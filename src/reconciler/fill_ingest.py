"""WS1 — Fill-event ingestion via per-tick order-status polling
(D-078, architecture-option-b-2026-06-09.md §2.1, ADR-010).

Invoked at the top of every `reconcile_now()`, before position
snapshotting and diff classification, so the tick that detects a fill
also explains it. Per tick:

  1. poll `get_order_by_id` for every journaled order with
     `final_status IS NULL` (≤ ~15 rows at v1 volume);
  2. map Alpaca terminal statuses to `final_status` and record the
     outcome via the one-time NULL→value `record_order_outcome`;
  3. when a recorded sell fill brings the journal's open qty for the
     symbol to zero, append a qty=0 `positions` tombstone
     (`source='fill_ingest'`) — the RC-1 fix: closed positions stop
     being permanent ghosts.

Effects: orders gain ground-truth outcomes (revives KS-3 and realized
P&L); the reconciler demotes to verification (FR-24 as written).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from broker.alpaca import BrokerUnavailable
from broker.protocol import SubmittedOrder
from journal.models import OrderRow, PositionRow
from observability.log_port import get_logger

log = get_logger(__name__)

# Alpaca lifecycle statuses that are terminal — the order will never fill
# further. Everything else (new, accepted, partially_filled, pending_*,
# held, ...) stays pending and is polled again next tick.
_TERMINAL_STATUSES = frozenset({"filled", "canceled", "expired", "rejected", "replaced"})


@dataclass(frozen=True)
class FillIngestReport:
    """Per-tick outcome summary, carried in structured logs and tests."""

    polled: int = 0
    recorded: int = 0
    tombstoned: list[str] = field(default_factory=list)
    deferred: bool = False  # broker unavailable; poll aborted this tick


class _JournalLike(Protocol):
    def get_orders_pending_fill(self) -> list[OrderRow]: ...
    def record_order_outcome(
        self,
        order_id: int,
        final_status: str,
        *,
        fill_price: float | None,
        fill_qty: int | None,
        fill_at: dt.datetime | None,
    ) -> None: ...
    def get_latest_position_for_symbol(self, symbol: str) -> PositionRow | None: ...
    def insert_position_tombstone(
        self,
        symbol: str,
        source: str,
        notes: str,
        *,
        snapshot_at: dt.datetime | None = None,
    ) -> int: ...


class _BrokerLike(Protocol):
    def get_order_by_id(self, broker_order_id: str) -> SubmittedOrder | None: ...


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _terminal_status(sub: SubmittedOrder) -> str | None:
    """Map a broker order status to our `final_status`, or None if the
    order is still live. A canceled/expired order that partially filled
    becomes `partially_filled_closed` — it carries real fill qty that the
    lifecycle classifier must see (delta §2.1 step 2)."""
    if sub.status not in _TERMINAL_STATUSES:
        return None
    if sub.status in ("canceled", "expired") and sub.filled_qty > 0:
        return "partially_filled_closed"
    return sub.status


class FillIngestor:
    """Poll-based fill ingestion (ADR-010). Sync, like the reconciler
    that hosts it; deterministic function of (pending rows × broker
    responses), testable with the established fake-broker pattern."""

    def __init__(
        self,
        *,
        broker: _BrokerLike,
        journal: _JournalLike,
        now_fn: Callable[[], dt.datetime] = _utcnow,
    ) -> None:
        self._broker = broker
        self._journal = journal
        self._now_fn = now_fn

    def run_tick(self) -> FillIngestReport:
        now = self._now_fn()
        pending = self._journal.get_orders_pending_fill()
        recorded = 0
        tombstoned: list[str] = []
        # W1-L3: per-symbol sum of sell qty recorded THIS tick. Two sells
        # each partial against the (stale-within-the-tick) latest snapshot
        # can together close the position; the tombstone must come from
        # here, not from a later external-close absorption with wrong
        # provenance and a spurious operator alert.
        sold_this_tick: dict[str, int] = {}
        for row in pending:
            try:
                sub = self._broker.get_order_by_id(row.broker_order_id)
            except BrokerUnavailable as e:
                # Broker down → nothing else will succeed this tick; the
                # reconcile body that follows will defer on its own calls.
                log.info(
                    "fill_ingest.broker_unavailable",
                    extra={
                        "event": "fill_ingest.broker_unavailable",
                        "error": str(e),
                    },
                )
                return FillIngestReport(
                    polled=len(pending),
                    recorded=recorded,
                    tombstoned=tombstoned,
                    deferred=True,
                )
            except Exception as e:  # noqa: BLE001 — one poisoned row must not block the rest
                log.error(
                    "fill_ingest.row_error",
                    extra={
                        "event": "fill_ingest.row_error",
                        "order_id": row.id,
                        "broker_order_id": row.broker_order_id,
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )
                continue
            if sub is None:
                # Broker has no record (404). Never invent an outcome —
                # leave pending so it stays visible here every tick.
                log.warning(
                    "fill_ingest.order_unknown_at_broker",
                    extra={
                        "event": "fill_ingest.order_unknown_at_broker",
                        "order_id": row.id,
                        "broker_order_id": row.broker_order_id,
                        "symbol": row.symbol,
                    },
                )
                continue
            final = _terminal_status(sub)
            if final is None:
                continue
            fill_qty = sub.filled_qty if sub.filled_qty > 0 else None
            self._journal.record_order_outcome(
                row.id if row.id is not None else 0,
                final,
                fill_price=sub.filled_avg_price,
                fill_qty=fill_qty,
                fill_at=sub.filled_at,
            )
            recorded += 1
            log.info(
                "fill_ingest.outcome_recorded",
                extra={
                    "event": "fill_ingest.outcome_recorded",
                    "order_id": row.id,
                    "symbol": row.symbol,
                    "side": row.side,
                    "role": row.role,
                    "final_status": final,
                    "fill_qty": sub.filled_qty,
                    "fill_price": sub.filled_avg_price,
                },
            )
            if (
                row.side == "sell"
                and final in ("filled", "partially_filled_closed")
                and sub.filled_qty > 0
            ):
                sold_this_tick[row.symbol] = (
                    sold_this_tick.get(row.symbol, 0) + sub.filled_qty
                )
            if self._closes_position(row, final, sold_this_tick):
                self._journal.insert_position_tombstone(
                    row.symbol,
                    "fill_ingest",
                    f"closed by order {row.id} "
                    f"(broker {row.broker_order_id}, role {row.role}); D-078",
                    snapshot_at=now,
                )
                tombstoned.append(row.symbol)
                log.info(
                    "fill_ingest.position_tombstoned",
                    extra={
                        "event": "fill_ingest.position_tombstoned",
                        "symbol": row.symbol,
                        "closing_order_id": row.id,
                    },
                )
        return FillIngestReport(
            polled=len(pending), recorded=recorded, tombstoned=tombstoned
        )

    def _closes_position(
        self, row: OrderRow, final: str, sold_this_tick: dict[str, int]
    ) -> bool:
        """Recorded sell fills close the position when the qty sold THIS
        tick (W1-L3: summed across same-tick fills, since the latest
        snapshot is stale within the tick) covers the journal's open qty.
        Partial closes leave the snapshot to the reconciler's next
        broker-truth write. Fires at most once per symbol per tick: the
        first crossing tombstones, after which the latest row is the
        qty-0 tombstone and the open-qty guard is False."""
        if row.side != "sell" or final not in ("filled", "partially_filled_closed"):
            return False
        sold = sold_this_tick.get(row.symbol, 0)
        if sold <= 0:
            return False
        open_pos = self._journal.get_latest_position_for_symbol(row.symbol)
        if open_pos is None or open_pos.qty <= 0:
            return False
        return sold >= open_pos.qty


__all__ = ["FillIngestReport", "FillIngestor"]
