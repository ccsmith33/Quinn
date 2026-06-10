# PDT-TRANSITION-D-077: transitional converter (ADR-012 §4.2). Survives
# the P2 tag-removal PR so a P2-era boot on a stale prod DB still
# self-heals; removed in the post-soak cleanup once `virtual_exits`
# shows zero active rows across a full week. Deliberately NOT tagged
# PDT-SUNSET-2026-06-04 — soak criterion 6 requires that grep to return
# nothing post-P2 while this module is still deployed.
"""One-shot idempotent converter: active virtual exits → real GTC broker orders.

ADR-012 standing invariant: at every instant, every open position has
either (a) an `active` virtual exit with the scanner running, or (b) a
live GTC protective order at the broker. The converter moves positions
from (a) to (b) with no window in between — a virtual exit is marked
`submitted` only AFTER the broker accepted the replacement order and
the order was journaled. On any failure the row stays `active` and the
still-running scanner keeps covering it.

Per execution group with `state='active'` rows on a broker-open position:

1. Idempotency pre-check: `get_order_by_client_id('pdt-convert-{vid}')`
   per row — a hit on a LIVE (or executed) order means a prior boot
   already converted this group and crashed before the journal write;
   journal the found order, mark the rows, submit nothing (the
   established crash-recovery pattern). A hit on a DEAD order
   (rejected/canceled/expired/suspended/replaced — async rejection, a
   broker sweep, or a deliberate operator cancel during triage) is NOT
   healed: journaling it would flip the group to `submitted`, stop the
   scanner, and silently leave the position with zero live protection
   (the 2026-05-07 unprotected-positions class). Instead the group is
   counted as an error, logged as `pdt_transition.heal_dead_order`, and
   left `active` so the scanner keeps covering it. Recovery is a
   deliberate OPERATOR step, not an automatic resubmit: a dead order
   here can mean the operator intentionally canceled it (the cutover
   runbook's manual-stop-cancellation gate), and a `replaced` order
   means live protection already exists at the broker under a different
   id — auto-resubmitting under a fresh client id could double-sell.
   The Task #7 runbook owns the triage step: inspect the dead order at
   the broker, then either place protection manually and mark the
   group's rows obsolete, or close the position (the converter then
   logs `skipped_no_position` and the scanner grace retires the rows).
2. Already-breached stop (`quote.last <= stop_price`): immediate market
   sell — a GTC stop below market triggers instantly anyway, with worse
   price control. A sibling TP row is marked obsolete (the market sell
   supersedes the whole group; submitting both would double-sell).
3. Otherwise `submit_oco_sell` (delta §7.3): GTC stop + GTC TP limit
   when both legs exist; plain GTC stop when only the stop does. A
   TP-only group (stop fired in a prior session) becomes a plain GTC
   limit sell via `submit_order`.
4. Journal before state-flip: honest `orders` rows (`role='stop'` /
   `'take_profit'`, notes 'D-077 conversion') — the 2026-05-08 lesson:
   unjournaled PDT sells make broker-position decreases unexplainable
   and halt the reconciler — then `mark_virtual_exit_submitted`.
5. Drain: every unreplayed `deferred_sells` row whose source virtual
   exit was converted this run ('invalidated: superseded by D-077
   conversion'), or whose position is already closed ('invalidated:
   position closed'), is invalidated with that note. The converter runs BEFORE the deferred
   replayer in `AgentLoop._boot` (ordering is load-bearing), so the
   replayer can never double-sell against a freshly-converted order.

Not this module's job: marking virtual exits obsolete on missing
positions. The scanner's consecutive-miss grace owns that call
(HOTFIX-2026-05-11); the converter just skips and leaves rows active.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Protocol

from broker.protocol import OrderRequest, Position, Quote, SubmittedOrder
from journal.models import DeferredSellRow, OrderRow, VirtualExitRow
from observability.log_port import get_logger

log = get_logger(__name__)

_CONVERSION_ORDER_NOTE = "D-077 conversion"
_DEFERRED_INVALIDATION_NOTE = "invalidated: superseded by D-077 conversion"
_DEFERRED_CLOSED_NOTE = "invalidated: position closed"

# W3-H2: a client-id pre-check hit is healable only while the order is
# live at the broker (or executed). Everything else — the review's dead
# set {rejected, canceled, expired, suspended, replaced} plus any status
# this module doesn't recognize ('calculated'/'stopped'/'held' fail safe
# into the dead path) — means the prior boot's conversion died between
# boots and the position is protected by nothing but the still-running
# scanner.
_HEALABLE_ORDER_STATUSES = frozenset(
    {
        "accepted",
        "new",
        "pending_new",
        "partially_filled",
        "filled",
        "done_for_day",
        "pending_cancel",
        "pending_replace",
    }
)


class ConverterBroker(Protocol):
    """Broker surface the converter needs. `submit_oco_sell` is WS2's
    delta §7.3 addition; structural typing keeps this module green
    before the WS2 merge (binding §5 order puts WS2 in first)."""

    def get_positions(self) -> list[Position]: ...
    def get_quote(self, symbol: str) -> Quote: ...
    def get_order_by_client_id(
        self, client_order_id: str
    ) -> SubmittedOrder | None: ...
    def submit_order(self, req: OrderRequest) -> SubmittedOrder: ...
    def submit_oco_sell(
        self,
        *,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float | None,
        client_order_id: str,
    ) -> tuple[SubmittedOrder, SubmittedOrder | None]: ...


class _JournalLike(Protocol):
    def list_active_virtual_exits(self) -> list[VirtualExitRow]: ...
    def mark_virtual_exit_submitted(
        self, vid: int, broker_order_id: str
    ) -> None: ...
    def mark_virtual_exit_obsolete(self, vid: int, reason: str) -> None: ...
    def insert_order(self, row: OrderRow) -> int: ...
    def list_unreplayed_deferred_sells(self) -> list[DeferredSellRow]: ...
    def mark_deferred_skipped(self, did: int, reason: str) -> None: ...


@dataclass(frozen=True)
class ConversionReport:
    """Counts from one `run()`. Logging only; callers do not branch."""

    converted_market: int
    converted_oco: int
    converted_stop: int
    converted_tp_limit: int
    healed: int
    skipped_no_position: int
    invalidated_deferred: int
    errors: int


def _client_id(vid: int) -> str:
    return f"pdt-convert-{vid}"


def _conversion_order_row(
    *, execution_id: int, role: str, req_or_so: SubmittedOrder
) -> OrderRow:
    """OrderRow from the broker's acknowledgement. Same journal shape the
    scanner/replayer used (orders.role: 'stop' | 'take_profit')."""
    so = req_or_so
    return OrderRow(
        execution_id=execution_id,
        role="take_profit" if role == "tp" else role,
        symbol=so.symbol,
        side="sell",
        order_type=so.order_type,
        qty=so.qty,
        limit_price=so.limit_price,
        stop_price=so.stop_price,
        tif="gtc" if so.order_type in ("stop", "limit") else "day",
        broker_order_id=so.broker_order_id,
        submitted_at=so.submitted_at,
        # D-078 §7.4: final_status is a deferred-completion field — NULL
        # until the FillIngestor records a terminal disposition. The
        # broker ack (so.status: 'accepted'/'new') is not terminal and
        # would hide the order from get_orders_pending_fill
        # (final_status IS NULL), making a converted GTC leg's fill
        # permanently unexplainable to the §2.2 classifier.
        final_status=None,
        notes=_CONVERSION_ORDER_NOTE,
    )


class PDTTransitionConverter:
    """Invoked from `AgentLoop._boot` before the deferred replayer, every
    boot, regardless of `pdt_enabled`, until the active set drains."""

    def __init__(self, *, broker: ConverterBroker, journal: _JournalLike) -> None:
        self._broker = broker
        self._journal = journal

    def run(self) -> ConversionReport:
        exits = self._journal.list_active_virtual_exits()
        unreplayed = self._journal.list_unreplayed_deferred_sells()
        if not exits and not unreplayed:
            return ConversionReport(0, 0, 0, 0, 0, 0, 0, 0)

        open_symbols = {
            p.symbol for p in self._broker.get_positions() if p.qty != 0
        }

        self._counts = {
            "market": 0, "oco": 0, "stop": 0, "tp_limit": 0,
            "healed": 0, "skipped": 0, "errors": 0,
        }
        converted_vids: set[int] = set()

        groups: dict[int, list[VirtualExitRow]] = {}
        for e in exits:
            if e.id is None:  # defensive — DB always assigns ids
                continue
            groups.setdefault(e.execution_id, []).append(e)

        for execution_id in sorted(groups):
            rows = groups[execution_id]
            try:
                self._convert_group(
                    execution_id, rows, open_symbols, converted_vids
                )
            except Exception as exc:  # noqa: BLE001 — per-group isolation;
                # rows stay 'active' so the scanner keeps covering them.
                self._counts["errors"] += 1
                log.error(
                    "pdt_transition.group_error",
                    extra={
                        "event": "pdt_transition.group_error",
                        "execution_id": execution_id,
                        "symbol": rows[0].symbol,
                        "error": str(exc),
                        "error_class": type(exc).__name__,
                    },
                )

        invalidated = 0
        for d in unreplayed:
            if d.id is None:
                continue
            # Distinct notes per delta §4.2 step 5: superseded-by-
            # conversion vs. position-already-closed (triage granularity).
            if d.virtual_exit_id in converted_vids:
                reason = _DEFERRED_INVALIDATION_NOTE
            elif d.symbol not in open_symbols:
                reason = _DEFERRED_CLOSED_NOTE
            else:
                continue
            self._journal.mark_deferred_skipped(d.id, reason=reason)
            invalidated += 1

        report = ConversionReport(
            converted_market=self._counts["market"],
            converted_oco=self._counts["oco"],
            converted_stop=self._counts["stop"],
            converted_tp_limit=self._counts["tp_limit"],
            healed=self._counts["healed"],
            skipped_no_position=self._counts["skipped"],
            invalidated_deferred=invalidated,
            errors=self._counts["errors"],
        )
        log.info(
            "pdt_transition.run",
            extra={
                "event": "pdt_transition.run",
                "converted_market": report.converted_market,
                "converted_oco": report.converted_oco,
                "converted_stop": report.converted_stop,
                "converted_tp_limit": report.converted_tp_limit,
                "healed": report.healed,
                "skipped_no_position": report.skipped_no_position,
                "invalidated_deferred": report.invalidated_deferred,
                "errors": report.errors,
            },
        )
        return report

    def _convert_group(
        self,
        execution_id: int,
        rows: list[VirtualExitRow],
        open_symbols: set[str],
        converted_vids: set[int],
    ) -> None:
        symbol = rows[0].symbol
        if symbol not in open_symbols:
            # Leave rows 'active' — scanner still covers them, and its
            # consecutive-miss grace owns the obsolete-marking decision.
            # Logged per group so the §4.4 operator verification can
            # triage any active-row residue straight from the log.
            self._counts["skipped"] += 1
            log.info(
                "pdt_transition.skipped_no_position",
                extra={
                    "event": "pdt_transition.skipped_no_position",
                    "execution_id": execution_id,
                    "symbol": symbol,
                    "virtual_exit_ids": [r.id for r in rows],
                },
            )
            return

        stop = next((r for r in rows if r.role == "stop"), None)
        tp = next((r for r in rows if r.role == "tp"), None)

        # -- Step 1: idempotency pre-check (prior-boot crash heal) -------
        dead: SubmittedOrder | None = None
        for r in rows:
            assert r.id is not None  # filtered by caller
            existing = self._broker.get_order_by_client_id(_client_id(r.id))
            if existing is None:
                continue
            if existing.status in _HEALABLE_ORDER_STATUSES:
                self._heal_group(execution_id, r, rows, existing, converted_vids)
                return
            dead = existing
        if dead is not None:
            # W3-H2: the prior boot's conversion DIED between boots
            # (async rejection, broker sweep, or operator cancel during
            # triage — an expected action under the runbook's manual-
            # stop-cancellation gate). Journaling it would flip the
            # group to 'submitted' and stop the scanner with zero live
            # protection left (the 2026-05-07 class). Leave the rows
            # 'active' (scanner covers), surface loudly, and let the
            # operator triage — see the module docstring for why this
            # is deliberately NOT an automatic resubmit.
            self._counts["errors"] += 1
            log.error(
                "pdt_transition.heal_dead_order",
                extra={
                    "event": "pdt_transition.heal_dead_order",
                    "execution_id": execution_id,
                    "symbol": symbol,
                    "client_order_id": dead.client_order_id,
                    "broker_order_id": dead.broker_order_id,
                    "order_status": dead.status,
                },
            )
            return

        # -- Step 2: already-breached stop → market sell ------------------
        if stop is not None:
            if stop.stop_price is None:
                raise ValueError(
                    f"stop virtual_exit id={stop.id} missing stop_price"
                )
            quote = self._broker.get_quote(symbol)
            if quote.last <= stop.stop_price:
                self._convert_breached(execution_id, stop, tp, converted_vids)
                return
            # -- Step 3a: stop (+ optional TP) → OCO / plain GTC stop -----
            self._convert_oco(execution_id, stop, tp, converted_vids)
            return

        # -- Step 3b: TP-only group (stop fired in a prior session) ------
        if tp is not None:
            self._convert_tp_only(execution_id, tp, converted_vids)

    def _heal_group(
        self,
        execution_id: int,
        matched: VirtualExitRow,
        rows: list[VirtualExitRow],
        existing: SubmittedOrder,
        converted_vids: set[int],
    ) -> None:
        """Prior boot converted at the broker but crashed before the
        journal write. Journal the found order; close out every row in
        the group against it. An OCO partner leg is not recoverable by
        client id — the matched order's journaled sell row still lets
        the lifecycle classifier explain either leg's fill (symbol-keyed).

        A prior boot may also have crashed AFTER the journal write but
        before mark_virtual_exit_submitted: the row already exists, and
        re-inserting trips orders.broker_order_id's UNIQUE constraint.
        That violation IS the already-journaled signal — consume it and
        carry on to the mark step, or the group errors on every boot
        and the rows stay pinned 'active' (W3-M1)."""
        try:
            self._journal.insert_order(
                _conversion_order_row(
                    execution_id=execution_id, role=matched.role, req_or_so=existing
                )
            )
        except sqlite3.IntegrityError:
            log.warning(
                "pdt_transition.heal_already_journaled",
                extra={
                    "event": "pdt_transition.heal_already_journaled",
                    "execution_id": execution_id,
                    "symbol": matched.symbol,
                    "broker_order_id": existing.broker_order_id,
                },
            )
        for r in rows:
            assert r.id is not None
            self._journal.mark_virtual_exit_submitted(
                r.id, broker_order_id=existing.broker_order_id
            )
            converted_vids.add(r.id)
        self._counts["healed"] += 1
        log.warning(
            "pdt_transition.healed_prior_boot",
            extra={
                "event": "pdt_transition.healed_prior_boot",
                "execution_id": execution_id,
                "symbol": matched.symbol,
                "client_order_id": _client_id(matched.id or 0),
                "broker_order_id": existing.broker_order_id,
            },
        )

    def _convert_breached(
        self,
        execution_id: int,
        stop: VirtualExitRow,
        tp: VirtualExitRow | None,
        converted_vids: set[int],
    ) -> None:
        assert stop.id is not None
        req = OrderRequest(
            symbol=stop.symbol,
            side="sell",
            qty=stop.qty,
            order_type="market",
            tif="day",
            client_order_id=_client_id(stop.id),
        )
        resp = self._broker.submit_order(req)
        self._journal.insert_order(
            _conversion_order_row(
                execution_id=execution_id, role="stop", req_or_so=resp
            )
        )
        self._journal.mark_virtual_exit_submitted(
            stop.id, broker_order_id=resp.broker_order_id
        )
        converted_vids.add(stop.id)
        if tp is not None and tp.id is not None:
            # The market sell empties the position; a TP order would
            # have nothing to sell. Obsolete is the honest state — no
            # broker order corresponds to this row.
            self._journal.mark_virtual_exit_obsolete(
                tp.id, reason="superseded: D-077 breached-stop market conversion"
            )
            converted_vids.add(tp.id)
        self._counts["market"] += 1
        log.info(
            "pdt_transition.converted",
            extra={
                "event": "pdt_transition.converted",
                "shape": "market_breached_stop",
                "execution_id": execution_id,
                "symbol": stop.symbol,
                "broker_order_id": resp.broker_order_id,
            },
        )

    def _convert_oco(
        self,
        execution_id: int,
        stop: VirtualExitRow,
        tp: VirtualExitRow | None,
        converted_vids: set[int],
    ) -> None:
        assert stop.id is not None
        assert stop.stop_price is not None  # caller-checked
        limit_price = tp.tp_price if tp is not None else None
        stop_so, tp_so = self._broker.submit_oco_sell(
            symbol=stop.symbol,
            qty=stop.qty,
            stop_price=stop.stop_price,
            limit_price=limit_price,
            client_order_id=_client_id(stop.id),
        )
        self._journal.insert_order(
            _conversion_order_row(
                execution_id=execution_id, role="stop", req_or_so=stop_so
            )
        )
        if tp_so is not None and tp is not None:
            self._journal.insert_order(
                _conversion_order_row(
                    execution_id=execution_id, role="tp", req_or_so=tp_so
                )
            )
        self._journal.mark_virtual_exit_submitted(
            stop.id, broker_order_id=stop_so.broker_order_id
        )
        converted_vids.add(stop.id)
        if tp is not None and tp.id is not None:
            # Close the TP row against its own leg when the broker
            # returned one; against the stop leg otherwise (defensive —
            # §7.3 returns a TP ack whenever limit_price was passed).
            tp_broker_id = (
                tp_so.broker_order_id if tp_so is not None
                else stop_so.broker_order_id
            )
            self._journal.mark_virtual_exit_submitted(
                tp.id, broker_order_id=tp_broker_id
            )
            converted_vids.add(tp.id)
            self._counts["oco"] += 1
        else:
            self._counts["stop"] += 1
        log.info(
            "pdt_transition.converted",
            extra={
                "event": "pdt_transition.converted",
                "shape": "oco" if tp is not None else "gtc_stop",
                "execution_id": execution_id,
                "symbol": stop.symbol,
                "broker_order_id": stop_so.broker_order_id,
            },
        )

    def _convert_tp_only(
        self,
        execution_id: int,
        tp: VirtualExitRow,
        converted_vids: set[int],
    ) -> None:
        assert tp.id is not None
        if tp.tp_price is None:
            raise ValueError(f"tp virtual_exit id={tp.id} missing tp_price")
        req = OrderRequest(
            symbol=tp.symbol,
            side="sell",
            qty=tp.qty,
            order_type="limit",
            tif="gtc",
            limit_price=tp.tp_price,
            client_order_id=_client_id(tp.id),
        )
        resp = self._broker.submit_order(req)
        self._journal.insert_order(
            _conversion_order_row(
                execution_id=execution_id, role="tp", req_or_so=resp
            )
        )
        self._journal.mark_virtual_exit_submitted(
            tp.id, broker_order_id=resp.broker_order_id
        )
        converted_vids.add(tp.id)
        self._counts["tp_limit"] += 1
        log.info(
            "pdt_transition.converted",
            extra={
                "event": "pdt_transition.converted",
                "shape": "gtc_tp_limit",
                "execution_id": execution_id,
                "symbol": tp.symbol,
                "broker_order_id": resp.broker_order_id,
            },
        )


__all__ = [
    "ConversionReport",
    "ConverterBroker",
    "PDTTransitionConverter",
]
