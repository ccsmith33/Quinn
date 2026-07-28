"""Protection-preserving order surgery on positions with resting legs.

Incidents 2026-07-14..17 (FEIM naked at +26%, VRDN close failing four
straight days with Alpaca 40310000 `insufficient qty available ...
held_for_orders`) share one missing discipline: any operation that
touches a position's resting protective legs must be sequenced so the
position can never end the operation with FEWER protective orders than
it started with (except a completed close, qty 0).

This module is the shared vocabulary for that discipline:

- `cancel_protective_legs`: confirm-or-abort cancellation of every live
  journaled protective leg (roles stop / trailing_stop / take_profit,
  `final_status IS NULL`) BEFORE a liquidation sell is submitted. A
  cancel that cannot be confirmed dead leaves the leg in `failed` — the
  caller must NOT submit its sell (the leg still holds the shares; the
  sell would be the VRDN 40310000 rejection all over again).
- `restore_protection`: re-place the just-cleared legs (stop, or
  stop + TP as one OCO) when the follow-up submission failed — the
  FEIM class killer. Journals honest replacement rows.

Cancel ordering is deliberate: take-profit legs first, stops LAST. If a
TP cancel fails nothing has been touched yet; if a stop cancel fails
after the TP cleared, the position still has its downside protection
live at the broker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from journal.models import OrderRow
from journal.repo import connect, insert_order
from observability.log_port import get_logger

log = get_logger(__name__)

# Delay in seconds after canceling orders to allow Alpaca's system to
# release the shares before submitting new orders (hotfix for 40310000
# "insufficient qty available" errors when shares are still "held_for_orders"
# immediately after cancellation).
_POST_CANCEL_DELAY_SECONDS = 0.15

# Roles that constitute a position's broker-side protection.
PROTECTIVE_ROLES = ("take_profit", "stop", "trailing_stop")

# Broker order statuses that mean the order can no longer fill (mirrors
# the thesis coordinator's set — see its module docstring for why
# done_for_day is deliberately absent).
DEAD_ORDER_STATUSES = frozenset(
    {"expired", "canceled", "rejected", "replaced", "suspended", "stopped"}
)


class _CancelBroker(Protocol):
    def cancel_order(self, broker_order_id: str) -> None: ...


@dataclass(frozen=True)
class ClearedLeg:
    """A protective leg confirmed no-longer-live at the broker."""

    row: OrderRow
    disposition: str  # 'canceled' | dead status | 'filled'


@dataclass(frozen=True)
class LegCancelOutcome:
    cleared: list[ClearedLeg] = field(default_factory=list)
    failed: list[OrderRow] = field(default_factory=list)  # still live

    @property
    def all_cleared(self) -> bool:
        return not self.failed

    @property
    def any_filled(self) -> bool:
        return any(c.disposition == "filled" for c in self.cleared)


def get_live_protective_legs(db_path: str, execution_id: int) -> list[OrderRow]:
    """Journal orders rows for the execution still pending a terminal
    disposition (`final_status IS NULL`) in the protective roles, TP
    legs first (cancel-ordering contract, see module docstring)."""
    placeholders = ", ".join("?" for _ in PROTECTIVE_ROLES)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM orders WHERE execution_id = ? "
            f"AND final_status IS NULL AND role IN ({placeholders}) "
            f"ORDER BY CASE WHEN role = 'take_profit' THEN 0 ELSE 1 END, id ASC",
            (execution_id, *PROTECTIVE_ROLES),
        ).fetchall()
    return [OrderRow(**dict(r)) for r in rows]


def resolve_leg_cancel(broker: Any, row: OrderRow) -> str | None:
    """Cancel one leg and confirm it can no longer fill.

    Returns the leg's disposition ('canceled', the actual dead status
    when it was already dead, or 'filled' when the cancel lost a race to
    a fill), or None when the leg is STILL LIVE at the broker (cancel
    not acked and the order is not dead) — the caller must treat None as
    "shares still held".
    """
    try:
        broker.cancel_order(row.broker_order_id)
        return "canceled"
    except Exception as cancel_err:  # noqa: BLE001 — classified below
        get_order_by_id = getattr(broker, "get_order_by_id", None)
        if get_order_by_id is None:
            return None  # cannot interrogate; assume still live
        try:
            current = get_order_by_id(row.broker_order_id)
        except Exception:  # noqa: BLE001
            return None
        if current is None:
            return "canceled"  # gone at the broker
        if current.status == "filled":
            return "filled"
        if current.status in DEAD_ORDER_STATUSES:
            return current.status
        log.warning(
            "protection.leg_cancel_unconfirmed",
            extra={
                "event": "protection.leg_cancel_unconfirmed",
                "broker_order_id": row.broker_order_id,
                "role": row.role,
                "broker_status": current.status,
                "error": str(cancel_err),
            },
        )
        return None


def cancel_protective_legs(
    db_path: str, broker: _CancelBroker, execution_id: int
) -> LegCancelOutcome:
    """Cancel every live protective leg for the execution, confirming
    each. Stops early on the first unconfirmed cancel (the remaining
    legs stay untouched and live). Adds a small delay after all legs
    are cleared to allow Alpaca's system to release the shares."""
    outcome = LegCancelOutcome()
    for row in get_live_protective_legs(db_path, execution_id):
        disposition = resolve_leg_cancel(broker, row)
        if disposition is None:
            outcome.failed.append(row)
            break  # leave remaining legs untouched — they still protect
        outcome.cleared.append(ClearedLeg(row=row, disposition=disposition))

    # If all legs were successfully cleared, wait a brief moment for
    # Alpaca's system to release the shares before the caller attempts
    # to submit new orders (hotfix for 40310000 errors).
    if outcome.all_cleared and outcome.cleared:
        time.sleep(_POST_CANCEL_DELAY_SECONDS)

    return outcome


def restore_protection(
    db_path: str,
    broker: Any,
    execution_id: int,
    cleared: list[ClearedLeg],
    *,
    client_order_id: str,
    notes: str,
) -> OrderRow | None:
    """Re-place the protection described by `cleared` (the FEIM-class
    compensation): the newest cleared stop leg (and TP leg, as one OCO
    when both exist). Returns the journaled replacement STOP row on
    success; None when nothing needed restoring or the re-placement
    itself failed (caller logs the ERROR and the reconciler's
    `position.naked` tripwire is the alerting net).

    A leg whose disposition is 'filled' is never restored — its fill is
    the exit, owned by fill ingestion.
    """
    stop_leg = next(
        (
            c.row
            for c in reversed(cleared)
            if c.row.role in ("stop", "trailing_stop") and c.disposition != "filled"
        ),
        None,
    )
    tp_leg = next(
        (
            c.row
            for c in reversed(cleared)
            if c.row.role == "take_profit" and c.disposition != "filled"
        ),
        None,
    )
    if stop_leg is None or stop_leg.stop_price is None:
        return None  # no stop to restore (TP-only positions keep their exit
        # via the still-live rows; downside restoration is the money path)

    try:
        stop_resp, tp_resp = broker.submit_oco_sell(
            symbol=stop_leg.symbol,
            qty=stop_leg.qty,
            stop_price=stop_leg.stop_price,
            limit_price=tp_leg.limit_price if tp_leg is not None else None,
            client_order_id=client_order_id,
        )
    except Exception as e:  # noqa: BLE001
        log.error(
            "protection.restore_failed",
            extra={
                "event": "protection.restore_failed",
                "execution_id": execution_id,
                "symbol": stop_leg.symbol,
                "stop_price": stop_leg.stop_price,
                "error": str(e),
                "error_class": type(e).__name__,
            },
        )
        return None

    new_stop_row_id = insert_order(
        db_path,
        OrderRow(
            execution_id=execution_id,
            role=stop_leg.role,
            symbol=stop_leg.symbol,
            side="sell",
            order_type="stop",
            qty=stop_leg.qty,
            tif="gtc",
            stop_price=stop_leg.stop_price,
            broker_order_id=stop_resp.broker_order_id,
            submitted_at=stop_resp.submitted_at,
            final_status=None,
            notes=notes,
        ),
    )
    if tp_leg is not None and tp_resp is not None:
        insert_order(
            db_path,
            OrderRow(
                execution_id=execution_id,
                role="take_profit",
                symbol=tp_leg.symbol,
                side="sell",
                order_type="limit",
                qty=tp_leg.qty,
                tif="gtc",
                limit_price=tp_leg.limit_price,
                broker_order_id=tp_resp.broker_order_id,
                submitted_at=tp_resp.submitted_at,
                final_status=None,
                notes=notes,
            ),
        )
    log.warning(
        "protection.restored",
        extra={
            "event": "protection.restored",
            "execution_id": execution_id,
            "symbol": stop_leg.symbol,
            "stop_price": stop_leg.stop_price,
            "tp_price": tp_leg.limit_price if tp_leg is not None else None,
            "broker_order_id": stop_resp.broker_order_id,
        },
    )
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE id = ?", (new_stop_row_id,)
        ).fetchone()
    return OrderRow(**dict(row)) if row is not None else None


__all__ = [
    "DEAD_ORDER_STATUSES",
    "PROTECTIVE_ROLES",
    "ClearedLeg",
    "LegCancelOutcome",
    "cancel_protective_legs",
    "get_live_protective_legs",
    "resolve_leg_cancel",
    "restore_protection",
]
