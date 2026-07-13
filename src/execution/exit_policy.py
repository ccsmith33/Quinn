"""D-079 §3.5/§3.6 — ExitPolicyTicker: trailing-stop ratchet + stale-entry
hygiene (ADR-011).

The let-winners-run mechanism. A deterministic component that runs once
per reconciler tick (the hook slot the PDT scanner vacates) and *raises*
the existing broker-side GTC stop via the atomic `replace_stop_order`
PATCH as price advances. The broker order is always the protection; the
ticker only improves it. Process down ⇒ stop frozen at the last
ratcheted level — still live at the broker (the decisive argument
against scanner-style virtual trailing, ADR-011).

Invariants (delta §3.5):
- Never lowers a stop.
- Never acts on a position without a live journaled broker stop.
- Never touches the TP leg (OCO linkage survives the PATCH).
- One state row per execution (`exit_policy_state`, migration 006) —
  operational memory only; the audit trail is the append-only `orders`
  replacement chain (NFR-16 undiluted).

Second duty (§3.6): cancel any unfilled GTC *entry* order whose
submission day (ET) has passed — restores de-facto DAY-entry semantics
now that the whole bracket group is GTC (§3.1). The cancel outcome is
recorded by fill ingestion on its next tick, not here.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from broker.protocol import BrokerAdapter
from config.calendar import ET
from config.loader import TrailStage
from journal.exit_policy import (
    ExitPolicyStateRow,
    get_exit_policy_state,
    upsert_exit_policy_state,
)
from journal.models import OrderRow
from journal.repo import (
    connect,
    get_orders_for_execution,
    get_proposal_by_id,
    has_open_position,
    insert_order,
)
from observability.log_port import get_logger

log = get_logger(__name__)

_PROTECTIVE_ROLES = ("stop", "trailing_stop")

# Default-trail clamp bounds (ADR-011): the proposal's own initial risk
# distance as a percent of entry, clamped to [1, 15]. Analyzer-proposed
# distances are schema-bounded (0.5–20) and pass through unclamped.
_DEFAULT_TRAIL_CLAMP_LO = 1.0
_DEFAULT_TRAIL_CLAMP_HI = 15.0


class _JournalLike(Protocol):
    db_path: str


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _ensure_aware(d: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes; treat them as UTC (same convention
    as the thesis coordinator)."""
    if d.tzinfo is None:
        return d.replace(tzinfo=dt.UTC)
    return d


# ---------------------------------------------------------------------------
# WS1 §7.4 port defaults — same pattern as the thesis coordinator:
# late-import repo.py's functions (WS1's file this epic); reads fall
# back to local SQL, the write falls back to a logged no-op so the
# branch stays green standalone and NFR-16a's single-writer rule holds.
# ---------------------------------------------------------------------------


def _default_record_order_outcome(
    db_path: str,
    order_id: int,
    final_status: str,
    *,
    fill_price: float | None,
    fill_qty: int | None,
    fill_at: dt.datetime | None,
) -> None:
    try:
        # WS1 §7.4 — lands with the merge train; missing name raises
        # ImportError until then, so the except arm below covers it.
        from journal.repo import (
            record_order_outcome,
        )
    except ImportError:
        log.warning(
            "exit_policy.record_order_outcome_unavailable",
            extra={
                "event": "exit_policy.record_order_outcome_unavailable",
                "order_id": order_id,
                "final_status": final_status,
            },
        )
        return
    record_order_outcome(
        db_path,
        order_id,
        final_status,
        fill_price=fill_price,
        fill_qty=fill_qty,
        fill_at=fill_at,
    )


class ExitPolicyTicker:
    """Deterministic per-tick exit-policy actuator (D-079 §3.5/§3.6).

    Implements the reconciler's exit-policy hook shape — a *sync*
    `run_tick`, like the scanner's, per WS1's seam (the reconciler
    invokes it without await); runs after the thesis hook in
    composition order.
    """

    def __init__(
        self,
        *,
        journal: _JournalLike,
        broker: BrokerAdapter,
        trail_activation_r: float = 1.0,
        min_ratchet_step_pct: float = 0.25,
        trail_stages: Sequence[TrailStage] = (),
        now_fn: Callable[[], dt.datetime] = _utcnow,
        record_order_outcome: Callable[..., None] | None = None,
    ) -> None:
        self._journal = journal
        self._broker = broker
        self._activation_r = trail_activation_r
        self._min_step_pct = min_ratchet_step_pct
        self._trail_stages = tuple(trail_stages)
        self._now_fn = now_fn
        self._record_order_outcome = record_order_outcome or (
            lambda order_id, final_status, *, fill_price, fill_qty, fill_at:
            _default_record_order_outcome(
                self._journal.db_path,
                order_id,
                final_status,
                fill_price=fill_price,
                fill_qty=fill_qty,
                fill_at=fill_at,
            )
        )

    def run_tick(self) -> None:
        """One exit-policy tick. Errors on a single execution/order are
        logged and never stop the rest of the tick (same protective
        shape as the retro/thesis hooks)."""
        self._cancel_stale_entries()
        self._ratchet_trailing_stops()

    # ------------------------------------------------------------------
    # §3.6 — stale-entry hygiene
    # ------------------------------------------------------------------

    def _cancel_stale_entries(self) -> None:
        """Cancel unfilled GTC entry orders whose ET submission day has
        passed. Canceling the bracket parent cancels unfilled children
        atomically; the journal outcome (`canceled`) arrives via fill
        ingestion on the next tick — this method writes nothing."""
        now_et_date = self._now_fn().astimezone(ET).date()
        with connect(self._journal.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM orders "
                "WHERE role = 'entry' AND tif = 'gtc' "
                "AND final_status IS NULL AND realized_fill_at IS NULL"
            ).fetchall()
        for raw in rows:
            row = OrderRow(**dict(raw))
            submitted_et_date = (
                _ensure_aware(row.submitted_at).astimezone(ET).date()
            )
            if submitted_et_date >= now_et_date:
                continue
            try:
                self._broker.cancel_order(row.broker_order_id)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "exit_policy.stale_entry_cancel_failed",
                    extra={
                        "event": "exit_policy.stale_entry_cancel_failed",
                        "order_id": row.id,
                        "broker_order_id": row.broker_order_id,
                        "symbol": row.symbol,
                        "error": str(e),
                    },
                )
                continue
            log.info(
                "exit_policy.stale_entry_canceled",
                extra={
                    "event": "exit_policy.stale_entry_canceled",
                    "order_id": row.id,
                    "broker_order_id": row.broker_order_id,
                    "symbol": row.symbol,
                    "submitted_et_date": str(submitted_et_date),
                },
            )

    # ------------------------------------------------------------------
    # §3.5 — trailing ratchet
    # ------------------------------------------------------------------

    def _ratchet_trailing_stops(self) -> None:
        for live_stop in self._live_protective_stops():
            try:
                self._ratchet_one(live_stop)
            except Exception as e:  # noqa: BLE001
                log.error(
                    "exit_policy.ratchet_error",
                    extra={
                        "event": "exit_policy.ratchet_error",
                        "execution_id": live_stop.execution_id,
                        "symbol": live_stop.symbol,
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )

    def _live_protective_stops(self) -> list[OrderRow]:
        """Newest pending-fill stop/trailing_stop row per execution —
        the 'never act without a live journaled broker stop' invariant
        holds by construction."""
        with connect(self._journal.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM orders o "
                "WHERE o.role IN ('stop', 'trailing_stop') "
                "AND o.final_status IS NULL "
                "AND o.id = ("
                "    SELECT MAX(o2.id) FROM orders o2 "
                "    WHERE o2.execution_id = o.execution_id "
                "    AND o2.role IN ('stop', 'trailing_stop') "
                "    AND o2.final_status IS NULL"
                ") "
                "ORDER BY o.execution_id ASC"
            ).fetchall()
        return [OrderRow(**dict(r)) for r in rows]

    def _ratchet_one(self, live_stop: OrderRow) -> None:
        execution_id = live_stop.execution_id
        symbol = live_stop.symbol
        if not has_open_position(self._journal.db_path, symbol):
            return
        if live_stop.stop_price is None or live_stop.stop_price <= 0:
            return

        state = get_exit_policy_state(
            self._journal.db_path, execution_id=execution_id
        )
        quote = self._broker.get_quote(symbol)
        last = float(quote.last)

        entry_price: float | None = None
        if state is None:
            geometry = self._initial_geometry(execution_id)
            if geometry is None:
                return
            entry_price, initial_risk, trail_pct = geometry
            # Engage when the winner has earned `activation_r` multiples
            # of its initial risk distance (default 1.0R).
            if last < entry_price + self._activation_r * initial_risk:
                return
            state = ExitPolicyStateRow(
                execution_id=execution_id,
                symbol=symbol,
                trail_distance_pct=trail_pct,
                trail_engaged=True,
                high_water_mark=last,
                stop_order_journal_id=live_stop.id,
            )
            upsert_exit_policy_state(self._journal.db_path, state)
            engaged_pct, engaged_stage = self._staged_trail_pct(
                trail_pct, entry_price=entry_price, high_water=last
            )
            log.info(
                "exit_policy.trail_engaged",
                extra={
                    "event": "exit_policy.trail_engaged",
                    "execution_id": execution_id,
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "initial_risk": initial_risk,
                    "trail_distance_pct": trail_pct,
                    "effective_trail_pct": engaged_pct,
                    "trail_stage_gain_pct": engaged_stage,
                    "high_water_mark": last,
                },
            )
            # Fall through — the engagement tick may already warrant a
            # first ratchet step.

        # Restart semantics: high-water resumes from the stored value; a
        # gap above it during downtime is simply a new high-water now.
        high_water = max(state.high_water_mark, last)
        # Staged tightening: the persisted width is the BASE; the
        # effective width is derived here every evaluation from config +
        # current high-water, so newly configured stages apply to
        # already-open positions on restart. Crossing a milestone
        # tightens from the CURRENT high-water immediately — one large
        # upward PATCH is intended.
        if self._trail_stages and entry_price is None:
            entry_price = self._entry_price(execution_id)
        effective_pct, active_stage_gain = self._staged_trail_pct(
            state.trail_distance_pct,
            entry_price=entry_price,
            high_water=high_water,
        )
        target = round(high_water * (1.0 - effective_pct / 100.0), 2)
        current_stop = float(live_stop.stop_price)

        should_replace = (
            target > current_stop * (1.0 + self._min_step_pct / 100.0)
            # An in-the-money sell stop (target at/above the current
            # quote after a fast fall) would fire instantly — the trail
            # accepts one tick of lag instead (ADR-011 consequences).
            and target < last
        )
        if not should_replace:
            if high_water > state.high_water_mark:
                upsert_exit_policy_state(
                    self._journal.db_path,
                    state.model_copy(update={"high_water_mark": high_water}),
                )
            return

        client_order_id = (
            f"trail-exec-{execution_id}-{self._now_fn().timestamp():.0f}"
        )
        try:
            replaced = self._broker.replace_stop_order(
                live_stop.broker_order_id,
                new_stop_price=target,
                client_order_id=client_order_id,
            )
        except Exception as e:  # noqa: BLE001
            # Atomic PATCH contract: the original stop is still live.
            # Persist the high-water advance and retry next tick.
            log.error(
                "exit_policy.ratchet_replace_failed",
                extra={
                    "event": "exit_policy.ratchet_replace_failed",
                    "execution_id": execution_id,
                    "symbol": symbol,
                    "old_broker_order_id": live_stop.broker_order_id,
                    "target_stop_price": target,
                    "error": str(e),
                },
            )
            if high_water > state.high_water_mark:
                upsert_exit_policy_state(
                    self._journal.db_path,
                    state.model_copy(update={"high_water_mark": high_water}),
                )
            return

        # Journal chain identical to §3.3: fresh live row, old completed
        # 'replaced' via the §7.4 single-writer.
        new_row_id = insert_order(
            self._journal.db_path,
            OrderRow(
                execution_id=execution_id,
                role="trailing_stop",
                symbol=symbol,
                side="sell",
                order_type="stop",
                qty=live_stop.qty,
                tif="gtc",
                stop_price=target,
                broker_order_id=replaced.broker_order_id,
                submitted_at=replaced.submitted_at,
                final_status=None,
                notes=(
                    f"trail_ratchet hw={high_water:.4f} "
                    f"trail={effective_pct:.2f}% "
                    f"replaced {live_stop.broker_order_id}"
                ),
            ),
        )
        self._record_order_outcome(
            live_stop.id, "replaced", fill_price=None, fill_qty=None, fill_at=None
        )
        upsert_exit_policy_state(
            self._journal.db_path,
            state.model_copy(
                update={
                    "high_water_mark": high_water,
                    "stop_order_journal_id": new_row_id,
                }
            ),
        )
        log.info(
            "exit_policy.stop_ratcheted",
            extra={
                "event": "exit_policy.stop_ratcheted",
                "execution_id": execution_id,
                "symbol": symbol,
                "old_stop_price": current_stop,
                "new_stop_price": target,
                "high_water_mark": high_water,
                "effective_trail_pct": effective_pct,
                "trail_stage_gain_pct": active_stage_gain,
                "old_broker_order_id": live_stop.broker_order_id,
                "new_broker_order_id": replaced.broker_order_id,
            },
        )

    def _initial_geometry(
        self, execution_id: int
    ) -> tuple[float, float, float] | None:
        """(entry_price, initial_risk, trail_distance_pct) for an
        execution, or None when the journal can't support a trail.

        Trail distance: analyzer-proposed `trail_distance_pct` when the
        proposal carries one (schema-bounded 0.5–20); otherwise the
        proposal's own initial risk distance as a percent of entry,
        clamped to [1, 15] (ADR-011 — self-calibrating, no new magic
        constant).
        """
        orders = get_orders_for_execution(self._journal.db_path, execution_id)
        entry = next((o for o in orders if o.role == "entry"), None)
        if entry is None:
            return None
        entry_price = entry.realized_fill_price or entry.pre_submission_last
        if entry_price is None or entry_price <= 0:
            return None
        initial_stops = [
            o for o in orders if o.role == "stop" and o.stop_price is not None
        ]
        if not initial_stops:
            return None
        stop_initial = min(initial_stops, key=lambda o: o.id or 0).stop_price
        assert stop_initial is not None  # filtered above
        initial_risk = entry_price - stop_initial
        if initial_risk <= 0:
            return None

        proposed = self._proposed_trail_pct(execution_id)
        if proposed is not None:
            return entry_price, initial_risk, proposed
        default_pct = initial_risk / entry_price * 100.0
        clamped = min(
            max(default_pct, _DEFAULT_TRAIL_CLAMP_LO), _DEFAULT_TRAIL_CLAMP_HI
        )
        return entry_price, initial_risk, clamped

    def _staged_trail_pct(
        self, base_pct: float, *, entry_price: float | None, high_water: float
    ) -> tuple[float, float | None]:
        """Effective trail width under staged tightening: min(base, the
        tightest stage whose gain milestone the high-water mark has
        crossed). Tighten-only — a stage wider than the current width is
        ignored. Returns (effective_pct, active stage's gain_pct or
        None). Never persisted: derived from config + current HWM every
        evaluation, so stages apply to already-open positions.
        """
        effective = base_pct
        active: float | None = None
        if not self._trail_stages or entry_price is None or entry_price <= 0:
            return effective, active
        gain_pct = (high_water / entry_price - 1.0) * 100.0
        for stage in self._trail_stages:
            if gain_pct >= stage.gain_pct and stage.trail_pct < effective:
                effective = stage.trail_pct
                active = stage.gain_pct
        return effective, active

    def _entry_price(self, execution_id: int) -> float | None:
        """Entry fill (or pre-submission last) for the staged-trail gain
        math; None when the journal can't support it (stages then leave
        the base width untouched)."""
        orders = get_orders_for_execution(self._journal.db_path, execution_id)
        entry = next((o for o in orders if o.role == "entry"), None)
        if entry is None:
            return None
        price = entry.realized_fill_price or entry.pre_submission_last
        if price is None or price <= 0:
            return None
        return float(price)

    def _proposed_trail_pct(self, execution_id: int) -> float | None:
        execution = self._get_execution(execution_id)
        if execution is None:
            return None
        proposal = get_proposal_by_id(
            self._journal.db_path, execution.proposal_id
        )
        if proposal is None or proposal.raw_response is None:
            return None
        try:
            from proposal.schemas import validate_trade_proposal

            trade = validate_trade_proposal(json.loads(proposal.raw_response))
        except Exception:  # noqa: BLE001 — malformed payload → default trail
            return None
        return trade.trail_distance_pct

    def _get_execution(self, execution_id: int) -> Any:
        with connect(self._journal.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM executions WHERE id = ?", (execution_id,)
            ).fetchone()
        if row is None:
            return None
        from journal.models import ExecutionRow

        return ExecutionRow(**dict(row))


__all__ = ["ExitPolicyTicker"]
