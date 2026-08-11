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

from broker.protocol import BrokerAdapter, OrderRequest
from config.calendar import ET
from config.loader import TrailStage
from execution.protection import DEAD_ORDER_STATUSES
from execution.quantize import quantize_price
from journal.exit_policy import (
    ExitPolicyStateRow,
    get_exit_policy_state,
    set_stop_order_journal_id,
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

# Consecutive identical-target replace failures before the ratchet
# escalates once (`exit_policy.ratchet_stuck`, ERROR — the ops/alerting
# key) and demotes further identical-target `ratchet_replace_failed`
# logs to DEBUG until the target moves (hotfix 2026-08-11: ATRC retried
# a broker-invalid sub-penny target at ERROR every tick forever).
_RATCHET_STUCK_THRESHOLD = 3


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
        breakeven_floor_gain_pct: float = 0.0,
        now_fn: Callable[[], dt.datetime] = _utcnow,
        record_order_outcome: Callable[..., None] | None = None,
    ) -> None:
        self._journal = journal
        self._broker = broker
        self._activation_r = trail_activation_r
        self._min_step_pct = min_ratchet_step_pct
        self._trail_stages = tuple(trail_stages)
        self._breakeven_floor_gain_pct = breakeven_floor_gain_pct
        self._now_fn = now_fn
        # Per-execution (target, consecutive-failure count) for the
        # replace-failure loop hygiene. In-memory only: a restart
        # resetting the streak just means up to 3 more ERROR lines.
        self._ratchet_failures: dict[int, tuple[float, int]] = {}
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
        self._heal_engaged_without_live_stop()
        self._heal_naked_positions_without_state()

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
                # Silent-skip is the FEIM-37 bug class: any arming skip
                # must say why. (This position still has its live stop —
                # the skip only means the trail cannot engage.)
                log.warning(
                    "exit_policy.engagement_skipped",
                    extra={
                        "event": "exit_policy.engagement_skipped",
                        "execution_id": execution_id,
                        "symbol": symbol,
                        "reason": "no_entry_geometry",
                    },
                )
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
        if (
            self._trail_stages or self._breakeven_floor_gain_pct > 0.0
        ) and entry_price is None:
            entry_price = self._entry_price(execution_id)
        effective_pct, active_stage_gain = self._staged_trail_pct(
            state.trail_distance_pct,
            entry_price=entry_price,
            high_water=high_water,
        )
        # Trail-derived component rounds DOWN to the increment (a
        # protective sell-stop a fraction lower is semantically
        # identical); the breakeven floor rounds entry UP inside
        # `_breakeven_floor`. Their max is always a clean, broker-valid
        # price (ATRC hotfix 2026-08-11 — Alpaca 42210000).
        trail_target = quantize_price(
            high_water * (1.0 - effective_pct / 100.0), direction="down"
        )
        target = self._breakeven_floor(
            trail_target, entry_price=entry_price, high_water=high_water
        )
        floor_bound = target > trail_target
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
            # Atomic PATCH contract: on a transient failure the original
            # stop is still live — persist the high-water advance and
            # retry next tick. BUT a PATCH against a DEAD order (the
            # tracked stop was canceled out from under the ratchet —
            # incident FEIM 2026-07-16) fails every tick forever without
            # ever re-establishing protection. Interrogate the target:
            # dead + position open → submit a FRESH stop at the computed
            # floor instead of retrying a doomed PATCH.
            if high_water > state.high_water_mark:
                upsert_exit_policy_state(
                    self._journal.db_path,
                    state.model_copy(update={"high_water_mark": high_water}),
                )
                state = state.model_copy(update={"high_water_mark": high_water})
            if self._selfheal_if_dead(
                live_stop=live_stop,
                target=target,
                last=last,
                replace_error=e,
            ):
                # The fresh stop was submitted at the (possibly floored)
                # target — mirror the normal-ratchet observability so the
                # floor's binding is never silent just because the PATCH
                # took the self-heal branch.
                if floor_bound:
                    self._log_breakeven_floor_applied(
                        execution_id=execution_id,
                        symbol=symbol,
                        entry_price=entry_price,
                        high_water=high_water,
                        trail_stop=trail_target,
                        floored_stop=target,
                    )
                self._ratchet_failures.pop(execution_id, None)
                return
            # Failure-loop hygiene: the retry itself is unchanged (next
            # tick recomputes and re-PATCHes), but after
            # `_RATCHET_STUCK_THRESHOLD` consecutive failures at the
            # SAME target the loop escalates once (`ratchet_stuck`,
            # ERROR — distinct event for ops/alerting) and further
            # identical-target failures log at DEBUG until the target
            # moves.
            prev = self._ratchet_failures.get(execution_id)
            count = prev[1] + 1 if prev is not None and prev[0] == target else 1
            self._ratchet_failures[execution_id] = (target, count)
            if count == _RATCHET_STUCK_THRESHOLD:
                log.error(
                    "exit_policy.ratchet_stuck",
                    extra={
                        "event": "exit_policy.ratchet_stuck",
                        "execution_id": execution_id,
                        "symbol": symbol,
                        "old_broker_order_id": live_stop.broker_order_id,
                        "target_stop_price": target,
                        "consecutive_failures": count,
                        "error": str(e),
                    },
                )
            log_fn = (
                log.debug if count >= _RATCHET_STUCK_THRESHOLD else log.error
            )
            log_fn(
                "exit_policy.ratchet_replace_failed",
                extra={
                    "event": "exit_policy.ratchet_replace_failed",
                    "execution_id": execution_id,
                    "symbol": symbol,
                    "old_broker_order_id": live_stop.broker_order_id,
                    "target_stop_price": target,
                    "consecutive_failures": count,
                    "error": str(e),
                },
            )
            return

        self._ratchet_failures.pop(execution_id, None)
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
        if floor_bound:
            # The floor was the binding constraint — it raised the
            # submitted stop above the trail-derived level. Logged once
            # per ratchet it binds (the min_ratchet_step gate above means
            # this only fires when the stop actually changed), not per
            # tick.
            self._log_breakeven_floor_applied(
                execution_id=execution_id,
                symbol=symbol,
                entry_price=entry_price,
                high_water=high_water,
                trail_stop=trail_target,
                floored_stop=target,
            )

    # ------------------------------------------------------------------
    # Self-heal (incident FEIM 2026-07-16): a trail-armed position whose
    # tracked stop was canceled out from under the ratchet must get a
    # FRESH stop, not a doomed PATCH retry (or, worse, silence once fill
    # ingestion records the cancel and the row leaves the live set).
    # ------------------------------------------------------------------

    def _selfheal_if_dead(
        self,
        *,
        live_stop: OrderRow,
        target: float,
        last: float,
        replace_error: Exception,
    ) -> bool:
        """PATCH-failure arm: True when the failed PATCH target turned
        out to be dead at the broker and a fresh stop was submitted (or
        the position is exiting some other way); False → the failure was
        transient, caller logs and retries next tick."""
        get_order_by_id = getattr(self._broker, "get_order_by_id", None)
        if get_order_by_id is None:
            return False
        try:
            current = get_order_by_id(live_stop.broker_order_id)
        except Exception:  # noqa: BLE001 — lookup failed; treat as transient
            return False
        if current is not None and current.status == "filled":
            return True  # stop filled — fill ingestion owns the exit
        if current is not None and current.status not in DEAD_ORDER_STATUSES:
            return False  # still live at the broker — transient PATCH failure
        disposition = current.status if current is not None else "canceled"
        log.warning(
            "exit_policy.tracked_stop_dead",
            extra={
                "event": "exit_policy.tracked_stop_dead",
                "execution_id": live_stop.execution_id,
                "symbol": live_stop.symbol,
                "old_broker_order_id": live_stop.broker_order_id,
                "broker_status": disposition,
                "replace_error": str(replace_error),
            },
        )
        self._place_fresh_stop(
            execution_id=live_stop.execution_id,
            symbol=live_stop.symbol,
            qty=live_stop.qty,
            target=target,
            last=last,
            old_row_id=live_stop.id,
            old_disposition=disposition,
            old_broker_order_id=live_stop.broker_order_id,
        )
        return True

    def _heal_engaged_without_live_stop(self) -> None:
        """Silent-blindness arm: a trail-engaged execution with NO live
        journaled stop at all (fill ingestion recorded the cancel, so
        `_live_protective_stops` no longer surfaces it — the ratchet
        would otherwise never look at the position again) gets a fresh
        GTC stop at its computed trail floor."""
        with connect(self._journal.db_path) as conn:
            rows = conn.execute(
                "SELECT s.* FROM exit_policy_state s "
                "WHERE s.trail_engaged = 1 AND NOT EXISTS ("
                "  SELECT 1 FROM orders o WHERE o.execution_id = s.execution_id "
                "  AND o.role IN ('stop', 'trailing_stop') "
                "  AND o.final_status IS NULL"
                ")"
            ).fetchall()
        for raw in rows:
            d = dict(raw)
            d["trail_engaged"] = bool(d["trail_engaged"])
            state = ExitPolicyStateRow(**d)
            try:
                self._heal_one_naked(state)
            except Exception as e:  # noqa: BLE001
                log.error(
                    "exit_policy.selfheal_error",
                    extra={
                        "event": "exit_policy.selfheal_error",
                        "execution_id": state.execution_id,
                        "symbol": state.symbol,
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )

    def _heal_one_naked(self, state: ExitPolicyStateRow) -> None:
        if not has_open_position(self._journal.db_path, state.symbol):
            return
        orders = get_orders_for_execution(
            self._journal.db_path, state.execution_id
        )
        entry = next((o for o in orders if o.role == "entry"), None)
        if entry is None or entry.qty <= 0:
            return
        qty = entry.realized_fill_qty or entry.qty
        quote = self._broker.get_quote(state.symbol)
        last = float(quote.last)
        entry_price = entry.realized_fill_price or entry.pre_submission_last
        high_water = max(state.high_water_mark, last)
        effective_pct, _ = self._staged_trail_pct(
            state.trail_distance_pct,
            entry_price=entry_price,
            high_water=high_water,
        )
        target = self._breakeven_floor(
            quantize_price(
                high_water * (1.0 - effective_pct / 100.0), direction="down"
            ),
            entry_price=entry_price,
            high_water=high_water,
        )
        log.warning(
            "exit_policy.naked_trail_detected",
            extra={
                "event": "exit_policy.naked_trail_detected",
                "execution_id": state.execution_id,
                "symbol": state.symbol,
                "high_water_mark": high_water,
                "target_stop_price": target,
            },
        )
        self._place_fresh_stop(
            execution_id=state.execution_id,
            symbol=state.symbol,
            qty=qty,
            target=target,
            last=last,
            old_row_id=None,
            old_disposition=None,
            old_broker_order_id=None,
            mode="rearmed_engaged",
        )

    def _heal_naked_positions_without_state(self) -> None:
        """Stateless arm (FEIM follow-up, execution 37): a position can
        be naked with NO `exit_policy_state` row at all — its stop was
        canceled by order surgery BEFORE the trail ever engaged, and the
        engagement scan only iterates live journaled stops, so it
        silently never armed. Broker truth drives this arm (the same
        condition the naked tripwire detects): every open broker
        position with ZERO live sell orders at the broker gets a fresh
        GTC stop immediately —

          - gain past the activation threshold → engage PROPERLY: place
            a trailing stop at the staged/computed floor and create the
            state row with HWM initialized from the current price
            (mode=engaged_fresh);
          - otherwise → restore the position's ORIGINAL entry-time stop
            price from the journal (mode=restored_original).

        Every skip logs WARN with a reason — silent-skip is the bug
        class this arm exists to kill. Legacy brokers/fakes without the
        get_positions/get_open_orders surface leave the arm inert."""
        get_positions = getattr(self._broker, "get_positions", None)
        get_open_orders = getattr(self._broker, "get_open_orders", None)
        if get_positions is None or get_open_orders is None:
            return
        try:
            open_positions = [p for p in get_positions() if p.qty > 0]
            if not open_positions:
                return
            open_orders = get_open_orders()
        except Exception as e:  # noqa: BLE001 — broker read failed; next tick
            log.warning(
                "exit_policy.selfheal_broker_read_failed",
                extra={
                    "event": "exit_policy.selfheal_broker_read_failed",
                    "error": str(e),
                    "error_class": type(e).__name__,
                },
            )
            return
        sell_covered = {o.symbol for o in open_orders if o.side == "sell"}
        for pos in open_positions:
            if pos.symbol in sell_covered:
                continue
            try:
                self._heal_one_stateless(pos)
            except Exception as e:  # noqa: BLE001
                log.error(
                    "exit_policy.selfheal_error",
                    extra={
                        "event": "exit_policy.selfheal_error",
                        "symbol": pos.symbol,
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )

    def _heal_one_stateless(self, pos: Any) -> None:
        execution_id = self._latest_accepted_execution_id(pos.symbol)
        if execution_id is None:
            log.warning(
                "exit_policy.selfheal_skipped",
                extra={
                    "event": "exit_policy.selfheal_skipped",
                    "symbol": pos.symbol,
                    "reason": "no_journal_lineage",
                },
            )
            return
        if get_exit_policy_state(
            self._journal.db_path, execution_id=execution_id
        ) is not None:
            # The engaged-state arm owns this execution (it ran earlier
            # this tick); if it declined, its own logs say why. A
            # duplicate stop from this arm would double-sell.
            return
        # A live journaled sell of ANY role (protective leg, thesis
        # close, displacement close) holds or is about to consume the
        # shares — placing a stop beside it would double-sell. The
        # broker check above normally catches this; the journal check
        # covers the submit-vs-open-orders visibility race.
        with connect(self._journal.db_path) as conn:
            live_sell = conn.execute(
                "SELECT 1 FROM orders WHERE execution_id = ? "
                "AND side = 'sell' AND final_status IS NULL LIMIT 1",
                (execution_id,),
            ).fetchone()
        if live_sell is not None:
            log.warning(
                "exit_policy.selfheal_skipped",
                extra={
                    "event": "exit_policy.selfheal_skipped",
                    "symbol": pos.symbol,
                    "execution_id": execution_id,
                    "reason": "live_journal_sell_not_at_broker",
                },
            )
            return
        geometry = self._initial_geometry(execution_id)
        if geometry is None:
            log.warning(
                "exit_policy.selfheal_skipped",
                extra={
                    "event": "exit_policy.selfheal_skipped",
                    "symbol": pos.symbol,
                    "execution_id": execution_id,
                    "reason": "no_entry_geometry",
                },
            )
            return
        entry_price, initial_risk, trail_pct = geometry
        quote = self._broker.get_quote(pos.symbol)
        last = float(quote.last)

        log.warning(
            "exit_policy.naked_trail_detected",
            extra={
                "event": "exit_policy.naked_trail_detected",
                "execution_id": execution_id,
                "symbol": pos.symbol,
                "qty": pos.qty,
                "last": last,
                "stateless": True,
            },
        )

        if last >= entry_price + self._activation_r * initial_risk:
            # Gain supports engagement — arm properly: trailing stop at
            # the staged/computed floor, state row with HWM = current
            # price. Arming does NOT require a pre-existing live stop
            # leg (the FEIM-37 silent-skip).
            effective_pct, _ = self._staged_trail_pct(
                trail_pct, entry_price=entry_price, high_water=last
            )
            target = self._breakeven_floor(
                quantize_price(
                    last * (1.0 - effective_pct / 100.0), direction="down"
                ),
                entry_price=entry_price,
                high_water=last,
            )
            new_row_id = self._place_fresh_stop(
                execution_id=execution_id,
                symbol=pos.symbol,
                qty=pos.qty,
                target=target,
                last=last,
                old_row_id=None,
                old_disposition=None,
                old_broker_order_id=None,
                mode="engaged_fresh",
            )
            if new_row_id is not None:
                upsert_exit_policy_state(
                    self._journal.db_path,
                    ExitPolicyStateRow(
                        execution_id=execution_id,
                        symbol=pos.symbol,
                        trail_distance_pct=trail_pct,
                        trail_engaged=True,
                        high_water_mark=last,
                        stop_order_journal_id=new_row_id,
                    ),
                )
            return

        # Below activation — restore the ORIGINAL entry-time stop level
        # (entry_price − initial_risk is exactly the earliest journaled
        # stop's price; `_initial_geometry` derived it from that row).
        # That price was broker-valid once, but the float subtraction
        # round-trip can drift off the increment grid — quantize DOWN
        # (protective sell-stop a hair lower is semantically identical).
        original_stop = quantize_price(
            entry_price - initial_risk, direction="down"
        )
        if original_stop <= 0:
            log.warning(
                "exit_policy.selfheal_skipped",
                extra={
                    "event": "exit_policy.selfheal_skipped",
                    "symbol": pos.symbol,
                    "execution_id": execution_id,
                    "reason": "invalid_original_stop",
                },
            )
            return
        self._place_fresh_stop(
            execution_id=execution_id,
            symbol=pos.symbol,
            qty=pos.qty,
            target=original_stop,
            last=last,
            old_row_id=None,
            old_disposition=None,
            old_broker_order_id=None,
            mode="restored_original",
            role="stop",
        )

    def _latest_accepted_execution_id(self, symbol: str) -> int | None:
        """Newest accepted execution for a held symbol — the same
        journal-lineage resolution displacement and the boot re-arm
        sweep use (broker truth says the position is open; the newest
        accepted execution is the entry that opened it)."""
        with connect(self._journal.db_path) as conn:
            row = conn.execute(
                "SELECT e.id AS execution_id FROM executions e "
                "JOIN proposals p ON p.id = e.proposal_id "
                "WHERE e.decision = 'accepted' AND p.symbol = ? "
                "ORDER BY e.id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        return int(row["execution_id"]) if row is not None else None

    def _place_fresh_stop(
        self,
        *,
        execution_id: int,
        symbol: str,
        qty: int,
        target: float,
        last: float,
        old_row_id: int | None,
        old_disposition: str | None,
        old_broker_order_id: str | None,
        mode: str = "replaced_dead",
        role: str = "trailing_stop",
    ) -> int | None:
        """Submit a fresh GTC stop at the computed price (clamped below
        the current quote so an in-the-money stop doesn't get rejected),
        journal it, complete the dead row (when known), and rotate the
        ratchet's pointer. Loud by design — this only runs when an open
        position lost (or never had) broker-side protection. Returns the
        new journal row id, or None when the submit failed."""
        # Submission seam: everything broker-bound is quantized DOWN to
        # a valid increment here as well. A floor-derived target is
        # already clean 2dp (`_breakeven_floor` ceils entry), so the
        # down-quantize is the identity for it — the floor invariant
        # (stop >= entry) survives.
        stop_price = (
            quantize_price(target, direction="down")
            if target < last
            else quantize_price(
                last * (1.0 - self._min_step_pct / 100.0), direction="down"
            )
        )
        client_order_id = (
            f"trail-heal-{execution_id}-{self._now_fn().timestamp():.0f}"
        )
        req = OrderRequest(
            symbol=symbol,
            side="sell",
            qty=qty,
            order_type="stop",
            tif="gtc",
            stop_price=stop_price,
            client_order_id=client_order_id,
        )
        try:
            resp = self._broker.submit_order(req)
        except Exception as e:  # noqa: BLE001
            log.error(
                "exit_policy.selfheal_submit_failed",
                extra={
                    "event": "exit_policy.selfheal_submit_failed",
                    "execution_id": execution_id,
                    "symbol": symbol,
                    "stop_price": stop_price,
                    "mode": mode,
                    "error": str(e),
                },
            )
            return None
        new_row_id = insert_order(
            self._journal.db_path,
            OrderRow(
                execution_id=execution_id,
                role=role,
                symbol=symbol,
                side="sell",
                order_type="stop",
                qty=qty,
                tif="gtc",
                stop_price=stop_price,
                broker_order_id=resp.broker_order_id,
                submitted_at=resp.submitted_at,
                final_status=None,
                notes=(
                    f"trail_selfheal fresh stop mode={mode}"
                    + (
                        f" replacing dead {old_broker_order_id} "
                        f"({old_disposition})"
                        if old_broker_order_id is not None
                        else ""
                    )
                ),
            ),
        )
        if old_row_id is not None and old_disposition is not None:
            self._record_order_outcome(
                old_row_id,
                old_disposition,
                fill_price=None,
                fill_qty=None,
                fill_at=None,
            )
        set_stop_order_journal_id(
            self._journal.db_path,
            execution_id=execution_id,
            order_journal_id=new_row_id,
        )
        log.warning(
            "exit_policy.stop_selfhealed",
            extra={
                "event": "exit_policy.stop_selfhealed",
                "execution_id": execution_id,
                "symbol": symbol,
                "stop_price": stop_price,
                "mode": mode,
                "broker_order_id": resp.broker_order_id,
                "replaced_dead_broker_order_id": old_broker_order_id,
            },
        )
        return new_row_id

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

    def _breakeven_floor(
        self, target: float, *, entry_price: float | None, high_water: float
    ) -> float:
        """Breakeven floor (study policy E): once the high-water gain vs
        entry clears `breakeven_floor_gain_pct`, the stop may never sit
        below entry. Applied AFTER the width/stage math as a pure lower
        bound — it only ever RAISES the target (never narrows the band,
        never lowers a computed stop). HWM-based (peak gain), not current
        price. Derived from config + current HWM every call, so a config
        change applies to already-open positions. Off (returns `target`
        unchanged) when the threshold is 0 or the entry price is unknown.

        The floor component is the ENTRY FILL, which can be sub-penny
        (ATRC 37.3297 — Alpaca rejects it, 42210000): quantize UP to the
        next valid increment. Up, not down, because the floor invariant
        is stop >= entry — ceiling preserves it; flooring would break it
        by a hair.
        """
        if self._breakeven_floor_gain_pct <= 0.0:
            return target
        if entry_price is None or entry_price <= 0:
            return target
        threshold = entry_price * (1.0 + self._breakeven_floor_gain_pct / 100.0)
        if high_water < threshold:
            return target
        return max(target, quantize_price(entry_price, direction="up"))

    def _log_breakeven_floor_applied(
        self,
        *,
        execution_id: int,
        symbol: str,
        entry_price: float | None,
        high_water: float,
        trail_stop: float,
        floored_stop: float,
    ) -> None:
        log.info(
            "exit_policy.breakeven_floor_applied",
            extra={
                "event": "exit_policy.breakeven_floor_applied",
                "execution_id": execution_id,
                "symbol": symbol,
                "entry_price": entry_price,
                "high_water_mark": high_water,
                "trail_stop": trail_stop,
                "floored_stop": floored_stop,
            },
        )

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
