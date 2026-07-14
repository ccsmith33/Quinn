"""Feature A — thesis-review coordinator.

Drives the open-position thesis-review pipeline:

  reconciler tick (market hours, post-reconcile)
   → for each due schedule:
     - confirm position is still open (stop/TP may have closed it)
     - build review context (price, days held, filings since entry)
     - call ThesisReviewer (Opus)
     - apply decision: hold | close | adjust_stop | adjust_take_profit
     - write follow-up schedule (hold / adjust_*) or none (close)

CRITICAL safety constraint (per task description + reviewer pre-read):
- For `adjust_stop`, the broker's atomic `replace_stop_order` call is
  used: Alpaca's PATCH /v2/orders/{id} replaces the stop_price in
  place. The broker either accepts (new order id, NO gap) or rejects
  (original stop still live). There is never a moment where the
  position is uncovered. If the replace fails, the old stop is intact
  and the coordinator reschedules the review for tomorrow.
- For `close`, the sell-market order goes out first; only after broker
  ack do we cancel the GTC stop and take-profit legs. A successful
  market sell + delayed cancel still leaves the GTC legs lingering on a
  zero-qty position, which Alpaca rejects on trigger — acceptable
  because the qty is already zero and the position is closed.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from broker.protocol import BrokerAdapter, OrderRequest
from config.loader import TrailStage
from journal.exit_policy import get_exit_policy_state, set_stop_order_journal_id
from journal.models import (
    OrderRow,
    ThesisReviewScheduleRow,
)
from journal.repo import (
    connect,
    find_due_thesis_reviews,
    get_orders_for_execution,
    get_proposal_by_id,
    has_open_position,
    insert_order,
    insert_thesis_review_schedule,
)
from observability.log_port import get_logger

from analyzer.thesis_review import (
    ThesisAdjustStop,
    ThesisAdjustTakeProfit,
    ThesisClose,
    ThesisHold,
    ThesisReviewContext,
    ThesisReviewer,
    ThesisReviewMalformed,
)

log = get_logger(__name__)


# Default cadence for follow-up reviews after a `hold`. Per task spec.
HOLD_RESCHEDULE_DAYS = 7

# Jurisdiction zones (event-reviews doctrine). Zone 1: not yet
# trail-armed — the reviewer's judgment is primary. Zone 3: armed and/or
# HWM gain past the first trail-stage milestone (or +20% when no stages
# are configured) — the staged-trail ALGORITHM is primary and the
# reviewer may act only on a cited new-information event. Zone 2 is the
# transitional band: not armed but HWM gain at/above the +10% arming
# vicinity.
_ZONE2_GAIN_PCT = 10.0
_DEFAULT_ZONE3_GAIN_PCT = 20.0


def compute_position_zone(
    *, trail_armed: bool, hwm_gain_pct: float, zone3_gain_pct: float
) -> int:
    """Map (armed, HWM gain) onto the 1/2/3 jurisdiction zones."""
    if trail_armed or hwm_gain_pct >= zone3_gain_pct:
        return 3
    if hwm_gain_pct >= _ZONE2_GAIN_PCT:
        return 2
    return 1

# Broker order statuses that mean the order can no longer fill — the
# §3.3 step-5 fallback re-places protection only when the PATCH target
# is in one of these (or gone). Anything else (accepted / new /
# partially_filled / pending_*) means the original order is still live
# at the broker, so the failed replace is transient: retry tomorrow.
#
# `done_for_day` is deliberately NOT here (W2-L3 alignment with the
# converter's healable set): an Alpaca GTC order parked done_for_day
# resumes next session — it is standing protection, and re-placing
# over it would race a second full-qty sell against it (the oversell
# class this epic excises). Treat as transient; the PATCH succeeds
# when the order resumes.
_DEAD_ORDER_STATUSES = frozenset(
    {"expired", "canceled", "rejected", "replaced",
     "suspended", "stopped"}
)


class _JournalLike(Protocol):
    db_path: str


# ---------------------------------------------------------------------------
# WS1 §7.4 journal-API ports (D-079 §3.3 steps 3/5). The coordinator
# consumes `get_live_protective_order` / `record_order_outcome` via
# injected callables; the defaults late-import WS1's repo functions so
# composition needs no change once the merge train lands, and degrade
# gracefully while this branch stands alone (repo.py is WS1's file this
# epic — delta §1).
# ---------------------------------------------------------------------------


def _default_get_live_protective_order(
    db_path: str, *, execution_id: int, roles: tuple[str, ...]
) -> OrderRow | None:
    """Latest orders row for the execution in `roles` that is still
    pending a terminal disposition (`final_status IS NULL` — the D-078
    §7.4 lifecycle contract)."""
    try:
        # WS1 §7.4 — lands with the merge train; missing name raises
        # ImportError until then, so the except arm below covers it.
        from journal.repo import (
            get_live_protective_order,
        )
    except ImportError:
        placeholders = ",".join("?" for _ in roles)
        with connect(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM orders "
                f"WHERE execution_id = ? AND role IN ({placeholders}) "
                "AND final_status IS NULL "
                "ORDER BY id DESC LIMIT 1",
                (execution_id, *roles),
            ).fetchone()
        return OrderRow(**dict(row)) if row is not None else None
    return get_live_protective_order(db_path, execution_id, roles=roles)


def _default_record_order_outcome(
    db_path: str,
    order_id: int,
    final_status: str,
    *,
    fill_price: float | None,
    fill_qty: int | None,
    fill_at: dt.datetime | None,
) -> None:
    """One-time NULL→value completion of the fill-outcome columns
    (NFR-16a) — `record_order_outcome` is WS1's sole-UPDATE write path,
    so there is deliberately NO local SQL fallback here. In the window
    before WS1 merges, the old row simply keeps `final_status` NULL;
    the latest-row-wins live lookup stays correct."""
    try:
        # WS1 §7.4 — lands with the merge train; missing name raises
        # ImportError until then, so the except arm below covers it.
        from journal.repo import (
            record_order_outcome,
        )
    except ImportError:
        log.warning(
            "thesis_coordinator.record_order_outcome_unavailable",
            extra={
                "event": "thesis_coordinator.record_order_outcome_unavailable",
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


class _FilingsLookup(Protocol):
    """Returns a small text summary of filings for a given issuer
    published on/after a cutoff date. Implementation provided at
    composition time so this module stays decoupled from journal repo."""

    def __call__(self, *, issuer_ticker: str | None, since: dt.datetime) -> str: ...


class ThesisReviewCoordinator:
    """Reconciler-driven coordinator for Feature A. One instance per agent
    process. Stateless across ticks — schedule + outcome rows are the
    source of truth.
    """

    def __init__(
        self,
        *,
        journal: _JournalLike,
        broker: BrokerAdapter,
        reviewer: ThesisReviewer,
        filings_lookup: _FilingsLookup,
        now_fn: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
        default_horizon_days: int = 14,
        # D-079 §3.4 belt-and-suspenders: a raised TP must not drop the
        # position's R:R below the §3.2 floor against the current stop.
        min_reward_risk: float = 1.5,
        # Jurisdiction zones: the first stage's gain_pct is the Zone-3
        # line (falls back to +20% when no stages are configured).
        trail_stages: Sequence[TrailStage] = (),
        # WS1 §7.4 ports — see the module-level defaults above.
        get_live_protective_order: (
            Callable[..., OrderRow | None] | None
        ) = None,
        record_order_outcome: Callable[..., None] | None = None,
    ) -> None:
        self._journal = journal
        self._broker = broker
        self._reviewer = reviewer
        self._filings_lookup = filings_lookup
        self._now_fn = now_fn
        self._default_horizon_days = default_horizon_days
        self._min_reward_risk = min_reward_risk
        self._zone3_gain_pct = (
            trail_stages[0].gain_pct if trail_stages else _DEFAULT_ZONE3_GAIN_PCT
        )
        self._get_live_protective_order = get_live_protective_order or (
            lambda *, execution_id, roles: _default_get_live_protective_order(
                self._journal.db_path, execution_id=execution_id, roles=roles
            )
        )
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

    async def run_tick(self) -> None:
        """One coordinator tick. Picks up all schedules whose `due_at <=
        now` and processes them sequentially. Errors on a single review
        are logged and do not stop subsequent ones."""
        now = self._now_fn()
        due = find_due_thesis_reviews(self._journal.db_path, now=now)
        if not due:
            return
        for schedule in due:
            try:
                await self._process_schedule(schedule, now=now)
            except Exception as e:  # noqa: BLE001
                log.error(
                    "thesis_coordinator.process_error",
                    extra={
                        "event": "thesis_coordinator.process_error",
                        "schedule_id": schedule.id,
                        "execution_id": schedule.execution_id,
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )

    async def _process_schedule(
        self, schedule: ThesisReviewScheduleRow, *, now: dt.datetime
    ) -> None:
        # Fetch the execution + proposal (proposal carries the original
        # thesis text, conviction, sizing, stops).
        execution = _get_execution(self._journal.db_path, schedule.execution_id)
        if execution is None:
            log.warning(
                "thesis_coordinator.execution_missing",
                extra={
                    "event": "thesis_coordinator.execution_missing",
                    "schedule_id": schedule.id,
                    "execution_id": schedule.execution_id,
                },
            )
            return
        proposal = get_proposal_by_id(self._journal.db_path, execution.proposal_id)
        if proposal is None:
            log.warning(
                "thesis_coordinator.proposal_missing",
                extra={
                    "event": "thesis_coordinator.proposal_missing",
                    "schedule_id": schedule.id,
                    "proposal_id": execution.proposal_id,
                },
            )
            return
        if proposal.symbol is None:
            return  # defensive: trade_proposal rows always have a symbol

        # Position-still-open guard. AC: if stop/TP fired between the
        # last reconcile and now, the schedule should be discarded
        # (don't write a thesis_reviews row for a closed position).
        if not has_open_position(self._journal.db_path, proposal.symbol):
            log.info(
                "thesis_coordinator.position_already_closed",
                extra={
                    "event": "thesis_coordinator.position_already_closed",
                    "schedule_id": schedule.id,
                    "execution_id": execution.id,
                    "symbol": proposal.symbol,
                },
            )
            return

        # Build review context.
        try:
            ctx = self._build_review_context(
                schedule=schedule,
                execution=execution,
                proposal=proposal,
                now=now,
            )
        except _BuildContextError as e:
            log.warning(
                "thesis_coordinator.context_build_failed",
                extra={
                    "event": "thesis_coordinator.context_build_failed",
                    "schedule_id": schedule.id,
                    "error": str(e),
                },
            )
            return

        log.info(
            "thesis_coordinator.review_starting",
            extra={
                "event": "thesis_coordinator.review_starting",
                "schedule_id": schedule.id,
                "execution_id": execution.id,
                "symbol": proposal.symbol,
                "days_held": ctx.days_held,
                "current_price": ctx.current_price,
                "pct_change_since_entry": ctx.pct_change_since_entry,
            },
        )

        result = await self._reviewer.review(ctx)

        log.info(
            "thesis_coordinator.review_decision",
            extra={
                "event": "thesis_coordinator.review_decision",
                "schedule_id": schedule.id,
                "execution_id": execution.id,
                "symbol": proposal.symbol,
                "decision": _decision_string(result),
            },
        )

        # Apply decision.
        if isinstance(result, ThesisHold):
            self._reschedule(execution.id, when=now + dt.timedelta(days=HOLD_RESCHEDULE_DAYS), reason="hold")
            return
        if isinstance(result, ThesisClose):
            self._apply_close(execution.id, proposal.symbol, ctx)
            return
        if isinstance(result, ThesisAdjustStop):
            self._apply_adjust_stop(
                execution_id=execution.id,
                symbol=proposal.symbol,
                new_stop_price=result.new_stop_price,
                now=now,
                schedule_id=schedule.id or 0,
                current_price=ctx.current_price,
            )
            return
        if isinstance(result, ThesisAdjustTakeProfit):
            self._apply_adjust_take_profit(
                execution_id=execution.id,
                symbol=proposal.symbol,
                new_tp_price=result.new_tp_price,
                now=now,
                schedule_id=schedule.id or 0,
                ctx=ctx,
            )
            return
        # Malformed → leave the position alone, schedule a retry for tomorrow
        # so a transient prompt failure doesn't strand the slot. Not part of
        # the AC but a reasonable defense-in-depth default.
        if isinstance(result, ThesisReviewMalformed):
            self._reschedule(
                execution.id,
                when=now + dt.timedelta(days=1),
                reason="hold",
            )
            return

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _apply_close(
        self,
        execution_id: int,
        symbol: str,
        ctx: ThesisReviewContext,
    ) -> None:
        """Submit a sell-market closing order, then cancel the open GTC
        stop and take-profit legs (best-effort).

        Order: market sell FIRST, then cancel GTCs. If the market sell
        fails, no cancel is attempted (the protective legs remain live).
        """
        # Determine quantity to sell — the sizing engine recorded
        # realized qty on entry, but the safest source is the latest
        # entry order's filled qty.
        orders = get_orders_for_execution(self._journal.db_path, execution_id)
        entry = next((o for o in orders if o.role == "entry"), None)
        if entry is None or entry.qty <= 0:
            log.warning(
                "thesis_coordinator.close_skipped_no_entry_qty",
                extra={
                    "event": "thesis_coordinator.close_skipped_no_entry_qty",
                    "execution_id": execution_id,
                },
            )
            return

        client_order_id = f"thesis-close-exec-{execution_id}"
        req = OrderRequest(
            symbol=symbol,
            side="sell",
            qty=entry.qty,
            order_type="market",
            tif="day",
            client_order_id=client_order_id,
        )
        try:
            submitted = self._broker.submit_order(req)
        except Exception as e:  # noqa: BLE001
            log.error(
                "thesis_coordinator.close_submit_failed",
                extra={
                    "event": "thesis_coordinator.close_submit_failed",
                    "execution_id": execution_id,
                    "error": str(e),
                },
            )
            return

        insert_order(
            self._journal.db_path,
            OrderRow(
                execution_id=execution_id,
                role="thesis_close",
                symbol=symbol,
                side="sell",
                order_type="market",
                qty=entry.qty,
                tif="day",
                broker_order_id=submitted.broker_order_id,
                submitted_at=submitted.submitted_at,
                final_status=submitted.status,
                notes="thesis_review:close",
            ),
        )

        # Cancel the GTC stop + take-profit, best effort. Failures here
        # are not fatal — the position is closing.
        for o in orders:
            if o.role in ("stop", "take_profit"):
                try:
                    self._broker.cancel_order(o.broker_order_id)
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "thesis_coordinator.cancel_failed",
                        extra={
                            "event": "thesis_coordinator.cancel_failed",
                            "execution_id": execution_id,
                            "role": o.role,
                            "broker_order_id": o.broker_order_id,
                            "error": str(e),
                        },
                    )

        log.info(
            "thesis_coordinator.close_submitted",
            extra={
                "event": "thesis_coordinator.close_submitted",
                "execution_id": execution_id,
                "symbol": symbol,
                "qty": entry.qty,
                "broker_order_id": submitted.broker_order_id,
            },
        )

    def _apply_adjust_stop(
        self,
        *,
        execution_id: int,
        symbol: str,
        new_stop_price: float,
        now: dt.datetime,
        schedule_id: int,
        current_price: float,
    ) -> None:
        """SAFETY-CRITICAL (D-079 §3.3): atomic stop-price replacement
        via the broker adapter's `replace_stop_order` (Alpaca PATCH
        /v2/orders/{id}). The broker either accepts the replacement
        atomically (new order id, position never uncovered) or rejects
        and the original stop remains live.

          1. Live stop = newest `orders` row, role IN
             ('stop','trailing_stop'), final_status IS NULL (§7.4 port).
          2. Atomic replace. Success → journal a replacement row (role
             preserved, final_status NULL), record the old row
             `replaced` (§7.4), rotate the trailing ratchet's pointer
             (§3.5 step 4), reschedule +7d.
          3. Failure → §3.3 step-5 fallback: branch on what actually
             happened to the PATCH target (`_adjust_stop_fallback`).

        Direction: RAISE-ONLY (event-reviews doctrine — one-directional
        review powers). The new stop must sit strictly ABOVE the live
        stop (never loosen protection / add risk) and strictly BELOW
        the current price for a long — a stop at/above the price is a
        `close` wearing an adjust_stop costume, and it would fire on
        the next tick.
        """
        orders = get_orders_for_execution(self._journal.db_path, execution_id)
        entry = next((o for o in orders if o.role == "entry"), None)
        old_stop = self._get_live_protective_order(
            execution_id=execution_id, roles=("stop", "trailing_stop")
        )
        if entry is None or old_stop is None or entry.qty <= 0:
            # No journaled live stop (e.g. PDT-era position before the
            # WS3 conversion). The scanner still protects those; the
            # window is one merge step (§3.3 item 6).
            #
            # D-087 fix: this branch previously returned WITHOUT writing
            # a follow-up schedule, which orphaned the position from all
            # future thesis re-review (the latest schedule row had
            # already fired but no successor existed, so
            # `find_due_thesis_reviews` never surfaced it again). A
            # missing journaled stop is a *transient* journal-vs-broker
            # gap that the WS3 conversion closes, NOT a terminal state —
            # so re-arm a +1d retry whenever the position is STILL OPEN.
            # If the position has since closed, stay terminal (no
            # reschedule): there is nothing left to review.
            position_open = has_open_position(self._journal.db_path, symbol)
            log.warning(
                "thesis_coordinator.adjust_stop_skipped",
                extra={
                    "event": "thesis_coordinator.adjust_stop_skipped",
                    "execution_id": execution_id,
                    "reason": "missing_entry_or_stop",
                    "position_open": position_open,
                    "rescheduled": position_open,
                },
            )
            if position_open:
                self._reschedule(
                    execution_id,
                    when=now + dt.timedelta(days=1),
                    reason="adjust_stop",
                )
            return

        if new_stop_price <= 0 or new_stop_price >= current_price:
            log.warning(
                "thesis_coordinator.adjust_stop_rejected",
                extra={
                    "event": "thesis_coordinator.adjust_stop_rejected",
                    "execution_id": execution_id,
                    "new_stop_price": new_stop_price,
                    "current_price": current_price,
                    "reason": "stop_not_below_current_price",
                },
            )
            self._reschedule(
                execution_id, when=now + dt.timedelta(days=1), reason="adjust_stop"
            )
            return

        # ONE-DIRECTIONAL review powers (event-reviews doctrine): the
        # reviewer may only RAISE a stop. A new stop at/below the live
        # stop would loosen protection — that is added risk, which no
        # review outcome is allowed to create. Reject and retry
        # tomorrow (the position keeps its current, tighter stop).
        if (
            old_stop.stop_price is not None
            and new_stop_price <= old_stop.stop_price
        ):
            log.warning(
                "thesis_coordinator.adjust_stop_rejected",
                extra={
                    "event": "thesis_coordinator.adjust_stop_rejected",
                    "execution_id": execution_id,
                    "new_stop_price": new_stop_price,
                    "current_stop_price": old_stop.stop_price,
                    "reason": "raise_only_stop_not_above_current_stop",
                },
            )
            self._reschedule(
                execution_id, when=now + dt.timedelta(days=1), reason="adjust_stop"
            )
            return

        client_order_id = f"thesis-adjstop-{schedule_id}"
        try:
            replaced = self._broker.replace_stop_order(
                old_stop.broker_order_id,
                new_stop_price=new_stop_price,
                client_order_id=client_order_id,
            )
        except Exception as e:  # noqa: BLE001
            self._adjust_stop_fallback(
                execution_id=execution_id,
                symbol=symbol,
                qty=entry.qty,
                old_stop=old_stop,
                new_stop_price=new_stop_price,
                now=now,
                schedule_id=schedule_id,
                replace_error=e,
            )
            return

        # Journal the replacement chain (§3.3 step 3): new row first
        # (role preserved, live = final_status NULL), then complete the
        # old row as 'replaced' via the §7.4 single-writer.
        new_row_id = insert_order(
            self._journal.db_path,
            OrderRow(
                execution_id=execution_id,
                role=old_stop.role,
                symbol=symbol,
                side="sell",
                order_type="stop",
                qty=entry.qty,
                tif="gtc",
                stop_price=new_stop_price,
                broker_order_id=replaced.broker_order_id,
                submitted_at=replaced.submitted_at,
                final_status=None,
                notes=(
                    f"thesis_review:adjust_stop atomic-replaced "
                    f"{old_stop.broker_order_id}"
                ),
            ),
        )
        self._record_order_outcome(
            old_stop.id, "replaced", fill_price=None, fill_qty=None, fill_at=None
        )
        # §3.3 step 4: keep the trailing ratchet pointed at the live
        # stop row. No-op when trailing isn't engaged for the execution.
        set_stop_order_journal_id(
            self._journal.db_path,
            execution_id=execution_id,
            order_journal_id=new_row_id,
        )

        log.info(
            "thesis_coordinator.stop_adjusted",
            extra={
                "event": "thesis_coordinator.stop_adjusted",
                "execution_id": execution_id,
                "symbol": symbol,
                "old_stop_price": old_stop.stop_price,
                "new_stop_price": new_stop_price,
                "old_broker_order_id": old_stop.broker_order_id,
                "new_broker_order_id": replaced.broker_order_id,
            },
        )

        self._reschedule(
            execution_id,
            when=now + dt.timedelta(days=HOLD_RESCHEDULE_DAYS),
            reason="adjust_stop",
        )

    def _adjust_stop_fallback(
        self,
        *,
        execution_id: int,
        symbol: str,
        qty: int,
        old_stop: OrderRow,
        new_stop_price: float,
        now: dt.datetime,
        schedule_id: int,
        replace_error: Exception,
    ) -> None:
        """§3.3 step 5 — re-place protection when the PATCH target is
        dead. Kills the bracket-mode reschedule-forever loop
        (exit-policy §5a) and covers the day-2+ book whose DAY legs
        expired together, until the WS3 conversion lands.

        Branches on `broker.get_order_by_id(old)` + broker position
        state:
          - old order filled            → lost race; fill ingestion owns
                                          the exit. Not an error, no
                                          reschedule (position closing).
          - position gone at broker     → no-op; the §2.2 tombstone path
                                          owns it.
          - old order still live        → transient PATCH failure; the
                                          original stop protects, retry
                                          tomorrow.
          - old dead + position open    → re-place fresh GTC protection,
                                          shape matching what died:
                                          surviving TP leg → cancel it
                                          and submit one OCO (two
                                          independent full-qty sells are
                                          impossible — §4.2 qty
                                          constraint); both legs dead →
                                          fresh OCO when the position
                                          has a TP on record, plain GTC
                                          stop otherwise.
        """
        get_order_by_id = getattr(self._broker, "get_order_by_id", None)
        if get_order_by_id is None:
            # Pre-WS1 adapter: cannot interrogate the PATCH target.
            # Conservative: assume the original stop survives (Alpaca's
            # PATCH contract) and retry tomorrow.
            log.error(
                "thesis_coordinator.stop_replace_failed",
                extra={
                    "event": "thesis_coordinator.stop_replace_failed",
                    "execution_id": execution_id,
                    "old_broker_order_id": old_stop.broker_order_id,
                    "error": str(replace_error),
                },
            )
            self._reschedule(
                execution_id, when=now + dt.timedelta(days=1), reason="adjust_stop"
            )
            return

        try:
            current = get_order_by_id(old_stop.broker_order_id)
        except Exception as e:  # noqa: BLE001
            log.error(
                "thesis_coordinator.stop_replace_failed",
                extra={
                    "event": "thesis_coordinator.stop_replace_failed",
                    "execution_id": execution_id,
                    "old_broker_order_id": old_stop.broker_order_id,
                    "error": f"{replace_error}; order lookup also failed: {e}",
                },
            )
            self._reschedule(
                execution_id, when=now + dt.timedelta(days=1), reason="adjust_stop"
            )
            return

        if current is not None and current.status == "filled":
            log.info(
                "thesis_coordinator.adjust_stop_lost_race",
                extra={
                    "event": "thesis_coordinator.adjust_stop_lost_race",
                    "execution_id": execution_id,
                    "symbol": symbol,
                    "old_broker_order_id": old_stop.broker_order_id,
                },
            )
            return

        if current is not None and current.status not in _DEAD_ORDER_STATUSES:
            # Original stop is still live at the broker — the PATCH
            # failure was transient. Position protected; retry tomorrow.
            log.error(
                "thesis_coordinator.stop_replace_failed",
                extra={
                    "event": "thesis_coordinator.stop_replace_failed",
                    "execution_id": execution_id,
                    "old_broker_order_id": old_stop.broker_order_id,
                    "broker_status": current.status,
                    "error": str(replace_error),
                },
            )
            self._reschedule(
                execution_id, when=now + dt.timedelta(days=1), reason="adjust_stop"
            )
            return

        position_open = any(
            p.symbol == symbol and p.qty > 0 for p in self._broker.get_positions()
        )
        if not position_open:
            log.info(
                "thesis_coordinator.adjust_stop_position_gone",
                extra={
                    "event": "thesis_coordinator.adjust_stop_position_gone",
                    "execution_id": execution_id,
                    "symbol": symbol,
                },
            )
            return

        # Old stop dead (expired/canceled/gone) + position open at the
        # broker → re-place fresh GTC protection.
        tp_row = self._get_live_protective_order(
            execution_id=execution_id, roles=("take_profit",)
        )
        tp_price: float | None = tp_row.limit_price if tp_row is not None else None
        tp_was_live = False
        if tp_row is not None:
            try:
                tp_current = get_order_by_id(tp_row.broker_order_id)
            except Exception:  # noqa: BLE001
                tp_current = None
            tp_was_live = (
                tp_current is not None
                and tp_current.status not in _DEAD_ORDER_STATUSES
                and tp_current.status != "filled"
            )
            if tp_was_live:
                # A second full-qty sell beside the surviving TP is
                # impossible (the TP holds the position's available
                # qty) — cancel it and rebuild the pair as one OCO.
                try:
                    self._broker.cancel_order(tp_row.broker_order_id)
                except Exception as e:  # noqa: BLE001
                    log.error(
                        "thesis_coordinator.adjust_stop_tp_cancel_failed",
                        extra={
                            "event": (
                                "thesis_coordinator.adjust_stop_tp_cancel_failed"
                            ),
                            "execution_id": execution_id,
                            "tp_broker_order_id": tp_row.broker_order_id,
                            "error": str(e),
                        },
                    )
                    # TP still holds the qty — an OCO would be rejected.
                    # The TP leg is live protection upside; retry the
                    # full rebuild tomorrow.
                    self._reschedule(
                        execution_id,
                        when=now + dt.timedelta(days=1),
                        reason="adjust_stop",
                    )
                    return

        try:
            stop_resp, tp_resp = self._broker.submit_oco_sell(
                symbol=symbol,
                qty=qty,
                stop_price=new_stop_price,
                limit_price=tp_price,
                client_order_id=f"thesis-adjstop-replace-{schedule_id}",
            )
        except Exception as e:  # noqa: BLE001
            log.error(
                "thesis_coordinator.adjust_stop_replace_dead_failed",
                extra={
                    "event": "thesis_coordinator.adjust_stop_replace_dead_failed",
                    "execution_id": execution_id,
                    "symbol": symbol,
                    "error": str(e),
                },
            )
            self._reschedule(
                execution_id, when=now + dt.timedelta(days=1), reason="adjust_stop"
            )
            return

        old_status = (
            current.status if current is not None
            and current.status in _DEAD_ORDER_STATUSES
            else "canceled"
        )
        new_stop_row_id = insert_order(
            self._journal.db_path,
            OrderRow(
                execution_id=execution_id,
                role="stop",
                symbol=symbol,
                side="sell",
                order_type="stop",
                qty=qty,
                tif="gtc",
                stop_price=new_stop_price,
                broker_order_id=stop_resp.broker_order_id,
                submitted_at=stop_resp.submitted_at,
                final_status=None,
                notes=(
                    "thesis_review:adjust_stop re-placed dead stop "
                    f"{old_stop.broker_order_id} ({old_status})"
                ),
            ),
        )
        self._record_order_outcome(
            old_stop.id, old_status, fill_price=None, fill_qty=None, fill_at=None
        )
        if tp_row is not None and tp_resp is not None:
            insert_order(
                self._journal.db_path,
                OrderRow(
                    execution_id=execution_id,
                    role="take_profit",
                    symbol=symbol,
                    side="sell",
                    order_type="limit",
                    qty=qty,
                    tif="gtc",
                    limit_price=tp_price,
                    broker_order_id=tp_resp.broker_order_id,
                    submitted_at=tp_resp.submitted_at,
                    final_status=None,
                    notes=(
                        "thesis_review:adjust_stop re-placed TP leg "
                        f"{tp_row.broker_order_id} as OCO"
                    ),
                ),
            )
            self._record_order_outcome(
                tp_row.id,
                "canceled" if tp_was_live else "expired",
                fill_price=None,
                fill_qty=None,
                fill_at=None,
            )
        set_stop_order_journal_id(
            self._journal.db_path,
            execution_id=execution_id,
            order_journal_id=new_stop_row_id,
        )

        log.info(
            "thesis_coordinator.adjust_stop_replaced_dead_protection",
            extra={
                "event": "thesis_coordinator.adjust_stop_replaced_dead_protection",
                "execution_id": execution_id,
                "symbol": symbol,
                "new_stop_price": new_stop_price,
                "tp_price": tp_price,
                "oco": tp_price is not None,
            },
        )
        self._reschedule(
            execution_id,
            when=now + dt.timedelta(days=HOLD_RESCHEDULE_DAYS),
            reason="adjust_stop",
        )

    def _apply_adjust_take_profit(
        self,
        *,
        execution_id: int,
        symbol: str,
        new_tp_price: float,
        now: dt.datetime,
        schedule_id: int,
        ctx: ThesisReviewContext,
    ) -> None:
        """D-079 §3.4 — raise-only TP actuator (let-winners-run),
        mirroring §3.3 via the atomic `replace_limit_order` PATCH.

        Validator-on-apply: `new_tp_price` must clear the current quote
        (raise-only — lowering a TP is expressible as `close`), and the
        resulting R:R against the *current* stop must not fall below
        the §3.2 floor (belt-and-suspenders: raising a TP can only
        improve R:R).
        """
        orders = get_orders_for_execution(self._journal.db_path, execution_id)
        entry = next((o for o in orders if o.role == "entry"), None)
        tp_row = self._get_live_protective_order(
            execution_id=execution_id, roles=("take_profit",)
        )
        if entry is None or tp_row is None or entry.qty <= 0:
            # No live TP leg to raise (no-TP proposal, or pre-conversion
            # PDT-era position). Nothing actionable — treat as hold.
            log.warning(
                "thesis_coordinator.adjust_tp_skipped",
                extra={
                    "event": "thesis_coordinator.adjust_tp_skipped",
                    "execution_id": execution_id,
                    "reason": "missing_entry_or_live_tp",
                },
            )
            self._reschedule(
                execution_id,
                when=now + dt.timedelta(days=HOLD_RESCHEDULE_DAYS),
                reason="adjust_take_profit",
            )
            return

        if new_tp_price <= ctx.current_price:
            log.warning(
                "thesis_coordinator.adjust_tp_rejected",
                extra={
                    "event": "thesis_coordinator.adjust_tp_rejected",
                    "execution_id": execution_id,
                    "new_tp_price": new_tp_price,
                    "current_price": ctx.current_price,
                    "reason": "raise_only_tp_not_above_current_price",
                },
            )
            self._reschedule(
                execution_id,
                when=now + dt.timedelta(days=1),
                reason="adjust_take_profit",
            )
            return

        stop_row = self._get_live_protective_order(
            execution_id=execution_id, roles=("stop", "trailing_stop")
        )
        entry_price = ctx.realized_fill_price or entry.pre_submission_last
        if (
            stop_row is not None
            and stop_row.stop_price is not None
            and entry_price is not None
        ):
            risk = entry_price - stop_row.stop_price
            # A ratcheted stop at/above entry means the position is
            # risk-free — any raise passes. Otherwise hold the floor.
            if risk > 0 and (new_tp_price - entry_price) / risk < self._min_reward_risk:
                log.warning(
                    "thesis_coordinator.adjust_tp_rejected",
                    extra={
                        "event": "thesis_coordinator.adjust_tp_rejected",
                        "execution_id": execution_id,
                        "new_tp_price": new_tp_price,
                        "entry_price": entry_price,
                        "current_stop_price": stop_row.stop_price,
                        "min_reward_risk": self._min_reward_risk,
                        "reason": "exit_geometry_below_floor",
                    },
                )
                self._reschedule(
                    execution_id,
                    when=now + dt.timedelta(days=1),
                    reason="adjust_take_profit",
                )
                return

        try:
            replaced = self._broker.replace_limit_order(
                tp_row.broker_order_id,
                new_limit_price=new_tp_price,
                client_order_id=f"thesis-adjtp-{schedule_id}",
            )
        except Exception as e:  # noqa: BLE001
            # Atomic PATCH contract: original TP still live at the
            # broker. Retry tomorrow.
            log.error(
                "thesis_coordinator.tp_replace_failed",
                extra={
                    "event": "thesis_coordinator.tp_replace_failed",
                    "execution_id": execution_id,
                    "old_broker_order_id": tp_row.broker_order_id,
                    "error": str(e),
                },
            )
            self._reschedule(
                execution_id,
                when=now + dt.timedelta(days=1),
                reason="adjust_take_profit",
            )
            return

        insert_order(
            self._journal.db_path,
            OrderRow(
                execution_id=execution_id,
                role="take_profit",
                symbol=symbol,
                side="sell",
                order_type="limit",
                qty=entry.qty,
                tif="gtc",
                limit_price=new_tp_price,
                broker_order_id=replaced.broker_order_id,
                submitted_at=replaced.submitted_at,
                final_status=None,
                notes=(
                    f"thesis_review:adjust_take_profit atomic-replaced "
                    f"{tp_row.broker_order_id}"
                ),
            ),
        )
        self._record_order_outcome(
            tp_row.id, "replaced", fill_price=None, fill_qty=None, fill_at=None
        )

        log.info(
            "thesis_coordinator.tp_adjusted",
            extra={
                "event": "thesis_coordinator.tp_adjusted",
                "execution_id": execution_id,
                "symbol": symbol,
                "old_tp_price": tp_row.limit_price,
                "new_tp_price": new_tp_price,
                "old_broker_order_id": tp_row.broker_order_id,
                "new_broker_order_id": replaced.broker_order_id,
            },
        )

        self._reschedule(
            execution_id,
            when=now + dt.timedelta(days=HOLD_RESCHEDULE_DAYS),
            reason="adjust_take_profit",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reschedule(
        self,
        execution_id: int,
        *,
        when: dt.datetime,
        reason: str,
    ) -> None:
        insert_thesis_review_schedule(
            self._journal.db_path,
            ThesisReviewScheduleRow(
                execution_id=execution_id,
                due_at=when,
                scheduled_reason=reason,
            ),
        )

    def _build_review_context(
        self,
        *,
        schedule: ThesisReviewScheduleRow,
        execution: Any,
        proposal: Any,
        now: dt.datetime,
    ) -> ThesisReviewContext:
        # Days held: prefer the entry order's submitted_at, fall back to
        # `executions.decided_at`. Both come from the journal.
        orders = get_orders_for_execution(self._journal.db_path, execution.id)
        entry = next((o for o in orders if o.role == "entry"), None)
        if entry is None:
            raise _BuildContextError("no entry order on execution")
        entry_time = _ensure_aware(entry.submitted_at)
        days_held = max(0, (now - entry_time).days)

        # Current price + entry fill price for pct change.
        try:
            quote = self._broker.get_quote(proposal.symbol)
        except Exception as e:  # noqa: BLE001
            raise _BuildContextError(f"broker quote failed: {e}") from e
        current_price = float(quote.last)
        fill_price = entry.realized_fill_price
        if fill_price is None or fill_price <= 0:
            # If the entry hasn't reported a fill price yet, use the
            # broker's avg_entry_price for the position. As a last
            # resort, use the entry's pre_submission_last (the snapshot
            # taken at submission).
            fill_price = entry.pre_submission_last
        pct_change = (
            (current_price - fill_price) / fill_price if fill_price else 0.0
        )

        # Filings since entry — caller-supplied lookup.
        filings_summary = self._filings_lookup(
            issuer_ticker=proposal.symbol,
            since=entry_time,
        )

        # Pull the trade-plan numbers from the proposal payload — the
        # ProposalRow columns don't carry stop/TP, those live in the
        # payload (validate_trade_proposal yields them).
        from proposal.schemas import validate_trade_proposal
        try:
            payload = json.loads(proposal.raw_response)
            trade = validate_trade_proposal(payload)
        except Exception as e:  # noqa: BLE001
            raise _BuildContextError(f"proposal payload invalid: {e}") from e

        # Zone-aware state (event-reviews doctrine): armed status + HWM
        # from the trailing ratchet's operational memory. A position with
        # no exit_policy_state row simply hasn't engaged the trail — the
        # current price is the best available high-water proxy.
        state = get_exit_policy_state(
            self._journal.db_path, execution_id=execution.id
        )
        trail_armed = bool(state.trail_engaged) if state is not None else False
        hwm = (
            max(state.high_water_mark, current_price)
            if state is not None
            else current_price
        )
        hwm_gain_pct = (
            (hwm - fill_price) / fill_price * 100.0 if fill_price else 0.0
        )
        zone = compute_position_zone(
            trail_armed=trail_armed,
            hwm_gain_pct=hwm_gain_pct,
            zone3_gain_pct=self._zone3_gain_pct,
        )

        return ThesisReviewContext(
            proposal=proposal,
            execution_id=execution.id,
            schedule_id=schedule.id,  # type: ignore[arg-type]
            days_held=days_held,
            current_price=current_price,
            pct_change_since_entry=pct_change,
            realized_fill_price=fill_price,
            realized_dollar_size=execution.realized_dollar_size,
            time_horizon_days=trade.time_horizon_days,
            stop_loss_price=trade.stop_loss_price,
            take_profit_price=trade.take_profit_price,
            filings_since_entry_summary=filings_summary,
            position_zone=zone,
            trail_armed=trail_armed,
            hwm_gain_pct=hwm_gain_pct,
            trigger_reason=schedule.scheduled_reason,
            trigger_detail=self._trigger_detail(schedule.scheduled_reason),
        )

    def _trigger_detail(self, reason: str) -> str | None:
        """Human-readable evidence line for event-triggered reviews —
        the filing's form/accession or the news headline the reviewer
        must weigh (and cite when acting at altitude). Calendar reasons
        ('entry', 'hold', ...) carry no detail."""
        if reason.startswith("event:filing:"):
            accession = reason.split(":", 2)[2]
            with connect(self._journal.db_path) as conn:
                row = conn.execute(
                    "SELECT form_type, filed_at, item_codes FROM filings "
                    "WHERE accession_number = ? LIMIT 1",
                    (accession,),
                ).fetchone()
            if row is None:
                return f"filing {accession} (not found in journal)"
            return (
                f"filing form={row['form_type']} acc={accession} "
                f"items={row['item_codes'] or '[]'} filed_at={row['filed_at']}"
            )
        if reason.startswith("event:news:"):
            # event:news:<article_id>:<headline (truncated)>
            parts = reason.split(":", 3)
            headline = parts[3] if len(parts) > 3 else ""
            return f"news headline: {headline}" if headline else "news headline"
        if reason.startswith("event:gain_cross:"):
            threshold = reason.split(":", 2)[2]
            return f"high-water gain crossed +{threshold}%"
        if reason == "event:anomaly":
            return "intraday price anomaly vs previous close"
        return None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class _BuildContextError(Exception):
    """Raised when the review context cannot be assembled (missing entry
    order, broker quote failure, malformed proposal payload)."""


def _decision_string(result: Any) -> str:
    if isinstance(result, ThesisHold):
        return "hold"
    if isinstance(result, ThesisClose):
        return "close"
    if isinstance(result, ThesisAdjustStop):
        return "adjust_stop"
    if isinstance(result, ThesisAdjustTakeProfit):
        return "adjust_take_profit"
    return "malformed"


def _get_execution(db_path: str, execution_id: int) -> Any:
    """Pull a single executions row. Local helper because the existing
    `get_execution_by_proposal_id` is keyed on proposal_id."""
    from journal.repo import connect

    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM executions WHERE id = ?", (execution_id,)
        ).fetchone()
    if row is None:
        return None
    from journal.models import ExecutionRow

    return ExecutionRow(**dict(row))


def _ensure_aware(d: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes for TIMESTAMP columns by default.
    Treat them as UTC so date math against `now()` is consistent."""
    if d.tzinfo is None:
        return d.replace(tzinfo=dt.UTC)
    return d


# ---------------------------------------------------------------------------
# Default filings lookup (used in production composition)
# ---------------------------------------------------------------------------


def make_journal_filings_lookup(db_path: str) -> _FilingsLookup:
    """Returns a `_FilingsLookup` that scans the `filings` table for
    rows whose `issuer_ticker` matches and `filed_at >= since`. Output
    is a compact text summary that fits in the prompt's block-3."""

    from journal.repo import connect as _connect

    def _lookup(*, issuer_ticker: str | None, since: dt.datetime) -> str:
        if issuer_ticker is None:
            return "(no issuer ticker on original proposal)"
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT accession_number, form_type, item_codes, filed_at
                FROM filings
                WHERE issuer_ticker = ? AND filed_at >= ?
                ORDER BY filed_at ASC
                LIMIT 10
                """,
                (issuer_ticker, since),
            ).fetchall()
        if not rows:
            return "(no filings since entry)"
        lines = [
            f"- {r['filed_at']}  form={r['form_type']}  items={r['item_codes'] or '[]'}  acc={r['accession_number']}"
            for r in rows
        ]
        return "\n".join(lines)

    return _lookup


__all__ = [
    "HOLD_RESCHEDULE_DAYS",
    "ThesisReviewCoordinator",
    "compute_position_zone",
    "make_journal_filings_lookup",
]
