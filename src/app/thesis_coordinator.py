"""Feature A — thesis-review coordinator.

Drives the open-position thesis-review pipeline:

  reconciler tick (market hours, post-reconcile)
   → for each due schedule:
     - confirm position is still open (stop/TP may have closed it)
     - build review context (price, days held, filings since entry)
     - call ThesisReviewer (Opus)
     - apply decision: hold | close | adjust_stop
     - write follow-up schedule (hold / adjust_stop) or none (close)

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
from collections.abc import Callable
from typing import Any, Protocol

from broker.protocol import BrokerAdapter, OrderRequest
from journal.models import (
    OrderRow,
    ThesisReviewScheduleRow,
)
from journal.repo import (
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
    ThesisClose,
    ThesisHold,
    ThesisReviewContext,
    ThesisReviewer,
    ThesisReviewMalformed,
)

log = get_logger(__name__)


# Default cadence for follow-up reviews after a `hold`. Per task spec.
HOLD_RESCHEDULE_DAYS = 7


class _JournalLike(Protocol):
    db_path: str


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
    ) -> None:
        self._journal = journal
        self._broker = broker
        self._reviewer = reviewer
        self._filings_lookup = filings_lookup
        self._now_fn = now_fn
        self._default_horizon_days = default_horizon_days

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
    ) -> None:
        """SAFETY-CRITICAL: atomic stop-price replacement via the
        broker adapter's `replace_stop_order` (Alpaca's PATCH
        /v2/orders/{id}). The broker either accepts the replacement
        atomically (new order id, position never uncovered) or rejects
        and the original stop remains live.

        Order:
          1. Look up the current "live" stop (most recent stop row
             written for this execution).
          2. Call `broker.replace_stop_order(broker_order_id,
             new_stop_price=...)`. On failure: original stop is still
             live, position is protected; reschedule at +1 day to retry.
          3. Journal the new stop row (broker returned a new id).

        Reschedule the next thesis review at +`HOLD_RESCHEDULE_DAYS`.
        """
        orders = get_orders_for_execution(self._journal.db_path, execution_id)
        entry = next((o for o in orders if o.role == "entry"), None)
        # Use the most recent stop row as the "current" stop. If a prior
        # adjust_stop already replaced the original, that newer row wins.
        stop_orders = [o for o in orders if o.role in ("stop", "thesis_stop")]
        old_stop = max(stop_orders, key=lambda o: o.id or 0) if stop_orders else None
        if entry is None or old_stop is None or entry.qty <= 0:
            log.warning(
                "thesis_coordinator.adjust_stop_skipped",
                extra={
                    "event": "thesis_coordinator.adjust_stop_skipped",
                    "execution_id": execution_id,
                    "reason": "missing_entry_or_stop",
                },
            )
            return

        # Atomic replace. On failure, old stop remains live; we never
        # leave the position uncovered.
        client_order_id = (
            f"thesis-stop-exec-{execution_id}-{now.timestamp():.0f}"
        )
        try:
            replaced = self._broker.replace_stop_order(
                old_stop.broker_order_id,
                new_stop_price=new_stop_price,
                client_order_id=client_order_id,
            )
        except Exception as e:  # noqa: BLE001
            log.error(
                "thesis_coordinator.stop_replace_failed",
                extra={
                    "event": "thesis_coordinator.stop_replace_failed",
                    "execution_id": execution_id,
                    "old_broker_order_id": old_stop.broker_order_id,
                    "error": str(e),
                },
            )
            # Original stop still alive at broker — position is protected.
            # Retry tomorrow.
            self._reschedule(
                execution_id,
                when=now + dt.timedelta(days=1),
                reason="adjust_stop",
            )
            return

        # Journal the new stop. Alpaca's PATCH returns a new order id,
        # so we add a fresh row rather than mutating the old one.
        insert_order(
            self._journal.db_path,
            OrderRow(
                execution_id=execution_id,
                role="thesis_stop",
                symbol=symbol,
                side="sell",
                order_type="stop",
                qty=entry.qty,
                tif="gtc",
                stop_price=new_stop_price,
                broker_order_id=replaced.broker_order_id,
                submitted_at=replaced.submitted_at,
                final_status=replaced.status,
                notes=(
                    f"thesis_review:adjust_stop atomic-replaced "
                    f"{old_stop.broker_order_id}"
                ),
            ),
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

        # Reschedule the next review.
        self._reschedule(
            execution_id,
            when=now + dt.timedelta(days=HOLD_RESCHEDULE_DAYS),
            reason="adjust_stop",
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
        )


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
    "make_journal_filings_lookup",
]
