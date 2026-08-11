"""Feature C — opportunistic retro-fill on slot-open.

When a position closes (KS5 slot frees), check for high-conviction
proposals (cv >= 9) that were skipped at capacity by Feature B. If any
exist within a 6-hour freshness window AND the issuer isn't already
held, run Opus retroactively, then drive the standard validator → sizer
→ submitter chain via the agent loop's `execute_proposal`.

Design notes
------------
- Trigger: invoked by the reconciler after every successful tick (which
  runs every `cfg.reconciler.interval_seconds_market` seconds during
  market hours). Off-hours suppression is inherited from the reconciler
  by virtue of the trigger location.

- Real-time priority: this coordinator never blocks the consumer task.
  The reconciler runs as a sibling task to the consumer, so a retro
  candidate cannot starve a fresh-signal proposal. If retro and fresh
  race for the slot, the validator/sizer's KS5 check picks the winner;
  the loser writes a benign `agent.retro_lost_to_fresh` event.

- Pyramiding guard: before passing the candidate to the executor, we
  check `has_open_position(symbol)`. The reconciler runs immediately
  before this hook with broker truth, so the journal's view of open
  positions is fresh.

- Conviction floor (>= 9) and freshness window (6h) are story-fixed
  constants per the task description, not config-tunable.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from journal.models import FilingRow, ProposalRow
from journal.repo import (
    find_retro_candidate,
    get_filing_by_id,
    get_proposal_review_by_proposal_id,
    has_open_position,
)
from observability.log_port import get_logger

log = get_logger(__name__)

# Feature C — story-fixed constants (out of scope for config in v1).
RETRO_CONVICTION_FLOOR = 9
RETRO_FRESHNESS_WINDOW = dt.timedelta(hours=6)

# SCOPE NOTE (full-book Opus-review gate): retro-fill rescues
# `pending_capacity` rows ONLY. `review_skipped_full_book` rows — the
# gate's deliberate skips — are retro-ineligible BY DESIGN
# (`find_retro_candidate` excludes any non-pending_capacity execution
# row): their designated rescue path is the WATCHLIST, whose deferred
# retry pays for the skipped review before entering, and the `AppConfig`
# cross-field validator guarantees the watchlist is on whenever the gate
# is. Conversely, sub-cv9 `pending_capacity` rows sit under this floor
# and are rescued by the loop's belt-and-braces watchlist enrollment,
# not by retro-fill.


class _OpusReviewerPort(Protocol):
    async def review(
        self, proposal: ProposalRow, source_filing: FilingRow, raw_text: str
    ) -> Any: ...


class _JournalLike(Protocol):
    db_path: str

    def get_open_positions(self) -> list: ...


ExecuteProposalFn = Callable[[int, FilingRow], Awaitable[None]]


class RetroFillCoordinator:
    """Opportunistic retro-fill on slot-open. Stateless across ticks; one
    instance per agent process.
    """

    def __init__(
        self,
        *,
        journal: _JournalLike,
        opus_reviewer: _OpusReviewerPort,
        ks5_max_concurrent: int,
        now_fn: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
        max_input_chars: int = 0,
    ) -> None:
        self._journal = journal
        self._opus = opus_reviewer
        self._ks5_max = ks5_max_concurrent
        self._now_fn = now_fn
        self._execute_fn: ExecuteProposalFn | None = None
        # COST LEVER (advisory #5) — same filing-body cap the day-0
        # analyzer applies (`analyzer.analyzer_max_input_chars`, wired by
        # `compose_agent`); the retro review must see the same capped
        # text a day-0 review would have. 0 = off.
        self._max_input_chars = max_input_chars

    def set_executor(self, fn: ExecuteProposalFn) -> None:
        """Wire the agent loop's `execute_proposal` after construction.
        (The loop is built after the coordinator in `compose_agent`, so
        the wiring is two-phase.)"""
        self._execute_fn = fn

    async def run_tick(self) -> None:
        """One retro-fill tick. Returns silently when:
          - no executor wired yet (early boot);
          - no slack (open >= ks5_max);
          - no candidate matches the (cv>=9, ≤6h, no-review, no-execution)
            filter;
          - candidate's symbol is already held (pyramiding guard).
        """
        if self._execute_fn is None:
            return

        open_positions = self._journal.get_open_positions()
        slack = self._ks5_max - len(open_positions)
        if slack <= 0:
            return

        now = self._now_fn()
        cutoff = now - RETRO_FRESHNESS_WINDOW
        candidate = find_retro_candidate(
            self._journal.db_path,
            min_conviction=RETRO_CONVICTION_FLOOR,
            not_older_than=cutoff,
        )
        if candidate is None:
            return

        # Pyramiding guard — never double up on a name we already hold.
        # CRITICAL per task description; reviewer will scrutinize.
        if candidate.symbol is None:
            return  # defensive: trade_proposal rows always have a symbol
        if has_open_position(self._journal.db_path, candidate.symbol):
            log.info(
                "agent.retro_pyramiding_skipped",
                extra={
                    "event": "agent.retro_pyramiding_skipped",
                    "proposal_id": candidate.id,
                    "symbol": candidate.symbol,
                    "conviction": candidate.conviction,
                },
            )
            return

        # Defense-in-depth: if a `proposal_reviews` row appeared between
        # the SQL filter and now, abort to avoid double-reviewing.
        existing_review = get_proposal_review_by_proposal_id(
            self._journal.db_path, candidate.id  # type: ignore[arg-type]
        )
        if existing_review is not None:
            return

        filing = get_filing_by_id(self._journal.db_path, candidate.filing_id)
        if filing is None:
            log.warning(
                "agent.retro_filing_missing",
                extra={
                    "event": "agent.retro_filing_missing",
                    "proposal_id": candidate.id,
                    "filing_id": candidate.filing_id,
                },
            )
            return
        # Advisory #5 — cap the filing body exactly as the day-0 analyzer
        # would have (same helper, same marker), so the retro review is
        # consistent with day-0 reviews and the cap's savings apply here.
        from analyzer.sonnet import cap_filing_text

        raw_text = cap_filing_text(
            _read_raw_text(filing.raw_text_path),
            cap=self._max_input_chars,
            filing_id=filing.id,
        )

        log.info(
            "agent.retro_candidate_selected",
            extra={
                "event": "agent.retro_candidate_selected",
                "proposal_id": candidate.id,
                "symbol": candidate.symbol,
                "conviction": candidate.conviction,
                "filing_id": candidate.filing_id,
                "open_positions": len(open_positions),
                "ks5_max": self._ks5_max,
            },
        )

        # Run Opus. The reviewer writes the proposal_reviews row before
        # returning — that's how `execute_proposal` later sees the decision.
        try:
            review_result = await self._opus.review(candidate, filing, raw_text)
        except Exception as e:  # noqa: BLE001 — never kill the reconciler tick
            log.error(
                "agent.retro_opus_error",
                extra={
                    "event": "agent.retro_opus_error",
                    "proposal_id": candidate.id,
                    "error": str(e),
                    "error_class": type(e).__name__,
                },
            )
            return

        decision = _extract_decision(review_result)
        log.info(
            "agent.retro_opus_decision",
            extra={
                "event": "agent.retro_opus_decision",
                "proposal_id": candidate.id,
                "decision": decision,
            },
        )
        if decision == "reject" or decision == "malformed":
            # Per task spec: don't try other retro candidates on rejection.
            # The execute path will write a rejected execution row via
            # `agent.execute_proposal` only if Opus's review row says reject.
            # We still pass through so the rejection is journaled.
            await self._execute_fn(candidate.id, filing)  # type: ignore[arg-type]
            return

        # Accept (ratify) or modify → drive the validator/sizer/submitter
        # chain. The validator's KS5 check is the safety net if a fresh
        # signal raced and won the slot between our slack check and here.
        try:
            await self._execute_fn(candidate.id, filing)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            log.error(
                "agent.retro_execute_error",
                extra={
                    "event": "agent.retro_execute_error",
                    "proposal_id": candidate.id,
                    "error": str(e),
                    "error_class": type(e).__name__,
                },
            )
            return

        # Inspect the executions row to log the outcome (for ops audit).
        # This is best-effort — the row is the source of truth either way.
        from journal.repo import get_execution_by_proposal_id

        execution = get_execution_by_proposal_id(
            self._journal.db_path, candidate.id  # type: ignore[arg-type]
        )
        outcome = (
            execution.decision if execution is not None else "no_execution_row"
        )
        log.info(
            "agent.retro_validator_outcome",
            extra={
                "event": "agent.retro_validator_outcome",
                "proposal_id": candidate.id,
                "outcome": outcome,
                "reject_reason": (
                    execution.reject_reason if execution is not None else None
                ),
            },
        )
        if execution is not None and execution.decision == "rejected" and (
            execution.reject_reason in ("ks5_concurrent_limit", "ks6_already_held")
        ):
            # Fresh signal won the race for the slot or our pyramiding
            # check missed a position opened concurrently.
            log.info(
                "agent.retro_lost_to_fresh",
                extra={
                    "event": "agent.retro_lost_to_fresh",
                    "proposal_id": candidate.id,
                    "reject_reason": execution.reject_reason,
                },
            )


def _extract_decision(review_result: Any) -> str:
    """Map an `OpusReviewResult` variant to its decision string."""
    if review_result is None:
        return "unknown"
    cls = type(review_result).__name__
    if cls == "OpusRatified":
        return "ratify"
    if cls == "OpusModified":
        return "modify"
    if cls == "OpusRejected":
        return "reject"
    if cls == "OpusMalformed":
        return "malformed"
    return "unknown"


def _read_raw_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        log.warning(
            "agent.retro_raw_text_missing",
            extra={"event": "agent.retro_raw_text_missing", "path": path},
        )
        return ""


__all__ = [
    "RETRO_CONVICTION_FLOOR",
    "RETRO_FRESHNESS_WINDOW",
    "RetroFillCoordinator",
]
