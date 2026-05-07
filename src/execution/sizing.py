"""S6.3 — Code-enforced position sizing + KS-4..KS-7 pre-trade caps.

PRD §6.1 specifies fixed-fraction-of-equity sizing by conviction tier,
hard-capped per KS-4. PRD §5.2 specifies KS-5 (concurrent positions),
KS-6 (single-name; covered by KS-4 + uniqueness here), and KS-7 (cash
reserve floor). The LLM's `size_pct_of_capital` is treated as a
*request* and may be capped silently — both the request and the
realized values are surfaced for journaling (FR-22).

This module is pure logic. All collaborators (account, positions,
quote, config) are values passed in; there is no I/O. S6.4 owns the
broker submission and the journal write.

Per D-007 (sacred), there is no `if (paper|mode|live):` branching here;
KS-4's $1,000 absolute floor is parameterized via `cfg.ks4_absolute_cap_usd`
per D-022 so the dial is a config value, not code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from broker.protocol import AccountSnapshot, OpenOrder, Position, Quote
from config.loader import ExecutionConfig
from observability.log_port import get_logger
from proposal.schemas import TradeProposal

log = get_logger(__name__)

# Conviction-tier sizing rates (PRD §6.1). cv >= 9 → high-tier rate; cv in
# [7, 8] → mid-tier rate; cv in [_CONVICTION_FLOOR, 6] also uses mid-tier
# rate (small fraction of equity, consistent with the prompt's "~5% for
# conviction 5–6" calibration).
_TIER_HIGH_THRESHOLD = 9  # cv >= 9 → high tier (10% default)
_TIER_MID_THRESHOLD = 7  # cv in [7, 8] → mid tier (5% default)

# Floor below which we will not size a trade. Tracks the configured Opus-
# review threshold (lowered from 7 → 5 on 2026-05-06): proposals that
# somehow reach sizing below this floor are journaled-and-skipped via
# `SizingRejected("conviction_too_low")` rather than crashing the agent
# loop, so a stray low-cv proposal is recoverable in production.
_CONVICTION_FLOOR = 5

# Sizing rejection vocabulary, written to `executions.reject_reason` by
# S5.6's agent loop (carry-fwd S6.4 reviewer M-1; S6.4's OrderSubmitter
# only persists the ACCEPTED branch).
SizingRejectReason = Literal[
    "ks5_concurrent_limit",
    "ks6_already_held",
    "ks7_cash_reserve",
    "size_too_small_for_one_share",
    "conviction_too_low",
]


@dataclass(frozen=True)
class SizingAccepted:
    """Sizing result for an accepted proposal.

    AC-8: both the request (LLM-asked) and realized (post-cap) dollar
    sizes are recorded so the journal can show the sizing audit trail.
    `realized_pct` is the actual fraction of equity placed (i.e. the
    realized dollar size divided by equity), not the tier rate — this
    matters when KS-4 or KS-7 caps reduce the size below the tier ask.
    """

    realized_dollar_size: float
    realized_pct: float
    qty: int
    realized_dollar_size_request: float


@dataclass(frozen=True)
class SizingRejected:
    reason: SizingRejectReason


SizingResult = SizingAccepted | SizingRejected


class SizingEngine:
    """Pure, deterministic sizing per PRD §6.1 + KS-4..KS-7."""

    def size(
        self,
        proposal: TradeProposal,
        account: AccountSnapshot,
        open_positions: list[Position],
        quote: Quote,
        cfg: ExecutionConfig,
        pending_buys: list[OpenOrder] | None = None,
    ) -> SizingResult:
        # Defense-in-depth: a proposal below the conviction floor should
        # have been routed to NoTrade by the analyzer or rejected by Opus.
        # If one slips through, journal-and-skip rather than crash the
        # agent loop — recoverable in production over an unhandled
        # AssertionError that would tear down the ingest path.
        if proposal.conviction < _CONVICTION_FLOOR:
            return self._reject("conviction_too_low", proposal)

        # KS-5 / KS-6 effective count = unique symbols across (filled
        # positions ∪ pending entry buys). Hotfix 2026-05-07: pre-market
        # market BUYs queue at Alpaca and only fill at the bell, so the
        # filled-only view saw 0 positions while the bot was happily
        # opening its 28th of the morning. Pending SELLs (stops / TPs
        # from prior days) are NOT counted — the underlying long is
        # already in `open_positions`.
        pending_buys = pending_buys or []
        pending_symbols = {o.symbol for o in pending_buys if o.is_entry}
        held_symbols = {p.symbol for p in open_positions}
        effective_symbols = held_symbols | pending_symbols

        # KS-5: concurrent-position cap (PRD §5.2).
        if len(effective_symbols) >= cfg.ks5_max_concurrent:
            return self._reject("ks5_concurrent_limit", proposal)

        # KS-6: single-name re-entry block (PRD §5.2 + tactical clarification
        # in S6.3 dev notes — KS-6 is "covered by KS-4" in v1, with the
        # additional rule that we don't re-enter a held name OR submit a
        # second entry while a prior entry on the same symbol is still
        # pending fill (avoids the wash-trade rejection cascade observed
        # in the 2026-05-07 incident).
        if proposal.symbol in effective_symbols:
            return self._reject("ks6_already_held", proposal)

        # PRD §6.1: conviction-tier rate.
        tier_pct = (
            cfg.sizing_high_pct
            if proposal.conviction >= _TIER_HIGH_THRESHOLD
            else cfg.sizing_mid_pct
        )
        tier_dollar = tier_pct * account.equity
        request_dollar = proposal.size_pct_of_capital * account.equity

        # KS-4 cap (PRD §5.2 + D-022): min(pct_cap * equity, absolute_cap).
        # Applied silently per AC-3.
        ks4_cap = min(cfg.ks4_pct_cap * account.equity, cfg.ks4_absolute_cap_usd)
        realized_dollar = min(tier_dollar, ks4_cap)

        # KS-7: cash-reserve floor. After the proposed spend, cash must
        # remain ≥ ks7_cash_reserve_pct * equity. If the tier+KS-4 size
        # violates this, *reduce* the size to maintain the reserve. Only
        # reject if even the minimum viable size (one share) cannot be
        # placed without breaching the floor.
        reserve_floor = cfg.ks7_cash_reserve_pct * account.equity
        max_spend_under_reserve = account.cash - reserve_floor
        if max_spend_under_reserve < quote.last:
            # One share would still breach the floor (or cash is already
            # below floor). Cannot proceed.
            return self._reject("ks7_cash_reserve", proposal)
        if realized_dollar > max_spend_under_reserve:
            realized_dollar = max_spend_under_reserve

        # AC-7: integer share quantity.
        qty = math.floor(realized_dollar / quote.last)
        if qty < 1:
            return self._reject("size_too_small_for_one_share", proposal)

        # Re-bind realized_dollar to qty * last so the journal value matches
        # the actual order. (Floor-rounding can leave $0–$last unspent.)
        realized_dollar_actual = qty * quote.last

        log.info(
            "execution.sizing.accept",
            extra={
                "event": "execution.sizing.accept",
                "symbol": proposal.symbol,
                "conviction": proposal.conviction,
                "realized_dollar_size": realized_dollar_actual,
                "realized_dollar_size_request": request_dollar,
                "qty": qty,
            },
        )
        return SizingAccepted(
            realized_dollar_size=realized_dollar_actual,
            realized_pct=realized_dollar_actual / account.equity,
            qty=qty,
            realized_dollar_size_request=request_dollar,
        )

    @staticmethod
    def _reject(reason: SizingRejectReason, proposal: TradeProposal) -> SizingRejected:
        log.info(
            "execution.sizing.reject",
            extra={
                "event": "execution.sizing.reject",
                "symbol": proposal.symbol,
                "reason": reason,
            },
        )
        return SizingRejected(reason=reason)


__all__ = [
    "SizingAccepted",
    "SizingEngine",
    "SizingRejectReason",
    "SizingRejected",
    "SizingResult",
]
