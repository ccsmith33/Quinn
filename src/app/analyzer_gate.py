"""Deterministic pre-analysis buyability gate (API cost lever).

Business context: earnings season pushed the Haiku/Sonnet filing analyzer
to 294 calls / $13.26 on a single box in one day, and a large share of
that spend went to filings whose issuer can NEVER become a trade — the
filer is outside the current universe snapshot, or its last known price
sits under the trade-time price floor. Both are checks the execution
validator already performs, but it performs them *after* the LLM has
been paid.

This module answers the same two questions from data already in memory
(the universe snapshot the validator and prefilter share), so the loop
can decline to spend tokens on a filing that cannot produce a buyable
proposal. It is deliberately COARSE: the validator still runs the
precise, quote-backed check at submission time. The snapshot's
`prev_close` is a day-stale price, which is exactly right for a gate
whose only job is to drop names that are nowhere near the band.

Scope, honestly stated: the prefilter's stage-1 gate already rejects
filings whose CIK is outside the snapshot, and the daily universe refresh
screens members on `prev_close >= $5.00` — the same value
`execution.price_floor_usd` defaults to. So against a FRESH snapshot and
the default floor this gate has almost nothing left to catch. It earns
its keep in the two cases that produced the spend: a snapshot that has
gone stale (refresh job failed; members that have since fallen through
the floor keep getting analyzed), and an operator raising
`price_floor_usd` above the snapshot's own screen, which takes effect
here immediately instead of only at the validator after the tokens are
spent.

FAIL OPEN is the load-bearing property. This gate must never silence the
analyzer wholesale, so every ambiguity — an unresolved ticker, an empty
universe snapshot, an unexpected error inside the universe object —
resolves to "analyze anyway". Only an affirmative, well-formed answer
("this ticker resolved, and it is not in the snapshot" / "it is in the
snapshot at a price below the floor") skips the call.

Pure logic: no I/O, no config reads, no logging. The agent loop owns the
journal/event side and the feature switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

# The validator enforces a price FLOOR only (`price_floor_usd`) — there is
# no upper price bound anywhere in the execution path, so this gate has no
# ceiling either. If one is ever added to `ProposalValidator.validate`, it
# belongs here too, sourced from the same config value.
SkipReason = Literal["universe", "price"]


class _UniverseLike(Protocol):
    """Structural view of `universe.api.Universe` — the reads this gate
    needs, so unit tests can pass a trivial stub."""

    def get_member(self, ticker: str) -> object | None: ...

    def is_in_universe_by_cik(self, cik: int) -> bool: ...

    def member_count(self) -> int: ...


@dataclass(frozen=True)
class Analyze:
    """Spend the tokens: the filer is buyable, or we cannot prove it isn't."""


@dataclass(frozen=True)
class SkipUnbuyable:
    """The filer provably cannot become a trade. Skip the LLM call."""

    reason: SkipReason
    symbol: str
    price: float | None


@dataclass(frozen=True)
class FailOpen:
    """Ambiguous input — analyze anyway, but say so. `detail` is the
    operator-facing cause, counted and logged by the caller so a gate
    that starts failing open on every filing is visible rather than
    silently free."""

    detail: str


GateDecision = Analyze | SkipUnbuyable | FailOpen


def evaluate_buyability(
    *,
    ticker: str | None,
    cik: int | None,
    universe: _UniverseLike,
    price_floor_usd: float,
) -> GateDecision:
    """Decide whether a prefiltered filing is worth an LLM call.

    `ticker` is the filing's `issuer_ticker` (None when the CIK→ticker
    resolver could not place the filer — a fail-open case, not a
    rejection: an unresolved ticker says nothing about buyability).
    `cik` is the filing's CIK, used only to distinguish a genuine
    non-member from a ticker/snapshot disagreement.
    """
    if not ticker:
        return FailOpen(detail="unresolved_ticker")

    try:
        if universe.member_count() <= 0:
            return FailOpen(detail="empty_universe_snapshot")
        member = universe.get_member(ticker)
        cik_is_member = cik is not None and universe.is_in_universe_by_cik(cik)
    except Exception as e:  # noqa: BLE001 — any universe fault fails open
        return FailOpen(detail=f"universe_error:{type(e).__name__}")

    if member is None:
        # The prefilter's stage-1 gate (`is_in_universe_by_cik`) already
        # rejects every filing whose CIK is outside the snapshot, so a
        # filing that reaches here with an in-universe CIK but no ticker
        # match means the SEC ticker resolver and the snapshot disagree —
        # multi-class listings (GOOG vs GOOGL), a stale ticker cache. That
        # is a resolver defect, not an answer about buyability, so it
        # fails open. Consequently the universe branch below only fires
        # on paths that bypass the prefilter's gate.
        if cik_is_member:
            return FailOpen(detail="ticker_snapshot_mismatch")
        return SkipUnbuyable(reason="universe", symbol=ticker, price=None)

    prev_close = getattr(member, "prev_close", None)
    if prev_close is None:
        return FailOpen(detail="no_snapshot_price")
    price = float(prev_close)
    if price < price_floor_usd:
        return SkipUnbuyable(reason="price", symbol=ticker, price=price)

    return Analyze()


__all__ = [
    "Analyze",
    "FailOpen",
    "GateDecision",
    "SkipReason",
    "SkipUnbuyable",
    "evaluate_buyability",
]
