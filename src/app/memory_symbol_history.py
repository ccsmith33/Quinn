"""SYMBOL-HISTORY memory provider.

When Quinn analyzes a filing ('analyze') or reviews an open position
('thesis_review') for a symbol it has met before, this provider injects
the most recent prior engagements — closed trades (entry/exit/realized
%/days held, plus the post-mortem lesson when one exists) and
analyzed-but-not-traded proposals (conviction + why the trade was
turned away).

Contract (see `app.memory_context`): returns None when the query has
no symbol, the purpose is not served, or the symbol has no prior
engagements. Rendering is DETERMINISTIC — dates as YYYY-MM-DD, prices
2dp, percents signed and rounded to 1dp, newest first, capped at
`_MAX_ENGAGEMENTS` — so identical journal state yields byte-identical
output. Registered as "symbol_history" in `app.composition` behind
`config.memory.symbol_history_enabled`.
"""

from __future__ import annotations

from app.memory_context import MemoryQuery, MemorySection, Provider
from journal.repo import JournalRepo, SymbolEngagement

_PROVIDER_NAME = "symbol_history"
_SERVED_PURPOSES = ("analyze", "thesis_review")
_MAX_ENGAGEMENTS = 3

# `executions.reject_reason` → operator-readable label. The validator's
# canonical vocabulary lives in `execution.validator.RejectReason`;
# anything unrecognized (future reasons) falls back to a generic
# "rejected (<reason>)" so rendering never raises.
_REJECT_LABELS = {
    "opus_reject": "rejected by review",
    "insufficient_capital": "rejected on capital",
    "pending_capacity": "skipped at capacity",
    "schema": "rejected by validator",
    "kill_switch": "rejected by validator",
    "universe": "rejected by validator",
    "price_floor": "rejected by validator",
    "exit_geometry": "rejected by validator",
    "direction_unsupported": "rejected by validator",
}


def make_symbol_history_provider(journal: JournalRepo) -> Provider:
    """Build the symbol_history provider closed over `journal`."""

    def provide(query: MemoryQuery) -> MemorySection | None:
        if query.symbol is None or query.purpose not in _SERVED_PURPOSES:
            return None
        engagements = journal.get_prior_engagements_for_symbol(
            query.symbol,
            limit=_MAX_ENGAGEMENTS,
            exclude_execution_id=query.execution_id,
        )
        if not engagements:
            return None
        lessons = _lessons_by_execution(journal, query.symbol)
        fragments = " ".join(_render(g, lessons) for g in engagements)
        return MemorySection(
            title="Symbol history",
            body=f"Prior engagements with {query.symbol}: {fragments}",
            provider_name=_PROVIDER_NAME,
        )

    return provide


def _lessons_by_execution(journal: JournalRepo, symbol: str) -> dict[int, str]:
    """Newest post-mortem `lesson` per execution_id (rows are served
    newest-first, so the first lesson seen per execution wins)."""
    lessons: dict[int, str] = {}
    for pm in journal.get_postmortems_for_symbol(symbol):
        if pm.lesson and pm.execution_id not in lessons:
            lessons[pm.execution_id] = pm.lesson
    return lessons


def _render(g: SymbolEngagement, lessons: dict[int, str]) -> str:
    if g.kind == "closed_trade":
        frag = (
            f"[{g.entry_date} bought {g.entry_price:.2f}, "
            f"closed {g.date} at {g.exit_price:.2f} "
            f"({g.realized_pct:+.1f}%), held {g.days_held}d"
        )
        lesson = lessons.get(g.execution_id) if g.execution_id is not None else None
        if lesson:
            frag += f" — postmortem: {lesson}"
        return frag + "]"
    label = (
        _REJECT_LABELS.get(g.reject_reason, f"rejected ({g.reject_reason})")
        if g.reject_reason is not None
        else "rejected"
    )
    conviction = f" cv{g.conviction}" if g.conviction is not None else ""
    return f"[{g.date} analyzed{conviction}, {label}]"


__all__ = ["make_symbol_history_provider"]
