# ruff: noqa: E501
# (E501 off file-wide: DOCTRINE_V1 is stored VERBATIM — its lines exceed
# the limit and must not be re-wrapped or split.)
"""DOCTRINE memory provider — study-derived priors for every LLM call.

Serves the distilled findings of the eight offline catalyst studies
(33,851 events, 2026-07/08) so the analyzer / proposal-review /
thesis-review LLMs know WHY the exit / eviction / capacity policies are
shaped the way they are.

Two pieces:
  * `seed_doctrine_v1(journal)` — idempotent boot-time seeder. Inserts
    doctrine v1 into `desk_memory` (kind='doctrine', active=1) IFF no
    active doctrine row exists. An operator-updated doctrine (higher
    version, or any active row) is never overwritten.
  * `make_doctrine_provider(journal)` — the `Provider` registered as
    "doctrine" on the shared `MemoryContextAssembler`. Returns the
    active doctrine content for ALL purposes / symbols (the doctrine is
    desk-level, not symbol-scoped); None when no active row exists.
    Deterministic: the body is the DB content verbatim — no clock, no
    formatting of its own.

`register_doctrine(assembler, memory_config, journal)` is the single
line `compose_agent` calls: it applies the `doctrine_enabled` gate,
seeds, and registers — keeping the composition diff minimal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.memory_context import (
    MemoryContextAssembler,
    MemoryQuery,
    MemorySection,
    Provider,
)
from journal.models import DeskMemoryRow
from observability.log_port import get_logger

if TYPE_CHECKING:
    from config.loader import MemoryConfig
    from journal.repo import JournalRepo

log = get_logger(__name__)

_PROVIDER_NAME = "doctrine"
_SECTION_TITLE = "Desk doctrine (study-derived priors)"

# Distilled by the orchestrator from the eight offline studies, then
# trimmed to terse fragments for the LLM reader (review advisory #5:
# this text rides UNCACHED in block-3 on every call, so it is budgeted
# <= ~300 tokens by the chars/4 estimate — every study number and
# finding retained, connective prose cut). Stored verbatim in
# `desk_memory` by the seeder; served verbatim by the provider. Content
# changes ship as a NEW version row (operator / orchestrator action),
# never by editing this constant in place — v1 is frozen so
# already-seeded journals and this source never disagree.
#
# Future optimization (deliberately NOT done now): move static memory
# like this into a prompt-cached segment so its per-call cost drops to
# the cache-read rate instead of riding uncached.
DOCTRINE_V1 = """\
QUINN DESK DOCTRINE v1 (33,851 catalyst events 2026-07/08, backtest priors)
- EDGE: >=+50% MFE: 14% of events, 130% net P&L (rest negative); >=+100%: 5%, 64%. Never cap winners; TPs = far ~2x cap-lifters.
- SLOW BLOOMERS: median big winner +1.0% day-1 close, 45% flat/down; quiet != failure; 1-2d-late entry keeps ~98% P&L.
- WIGGLE: 8-12% dips off pre-peak highs normal early; sub-+12% trails strangle (3 tests); floor-to-entry @+12% peak: 27.6%->8.9% winner->loser, ~0 runner-cost, 0.0% fade-to-loss armed.
- VELOCITY=VARIANCE: day-0 +8% poppers fade more (med -5.7%) yet hold 28% of >=+50% monsters (MFE 58%/31%); no speed-harvest; floor absorbs fade.
- EVICTION: day-3/5 buckets all -EV vs redeploy even down>10% (~2.1%/trade/29d, ~0.056%/slot-day); 45% beaten-down young retag +10%; thesis-break evicts, price never; conv>=8 swaps ~+8.4%/trade +EV.
- PEAKS: ~47% day-0 highs 1st 30min; peaks day 3-7+; trails +20%->8%, +35%->5%: +163% vs +29% +5%-harvest.
- ODDS: P(+30|+10)=43% (hi-conv 59%); P(+50|+30)=54%; P(2x|+50)=38%; +10% = nearer start.
- CAVEATS: daily-bar conservative fills, survivorship-lite, single regime; priors not certainties; hard safety (stops/KS/exemptions) outranks."""


def seed_doctrine_v1(journal: JournalRepo) -> bool:
    """Insert doctrine v1 iff no active doctrine row exists.

    Idempotent per journal: the second call is a no-op, and an existing
    active doctrine row (any version, any content) is never overwritten
    or duplicated. Returns True when a row was inserted.
    """
    if journal.get_active_desk_memory("doctrine") is not None:
        return False
    row_id = journal.insert_desk_memory(
        DeskMemoryRow(kind="doctrine", content=DOCTRINE_V1, version=1, active=1)
    )
    log.info(
        "memory.doctrine_seeded",
        extra={
            "event": "memory.doctrine_seeded",
            "desk_memory_id": row_id,
            "version": 1,
        },
    )
    return True


def make_doctrine_provider(journal: JournalRepo) -> Provider:
    """Build the "doctrine" provider bound to `journal`.

    Reads the active doctrine row per call (so an operator publishing a
    new version takes effect without a restart) and serves it verbatim
    for every purpose / symbol. Returns None when no active row exists —
    the rail then simply omits the section.
    """

    def provide(_query: MemoryQuery) -> MemorySection | None:
        row = journal.get_active_desk_memory("doctrine")
        if row is None:
            return None
        return MemorySection(
            title=_SECTION_TITLE,
            body=row.content,
            provider_name=_PROVIDER_NAME,
        )

    return provide


def register_doctrine(
    assembler: MemoryContextAssembler,
    memory_config: MemoryConfig,
    journal: JournalRepo,
) -> None:
    """Gate + seed + register — the one-call composition seam.

    No-op when `memory_config.doctrine_enabled` is false (nothing seeded,
    nothing registered). The caller only invokes this when the memory
    master gate is already on.
    """
    if not memory_config.doctrine_enabled:
        return
    seed_doctrine_v1(journal)
    assembler.register(_PROVIDER_NAME, make_doctrine_provider(journal))


__all__ = [
    "DOCTRINE_V1",
    "make_doctrine_provider",
    "register_doctrine",
    "seed_doctrine_v1",
]
