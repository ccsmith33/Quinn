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

# Distilled by the orchestrator from the eight offline studies. Stored
# verbatim in `desk_memory` by the seeder; served verbatim by the
# provider. Content changes ship as a NEW version row (operator /
# orchestrator action), never by editing this constant in place — v1 is
# frozen so already-seeded journals and this source never disagree.
DOCTRINE_V1 = """\
QUINN DESK DOCTRINE v1 (distilled from 33,851-event catalyst studies, 2026-07/08)
- EDGE CONCENTRATION: names reaching >=+50% MFE are 14% of events but 130% of net P&L (the rest nets negative); >=+100% names are 5% of events, 64% of P&L. The entire business is not capping winners. TPs only as far cap-lifters (~2x), never tight.
- SLOW BLOOMERS: the median eventual big winner is up only +1.0% at day-1 close; 45% are flat-or-down. A quiet first days is NOT thesis failure. Entering 1-2 days late retains ~98% of big-winner P&L.
- WIGGLE TOLERANCE: eventual runners routinely draw down 8-12% below their pre-peak highs at low altitude. Tightening trails below +12% peak gain was tested three ways and strangled more runners than it saved. The breakeven floor at +12% peak (stop never below entry) cuts winner-to-loser from 27.6% to 8.9% at ~zero runner cost — fade-to-loss is 0.0% in every cohort once the floor arms.
- VELOCITY IS VARIANCE, NOT EV: fast day-0 +8% poppers fade more often (median -5.7% from the tag) BUT hold 28% of all eventual >=+50% monsters (mean MFE 58% vs 31% for slow builders). Do not harvest speed; the floor absorbs the fade risk.
- EVICTION: every day-3 and day-5 price-state bucket (including down >10%) is NEGATIVE-EV to evict against average redeployment (~2.1%/trade over ~29 days, ~0.056%/slot-day net of friction) — 45% of beaten-down young names tag +10% above their low again. Eviction on price alone destroys value; eviction on THESIS BREAK (catalyst resolved against us, thesis invalidated) is the valid trigger. Demand-driven swaps into conviction>=8 signals (~+8.4%/trade historically) ARE strongly +EV.
- PEAKS AND EXITS: ~47% of day-0 session highs print in the first 30 minutes, but real winners typically peak day 3-7+; staged trail tightening (+20% gain -> 8% band, +35% -> 5%) preserved runners for +163% counterfactual vs +29% for harvest-at-+5%.
- CONDITIONAL ODDS: P(reach +30% | reached +10%) = 43% (59% for high conviction); P(+50 | +30) = 54%; P(double | +50) = 38%. A position at +10% is closer to its beginning than its end.
- CAVEATS: all figures are backtest (daily-bar conservative fills, survivorship-lite universe, single-regime tape). They set priors, not certainties; deterministic safety (stops, kill switches, exemptions) always outranks this doctrine."""


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
