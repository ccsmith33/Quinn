"""DOCTRINE memory provider — seeder idempotency + provider semantics.

Covers: the seeder inserts doctrine v1 exactly once (second call no-op;
a pre-existing active doctrine row is NEVER overwritten or duplicated);
the provider returns the active doctrine content for every purpose /
symbol and None when no active row exists; registration is gated on
`config.memory.doctrine_enabled`; output is deterministic.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.memory_context import (
    MEMORY_CONTAINMENT_HEADER,
    MemoryContextAssembler,
    MemoryQuery,
)
from app.memory_doctrine import (
    DOCTRINE_V1,
    make_doctrine_provider,
    register_doctrine,
    seed_doctrine_v1,
)
from config.loader import MemoryConfig
from journal.migrate import apply_migrations
from journal.models import DeskMemoryRow
from journal.repo import JournalRepo


@pytest.fixture
def journal(tmp_path: Path) -> JournalRepo:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return JournalRepo(str(p))


def _doctrine_rows(journal: JournalRepo) -> list[tuple]:
    with sqlite3.connect(journal.db_path) as conn:
        return conn.execute(
            "SELECT kind, content, version, active FROM desk_memory "
            "WHERE kind = 'doctrine' ORDER BY id"
        ).fetchall()


def _q(
    symbol: str | None = "ACME", purpose: str = "analyze"
) -> MemoryQuery:
    return MemoryQuery(symbol=symbol, purpose=purpose)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# doctrine content
# ---------------------------------------------------------------------------


def test_doctrine_v1_content_shape() -> None:
    assert DOCTRINE_V1.startswith(
        "QUINN DESK DOCTRINE v1 "
        "(33,851 catalyst events 2026-07/08, backtest priors)"
    )
    # Every distilled finding is present, and the safety caveat closes it.
    for heading in (
        "- EDGE:",
        "- SLOW BLOOMERS:",
        "- WIGGLE:",
        "- VELOCITY=VARIANCE:",
        "- EVICTION:",
        "- PEAKS:",
        "- ODDS:",
        "- CAVEATS:",
    ):
        assert heading in DOCTRINE_V1
    assert DOCTRINE_V1.endswith("(stops/KS/exemptions) outranks.")


def test_doctrine_v1_token_budget() -> None:
    """Review advisory #5: the doctrine rides UNCACHED on every LLM call,
    so it must stay <= ~300 tokens (chars/4 estimate)."""
    assert len(DOCTRINE_V1) / 4 <= 300


def test_doctrine_v1_keeps_every_study_number() -> None:
    """The trim cut prose, never data — every figure from the eight
    studies survives."""
    for figure in (
        "33,851",
        ">=+50% MFE",
        "14%",
        "130%",
        ">=+100%",
        "64%",
        "~2x",
        "+1.0%",
        "45% flat/down",
        "1-2d",
        "~98%",
        "8-12%",
        "27.6%->8.9%",
        "0.0% fade-to-loss",
        "-5.7%",
        "28%",
        "58%/31%",
        "day-3/5",
        "down>10%",
        "~2.1%/trade",
        "~0.056%/slot-day",
        "conv>=8",
        "~+8.4%/trade",
        "~47%",
        "30min",
        "day 3-7+",
        "+20%->8%",
        "+35%->5%",
        "+163%",
        "+29%",
        "P(+30|+10)=43%",
        "59%",
        "P(+50|+30)=54%",
        "P(2x|+50)=38%",
    ):
        assert figure in DOCTRINE_V1, figure


# ---------------------------------------------------------------------------
# seeder
# ---------------------------------------------------------------------------


def test_seed_inserts_doctrine_v1_when_absent(journal: JournalRepo) -> None:
    assert seed_doctrine_v1(journal) is True

    rows = _doctrine_rows(journal)
    assert len(rows) == 1
    kind, content, version, active = rows[0]
    assert kind == "doctrine"
    assert content == DOCTRINE_V1
    assert version == 1
    assert active == 1


def test_seed_second_call_is_noop(journal: JournalRepo) -> None:
    assert seed_doctrine_v1(journal) is True
    assert seed_doctrine_v1(journal) is False
    assert len(_doctrine_rows(journal)) == 1


def test_seed_never_overwrites_existing_active_doctrine(
    journal: JournalRepo,
) -> None:
    journal.insert_desk_memory(
        DeskMemoryRow(
            kind="doctrine", content="OPERATOR DOCTRINE v3", version=3
        )
    )

    assert seed_doctrine_v1(journal) is False

    rows = _doctrine_rows(journal)
    assert len(rows) == 1
    assert rows[0][1] == "OPERATOR DOCTRINE v3"
    assert rows[0][2] == 3


def test_seed_ignores_inactive_synthesis_and_deactivated_rows(
    journal: JournalRepo,
) -> None:
    # A deactivated doctrine row and an active synthesis row do NOT count
    # as an active doctrine — the seeder must still insert v1.
    journal.insert_desk_memory(
        DeskMemoryRow(kind="doctrine", content="retired", version=1, active=0)
    )
    journal.insert_desk_memory(
        DeskMemoryRow(kind="synthesis", content="post-mortems", version=1)
    )

    assert seed_doctrine_v1(journal) is True
    active = journal.get_active_desk_memory("doctrine")
    assert active is not None
    assert active.content == DOCTRINE_V1


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------


def test_provider_returns_active_doctrine_section(
    journal: JournalRepo,
) -> None:
    seed_doctrine_v1(journal)
    provide = make_doctrine_provider(journal)

    section = provide(_q())

    assert section is not None
    assert section.title == "Desk doctrine (study-derived priors)"
    assert section.body == DOCTRINE_V1
    assert section.provider_name == "doctrine"


def test_provider_serves_all_purposes_and_symbols(
    journal: JournalRepo,
) -> None:
    seed_doctrine_v1(journal)
    provide = make_doctrine_provider(journal)

    for purpose in ("analyze", "proposal_review", "thesis_review"):
        for symbol in ("ACME", None):
            section = provide(_q(symbol=symbol, purpose=purpose))
            assert section is not None
            assert section.body == DOCTRINE_V1


def test_provider_returns_none_when_no_active_row(
    journal: JournalRepo,
) -> None:
    provide = make_doctrine_provider(journal)
    assert provide(_q()) is None

    # A deactivated row is still not served.
    journal.insert_desk_memory(
        DeskMemoryRow(kind="doctrine", content="retired", version=1, active=0)
    )
    assert provide(_q()) is None


def test_provider_serves_latest_active_version_from_db(
    journal: JournalRepo,
) -> None:
    seed_doctrine_v1(journal)
    # Publish v2 via the atomic swap helper — the canonical way to
    # replace the active version under the 009 one-active-per-kind index.
    journal.insert_desk_memory_replacing_active(
        DeskMemoryRow(kind="doctrine", content="DOCTRINE v2", version=2)
    )

    section = make_doctrine_provider(journal)(_q())
    assert section is not None
    assert section.body == "DOCTRINE v2"


# ---------------------------------------------------------------------------
# one-active-per-kind invariant (009 partial unique index, advisory #9)
# ---------------------------------------------------------------------------


def test_second_active_doctrine_row_is_rejected(
    journal: JournalRepo,
) -> None:
    seed_doctrine_v1(journal)
    with pytest.raises(sqlite3.IntegrityError):
        journal.insert_desk_memory(
            DeskMemoryRow(kind="doctrine", content="rogue v2", version=2)
        )
    # The active row is untouched.
    active = journal.get_active_desk_memory("doctrine")
    assert active is not None
    assert active.content == DOCTRINE_V1


def test_one_active_per_kind_allows_other_kinds_and_inactive_rows(
    journal: JournalRepo,
) -> None:
    seed_doctrine_v1(journal)
    # An active synthesis coexists with the active doctrine...
    journal.insert_desk_memory(
        DeskMemoryRow(kind="synthesis", content="patterns", version=1)
    )
    # ...and any number of retired rows of either kind are fine.
    journal.insert_desk_memory(
        DeskMemoryRow(kind="doctrine", content="old", version=0, active=0)
    )
    journal.insert_desk_memory(
        DeskMemoryRow(kind="doctrine", content="older", version=0, active=0)
    )
    assert journal.get_active_desk_memory("doctrine") is not None
    assert journal.get_active_desk_memory("synthesis") is not None


def test_provider_is_deterministic(journal: JournalRepo) -> None:
    seed_doctrine_v1(journal)
    provide = make_doctrine_provider(journal)

    first = provide(_q())
    second = provide(_q())
    assert first == second

    # And through the rail: byte-identical assembled output.
    a = MemoryContextAssembler()
    a.register("doctrine", provide)
    assert a.assemble(_q()) == a.assemble(_q())


# ---------------------------------------------------------------------------
# registration gate
# ---------------------------------------------------------------------------


def test_register_doctrine_gated_off(journal: JournalRepo) -> None:
    cfg = MemoryConfig(enabled=True, doctrine_enabled=False)
    a = MemoryContextAssembler()

    register_doctrine(a, cfg, journal)

    # Not registered AND not seeded — the gate silences the whole path.
    assert a.assemble(_q()) is None
    assert _doctrine_rows(journal) == []


def test_register_doctrine_gated_on_seeds_and_registers(
    journal: JournalRepo,
) -> None:
    cfg = MemoryConfig(enabled=True, doctrine_enabled=True)
    a = MemoryContextAssembler()

    register_doctrine(a, cfg, journal)

    out = a.assemble(_q())
    assert out is not None
    assert out.startswith(
        MEMORY_CONTAINMENT_HEADER
        + "\n\n## MEMORY: Desk doctrine (study-derived priors)\n"
    )
    assert DOCTRINE_V1 in out
    assert len(_doctrine_rows(journal)) == 1


def test_register_doctrine_idempotent_across_boots(
    journal: JournalRepo,
) -> None:
    cfg = MemoryConfig(enabled=True, doctrine_enabled=True)

    # Two boots (fresh assembler each time, shared journal): one row.
    for _ in range(2):
        a = MemoryContextAssembler()
        register_doctrine(a, cfg, journal)
    assert len(_doctrine_rows(journal)) == 1
