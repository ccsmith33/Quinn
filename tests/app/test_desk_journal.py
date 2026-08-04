"""Desk journal (LLM memory layer) — post-mortem generation, weekly
synthesis, and the `desk_journal` memory provider.

Covers: the close-detection query (closed-without-postmortem, excludes
open / rejected / already-postmortemed, honours the cap); exit-quality
computed in CODE; the happy path through the REAL AnthropicClient (fake
SDK underneath) so the `llm_calls` telemetry row with purpose
'postmortem' is proven; malformed-output skip + next-day retry; the
daily cap and once-per-day gating; weekly ISO-week (ET) synthesis gating
(<3 new post-mortems → skip; version increments; prior deactivated;
same-week no-op, restart-proof); provider render / None / determinism.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from analyzer.anthropic_client import AnthropicClient
from app.desk_journal import (
    DESK_JOURNAL_SECTION_TITLE,
    HAIKU_MODEL_ID,
    DeskJournalTicker,
    compute_exit_quality_pct,
    make_desk_journal_provider,
)
from app.memory_context import MemoryContextAssembler, MemoryQuery
from config.loader import MemoryConfig
from journal.migrate import apply_migrations
from journal.models import (
    DeskMemoryRow,
    ExecutionRow,
    FilingRow,
    OrderRow,
    PromptRow,
    ProposalRow,
    TradePostmortemRow,
)
from journal.repo import (
    JournalRepo,
    find_closed_executions_without_postmortem,
    get_active_desk_memory,
    get_llm_calls_by_decision_id,
    get_postmortems_for_symbol,
    insert_desk_memory,
    insert_execution,
    insert_filing,
    insert_order,
    insert_prompt,
    insert_proposal,
    insert_trade_postmortem,
)
from prompts.loader import PromptBuilder

PROMPTS_DIR = Path("src/prompts")

_ENTRY_AT = dt.datetime(2026, 7, 10, 14, 0, 0)
_EXIT_AT = dt.datetime(2026, 7, 16, 15, 0, 0)

# A Wednesday 15:00 UTC = 11:00 ET, ISO week 30 of 2026.
_NOW_WEEK30 = dt.datetime(2026, 7, 22, 15, 0, 0, tzinfo=dt.UTC)
# The following Wednesday — ISO week 31.
_NOW_WEEK31 = dt.datetime(2026, 7, 29, 15, 0, 0, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# Fakes (same shape as tests/analyzer/test_anthropic_client.py)
# ---------------------------------------------------------------------------


@dataclass
class _FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeResponse:
    content: list[_FakeTextBlock]
    usage: _FakeUsage


def _response(text: str) -> _FakeResponse:
    return _FakeResponse(content=[_FakeTextBlock(text=text)], usage=_FakeUsage())


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses: list[Any] = []

    def queue(self, response: Any) -> None:
        self._responses.append(response)

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("no fake response queued")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeAsyncAnthropic:
    def __init__(self, **_: Any) -> None:
        self.messages = _FakeMessages()


# ---------------------------------------------------------------------------
# Fixtures / seed helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return str(p)


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder(PROMPTS_DIR)


@pytest.fixture
def registered_db(db: str, builder: PromptBuilder) -> str:
    """DB with the two desk-journal composed prompt versions registered so
    the `llm_calls.prompt_version` FK resolves (composition does this at
    boot via `register_composed_prompt_versions`)."""
    for name in ("desk_postmortem_v1", "desk_synthesis_v1"):
        insert_prompt(
            db,
            PromptRow(
                prompt_version=builder.prompt_version(name),
                name=name,
                file_path=str(PROMPTS_DIR / f"{name}.txt"),
                content_hash="a" * 64,
            ),
        )
    return db


_SEED_PROMPT_VERSION = "pv@aaaaaaaaaaaa"


def _seed_execution(
    db_path: str,
    symbol: str,
    *,
    thesis: str = "8-K beat; expect re-rating",
    conviction: int = 7,
    decision: str = "accepted",
) -> int:
    with sqlite3.connect(db_path) as conn:
        have = conn.execute(
            "SELECT 1 FROM prompts WHERE prompt_version = ?",
            (_SEED_PROMPT_VERSION,),
        ).fetchone()
    if have is None:
        insert_prompt(
            db_path,
            PromptRow(
                prompt_version=_SEED_PROMPT_VERSION,
                name="sonnet_filing_analysis_v2",
                file_path="src/prompts/sonnet_filing_analysis_v2.txt",
                content_hash="b" * 64,
            ),
        )
    fid = insert_filing(
        db_path,
        FilingRow(
            accession_number=f"acc-{symbol}",
            cik=1234567,
            form_type="8-K",
            filed_at=dt.datetime(2026, 7, 1, 14, 30, 0),
            fetched_at=dt.datetime(2026, 7, 1, 14, 31, 0),
            raw_text_path=f"/raw/{symbol}.txt",
            content_hash=f"h-{symbol}",
        ),
    )
    pid = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=fid,
            decision_id=f"dec-{symbol}",
            model_id="claude-haiku-4-5-20251001",
            prompt_version=_SEED_PROMPT_VERSION,
            raw_response="{}",
            kind="trade_proposal",
            symbol=symbol,
            direction="long",
            size_pct_requested=0.05,
            conviction=conviction,
            thesis=thesis,
            input_tokens=10,
            output_tokens=10,
            latency_ms=100,
            cost_usd=0.001,
        ),
    )
    return insert_execution(
        db_path,
        ExecutionRow(
            proposal_id=pid, decision=decision, submitted_orders_json="[]"
        ),
    )


def _add_order(
    db_path: str,
    execution_id: int,
    *,
    role: str,
    side: str,
    symbol: str,
    fill_price: float | None,
    fill_at: dt.datetime | None,
    final_status: str | None = "filled",
) -> int:
    return insert_order(
        db_path,
        OrderRow(
            execution_id=execution_id,
            role=role,
            symbol=symbol,
            side=side,
            order_type="market",
            qty=10,
            tif="day",
            broker_order_id=f"b-{execution_id}-{role}-{side}",
            submitted_at=fill_at or _ENTRY_AT,
            final_status=final_status,
            realized_fill_price=fill_price,
            realized_fill_qty=10 if fill_price is not None else None,
            realized_fill_at=fill_at,
        ),
    )


def _seed_closed_trade(
    db_path: str,
    symbol: str,
    *,
    entry_price: float = 10.0,
    exit_price: float = 11.5,
    exit_role: str = "trailing_stop",
    hwm: float | None = 12.0,
    conviction: int = 7,
    thesis: str = "8-K beat; expect re-rating",
) -> int:
    eid = _seed_execution(db_path, symbol, thesis=thesis, conviction=conviction)
    _add_order(
        db_path, eid, role="entry", side="buy", symbol=symbol,
        fill_price=entry_price, fill_at=_ENTRY_AT,
    )
    _add_order(
        db_path, eid, role=exit_role, side="sell", symbol=symbol,
        fill_price=exit_price, fill_at=_EXIT_AT,
    )
    if hwm is not None:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO exit_policy_state "
                "(execution_id, symbol, trail_distance_pct, trail_engaged, "
                "high_water_mark) VALUES (?, ?, 10.0, 1, ?)",
                (eid, symbol, hwm),
            )
    return eid


def _seed_postmortems(db_path: str, n: int, *, prefix: str = "PM") -> list[int]:
    ids = []
    for i in range(n):
        eid = _seed_closed_trade(db_path, f"{prefix}{i}")
        ids.append(
            insert_trade_postmortem(
                db_path,
                TradePostmortemRow(
                    execution_id=eid,
                    symbol=f"{prefix}{i}",
                    thesis_summary=f"thesis {i}",
                    outcome_summary=f"outcome {i}",
                    exit_quality_pct=50.0 + i,
                    lesson=f"lesson {i}",
                ),
            )
        )
    return ids


def _make_ticker(
    db_path: str,
    builder: PromptBuilder,
    *,
    daily_cap: int = 10,
    now: dt.datetime = _NOW_WEEK30,
) -> tuple[DeskJournalTicker, _FakeAsyncAnthropic, list[dt.datetime]]:
    fake = _FakeAsyncAnthropic()
    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id=HAIKU_MODEL_ID,
        db_path=db_path,
        _client=fake,
    )
    clock = [now]
    ticker = DeskJournalTicker(
        journal=JournalRepo(db_path),
        client=client,
        prompt_builder=builder,
        daily_cap=daily_cap,
        now_fn=lambda: clock[0],
    )
    return ticker, fake, clock


def _pm_json(thesis: str = "t", outcome: str = "o", lesson: str = "l") -> str:
    return json.dumps(
        {"thesis_summary": thesis, "outcome_summary": outcome, "lesson": lesson}
    )


# ---------------------------------------------------------------------------
# Close-detection query
# ---------------------------------------------------------------------------


def test_close_detection_returns_closed_without_postmortem(db: str) -> None:
    eid = _seed_closed_trade(db, "ACME", exit_role="stop", hwm=None)
    rows = find_closed_executions_without_postmortem(db, limit=10)
    assert [r["execution_id"] for r in rows] == [eid]
    r = rows[0]
    assert r["symbol"] == "ACME"
    assert r["thesis"] == "8-K beat; expect re-rating"
    assert r["conviction"] == 7
    assert r["entry_price"] == 10.0
    assert r["exit_price"] == 11.5
    assert r["exit_role"] == "stop"
    assert r["high_water_mark"] is None


def test_close_detection_excludes_open_rejected_and_postmortemed(
    db: str,
) -> None:
    # Open: entry filled, no exit fill (pending stop).
    open_eid = _seed_execution(db, "OPEN")
    _add_order(
        db, open_eid, role="entry", side="buy", symbol="OPEN",
        fill_price=10.0, fill_at=_ENTRY_AT,
    )
    _add_order(
        db, open_eid, role="stop", side="sell", symbol="OPEN",
        fill_price=None, fill_at=None, final_status=None,
    )
    # Rejected execution with (impossible but defensive) filled orders.
    rej_eid = _seed_execution(db, "REJD", decision="rejected")
    _add_order(
        db, rej_eid, role="entry", side="buy", symbol="REJD",
        fill_price=10.0, fill_at=_ENTRY_AT,
    )
    _add_order(
        db, rej_eid, role="stop", side="sell", symbol="REJD",
        fill_price=9.0, fill_at=_EXIT_AT,
    )
    # Closed but already post-mortemed.
    done_eid = _seed_closed_trade(db, "DONE")
    insert_trade_postmortem(
        db, TradePostmortemRow(execution_id=done_eid, symbol="DONE")
    )
    # Entry never filled (canceled before fill), exit absent.
    nofill_eid = _seed_execution(db, "NOFL")
    _add_order(
        db, nofill_eid, role="entry", side="buy", symbol="NOFL",
        fill_price=None, fill_at=None, final_status="canceled",
    )

    assert find_closed_executions_without_postmortem(db, limit=10) == []


def test_close_detection_cap_and_oldest_first(db: str) -> None:
    ids = [_seed_closed_trade(db, f"S{i}") for i in range(4)]
    rows = find_closed_executions_without_postmortem(db, limit=3)
    assert [r["execution_id"] for r in rows] == ids[:3]


# ---------------------------------------------------------------------------
# Exit quality — computed in code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "exit_", "peak", "expected"),
    [
        (10.0, 11.5, 12.0, 75.0),      # captured 1.5 of 2.0 peak gain
        (10.0, 12.0, None, 100.0),     # no HWM row: exit IS the peak
        (10.0, 12.0, 11.0, 100.0),     # HWM lagged a gap exit
        (10.0, 9.0, 12.0, -50.0),      # round-tripped peak into a loss
        (10.0, 9.0, None, None),       # never above entry: undefined
        (10.0, 10.0, 10.0, None),      # flat: no favorable move
        (None, 11.0, 12.0, None),      # missing entry price
        (10.0, None, 12.0, None),      # missing exit price
    ],
)
def test_compute_exit_quality_pct(
    entry: float | None,
    exit_: float | None,
    peak: float | None,
    expected: float | None,
) -> None:
    assert (
        compute_exit_quality_pct(
            entry_price=entry, exit_price=exit_, peak_price=peak
        )
        == expected
    )


# ---------------------------------------------------------------------------
# Post-mortem generation
# ---------------------------------------------------------------------------


def test_postmortem_happy_path_parses_inserts_and_journals_llm_call(
    registered_db: str, builder: PromptBuilder
) -> None:
    eid = _seed_closed_trade(registered_db, "ACME")
    ticker, fake, _ = _make_ticker(registered_db, builder)
    fake.messages.queue(
        _response(
            _pm_json(
                "8-K beat, re-rating expected",
                "rose 15% in 6 days, trail stopped out",
                "trail captured most of the move",
            )
        )
    )

    asyncio.run(ticker.run_tick())

    rows = get_postmortems_for_symbol(registered_db, "ACME")
    assert len(rows) == 1
    pm = rows[0]
    assert pm.execution_id == eid
    assert pm.thesis_summary == "8-K beat, re-rating expected"
    assert pm.outcome_summary == "rose 15% in 6 days, trail stopped out"
    assert pm.lesson == "trail captured most of the move"
    # exit_quality is CODE-computed: (11.5-10)/(12-10) = 75.0 — the LLM
    # never supplied it.
    assert pm.exit_quality_pct == 75.0

    # Cost landed in llm_calls under purpose 'postmortem'.
    calls = get_llm_calls_by_decision_id(
        registered_db, f"postmortem-{eid:08d}"
    )
    assert len(calls) == 1
    assert calls[0].purpose == "postmortem"
    assert calls[0].model_id == HAIKU_MODEL_ID
    assert calls[0].error_class is None

    # The request carried the code-computed facts + verbatim thesis.
    sent = fake.messages.calls[0]
    assert sent["model"] == HAIKU_MODEL_ID
    user_text = sent["messages"][0]["content"][0]["text"]
    assert "exit_quality_pct: 75.0" in user_text
    assert "exit_mechanism: trail" in user_text
    assert "days_held: 6" in user_text
    assert "8-K beat; expect re-rating" in user_text


def test_postmortem_malformed_output_skipped_then_retried_next_day(
    registered_db: str, builder: PromptBuilder
) -> None:
    _seed_closed_trade(registered_db, "ACME")
    ticker, fake, clock = _make_ticker(registered_db, builder)
    fake.messages.queue(_response("sorry, no JSON today"))

    asyncio.run(ticker.run_tick())  # malformed → skip, never crash
    assert get_postmortems_for_symbol(registered_db, "ACME") == []

    # Same day: once-per-day gate — no retry, no LLM call.
    asyncio.run(ticker.run_tick())
    assert len(fake.messages.calls) == 1

    # Next day: candidate is still queued (no row) → retried and written.
    clock[0] = _NOW_WEEK30 + dt.timedelta(days=1)
    fake.messages.queue(_response(_pm_json()))
    asyncio.run(ticker.run_tick())
    assert len(get_postmortems_for_symbol(registered_db, "ACME")) == 1


def test_postmortem_daily_cap_drains_backlog_gradually(
    registered_db: str, builder: PromptBuilder
) -> None:
    for i in range(5):
        _seed_closed_trade(registered_db, f"S{i}")
    ticker, fake, clock = _make_ticker(registered_db, builder, daily_cap=3)
    for _ in range(3):
        fake.messages.queue(_response(_pm_json()))
    # Day 1 also crosses the >=3-new-postmortems line, so the weekly
    # synthesis legitimately fires once right after the capped batch.
    fake.messages.queue(_response("first synthesis (n=3)"))

    asyncio.run(ticker.run_tick())
    with sqlite3.connect(registered_db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM trade_postmortems").fetchone()[0]
    assert n == 3
    assert len(fake.messages.calls) == 4  # 3 postmortems + 1 synthesis

    # Day 2 drains the remaining 2.
    clock[0] = _NOW_WEEK30 + dt.timedelta(days=1)
    for _ in range(2):
        fake.messages.queue(_response(_pm_json()))
    asyncio.run(ticker.run_tick())
    with sqlite3.connect(registered_db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM trade_postmortems").fetchone()[0]
    assert n == 5


def test_postmortem_llm_failure_of_one_trade_does_not_block_others(
    registered_db: str, builder: PromptBuilder
) -> None:
    _seed_closed_trade(registered_db, "AAAA")
    _seed_closed_trade(registered_db, "BBBB")
    ticker, fake, _ = _make_ticker(registered_db, builder)

    # First call blows up inside the SDK (non-transient → client raises);
    # second succeeds.
    fake.messages.queue(RuntimeError("boom"))
    fake.messages.queue(_response(_pm_json()))
    asyncio.run(ticker.run_tick())

    assert get_postmortems_for_symbol(registered_db, "AAAA") == []
    assert len(get_postmortems_for_symbol(registered_db, "BBBB")) == 1


# ---------------------------------------------------------------------------
# Weekly synthesis
# ---------------------------------------------------------------------------


def test_synthesis_skipped_below_three_new_postmortems(
    registered_db: str, builder: PromptBuilder
) -> None:
    _seed_postmortems(registered_db, 2)
    ticker, fake, _ = _make_ticker(registered_db, builder)

    asyncio.run(ticker.run_tick())

    assert get_active_desk_memory(registered_db, "synthesis") is None
    assert fake.messages.calls == []  # no LLM spend at all


def test_synthesis_first_version_written_and_capped(
    registered_db: str, builder: PromptBuilder
) -> None:
    _seed_postmortems(registered_db, 3)
    ticker, fake, _ = _make_ticker(registered_db, builder)
    # 17 lines + blank noise → code caps at 15.
    fake.messages.queue(
        _response("\n".join(f"pattern line {i} (n=3)" for i in range(17)) + "\n\n")
    )

    asyncio.run(ticker.run_tick())

    row = get_active_desk_memory(registered_db, "synthesis")
    assert row is not None
    assert row.version == 1
    assert row.active == 1
    lines = row.content.splitlines()
    assert len(lines) == 15
    assert lines[0] == "pattern line 0 (n=3)"

    calls = get_llm_calls_by_decision_id(registered_db, "synthesis-2026W30")
    assert len(calls) == 1
    assert calls[0].purpose == "synthesis"
    assert calls[0].model_id == HAIKU_MODEL_ID

    # The digest carried the postmortems and their sample count.
    digest = fake.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "count: 3" in digest
    assert "[PM0]" in digest

    # Same week, later tick: no second synthesis.
    fake.messages.calls.clear()
    asyncio.run(ticker.run_tick())
    assert fake.messages.calls == []


def test_synthesis_version_increments_and_prior_deactivated(
    registered_db: str, builder: PromptBuilder
) -> None:
    prev_id = insert_desk_memory(
        registered_db,
        DeskMemoryRow(kind="synthesis", content="old patterns", version=1),
    )
    # Backdate the prior synthesis into ISO week 29 so week 30 is "new".
    with sqlite3.connect(registered_db) as conn:
        conn.execute(
            "UPDATE desk_memory SET created_at = '2026-07-15 12:00:00' "
            "WHERE id = ?",
            (prev_id,),
        )
    _seed_postmortems(registered_db, 3)  # created now → after the backdate
    ticker, fake, _ = _make_ticker(registered_db, builder)
    fake.messages.queue(_response("fresh pattern (2 of 3 trades)"))

    asyncio.run(ticker.run_tick())

    row = get_active_desk_memory(registered_db, "synthesis")
    assert row is not None
    assert row.version == 2
    assert row.content == "fresh pattern (2 of 3 trades)"
    with sqlite3.connect(registered_db) as conn:
        old_active = conn.execute(
            "SELECT active FROM desk_memory WHERE id = ?", (prev_id,)
        ).fetchone()[0]
    assert old_active == 0


def test_synthesis_same_week_restart_proof(
    registered_db: str, builder: PromptBuilder
) -> None:
    """A fresh ticker (restart: in-memory week marker lost) must not
    re-synthesize inside the same ISO week — durable created_at check."""
    sid = insert_desk_memory(
        registered_db,
        DeskMemoryRow(kind="synthesis", content="this week", version=1),
    )
    with sqlite3.connect(registered_db) as conn:
        # Same ISO week (ET) as _NOW_WEEK30.
        conn.execute(
            "UPDATE desk_memory SET created_at = '2026-07-21 12:00:00' "
            "WHERE id = ?",
            (sid,),
        )
    _seed_postmortems(registered_db, 5)
    ticker, fake, _ = _make_ticker(registered_db, builder)

    asyncio.run(ticker.run_tick())

    assert fake.messages.calls == []
    row = get_active_desk_memory(registered_db, "synthesis")
    assert row is not None and row.version == 1

    # Next ISO week the same backlog does synthesize.
    ticker2, fake2, _ = _make_ticker(registered_db, builder, now=_NOW_WEEK31)
    fake2.messages.queue(_response("week-31 patterns (n=5)"))
    asyncio.run(ticker2.run_tick())
    row = get_active_desk_memory(registered_db, "synthesis")
    assert row is not None and row.version == 2


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def test_provider_returns_none_without_synthesis(db: str) -> None:
    provider = make_desk_journal_provider(db)
    assert provider(MemoryQuery(symbol="ACME", purpose="analyze")) is None


def test_provider_serves_active_synthesis_for_all_purposes(db: str) -> None:
    insert_desk_memory(
        db,
        DeskMemoryRow(kind="synthesis", content="cv6 entries stopped early (3 of 4)"),
    )
    provider = make_desk_journal_provider(db)
    for purpose in ("analyze", "proposal_review", "thesis_review"):
        section = provider(MemoryQuery(symbol=None, purpose=purpose))  # type: ignore[arg-type]
        assert section is not None
        assert section.title == DESK_JOURNAL_SECTION_TITLE
        assert section.body == "cv6 entries stopped early (3 of 4)"
        assert section.provider_name == "desk_journal"


def test_provider_output_is_deterministic_through_assembler(db: str) -> None:
    insert_desk_memory(
        db, DeskMemoryRow(kind="synthesis", content="stable patterns (n=4)")
    )
    assembler = MemoryContextAssembler()
    assembler.register("desk_journal", make_desk_journal_provider(db))
    query = MemoryQuery(symbol="ACME", purpose="thesis_review")
    first = assembler.assemble(query)
    second = assembler.assemble(query)
    assert first is not None
    assert first == second  # byte-identical for identical inputs
    assert first == (
        f"## MEMORY: {DESK_JOURNAL_SECTION_TITLE}\nstable patterns (n=4)"
    )


def test_provider_ignores_doctrine_rows(db: str) -> None:
    insert_desk_memory(db, DeskMemoryRow(kind="doctrine", content="doctrine"))
    provider = make_desk_journal_provider(db)
    assert provider(MemoryQuery(symbol=None, purpose="analyze")) is None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_memory_config_postmortem_daily_cap_default_and_bounds() -> None:
    assert MemoryConfig().postmortem_daily_cap == 10
    assert MemoryConfig(postmortem_daily_cap=50).postmortem_daily_cap == 50
    with pytest.raises(ValidationError):
        MemoryConfig(postmortem_daily_cap=0)
    with pytest.raises(ValidationError):
        MemoryConfig(postmortem_daily_cap=51)
