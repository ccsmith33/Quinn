"""Memory-layer wiring — proves the assembler output is threaded into the
analyzer + thesis-review LLM context, and that with the memory master gate
OFF (no assembler) the built context is byte-identical to pre-memory.

Two layers of coverage:
  * PromptBuilder byte-identity — `memory_block=None` reproduces the old
    per-call block exactly; a supplied block is appended at the tail
    (for thesis review, AFTER the capacity block).
  * Analyzer / reviewer seam — `_assemble_memory` builds the right
    MemoryQuery and returns None when disabled; and analyze() actually
    passes the assembled block to the builder.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from analyzer.sonnet import SonnetAnalyzer
from analyzer.thesis_review import ThesisReviewContext, ThesisReviewer
from app.memory_context import (
    MEMORY_CONTAINMENT_HEADER,
    MemoryContextAssembler,
    MemoryQuery,
    MemorySection,
)
from journal.migrate import apply_migrations
from journal.models import FilingRow, LlmCallRow, PromptRow, ProposalRow
from journal.repo import (
    get_filing_by_id,
    insert_filing,
    insert_llm_call,
    insert_prompt,
)
from prompts.loader import (
    AnalyzerContext,
    ApiRequest,
    Block,
    Message,
    PromptBuilder,
)

PROMPTS_DIR = Path("src/prompts")
_HDR = MEMORY_CONTAINMENT_HEADER + "\n\n"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _filing_row(**overrides: Any) -> FilingRow:
    base = dict(
        accession_number="0001234567-26-000456",
        cik=1234567,
        form_type="8-K",
        filed_at=dt.datetime(2026, 4, 28, 14, 30, 0),
        fetched_at=dt.datetime(2026, 4, 28, 14, 31, 0),
        raw_text_path="/raw/acme.txt",
        content_hash="aaa222",
        item_codes='["1.01"]',
        issuer_ticker="ACME",
    )
    base.update(overrides)
    return FilingRow(**base)


def _proposal_row(**overrides: Any) -> ProposalRow:
    base = dict(
        id=99,
        filing_id=42,
        decision_id="dec_xyz_001",
        model_id="claude-sonnet-4-6",
        prompt_version="sonnet_filing_analysis_v2@aaaaaaaaaaaa",
        raw_response="{}",
        kind="trade_proposal",
        symbol="ACME",
        direction="long",
        size_pct_requested=0.05,
        conviction=8,
        thesis="material 8-K",
        input_tokens=10,
        output_tokens=10,
        latency_ms=100,
        cost_usd=0.001,
    )
    base.update(overrides)
    return ProposalRow(**base)


def _thesis_ctx(**overrides: Any) -> ThesisReviewContext:
    base: dict[str, Any] = dict(
        proposal=_proposal_row(),
        execution_id=7,
        schedule_id=3,
        days_held=6,
        current_price=12.34,
        pct_change_since_entry=0.10,
        realized_fill_price=11.0,
        realized_dollar_size=1000.0,
        time_horizon_days=14,
        stop_loss_price=9.5,
        take_profit_price=None,
        filings_since_entry_summary="none",
    )
    base.update(overrides)
    return ThesisReviewContext(**base)


def _assembler_with_section(body: str = "base rates") -> MemoryContextAssembler:
    a = MemoryContextAssembler()

    def provide(_q: MemoryQuery) -> MemorySection | None:
        return MemorySection(title="Doctrine", body=body, provider_name="doctrine")

    a.register("doctrine", provide)
    return a


def _block3(request: ApiRequest) -> str:
    """The per-call user message text (block 3)."""
    return request.messages[0].content[0].text


# ---------------------------------------------------------------------------
# PromptBuilder byte-identity — sonnet filing analysis
# ---------------------------------------------------------------------------


def test_sonnet_memory_none_is_byte_identical() -> None:
    builder = PromptBuilder(PROMPTS_DIR)
    filing = _filing_row()
    ctx = AnalyzerContext(
        universe_summary="ACME",
        kill_switch_state="ok",
        open_positions_count=0,
        decision_id="d1",
    )
    baseline = builder.build_sonnet_filing_analysis(filing, "body", ctx)
    with_none = builder.build_sonnet_filing_analysis(
        filing, "body", ctx, memory_block=None
    )
    assert _block3(with_none) == _block3(baseline)
    assert "MEMORY" not in _block3(baseline)


def test_sonnet_memory_block_appended_at_tail() -> None:
    builder = PromptBuilder(PROMPTS_DIR)
    filing = _filing_row()
    ctx = AnalyzerContext(
        universe_summary="ACME",
        kill_switch_state="ok",
        open_positions_count=0,
        decision_id="d1",
    )
    baseline = _block3(builder.build_sonnet_filing_analysis(filing, "body", ctx))
    mem = "## MEMORY: Doctrine\nbase rates"
    got = _block3(
        builder.build_sonnet_filing_analysis(
            filing, "body", ctx, memory_block=mem
        )
    )
    assert got == f"{baseline}---\n{mem}\n"


# ---------------------------------------------------------------------------
# PromptBuilder byte-identity — thesis review (memory AFTER capacity block)
# ---------------------------------------------------------------------------


def test_thesis_memory_none_is_byte_identical() -> None:
    builder = PromptBuilder(PROMPTS_DIR)
    p = _proposal_row()
    kw: dict[str, Any] = dict(
        proposal=p,
        execution_id=7,
        days_held=6,
        current_price=12.34,
        pct_change_since_entry=0.10,
        realized_fill_price=11.0,
        realized_dollar_size=1000.0,
        time_horizon_days=14,
        stop_loss_price=9.5,
        take_profit_price=None,
        filings_since_entry_summary="none",
        capacity_pressure_block="CAP",
    )
    baseline = _block3(builder.build_opus_thesis_review(**kw))
    with_none = _block3(builder.build_opus_thesis_review(**kw, memory_block=None))
    assert with_none == baseline
    assert "MEMORY" not in baseline


def test_thesis_memory_appended_after_capacity_block() -> None:
    builder = PromptBuilder(PROMPTS_DIR)
    p = _proposal_row()
    kw: dict[str, Any] = dict(
        proposal=p,
        execution_id=7,
        days_held=6,
        current_price=12.34,
        pct_change_since_entry=0.10,
        realized_fill_price=11.0,
        realized_dollar_size=1000.0,
        time_horizon_days=14,
        stop_loss_price=9.5,
        take_profit_price=None,
        filings_since_entry_summary="none",
        capacity_pressure_block="CAP",
    )
    baseline = _block3(builder.build_opus_thesis_review(**kw))
    mem = "## MEMORY: Doctrine\nbase rates"
    got = _block3(builder.build_opus_thesis_review(**kw, memory_block=mem))
    # memory sits at the very end, strictly after the capacity block
    assert got == f"{baseline}---\n{mem}\n"
    assert got.index("CAP") < got.index("MEMORY")


# ---------------------------------------------------------------------------
# analyzer / reviewer seam — MemoryQuery construction + disabled short-circuit
# ---------------------------------------------------------------------------


def _sonnet_analyzer(assembler: MemoryContextAssembler | None) -> SonnetAnalyzer:
    return SonnetAnalyzer(
        client=object(),  # unused by _assemble_memory
        store=object(),
        prompt_builder=object(),
        opus_reviewer=object(),
        sonnet_model_id="m",
        opus_review_conviction_threshold=7,
        db_path=":memory:",
        memory_assembler=assembler,
    )


def test_analyzer_assemble_memory_disabled_returns_none() -> None:
    assert _sonnet_analyzer(None)._assemble_memory(_filing_row()) is None


def test_analyzer_assemble_memory_builds_analyze_query() -> None:
    seen: list[MemoryQuery] = []

    a = MemoryContextAssembler()

    def capture(q: MemoryQuery) -> MemorySection | None:
        seen.append(q)
        return MemorySection(title="T", body="B", provider_name="p")

    a.register("p", capture)
    out = _sonnet_analyzer(a)._assemble_memory(_filing_row(issuer_ticker="ACME"))
    assert out == _HDR + "## MEMORY: T\nB"
    assert seen[0].symbol == "ACME"
    assert seen[0].purpose == "analyze"
    assert seen[0].execution_id is None
    assert seen[0].conviction is None


def _thesis_reviewer(assembler: MemoryContextAssembler | None) -> ThesisReviewer:
    return ThesisReviewer(
        client=object(),
        prompt_builder=object(),
        opus_model_id="m",
        db_path=":memory:",
        memory_assembler=assembler,
    )


def test_reviewer_assemble_memory_disabled_returns_none() -> None:
    assert _thesis_reviewer(None)._assemble_memory(_thesis_ctx()) is None


def test_reviewer_assemble_memory_builds_thesis_review_query() -> None:
    seen: list[MemoryQuery] = []

    a = MemoryContextAssembler()

    def capture(q: MemoryQuery) -> MemorySection | None:
        seen.append(q)
        return MemorySection(title="T", body="B", provider_name="p")

    a.register("p", capture)
    ctx = _thesis_ctx(
        proposal=_proposal_row(symbol="ZZ", conviction=9), execution_id=42
    )
    out = _thesis_reviewer(a)._assemble_memory(ctx)
    assert out == _HDR + "## MEMORY: T\nB"
    assert seen[0].symbol == "ZZ"
    assert seen[0].purpose == "thesis_review"
    assert seen[0].execution_id == 42
    assert seen[0].conviction == 9


# ---------------------------------------------------------------------------
# analyze() end-to-end — the assembled block reaches the builder
# ---------------------------------------------------------------------------


@dataclass
class _RecordingBuilder:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def build_sonnet_filing_analysis(
        self,
        filing: FilingRow,
        raw_text: str,
        ctx: AnalyzerContext,
        *,
        memory_block: str | None = None,
    ) -> ApiRequest:
        self.calls.append({"memory_block": memory_block})
        return ApiRequest(
            system=[Block(text="b1", cache_control={"type": "ephemeral"})],
            messages=[Message(role="user", content=[Block(text=raw_text)])],
            prompt_version="pv@aaaaaaaaaaaa",
        )

    def prompt_version(self, name: str) -> str:
        return "pv@aaaaaaaaaaaa"


@dataclass
class _LlmCallWritingClient:
    db_path: str

    async def call(
        self,
        request: Any,
        *,
        model_id: str,
        purpose: str,
        decision_id: str,
        max_tokens: int | None = None,
    ) -> str:
        insert_llm_call(
            self.db_path,
            LlmCallRow(
                decision_id=decision_id,
                purpose=purpose,
                model_id=model_id,
                prompt_version=request.prompt_version,
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                latency_ms=1,
                cost_usd=0.0,
                error_class=None,
            ),
        )
        return json.dumps(
            {
                "decision": "no_trade",
                "thesis_or_reason": (
                    "Routine 8-K Item 8.01 governance disclosure, no material "
                    "catalyst; conviction below threshold for a proposal."
                ),
                "signals_considered": ["Item 8.01 — routine governance"],
            }
        )


@dataclass
class _NoopStore:
    def store(self, result: Any, **_: Any) -> int:
        return 1


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    # llm_calls.prompt_version FKs to prompts — seed the pv the recording
    # builder returns so the fake client's llm_calls insert satisfies it.
    insert_prompt(
        str(p),
        PromptRow(
            prompt_version="pv@aaaaaaaaaaaa",
            name="sonnet_filing_analysis_v2",
            file_path="src/prompts/sonnet_filing_analysis_v2.txt",
            content_hash="a" * 64,
        ),
    )
    return str(p)


def _analyzer_with(db: str, assembler: MemoryContextAssembler | None) -> tuple[
    SonnetAnalyzer, _RecordingBuilder
]:
    builder = _RecordingBuilder()
    analyzer = SonnetAnalyzer(
        client=_LlmCallWritingClient(db_path=db),
        store=_NoopStore(),
        prompt_builder=builder,
        opus_reviewer=object(),
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=7,
        db_path=db,
        memory_assembler=assembler,
    )
    return analyzer, builder


@pytest.fixture
def stored_filing(db: str) -> FilingRow:
    fid = insert_filing(db, _filing_row())
    f = get_filing_by_id(db, fid)
    assert f is not None
    return f


@pytest.mark.asyncio
async def test_analyze_passes_none_when_memory_disabled(
    db: str, stored_filing: FilingRow
) -> None:
    ctx = AnalyzerContext(
        universe_summary="ACME",
        kill_switch_state="ok",
        open_positions_count=0,
        decision_id="d",
    )
    analyzer, builder = _analyzer_with(db, None)
    await analyzer.analyze(stored_filing, raw_text="body", ctx=ctx)
    assert builder.calls[0]["memory_block"] is None


@pytest.mark.asyncio
async def test_analyze_passes_assembled_block_when_enabled(
    db: str, stored_filing: FilingRow
) -> None:
    ctx = AnalyzerContext(
        universe_summary="ACME",
        kill_switch_state="ok",
        open_positions_count=0,
        decision_id="d",
    )
    analyzer, builder = _analyzer_with(db, _assembler_with_section())
    await analyzer.analyze(stored_filing, raw_text="body", ctx=ctx)
    assert (
        builder.calls[0]["memory_block"]
        == _HDR + "## MEMORY: Doctrine\nbase rates"
    )
