"""S5.3 — Sonnet filing analyzer tests.

Architecture references: §2.3 (LLM Analyzer), §3.2 (TradeProposal schema),
§3.3 (NoTradeRecord schema), FR-15, FR-16, FR-18, FR-19, NFR-1, ADR-005.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from journal.migrate import apply_migrations
from journal.models import (
    FilingRow,
    LlmCallRow,
    PromptRow,
)
from journal.repo import (
    get_proposal_by_decision_id,
    get_proposal_review_by_proposal_id,
    insert_filing,
    insert_llm_call,
    insert_prompt,
)
from prompts.loader import AnalyzerContext, ApiRequest, Block, Message

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class _FakeAnthropicClient:
    """AnthropicClient stub that mirrors the real one's contract: writes
    one `llm_calls` row per successful call and returns the queued text."""

    db_path: str
    responses: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    fixed_telemetry: dict[str, Any] = field(
        default_factory=lambda: {
            "input_tokens": 1500,
            "output_tokens": 800,
            "cache_read_tokens": 4000,
            "cache_creation_tokens": 200,
            "latency_ms": 4321,
            "cost_usd": 0.0234,
        }
    )

    async def call(
        self,
        request: Any,
        *,
        model_id: str,
        purpose: str,
        decision_id: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "request": request,
                "model_id": model_id,
                "purpose": purpose,
                "decision_id": decision_id,
            }
        )
        if not self.responses:
            raise AssertionError("no fake response queued")
        text = self.responses.pop(0)
        insert_llm_call(
            self.db_path,
            LlmCallRow(
                decision_id=decision_id,
                purpose=purpose,
                model_id=model_id,
                prompt_version=request.prompt_version,
                input_tokens=self.fixed_telemetry["input_tokens"],
                output_tokens=self.fixed_telemetry["output_tokens"],
                cache_read_tokens=self.fixed_telemetry["cache_read_tokens"],
                cache_creation_tokens=self.fixed_telemetry["cache_creation_tokens"],
                latency_ms=self.fixed_telemetry["latency_ms"],
                cost_usd=self.fixed_telemetry["cost_usd"],
                error_class=None,
            ),
        )
        return text


@dataclass
class _FakePromptBuilder:
    """PromptBuilder stub returning a fixed ApiRequest. The real
    PromptBuilder is exercised end-to-end by the integration test."""

    sonnet_prompt_version: str = "sonnet_filing_analysis_v1@b5b5b5b50001"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def build_sonnet_filing_analysis(
        self, filing: FilingRow, raw_text: str, ctx: AnalyzerContext
    ) -> ApiRequest:
        self.calls.append({"filing": filing, "raw_text": raw_text, "ctx": ctx})
        return ApiRequest(
            system=[
                Block(text="block1", cache_control={"type": "ephemeral"}),
                Block(text=f"ctx={ctx.decision_id}", cache_control={"type": "ephemeral"}),
            ],
            messages=[Message(role="user", content=[Block(text=raw_text)])],
            prompt_version=self.sonnet_prompt_version,
        )

    def prompt_version(self, name: str) -> str:
        if name == "sonnet_filing_analysis_v1":
            return self.sonnet_prompt_version
        raise KeyError(name)


@dataclass
class _FakeOpusReviewer:
    """OpusReviewer stub that records review() invocations."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    return_value: Any = None

    async def review(
        self, proposal: Any, source_filing: FilingRow, raw_text: str
    ) -> Any:
        self.calls.append(
            {"proposal": proposal, "source_filing": source_filing, "raw_text": raw_text}
        )
        return self.return_value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return str(db_path)


@pytest.fixture
def sonnet_prompt_version(db: str) -> str:
    pv = "sonnet_filing_analysis_v1@b5b5b5b50001"
    insert_prompt(
        db,
        PromptRow(
            prompt_version=pv,
            name="sonnet_filing_analysis_v1",
            file_path="src/prompts/sonnet_filing_analysis_v1.txt",
            content_hash="b5b5b5b50001" + "0" * 52,
        ),
    )
    return pv


@pytest.fixture
def filing(db: str) -> FilingRow:
    fid = insert_filing(
        db,
        FilingRow(
            accession_number="0001234567-26-000456",
            cik=1234567,
            form_type="8-K",
            filed_at=dt.datetime(2026, 4, 28, 14, 30, 0),
            fetched_at=dt.datetime(2026, 4, 28, 14, 31, 0),
            raw_text_path="/var/lib/quinn/raw/0001234567-26-000456.txt",
            content_hash="aaa222",
            item_codes='["1.01"]',
            issuer_ticker="ACME",
        ),
    )
    from journal.repo import get_filing_by_id
    f = get_filing_by_id(db, fid)
    assert f is not None
    return f


@pytest.fixture
def ctx() -> AnalyzerContext:
    return AnalyzerContext(
        universe_summary="ACME ($150M cap, NYSE)",
        kill_switch_state="ok",
        open_positions_count=2,
        decision_id="placeholder-rebuilt-by-analyzer",
    )


def _valid_proposal_payload(*, conviction: int = 8) -> dict[str, Any]:
    return {
        "symbol": "ACME",
        "direction": "long",
        "size_pct_of_capital": 0.10,
        "entry_style": "market_open",
        "stop_loss_price": 9.50,
        "time_horizon_days": 14,
        "conviction": conviction,
        "thesis": (
            "ACME announced a material acquisition in 8-K Item 1.01 "
            "with concrete pricing terms; integration timeline is plausible."
        ),
        "signals": ["Item 1.01 — Material Definitive Agreement"],
        "exit_conditions": ["Exit on contradicting filing within 5 trading days"],
        "risk_factors": ["Closing conditions not yet met"],
    }


def _valid_no_trade_payload() -> dict[str, Any]:
    return {
        "decision": "no_trade",
        "thesis_or_reason": (
            "8-K Item 8.01 contains routine corporate-governance disclosure with "
            "no material catalyst; conviction below threshold."
        ),
        "signals_considered": ["Item 8.01 — routine governance"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_proposal_response_persisted(
    db: str, filing: FilingRow, ctx: AnalyzerContext, sonnet_prompt_version: str
) -> None:
    """AC-1, AC-2, AC-6: a valid TradeProposal response is parsed,
    schema-validated, and stored via ProposalStore. The decision_id is
    deterministic from (filing.id, model_id, prompt_version)."""
    from analyzer.results import ProposalEmitted
    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    payload = _valid_proposal_payload()
    raw = json.dumps(payload)
    client = _FakeAnthropicClient(db_path=db, responses=[raw])
    builder = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store = ProposalStore(db_path=db)
    opus = _FakeOpusReviewer()

    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=7,
        db_path=db,
    )
    result = await analyzer.analyze(filing, raw_text="filing body", ctx=ctx)

    assert isinstance(result, ProposalEmitted)
    # AC-2: model + purpose
    assert client.calls[0]["model_id"] == "claude-sonnet-4-6"
    assert client.calls[0]["purpose"] == "analyze"

    # AC-9: decision_id deterministic
    expected = hashlib.sha256(
        f"{filing.id}|claude-sonnet-4-6|{sonnet_prompt_version}".encode()
    ).hexdigest()[:32]
    assert client.calls[0]["decision_id"] == expected

    # AC-6: stored to proposals
    row = get_proposal_by_decision_id(db, expected)
    assert row is not None
    assert row.kind == "trade_proposal"
    assert row.symbol == "ACME"
    assert row.conviction == 8


@pytest.mark.asyncio
async def test_no_trade_response_persisted(
    db: str, filing: FilingRow, ctx: AnalyzerContext, sonnet_prompt_version: str
) -> None:
    """AC-1, AC-6: a valid NoTradeRecord is parsed and stored with kind='no_trade'."""
    from analyzer.results import NoTrade
    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    raw = json.dumps(_valid_no_trade_payload())
    client = _FakeAnthropicClient(db_path=db, responses=[raw])
    builder = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store = ProposalStore(db_path=db)
    opus = _FakeOpusReviewer()

    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=7,
        db_path=db,
    )
    result = await analyzer.analyze(filing, raw_text="filing body", ctx=ctx)
    assert isinstance(result, NoTrade)
    decision_id = client.calls[0]["decision_id"]
    row = get_proposal_by_decision_id(db, decision_id)
    assert row is not None
    assert row.kind == "no_trade"
    assert row.symbol is None
    # FR-18: no_trade NEVER triggers Opus review.
    assert opus.calls == []


@pytest.mark.asyncio
async def test_malformed_response_retried_once(
    db: str, filing: FilingRow, ctx: AnalyzerContext, sonnet_prompt_version: str
) -> None:
    """AC-4: schema/parse failure → retry once with stricter instruction.
    Second response is valid → use it."""
    from analyzer.results import ProposalEmitted
    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    bad = "this is not json at all"
    good = json.dumps(_valid_proposal_payload())
    client = _FakeAnthropicClient(db_path=db, responses=[bad, good])
    builder = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store = ProposalStore(db_path=db)
    opus = _FakeOpusReviewer()

    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=7,
        db_path=db,
    )
    result = await analyzer.analyze(filing, raw_text="filing body", ctx=ctx)
    assert isinstance(result, ProposalEmitted)
    assert len(client.calls) == 2  # original + retry
    # Retry uses an augmented user message but the same decision_id.
    assert client.calls[0]["decision_id"] == client.calls[1]["decision_id"]


@pytest.mark.asyncio
async def test_malformed_after_retry_records_analyzer_malformed(
    db: str, filing: FilingRow, ctx: AnalyzerContext, sonnet_prompt_version: str
) -> None:
    """AC-4, AC-6: two malformed responses → AnalyzerMalformed; row written
    with kind='no_trade' and reasoning_notes capturing the parse failure."""
    from analyzer.results import AnalyzerMalformed
    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    client = _FakeAnthropicClient(
        db_path=db, responses=["not json", "still {malformed"]
    )
    builder = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store = ProposalStore(db_path=db)
    opus = _FakeOpusReviewer()

    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=7,
        db_path=db,
    )
    result = await analyzer.analyze(filing, raw_text="filing body", ctx=ctx)
    assert isinstance(result, AnalyzerMalformed)
    assert len(client.calls) == 2
    # FR-30 single-source-of-truth: the failure is in the journal.
    decision_id = client.calls[0]["decision_id"]
    row = get_proposal_by_decision_id(db, decision_id)
    assert row is not None
    assert row.kind == "no_trade"
    assert row.reasoning_notes is not None
    assert "analyzer_malformed" in row.reasoning_notes
    # FR-18: malformed NEVER triggers Opus review (defense-in-depth).
    assert opus.calls == []


@pytest.mark.asyncio
async def test_high_conviction_triggers_opus_review(
    db: str, filing: FilingRow, ctx: AnalyzerContext, sonnet_prompt_version: str
) -> None:
    """AC-7: conviction >= threshold triggers Opus review; below threshold
    does NOT (FR-18)."""
    from analyzer.opus import OpusRatified
    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    # cv=8: >= threshold of 7 → Opus invoked
    raw_high = json.dumps(_valid_proposal_payload(conviction=8))
    client_h = _FakeAnthropicClient(db_path=db, responses=[raw_high])
    builder_h = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store_h = ProposalStore(db_path=db)
    opus_h = _FakeOpusReviewer()
    opus_h.return_value = OpusRatified(proposal=None, rationale="test")  # type: ignore[arg-type]

    analyzer_h = SonnetAnalyzer(
        client=client_h,
        store=store_h,
        prompt_builder=builder_h,
        opus_reviewer=opus_h,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=7,
        db_path=db,
    )
    await analyzer_h.analyze(filing, raw_text="filing body", ctx=ctx)
    assert len(opus_h.calls) == 1
    assert opus_h.calls[0]["source_filing"] == filing

    # cv=6: below threshold → Opus NOT invoked. Use a different filing so
    # the second analyze doesn't hit the idempotency-on-decision_id short
    # circuit.
    other_fid = insert_filing(
        db,
        FilingRow(
            accession_number="0001234567-26-000457",
            cik=1234567,
            form_type="8-K",
            filed_at=dt.datetime(2026, 4, 28, 14, 30, 0),
            fetched_at=dt.datetime(2026, 4, 28, 14, 31, 0),
            raw_text_path="/var/lib/quinn/raw/0001234567-26-000457.txt",
            content_hash="aaa333",
            item_codes='["1.01"]',
            issuer_ticker="ACME",
        ),
    )
    from journal.repo import get_filing_by_id
    other_filing = get_filing_by_id(db, other_fid)
    assert other_filing is not None

    raw_low = json.dumps(_valid_proposal_payload(conviction=6))
    client_l = _FakeAnthropicClient(db_path=db, responses=[raw_low])
    builder_l = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store_l = ProposalStore(db_path=db)
    opus_l = _FakeOpusReviewer()
    analyzer_l = SonnetAnalyzer(
        client=client_l,
        store=store_l,
        prompt_builder=builder_l,
        opus_reviewer=opus_l,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=7,
        db_path=db,
    )
    await analyzer_l.analyze(other_filing, raw_text="filing body", ctx=ctx)
    assert opus_l.calls == []


@pytest.mark.asyncio
async def test_opus_skipped_when_ks5_at_capacity(
    db: str, filing: FilingRow, sonnet_prompt_version: str, caplog: Any
) -> None:
    """Feature B: when KS5 is at capacity (open_positions_count >=
    ks5_max_concurrent), do NOT call Opus even on a high-conviction
    proposal. The proposal still persists; absence of a proposal_reviews
    row is the "pending_capacity" signal Feature C will key on."""
    import logging

    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    raw = json.dumps(_valid_proposal_payload(conviction=9))
    client = _FakeAnthropicClient(db_path=db, responses=[raw])
    builder = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store = ProposalStore(db_path=db)
    opus = _FakeOpusReviewer()

    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=5,
        ks5_max_concurrent=5,
        db_path=db,
    )

    ctx_full = AnalyzerContext(
        universe_summary="ACME ($150M cap, NYSE)",
        kill_switch_state="ok",
        open_positions_count=5,
        decision_id="placeholder-rebuilt-by-analyzer",
    )

    with caplog.at_level(logging.INFO):
        await analyzer.analyze(filing, raw_text="filing body", ctx=ctx_full)

    # AC: Opus NOT called.
    assert opus.calls == []
    # AC: proposal stored, no proposal_reviews row.
    decision_id = client.calls[0]["decision_id"]
    row = get_proposal_by_decision_id(db, decision_id)
    assert row is not None
    assert row.kind == "trade_proposal"
    review = get_proposal_review_by_proposal_id(db, row.id)  # type: ignore[arg-type]
    assert review is None
    # AC: structured log event emitted.
    skip_records = [
        r for r in caplog.records
        if getattr(r, "event", None) == "agent.opus_skipped_at_capacity"
    ]
    assert len(skip_records) == 1
    rec = skip_records[0]
    assert rec.filing_id == filing.id
    assert rec.proposal_id == row.id
    assert rec.conviction == 9
    assert rec.current_open_positions == 5
    assert rec.ks5_max == 5


@pytest.mark.asyncio
async def test_opus_capacity_gate_uses_broker_counter_not_journal_ctx(
    db: str, filing: FilingRow, sonnet_prompt_version: str, caplog: Any
) -> None:
    """Feature B source-of-truth alignment: the capacity gate must
    consult the broker-backed counter (the same source `SizingEngine`
    uses at sizing.py:103), NOT `ctx.open_positions_count` (which is
    journal-derived). Mismatch would cause Feature B to refuse Opus
    spend on a trade that would actually fit at the sizer's gate.

    This test sets ctx.open_positions_count=4 (slack per journal) but
    wires the broker counter to return 5 (full per broker). Expected:
    Opus is SKIPPED — broker truth wins.
    """
    import logging

    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    raw = json.dumps(_valid_proposal_payload(conviction=9))
    client = _FakeAnthropicClient(db_path=db, responses=[raw])
    builder = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store = ProposalStore(db_path=db)
    opus = _FakeOpusReviewer()

    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=5,
        ks5_max_concurrent=5,
        open_positions_counter=lambda: 5,  # broker truth — at cap
        db_path=db,
    )
    ctx_journal_says_slack = AnalyzerContext(
        universe_summary="ACME",
        kill_switch_state="ok",
        open_positions_count=4,  # journal disagrees — would say "fit"
        decision_id="placeholder",
    )
    with caplog.at_level(logging.INFO):
        await analyzer.analyze(filing, raw_text="x", ctx=ctx_journal_says_slack)

    assert opus.calls == [], "broker-truth (5/5) must override journal (4/5)"
    skip_records = [
        r for r in caplog.records
        if getattr(r, "event", None) == "agent.opus_skipped_at_capacity"
    ]
    assert len(skip_records) == 1
    # Logged count is the broker count, not the ctx count.
    assert skip_records[0].current_open_positions == 5


@pytest.mark.asyncio
async def test_opus_capacity_gate_broker_counter_overrides_ctx_at_slack(
    db: str, filing: FilingRow, sonnet_prompt_version: str
) -> None:
    """Inverse mismatch: journal says full (ctx=5), broker says slack
    (counter=4). Broker wins → Opus IS called. Pins the contract that
    Feature B does NOT block on stale journal state."""
    from analyzer.opus import OpusRatified
    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    raw = json.dumps(_valid_proposal_payload(conviction=9))
    client = _FakeAnthropicClient(db_path=db, responses=[raw])
    builder = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store = ProposalStore(db_path=db)
    opus = _FakeOpusReviewer()
    opus.return_value = OpusRatified(proposal=None, rationale="x")  # type: ignore[arg-type]

    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=5,
        ks5_max_concurrent=5,
        open_positions_counter=lambda: 4,  # broker truth — slack
        db_path=db,
    )
    ctx_journal_says_full = AnalyzerContext(
        universe_summary="ACME",
        kill_switch_state="ok",
        open_positions_count=5,  # journal disagrees — would say "skip"
        decision_id="placeholder",
    )
    await analyzer.analyze(filing, raw_text="x", ctx=ctx_journal_says_full)
    assert len(opus.calls) == 1, (
        "broker-truth (4/5 slack) must override stale journal (5/5)"
    )


@pytest.mark.asyncio
async def test_opus_capacity_gate_falls_back_to_ctx_when_counter_raises(
    db: str, filing: FilingRow, sonnet_prompt_version: str
) -> None:
    """If the broker counter raises (transient outage), the gate falls
    back to the journal-derived ctx count to stay consistent with the
    rest of the pipeline. cv9, ctx says 5/5 → still skip."""
    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    raw = json.dumps(_valid_proposal_payload(conviction=9))
    client = _FakeAnthropicClient(db_path=db, responses=[raw])
    builder = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store = ProposalStore(db_path=db)
    opus = _FakeOpusReviewer()

    def _raising_counter() -> int:
        raise RuntimeError("simulated broker outage")

    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=5,
        ks5_max_concurrent=5,
        open_positions_counter=_raising_counter,
        db_path=db,
    )
    ctx_at_cap = AnalyzerContext(
        universe_summary="ACME",
        kill_switch_state="ok",
        open_positions_count=5,
        decision_id="placeholder",
    )
    await analyzer.analyze(filing, raw_text="x", ctx=ctx_at_cap)
    # ctx fallback says 5/5 → skip.
    assert opus.calls == []


@pytest.mark.asyncio
async def test_opus_called_when_ks5_has_slack(
    db: str, filing: FilingRow, sonnet_prompt_version: str
) -> None:
    """Feature B: when KS5 has slack (open_positions_count <
    ks5_max_concurrent), Opus is called normally — existing behavior
    preserved."""
    from analyzer.opus import OpusRatified
    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    raw = json.dumps(_valid_proposal_payload(conviction=9))
    client = _FakeAnthropicClient(db_path=db, responses=[raw])
    builder = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store = ProposalStore(db_path=db)
    opus = _FakeOpusReviewer()
    opus.return_value = OpusRatified(proposal=None, rationale="x")  # type: ignore[arg-type]

    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=5,
        ks5_max_concurrent=5,
        db_path=db,
    )

    ctx_slack = AnalyzerContext(
        universe_summary="ACME ($150M cap, NYSE)",
        kill_switch_state="ok",
        open_positions_count=4,
        decision_id="placeholder-rebuilt-by-analyzer",
    )
    await analyzer.analyze(filing, raw_text="filing body", ctx=ctx_slack)

    assert len(opus.calls) == 1


@pytest.mark.asyncio
async def test_system_fields_injected(
    db: str, filing: FilingRow, ctx: AnalyzerContext, sonnet_prompt_version: str
) -> None:
    """AC-5: persisted raw_response carries system-injected `prompt_version`
    and `source_filings` — populated by ProposalStore from analyzer wiring,
    NOT by the LLM."""
    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    # The LLM response does NOT include prompt_version or source_filings;
    # the store must inject them.
    payload = _valid_proposal_payload()
    assert "prompt_version" not in payload
    assert "source_filings" not in payload
    raw = json.dumps(payload)
    client = _FakeAnthropicClient(db_path=db, responses=[raw])
    builder = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store = ProposalStore(db_path=db)
    opus = _FakeOpusReviewer()
    opus.return_value = None  # not exercised here

    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=7,
        db_path=db,
    )
    await analyzer.analyze(filing, raw_text="filing body", ctx=ctx)

    decision_id = client.calls[0]["decision_id"]
    row = get_proposal_by_decision_id(db, decision_id)
    assert row is not None
    enriched = json.loads(row.raw_response)
    assert enriched["prompt_version"] == sonnet_prompt_version
    assert enriched["source_filings"] == [
        {
            "accession_number": filing.accession_number,
            "filing_type": filing.form_type,
            "item_codes": ["1.01"],
        }
    ]


def test_decision_id_deterministic(sonnet_prompt_version: str) -> None:
    """AC-9: same (filing.id, model_id, prompt_version) → same decision_id."""
    from analyzer.sonnet import compute_decision_id

    a = compute_decision_id(
        filing_id=42,
        model_id="claude-sonnet-4-6",
        prompt_version=sonnet_prompt_version,
    )
    b = compute_decision_id(
        filing_id=42,
        model_id="claude-sonnet-4-6",
        prompt_version=sonnet_prompt_version,
    )
    assert a == b
    assert len(a) == 32
    # Different inputs → different ids.
    c = compute_decision_id(
        filing_id=43,  # different filing
        model_id="claude-sonnet-4-6",
        prompt_version=sonnet_prompt_version,
    )
    assert a != c
    d = compute_decision_id(
        filing_id=42,
        model_id="claude-opus-4-7",  # different model
        prompt_version=sonnet_prompt_version,
    )
    assert a != d


def test_augment_request_for_retry_preserves_system_blocks() -> None:
    """S5.4 review T-1 carry-forward: byte-stability of the cacheable
    prefix across the retry. The augmented request MUST have identical
    system-block bytes to the original — otherwise NFR-12 cache-hit on
    the retry collapses and the prompt-version hash drifts.

    Mirrors S5.4's `_augment_request_for_retry` byte-stability invariant.
    """
    from analyzer.sonnet import augment_request_for_retry

    original = ApiRequest(
        system=[
            Block(text="block1 system", cache_control={"type": "ephemeral"}),
            Block(text="block2 context", cache_control={"type": "ephemeral"}),
        ],
        messages=[
            Message(role="user", content=[Block(text="block3 filing payload")])
        ],
        prompt_version="sonnet_filing_analysis_v1@deadbeef0001",
    )
    retried = augment_request_for_retry(original, prior_error="Expecting value")

    # System blocks (cacheable prefix) byte-stable.
    assert retried.system == original.system
    # prompt_version unchanged.
    assert retried.prompt_version == original.prompt_version
    # User message gets the correction appended to its LAST block.
    assert retried.messages[0].content[-1].text.startswith("block3 filing payload")
    assert "[SYSTEM RETRY]" in retried.messages[0].content[-1].text
    # Original is not mutated (frozen dataclasses, but pin behavior).
    assert original.messages[0].content[0].text == "block3 filing payload"


def test_parse_strips_json_fences() -> None:
    """AC-3: the parser strips ```json fences from the response."""
    from analyzer.parse import extract_json_object

    fenced = "```json\n" + json.dumps({"a": 1, "b": "x"}) + "\n```"
    assert extract_json_object(fenced) == {"a": 1, "b": "x"}
    plain_fenced = "```\n" + json.dumps({"a": 1}) + "\n```"
    assert extract_json_object(plain_fenced) == {"a": 1}
    bare = json.dumps({"a": 1})
    assert extract_json_object(bare) == {"a": 1}
    # Whitespace tolerated.
    padded = "   " + json.dumps({"a": 1}) + "  \n"
    assert extract_json_object(padded) == {"a": 1}


def test_parse_rejects_non_object() -> None:
    """AC-3: lists / scalars / multi-objects are rejected."""
    from analyzer.parse import ResponseParseError, extract_json_object

    with pytest.raises(ResponseParseError):
        extract_json_object("[1, 2, 3]")
    with pytest.raises(ResponseParseError):
        extract_json_object('"a string"')
    with pytest.raises(ResponseParseError):
        extract_json_object("not json at all")


@pytest.mark.asyncio
async def test_p95_latency_budget_under_60s_on_stubbed_sonnet(
    db: str, filing: FilingRow, ctx: AnalyzerContext, sonnet_prompt_version: str
) -> None:
    """AC-8: NFR-1 sanity check on the analyzer's *non-Sonnet* overhead.

    The 60s NFR-1 budget is dominated by the actual Sonnet API call; the
    analyzer's parse + validate + persist + opus-gate work must be a small
    fraction. With a synchronous in-process stub, run 50 analyses and
    assert p95 wall-clock is comfortably under 60s. This is a smoke test
    for the framework cost, not a Sonnet latency check.
    """
    from analyzer.results import ProposalEmitted
    from analyzer.sonnet import SonnetAnalyzer
    from proposal.store import ProposalStore

    n = 50
    samples = [json.dumps(_valid_proposal_payload(conviction=6)) for _ in range(n)]
    client = _FakeAnthropicClient(db_path=db, responses=samples)
    builder = _FakePromptBuilder(sonnet_prompt_version=sonnet_prompt_version)
    store = ProposalStore(db_path=db)
    opus = _FakeOpusReviewer()
    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=7,  # cv 6 stays under → no opus call
        db_path=db,
    )
    # Use distinct filings so each call gets a unique decision_id and
    # the proposal-store idempotency short-circuit doesn't shadow latency.
    filings: list[FilingRow] = [filing]
    for i in range(1, n):
        fid = insert_filing(
            db,
            FilingRow(
                accession_number=f"0001234567-26-{i:06d}",
                cik=1234567,
                form_type="8-K",
                filed_at=dt.datetime(2026, 4, 28, 14, 30, 0),
                fetched_at=dt.datetime(2026, 4, 28, 14, 31, 0),
                raw_text_path=f"/tmp/{i}.txt",
                content_hash=f"h-{i}",
                item_codes='["1.01"]',
                issuer_ticker="ACME",
            ),
        )
        from journal.repo import get_filing_by_id
        f = get_filing_by_id(db, fid)
        assert f is not None
        filings.append(f)

    durations: list[float] = []
    for f in filings:
        start = time.perf_counter()
        result = await analyzer.analyze(f, raw_text="x", ctx=ctx)
        durations.append(time.perf_counter() - start)
        assert isinstance(result, ProposalEmitted)
    durations.sort()
    p95 = durations[int(0.95 * len(durations))]
    # Generous: real Sonnet calls are 5–30s; the stub here is in-process
    # so we expect single-digit ms per call. Bound at 1s for p95 to catch
    # accidental sleeps / I/O without flaking on slow CI.
    assert p95 < 1.0, f"p95 {p95:.3f}s exceeded 1s budget on stubbed Sonnet"


@pytest.mark.asyncio
async def test_e2e_with_real_prompt_builder_and_store(
    db: str, filing: FilingRow, ctx: AnalyzerContext
) -> None:
    """AC-1..AC-7 integration: real PromptBuilder (composed prompt-version),
    real ProposalStore (FK-enforced row write), real OpusReviewer (with a
    fake AnthropicClient as the only stub).

    Also exercises the S5.6 carry-forward sequencing: composed
    prompt_version must be registered in the `prompts` table BEFORE the
    ProposalStore write fires the FK. This test does the registration
    explicitly, mirroring what S5.6 wiring will own.
    """
    from analyzer.opus import OpusReviewer
    from analyzer.results import ProposalEmitted
    from analyzer.sonnet import SonnetAnalyzer, compute_decision_id
    from journal.models import PromptRow
    from prompts.loader import PromptBuilder
    from prompts.lock import _entries  # type: ignore[attr-defined]
    from proposal.store import ProposalStore

    builder = PromptBuilder(prompt_dir=Path("src/prompts"))
    sonnet_pv = builder.prompt_version("sonnet_filing_analysis_v1")
    opus_pv = builder.prompt_version("opus_proposal_review_v1")

    # Register composed prompt-versions BEFORE the call (S5.6 carry-forward).
    # This is what S5.6 wiring will own; this test pins the contract.
    for pv, name in ((sonnet_pv, "sonnet_filing_analysis_v1"),
                     (opus_pv, "opus_proposal_review_v1")):
        # Verify the lock helper can see the underlying file (exercises
        # the lock helper's import path as a side effect).
        _ = _entries(Path("src/prompts"))  # smoke
        insert_prompt(
            db,
            PromptRow(
                prompt_version=pv,
                name=name,
                file_path=f"src/prompts/{name}.txt",
                content_hash="x" * 64,  # synthetic hash; real one comes from S5.1's helper
            ),
        )

    high_conviction_payload = _valid_proposal_payload(conviction=8)
    raw_sonnet = json.dumps(high_conviction_payload)
    raw_opus = json.dumps(
        {
            "decision": "ratify",
            "rationale": (
                "The thesis is supported by the cited filing language; sizing "
                "and stops are reasonable for the conviction tier."
            ),
        }
    )
    client = _FakeAnthropicClient(db_path=db, responses=[raw_sonnet, raw_opus])
    store = ProposalStore(db_path=db)
    opus = OpusReviewer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db,
    )

    analyzer = SonnetAnalyzer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_reviewer=opus,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=7,
        db_path=db,
    )
    result = await analyzer.analyze(filing, raw_text="ACME 8-K body...", ctx=ctx)

    assert isinstance(result, ProposalEmitted)
    decision_id = compute_decision_id(
        filing_id=filing.id,
        model_id="claude-sonnet-4-6",
        prompt_version=sonnet_pv,
    )

    # Sonnet row landed.
    proposal = get_proposal_by_decision_id(db, decision_id)
    assert proposal is not None
    assert proposal.kind == "trade_proposal"
    assert proposal.symbol == "ACME"
    assert proposal.conviction == 8
    # AC-5: system fields injected into raw_response.
    enriched = json.loads(proposal.raw_response)
    assert enriched["prompt_version"] == sonnet_pv
    assert len(enriched["source_filings"]) == 1

    # Opus row landed (cv 8 ≥ 7 threshold).
    review = get_proposal_review_by_proposal_id(db, proposal.id)
    assert review is not None
    assert review.decision == "ratify"
    assert review.model_id == "claude-opus-4-7"
    assert review.prompt_version == opus_pv

    # Sonnet + Opus both went through the AnthropicClient; only one
    # `purpose="analyze"` row and one `purpose="review"` row exist.
    from journal.repo import get_llm_calls_by_decision_id
    calls = get_llm_calls_by_decision_id(db, decision_id)
    purposes = sorted(c.purpose for c in calls)
    assert purposes == ["analyze", "review"]
