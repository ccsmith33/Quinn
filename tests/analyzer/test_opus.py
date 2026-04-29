"""S5.4 — Opus high-conviction proposal reviewer tests.

Architecture references: §2.3 (LLM Analyzer Opus path), §3.4 (Opus review
schema), FR-17, FR-18, FR-28, ADR-005.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from journal.migrate import apply_migrations
from journal.models import (
    FilingRow,
    LlmCallRow,
    PromptRow,
    ProposalReviewRow,
    ProposalRow,
)
from journal.repo import insert_filing, insert_prompt, insert_proposal

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class _FakeAnthropicClient:
    """Stub AnthropicClient: returns a queued response and writes the
    matching `llm_calls` row that S5.4 reads back for telemetry."""

    db_path: str
    responses: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    fixed_telemetry: dict[str, Any] = field(
        default_factory=lambda: {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_tokens": 800,
            "cache_creation_tokens": 200,
            "latency_ms": 1234,
            "cost_usd": 0.0123,
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
        # Mirror what the real AnthropicClient does: write a `llm_calls` row.
        from journal.repo import insert_llm_call

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
class _FakeProposalStore:
    """Stub ProposalStore that only implements `store_review`."""

    reviews: list[dict[str, Any]] = field(default_factory=list)

    def store_review(
        self,
        proposal_id: int,
        review: Any,
        *,
        model_id: str,
        prompt_version: str,
        raw_response: str,
        telemetry: Any,
    ) -> None:
        self.reviews.append(
            {
                "proposal_id": proposal_id,
                "review": review,
                "model_id": model_id,
                "prompt_version": prompt_version,
                "raw_response": raw_response,
                "telemetry": telemetry,
            }
        )


@dataclass
class _FakePromptBuilder:
    """Stub PromptBuilder.build_opus_proposal_review."""

    prompt_version: str = "opus_proposal_review_v1@cafef00d0001"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def build_opus_proposal_review(
        self, proposal: ProposalRow, source_text_summary: str
    ) -> Any:
        self.calls.append({"proposal": proposal, "summary": source_text_summary})
        from prompts.loader import ApiRequest, Block, Message

        return ApiRequest(
            system=[Block(text="block1", cache_control={"type": "ephemeral"})],
            messages=[Message(role="user", content=[Block(text="block3")])],
            prompt_version=self.prompt_version,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return str(db_path)


@pytest.fixture
def opus_prompt_version(db: str) -> str:
    pv = "opus_proposal_review_v1@cafef00d0001"
    insert_prompt(
        db,
        PromptRow(
            prompt_version=pv,
            name="opus_proposal_review_v1",
            file_path="src/prompts/opus_proposal_review_v1.txt",
            content_hash="cafef00d0001" + "0" * 52,
        ),
    )
    return pv


@pytest.fixture
def proposal(db: str, opus_prompt_version: str) -> ProposalRow:
    """Insert a high-conviction proposal and return the row with `id` set."""
    filing_id = insert_filing(
        db,
        FilingRow(
            accession_number="0001234567-26-000099",
            cik=1234567,
            form_type="8-K",
            filed_at=dt.datetime(2026, 4, 28, 14, 30, 0),
            fetched_at=dt.datetime(2026, 4, 28, 14, 31, 0),
            raw_text_path="/var/lib/quinn/raw/0001234567-26-000099.txt",
            content_hash="abc123",
            item_codes='["1.01"]',
            issuer_ticker="ACME",
        ),
    )
    proposal_row = ProposalRow(
        filing_id=filing_id,
        decision_id="dec-opus-001",
        model_id="claude-sonnet-4-6",
        prompt_version=opus_prompt_version,  # share the same prompt version
        raw_response='{"symbol": "ACME", "conviction": 8}',
        kind="trade_proposal",
        symbol="ACME",
        direction="long",
        size_pct_requested=0.10,
        conviction=8,
        thesis="Strong material event.",
        input_tokens=100,
        output_tokens=50,
        latency_ms=2000,
        cost_usd=0.005,
    )
    pid = insert_proposal(db, proposal_row)
    return proposal_row.model_copy(update={"id": pid})


@pytest.fixture
def filing(db: str, proposal: ProposalRow) -> FilingRow:
    from journal.repo import get_filing_by_id
    f = get_filing_by_id(db, proposal.filing_id)
    assert f is not None
    return f


def _ratify_response() -> str:
    return json.dumps(
        {
            "decision": "ratify",
            "rationale": (
                "The thesis is supported by the filing language; size and stops "
                "are reasonable for the conviction tier; no contradicting items."
            ),
        }
    )


def _modify_response(**mods: Any) -> str:
    return json.dumps(
        {
            "decision": "modify",
            "rationale": (
                "The thesis is largely sound but the proposed stop is too wide "
                "for the recent volatility; tightening is warranted to bound risk."
            ),
            "modifications": mods,
        }
    )


def _reject_response() -> str:
    return json.dumps(
        {
            "decision": "reject",
            "rationale": (
                "Filing language is non-material relative to the proposed conviction; "
                "the analyst over-weighted Item 1.01 housekeeping disclosure as a catalyst."
            ),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ratify_passes_through_unchanged(
    db: str, proposal: ProposalRow, filing: FilingRow
) -> None:
    """AC-1, AC-4: ratify decision returns OpusRatified with the original
    proposal carried through; review row is persisted."""
    from analyzer.opus import OpusRatified, OpusReviewer

    client = _FakeAnthropicClient(db_path=db, responses=[_ratify_response()])
    store = _FakeProposalStore()
    builder = _FakePromptBuilder()

    reviewer = OpusReviewer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db,
    )

    result = await reviewer.review(proposal, filing, raw_text="filing body")

    assert isinstance(result, OpusRatified)
    assert result.proposal == proposal
    # Single Opus call, with proper plumbing.
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model_id"] == "claude-opus-4-7"
    assert call["purpose"] == "review"
    assert call["decision_id"] == proposal.decision_id
    # Review row written.
    assert len(store.reviews) == 1
    review_call = store.reviews[0]
    assert review_call["proposal_id"] == proposal.id
    assert review_call["model_id"] == "claude-opus-4-7"
    assert review_call["prompt_version"] == builder.prompt_version
    assert review_call["raw_response"] == _ratify_response()
    # Telemetry round-trips from llm_calls.
    tel = review_call["telemetry"]
    assert tel.input_tokens == 1000
    assert tel.output_tokens == 500
    assert tel.cache_read_tokens == 800
    assert tel.cache_creation_tokens == 200
    assert tel.latency_ms == 1234
    assert tel.cost_usd == pytest.approx(0.0123)


@pytest.mark.asyncio
async def test_modify_overlays_applied(
    db: str, proposal: ProposalRow, filing: FilingRow
) -> None:
    """AC-1, AC-5: modification adjusts size; the working_proposal carries
    the overlay; the original proposal row is unchanged (NFR-16)."""
    from analyzer.opus import OpusModified, OpusReviewer

    client = _FakeAnthropicClient(
        db_path=db,
        responses=[_modify_response(size_pct_of_capital=0.05, stop_loss_price=42.50)],
    )
    store = _FakeProposalStore()
    builder = _FakePromptBuilder()
    reviewer = OpusReviewer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db,
    )

    result = await reviewer.review(proposal, filing, raw_text="filing body")

    assert isinstance(result, OpusModified)
    # Working copy carries the overlay; original row unchanged.
    assert result.working_proposal.size_pct_requested == pytest.approx(0.05)
    assert result.proposal.size_pct_requested == pytest.approx(0.10)
    # Modifications dict is preserved in full (incl. fields not on ProposalRow).
    assert result.modifications == {"size_pct_of_capital": 0.05, "stop_loss_price": 42.50}
    assert len(store.reviews) == 1


@pytest.mark.asyncio
async def test_reject_halts_proposal(
    db: str, proposal: ProposalRow, filing: FilingRow
) -> None:
    """AC-1, AC-4: reject decision returns OpusRejected; review row persisted."""
    from analyzer.opus import OpusRejected, OpusReviewer

    client = _FakeAnthropicClient(db_path=db, responses=[_reject_response()])
    store = _FakeProposalStore()
    builder = _FakePromptBuilder()
    reviewer = OpusReviewer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db,
    )

    result = await reviewer.review(proposal, filing, raw_text="filing body")
    assert isinstance(result, OpusRejected)
    assert "non-material" in result.rationale
    assert len(store.reviews) == 1
    assert store.reviews[0]["raw_response"] == _reject_response()


@pytest.mark.asyncio
async def test_malformed_retried_once_then_records_malformed(
    db: str, proposal: ProposalRow, filing: FilingRow
) -> None:
    """AC-3: malformed response → retry once with stricter instruction;
    if still malformed, return OpusMalformed and persist a review row."""
    from analyzer.opus import OpusMalformed, OpusReviewer

    bad1 = "this is not json at all"
    bad2 = "{still not a valid review document}"
    client = _FakeAnthropicClient(db_path=db, responses=[bad1, bad2])
    store = _FakeProposalStore()
    builder = _FakePromptBuilder()
    reviewer = OpusReviewer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db,
    )

    result = await reviewer.review(proposal, filing, raw_text="filing body")
    assert isinstance(result, OpusMalformed)
    assert result.raw_response == bad2
    # Anthropic was called exactly twice: original + retry.
    assert len(client.calls) == 2
    # Review row persisted (FR-28 — every review including malformed).
    assert len(store.reviews) == 1
    assert store.reviews[0]["raw_response"] == bad2


@pytest.mark.asyncio
async def test_malformed_then_valid_on_retry(
    db: str, proposal: ProposalRow, filing: FilingRow
) -> None:
    """Retry-on-malformed: if the second response is valid, use it."""
    from analyzer.opus import OpusRatified, OpusReviewer

    client = _FakeAnthropicClient(
        db_path=db, responses=["not json", _ratify_response()]
    )
    store = _FakeProposalStore()
    builder = _FakePromptBuilder()
    reviewer = OpusReviewer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db,
    )

    result = await reviewer.review(proposal, filing, raw_text="filing body")
    assert isinstance(result, OpusRatified)
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_modify_invalid_overlay_treated_as_reject(
    db: str, proposal: ProposalRow, filing: FilingRow
) -> None:
    """AC-5: modification with size_pct over KS-4 cap → treated as reject."""
    from analyzer.opus import OpusRejected, OpusReviewer

    client = _FakeAnthropicClient(
        db_path=db,
        responses=[_modify_response(size_pct_of_capital=0.99)],  # > KS-4 cap
    )
    store = _FakeProposalStore()
    builder = _FakePromptBuilder()
    reviewer = OpusReviewer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db,
    )

    result = await reviewer.review(proposal, filing, raw_text="filing body")
    assert isinstance(result, OpusRejected)
    assert "invalid overlay" in result.rationale
    # Review row still persisted — the journal records the Opus decision
    # in full, including overlay-failure rationale (FR-28).
    assert len(store.reviews) == 1


@pytest.mark.asyncio
async def test_review_row_persisted_for_all_paths(
    db: str, proposal: ProposalRow, filing: FilingRow
) -> None:
    """AC-4 + FR-28: every Opus path (ratify, modify, reject, malformed)
    writes one review row."""
    from analyzer.opus import (
        OpusMalformed,
        OpusModified,
        OpusRatified,
        OpusRejected,
        OpusReviewer,
    )

    paths = [
        (_ratify_response(), OpusRatified),
        (_modify_response(size_pct_of_capital=0.05), OpusModified),
        (_reject_response(), OpusRejected),
    ]
    for raw, expected_cls in paths:
        client = _FakeAnthropicClient(db_path=db, responses=[raw])
        store = _FakeProposalStore()
        builder = _FakePromptBuilder()
        reviewer = OpusReviewer(
            client=client,
            store=store,
            prompt_builder=builder,
            opus_model_id="claude-opus-4-7",
            ks4_pct_cap=0.20,
            db_path=db,
        )
        result = await reviewer.review(proposal, filing, raw_text="x")
        assert isinstance(result, expected_cls)
        assert len(store.reviews) == 1

    # Malformed path also writes a row.
    client = _FakeAnthropicClient(db_path=db, responses=["bad", "still bad"])
    store = _FakeProposalStore()
    builder = _FakePromptBuilder()
    reviewer = OpusReviewer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db,
    )
    result = await reviewer.review(proposal, filing, raw_text="x")
    assert isinstance(result, OpusMalformed)
    assert len(store.reviews) == 1


@pytest.mark.asyncio
async def test_decision_id_carried_through(
    db: str, proposal: ProposalRow, filing: FilingRow
) -> None:
    """AC-2: Opus call uses the same decision_id as the original proposal —
    so the audit trail joins proposal ↔ review by decision_id."""
    from analyzer.opus import OpusReviewer

    client = _FakeAnthropicClient(db_path=db, responses=[_ratify_response()])
    store = _FakeProposalStore()
    builder = _FakePromptBuilder()
    reviewer = OpusReviewer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db,
    )
    await reviewer.review(proposal, filing, raw_text="x")
    assert client.calls[0]["decision_id"] == proposal.decision_id


@pytest.mark.asyncio
async def test_persisted_review_row_in_journal(
    db: str, proposal: ProposalRow, filing: FilingRow
) -> None:
    """Integration: when paired with a real-style ProposalStore that writes
    `proposal_reviews`, the Opus decision lands as a row in the journal.

    Uses an inline thin store to exercise the journal repo end-to-end
    (S5.5 will provide a richer ProposalStore; this test pins the contract
    `OpusReviewer → store_review → proposal_reviews`)."""
    from analyzer.opus import OpusModified, OpusReviewer
    from analyzer.telemetry import CallTelemetry
    from journal.repo import get_proposal_review_by_proposal_id

    @dataclass
    class _ThinStore:
        db_path: str

        def store_review(
            self,
            proposal_id: int,
            review: Any,
            *,
            model_id: str,
            prompt_version: str,
            raw_response: str,
            telemetry: CallTelemetry,
        ) -> None:
            from journal.repo import insert_proposal_review

            decision = (
                "ratify" if isinstance(review, OpusRatifiedT)
                else "modify" if isinstance(review, OpusModifiedT)
                else "reject" if isinstance(review, OpusRejectedT)
                else "malformed"
            )
            mods_json = (
                json.dumps(review.modifications)
                if isinstance(review, OpusModifiedT)
                else None
            )
            insert_proposal_review(
                self.db_path,
                ProposalReviewRow(
                    proposal_id=proposal_id,
                    model_id=model_id,
                    prompt_version=prompt_version,
                    decision=decision,
                    raw_response=raw_response,
                    rationale=getattr(review, "rationale", "") or "",
                    modifications_json=mods_json,
                    input_tokens=telemetry.input_tokens,
                    output_tokens=telemetry.output_tokens,
                    cache_read_tokens=telemetry.cache_read_tokens,
                    cache_creation_tokens=telemetry.cache_creation_tokens,
                    latency_ms=telemetry.latency_ms,
                    cost_usd=telemetry.cost_usd,
                ),
            )

    from analyzer.opus import (
        OpusModified as OpusModifiedT,
    )
    from analyzer.opus import (
        OpusRatified as OpusRatifiedT,
    )
    from analyzer.opus import (
        OpusRejected as OpusRejectedT,
    )

    client = _FakeAnthropicClient(
        db_path=db,
        responses=[_modify_response(size_pct_of_capital=0.07)],
    )
    store = _ThinStore(db_path=db)
    builder = _FakePromptBuilder()
    reviewer = OpusReviewer(
        client=client,
        store=store,
        prompt_builder=builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db,
    )

    result = await reviewer.review(proposal, filing, raw_text="filing body")
    assert isinstance(result, OpusModified)

    row = get_proposal_review_by_proposal_id(db, proposal.id)
    assert row is not None
    assert row.proposal_id == proposal.id
    assert row.model_id == "claude-opus-4-7"
    assert row.decision == "modify"
    assert row.modifications_json is not None
    mods = json.loads(row.modifications_json)
    assert mods["size_pct_of_capital"] == pytest.approx(0.07)
    assert row.input_tokens == 1000
    assert row.cost_usd == pytest.approx(0.0123)
