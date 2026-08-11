"""Cost-ledger fix (2026-08-04 blind spot) — thesis-review llm_calls journaling.

Thesis reviews historically journaled under `purpose="review"` — the same
tag as Opus proposal reviews — so the event-review layer was invisible in
every purpose-keyed ledger view (`SELECT DISTINCT purpose FROM llm_calls`
showed no thesis layer at all). These tests pin the corrected contract:

- one `llm_calls` row per thesis-review call, `purpose='thesis_review'`
- real token counts + `cost_usd` from the pricing table
- `prompt_version` FK resolves against the `prompts` registry (the
  2026-07-01 incident class: unregistered pv -> FK failure -> lost call)
- `thesis_reviews` row content unchanged (its token columns remain the
  per-review record, sourced from the same llm_calls row — no drift)
- bounded-retry semantics still produce exactly one row per call
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import SecretStr

import prompts.loader as loader_mod
from analyzer.anthropic_client import AnthropicClient
from analyzer.pricing import compute_cost
from analyzer.thesis_review import (
    ThesisHold,
    ThesisReviewContext,
    ThesisReviewer,
)
from app.composition import register_composed_prompt_versions
from journal.migrate import apply_migrations
from journal.models import (
    ExecutionRow,
    FilingRow,
    PromptRow,
    ProposalRow,
    ThesisReviewScheduleRow,
)
from journal.repo import (
    connect,
    get_llm_calls_by_decision_id,
    get_prompt_by_version,
    insert_execution,
    insert_filing,
    insert_prompt,
    insert_proposal,
    insert_thesis_review_schedule,
)
from prompts.loader import ACTIVE_OPUS_THESIS_REVIEW_PROMPT, PromptBuilder

_OPUS_MODEL = "claude-opus-4-6"
_PROMPTS_DIR = Path(loader_mod.__file__).parent

_HOLD_RESPONSE = json.dumps(
    {
        "decision": "hold",
        "rationale": (
            "Thesis intact: integration milestones on schedule and no "
            "contradicting filings since entry; hold to horizon."
        ),
    }
)


# ---------------------------------------------------------------------------
# Fake SDK transport (real AnthropicClient wraps it, so the production
# telemetry write path is exercised end-to-end)
# ---------------------------------------------------------------------------


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int
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


def _success(input_tokens: int = 2000, output_tokens: int = 300) -> _FakeResponse:
    return _FakeResponse(
        content=[_FakeTextBlock(text=_HOLD_RESPONSE)],
        usage=_FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _rate_limit_error() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        429,
        request=request,
        json={"type": "error", "error": {"type": "rate_limit_error", "message": "rl"}},
    )
    return anthropic.RateLimitError(
        message="rate limited",
        response=response,
        body={"type": "error", "error": {"type": "rate_limit_error"}},
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
def builder(db: str) -> PromptBuilder:
    """Real PromptBuilder over the shipped prompt files, with every composed
    pv registered the way `compose_agent` does at boot — the FK seam under
    test is the production one, not a hand-inserted stand-in."""
    b = PromptBuilder(_PROMPTS_DIR)
    register_composed_prompt_versions(b, db)
    return b


@pytest.fixture
def review_ctx(db: str) -> ThesisReviewContext:
    """Filing -> proposal -> execution(accepted) -> due schedule chain so
    `insert_thesis_review`'s own FKs resolve."""
    now = dt.datetime.now(dt.UTC)
    sonnet_pv = "sonnet_filing_analysis_v2@deadbeef0001"
    insert_prompt(
        db,
        PromptRow(
            prompt_version=sonnet_pv,
            name="sonnet_filing_analysis_v2",
            file_path="src/prompts/sonnet_filing_analysis_v2.txt",
            content_hash="deadbeef0001" + "0" * 52,
        ),
    )
    fid = insert_filing(
        db,
        FilingRow(
            accession_number="0001234567-26-000001",
            cik=320193,
            form_type="8-K",
            filed_at=now - dt.timedelta(days=9),
            fetched_at=now - dt.timedelta(days=9),
            raw_text_path="/tmp/f.txt",
            content_hash="h" * 64,
            item_codes='["1.01"]',
            issuer_ticker="ACME",
        ),
    )
    proposal = ProposalRow(
        filing_id=fid,
        decision_id="orig-decision-1",
        model_id="claude-sonnet-4-6",
        prompt_version=sonnet_pv,
        raw_response="{}",
        kind="trade_proposal",
        symbol="ACME",
        direction="long",
        size_pct_requested=0.10,
        conviction=8,
        thesis="Material definitive agreement with concrete pricing terms.",
        input_tokens=1500,
        output_tokens=800,
        latency_ms=2000,
        cost_usd=0.02,
    )
    pid = insert_proposal(db, proposal)
    eid = insert_execution(
        db,
        ExecutionRow(
            proposal_id=pid,
            decision="accepted",
            realized_size_pct=0.10,
            realized_dollar_size=10_000.0,
            submitted_orders_json="[]",
        ),
    )
    sid = insert_thesis_review_schedule(
        db,
        ThesisReviewScheduleRow(
            execution_id=eid,
            due_at=now,
            scheduled_reason="entry",
        ),
    )
    return ThesisReviewContext(
        proposal=proposal.model_copy(update={"id": pid}),
        execution_id=eid,
        schedule_id=sid,
        days_held=9,
        current_price=105.0,
        pct_change_since_entry=0.05,
        realized_fill_price=100.0,
        realized_dollar_size=10_000.0,
        time_horizon_days=7,
        stop_loss_price=90.0,
        take_profit_price=150.0,
        filings_since_entry_summary="No filings since entry.",
    )


def _make_reviewer(
    db: str, builder: PromptBuilder, fake: _FakeAsyncAnthropic
) -> ThesisReviewer:
    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id="claude-sonnet-4-6",
        db_path=db,
        _client=fake,
    )
    return ThesisReviewer(
        client=client,
        prompt_builder=builder,
        opus_model_id=_OPUS_MODEL,
        db_path=db,
        max_output_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thesis_review_writes_one_llm_calls_row_purpose_thesis_review(
    db: str, builder: PromptBuilder, review_ctx: ThesisReviewContext
) -> None:
    """One thesis review -> exactly one llm_calls row: purpose
    'thesis_review', real token counts, pricing-table cost, resolving
    model/prompt_version — and NOT the proposal-review 'review' tag the
    layer used to hide under."""
    fake = _FakeAsyncAnthropic()
    fake.messages.queue(_success(input_tokens=2000, output_tokens=300))
    reviewer = _make_reviewer(db, builder, fake)

    result = await reviewer.review(review_ctx)
    assert isinstance(result, ThesisHold)

    decision_id = f"thesis-{review_ctx.schedule_id:08d}"
    rows = get_llm_calls_by_decision_id(db, decision_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.purpose == "thesis_review"
    assert row.model_id == _OPUS_MODEL
    assert row.input_tokens == 2000
    assert row.output_tokens == 300
    assert row.cost_usd == pytest.approx(
        compute_cost(
            _OPUS_MODEL,
            input_tokens=2000,
            output_tokens=300,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )
    )
    assert row.cost_usd > 0
    assert row.error_class is None

    # prompt_version FK resolves against the boot-time registration
    # (2026-07-01 incident class guard).
    expected_pv = builder.prompt_version(ACTIVE_OPUS_THESIS_REVIEW_PROMPT)
    assert row.prompt_version == expected_pv
    assert get_prompt_by_version(db, expected_pv) is not None
    with connect(db) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.asyncio
async def test_thesis_reviews_row_content_unchanged(
    db: str, builder: PromptBuilder, review_ctx: ThesisReviewContext
) -> None:
    """The thesis_reviews outcome row keeps its own per-review token
    record and stays consistent with the (single) llm_calls row — the fix
    must not double-write or drift the two."""
    fake = _FakeAsyncAnthropic()
    fake.messages.queue(_success(input_tokens=2000, output_tokens=300))
    reviewer = _make_reviewer(db, builder, fake)

    await reviewer.review(review_ctx)

    with connect(db) as conn:
        trs = conn.execute(
            "SELECT * FROM thesis_reviews WHERE schedule_id = ?",
            (review_ctx.schedule_id,),
        ).fetchall()
    assert len(trs) == 1
    tr = dict(trs[0])
    assert tr["decision"] == "hold"
    assert tr["model_id"] == _OPUS_MODEL
    assert tr["prompt_version"] == builder.prompt_version(
        ACTIVE_OPUS_THESIS_REVIEW_PROMPT
    )
    assert tr["raw_response"] == _HOLD_RESPONSE

    llm = get_llm_calls_by_decision_id(db, f"thesis-{review_ctx.schedule_id:08d}")[0]
    assert tr["input_tokens"] == llm.input_tokens == 2000
    assert tr["output_tokens"] == llm.output_tokens == 300
    assert tr["cost_usd"] == pytest.approx(llm.cost_usd)
    assert tr["latency_ms"] == llm.latency_ms


@pytest.mark.asyncio
async def test_thesis_review_retry_then_success_writes_exactly_one_row(
    db: str,
    builder: PromptBuilder,
    review_ctx: ThesisReviewContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded-retry semantics: a transient 429 followed by success still
    journals exactly one llm_calls row (the terminal outcome), so the
    ledger stays idempotent under the client's retry loop."""
    import analyzer.anthropic_client as ac_mod

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(ac_mod, "_async_sleep", _no_sleep)

    fake = _FakeAsyncAnthropic()
    fake.messages.queue(_rate_limit_error())
    fake.messages.queue(_success())
    reviewer = _make_reviewer(db, builder, fake)

    result = await reviewer.review(review_ctx)
    assert isinstance(result, ThesisHold)

    rows = get_llm_calls_by_decision_id(db, f"thesis-{review_ctx.schedule_id:08d}")
    assert len(rows) == 1
    assert rows[0].purpose == "thesis_review"
    assert rows[0].error_class is None
    assert rows[0].cost_usd > 0


def test_boot_registration_covers_active_thesis_prompt(db: str) -> None:
    """`register_composed_prompt_versions` must pre-register the ACTIVE
    thesis-review pv so the llm_calls FK can never fire after the spend
    (the 2026-07-01 silently-discarded-calls incident class)."""
    b = PromptBuilder(_PROMPTS_DIR)
    register_composed_prompt_versions(b, db)
    pv = b.prompt_version(ACTIVE_OPUS_THESIS_REVIEW_PROMPT)
    registered = get_prompt_by_version(db, pv)
    assert registered is not None
    assert registered.name == ACTIVE_OPUS_THESIS_REVIEW_PROMPT
