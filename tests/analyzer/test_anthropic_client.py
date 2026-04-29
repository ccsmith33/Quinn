"""S5.2 — Anthropic client + telemetry tests.

Architecture references: §2.3 (LLM Analyzer failure modes), §10.1
(Observability), ADR-005 (cache-aware request shape), FR-29, NFR-5,
NFR-8, NFR-12.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import pytest
from pydantic import SecretStr

from journal.migrate import apply_migrations
from journal.models import PromptRow
from journal.repo import get_llm_calls_by_decision_id, insert_prompt
from prompts.loader import ApiRequest, Block, Message

# ---------------------------------------------------------------------------
# Fakes
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
    model: str = "claude-sonnet-4-6"


class _FakeMessages:
    """Stand-in for `client.messages` — captures call kwargs and returns
    the response from the queued list (or raises a queued exception)."""

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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return str(db_path)


@pytest.fixture
def registered_prompt_version(db: str) -> str:
    """Insert a row into `prompts` so the FK from `llm_calls.prompt_version`
    resolves. Returns the registered version id."""
    pv = "sonnet_filing_analysis_v1@deadbeef0001"
    insert_prompt(
        db,
        PromptRow(
            prompt_version=pv,
            name="sonnet_filing_analysis_v1",
            file_path="src/prompts/sonnet_filing_analysis_v1.txt",
            content_hash="deadbeef0001" + "0" * 52,
        ),
    )
    return pv


def _make_request(prompt_version: str) -> ApiRequest:
    return ApiRequest(
        system=[
            Block(text="block1", cache_control={"type": "ephemeral"}),
            Block(text="block2", cache_control={"type": "ephemeral"}),
        ],
        messages=[Message(role="user", content=[Block(text="block3")])],
        prompt_version=prompt_version,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_records_llm_calls_row(
    db: str, registered_prompt_version: str
) -> None:
    """AC-2: every successful call writes one row to `llm_calls` with the
    expected fields (decision_id, purpose, model_id, prompt_version, tokens,
    latency_ms, cost_usd, error_class)."""
    from analyzer.anthropic_client import AnthropicClient

    fake = _FakeAsyncAnthropic()
    fake.messages.queue(
        _FakeResponse(
            content=[_FakeTextBlock(text="ok")],
            usage=_FakeUsage(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=200,
                cache_creation_input_tokens=300,
            ),
            model="claude-sonnet-4-6",
        )
    )

    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id="claude-sonnet-4-6",
        db_path=db,
        _client=fake,
    )

    request = _make_request(registered_prompt_version)
    text = await client.call(
        request,
        model_id="claude-sonnet-4-6",
        purpose="analyze",
        decision_id="dec-001",
    )

    assert text == "ok"
    rows = get_llm_calls_by_decision_id(db, "dec-001")
    assert len(rows) == 1
    row = rows[0]
    assert row.decision_id == "dec-001"
    assert row.purpose == "analyze"
    assert row.model_id == "claude-sonnet-4-6"
    assert row.prompt_version == registered_prompt_version
    assert row.input_tokens == 100
    assert row.output_tokens == 50
    assert row.cache_read_tokens == 200
    assert row.cache_creation_tokens == 300
    assert row.latency_ms >= 0
    assert row.cost_usd > 0.0
    assert row.error_class is None


def _rate_limit_error() -> anthropic.RateLimitError:
    """Build a RateLimitError without making a real HTTP call."""
    import httpx

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


def _server_error(status: int = 503) -> anthropic.APIStatusError:
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        status,
        request=request,
        json={"type": "error", "error": {"type": "api_error", "message": "boom"}},
    )
    return anthropic.APIStatusError(
        message="server error",
        response=response,
        body={"type": "error", "error": {"type": "api_error"}},
    )


def _success(model: str = "claude-sonnet-4-6") -> _FakeResponse:
    return _FakeResponse(
        content=[_FakeTextBlock(text="ok")],
        usage=_FakeUsage(input_tokens=10, output_tokens=5),
        model=model,
    )


@pytest.mark.asyncio
async def test_429_retried_then_succeeds(
    db: str, registered_prompt_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-3: 429 → retry with full-jitter backoff; success after retries
    records exactly one `llm_calls` row (the successful attempt)."""
    # Make backoff sleeps no-op so the test runs fast.
    import analyzer.anthropic_client as ac_mod
    from analyzer.anthropic_client import AnthropicClient

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(ac_mod, "_async_sleep", _no_sleep)

    fake = _FakeAsyncAnthropic()
    fake.messages.queue(_rate_limit_error())
    fake.messages.queue(_rate_limit_error())
    fake.messages.queue(_success())

    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id="claude-sonnet-4-6",
        db_path=db,
        _client=fake,
    )

    text = await client.call(
        _make_request(registered_prompt_version),
        model_id="claude-sonnet-4-6",
        purpose="analyze",
        decision_id="dec-002",
    )

    assert text == "ok"
    assert len(fake.messages.calls) == 3
    rows = get_llm_calls_by_decision_id(db, "dec-002")
    # Successful attempts record one row; the transient retries do not.
    assert len(rows) == 1
    assert rows[0].error_class is None


@pytest.mark.asyncio
async def test_429_exhausted_records_error_class(
    db: str, registered_prompt_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-3: when retries exhaust, raise `AnthropicUnavailable` and record
    one `llm_calls` row with `error_class="rate_limit"`."""
    import analyzer.anthropic_client as ac_mod
    from analyzer.anthropic_client import AnthropicClient, AnthropicUnavailable

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(ac_mod, "_async_sleep", _no_sleep)

    fake = _FakeAsyncAnthropic()
    for _ in range(ac_mod._RETRY_MAX_ATTEMPTS):
        fake.messages.queue(_rate_limit_error())

    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id="claude-sonnet-4-6",
        db_path=db,
        _client=fake,
    )

    with pytest.raises(AnthropicUnavailable):
        await client.call(
            _make_request(registered_prompt_version),
            model_id="claude-sonnet-4-6",
            purpose="analyze",
            decision_id="dec-003",
        )

    assert len(fake.messages.calls) == ac_mod._RETRY_MAX_ATTEMPTS
    rows = get_llm_calls_by_decision_id(db, "dec-003")
    assert len(rows) == 1
    assert rows[0].error_class == "rate_limit"
    # Tokens / cost zero on terminal failure (no usage to record).
    assert rows[0].input_tokens == 0
    assert rows[0].output_tokens == 0
    assert rows[0].cost_usd == 0.0


@pytest.mark.asyncio
async def test_503_retried_then_records_error_class(
    db: str, registered_prompt_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-3: 503 is also retryable and labeled `service_unavailable` on
    exhaustion."""
    import analyzer.anthropic_client as ac_mod
    from analyzer.anthropic_client import AnthropicClient, AnthropicUnavailable

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(ac_mod, "_async_sleep", _no_sleep)

    fake = _FakeAsyncAnthropic()
    for _ in range(ac_mod._RETRY_MAX_ATTEMPTS):
        fake.messages.queue(_server_error(503))

    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id="claude-sonnet-4-6",
        db_path=db,
        _client=fake,
    )

    with pytest.raises(AnthropicUnavailable):
        await client.call(
            _make_request(registered_prompt_version),
            model_id="claude-sonnet-4-6",
            purpose="review",
            decision_id="dec-004",
        )
    rows = get_llm_calls_by_decision_id(db, "dec-004")
    assert len(rows) == 1
    assert rows[0].error_class == "service_unavailable"
    assert rows[0].purpose == "review"


@pytest.mark.asyncio
async def test_non_retryable_error_propagates_without_journaling(
    db: str, registered_prompt_version: str
) -> None:
    """AC-3 implication: non-transient client errors (e.g., 400) propagate
    unchanged and DO NOT write a `llm_calls` row — those are caller bugs,
    not transport failures, and pollute the cost telemetry."""
    import httpx

    from analyzer.anthropic_client import AnthropicClient

    fake = _FakeAsyncAnthropic()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        400,
        request=request,
        json={"type": "error", "error": {"type": "invalid_request_error"}},
    )
    bad_request = anthropic.BadRequestError(
        message="bad request",
        response=response,
        body={"type": "error", "error": {"type": "invalid_request_error"}},
    )
    fake.messages.queue(bad_request)

    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id="claude-sonnet-4-6",
        db_path=db,
        _client=fake,
    )

    with pytest.raises(anthropic.BadRequestError):
        await client.call(
            _make_request(registered_prompt_version),
            model_id="claude-sonnet-4-6",
            purpose="analyze",
            decision_id="dec-005",
        )
    # No retry, no row.
    assert len(fake.messages.calls) == 1
    assert get_llm_calls_by_decision_id(db, "dec-005") == []


@pytest.mark.asyncio
async def test_cost_computed_from_token_counts(
    db: str, registered_prompt_version: str
) -> None:
    """AC-2 cost lookup: cost is computed from the per-model pricing table.

    Sonnet 4.6 = $3/$15 per 1M (input/output), cache-read = 0.1×, cache-write = 1.25×.
    """
    from analyzer.anthropic_client import AnthropicClient

    fake = _FakeAsyncAnthropic()
    fake.messages.queue(
        _FakeResponse(
            content=[_FakeTextBlock(text="ok")],
            usage=_FakeUsage(
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                cache_read_input_tokens=1_000_000,
                cache_creation_input_tokens=1_000_000,
            ),
        )
    )
    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id="claude-sonnet-4-6",
        db_path=db,
        _client=fake,
    )
    await client.call(
        _make_request(registered_prompt_version),
        model_id="claude-sonnet-4-6",
        purpose="analyze",
        decision_id="dec-006",
    )
    rows = get_llm_calls_by_decision_id(db, "dec-006")
    expected = 3.00 + 15.00 + (3.00 * 0.1) + (3.00 * 1.25)
    assert rows[0].cost_usd == pytest.approx(expected, rel=1e-9)


@pytest.mark.asyncio
async def test_cache_tokens_extracted_from_response(
    db: str, registered_prompt_version: str
) -> None:
    """AC-6 (offline half): cache_read / cache_creation tokens populated
    on `usage` are correctly recorded into the `llm_calls` row."""
    from analyzer.anthropic_client import AnthropicClient

    fake = _FakeAsyncAnthropic()
    fake.messages.queue(
        _FakeResponse(
            content=[_FakeTextBlock(text="ok")],
            usage=_FakeUsage(
                input_tokens=42,
                output_tokens=7,
                cache_read_input_tokens=4096,
                cache_creation_input_tokens=2048,
            ),
        )
    )
    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id="claude-sonnet-4-6",
        db_path=db,
        _client=fake,
    )
    await client.call(
        _make_request(registered_prompt_version),
        model_id="claude-sonnet-4-6",
        purpose="analyze",
        decision_id="dec-007",
    )
    [row] = get_llm_calls_by_decision_id(db, "dec-007")
    assert row.cache_read_tokens == 4096
    assert row.cache_creation_tokens == 2048


@pytest.mark.asyncio
async def test_cache_control_markers_pass_through(
    db: str, registered_prompt_version: str
) -> None:
    """AC-5: the client does not strip or alter `cache_control` markers
    on system blocks — they reach the SDK call kwargs verbatim."""
    from analyzer.anthropic_client import AnthropicClient

    fake = _FakeAsyncAnthropic()
    fake.messages.queue(_success())
    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id="claude-sonnet-4-6",
        db_path=db,
        _client=fake,
    )
    await client.call(
        _make_request(registered_prompt_version),
        model_id="claude-sonnet-4-6",
        purpose="analyze",
        decision_id="dec-008",
    )
    [kwargs] = fake.messages.calls
    system = kwargs["system"]
    # Both system blocks carry ephemeral cache_control per ADR-005.
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    # Block-3 (filing payload) carries no cache_control.
    msg = kwargs["messages"][0]
    assert "cache_control" not in msg["content"][0]


@pytest.mark.asyncio
async def test_call_emits_structured_log(
    db: str,
    registered_prompt_version: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC-7: every API call emits an `event="anthropic.call"` log line with
    the same fields as the `llm_calls` row."""
    import logging

    from analyzer.anthropic_client import AnthropicClient

    fake = _FakeAsyncAnthropic()
    fake.messages.queue(
        _FakeResponse(
            content=[_FakeTextBlock(text="ok")],
            usage=_FakeUsage(
                input_tokens=11,
                output_tokens=22,
                cache_read_input_tokens=33,
                cache_creation_input_tokens=44,
            ),
        )
    )
    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id="claude-sonnet-4-6",
        db_path=db,
        _client=fake,
    )
    with caplog.at_level(logging.INFO, logger="analyzer.anthropic_client"):
        await client.call(
            _make_request(registered_prompt_version),
            model_id="claude-sonnet-4-6",
            purpose="analyze",
            decision_id="dec-009",
        )
    matching = [r for r in caplog.records if getattr(r, "event", None) == "anthropic.call"]
    assert len(matching) == 1
    record = matching[0]
    assert record.decision_id == "dec-009"
    assert record.purpose == "analyze"
    assert record.model_id == "claude-sonnet-4-6"
    assert record.prompt_version == registered_prompt_version
    assert record.input_tokens == 11
    assert record.output_tokens == 22
    assert record.cache_read_tokens == 33
    assert record.cache_creation_tokens == 44
    assert record.error_class is None


@pytest.mark.asyncio
async def test_connection_error_classified_and_retried(
    db: str, registered_prompt_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-3: APIConnectionError → retried, classified as 'connection'."""
    import httpx

    import analyzer.anthropic_client as ac_mod
    from analyzer.anthropic_client import AnthropicClient, AnthropicUnavailable

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(ac_mod, "_async_sleep", _no_sleep)

    fake = _FakeAsyncAnthropic()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    for _ in range(ac_mod._RETRY_MAX_ATTEMPTS):
        fake.messages.queue(anthropic.APIConnectionError(request=request))

    client = AnthropicClient(
        api_key=SecretStr("test-key"),
        default_model_id="claude-sonnet-4-6",
        db_path=db,
        _client=fake,
    )
    with pytest.raises(AnthropicUnavailable):
        await client.call(
            _make_request(registered_prompt_version),
            model_id="claude-sonnet-4-6",
            purpose="analyze",
            decision_id="dec-010",
        )
    [row] = get_llm_calls_by_decision_id(db, "dec-010")
    assert row.error_class == "connection"
