"""S5.2 — Live Anthropic API smoke test (AC-6 second half).

Gated behind `QUINN_RUN_LIVE_ANTHROPIC=1` so it does not run in normal
development or CI without explicit opt-in (paid API call). Validates
that the cache-hit path exercised in production actually produces
`cache_read_input_tokens > 0` after the cache warm-up.

Per ADR-005 cache test scenario: "100 calls in an hour → cache-read
tokens > 0 on calls 2..100". This smoke test exercises a small number
of calls (2) to validate the cache contract end-to-end without burning
budget.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_LIVE_ENV = "QUINN_RUN_LIVE_ANTHROPIC"
_API_KEY_ENV = "ANTHROPIC_API_KEY"

pytestmark = pytest.mark.skipif(
    os.environ.get(_LIVE_ENV) != "1",
    reason=f"Set {_LIVE_ENV}=1 to run live Anthropic smoke tests",
)


@pytest.mark.asyncio
async def test_real_api_smoke_cache_read(tmp_path: Path) -> None:
    """End-to-end: send a request twice with the same large cacheable
    prefix; the second call should report `cache_read_input_tokens > 0`.

    Requires `ANTHROPIC_API_KEY` in the environment alongside the gate.
    """
    api_key = os.environ.get(_API_KEY_ENV)
    if not api_key:
        pytest.fail(f"{_LIVE_ENV}=1 set but {_API_KEY_ENV} is not in env")

    from pydantic import SecretStr

    from analyzer.anthropic_client import AnthropicClient
    from journal.migrate import apply_migrations
    from journal.models import PromptRow
    from journal.repo import get_llm_calls_by_decision_id, insert_prompt
    from prompts.loader import ApiRequest, Block, Message

    db_path = str(tmp_path / "journal.db")
    apply_migrations(db_path)
    pv = "live_smoke_v1@" + ("a" * 12)
    insert_prompt(
        db_path,
        PromptRow(
            prompt_version=pv,
            name="live_smoke_v1",
            file_path="/tmp/live_smoke.txt",
            content_hash="a" * 64,
        ),
    )

    # Block-1 must be ≥1024 tokens for Sonnet 4.6 to be cacheable
    # (per `shared/prompt-caching.md` minimum cacheable prefix).
    cacheable_prefix = ("Quinn live smoke test prefix. " * 600).strip()

    def _request(block3_text: str) -> ApiRequest:
        return ApiRequest(
            system=[Block(text=cacheable_prefix, cache_control={"type": "ephemeral"})],
            messages=[Message(role="user", content=[Block(text=block3_text)])],
            prompt_version=pv,
        )

    client = AnthropicClient(
        api_key=SecretStr(api_key),
        default_model_id="claude-haiku-4-5",
        db_path=db_path,
    )

    # First call — writes cache.
    await client.call(
        _request("first call payload"),
        model_id="claude-haiku-4-5",
        purpose="audit",
        decision_id="live-smoke-1",
    )
    # Second call — should read cache.
    await client.call(
        _request("second call payload"),
        model_id="claude-haiku-4-5",
        purpose="audit",
        decision_id="live-smoke-2",
    )
    [second] = get_llm_calls_by_decision_id(db_path, "live-smoke-2")
    assert second.cache_read_tokens > 0, (
        f"expected cache hit on second call; usage was: {second!r}"
    )
