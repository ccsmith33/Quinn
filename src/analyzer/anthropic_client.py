"""Anthropic SDK wrapper — cache-aware request construction, per-call
telemetry into `llm_calls`, and bounded retry on transient errors.

Architecture references: §2.3 (LLM Analyzer failure modes), §10.1
(Observability), ADR-005 (3-block cache structure), FR-29, NFR-5
(retry on 429/503), NFR-8 (per-call telemetry), NFR-12 (cache-hit rate).

Per S5.2 dev notes: the client is transport-only — JSON parsing of the
response is the caller's job (S5.3 / S5.4). decision_id is generated
upstream as `hash(filing_id + model_id + prompt_version)` per
architecture §8.1 `proposals.decision_id`; this client just records it.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Literal, Protocol

import anthropic
from pydantic import SecretStr

from journal.models import LlmCallRow
from journal.repo import insert_llm_call
from observability.log_port import get_logger
from prompts.loader import ApiRequest

from .pricing import compute_cost

log = get_logger(__name__)

# "postmortem" / "synthesis" are the desk-journal Haiku calls (LLM memory
# layer) — journaled through the same llm_calls telemetry as trading calls.
# "thesis_review" is the Feature A open-position event-review layer
# (2026-08-04 cost-ledger fix: it used to ride the "review" tag, which
# made it indistinguishable from Opus proposal reviews in every
# purpose-keyed spend rollup).
Purpose = Literal[
    "analyze", "review", "thesis_review", "replay", "audit",
    "postmortem", "synthesis",
]

# Retry policy per S5.2 AC-3: full-jitter exponential backoff,
# base 1s, cap 60s, max 5 attempts.
_RETRY_BASE_SECONDS = 1.0
_RETRY_CAP_SECONDS = 60.0
_RETRY_MAX_ATTEMPTS = 5


class AnthropicUnavailable(Exception):
    """Raised when retries are exhausted on a transient transport error."""


async def _async_sleep(seconds: float) -> None:
    """Indirection so tests can monkeypatch backoff to no-op."""
    await asyncio.sleep(seconds)


def _classify_transient(exc: BaseException) -> str | None:
    """Return an `error_class` label if `exc` is retryable, else None.

    Per AC-3: 429 → "rate_limit"; 503 → "service_unavailable"; connection
    errors → "connection". Other 4xx are non-retryable client errors.
    """
    if isinstance(exc, anthropic.RateLimitError):
        return "rate_limit"
    if isinstance(exc, anthropic.APIConnectionError):
        return "connection"
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status == 503:
            return "service_unavailable"
        if status is not None and status >= 500:
            return "server_error"
    return None


def _backoff_seconds(attempt: int) -> float:
    """Full-jitter exponential backoff: random.uniform(0, min(cap, base*2^n))."""
    upper = min(_RETRY_CAP_SECONDS, _RETRY_BASE_SECONDS * (2 ** attempt))
    return random.uniform(0.0, upper)


class _AsyncMessages(Protocol):
    """Subset of `anthropic.AsyncAnthropic.messages` used here."""

    async def create(self, **kwargs: Any) -> Any: ...  # noqa: E704


class _AsyncClient(Protocol):
    """Subset of `anthropic.AsyncAnthropic` used here."""

    messages: _AsyncMessages


class AnthropicClient:
    """Wraps `anthropic.AsyncAnthropic` with telemetry + retry.

    The constructor accepts an optional `_client` for tests (dependency
    injection seam — see D-036). In production, leave `_client=None` and
    the SDK's `AsyncAnthropic` is constructed from `api_key`.
    """

    def __init__(
        self,
        api_key: SecretStr,
        default_model_id: str,
        db_path: str,
        *,
        _client: _AsyncClient | None = None,
    ) -> None:
        self._default_model_id = default_model_id
        self._db_path = db_path
        # Disable SDK auto-retries (max_retries=0) — our explicit policy
        # supersedes it (D-036): tests need deterministic retry counts.
        self._client: _AsyncClient = _client or anthropic.AsyncAnthropic(  # type: ignore[assignment]
            api_key=api_key.get_secret_value(),
            max_retries=0,
        )

    async def call(
        self,
        request: ApiRequest,
        *,
        model_id: str,
        purpose: Purpose,
        decision_id: str,
        max_tokens: int | None = None,
    ) -> str:
        """Send `request` to Anthropic and return the raw text response.

        Writes one `llm_calls` row per call (success or terminal failure),
        emits a structured `event="anthropic.call"` log line, and raises
        `AnthropicUnavailable` if retries are exhausted.

        Per S5.6 carry-forward (S5.2 reviewer C-1, medium): callers MAY
        pass `max_tokens` to override the SDK kwargs default. Production
        wiring (S5.6) passes `config.analyzer.{sonnet,opus}_max_output_tokens`
        per call so the cost ceiling is config-controlled. See D-052.
        """
        sdk_kwargs = _to_sdk_kwargs(request, model_id=model_id, max_tokens=max_tokens)
        start = time.monotonic()
        response = await self._call_with_retry(
            sdk_kwargs,
            decision_id=decision_id,
            purpose=purpose,
            model_id=model_id,
            prompt_version=request.prompt_version,
            start=start,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        usage = response.usage
        input_tokens = int(usage.input_tokens)
        output_tokens = int(usage.output_tokens)
        cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_creation_tokens = int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )
        cost_usd = compute_cost(
            model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
        text = _extract_text(response)
        self._record_call(
            decision_id=decision_id,
            purpose=purpose,
            model_id=model_id,
            prompt_version=request.prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            error_class=None,
        )
        return text

    async def _call_with_retry(
        self,
        sdk_kwargs: dict[str, Any],
        *,
        decision_id: str,
        purpose: str,
        model_id: str,
        prompt_version: str,
        start: float,
    ) -> Any:
        """Send the request with bounded retry on transient transport errors.

        On exhaustion, records a single `llm_calls` row with `error_class`
        set, then raises `AnthropicUnavailable`. Non-transient errors
        (4xx other than 429) propagate without a row write — they are bugs
        in the caller, not transport issues, and journaling them would
        muddy the analytics view.
        """
        last_error_class: str | None = None
        last_exc: BaseException | None = None
        for attempt in range(_RETRY_MAX_ATTEMPTS):
            try:
                return await self._client.messages.create(**sdk_kwargs)
            except BaseException as exc:  # noqa: BLE001 — re-raise after classifying
                error_class = _classify_transient(exc)
                if error_class is None:
                    raise
                last_error_class = error_class
                last_exc = exc
                if attempt + 1 >= _RETRY_MAX_ATTEMPTS:
                    break
                await _async_sleep(_backoff_seconds(attempt))
        latency_ms = int((time.monotonic() - start) * 1000)
        self._record_call(
            decision_id=decision_id,
            purpose=purpose,
            model_id=model_id,
            prompt_version=prompt_version,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            error_class=last_error_class,
        )
        raise AnthropicUnavailable(
            f"retries exhausted ({_RETRY_MAX_ATTEMPTS} attempts); "
            f"last error_class={last_error_class}"
        ) from last_exc

    def _record_call(
        self,
        *,
        decision_id: str,
        purpose: str,
        model_id: str,
        prompt_version: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_creation_tokens: int,
        latency_ms: int,
        cost_usd: float,
        error_class: str | None,
    ) -> None:
        row = LlmCallRow(
            decision_id=decision_id,
            purpose=purpose,
            model_id=model_id,
            prompt_version=prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            error_class=error_class,
        )
        insert_llm_call(self._db_path, row)
        log.info(
            "anthropic.call",
            extra={
                "event": "anthropic.call",
                "decision_id": decision_id,
                "purpose": purpose,
                "model_id": model_id,
                "prompt_version": prompt_version,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "error_class": error_class,
            },
        )


_DEFAULT_MAX_TOKENS = 16000


def _to_sdk_kwargs(
    request: ApiRequest, *, model_id: str, max_tokens: int | None = None
) -> dict[str, Any]:
    """Convert our typed `ApiRequest` into `client.messages.create` kwargs.

    Cache-control markers on `Block`s pass through verbatim (AC-5). The SDK
    accepts a list of content blocks for `system`, and a list of message
    objects for `messages`.

    `max_tokens` is the per-call cost ceiling. Callers SHOULD pass
    `config.analyzer.{sonnet,opus}_max_output_tokens` (S5.6 carry-fwd
    S5.2 C-1, D-052). Falls back to `_DEFAULT_MAX_TOKENS` only when no
    explicit value is supplied — present for backward compatibility
    with pre-S5.6 callers.
    """
    system_blocks = [_block_to_sdk(b) for b in request.system]
    messages = [
        {
            "role": m.role,
            "content": [_block_to_sdk(b) for b in m.content],
        }
        for m in request.messages
    ]
    return {
        "model": model_id,
        "max_tokens": max_tokens if max_tokens is not None else _DEFAULT_MAX_TOKENS,
        "system": system_blocks,
        "messages": messages,
    }


def _block_to_sdk(block: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "text", "text": block.text}
    if block.cache_control is not None:
        out["cache_control"] = block.cache_control
    return out


def _extract_text(response: Any) -> str:
    """Pull the first text block out of `response.content` (AC-4)."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return str(block.text)
    return ""
