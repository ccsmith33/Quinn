"""S3.1 — EDGAR HTTP client tests.

ADR-002 §6 governs the rate-limit posture: 10 req/sec global, declared
User-Agent, exponential backoff with full jitter on 429/5xx, max 5
attempts, then a circuit breaker (NFR-5).

All HTTP is faked via `httpx.MockTransport`; no real EDGAR calls.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import httpx
import pytest

from ingestion.edgar_client import (
    CircuitOpen,
    EdgarClient,
    EdgarUnavailable,
)

_UA = "Quinn-Research/v1 contact@operator.example"


def _ok_transport(seen: list[httpx.Request] | None = None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handle)


def _scripted_transport(
    responses: list[Callable[[httpx.Request], httpx.Response]],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Cycle through a list of response factories, one per call."""
    seen: list[httpx.Request] = []
    idx = {"i": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        i = min(idx["i"], len(responses) - 1)
        idx["i"] += 1
        return responses[i](request)

    return httpx.MockTransport(handle), seen


# ---------------------------------------------------------------------------
# AC-2: User-Agent is required
# ---------------------------------------------------------------------------


def test_user_agent_required() -> None:
    with pytest.raises(ValueError):
        EdgarClient(user_agent="")
    with pytest.raises(ValueError):
        EdgarClient(user_agent=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_user_agent_header_set_on_every_request() -> None:
    seen: list[httpx.Request] = []
    transport = _ok_transport(seen)
    client = EdgarClient(user_agent=_UA, transport=transport)
    try:
        await client.get("https://www.sec.gov/x")
        await client.get("https://data.sec.gov/y")
    finally:
        await client.aclose()
    assert len(seen) == 2
    for req in seen:
        assert req.headers["user-agent"] == _UA


# ---------------------------------------------------------------------------
# AC-3: Global rate limit — 10 req/sec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_throttles_concurrent_requests() -> None:
    transport = _ok_transport()
    client = EdgarClient(user_agent=_UA, transport=transport)
    try:
        start = time.monotonic()
        await asyncio.gather(
            *[client.get(f"https://www.sec.gov/x?i={i}") for i in range(30)]
        )
        elapsed = time.monotonic() - start
    finally:
        await client.aclose()
    # 30 requests at 10 req/sec ceiling → ≥ 2.9s (allow tiny scheduling slack).
    assert elapsed >= 2.9, f"rate-limit too permissive: 30 reqs in {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# AC-4: 429/5xx retry then success / exhaustion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_429_retries_then_succeeds() -> None:
    transport, seen = _scripted_transport(
        [
            lambda _r: httpx.Response(429, text="slow down"),
            lambda _r: httpx.Response(200, json={"ok": True}),
        ]
    )
    client = EdgarClient(user_agent=_UA, transport=transport, retry_base_seconds=0.0)
    try:
        resp = await client.get("https://www.sec.gov/x")
    finally:
        await client.aclose()
    assert resp.status_code == 200
    assert len(seen) == 2
    assert client.retry_count >= 1


@pytest.mark.asyncio
async def test_5xx_retries_then_succeeds() -> None:
    transport, seen = _scripted_transport(
        [
            lambda _r: httpx.Response(503, text="bad"),
            lambda _r: httpx.Response(502, text="bad"),
            lambda _r: httpx.Response(200, json={"ok": True}),
        ]
    )
    client = EdgarClient(user_agent=_UA, transport=transport, retry_base_seconds=0.0)
    try:
        resp = await client.get("https://www.sec.gov/x")
    finally:
        await client.aclose()
    assert resp.status_code == 200
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_429_exhaustion_raises_unavailable() -> None:
    transport, seen = _scripted_transport(
        [lambda _r: httpx.Response(429, text="nope")]
    )
    client = EdgarClient(user_agent=_UA, transport=transport, retry_base_seconds=0.0)
    try:
        with pytest.raises(EdgarUnavailable):
            await client.get("https://www.sec.gov/x")
    finally:
        await client.aclose()
    assert len(seen) == 5


# ---------------------------------------------------------------------------
# AC-5 / AC-6: circuit breaker + telemetry hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold() -> None:
    transport, _seen = _scripted_transport(
        [lambda _r: httpx.Response(429, text="nope")]
    )
    events: list[tuple[str, dict[str, object]]] = []
    client = EdgarClient(
        user_agent=_UA,
        transport=transport,
        retry_base_seconds=0.0,
        circuit_threshold=3,
        circuit_window_seconds=300.0,
        circuit_cooldown_seconds=300.0,
        on_event=lambda name, ctx: events.append((name, dict(ctx))),
    )
    try:
        for _ in range(3):
            with pytest.raises(EdgarUnavailable):
                await client.get("https://www.sec.gov/x")
        with pytest.raises(CircuitOpen):
            await client.get("https://www.sec.gov/x")
    finally:
        await client.aclose()
    names = [n for n, _ in events]
    assert "circuit_open" in names
    # AC-6: retry events are emitted
    assert any(n == "retry" for n, _ in events)


@pytest.mark.asyncio
async def test_circuit_recovers_after_cooldown() -> None:
    # First 3 calls raise EdgarUnavailable (every attempt is a 429); after the
    # cooldown elapses, upstream is healthy again, the next call succeeds, and
    # the circuit closes.
    state = {"healthy": False}

    def handle(_r: httpx.Request) -> httpx.Response:
        if state["healthy"]:
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(429, text="nope")

    transport = httpx.MockTransport(handle)
    events: list[tuple[str, dict[str, object]]] = []
    client = EdgarClient(
        user_agent=_UA,
        transport=transport,
        retry_base_seconds=0.0,
        circuit_threshold=3,
        circuit_window_seconds=300.0,
        circuit_cooldown_seconds=0.05,
        on_event=lambda name, ctx: events.append((name, dict(ctx))),
    )
    try:
        for _ in range(3):
            with pytest.raises(EdgarUnavailable):
                await client.get("https://www.sec.gov/x")
        # Circuit is now open.
        with pytest.raises(CircuitOpen):
            await client.get("https://www.sec.gov/x")
        # Wait for cooldown, then heal the upstream.
        await asyncio.sleep(0.06)
        state["healthy"] = True
        resp = await client.get("https://www.sec.gov/x")
        assert resp.status_code == 200
    finally:
        await client.aclose()
    names = [n for n, _ in events]
    assert "circuit_open" in names
    assert "circuit_close" in names


# ---------------------------------------------------------------------------
# AC-1: get_bytes variant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_bytes_returns_raw_body() -> None:
    payload = b"<?xml version='1.0'?><root>hi</root>"

    def handle(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    transport = httpx.MockTransport(handle)
    client = EdgarClient(user_agent=_UA, transport=transport)
    try:
        body = await client.get_bytes("https://www.sec.gov/x.xml")
    finally:
        await client.aclose()
    assert body == payload


# ---------------------------------------------------------------------------
# AC-1: query params propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_passes_query_params() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handle)
    client = EdgarClient(user_agent=_UA, transport=transport)
    try:
        await client.get(
            "https://www.sec.gov/cgi-bin/browse-edgar",
            params={"action": "getcurrent", "type": "8-K"},
        )
    finally:
        await client.aclose()
    assert seen[0].url.params["action"] == "getcurrent"
    assert seen[0].url.params["type"] == "8-K"
