"""Tests for `ingestion.ticker_resolver.TickerResolver`.

Production hotfix: prior to this resolver, `detail_fetcher.py` hard-coded
`issuer_ticker=None` for all non-Form-4 filings, causing the analyzer to
treat every issuer as not-in-universe and refuse all proposals (0 trades
across the lifetime of the system before the fix).

The resolver consults SEC's `company_tickers.json` (CIK→ticker map) with
a local on-disk cache. Failure modes (per task #1):
  - CIK not in map → returns None, no exception
  - SEC fetch fails → falls back to cached map; no cache → returns None
  - Cache corrupted → re-fetches; never crashes

Tests use `httpx.MockTransport` injected into the shared `EdgarClient`.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ingestion.edgar_client import EdgarClient
from ingestion.ticker_resolver import TickerResolver

_UA = "Quinn-Research/v1 contact@operator.example"
_TICKERS_URL_TAIL = "/files/company_tickers.json"


# Real-shape SEC payload: dict keyed by stringified row index, with cik_str
# (zero-padded numeric or int), ticker, title.
_SEC_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    "2": {"cik_str": 1318605, "ticker": "TSLA", "title": "Tesla, Inc."},
}


def _mock_transport(
    *,
    payload: bytes | None = None,
    status: int = 200,
    on_call: list[str] | None = None,
    raise_exc: bool = False,
) -> httpx.MockTransport:
    """Build a transport serving company_tickers.json once per call.

    `on_call` (optional) is a list passed by reference; each request appends
    its URL so tests can assert "fetched once / fetched zero times".
    """
    if payload is None:
        payload = json.dumps(_SEC_PAYLOAD).encode()

    def handle(request: httpx.Request) -> httpx.Response:
        if on_call is not None:
            on_call.append(str(request.url))
        if raise_exc:
            raise httpx.ConnectError("simulated DNS failure")
        if _TICKERS_URL_TAIL in str(request.url):
            return httpx.Response(status, content=payload)
        return httpx.Response(404, text=f"unhandled {request.url}")

    return httpx.MockTransport(handle)


def _client(transport: httpx.MockTransport) -> EdgarClient:
    return EdgarClient(user_agent=_UA, transport=transport, retry_base_seconds=0.0)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_known_cik_returns_ticker(tmp_path: Path) -> None:
    """CIK 0000320193 → AAPL (the production canary case)."""
    transport = _mock_transport()
    edgar = _client(transport)
    resolver = TickerResolver(edgar=edgar, cache_path=tmp_path / "cik_map.json")
    try:
        ticker = await resolver.resolve(320193)
    finally:
        await edgar.aclose()
    assert ticker == "AAPL"


@pytest.mark.asyncio
async def test_resolve_unknown_cik_returns_none(tmp_path: Path) -> None:
    """CIK not present in SEC map → None, no exception, no crash."""
    transport = _mock_transport()
    edgar = _client(transport)
    resolver = TickerResolver(edgar=edgar, cache_path=tmp_path / "cik_map.json")
    try:
        ticker = await resolver.resolve(999_999_999)
    finally:
        await edgar.aclose()
    assert ticker is None


@pytest.mark.asyncio
async def test_resolve_caches_first_fetch(tmp_path: Path) -> None:
    """Multiple resolves within one process do not re-hit SEC."""
    calls: list[str] = []
    transport = _mock_transport(on_call=calls)
    edgar = _client(transport)
    resolver = TickerResolver(edgar=edgar, cache_path=tmp_path / "cik_map.json")
    try:
        await resolver.resolve(320193)
        await resolver.resolve(789019)
        await resolver.resolve(1318605)
    finally:
        await edgar.aclose()
    # Exactly one HTTP call was made (the rest are in-memory hits).
    assert len([c for c in calls if _TICKERS_URL_TAIL in c]) == 1


@pytest.mark.asyncio
async def test_resolve_writes_disk_cache(tmp_path: Path) -> None:
    """First successful fetch persists the map to `cache_path`."""
    transport = _mock_transport()
    edgar = _client(transport)
    cache_path = tmp_path / "cik_map.json"
    resolver = TickerResolver(edgar=edgar, cache_path=cache_path)
    try:
        await resolver.resolve(320193)
    finally:
        await edgar.aclose()
    assert cache_path.exists()
    cached = json.loads(cache_path.read_text())
    # Cache shape contract: stringified-CIK → ticker.
    assert cached["320193"] == "AAPL"
    assert cached["789019"] == "MSFT"


# ---------------------------------------------------------------------------
# Failure modes — the part that protects ingest from crashing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sec_unreachable_falls_back_to_disk_cache(tmp_path: Path) -> None:
    """SEC unreachable + warm disk cache → resolve from cache, no exception."""
    cache_path = tmp_path / "cik_map.json"
    cache_path.write_text(json.dumps({"320193": "AAPL", "789019": "MSFT"}))

    transport = _mock_transport(raise_exc=True)
    edgar = _client(transport)
    resolver = TickerResolver(edgar=edgar, cache_path=cache_path)
    try:
        ticker = await resolver.resolve(320193)
    finally:
        await edgar.aclose()
    assert ticker == "AAPL"


@pytest.mark.asyncio
async def test_sec_unreachable_no_cache_returns_none(tmp_path: Path) -> None:
    """SEC unreachable + no cache → None (ingest continues with NULL ticker)."""
    transport = _mock_transport(raise_exc=True)
    edgar = _client(transport)
    resolver = TickerResolver(edgar=edgar, cache_path=tmp_path / "cik_map.json")
    try:
        ticker = await resolver.resolve(320193)
    finally:
        await edgar.aclose()
    assert ticker is None


@pytest.mark.asyncio
async def test_sec_5xx_falls_back_to_disk_cache(tmp_path: Path) -> None:
    """SEC 5xx (after retries exhaust) + warm cache → cache hit."""
    cache_path = tmp_path / "cik_map.json"
    cache_path.write_text(json.dumps({"320193": "AAPL"}))

    # EdgarClient retries 5xx; setting max_attempts=1 short-circuits to one
    # attempt so the test runs fast. Resolver must accept that.
    transport = _mock_transport(status=503, payload=b"server error")
    edgar = EdgarClient(
        user_agent=_UA, transport=transport, retry_base_seconds=0.0, max_attempts=1
    )
    resolver = TickerResolver(edgar=edgar, cache_path=cache_path)
    try:
        ticker = await resolver.resolve(320193)
    finally:
        await edgar.aclose()
    assert ticker == "AAPL"


@pytest.mark.asyncio
async def test_sec_4xx_falls_back_to_disk_cache(tmp_path: Path) -> None:
    """SEC 4xx (non-retryable) + warm cache → cache hit, no crash."""
    cache_path = tmp_path / "cik_map.json"
    cache_path.write_text(json.dumps({"320193": "AAPL"}))

    transport = _mock_transport(status=403, payload=b"forbidden")
    edgar = _client(transport)
    resolver = TickerResolver(edgar=edgar, cache_path=cache_path)
    try:
        ticker = await resolver.resolve(320193)
    finally:
        await edgar.aclose()
    assert ticker == "AAPL"


@pytest.mark.asyncio
async def test_corrupt_cache_triggers_refetch(tmp_path: Path) -> None:
    """Garbage on disk → discard, re-fetch from SEC, succeed."""
    cache_path = tmp_path / "cik_map.json"
    cache_path.write_text("{this is not valid json")

    transport = _mock_transport()
    edgar = _client(transport)
    resolver = TickerResolver(edgar=edgar, cache_path=cache_path)
    try:
        ticker = await resolver.resolve(320193)
    finally:
        await edgar.aclose()
    assert ticker == "AAPL"
    # After re-fetch, the cache should be replaced with valid JSON.
    cached = json.loads(cache_path.read_text())
    assert cached["320193"] == "AAPL"


@pytest.mark.asyncio
async def test_corrupt_cache_and_sec_unreachable_returns_none(tmp_path: Path) -> None:
    """Worst case — corrupt cache + SEC down → None, no exception."""
    cache_path = tmp_path / "cik_map.json"
    cache_path.write_text("{garbage")

    transport = _mock_transport(raise_exc=True)
    edgar = _client(transport)
    resolver = TickerResolver(edgar=edgar, cache_path=cache_path)
    try:
        ticker = await resolver.resolve(320193)
    finally:
        await edgar.aclose()
    assert ticker is None


@pytest.mark.asyncio
async def test_malformed_sec_payload_returns_none(tmp_path: Path) -> None:
    """SEC returns 200 but body is not the expected JSON shape → None,
    do not poison the cache.
    """
    transport = _mock_transport(payload=b"<html>maintenance</html>")
    edgar = _client(transport)
    cache_path = tmp_path / "cik_map.json"
    resolver = TickerResolver(edgar=edgar, cache_path=cache_path)
    try:
        ticker = await resolver.resolve(320193)
    finally:
        await edgar.aclose()
    assert ticker is None
    # Cache must NOT have been written (we don't want garbage in there).
    assert not cache_path.exists()


# ---------------------------------------------------------------------------
# Concurrency — single-flight against SEC under simultaneous resolves
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_resolves_dedupe_to_single_fetch(tmp_path: Path) -> None:
    """Two coroutines racing for the first resolve must NOT each fire a
    separate SEC GET. The resolver should single-flight the cold-start fetch.
    """
    import asyncio

    calls: list[str] = []
    transport = _mock_transport(on_call=calls)
    edgar = _client(transport)
    resolver = TickerResolver(edgar=edgar, cache_path=tmp_path / "cik_map.json")
    try:
        results = await asyncio.gather(
            resolver.resolve(320193),
            resolver.resolve(789019),
            resolver.resolve(1318605),
        )
    finally:
        await edgar.aclose()
    assert results == ["AAPL", "MSFT", "TSLA"]
    assert len([c for c in calls if _TICKERS_URL_TAIL in c]) == 1


# ---------------------------------------------------------------------------
# Disk-cache reuse across resolver instances (warm-start path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disk_cache_used_on_warm_start_no_network(tmp_path: Path) -> None:
    """A pre-existing valid cache is consulted first; the resolver does not
    hit SEC for a CIK that is already in the disk cache."""
    cache_path = tmp_path / "cik_map.json"
    cache_path.write_text(json.dumps({"320193": "AAPL"}))

    calls: list[str] = []
    transport = _mock_transport(on_call=calls)
    edgar = _client(transport)
    resolver = TickerResolver(
        edgar=edgar, cache_path=cache_path, refresh_on_load=False
    )
    try:
        ticker = await resolver.resolve(320193)
    finally:
        await edgar.aclose()
    assert ticker == "AAPL"
    # Zero network calls — disk cache hit only.
    assert calls == []
