"""S2.1 — SEC company_tickers.json fetcher."""

from __future__ import annotations

import json

import httpx
import pytest

from universe.sec_tickers import SEC_TICKERS_URL, SecTicker, fetch_sec_tickers

# Real upstream payload shape:
# {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
_FAKE_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    "2": {"cik_str": 1234567, "ticker": "ACME", "title": "Acme Microcap Corp"},
}


def _ok_handler(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_FAKE_PAYLOAD)

    return httpx.MockTransport(handle)


def test_fetches_and_parses() -> None:
    seen: list[httpx.Request] = []
    transport = _ok_handler(seen)
    result = fetch_sec_tickers(
        user_agent="Quinn-Research/v1 contact@operator.example",
        transport=transport,
    )
    assert isinstance(result, list)
    assert len(result) == 3
    by_ticker = {t.ticker: t for t in result}
    assert isinstance(by_ticker["AAPL"], SecTicker)
    assert by_ticker["AAPL"].cik == 320193
    assert by_ticker["AAPL"].name == "Apple Inc."
    assert by_ticker["ACME"].cik == 1234567
    assert seen[0].url == httpx.URL(SEC_TICKERS_URL)


def test_user_agent_header_set() -> None:
    seen: list[httpx.Request] = []
    transport = _ok_handler(seen)
    ua = "Quinn-Research/v1 contact@operator.example"
    fetch_sec_tickers(user_agent=ua, transport=transport)
    assert seen[0].headers["user-agent"] == ua


def test_raises_on_http_error() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    transport = httpx.MockTransport(handle)
    with pytest.raises(httpx.HTTPStatusError):
        fetch_sec_tickers(user_agent="Q/1", transport=transport)


def test_raises_on_malformed_json() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    transport = httpx.MockTransport(handle)
    with pytest.raises((json.JSONDecodeError, ValueError)):
        fetch_sec_tickers(user_agent="Q/1", transport=transport)
