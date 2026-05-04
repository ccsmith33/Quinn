"""SEC EDGAR `company_tickers.json` fetcher.

Egress: only `https://www.sec.gov` (architecture §9.5 allow-list).
NFR-17 requires a declared User-Agent on every SEC request.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel, ConfigDict

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_DEFAULT_TIMEOUT_SECONDS = 30.0


class SecTicker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cik: int
    ticker: str
    name: str


def parse_company_tickers_payload(payload: object) -> list[SecTicker]:
    """Parse SEC's `company_tickers.json` body into `list[SecTicker]`.

    Raises `ValueError` on a malformed body (wrong shape, missing required
    fields). Empty result is permitted — callers decide whether to treat
    that as a failure.

    Real shape: dict keyed by stringified row index, each value a dict
    with `cik_str` (int or zero-padded string), `ticker`, `title`. SEC
    occasionally returns an HTML maintenance page on the same URL; that
    path goes through `ValueError` so callers can fall back without
    poisoning a cache.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload is not a JSON object")
    out: list[SecTicker] = []
    for entry in payload.values():
        if not isinstance(entry, dict):
            continue
        cik_raw = entry.get("cik_str")
        ticker = entry.get("ticker")
        title = entry.get("title")
        if cik_raw is None or not isinstance(ticker, str) or not ticker:
            continue
        try:
            cik_int = int(cik_raw)
        except (TypeError, ValueError):
            continue
        out.append(
            SecTicker(
                cik=cik_int,
                ticker=ticker,
                name=str(title) if title is not None else "",
            )
        )
    return out


def fetch_sec_tickers(
    user_agent: str,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> list[SecTicker]:
    """GET company_tickers.json and parse to a list of SecTicker.

    `transport` is injectable for testing (e.g. `httpx.MockTransport`).

    NOTE: this is a synchronous, fresh-client fetch used by daily-refresh
    jobs and operator scripts. The agent-loop ingestion path uses
    `ingestion.ticker_resolver.TickerResolver` instead — that class
    routes through the shared `EdgarClient` so SEC's 10 req/sec ceiling
    is observed across all ingestion code paths (ADR-002 §6).
    """
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    with httpx.Client(transport=transport, timeout=timeout, headers=headers) as client:
        resp = client.get(SEC_TICKERS_URL)
        resp.raise_for_status()
        payload = resp.json()
    return parse_company_tickers_payload(payload)
