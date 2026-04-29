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


def fetch_sec_tickers(
    user_agent: str,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> list[SecTicker]:
    """GET company_tickers.json and parse to a list of SecTicker.

    `transport` is injectable for testing (e.g. `httpx.MockTransport`).
    """
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    with httpx.Client(transport=transport, timeout=timeout, headers=headers) as client:
        resp = client.get(SEC_TICKERS_URL)
        resp.raise_for_status()
        payload = resp.json()
    out: list[SecTicker] = []
    for entry in payload.values():
        out.append(
            SecTicker(
                cik=int(entry["cik_str"]),
                ticker=str(entry["ticker"]),
                name=str(entry["title"]),
            )
        )
    return out
