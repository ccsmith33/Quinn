"""Universe snapshot composition helpers (S2.2).

Hashing follows the architecture's content-hash convention used elsewhere
(e.g. filings, prompts): canonical-JSON serialization → SHA-256 → hex digest.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .alpaca_assets import AlpacaAsset
from .sec_tickers import SecTicker

ALLOWED_EXCHANGES = frozenset({"NYSE", "NASDAQ", "ARCA", "AMEX"})
MARKET_CAP_FLOOR_USD = 50_000_000.0
MARKET_CAP_CEILING_USD = 2_000_000_000.0
PREV_CLOSE_FLOOR_USD = 5.00


def _canonical_hash(payload: Any, *, seed: str) -> str:
    """Hash a JSON-serializable payload with an optional seed prefix.

    The seed mixes test-supplied entropy into the hash without altering the
    payload's logical content. Production callers pass an empty seed.
    """
    blob = json.dumps([seed, payload], sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


def hash_sec_tickers(tickers: list[SecTicker], *, seed: str = "") -> str:
    rows = [(t.cik, t.ticker, t.name) for t in tickers]
    rows.sort()
    return _canonical_hash([list(r) for r in rows], seed=seed)


def hash_alpaca_assets(assets: list[AlpacaAsset], *, seed: str = "") -> str:
    rows = [
        (a.symbol, a.exchange, a.status, a.tradable, a.fractionable, a.asset_class)
        for a in assets
    ]
    rows.sort()
    return _canonical_hash([list(r) for r in rows], seed=seed)
