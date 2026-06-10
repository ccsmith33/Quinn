"""Daily bars + split events from the Alpaca *market data* API, with a local
JSON cache so repeat runs (and offline runs) need no network.

Safety: this module talks only to ``data.alpaca.markets``. It never reads
``ALPACA_ENDPOINT`` and imports no trading client — it is structurally
incapable of reaching the trading API. Credentials are the two data keys
from the environment (same convention as ``ops/scripts/place_stops.py``);
``src/config/secrets.load_secrets`` is deliberately not used because it
demands the full production secret set this tool has no business holding.

Bars are fetched with ``adjustment=raw`` (split-unadjusted) because the
journaled proposal geometry is in as-of-proposal-time price terms; the
simulator applies split events to the geometry explicitly instead.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ops.replay.models import DailyBar, SplitEvent

DATA_BASE = "https://data.alpaca.markets"
_CACHE_VERSION = 1


class BarsUnavailable(Exception):
    """Raised when bars for a symbol cannot be served from cache or API."""


def _auth_headers() -> dict[str, str]:
    key_id = os.environ.get("ALPACA_API_KEY_ID", "")
    secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
    if not key_id or not secret:
        raise BarsUnavailable(
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set; "
            "export them or run with --offline against a populated cache"
        )
    return {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}


def _get_json(url: str, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise BarsUnavailable(f"HTTP {e.code} GET {url.split('?')[0]}: {body}") from e
    except urllib.error.URLError as e:
        raise BarsUnavailable(f"network error: {e.reason}") from e


def _fetch_bars(
    symbol: str, start: dt.date, end: dt.date, feed: str, headers: dict[str, str]
) -> list[DailyBar]:
    bars: list[DailyBar] = []
    page_token: str | None = None
    while True:
        params = {
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": "raw",
            "feed": feed,
            "limit": "10000",
        }
        if page_token:
            params["page_token"] = page_token
        url = f"{DATA_BASE}/v2/stocks/{symbol}/bars?{urllib.parse.urlencode(params)}"
        payload = _get_json(url, headers)
        for b in payload.get("bars") or []:
            bars.append(
                DailyBar(
                    date=dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00")).date(),
                    open=float(b["o"]),
                    high=float(b["h"]),
                    low=float(b["l"]),
                    close=float(b["c"]),
                    volume=float(b.get("v", 0)),
                )
            )
        page_token = payload.get("next_page_token")
        if not page_token:
            return sorted(bars, key=lambda b: b.date)


def _fetch_splits(
    symbol: str, start: dt.date, end: dt.date, headers: dict[str, str]
) -> list[SplitEvent]:
    params = {
        "types": "forward_split,reverse_split",
        "symbols": symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": "1000",
    }
    url = f"{DATA_BASE}/v1beta1/corporate-actions?{urllib.parse.urlencode(params)}"
    payload = _get_json(url, headers)
    actions = payload.get("corporate_actions") or {}
    splits: list[SplitEvent] = []
    for kind in ("forward_splits", "reverse_splits"):
        for s in actions.get(kind) or []:
            old_rate = float(s["old_rate"])
            if old_rate == 0:
                continue
            splits.append(
                SplitEvent(
                    ex_date=dt.date.fromisoformat(s["ex_date"]),
                    ratio=float(s["new_rate"]) / old_rate,
                )
            )
    return sorted(splits, key=lambda s: s.ex_date)


class BarsClient:
    """Cache-first daily-bars source.

    Cache layout: one JSON file per symbol under `cache_dir`, holding the
    fetched date range, bars, and split events. A cache file is reused when
    it covers the requested range and was fetched from the same feed;
    otherwise the full range is refetched (daily-bar volumes are tiny —
    simplicity over delta-fetching).
    """

    def __init__(self, cache_dir: str | Path, feed: str = "sip", offline: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.feed = feed
        self.offline = offline

    def get(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> tuple[list[DailyBar], list[SplitEvent]]:
        cached = self._read_cache(symbol, start, end)
        if cached is not None:
            return cached
        if self.offline:
            raise BarsUnavailable(
                f"--offline and cache miss for {symbol} [{start}..{end}] "
                f"in {self.cache_dir}"
            )
        headers = _auth_headers()
        bars = _fetch_bars(symbol, start, end, self.feed, headers)
        splits = _fetch_splits(symbol, start, end, headers)
        self._write_cache(symbol, start, end, bars, splits)
        return bars, splits

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}.json"

    def _read_cache(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> tuple[list[DailyBar], list[SplitEvent]] | None:
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if doc.get("version") != _CACHE_VERSION:
            return None
        if doc.get("feed") != self.feed:
            # a cache written from another feed must not silently serve this run
            return None
        if dt.date.fromisoformat(doc["start"]) > start or dt.date.fromisoformat(doc["end"]) < end:
            return None
        bars = [
            DailyBar(
                date=dt.date.fromisoformat(b["date"]),
                open=b["open"],
                high=b["high"],
                low=b["low"],
                close=b["close"],
                volume=b.get("volume", 0.0),
            )
            for b in doc["bars"]
        ]
        splits = [
            SplitEvent(ex_date=dt.date.fromisoformat(s["ex_date"]), ratio=s["ratio"])
            for s in doc["splits"]
        ]
        return bars, splits

    def _write_cache(
        self,
        symbol: str,
        start: dt.date,
        end: dt.date,
        bars: list[DailyBar],
        splits: list[SplitEvent],
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        doc = {
            "version": _CACHE_VERSION,
            "symbol": symbol,
            "fetched_at": dt.datetime.now(dt.UTC).isoformat(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": self.feed,
            "bars": [
                {
                    "date": b.date.isoformat(),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in bars
            ],
            "splits": [{"ex_date": s.ex_date.isoformat(), "ratio": s.ratio} for s in splits],
        }
        self._cache_path(symbol).write_text(json.dumps(doc, indent=1))
