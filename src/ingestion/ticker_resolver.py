"""CIK → ticker resolver backed by SEC's `company_tickers.json`.

Production hotfix (2026-05-04): prior to this module, `detail_fetcher.py`
hard-coded `issuer_ticker=None` for every non-Form-4 path, leaving 35/44
of today's filings with NULL ticker. The analyzer's universe-summary
context-builder (`app/loop.py:_summarize_universe`) then declared every
issuer "not in current universe member list", and Sonnet correctly
refused all 36 proposals. Zero trades had ever executed.

Architecture compliance (ADR-002 §6):
  - The cold-start fetch routes through the shared `EdgarClient` so SEC's
    10 req/sec ceiling, exponential backoff, circuit breaker, and the
    declared User-Agent (NFR-17) are honored uniformly across every
    ingestion code path. This module does NOT instantiate its own
    `httpx.AsyncClient`.
  - Parsing of `company_tickers.json` is delegated to
    `universe.sec_tickers.parse_company_tickers_payload` so the wire
    format lives in exactly one place. This module is the agent-loop
    async consumer of that shared parser.

Behavior:
  - first call fetches `SEC_TICKERS_URL` via the shared `EdgarClient`
  - the parsed map is cached in memory and persisted to a local JSON
    file (`cache_path`) for warm-start re-use across processes
  - on SEC failure, the resolver falls back to the disk cache; if the
    cache is missing or unreadable, it returns None for every CIK
  - corrupt cache files are discarded and re-fetched, never crashed on

Failure semantics (task #1 acceptance criteria):
  - missing CIK → `resolve()` returns None
  - SEC fetch failure (timeout / 4xx / 5xx exhaust / DNS) → fall back to
    cache; if no cache → None; never raises
  - corrupt cache → re-fetch from SEC; if SEC also unavailable → None

Concurrency: the resolver single-flights the cold-start fetch under an
asyncio.Lock so simultaneous resolves from multiple ingestion paths
(RSS loop + reconciler) do not stampede SEC. Disk-cache writes are
atomic (tmp + replace) so a partial write cannot poison a warm-start
read for another process.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from ingestion.edgar_client import CircuitOpen, EdgarClient, EdgarUnavailable
from observability.log_port import get_logger
from universe.sec_tickers import (
    SEC_TICKERS_URL,
    parse_company_tickers_payload,
)

_log = get_logger(__name__)

# Re-exported for back-compat; canonical source is `universe.sec_tickers`.
__all__ = ["SEC_TICKERS_URL", "TickerResolver"]


class TickerResolver:
    """CIK → ticker map with SEC-backed lazy load + on-disk cache.

    `refresh_on_load` (default True): when a non-empty disk cache exists at
    construction time, the resolver still attempts a background refresh on
    the first `resolve()` call. Set False to use the cache verbatim and
    skip SEC entirely (used by the warm-start unit test and by the backfill
    script when the operator wants to avoid extra SEC calls).
    """

    def __init__(
        self,
        *,
        edgar: EdgarClient,
        cache_path: Path,
        refresh_on_load: bool = True,
    ) -> None:
        self._edgar = edgar
        self._cache_path = Path(cache_path)
        self._refresh_on_load = refresh_on_load
        self._map: dict[int, str] | None = None
        self._lock = asyncio.Lock()

    async def resolve(self, cik: int) -> str | None:
        """Return the ticker for `cik`, or None if unresolved.

        Never raises on network or cache failure.
        """
        await self._ensure_loaded()
        if self._map is None:
            return None
        return self._map.get(int(cik))

    async def _ensure_loaded(self) -> None:
        if self._map is not None:
            return
        async with self._lock:
            if self._map is not None:
                return
            disk_map = self._load_disk_cache()
            if disk_map is not None and not self._refresh_on_load:
                self._map = disk_map
                return
            sec_map = await self._fetch_from_sec()
            if sec_map is not None:
                self._map = sec_map
                self._write_disk_cache(sec_map)
                return
            # SEC failed — fall back to disk cache if available.
            if disk_map is not None:
                _log.warning(
                    "ticker_resolver.sec_unavailable_using_cache",
                    extra={
                        "event": "ticker_resolver.sec_unavailable_using_cache",
                        "cache_size": len(disk_map),
                    },
                )
                self._map = disk_map
                return
            # Neither SEC nor cache; leave _map None so resolve() returns None
            # but DO record a sentinel so we don't loop on every call.
            _log.warning(
                "ticker_resolver.sec_unavailable_no_cache",
                extra={"event": "ticker_resolver.sec_unavailable_no_cache"},
            )
            self._map = {}

    async def _fetch_from_sec(self) -> dict[int, str] | None:
        try:
            resp = await self._edgar.get(SEC_TICKERS_URL)
        except (EdgarUnavailable, CircuitOpen, httpx.HTTPError):
            return None
        except Exception:  # noqa: BLE001 — defensive at the SEC boundary
            _log.exception("ticker_resolver.unexpected_fetch_error")
            return None
        if resp.status_code != 200:
            _log.warning(
                "ticker_resolver.sec_non_200",
                extra={
                    "event": "ticker_resolver.sec_non_200",
                    "status_code": resp.status_code,
                },
            )
            return None
        try:
            payload = resp.json()
        except (json.JSONDecodeError, ValueError):
            _log.warning("ticker_resolver.sec_payload_not_json")
            return None
        try:
            entries = parse_company_tickers_payload(payload)
        except ValueError:
            # Wrong shape (e.g., HTML maintenance page returned with 200).
            # Don't poison the cache; treat as a fetch failure so the
            # disk-cache fallback path runs.
            _log.warning("ticker_resolver.sec_payload_malformed")
            return None
        out = {e.cik: e.ticker for e in entries}
        return out if out else None

    def _load_disk_cache(self) -> dict[int, str] | None:
        if not self._cache_path.exists():
            return None
        try:
            raw = self._cache_path.read_text()
        except OSError:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            _log.warning(
                "ticker_resolver.cache_corrupt_will_refetch",
                extra={"event": "ticker_resolver.cache_corrupt_will_refetch"},
            )
            return None
        if not isinstance(payload, dict):
            return None
        out: dict[int, str] = {}
        for k, v in payload.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        return out if out else None

    def _write_disk_cache(self, m: dict[int, str]) -> None:
        # Atomic write: dump to <path>.tmp then os.replace().
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            payload = {str(cik): ticker for cik, ticker in m.items()}
            tmp.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
            tmp.replace(self._cache_path)
        except OSError:
            _log.warning(
                "ticker_resolver.cache_write_failed",
                extra={
                    "event": "ticker_resolver.cache_write_failed",
                    "cache_path": str(self._cache_path),
                },
            )


