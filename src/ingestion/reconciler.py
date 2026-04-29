"""Submissions-API reconciler (S3.4, ADR-002 §4).

The completeness safety net for E3 ingestion: every 6 hours, for each CIK
in the current universe, fetch the SEC submissions JSON
(`https://data.sec.gov/submissions/CIK{cik:010d}.json`) and diff its
recent-filings list against the journal's `filings` table. Any in-universe
filing the RSS firehose missed (S3.2 was down, polling-window gap, etc.)
is enqueued into the same detail-fetch queue (S3.3 consumer).

Bounded request budget: a single pass issues at most
`MAX_REQUESTS_PER_PASS` (= 100) submissions calls. With the universe
bounded at ~3-4k CIKs, full coverage takes ~30-40 reconciler passes worth
of distinct CIK shards over time — the per-pass cap protects the shared
EdgarClient's 10 req/sec budget against a single pass monopolizing the
reconciler interval. The shared client (ADR-002 §6) does the actual rate
limiting; the reconciler does NOT bypass it.

Failure posture (NFR-5):
- Per-CIK transport / status error → log warning, skip that CIK, keep going.
- `EdgarUnavailable` (retries exhausted) on a CIK → same: skip + continue.
- `CircuitOpen` → abort the current pass; next interval will retry.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from collections.abc import Callable, Iterable
from typing import Protocol

from ingestion.edgar_client import CircuitOpen, EdgarClient, EdgarUnavailable
from ingestion.parsers.submissions import SubmissionsEntry, parse_submissions
from ingestion.rss_loop import FORM_TYPES, DiscoveredFiling
from journal.repo import get_filing_by_accession
from observability.log_port import get_logger

DEFAULT_INTERVAL_SECONDS = 21600  # 6 hours per AC-2
DEFAULT_LOOKBACK_DAYS = 7  # AC-3
MAX_REQUESTS_PER_PASS = 100  # ADR-002 §5 + carry-forward note in #38
DEFAULT_BATCH_SIZE = 50  # dev-note "in batches of 50"

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_FORM_TYPES_SET = frozenset(FORM_TYPES)

_log = get_logger(__name__)


class _UniverseProtocol(Protocol):
    """Reconciler reads exactly two methods from `Universe`.

    Typed as a Protocol so tests can pass a small fake without spinning
    up a journal-backed `Universe` instance.
    """

    def is_in_universe_by_cik(self, cik: int) -> bool: ...
    def iter_ciks(self) -> list[int] | Iterable[int]: ...


class Reconciler:
    """6-hour submissions-API loop closing discovery-side gaps."""

    def __init__(
        self,
        *,
        edgar: EdgarClient,
        universe: _UniverseProtocol,
        queue: asyncio.Queue[DiscoveredFiling],
        db_path: str,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_requests_per_pass: int = MAX_REQUESTS_PER_PASS,
        wall_clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self._edgar = edgar
        self._universe = universe
        self._queue = queue
        self._db_path = db_path
        self._interval = interval_seconds
        self._lookback = dt.timedelta(days=lookback_days)
        self._batch_size = batch_size
        self._max_requests = max_requests_per_pass
        self._wall_clock = wall_clock or (lambda: dt.datetime.now(tz=dt.UTC))
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self.filings_recovered_total: int = 0  # AC-6 metric

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def force_reconcile_now(self) -> int:
        """Run one pass synchronously; return the recovered count.

        Used by the agent main on operator demand or after an extended
        outage; does NOT affect the periodic schedule.
        """
        return await self.reconcile_once()

    async def _run(self) -> None:
        # Wait one interval before the first pass so the discovery loop
        # has a chance to initialize and dedupe its cold-start window
        # (D-041); aligning the first pass with the cadence avoids a
        # thundering boot.
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._interval,
                )
                return
            except TimeoutError:
                pass
            try:
                await self.reconcile_once()
            except CircuitOpen:
                _log.warning("reconciler: edgar circuit open; deferring pass")
            except Exception:
                _log.exception("reconciler: unhandled error in pass")

    # ------------------------------------------------------------------
    # Reconcile pass
    # ------------------------------------------------------------------

    async def reconcile_once(self) -> int:
        """Issue submissions queries up to `max_requests_per_pass`, diff
        each against the journal, and enqueue missing accessions.

        Returns the number of filings recovered in this pass.
        """
        ciks = list(self._universe.iter_ciks())
        cutoff = self._wall_clock() - self._lookback
        recovered_this_pass = 0
        budget = self._max_requests

        for batch_start in range(0, len(ciks), self._batch_size):
            if budget <= 0:
                break
            batch = ciks[batch_start : batch_start + self._batch_size]
            for cik in batch:
                if budget <= 0:
                    break
                budget -= 1
                try:
                    entries = await self._fetch_submissions(cik)
                except CircuitOpen:
                    raise  # abort the entire pass; outer caller handles
                except EdgarUnavailable as exc:
                    _log.warning(
                        "reconciler: skipping cik=%d after EdgarUnavailable: %s",
                        cik, exc,
                    )
                    continue
                except _SkipCik as exc:
                    _log.warning("reconciler: skipping cik=%d: %s", cik, exc)
                    continue
                recovered_this_pass += await self._diff_and_enqueue(
                    cik, entries, cutoff
                )
        self.filings_recovered_total += recovered_this_pass
        if recovered_this_pass:
            _log.info(
                "reconciler pass complete; recovered %d filings",
                recovered_this_pass,
            )
        return recovered_this_pass

    async def _fetch_submissions(self, cik: int) -> list[SubmissionsEntry]:
        url = _SUBMISSIONS_URL.format(cik=cik)
        resp = await self._edgar.get(url)
        # Carry-forward S3.1 review L-1: 4xx/5xx responses are returned,
        # not raised; explicitly status-check before parsing.
        if resp.status_code != 200:
            raise _SkipCik(f"submissions returned status {resp.status_code}")
        try:
            payload = resp.json()
        except json.JSONDecodeError as e:
            raise _SkipCik(f"submissions JSON malformed: {e}") from e
        if not isinstance(payload, dict):
            raise _SkipCik("submissions payload is not an object")
        return parse_submissions(payload, cik=cik)

    async def _diff_and_enqueue(
        self,
        cik: int,
        entries: list[SubmissionsEntry],
        cutoff: dt.datetime,
    ) -> int:
        # Universe gate is implicit — we only iterate `iter_ciks()` so every
        # `cik` here is in-universe. We still re-check defensively because
        # `Universe.reload_if_newer()` could land between iter and diff
        # (architecture §10.5: single-threaded asyncio means this is a
        # cooperative-yield boundary, not a TOCTOU race).
        if not self._universe.is_in_universe_by_cik(cik):
            return 0
        recovered = 0
        for entry in entries:
            if entry.form_type not in _FORM_TYPES_SET:
                continue
            if entry.filed_at < cutoff:
                continue
            existing = get_filing_by_accession(self._db_path, entry.accession_number)
            if existing is not None:
                continue
            await self._queue.put(
                DiscoveredFiling(
                    accession_number=entry.accession_number,
                    cik=entry.cik,
                    form_type=entry.form_type,
                    filed_at=entry.filed_at,
                    discovered_at=self._wall_clock(),
                )
            )
            recovered += 1
            _log.info(
                "reconciler recovered filing",
                extra={
                    "accession": entry.accession_number,
                    "cik": entry.cik,
                    "form_type": entry.form_type,
                },
            )
        return recovered


class _SkipCik(Exception):
    """Internal: per-CIK recoverable error — log + continue with next CIK."""
