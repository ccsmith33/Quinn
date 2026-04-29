"""RSS discovery loop (S3.2, ADR-002 §1–§3).

The high-frequency discovery channel: poll EDGAR full-text Atom feeds for
the six v1 form types (10-K, 10-Q, 8-K, S-1, DEF 14A, Form 4) every 60s
during NYSE market hours and every 5min off-hours. Apply the universe
gate (FR-11) per discovered accession; in-universe entries are pushed to
an in-process `asyncio.Queue` consumed by the detail fetcher (S3.3).

This module deliberately does NOT touch the journal — out-of-universe
filings must be dropped before any DB write (FR-11 spirit). Persistence
of fetched filings is S3.3's responsibility.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from config.calendar import is_market_hours
from ingestion.edgar_client import CircuitOpen, EdgarClient, EdgarUnavailable
from ingestion.rss_parsers import RssEntry, parse_atom
from observability.log_port import get_logger

# Six v1 form types per ADR-002 §1 / FR-1..FR-3.
FORM_TYPES: tuple[str, ...] = ("10-K", "10-Q", "8-K", "S-1", "DEF 14A", "4")

_RSS_BASE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
# Bound the per-form-type seen-set so the cursor file stays small. EDGAR
# RSS returns at most 40 entries per call by default; 1000 covers many
# polls of memory and is well under typical filesystem write cost.
_CURSOR_RING_SIZE = 1000

_log = get_logger(__name__)


class _UniverseProtocol(Protocol):
    """The narrow surface RssDiscoveryLoop needs from `Universe`.

    Intentionally typed as a `Protocol` (not `universe.api.Universe`) so
    tests can pass a small fake without spinning up a journal DB.
    """

    def is_in_universe_by_cik(self, cik: int) -> bool:
        ...


class DiscoveredFiling(BaseModel):
    """Sink-queue payload. S3.3 consumes this and fetches the primary doc."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accession_number: str
    cik: int
    form_type: str
    filed_at: dt.datetime
    discovered_at: dt.datetime  # wall-clock when we observed it (NFR-1 latency)


class RssDiscoveryLoop:
    """Async lifecycle: `start()` schedules a background task that polls the
    six form-type feeds at the cadence dictated by `is_market_hours(now())`;
    `stop()` cancels and persists the cursor.

    Tests typically call `tick_once()` directly to skip the sleep loop.
    """

    def __init__(
        self,
        *,
        edgar: EdgarClient,
        universe: _UniverseProtocol,
        queue: asyncio.Queue[DiscoveredFiling],
        cursor_path: Path,
        poll_market_seconds: int,
        poll_offhours_seconds: int,
        form_types: Iterable[str] = FORM_TYPES,
        wall_clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self._edgar = edgar
        self._universe = universe
        self._queue = queue
        self._cursor_path = Path(cursor_path)
        self._poll_market = poll_market_seconds
        self._poll_offhours = poll_offhours_seconds
        self._form_types = tuple(form_types)
        self._wall_clock = wall_clock or (lambda: dt.datetime.now(tz=dt.UTC))
        # Per-form-type ring of seen accessions. Implemented as a list +
        # set: list preserves insertion order for ring eviction; set gives
        # O(1) membership.
        self._seen_order: dict[str, list[str]] = {f: [] for f in self._form_types}
        self._seen: dict[str, set[str]] = {f: set() for f in self._form_types}
        self._loaded_cursor = self._load_cursor()
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

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
        await self.persist_cursor()

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.tick_once()
            except CircuitOpen:
                # AC-5 contract from S3.1: caller skips the cycle.
                _log.warning("rss_loop: edgar circuit open; skipping cycle")
            except Exception:
                _log.exception("rss_loop: unhandled error in tick")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.next_sleep_seconds(),
                )
            except TimeoutError:
                continue

    def next_sleep_seconds(self) -> int:
        """Cadence selector — 60s during NYSE session, 5min otherwise."""
        return (
            self._poll_market
            if is_market_hours(self._wall_clock())
            else self._poll_offhours
        )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def tick_once(self) -> None:
        for form_type in self._form_types:
            try:
                body = await self._edgar.get_bytes(
                    _RSS_BASE_URL,
                    params={
                        "action": "getcurrent",
                        "type": form_type,
                        "output": "atom",
                    },
                )
            except EdgarUnavailable as exc:
                _log.warning(
                    "rss_loop: edgar unavailable for %s (%s); will retry next cycle",
                    form_type,
                    exc,
                )
                continue
            try:
                entries = parse_atom(body)
            except Exception:
                _log.exception("rss_loop: failed to parse %s feed", form_type)
                continue
            await self._dispatch(form_type, entries)

    async def _dispatch(self, form_type: str, entries: list[RssEntry]) -> None:
        # Snapshot whether this is a fresh start for this form type — if so,
        # we must NOT back-fill via RSS (AC-6). We seed the seen-set with
        # everything in the first poll and queue nothing.
        first_poll = (
            form_type not in self._loaded_cursor
            and not self._seen[form_type]
        )
        for entry in entries:
            if entry.accession_number in self._seen[form_type]:
                continue
            self._mark_seen(form_type, entry.accession_number)
            if first_poll:
                continue
            if not self._universe.is_in_universe_by_cik(entry.cik):
                _log.debug(
                    "rss_loop: dropped out-of-universe filing cik=%d acc=%s form=%s",
                    entry.cik,
                    entry.accession_number,
                    entry.form_type,
                )
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
        if first_poll:
            # Mark this form type as "seeded" so subsequent ticks treat new
            # accessions normally.
            self._loaded_cursor[form_type] = True

    def _mark_seen(self, form_type: str, accession: str) -> None:
        self._seen[form_type].add(accession)
        order = self._seen_order[form_type]
        order.append(accession)
        if len(order) > _CURSOR_RING_SIZE:
            evicted = order.pop(0)
            self._seen[form_type].discard(evicted)

    # ------------------------------------------------------------------
    # Cursor persistence
    # ------------------------------------------------------------------

    def _load_cursor(self) -> dict[str, bool]:
        """Returns a per-form-type "seeded" flag map. Side effect: hydrates
        `self._seen` / `self._seen_order` from disk so previously-seen
        accessions are not re-emitted across restarts.
        """
        seeded: dict[str, bool] = {}
        if not self._cursor_path.exists():
            return seeded
        try:
            raw = json.loads(self._cursor_path.read_text())
        except json.JSONDecodeError:
            _log.warning(
                "rss_loop: cursor file %s is corrupt; ignoring",
                self._cursor_path,
            )
            return seeded
        for form_type, accessions in raw.items():
            if form_type not in self._seen:
                continue
            for acc in accessions:
                self._mark_seen(form_type, acc)
            # The presence of the form-type key in the cursor file marks
            # this form type as "seeded" — even if its accession list is
            # empty (legitimately possible after a long quiet period that
            # aged out of the ring buffer). Without this, a restart with
            # an empty list would re-seed and silently drop the next poll.
            seeded[form_type] = True
        return seeded

    async def persist_cursor(self) -> None:
        payload = {f: list(self._seen_order[f]) for f in self._form_types}
        self._cursor_path.parent.mkdir(parents=True, exist_ok=True)
        self._cursor_path.write_text(json.dumps(payload, indent=0))
