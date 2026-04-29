"""S3.2 — RSS discovery loop tests.

Tests use `httpx.MockTransport` for the EDGAR client, an in-memory fake
universe, an in-process `asyncio.Queue` as the detail-fetch sink, and a
fake clock so cadence and latency assertions are deterministic.

ADR-002 §1–§3 govern the design; the two scenarios this story owns are:
- "out-of-universe filing dropped at universe gate with no detail fetch"
- "new 8-K item 2.02 ingested within 90 s of publication" (discovery
  component).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path

import httpx
import pytest

from ingestion.edgar_client import EdgarClient
from ingestion.rss_loop import (
    FORM_TYPES,
    DiscoveredFiling,
    RssDiscoveryLoop,
)

_UA = "Quinn-Research/v1 contact@operator.example"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeUniverse:
    def __init__(self, ciks: set[int]) -> None:
        self._ciks = ciks
        self.lookups: list[int] = []

    def is_in_universe_by_cik(self, cik: int) -> bool:
        self.lookups.append(cik)
        return cik in self._ciks


def _atom(entries: list[dict[str, str]]) -> bytes:
    """Build a minimal Atom feed body from a list of {accession, cik, form, updated}."""
    items = []
    for e in entries:
        items.append(
            f"""  <entry>
    <title>{e["form"]} - ISSUER ({int(e["cik"]):010d})</title>
    <updated>{e["updated"]}</updated>
    <category term="{e["form"]}"/>
    <id>urn:tag:sec.gov,2008:accession-number={e["accession"]}</id>
  </entry>
"""
        )
    body = (
        '<?xml version="1.0"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        "  <title>EDGAR Latest</title>\n"
        + "".join(items)
        + "</feed>\n"
    )
    return body.encode("utf-8")


class _ScriptedTransport:
    """Transport whose response per form-type is pulled from a queue.

    For each form type we keep an ordered list of feed bodies; each call
    pops the next one (or repeats the last). This lets a test stage what
    the firehose looks like across successive polls.
    """

    def __init__(self, per_form: dict[str, list[bytes]]) -> None:
        self._per_form = {k: list(v) for k, v in per_form.items()}
        self._idx: dict[str, int] = dict.fromkeys(per_form, 0)
        self.requests: list[httpx.Request] = []

    def as_mock(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            form = request.url.params.get("type", "")
            bodies = self._per_form.get(form, [])
            if not bodies:
                return httpx.Response(200, content=_atom([]))
            i = min(self._idx[form], len(bodies) - 1)
            self._idx[form] += 1
            return httpx.Response(
                200,
                content=bodies[i],
                headers={"content-type": "application/atom+xml"},
            )

        return httpx.MockTransport(handle)


class _FakeClock:
    """Wall-clock fake. The loop reads `now()` only for market-hours; we
    don't need to fake monotonic time because we drive the loop with
    explicit `tick_once()` calls rather than waiting on real sleeps."""

    def __init__(self, start: dt.datetime) -> None:
        self.t = start

    def now(self) -> dt.datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = self.t + dt.timedelta(seconds=seconds)


def _client(transport: httpx.MockTransport) -> EdgarClient:
    return EdgarClient(user_agent=_UA, transport=transport, retry_base_seconds=0.0)


# ---------------------------------------------------------------------------
# AC-2: poll URLs cover the six form types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_form_types_cover_six(tmp_path: Path) -> None:
    expected = {"10-K", "10-Q", "8-K", "S-1", "DEF 14A", "4"}
    assert set(FORM_TYPES) == expected


# ---------------------------------------------------------------------------
# AC-4 + AC-7: out-of-universe entries are dropped before any detail-fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_out_of_universe_dropped(tmp_path: Path) -> None:
    transport = _ScriptedTransport(
        {
            "8-K": [
                _atom(
                    [
                        {
                            "accession": "0001234567-26-000099",
                            "cik": "1234567",
                            "form": "8-K",
                            "updated": "2026-04-29T13:55:00-04:00",
                        },
                        {
                            "accession": "0009999999-26-000001",
                            "cik": "9999999",
                            "form": "8-K",
                            "updated": "2026-04-29T13:56:00-04:00",
                        },
                    ]
                )
            ]
        }
    ).as_mock()
    universe = _FakeUniverse(ciks={1234567})  # 9999999 is OOU
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    cursor = tmp_path / "rss_cursor.json"
    # Pre-seed the cursor so this is a "warm start" — new entries should
    # flow normally rather than be swallowed by cold-start seeding.
    cursor.write_text(json.dumps({"8-K": []}))
    edgar = _client(transport)
    loop = RssDiscoveryLoop(
        edgar=edgar,
        universe=universe,
        queue=queue,
        cursor_path=cursor,
        poll_market_seconds=60,
        poll_offhours_seconds=300,
        form_types=("8-K",),
    )
    try:
        await loop.tick_once()
    finally:
        await edgar.aclose()
    assert queue.qsize() == 1
    item = queue.get_nowait()
    assert item.cik == 1234567
    # Universe was consulted for both entries.
    assert sorted(universe.lookups) == [1234567, 9999999]


# ---------------------------------------------------------------------------
# AC-5: in-universe entries are queued
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_entry_queued(tmp_path: Path) -> None:
    transport = _ScriptedTransport(
        {
            "8-K": [
                _atom(
                    [
                        {
                            "accession": "0001234567-26-000099",
                            "cik": "1234567",
                            "form": "8-K",
                            "updated": "2026-04-29T13:55:00-04:00",
                        }
                    ]
                )
            ]
        }
    ).as_mock()
    universe = _FakeUniverse(ciks={1234567})
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    cursor = tmp_path / "rss_cursor.json"
    cursor.write_text(json.dumps({"8-K": []}))  # warm start
    edgar = _client(transport)
    loop = RssDiscoveryLoop(
        edgar=edgar,
        universe=universe,
        queue=queue,
        cursor_path=cursor,
        poll_market_seconds=60,
        poll_offhours_seconds=300,
        form_types=("8-K",),
    )
    try:
        await loop.tick_once()
    finally:
        await edgar.aclose()
    item = queue.get_nowait()
    assert item.accession_number == "0001234567-26-000099"
    assert item.form_type == "8-K"
    assert item.cik == 1234567


# ---------------------------------------------------------------------------
# AC-6: cursor persistence — second tick does not re-emit prior entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cursor_persistence_round_trip(tmp_path: Path) -> None:
    cursor = tmp_path / "rss_cursor.json"
    poll1 = _atom(
        [
            {
                "accession": "0001234567-26-000099",
                "cik": "1234567",
                "form": "8-K",
                "updated": "2026-04-29T13:55:00-04:00",
            },
            {
                "accession": "0001234567-26-000100",
                "cik": "1234567",
                "form": "8-K",
                "updated": "2026-04-29T13:56:00-04:00",
            },
        ]
    )
    # Second poll repeats the same two entries plus a new one.
    poll2 = _atom(
        [
            {
                "accession": "0001234567-26-000099",
                "cik": "1234567",
                "form": "8-K",
                "updated": "2026-04-29T13:55:00-04:00",
            },
            {
                "accession": "0001234567-26-000100",
                "cik": "1234567",
                "form": "8-K",
                "updated": "2026-04-29T13:56:00-04:00",
            },
            {
                "accession": "0001234567-26-000101",
                "cik": "1234567",
                "form": "8-K",
                "updated": "2026-04-29T13:57:30-04:00",
            },
        ]
    )

    transport_factory = _ScriptedTransport({"8-K": [poll1]}).as_mock()
    universe = _FakeUniverse(ciks={1234567})
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    # Warm-start: seed the cursor file so the run-1 entries flow through.
    cursor.write_text(json.dumps({"8-K": []}))

    # Run 1: see two entries, persist on shutdown.
    edgar1 = _client(transport_factory)
    loop1 = RssDiscoveryLoop(
        edgar=edgar1,
        universe=universe,
        queue=queue,
        cursor_path=cursor,
        poll_market_seconds=60,
        poll_offhours_seconds=300,
        form_types=("8-K",),
    )
    await loop1.tick_once()
    await loop1.persist_cursor()
    await edgar1.aclose()
    assert queue.qsize() == 2
    while not queue.empty():
        queue.get_nowait()

    # Cursor file must exist and be JSON.
    assert cursor.exists()
    cur = json.loads(cursor.read_text())
    assert "8-K" in cur

    # Run 2: poll returns the same two PLUS a third — only the third is queued.
    transport2 = _ScriptedTransport({"8-K": [poll2]}).as_mock()
    edgar2 = _client(transport2)
    loop2 = RssDiscoveryLoop(
        edgar=edgar2,
        universe=universe,
        queue=queue,
        cursor_path=cursor,
        poll_market_seconds=60,
        poll_offhours_seconds=300,
        form_types=("8-K",),
    )
    await loop2.tick_once()
    await edgar2.aclose()
    assert queue.qsize() == 1
    new_one = queue.get_nowait()
    assert new_one.accession_number == "0001234567-26-000101"


@pytest.mark.asyncio
async def test_no_cursor_starts_at_latest(tmp_path: Path) -> None:
    # AC-6: cursor file missing → start from most recent entry; do NOT
    # back-fill via RSS. We assert this by verifying that after a fresh
    # start with a non-empty feed, no entries are queued, and the cursor
    # is then advanced so subsequent polls yield only newer entries.
    poll1 = _atom(
        [
            {
                "accession": "0001234567-26-000099",
                "cik": "1234567",
                "form": "8-K",
                "updated": "2026-04-29T13:55:00-04:00",
            },
            {
                "accession": "0001234567-26-000100",
                "cik": "1234567",
                "form": "8-K",
                "updated": "2026-04-29T13:56:00-04:00",
            },
        ]
    )
    poll2 = _atom(
        [
            {
                "accession": "0001234567-26-000099",
                "cik": "1234567",
                "form": "8-K",
                "updated": "2026-04-29T13:55:00-04:00",
            },
            {
                "accession": "0001234567-26-000100",
                "cik": "1234567",
                "form": "8-K",
                "updated": "2026-04-29T13:56:00-04:00",
            },
            {
                "accession": "0001234567-26-000101",
                "cik": "1234567",
                "form": "8-K",
                "updated": "2026-04-29T13:57:30-04:00",
            },
        ]
    )
    transport = _ScriptedTransport({"8-K": [poll1, poll2]}).as_mock()
    universe = _FakeUniverse(ciks={1234567})
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    cursor = tmp_path / "rss_cursor.json"
    assert not cursor.exists()
    edgar = _client(transport)
    loop = RssDiscoveryLoop(
        edgar=edgar,
        universe=universe,
        queue=queue,
        cursor_path=cursor,
        poll_market_seconds=60,
        poll_offhours_seconds=300,
        form_types=("8-K",),
    )
    try:
        await loop.tick_once()  # initial — establishes cursor, queues nothing
        assert queue.qsize() == 0
        await loop.tick_once()  # second — only the new accession is queued
        assert queue.qsize() == 1
        item = queue.get_nowait()
        assert item.accession_number == "0001234567-26-000101"
    finally:
        await edgar.aclose()


# ---------------------------------------------------------------------------
# AC-3: cadence — market vs off-hours
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_vs_offhours_cadence(tmp_path: Path) -> None:
    universe = _FakeUniverse(ciks=set())
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    transport = _ScriptedTransport({"8-K": [_atom([])]}).as_mock()
    edgar = _client(transport)
    market_clock = _FakeClock(dt.datetime(2026, 4, 29, 11, 0))
    offhours_clock = _FakeClock(dt.datetime(2026, 4, 29, 22, 0))
    try:
        loop = RssDiscoveryLoop(
            edgar=edgar,
            universe=universe,
            queue=queue,
            cursor_path=tmp_path / "rss_cursor.json",
            poll_market_seconds=60,
            poll_offhours_seconds=300,
            form_types=("8-K",),
            wall_clock=market_clock.now,
        )
        assert loop.next_sleep_seconds() == 60

        loop2 = RssDiscoveryLoop(
            edgar=edgar,
            universe=universe,
            queue=queue,
            cursor_path=tmp_path / "rss_cursor.json",
            poll_market_seconds=60,
            poll_offhours_seconds=300,
            form_types=("8-K",),
            wall_clock=offhours_clock.now,
        )
        assert loop2.next_sleep_seconds() == 300

        # Saturday → off-hours
        weekend_clock = _FakeClock(dt.datetime(2026, 5, 2, 11, 0))
        loop3 = RssDiscoveryLoop(
            edgar=edgar,
            universe=universe,
            queue=queue,
            cursor_path=tmp_path / "rss_cursor.json",
            poll_market_seconds=60,
            poll_offhours_seconds=300,
            form_types=("8-K",),
            wall_clock=weekend_clock.now,
        )
        assert loop3.next_sleep_seconds() == 300
    finally:
        await edgar.aclose()


# ---------------------------------------------------------------------------
# AC-8: discovery latency under 90s p95 with 60s poll cadence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_latency_under_90s_p95_with_60s_poll(tmp_path: Path) -> None:
    """Synthetic EDGAR feed where each successive poll exposes one new entry,
    with a publish time that lags `now` by a uniformly random amount in
    [0, 60s] (modelling the polling-window phase). Run 50 polls; assert p95
    of (now - publish_time) is under 90s."""

    universe = _FakeUniverse(ciks={1234567})
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()

    n_polls = 50
    poll_interval_s = 60.0
    # Wall-clock start.
    start = dt.datetime(2026, 4, 29, 9, 30, tzinfo=dt.UTC)
    # Generate `n_polls` publish times, each landing somewhere inside the
    # window of its corresponding poll.
    rng_state = [0.13, 0.91, 0.27, 0.75, 0.04, 0.58, 0.42, 0.88, 0.31, 0.66] * 5
    publish_times: list[dt.datetime] = []
    accessions: list[str] = []
    for i in range(n_polls):
        # poll i happens at start + i*60s; entry was published a fraction
        # of one window earlier.
        frac = rng_state[i % len(rng_state)]
        offset_s = i * poll_interval_s - frac * poll_interval_s
        publish_times.append(start + dt.timedelta(seconds=offset_s))
        accessions.append(f"0001234567-26-{i:06d}")

    # Build per-poll feeds: each poll's feed contains all entries with
    # publish_time <= now (cumulative; matches real EDGAR which serves a
    # rolling-latest list).
    per_form_polls: list[bytes] = []
    for i in range(n_polls):
        now_i = start + dt.timedelta(seconds=i * poll_interval_s)
        visible = [
            {
                "accession": accessions[j],
                "cik": "1234567",
                "form": "8-K",
                "updated": publish_times[j].astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            }
            for j in range(n_polls)
            if publish_times[j] <= now_i
        ]
        per_form_polls.append(_atom(visible[-40:]))  # mimic 40-entry feed cap

    transport = _ScriptedTransport({"8-K": per_form_polls}).as_mock()
    edgar = _client(transport)
    loop = RssDiscoveryLoop(
        edgar=edgar,
        universe=universe,
        queue=queue,
        cursor_path=tmp_path / "rss_cursor.json",
        poll_market_seconds=60,
        poll_offhours_seconds=300,
        form_types=("8-K",),
    )

    queue_times: dict[str, dt.datetime] = {}
    try:
        for i in range(n_polls):
            now_i = start + dt.timedelta(seconds=i * poll_interval_s)
            # Inject `now` as the time the loop "sees" by passing wall_clock
            # captures; here we simply record the queue-time as `now_i`
            # since the only purpose is latency math.
            await loop.tick_once()
            while not queue.empty():
                item = queue.get_nowait()
                queue_times.setdefault(item.accession_number, now_i)
    finally:
        await edgar.aclose()

    latencies_s: list[float] = []
    for j, acc in enumerate(accessions):
        if acc in queue_times:
            latencies_s.append((queue_times[acc] - publish_times[j]).total_seconds())
    assert len(latencies_s) >= int(n_polls * 0.95), (
        f"too many entries lost: {len(latencies_s)}/{n_polls}"
    )
    latencies_s.sort()
    p95 = latencies_s[int(0.95 * len(latencies_s))]
    assert p95 < 90.0, f"discovery p95 latency {p95:.1f}s exceeds 90s NFR-1 budget"


# ---------------------------------------------------------------------------
# AC-1: start/stop lifecycle drains nothing into the queue when no entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_stop_lifecycle(tmp_path: Path) -> None:
    transport = _ScriptedTransport({form: [_atom([])] for form in FORM_TYPES}).as_mock()
    universe = _FakeUniverse(ciks=set())
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    edgar = _client(transport)
    loop = RssDiscoveryLoop(
        edgar=edgar,
        universe=universe,
        queue=queue,
        cursor_path=tmp_path / "rss_cursor.json",
        poll_market_seconds=60,
        poll_offhours_seconds=300,
    )
    try:
        await loop.start()
        # Give the loop a moment to do at least one tick.
        await asyncio.sleep(0.05)
        await loop.stop()
    finally:
        await edgar.aclose()
    # No entries to queue (empty universe + empty feed).
    assert queue.qsize() == 0
    # Cursor was persisted on stop.
    assert (tmp_path / "rss_cursor.json").exists()


# ---------------------------------------------------------------------------
# Universe gate is consulted BEFORE any detail fetch (sanity: this story
# only enqueues; S3.3 fetches. We assert the gate runs and out-of-universe
# CIKs are not enqueued — the "no detail fetch" half of AC-7).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_universe_gate_consulted_before_queue(tmp_path: Path) -> None:
    transport = _ScriptedTransport(
        {
            "8-K": [
                _atom(
                    [
                        {
                            "accession": "0009999999-26-000001",
                            "cik": "9999999",
                            "form": "8-K",
                            "updated": "2026-04-29T13:55:00-04:00",
                        }
                    ]
                )
            ]
        }
    ).as_mock()
    universe = _FakeUniverse(ciks={1234567})  # 9999999 is OOU
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    cursor = tmp_path / "rss_cursor.json"
    cursor.write_text(json.dumps({"8-K": []}))  # warm start
    edgar = _client(transport)
    loop = RssDiscoveryLoop(
        edgar=edgar,
        universe=universe,
        queue=queue,
        cursor_path=cursor,
        poll_market_seconds=60,
        poll_offhours_seconds=300,
        form_types=("8-K",),
    )
    try:
        await loop.tick_once()
    finally:
        await edgar.aclose()
    assert universe.lookups == [9999999]
    assert queue.qsize() == 0
    # No journal write happened — this is enforced by the loop having no
    # journal dependency at all (verified by import: `import ingestion.rss_loop`
    # does not touch `journal.repo`).
    import ingestion.rss_loop as _mod  # noqa: F401, PLC0415

    src = Path(_mod.__file__).read_text()
    # Defensive: the loop should not import journal.repo. (FR-11 spirit.)
    assert "journal.repo" not in src
