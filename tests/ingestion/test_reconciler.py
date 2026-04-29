"""S3.4 — submissions-API reconciler tests.

ADR-002 §4: every 6 hours, for each in-universe CIK, fetch the submissions
JSON and queue any in-universe filings that are NOT already in `filings`.
The two ADR-002 scenarios this story owns:
- "Discovery loop killed for 30 min → reconciler picks up the gap"
- "EDGAR 429 → backoff + reconciler eventually succeeds"
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path

import httpx
import pytest

from ingestion.edgar_client import EdgarClient
from ingestion.reconciler import (
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_LOOKBACK_DAYS,
    MAX_REQUESTS_PER_PASS,
    Reconciler,
)
from ingestion.rss_loop import DiscoveredFiling
from journal.migrate import apply_migrations
from journal.models import FilingRow
from journal.repo import insert_filing

_UA = "Quinn-Research/v1 contact@operator.example"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return str(db_path)


def _seed_filing(
    db_path: str,
    *,
    accession: str,
    cik: int,
    form_type: str = "8-K",
    filed_at: dt.datetime | None = None,
) -> None:
    row = FilingRow(
        accession_number=accession,
        cik=cik,
        form_type=form_type,
        filed_at=filed_at or dt.datetime(2026, 4, 28, 12, 0, tzinfo=dt.UTC),
        fetched_at=dt.datetime(2026, 4, 28, 12, 1, tzinfo=dt.UTC),
        raw_text_path=f"/var/lib/quinn/raw/{accession}.txt",
        content_hash="seed",
        item_codes=None,
        issuer_ticker=None,
        ingest_state="ok",
        ingest_error=None,
    )
    insert_filing(db_path, row)


def _submissions(
    *,
    accessions: list[tuple[str, str, str]],  # (accession, form, filingDate)
) -> bytes:
    return json.dumps(
        {
            "cik": "1234567",
            "filings": {
                "recent": {
                    "accessionNumber": [a[0] for a in accessions],
                    "form": [a[1] for a in accessions],
                    "filingDate": [a[2] for a in accessions],
                    "primaryDocument": [
                        f"{a[0]}-primary.htm" for a in accessions
                    ],
                },
                "files": [],
            },
        }
    ).encode()


class _FakeUniverse:
    def __init__(self, ciks: list[int]) -> None:
        self._ciks = list(ciks)

    def is_in_universe_by_cik(self, cik: int) -> bool:
        return cik in self._ciks

    def iter_ciks(self) -> list[int]:
        return list(self._ciks)


def _route_submissions(
    *,
    cik_to_body: dict[int, bytes],
    request_log: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request_log is not None:
            request_log.append(request)
        url = str(request.url)
        for cik, body in cik_to_body.items():
            tag = f"CIK{cik:010d}.json"
            if tag in url:
                return httpx.Response(200, content=body)
        return httpx.Response(404, text=f"unhandled {url}")

    return httpx.MockTransport(handle)


def _client(transport: httpx.MockTransport) -> EdgarClient:
    return EdgarClient(user_agent=_UA, transport=transport, retry_base_seconds=0.0)


def _today_minus(days: int) -> str:
    """ISO date `days` days before today (lookback-window-aware fixtures)."""
    return (dt.date.today() - dt.timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# AC-1 / AC-3: diff finds missing accession; already-present is not re-queued
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diff_finds_missing_accession(db: str) -> None:
    # Universe has one CIK; 5 of 6 submissions are already in `filings`,
    # the 6th must be queued.
    cik = 1234567
    seeded = [
        ("0001234567-26-000001", "8-K", _today_minus(5)),
        ("0001234567-26-000002", "8-K", _today_minus(4)),
        ("0001234567-26-000003", "10-Q", _today_minus(3)),
        ("0001234567-26-000004", "8-K", _today_minus(2)),
        ("0001234567-26-000005", "8-K", _today_minus(1)),
    ]
    for acc, form, date in seeded:
        _seed_filing(
            db,
            accession=acc,
            cik=cik,
            form_type=form,
            filed_at=dt.datetime.fromisoformat(f"{date}T12:00:00+00:00"),
        )
    new_one = ("0001234567-26-000006", "8-K", _today_minus(0))
    body = _submissions(accessions=seeded + [new_one])
    transport = _route_submissions(cik_to_body={cik: body})
    universe = _FakeUniverse([cik])
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    edgar = _client(transport)
    rec = Reconciler(
        edgar=edgar,
        universe=universe,
        queue=queue,
        db_path=db,
    )
    try:
        recovered = await rec.reconcile_once()
    finally:
        await edgar.aclose()
    assert recovered == 1
    assert queue.qsize() == 1
    item = queue.get_nowait()
    assert item.accession_number == "0001234567-26-000006"
    assert item.cik == cik
    assert item.form_type == "8-K"


@pytest.mark.asyncio
async def test_already_present_not_re_queued(db: str) -> None:
    cik = 1234567
    accession = "0001234567-26-000001"
    _seed_filing(db, accession=accession, cik=cik, form_type="8-K")
    body = _submissions(accessions=[(accession, "8-K", _today_minus(0))])
    transport = _route_submissions(cik_to_body={cik: body})
    universe = _FakeUniverse([cik])
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    edgar = _client(transport)
    rec = Reconciler(edgar=edgar, universe=universe, queue=queue, db_path=db)
    try:
        recovered = await rec.reconcile_once()
    finally:
        await edgar.aclose()
    assert recovered == 0
    assert queue.empty()


# ---------------------------------------------------------------------------
# AC-3: only 6 v1 form types are reconciled; older-than-lookback is skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_v1_forms_skipped(db: str) -> None:
    cik = 1234567
    body = _submissions(
        accessions=[
            ("0001234567-26-000020", "SC 13G", _today_minus(0)),  # not in v1 set
            ("0001234567-26-000021", "424B5", _today_minus(0)),  # not in v1 set
        ]
    )
    transport = _route_submissions(cik_to_body={cik: body})
    universe = _FakeUniverse([cik])
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    edgar = _client(transport)
    rec = Reconciler(edgar=edgar, universe=universe, queue=queue, db_path=db)
    try:
        recovered = await rec.reconcile_once()
    finally:
        await edgar.aclose()
    assert recovered == 0
    assert queue.empty()


@pytest.mark.asyncio
async def test_outside_lookback_skipped(db: str) -> None:
    cik = 1234567
    # Filing is 30 days old; default lookback is 7.
    body = _submissions(
        accessions=[("0001234567-26-000030", "8-K", _today_minus(30))]
    )
    transport = _route_submissions(cik_to_body={cik: body})
    universe = _FakeUniverse([cik])
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    edgar = _client(transport)
    rec = Reconciler(edgar=edgar, universe=universe, queue=queue, db_path=db)
    try:
        recovered = await rec.reconcile_once()
    finally:
        await edgar.aclose()
    assert recovered == 0


# ---------------------------------------------------------------------------
# AC-3: only universe CIKs are queried (universe gate runs BEFORE submissions
# fetch — the loop iterates `universe.iter_ciks()` directly).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_out_of_universe_cik(db: str) -> None:
    in_cik = 1234567
    out_cik = 9999999
    request_log: list[httpx.Request] = []
    body = _submissions(
        accessions=[("0001234567-26-000040", "8-K", _today_minus(0))]
    )
    transport = _route_submissions(
        cik_to_body={in_cik: body},
        request_log=request_log,
    )
    universe = _FakeUniverse([in_cik])  # out_cik is NOT in the universe
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    edgar = _client(transport)
    rec = Reconciler(edgar=edgar, universe=universe, queue=queue, db_path=db)
    try:
        await rec.reconcile_once()
    finally:
        await edgar.aclose()
    # Exactly one CIK was queried — the in-universe one.
    assert len(request_log) == 1
    assert f"CIK{in_cik:010d}.json" in str(request_log[0].url)
    assert f"CIK{out_cik:010d}.json" not in str(request_log[0].url)


# ---------------------------------------------------------------------------
# AC-4: ADR-002 scenario — discovery loop killed for 30 min → reconciler picks
# up filings published during the gap on its next pass.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconciler_recovers_after_discovery_gap(db: str) -> None:
    cik = 1234567
    # Pre-gap: this filing was already journaled by the live discovery loop.
    _seed_filing(
        db,
        accession="0001234567-26-000100",
        cik=cik,
        form_type="8-K",
        filed_at=dt.datetime.fromisoformat(f"{_today_minus(0)}T08:00:00+00:00"),
    )
    # During the 30-min gap, two new filings landed at EDGAR — the live RSS
    # loop missed them; submissions-API now reflects them.
    body = _submissions(
        accessions=[
            ("0001234567-26-000100", "8-K", _today_minus(0)),
            ("0001234567-26-000101", "8-K", _today_minus(0)),  # missed during gap
            ("0001234567-26-000102", "10-Q", _today_minus(0)),  # missed during gap
        ]
    )
    transport = _route_submissions(cik_to_body={cik: body})
    universe = _FakeUniverse([cik])
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    edgar = _client(transport)
    rec = Reconciler(edgar=edgar, universe=universe, queue=queue, db_path=db)
    try:
        recovered = await rec.reconcile_once()
    finally:
        await edgar.aclose()
    assert recovered == 2
    queued: list[DiscoveredFiling] = []
    while not queue.empty():
        queued.append(queue.get_nowait())
    queued_accessions = {q.accession_number for q in queued}
    assert queued_accessions == {
        "0001234567-26-000101",
        "0001234567-26-000102",
    }


# ---------------------------------------------------------------------------
# AC-5: bounded request budget per pass — reconciler caps at MAX_REQUESTS_PER_PASS
# even with a larger universe; defers to shared EdgarClient rate limit.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_budget_bounded(db: str) -> None:
    # Universe with 3 × budget CIKs; reconciler should issue exactly
    # MAX_REQUESTS_PER_PASS submissions queries in one pass.
    over_budget = MAX_REQUESTS_PER_PASS * 3
    ciks = list(range(1, over_budget + 1))
    cik_to_body = {
        cik: _submissions(accessions=[]) for cik in ciks
    }
    request_log: list[httpx.Request] = []
    transport = _route_submissions(
        cik_to_body=cik_to_body,
        request_log=request_log,
    )
    universe = _FakeUniverse(ciks)
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    edgar = _client(transport)
    rec = Reconciler(edgar=edgar, universe=universe, queue=queue, db_path=db)
    try:
        await rec.reconcile_once()
    finally:
        await edgar.aclose()
    assert len(request_log) == MAX_REQUESTS_PER_PASS


# ---------------------------------------------------------------------------
# AC-5: per-CIK transport failure does not kill the pass.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_cik_failure_does_not_block_pass(db: str) -> None:
    cik_ok = 1111111
    cik_fail = 2222222
    cik_ok_2 = 3333333

    body_ok = _submissions(
        accessions=[("0001111111-26-000001", "8-K", _today_minus(0))]
    )
    body_ok_2 = _submissions(
        accessions=[("0003333333-26-000001", "8-K", _today_minus(0))]
    )

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if f"CIK{cik_ok:010d}.json" in url:
            return httpx.Response(200, content=body_ok)
        if f"CIK{cik_fail:010d}.json" in url:
            return httpx.Response(500, text="upstream error")
        if f"CIK{cik_ok_2:010d}.json" in url:
            return httpx.Response(200, content=body_ok_2)
        return httpx.Response(404, text="?")

    transport = httpx.MockTransport(handle)
    universe = _FakeUniverse([cik_ok, cik_fail, cik_ok_2])
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    edgar = _client(transport)
    rec = Reconciler(edgar=edgar, universe=universe, queue=queue, db_path=db)
    try:
        recovered = await rec.reconcile_once()
    finally:
        await edgar.aclose()
    # Two healthy CIKs each yielded one missing filing; failure CIK is skipped.
    assert recovered == 2
    queued = []
    while not queue.empty():
        queued.append(queue.get_nowait())
    queued_ciks = {q.cik for q in queued}
    assert queued_ciks == {cik_ok, cik_ok_2}


# ---------------------------------------------------------------------------
# AC-6: recovered counter exposed; force_reconcile_now hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_force_reconcile_now_runs_immediately_and_increments_counter(
    db: str,
) -> None:
    cik = 1234567
    body = _submissions(
        accessions=[("0001234567-26-000200", "8-K", _today_minus(0))]
    )
    transport = _route_submissions(cik_to_body={cik: body})
    universe = _FakeUniverse([cik])
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    edgar = _client(transport)
    rec = Reconciler(edgar=edgar, universe=universe, queue=queue, db_path=db)
    try:
        recovered = await rec.force_reconcile_now()
    finally:
        await edgar.aclose()
    assert recovered == 1
    assert rec.filings_recovered_total == 1


@pytest.mark.asyncio
async def test_default_interval_is_six_hours() -> None:
    assert DEFAULT_INTERVAL_SECONDS == 21600  # AC-2
    assert DEFAULT_LOOKBACK_DAYS == 7  # AC-3


# ---------------------------------------------------------------------------
# AC-1: start/stop lifecycle drains nothing in a synthetic short interval.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_stop_lifecycle(db: str) -> None:
    transport = _route_submissions(cik_to_body={})
    universe = _FakeUniverse([])
    queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()
    edgar = _client(transport)
    # Use a short interval to verify the lifecycle without waiting 6h.
    rec = Reconciler(
        edgar=edgar,
        universe=universe,
        queue=queue,
        db_path=db,
        interval_seconds=1,
    )
    try:
        await rec.start()
        await asyncio.sleep(0.05)
        await rec.stop()
    finally:
        await edgar.aclose()
    assert queue.empty()
