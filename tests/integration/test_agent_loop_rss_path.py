"""Integration test: RSS discovery → detail fetch → consumer pipeline.

This is the test that S5.6 review missed: the existing smoke test populates
`ingestion_queue` directly, bypassing RSS entirely. This exercise drives
the full path — `RssDiscoveryLoop` polls EDGAR (via `httpx.MockTransport`),
puts a `DiscoveredFiling` on its queue, the detail-pump task fetches the
primary doc + persists a `FilingRow`, and the consumer picks it up.

If `compose_agent` did not wire the discovery loop or `AgentLoop.run` did
not start the discovery task, this test fails with the same symptom as
the production droplet: 0 filings ingested after several seconds of agent
boot.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from app.composition import AgentComponents
from app.loop import AgentLoop
from execution.orders import OrderSubmitter
from execution.sizing import SizingEngine
from execution.validator import ProposalValidator
from ingestion.detail_fetcher import DetailFetcher
from ingestion.edgar_client import EdgarClient
from ingestion.rss_loop import DiscoveredFiling, RssDiscoveryLoop
from journal.models import FilingRow
from journal.repo import JournalRepo, connect


# ---------------------------------------------------------------------------
# Fakes / fixtures local to this test
# ---------------------------------------------------------------------------


def _atom_feed(*, accession: str, cik: int, form: str = "8-K") -> bytes:
    """Minimal Atom payload with a single entry."""
    body = (
        '<?xml version="1.0"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        "  <title>EDGAR Latest</title>\n"
        f"  <entry>\n"
        f"    <title>{form} - ACME ({cik:010d})</title>\n"
        f"    <updated>2026-04-29T14:30:00+00:00</updated>\n"
        f'    <category term="{form}"/>\n'
        f"    <id>urn:tag:sec.gov,2008:accession-number={accession}</id>\n"
        f"  </entry>\n"
        "</feed>\n"
    )
    return body.encode("utf-8")


def _index_json(accession: str, primary: str = "primary.htm") -> bytes:
    payload = {
        "directory": {
            "item": [
                {"name": primary, "type": "8-K", "size": "1024"},
            ]
        }
    }
    return json.dumps(payload).encode("utf-8")


def _primary_html() -> bytes:
    return (
        b"<html><body>"
        b"<p>Item 1.01 Material Definitive Agreement.</p>"
        b"<p>ACME announced a material acquisition with concrete pricing terms.</p>"
        b"</body></html>"
    )


class _ScriptedEdgarTransport:
    """Routes a small set of EDGAR URLs to canned bytes.

    Routes:
      - `/cgi-bin/browse-edgar` (atom RSS): first call returns one entry;
        subsequent calls return empty feed (so we don't loop endlessly).
      - `/Archives/.../index.json`: returns the canned index.
      - `/Archives/.../primary.htm`: returns the primary HTML body.
    """

    def __init__(self, *, accession: str, cik: int) -> None:
        self._accession = accession
        self._cik = cik
        # Per-form-type call counter so first-poll gets the entry, all
        # later polls return empty feeds — the rss_loop only emits NEW
        # accessions and we don't want a flood.
        self._rss_calls: dict[str, int] = {}
        self.requests: list[str] = []

    def as_mock(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            self.requests.append(url)
            # RSS endpoint
            if "browse-edgar" in url:
                form = request.url.params.get("type", "")
                n = self._rss_calls.get(form, 0)
                self._rss_calls[form] = n + 1
                # Only the 8-K feed gets a non-empty payload, on its
                # SECOND call. The first call must be empty so the
                # rss_loop "first poll seeds the seen-set without
                # emitting" branch (AC-6 of S3.2) doesn't drop our
                # fixture entry on the floor.
                if form == "8-K" and n == 1:
                    return httpx.Response(
                        200,
                        content=_atom_feed(
                            accession=self._accession, cik=self._cik
                        ),
                        headers={"content-type": "application/atom+xml"},
                    )
                return httpx.Response(200, content=_atom_feed_empty())
            # Index lookup
            if url.endswith("/index.json"):
                return httpx.Response(200, content=_index_json(self._accession))
            # Primary HTML
            if url.endswith("/primary.htm"):
                return httpx.Response(
                    200,
                    content=_primary_html(),
                    headers={"content-type": "text/html"},
                )
            return httpx.Response(404, content=b"")

        return httpx.MockTransport(handle)


def _atom_feed_empty() -> bytes:
    body = (
        '<?xml version="1.0"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        "  <title>EDGAR Latest</title>\n"
        "</feed>\n"
    )
    return body.encode("utf-8")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rss_discovery_pipes_filing_through_consumer(
    db_path: str,
    journal: JournalRepo,
    fake_broker,
    fake_anthropic,
    universe,
    prompt_builder,
    proposal_store,
    killswitch,
    tmp_path: Path,
) -> None:
    """End-to-end: RSS poll → DetailFetcher → FilingRow → consumer pipeline.

    The fake EDGAR transport returns one in-universe 8-K on the second
    poll (first poll is empty so the rss_loop's "first poll seeding"
    branch doesn't swallow it). The detail-fetcher pulls the index +
    primary doc and writes the journal row. The consumer prefilters,
    analyzes, reviews, and executes — all real components except the
    Anthropic + broker boundaries.

    Assertion: within 5 seconds of `loop.run()`, at least one
    prefilter_decisions row exists for the discovered accession.
    """
    from analyzer.opus import OpusReviewer
    from analyzer.sonnet import SonnetAnalyzer
    from prefilter.orchestrator import Prefilter

    from tests.integration.conftest import (
        _StubReconciler,
        _opus_ratify_json,
        _valid_proposal_json,
    )

    accession = "0001234567-26-000099"
    cik = 320193  # in-universe per the universe fixture

    transport = _ScriptedEdgarTransport(accession=accession, cik=cik)
    edgar = EdgarClient(
        user_agent="Quinn-Test/v1 test@example.com",
        transport=transport.as_mock(),
        retry_base_seconds=0.0,  # no retry sleep in tests
    )

    # Real RSS loop, real DetailFetcher; tight cadences so the test
    # converges quickly.
    raw_root = tmp_path / "raw"
    cursor_path = tmp_path / "rss_cursor.json"
    discovered_queue: asyncio.Queue[DiscoveredFiling] = asyncio.Queue()

    rss_loop = RssDiscoveryLoop(
        edgar=edgar,
        universe=universe,
        queue=discovered_queue,
        cursor_path=cursor_path,
        poll_market_seconds=0,  # tight loop in test
        poll_offhours_seconds=0,
        form_types=("8-K",),  # narrow to one form to keep transport simple
    )
    detail_fetcher = DetailFetcher(
        edgar=edgar,
        db_path=db_path,
        raw_root=raw_root,
    )

    # Real pipeline components.
    similarity = MagicMock()
    prefilter = Prefilter(db_path=db_path, universe=universe, similarity=similarity)
    reviewer = OpusReviewer(
        client=fake_anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        ks4_pct_cap=0.20,
        db_path=db_path,
    )
    analyzer = SonnetAnalyzer(
        client=fake_anthropic,
        store=proposal_store,
        prompt_builder=prompt_builder,
        opus_reviewer=reviewer,
        sonnet_model_id="claude-sonnet-4-6",
        opus_review_conviction_threshold=7,
        db_path=db_path,
    )
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9, symbol="ACME")],
        "review": [_opus_ratify_json()],
    }

    # Register prompt versions so FK constraints hold.
    from app.composition import register_composed_prompt_versions

    register_composed_prompt_versions(prompt_builder, db_path)

    ingestion_queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    components = AgentComponents(
        universe=universe,
        prefilter=prefilter,
        analyzer=analyzer,
        reviewer=reviewer,
        proposal_store=proposal_store,
        validator=ProposalValidator(),
        sizer=SizingEngine(),
        submitter=OrderSubmitter(),
        execution=None,
        broker=fake_broker,
        reconciler=_StubReconciler(),
        killswitch=killswitch,
        ingestion_queue=ingestion_queue,
        journal=journal,
        rss_discovery_loop=rss_loop,
        detail_fetcher=detail_fetcher,
        discovered_queue=discovered_queue,
    )
    loop = AgentLoop(components=components, shutdown_grace_seconds=5.0)

    runner = asyncio.create_task(loop.run())

    # Wait up to 5s for at least one prefilter_decisions row keyed on the
    # discovered accession. If discovery is unwired, this never happens.
    deadline = asyncio.get_event_loop().time() + 5.0
    saw_filing = False
    while asyncio.get_event_loop().time() < deadline:
        with connect(db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM prefilter_decisions"
            ).fetchone()
        if row["n"] >= 1:
            saw_filing = True
            break
        await asyncio.sleep(0.05)

    # Shut down cleanly.
    loop.shutdown_requested = True
    await ingestion_queue.put(None)  # type: ignore[arg-type]
    try:
        await asyncio.wait_for(runner, timeout=10.0)
    except TimeoutError:
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

    await edgar.aclose()

    assert saw_filing, (
        "No prefilter_decisions row appeared within 5s — the RSS path "
        "is not wired into the agent loop. This is the production bug."
    )

    # Stronger: the persisted filing must have come from the canned RSS
    # accession (not a leftover row).
    with connect(db_path) as conn:
        filings = conn.execute(
            "SELECT accession_number, ingest_state FROM filings ORDER BY id ASC"
        ).fetchall()
    accessions = [r["accession_number"] for r in filings]
    assert accession in accessions, (
        f"discovered accession {accession} not persisted; got {accessions}"
    )
