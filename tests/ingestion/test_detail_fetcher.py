"""S3.3 — Detail fetcher tests.

End-to-end: receive a `DiscoveredFiling` (S3.2 output), fetch primary
document via the shared `EdgarClient`, parse per form-type, persist a
row to the `filings` table, and write raw plaintext / Form-4 JSON to
disk under `raw_root/<accession>.{txt|json}`.

ADR-002 §3 owned scenarios:
- "primary doc is fetched per form type"
- "Form 4 with footnoted codes parses correctly"
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import httpx
import pytest

from ingestion.detail_fetcher import DetailFetcher
from ingestion.edgar_client import EdgarClient
from ingestion.normalize import content_hash, normalize_text
from ingestion.rss_loop import DiscoveredFiling
from journal.migrate import apply_migrations
from journal.repo import get_filing_by_accession, get_filing_by_id

_UA = "Quinn-Research/v1 contact@operator.example"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return str(db_path)


def _discovered(
    accession: str = "0001234567-26-000099",
    cik: int = 1234567,
    form_type: str = "8-K",
) -> DiscoveredFiling:
    return DiscoveredFiling(
        accession_number=accession,
        cik=cik,
        form_type=form_type,
        filed_at=dt.datetime(2026, 4, 29, 13, 55, tzinfo=dt.UTC),
        discovered_at=dt.datetime(2026, 4, 29, 13, 55, 30, tzinfo=dt.UTC),
    )


def _index_json(items: list[dict[str, str]]) -> bytes:
    return json.dumps({"directory": {"item": items, "name": "fake"}}).encode()


def _accession_no_dashes(acc: str) -> str:
    return acc.replace("-", "")


def _route(
    *,
    cik: int,
    accession: str,
    primary_name: str,
    primary_body: bytes,
    primary_type: str,
    extra_items: list[dict[str, str]] | None = None,
    primary_status: int = 200,
    index_status: int = 200,
) -> httpx.MockTransport:
    """Build a transport that serves index.json and the primary doc on the
    canonical EDGAR Archives URLs."""
    acc_nd = _accession_no_dashes(accession)
    items = list(extra_items or [])
    items.append(
        {
            "name": primary_name,
            "type": primary_type,
            "size": str(len(primary_body)),
        }
    )

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "index.json" in url and acc_nd in url:
            return httpx.Response(index_status, content=_index_json(items))
        if primary_name in url:
            return httpx.Response(primary_status, content=primary_body)
        return httpx.Response(404, text=f"unhandled {url}")

    return httpx.MockTransport(handle)


def _client(transport: httpx.MockTransport) -> EdgarClient:
    return EdgarClient(user_agent=_UA, transport=transport, retry_base_seconds=0.0)


# ---------------------------------------------------------------------------
# AC-2 + AC-5 + AC-6: prose form (8-K) end to end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filing_persisted_with_content_hash(db: str, tmp_path: Path) -> None:
    body = b"""<html><body>
        <h1>FORM 8-K</h1>
        <p>Item 2.02 Results of Operations and Financial Condition</p>
        <p>On April 29 2026, ACME announced earnings.</p>
        <p>Item 9.01 Financial Statements and Exhibits</p>
        </body></html>"""
    transport = _route(
        cik=1234567,
        accession="0001234567-26-000099",
        primary_name="acme-8k.htm",
        primary_body=body,
        primary_type="8-K",
    )
    edgar = _client(transport)
    raw_root = tmp_path / "raw"
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=raw_root)
    try:
        filing_id = await fetcher.fetch_and_persist(_discovered())
    finally:
        await edgar.aclose()

    row = get_filing_by_id(db, filing_id)
    assert row is not None
    assert row.accession_number == "0001234567-26-000099"
    assert row.form_type == "8-K"
    assert row.cik == 1234567
    assert row.ingest_state == "ok"
    assert row.ingest_error is None
    # AC-2: raw plaintext written to disk under raw_root.
    raw_path = Path(row.raw_text_path)
    assert raw_path.exists()
    text = raw_path.read_text()
    assert "FORM 8-K" in text
    assert "<" not in text
    # AC-5: content_hash matches sha256 of normalized plaintext.
    assert row.content_hash == content_hash(text)
    # AC-4: 8-K item codes captured as a JSON array.
    assert row.item_codes is not None
    items = json.loads(row.item_codes)
    assert items == ["2.02", "9.01"]


@pytest.mark.asyncio
async def test_primary_doc_fetched_per_form_type(db: str, tmp_path: Path) -> None:
    """ADR-002 scenario: each prose-form filing's primary document is the one
    flagged in `index.json` whose `type` matches the filing's form_type."""
    body_primary = b"<html><body><p>Primary 10-Q content</p></body></html>"
    transport = _route(
        cik=1234567,
        accession="0001234567-26-000200",
        primary_name="acme-10q.htm",
        primary_body=body_primary,
        primary_type="10-Q",
        extra_items=[
            # Distractors (lower-priority) — must be ignored.
            {"name": "exhibit-99.1.htm", "type": "EX-99.1", "size": "1000"},
            {"name": "exhibit-31.1.htm", "type": "EX-31.1", "size": "500"},
        ],
    )
    edgar = _client(transport)
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            _discovered(accession="0001234567-26-000200", form_type="10-Q")
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    text = Path(row.raw_text_path).read_text()
    assert "Primary 10-Q content" in text
    # Distractor body should not bleed into raw text.
    assert "exhibit" not in text.lower()


# ---------------------------------------------------------------------------
# AC-3 + AC-8: Form 4 XML
# ---------------------------------------------------------------------------


_FORM4_XML = b"""<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <periodOfReport>2026-04-29</periodOfReport>
  <issuer>
    <issuerCik>0001234567</issuerCik>
    <issuerName>ACME MICROCAP CORP</issuerName>
    <issuerTradingSymbol>ACME</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0009999000</rptOwnerCik>
      <rptOwnerName>SMITH JOHN</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship><isOfficer>1</isOfficer><officerTitle>CEO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common</value></securityTitle>
      <transactionDate><value>2026-04-28</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare>
          <value>5.25</value><footnoteId id="F1"/>
        </transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <footnotes><footnote id="F1">Average price across multiple trades.</footnote></footnotes>
</ownershipDocument>
"""


@pytest.mark.asyncio
async def test_form4_persisted_as_normalized_json(db: str, tmp_path: Path) -> None:
    transport = _route(
        cik=1234567,
        accession="0001234567-26-000300",
        primary_name="form4.xml",
        primary_body=_FORM4_XML,
        primary_type="4",
    )
    edgar = _client(transport)
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            _discovered(accession="0001234567-26-000300", form_type="4")
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    raw_path = Path(row.raw_text_path)
    assert raw_path.suffix == ".json"
    payload = json.loads(raw_path.read_text())
    # Issuer fields preserved.
    assert payload["issuer_cik"] == 1234567
    assert payload["issuer_ticker"] == "ACME"
    # Reporting owner preserved.
    assert payload["reporting_owner_cik"] == 9999000
    assert payload["reporting_owner_name"] == "SMITH JOHN"
    # Transaction structured.
    assert len(payload["transactions"]) == 1
    tx = payload["transactions"][0]
    assert tx["transaction_code"] == "P"
    assert tx["shares"] == "10000"
    assert tx["price_per_share"] == "5.25"
    assert tx["acquired_disposed"] == "A"
    # Footnote text preserved for the LLM (ADR-002 scenario).
    assert "Average price" in payload["footnotes"]


# ---------------------------------------------------------------------------
# AC-6: idempotency — duplicate accession does not crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_accession_no_crash(db: str, tmp_path: Path) -> None:
    body = b"<html><body><p>Hello</p></body></html>"
    transport = _route(
        cik=1234567,
        accession="0001234567-26-000400",
        primary_name="x.htm",
        primary_body=body,
        primary_type="8-K",
    )
    edgar = _client(transport)
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        first = await fetcher.fetch_and_persist(
            _discovered(accession="0001234567-26-000400")
        )
        second = await fetcher.fetch_and_persist(
            _discovered(accession="0001234567-26-000400")
        )
    finally:
        await edgar.aclose()
    assert first == second
    # Single row in the table.
    row = get_filing_by_accession(db, "0001234567-26-000400")
    assert row is not None
    assert row.id == first


# ---------------------------------------------------------------------------
# AC-7: parse / fetch failure → ingest_state="partial", queue not blocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_failure_records_partial(db: str, tmp_path: Path) -> None:
    # Index returns 200 with no usable primary document → fetcher records partial.
    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "index.json" in url:
            return httpx.Response(
                200,
                content=_index_json(
                    [
                        {"name": "exhibit.htm", "type": "EX-99.1", "size": "10"},
                    ]
                ),
            )
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handle)
    edgar = _client(transport)
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            _discovered(accession="0001234567-26-000500", form_type="10-K")
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    assert row.ingest_state == "partial"
    assert row.ingest_error is not None
    assert "primary" in row.ingest_error.lower()


@pytest.mark.asyncio
async def test_index_404_records_partial(db: str, tmp_path: Path) -> None:
    """L-1 from S3.1 review: EdgarClient returns 4xx rather than raising;
    detail fetcher must check status explicitly and record partial."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handle)
    edgar = _client(transport)
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            _discovered(accession="0001234567-26-000501", form_type="10-K")
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    assert row.ingest_state == "partial"
    assert row.ingest_error is not None
    assert "404" in row.ingest_error or "not found" in row.ingest_error.lower()


@pytest.mark.asyncio
async def test_form4_malformed_xml_records_partial(db: str, tmp_path: Path) -> None:
    transport = _route(
        cik=1234567,
        accession="0001234567-26-000502",
        primary_name="form4.xml",
        primary_body=b"<not even xml<",
        primary_type="4",
    )
    edgar = _client(transport)
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            _discovered(accession="0001234567-26-000502", form_type="4")
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    assert row.ingest_state == "partial"
    assert row.ingest_error is not None


# ---------------------------------------------------------------------------
# AC-1: returns FilingId
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_and_persist_returns_filing_id(db: str, tmp_path: Path) -> None:
    transport = _route(
        cik=1234567,
        accession="0001234567-26-000600",
        primary_name="x.htm",
        primary_body=b"<html><body><p>hi</p></body></html>",
        primary_type="8-K",
    )
    edgar = _client(transport)
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            _discovered(accession="0001234567-26-000600")
        )
    finally:
        await edgar.aclose()
    assert isinstance(filing_id, int)
    assert filing_id > 0


# ---------------------------------------------------------------------------
# Normalizer + content-hash consistency
# ---------------------------------------------------------------------------


def test_content_hash_is_stable_under_whitespace_changes() -> None:
    a = "Hello   WORLD  \n"
    b = "hello world"
    assert content_hash(a) == content_hash(b)


def test_normalize_text_lowercases_and_collapses() -> None:
    assert normalize_text("  HELLO    World\n\n") == "hello world"
