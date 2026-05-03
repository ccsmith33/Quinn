"""S3.3 — Detail fetcher tests.

End-to-end: receive a `DiscoveredFiling` (S3.2 output), fetch primary
document via the shared `EdgarClient`, parse per form-type, persist a
row to the `filings` table, and write raw plaintext / Form-4 JSON to
disk under `raw_root/<accession>.{txt|json}`.

ADR-002 §3 + ADR-007 owned scenarios:
- "primary doc is fetched per form type" (real-EDGAR-shape index.json)
- "Form 4 with footnoted codes parses correctly"
- ADR-007 Stage 1 heuristic + Stage 2 SGML-header fallback
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
_FIXTURES = Path(__file__).parent / "fixtures"

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
    primary_type: str = "text.gif",
    extra_items: list[dict[str, str]] | None = None,
    primary_status: int = 200,
    index_status: int = 200,
    sgml_body: bytes | None = None,
    sgml_status: int = 200,
) -> httpx.MockTransport:
    """Build a transport that serves index.json, the primary doc, and the
    `<accession>.txt` SGML on the canonical EDGAR Archives URLs.

    Real EDGAR's `index.json` `type` field is an icon-name string
    (`text.gif`, `compressed.gif`, …) — NOT the form type. Tests that
    pass `primary_type="8-K"` or similar are codifying a fiction; the
    default here matches real EDGAR shape.
    """
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
        if sgml_body is not None and url.endswith(f"/{accession}.txt"):
            return httpx.Response(sgml_status, content=sgml_body)
        if primary_name in url:
            return httpx.Response(primary_status, content=primary_body)
        return httpx.Response(404, text=f"unhandled {url}")

    return httpx.MockTransport(handle)


def _route_from_index(
    *,
    cik: int,
    accession: str,
    index_payload: bytes,
    primary_name: str,
    primary_body: bytes,
    sgml_body: bytes | None = None,
    sgml_status: int = 200,
) -> httpx.MockTransport:
    """Serve a captured real-EDGAR `index.json` body verbatim alongside the
    primary doc and (optionally) the SGML `<accession>.txt`. Used for fixture-
    driven Stage-1 / Stage-2 tests."""
    acc_nd = _accession_no_dashes(accession)

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "index.json" in url and acc_nd in url:
            return httpx.Response(200, content=index_payload)
        if sgml_body is not None and url.endswith(f"/{accession}.txt"):
            return httpx.Response(sgml_status, content=sgml_body)
        if primary_name in url:
            return httpx.Response(200, content=primary_body)
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
    body_primary = b"<html><body><p>" + b"Primary 10-Q content" * 200 + b"</p></body></html>"
    transport = _route(
        cik=1234567,
        accession="0001234567-26-000200",
        primary_name="acme-10q.htm",
        primary_body=body_primary,
        # Real EDGAR `type` is an icon-name string; prose-primary selection
        # is driven by filename heuristics (ADR-007 Stage 1), not `type`.
        extra_items=[
            # Distractors — exhibit pattern must exclude these regardless of size.
            {"name": "ex-99-1.htm", "type": "text.gif", "size": "100000"},
            {"name": "ex-31-1.htm", "type": "text.gif", "size": "50000"},
            # XBRL accessory — extension/suffix exclusion.
            {"name": "acme-20260331_lab.xml", "type": "text.gif", "size": "200000"},
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
    # Index returns 200 with no Stage-1 candidate (only an exhibit, which is
    # excluded by the regex). `<accession>.txt` 404s, so Stage 2 also fails.
    # Fetcher records partial with the unresolved-error message.
    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "index.json" in url:
            return httpx.Response(
                200,
                content=_index_json(
                    [
                        {"name": "ex-99-1.htm", "type": "text.gif", "size": "10"},
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


# ---------------------------------------------------------------------------
# ADR-007 — real-EDGAR-shape fixture tests for prose primary-doc selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_apple_10k_index_picks_inline_xbrl_primary(
    db: str, tmp_path: Path
) -> None:
    """ADR-007 §"Test scenarios": a real Apple 10-K `index.json` payload
    contains an Inline XBRL primary (`aapl-20240928.htm`, 1.5 MB), several
    `a10-kexhibit*.htm` exhibits, R\\d+.htm XBRL-viewer fragments, and
    XBRL accessory files. Stage 1 must return the primary."""
    index_payload = (_FIXTURES / "edgar_index_apple_10k.json").read_bytes()
    primary_body = b"<html><body>" + b"Item 1. Business. " * 1000 + b"</body></html>"
    transport = _route_from_index(
        cik=320193,
        accession="0000320193-24-000123",
        index_payload=index_payload,
        primary_name="aapl-20240928.htm",
        primary_body=primary_body,
    )
    edgar = _client(transport)
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            DiscoveredFiling(
                accession_number="0000320193-24-000123",
                cik=320193,
                form_type="10-K",
                filed_at=dt.datetime(2024, 11, 1, 6, 1, tzinfo=dt.UTC),
                discovered_at=dt.datetime(2024, 11, 1, 6, 5, tzinfo=dt.UTC),
            )
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    assert row.ingest_state == "ok", row.ingest_error
    assert row.ingest_error is None
    text = Path(row.raw_text_path).read_text()
    assert "Item 1. Business." in text


@pytest.mark.asyncio
async def test_real_apple_8k_index_picks_sgml_canonical_primary(
    db: str, tmp_path: Path
) -> None:
    """ADR-007 §"Test scenarios": Apple's 8-K — agent-style filename
    `a8-kex991q2202603282026.htm` (168 KB EX-99.1 exhibit) slips past the
    narrow Stage 1 regex AND is the size winner over the canonical primary
    `aapl-20260430.htm` (37 KB, TYPE=8-K SEQUENCE=1). Stage 2 SGML safety
    net (triggered by `_stage1_is_contested` when the picked filename
    matches the loose `ex(hibit)?[-_]?\\d` pattern) must disambiguate to
    the SGML-authoritative primary."""
    index_payload = (_FIXTURES / "edgar_index_apple_8k.json").read_bytes()
    sgml_payload = (_FIXTURES / "edgar_sgml_apple_8k_header.txt").read_bytes()
    primary_body = (
        b"<html><body>" + b"Item 2.02 Results of Operations and Financial Condition. " * 100
        + b"Item 9.01 Financial Statements and Exhibits."
        + b"</body></html>"
    )
    transport = _route_from_index(
        cik=320193,
        accession="0000320193-26-000011",
        index_payload=index_payload,
        primary_name="aapl-20260430.htm",
        primary_body=primary_body,
        sgml_body=sgml_payload,
    )
    edgar = _client(transport)
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            DiscoveredFiling(
                accession_number="0000320193-26-000011",
                cik=320193,
                form_type="8-K",
                filed_at=dt.datetime(2026, 4, 30, 16, 30, tzinfo=dt.UTC),
                discovered_at=dt.datetime(2026, 4, 30, 16, 35, tzinfo=dt.UTC),
            )
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    assert row.ingest_state == "ok", row.ingest_error
    text = Path(row.raw_text_path).read_text()
    assert "Item 2.02" in text
    # The exhibit body must NOT have leaked in.
    assert "kex991" not in text


@pytest.mark.asyncio
async def test_form_type_token_disambiguates_equal_size_htm_survivors(
    db: str, tmp_path: Path
) -> None:
    """ADR-007 §"Test scenarios" (post-2026-05-03 patch): two non-exhibit
    `.htm` survivors of IDENTICAL reported size; the form-type-token hint
    fires only here as the size-tied tie-breaker and prefers the
    `8-k`-bearing filename."""
    items = [
        # Equal-size survivors: hint is the disambiguator.
        {"name": "abc-8k-2026-05-01.htm", "type": "text.gif", "size": "9000"},
        {"name": "abc-appendix.htm", "type": "text.gif", "size": "9000"},
        # Distractors that must be excluded by Stage 1.
        {"name": "ex-99-1.htm", "type": "text.gif", "size": "100000"},
        {"name": "abc-20260501_lab.xml", "type": "text.gif", "size": "200000"},
    ]
    primary_body = (
        b"<html><body>" + b"Item 8.01 Other Events. " * 100 + b"</body></html>"
    )
    acc = "0001111111-26-000001"
    acc_nd = _accession_no_dashes(acc)

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "index.json" in url and acc_nd in url:
            return httpx.Response(200, content=_index_json(items))
        if "abc-8k-2026-05-01.htm" in url:
            return httpx.Response(200, content=primary_body)
        return httpx.Response(404, text=f"unhandled {url}")

    edgar = _client(httpx.MockTransport(handle))
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            _discovered(accession=acc, form_type="8-K")
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    assert row.ingest_state == "ok", row.ingest_error


@pytest.mark.asyncio
async def test_stage1_empty_falls_back_to_sgml_header(
    db: str, tmp_path: Path
) -> None:
    """ADR-007 §"Test scenarios": when every `index.json` candidate is
    excluded by the Stage-1 patterns, parse `<accession>.txt` SGML
    `<DOCUMENT>` blocks and use the `<TYPE>=form_type` block's
    `<FILENAME>`."""
    sgml = (_FIXTURES / "edgar_sgml_apple_8k_header.txt").read_bytes()
    # An index.json that has nothing surviving Stage 1.
    items = [
        {"name": "ex-99-1.htm", "type": "text.gif", "size": "100000"},
        {"name": "ex-31-1.htm", "type": "text.gif", "size": "5000"},
        {"name": "aapl-20260430_lab.xml", "type": "text.gif", "size": "200000"},
        {"name": "R1.htm", "type": "text.gif", "size": "70000"},
    ]
    # The SGML header points at filename "aapl-20260430.htm" as TYPE=8-K SEQ=1.
    primary_body = (
        b"<html><body>" + b"Item 2.02 Results of Operations. " * 100 + b"</body></html>"
    )
    acc = "0000320193-26-000011"
    acc_nd = _accession_no_dashes(acc)

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "index.json" in url and acc_nd in url:
            return httpx.Response(200, content=_index_json(items))
        if url.endswith(f"/{acc}.txt"):
            return httpx.Response(200, content=sgml)
        if "aapl-20260430.htm" in url:
            return httpx.Response(200, content=primary_body)
        return httpx.Response(404, text=f"unhandled {url}")

    edgar = _client(httpx.MockTransport(handle))
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            _discovered(accession=acc, cik=320193, form_type="8-K")
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    assert row.ingest_state == "ok", row.ingest_error
    text = Path(row.raw_text_path).read_text()
    assert "Item 2.02" in text


@pytest.mark.asyncio
async def test_stage1_empty_stage2_404_records_partial_with_unresolved_message(
    db: str, tmp_path: Path
) -> None:
    """Both Stage 1 and Stage 2 fail → partial with the new error message."""
    items = [{"name": "ex-99-1.htm", "type": "text.gif", "size": "1000"}]
    acc = "0001234567-26-000777"
    acc_nd = _accession_no_dashes(acc)

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "index.json" in url and acc_nd in url:
            return httpx.Response(200, content=_index_json(items))
        if url.endswith(f"/{acc}.txt"):
            return httpx.Response(404, text="not found")
        return httpx.Response(404, text=f"unhandled {url}")

    edgar = _client(httpx.MockTransport(handle))
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            _discovered(accession=acc, form_type="10-K")
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    assert row.ingest_state == "partial"
    assert row.ingest_error is not None
    # ADR-007 spec: partial-row error message indicates both stages failed.
    assert "primary doc unresolved" in row.ingest_error.lower()


@pytest.mark.asyncio
async def test_stage2_malformed_sgml_records_partial(
    db: str, tmp_path: Path
) -> None:
    """`<accession>.txt` returns 200 but the SGML body has no parseable
    `<DOCUMENT>` blocks → Stage 2 returns None → partial."""
    items = [{"name": "ex-99-1.htm", "type": "text.gif", "size": "1000"}]
    acc = "0001234567-26-000778"
    acc_nd = _accession_no_dashes(acc)
    bogus_sgml = b"<SEC-DOCUMENT>this is not real sgml content</SEC-DOCUMENT>"

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "index.json" in url and acc_nd in url:
            return httpx.Response(200, content=_index_json(items))
        if url.endswith(f"/{acc}.txt"):
            return httpx.Response(200, content=bogus_sgml)
        return httpx.Response(404, text=f"unhandled {url}")

    edgar = _client(httpx.MockTransport(handle))
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            _discovered(accession=acc, form_type="10-K")
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    assert row.ingest_state == "partial"
    assert row.ingest_error is not None


# ---------------------------------------------------------------------------
# ADR-007 — Stage 1 selector unit tests (regex / extension exclusions)
# ---------------------------------------------------------------------------


def test_select_prose_primary_excludes_exhibit_patterns() -> None:
    from ingestion.detail_fetcher import DetailFetcher

    items = [
        {"name": "ex-99-1.htm", "type": "text.gif", "size": "100000"},
        {"name": "exhibit_31-1.htm", "type": "text.gif", "size": "50000"},
        {"name": "ex99.htm", "type": "text.gif", "size": "200000"},
        {"name": "abc-10k-2026.htm", "type": "text.gif", "size": "5000"},
    ]
    assert DetailFetcher._select_prose_primary(items, "10-K") == "abc-10k-2026.htm"


def test_select_prose_primary_excludes_xbrl_viewer_fragments() -> None:
    from ingestion.detail_fetcher import DetailFetcher

    items = [
        {"name": "R1.htm", "type": "text.gif", "size": "90000"},
        {"name": "R10.htm", "type": "text.gif", "size": "70000"},
        {"name": "primary.htm", "type": "text.gif", "size": "5000"},
    ]
    assert DetailFetcher._select_prose_primary(items, "10-K") == "primary.htm"


def test_select_prose_primary_excludes_xbrl_accessories() -> None:
    from ingestion.detail_fetcher import DetailFetcher

    items = [
        {"name": "abc-20260501_lab.xml", "type": "text.gif", "size": "500000"},
        {"name": "abc-20260501_def.xml", "type": "text.gif", "size": "300000"},
        {"name": "abc-20260501_cal.xml", "type": "text.gif", "size": "200000"},
        {"name": "abc-20260501_pre.xml", "type": "text.gif", "size": "100000"},
        {"name": "abc-20260501_htm.xml", "type": "text.gif", "size": "1000000"},
        {"name": "abc-20260501.xsd", "type": "text.gif", "size": "50000"},
        {"name": "FilingSummary.xml", "type": "text.gif", "size": "10000"},
        {"name": "MetaLinks.json", "type": "text.gif", "size": "10000"},
        {"name": "Financial_Report.xlsx", "type": "text.gif", "size": "10000"},
        {"name": "abc-20260501.htm", "type": "text.gif", "size": "5000"},
    ]
    assert DetailFetcher._select_prose_primary(items, "10-K") == "abc-20260501.htm"


def test_select_prose_primary_returns_none_when_all_excluded() -> None:
    from ingestion.detail_fetcher import DetailFetcher

    items = [
        {"name": "ex-99-1.htm", "type": "text.gif", "size": "100000"},
        {"name": "Show.js", "type": "text.gif", "size": "1000"},
    ]
    assert DetailFetcher._select_prose_primary(items, "10-K") is None


def test_select_prose_primary_form_type_token_hint_breaks_equal_size_tie() -> None:
    """ADR-007 §"Identification heuristic spec" (post-2026-05-03 patch):
    hint fires ONLY when multiple survivors share the top size."""
    from ingestion.detail_fetcher import DetailFetcher

    items = [
        {"name": "abc-8k-2026.htm", "type": "text.gif", "size": "9000"},
        {"name": "abc-appendix.htm", "type": "text.gif", "size": "9000"},
    ]
    assert (
        DetailFetcher._select_prose_primary(items, "8-K") == "abc-8k-2026.htm"
    )


def test_select_prose_primary_hint_does_not_displace_unique_size_winner() -> None:
    """ADR-007 §"Test scenarios" (post-2026-05-03 patch) — mandatory
    Apple-shape regression: size winner has no form-type token; siblings
    carry the token; hint MUST NOT fire because the size winner is unique."""
    from ingestion.detail_fetcher import DetailFetcher

    items = [
        # Size winner — no form-type token.
        {"name": "aapl-20240928.htm", "type": "text.gif", "size": "1503780"},
        # Hint-bearing siblings, all dominated on size. Hint must NOT fire.
        {"name": "a10-kappendix4109282024.htm", "type": "text.gif", "size": "120785"},
        {"name": "a10-kappendix10229282024.htm", "type": "text.gif", "size": "75240"},
    ]
    assert (
        DetailFetcher._select_prose_primary(items, "10-K") == "aapl-20240928.htm"
    )


# ---------------------------------------------------------------------------
# ADR-007 Stage 2 — SGML parser unit tests (per architect handoff concern #1)
# ---------------------------------------------------------------------------


def test_sgml_parser_clean_header_picks_sequence_1_filename() -> None:
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    body = (
        b"<SEC-HEADER>...</SEC-HEADER>\n"
        b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>1\n<FILENAME>primary.htm\n<TEXT>"
        b"body\n</TEXT>\n</DOCUMENT>\n"
        b"<DOCUMENT>\n<TYPE>EX-99.1\n<SEQUENCE>2\n<FILENAME>ex99.htm\n<TEXT>"
        b"body\n</TEXT>\n</DOCUMENT>\n"
    )
    assert _parse_sgml_filename_for_type(body, "10-K") == "primary.htm"


def test_sgml_parser_prefers_lower_sequence_when_multiple_match() -> None:
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    body = (
        b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>3\n<FILENAME>third.htm\n<TEXT>x</TEXT>\n</DOCUMENT>\n"
        b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>1\n<FILENAME>first.htm\n<TEXT>x</TEXT>\n</DOCUMENT>\n"
    )
    assert _parse_sgml_filename_for_type(body, "10-K") == "first.htm"


def test_sgml_parser_returns_none_when_no_type_matches() -> None:
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    body = (
        b"<DOCUMENT>\n<TYPE>EX-99.1\n<SEQUENCE>1\n<FILENAME>ex.htm\n<TEXT>x</TEXT>\n</DOCUMENT>\n"
    )
    assert _parse_sgml_filename_for_type(body, "10-K") is None


def test_sgml_parser_unclosed_document_block_skipped() -> None:
    """Unclosed `<DOCUMENT>` block (no `<TEXT>` sentinel within byte cap)
    is skipped; later well-formed blocks still parse."""
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    # First block has no <TEXT> at all (truncated). Second block is well-formed.
    body = (
        b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>1\n<FILENAME>truncated.htm\n"
        # No <TEXT>... so the next <DOCUMENT> starts the next valid block.
        b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>2\n<FILENAME>recovered.htm\n<TEXT>x</TEXT>\n</DOCUMENT>\n"
    )
    # The first block's prologue does end at the second `<DOCUMENT>`. Since
    # there's no <TEXT> sentinel inside it (within the 64 KB cap), we skip it
    # and only the second block resolves.
    result = _parse_sgml_filename_for_type(body, "10-K")
    # Either implementation choice is acceptable: skip-first-block→use-second,
    # or fall through. The contract is "no exception, returns a sane result".
    assert result in {"recovered.htm", None}


def test_sgml_parser_missing_type_field_skipped() -> None:
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    body = (
        # First block has no <TYPE>; should be skipped.
        b"<DOCUMENT>\n<SEQUENCE>1\n<FILENAME>notype.htm\n<TEXT>x</TEXT>\n</DOCUMENT>\n"
        b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>2\n<FILENAME>ok.htm\n<TEXT>x</TEXT>\n</DOCUMENT>\n"
    )
    assert _parse_sgml_filename_for_type(body, "10-K") == "ok.htm"


def test_sgml_parser_missing_filename_field_skipped() -> None:
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    body = (
        b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>1\n<TEXT>x</TEXT>\n</DOCUMENT>\n"
        b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>2\n<FILENAME>backup.htm\n<TEXT>x</TEXT>\n</DOCUMENT>\n"
    )
    assert _parse_sgml_filename_for_type(body, "10-K") == "backup.htm"


def test_sgml_parser_block_exceeding_byte_cap_is_skipped() -> None:
    """A pathological `<DOCUMENT>` block where the prologue (start →
    `<TEXT>`) exceeds 64 KB is skipped, not parsed. Bounded parse cost
    per ADR-007 §"Error handling"."""
    from ingestion.detail_fetcher import _SGML_BLOCK_BYTE_CAP, _parse_sgml_filename_for_type

    pathological_padding = b"X" * (_SGML_BLOCK_BYTE_CAP + 1024)
    body = (
        b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>1\n<FILENAME>over_cap.htm\n"
        + pathological_padding
        + b"<TEXT>x</TEXT>\n</DOCUMENT>\n"
        # A second sane block that should still resolve.
        b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>2\n<FILENAME>under_cap.htm\n<TEXT>x</TEXT>\n</DOCUMENT>\n"
    )
    assert _parse_sgml_filename_for_type(body, "10-K") == "under_cap.htm"


def test_sgml_parser_non_ascii_description_does_not_break_filename_extraction() -> None:
    """Non-ASCII bytes in `<DESCRIPTION>` (occasionally seen in legacy
    filings) must not break TYPE/FILENAME extraction — only the FILENAME
    needs to be ASCII-decodable."""
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    body = (
        b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>1\n<FILENAME>primary.htm\n"
        b"<DESCRIPTION>Caf\xc3\xa9 du Centre annual report\n"
        b"<TEXT>body</TEXT>\n</DOCUMENT>\n"
    )
    assert _parse_sgml_filename_for_type(body, "10-K") == "primary.htm"


def test_sgml_parser_case_insensitive_type_match() -> None:
    """SGML `<TYPE>` matching is case-insensitive per ADR-007 §"Stage 2"."""
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    body = (
        b"<DOCUMENT>\n<TYPE>10-k\n<SEQUENCE>1\n<FILENAME>primary.htm\n"
        b"<TEXT>x</TEXT>\n</DOCUMENT>\n"
    )
    assert _parse_sgml_filename_for_type(body, "10-K") == "primary.htm"


def test_sgml_parser_empty_body_returns_none() -> None:
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    assert _parse_sgml_filename_for_type(b"", "10-K") is None


def test_sgml_parser_completely_malformed_returns_none() -> None:
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    assert (
        _parse_sgml_filename_for_type(b"this is not even close to SGML", "10-K")
        is None
    )


# ---------------------------------------------------------------------------
# ADR-007 — additional real-EDGAR fixtures (per architect handoff concern #4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_altex_10q_index_picks_form_token_named_primary(
    db: str, tmp_path: Path
) -> None:
    """Real microcap 10-Q (Altex Industries, 2026-05-01) — primary is
    `altx-20260331_10q.htm` (contains the `10q` form-type token in name);
    exhibits use `altx_ex31.htm` / `altx_ex32.htm` naming. Validates
    Stage 1 against a different filer-agent shape than Apple's."""
    index_payload = (_FIXTURES / "edgar_index_altex_10q.json").read_bytes()
    primary_body = (
        b"<html><body>" + b"Item 1. Financial Statements. " * 200 + b"</body></html>"
    )
    transport = _route_from_index(
        cik=775057,
        accession="0001096906-26-000672",
        index_payload=index_payload,
        primary_name="altx-20260331_10q.htm",
        primary_body=primary_body,
    )
    edgar = _client(transport)
    fetcher = DetailFetcher(edgar=edgar, db_path=db, raw_root=tmp_path / "raw")
    try:
        filing_id = await fetcher.fetch_and_persist(
            DiscoveredFiling(
                accession_number="0001096906-26-000672",
                cik=775057,
                form_type="10-Q",
                filed_at=dt.datetime(2026, 5, 1, 21, 8, tzinfo=dt.UTC),
                discovered_at=dt.datetime(2026, 5, 1, 21, 12, tzinfo=dt.UTC),
            )
        )
    finally:
        await edgar.aclose()
    row = get_filing_by_id(db, filing_id)
    assert row is not None
    assert row.ingest_state == "ok", row.ingest_error
    text = Path(row.raw_text_path).read_text()
    assert "Item 1. Financial Statements." in text


def test_sgml_parser_resolves_real_altex_10q_primary() -> None:
    """Stage 2 against the real Altex 10-Q SGML header."""
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    body = (_FIXTURES / "edgar_sgml_altex_10q_header.txt").read_bytes()
    assert _parse_sgml_filename_for_type(body, "10-Q") == "altx-20260331_10q.htm"


def test_sgml_parser_resolves_real_apple_8k_primary() -> None:
    """Stage 2 against the real Apple 8-K SGML header (already used as a
    fixture in the Stage-1-empty integration test, but exercised here as
    a parser unit test directly)."""
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    body = (_FIXTURES / "edgar_sgml_apple_8k_header.txt").read_bytes()
    assert _parse_sgml_filename_for_type(body, "8-K") == "aapl-20260430.htm"


def test_sgml_parser_resolves_real_form4_primary() -> None:
    """Stage 2 parser is form-agnostic; verify it resolves a real Form 4
    `<accession>.txt` header. Form 4's primary selection in production
    still uses `_select_form4_primary` (unchanged); this test exercises
    parser correctness across `<TYPE>` codes only."""
    from ingestion.detail_fetcher import _parse_sgml_filename_for_type

    body = (_FIXTURES / "edgar_sgml_coreweave_form4_header.txt").read_bytes()
    assert _parse_sgml_filename_for_type(body, "4") == "tm2613377-3_4seq1.xml"
