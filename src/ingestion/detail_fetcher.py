"""Detail fetcher (S3.3, ADR-002 §3).

Consumes the queue from S3.2 and turns each `DiscoveredFiling` into a
persisted `filings` row plus a raw artifact on disk:

  - prose forms (10-K, 10-Q, 8-K, S-1, DEF 14A): primary HTML/XBRL → plaintext
    at `<raw_root>/<accession>.txt`
  - Form 4: primary XML → normalized JSON at `<raw_root>/<accession>.json`

EDGAR Archives URL pattern (per ADR-002 §3):
  - index:    `https://www.sec.gov/Archives/edgar/data/<cik>/<accession_no_dashes>/index.json`
  - primary:  `https://www.sec.gov/Archives/edgar/data/<cik>/<accession_no_dashes>/<filename>`

Failure posture (AC-7): any non-fatal fetch / parse error is recorded as
`ingest_state="partial"` with `ingest_error` populated; the queue keeps
flowing. Only a database integrity error (other than duplicate-accession,
which is the idempotency happy path) propagates.

Carry-forward from S3.1 review L-1: `EdgarClient.get_bytes(...)` returns
non-2xx responses as a `Response` rather than raising. This module does
the explicit status check and falls into the partial path on 4xx.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from ingestion.edgar_client import EdgarClient, EdgarUnavailable
from ingestion.normalize import content_hash
from ingestion.parsers.eight_k import extract_item_codes
from ingestion.parsers.form4 import Form4ParseError, parse_form4_xml
from ingestion.parsers.html_to_text import html_to_text
from ingestion.rss_loop import DiscoveredFiling
from journal.models import FilingRow
from journal.repo import DuplicateAccession, insert_filing
from observability.log_port import get_logger

_log = get_logger(__name__)
_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
_PROSE_FORMS = frozenset({"10-K", "10-Q", "8-K", "S-1", "DEF 14A"})


def _accession_no_dashes(accession_number: str) -> str:
    return accession_number.replace("-", "")


def _index_url(cik: int, accession_number: str) -> str:
    return (
        f"{_ARCHIVES_BASE}/{cik}/"
        f"{_accession_no_dashes(accession_number)}/index.json"
    )


def _doc_url(cik: int, accession_number: str, filename: str) -> str:
    return (
        f"{_ARCHIVES_BASE}/{cik}/"
        f"{_accession_no_dashes(accession_number)}/{filename}"
    )


class _FetchError(Exception):
    """Internal: a recoverable error during fetch / parse — folds into
    the partial-row branch rather than propagating to the caller."""


class DetailFetcher:
    """Pulls primary documents and persists `filings` rows.

    The shared `EdgarClient` (S3.1, ADR-002 §6) is injected; never
    instantiated here.
    """

    def __init__(
        self,
        *,
        edgar: EdgarClient,
        db_path: str,
        raw_root: Path,
    ) -> None:
        self._edgar = edgar
        self._db_path = db_path
        self._raw_root = Path(raw_root)
        self._raw_root.mkdir(parents=True, exist_ok=True)

    async def fetch_and_persist(self, filing: DiscoveredFiling) -> int:
        """Fetch + parse + persist one filing. Returns the journal `filings.id`.

        On a duplicate accession (already-persisted), returns the existing id
        without re-fetching. On any non-fatal error during fetch/parse, a
        partial row is inserted with `ingest_state="partial"` and the error
        message; the returned id still references that row.
        """
        try:
            return await self._happy_path(filing)
        except DuplicateAccession as dup:
            _log.info(
                "filing already persisted; skipping",
                extra={
                    "accession": filing.accession_number,
                    "filing_id": dup.existing_id,
                },
            )
            return dup.existing_id
        except _FetchError as exc:
            return self._persist_partial(filing, str(exc))
        except EdgarUnavailable as exc:
            return self._persist_partial(filing, f"edgar_unavailable: {exc}")

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    async def _happy_path(self, filing: DiscoveredFiling) -> int:
        index_items = await self._fetch_index(filing)
        if filing.form_type == "4":
            primary_name = self._select_form4_primary(index_items)
            if primary_name is None:
                raise _FetchError("no Form 4 primary XML found in index")
            xml_bytes = await self._fetch_doc(filing, primary_name)
            try:
                doc = parse_form4_xml(xml_bytes)
            except Form4ParseError as e:
                raise _FetchError(f"form4 parse failed: {e}") from e
            blob = doc.to_normalized_json()
            raw_path = self._raw_root / f"{filing.accession_number}.json"
            raw_path.write_text(blob)
            row = FilingRow(
                accession_number=filing.accession_number,
                cik=filing.cik,
                form_type=filing.form_type,
                filed_at=filing.filed_at,
                fetched_at=self._now(),
                raw_text_path=str(raw_path),
                content_hash=content_hash(blob),
                item_codes=None,
                issuer_ticker=doc.issuer_ticker,
                ingest_state="ok",
                ingest_error=None,
            )
            return insert_filing(self._db_path, row)

        # Prose forms.
        primary_name = self._select_prose_primary(index_items, filing.form_type)
        if primary_name is None:
            raise _FetchError(
                f"no primary document found in index for form_type={filing.form_type}"
            )
        html_bytes = await self._fetch_doc(filing, primary_name)
        text = html_to_text(html_bytes)
        if not text:
            raise _FetchError("primary document yielded empty plaintext")
        raw_path = self._raw_root / f"{filing.accession_number}.txt"
        raw_path.write_text(text)
        item_codes_json: str | None = None
        if filing.form_type == "8-K":
            codes = extract_item_codes(text)
            item_codes_json = json.dumps(codes)
        row = FilingRow(
            accession_number=filing.accession_number,
            cik=filing.cik,
            form_type=filing.form_type,
            filed_at=filing.filed_at,
            fetched_at=self._now(),
            raw_text_path=str(raw_path),
            content_hash=content_hash(text),
            item_codes=item_codes_json,
            issuer_ticker=None,
            ingest_state="ok",
            ingest_error=None,
        )
        return insert_filing(self._db_path, row)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fetch_index(self, filing: DiscoveredFiling) -> list[dict[str, str]]:
        url = _index_url(filing.cik, filing.accession_number)
        resp = await self._edgar.get(url)
        if resp.status_code != 200:
            raise _FetchError(
                f"index.json fetch returned status {resp.status_code} for {url}"
            )
        try:
            payload = resp.json()
        except json.JSONDecodeError as e:
            raise _FetchError(f"index.json was not valid JSON: {e}") from e
        directory = payload.get("directory") if isinstance(payload, dict) else None
        if not isinstance(directory, dict):
            raise _FetchError("index.json missing 'directory' object")
        items = directory.get("item")
        if not isinstance(items, list):
            raise _FetchError("index.json 'directory.item' is not a list")
        return [i for i in items if isinstance(i, dict)]

    async def _fetch_doc(self, filing: DiscoveredFiling, filename: str) -> bytes:
        url = _doc_url(filing.cik, filing.accession_number, filename)
        resp = await self._edgar.get(url)
        if resp.status_code != 200:
            raise _FetchError(
                f"primary doc fetch returned status {resp.status_code} for {url}"
            )
        return resp.content

    @staticmethod
    def _select_prose_primary(
        items: list[dict[str, str]], form_type: str
    ) -> str | None:
        """Prefer entries whose `type` matches the filing form_type; tie-break
        by descending size; restrict to .htm / .html / .xml extensions to
        avoid picking up exhibit images or .txt indices.
        """
        candidates: list[tuple[int, str]] = []
        for it in items:
            name = it.get("name", "")
            ext = Path(name).suffix.lower()
            if ext not in {".htm", ".html", ".xml"}:
                continue
            if it.get("type") != form_type:
                continue
            try:
                size = int(it.get("size", "0") or 0)
            except (TypeError, ValueError):
                size = 0
            candidates.append((size, name))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    @staticmethod
    def _select_form4_primary(items: list[dict[str, str]]) -> str | None:
        for it in items:
            name = it.get("name", "")
            if Path(name).suffix.lower() == ".xml" and it.get("type") == "4":
                return name
        # Fallback: any .xml file in the index (some legacy filings drop the
        # `type` field on the primary).
        for it in items:
            name = it.get("name", "")
            if Path(name).suffix.lower() == ".xml":
                return name
        return None

    # ------------------------------------------------------------------
    # Partial / error path
    # ------------------------------------------------------------------

    def _persist_partial(self, filing: DiscoveredFiling, error: str) -> int:
        # `raw_text_path` is NOT NULL in the schema; for partials we record a
        # synthetic path under raw_root that does not exist on disk, so the
        # row is identifiable but the caller knows nothing was written.
        synthetic_path = str(
            self._raw_root / f"{filing.accession_number}.partial"
        )
        row = FilingRow(
            accession_number=filing.accession_number,
            cik=filing.cik,
            form_type=filing.form_type,
            filed_at=filing.filed_at,
            fetched_at=self._now(),
            raw_text_path=synthetic_path,
            content_hash="",
            item_codes=None,
            issuer_ticker=None,
            ingest_state="partial",
            ingest_error=error,
        )
        try:
            return insert_filing(self._db_path, row)
        except DuplicateAccession as dup:
            _log.info(
                "partial filing collided with existing row; keeping prior",
                extra={"accession": filing.accession_number},
            )
            return dup.existing_id

    @staticmethod
    def _now() -> dt.datetime:
        return dt.datetime.now(tz=dt.UTC)
