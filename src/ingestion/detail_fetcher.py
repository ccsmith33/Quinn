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
import re
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

# ADR-007 Stage 1: regex exclusion patterns and form-type filename hints for
# the prose primary-doc heuristic. EDGAR `index.json` `type` field is icon-
# name strings (text.gif, …), so selection is filename-driven.
_STAGE1_ALLOW_EXTENSIONS = frozenset({".htm", ".html", ".xml"})
_STAGE1_EXCLUDE_PATTERNS = (
    # ADR-007 §"Identification heuristic spec" — narrow exhibit pattern. The
    # `ex` token must be at the start of the basename or preceded by `-`/`_`;
    # this rejects the EDGAR-canonical `ex-99-1.htm` / `exhibit_10-1.htm` /
    # `ex99.htm` shapes without false-positive risk on primaries that may
    # legitimately contain `ex` after a letter (`apex-2024.htm`,
    # `vertex-10k.htm`). Agent-style filenames like Apple's `a8-kex991*.htm`
    # slip past this filter intentionally — they're caught downstream by
    # `_stage1_is_contested` triggering the Stage 2 SGML safety net.
    re.compile(r"(^|[-_])ex(hibit)?[-_]?\d", re.IGNORECASE),
    re.compile(r"^r\d+\.htm$", re.IGNORECASE),
    re.compile(
        r"\.xsd$|_cal\.xml$|_def\.xml$|_lab\.xml$|_pre\.xml$|_htm\.xml$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^FilingSummary\.xml$|^MetaLinks\.json$|financial.report\.xlsx$",
        re.IGNORECASE,
    ),
)
# Looser pattern used ONLY by `_stage1_is_contested` to detect agent-style
# exhibit names (e.g. `a8-kex991*.htm`) that slipped past the narrow Stage 1
# exclusion. Triggers Stage 2 SGML resolution; never excludes a candidate.
_STAGE1_LOOSE_EXHIBIT_PATTERN = re.compile(r"ex(hibit)?[-_]?\d", re.IGNORECASE)
_STAGE1_FORM_TYPE_HINTS: dict[str, tuple[str, ...]] = {
    "10-K": ("10-k", "10k"),
    "10-Q": ("10-q", "10q"),
    "8-K": ("8-k", "8k"),
    "S-1": ("s-1", "s1"),
    "DEF 14A": ("def14a", "def-14a"),
    "425": ("425",),
}

# ADR-007 Stage 2: per-block byte cap to bound parse cost on pathological SGML.
_SGML_BLOCK_BYTE_CAP = 64 * 1024
_SGML_FIELD_RES: dict[str, re.Pattern[bytes]] = {
    "TYPE": re.compile(rb"^<TYPE>(.+?)\s*$", re.MULTILINE),
    "SEQUENCE": re.compile(rb"^<SEQUENCE>(.+?)\s*$", re.MULTILINE),
    "FILENAME": re.compile(rb"^<FILENAME>(.+?)\s*$", re.MULTILINE),
}


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


def _sgml_url(cik: int, accession_number: str) -> str:
    """ADR-007 Stage 2: canonical full-submission `<accession>.txt` URL."""
    return (
        f"{_ARCHIVES_BASE}/{cik}/"
        f"{_accession_no_dashes(accession_number)}/{accession_number}.txt"
    )


def _stage1_is_contested(
    items: list[dict[str, str]], stage1_pick: str, dominance_ratio: float = 2.0
) -> bool:
    """Return True when Stage 1's pick should be cross-checked against the
    SGML-header authority. Two triggers:

    1. **Looks-exhibit-like.** Stage 1's exclusion regex is intentionally
       narrow (ADR-007 §"Identification heuristic spec"); agent-style
       filenames such as Apple's `a8-kex991*.htm` slip past it. A looser
       post-pick check on the picked filename catches these.
    2. **Non-dominant size.** ADR-007 §"Why (c) over (a)-only" calls out
       exhibit-larger-than-primary as a known failure mode for size-only
       selection; if the picked size is less than `dominance_ratio` × the
       next-largest viable candidate, ask Stage 2 to disambiguate.

    "Viable" = same allow-list extension as Stage 1 (.htm/.html/.xml) and
    not excluded by Stage 1's exclusion regexes — i.e., the candidates
    Stage 1 was actually choosing among.
    """
    if _STAGE1_LOOSE_EXHIBIT_PATTERN.search(stage1_pick):
        return True
    sizes: list[int] = []
    pick_size = 0
    for it in items:
        name = it.get("name", "")
        ext = Path(name).suffix.lower()
        if ext not in _STAGE1_ALLOW_EXTENSIONS:
            continue
        if any(p.search(name) for p in _STAGE1_EXCLUDE_PATTERNS):
            continue
        try:
            size = int(it.get("size", "0") or 0)
        except (TypeError, ValueError):
            size = 0
        sizes.append(size)
        if name == stage1_pick:
            pick_size = size
    if len(sizes) < 2:
        return False
    sizes.sort(reverse=True)
    runner_up = sizes[1]
    if pick_size <= 0 or runner_up <= 0:
        return False
    return pick_size < runner_up * dominance_ratio


def _parse_sgml_filename_for_type(body: bytes, form_type: str) -> str | None:
    """ADR-007 Stage 2 parser.

    Scan `<DOCUMENT>` blocks; for each, read only its prologue (the
    metadata lines before its `<TEXT>` body, capped at
    `_SGML_BLOCK_BYTE_CAP`) and extract TYPE/SEQUENCE/FILENAME. Return
    the FILENAME of the lowest-SEQUENCE block whose TYPE matches
    `form_type` (case-insensitive). Returns None on any malformation.
    """
    target = form_type.upper().encode()
    matched: list[tuple[int, bytes]] = []
    pos = 0
    open_tag = b"<DOCUMENT>"
    text_tag = b"<TEXT>"
    while True:
        start = body.find(open_tag, pos)
        if start < 0:
            break
        # Bound the prologue scan to the smaller of: byte cap, next <TEXT>,
        # or next <DOCUMENT>. If another <DOCUMENT> opens before this
        # block's <TEXT>, the block is unclosed and we skip it.
        scan_limit = start + _SGML_BLOCK_BYTE_CAP
        prologue_end = body.find(text_tag, start + len(open_tag), scan_limit)
        next_doc_at = body.find(open_tag, start + len(open_tag), scan_limit)
        if prologue_end < 0:
            # Malformed block (no <TEXT> inside cap) — skip it.
            pos = start + len(open_tag)
            continue
        if next_doc_at >= 0 and next_doc_at < prologue_end:
            # Another <DOCUMENT> opens before this one closes its prologue
            # → unclosed/truncated block. Resume at the next opener.
            pos = next_doc_at
            continue
        prologue = body[start:prologue_end]
        type_match = _SGML_FIELD_RES["TYPE"].search(prologue)
        filename_match = _SGML_FIELD_RES["FILENAME"].search(prologue)
        seq_match = _SGML_FIELD_RES["SEQUENCE"].search(prologue)
        if type_match is None or filename_match is None:
            pos = prologue_end + len(text_tag)
            continue
        if type_match.group(1).strip().upper() != target:
            pos = prologue_end + len(text_tag)
            continue
        try:
            seq = int(seq_match.group(1).strip()) if seq_match else 999
        except ValueError:
            seq = 999
        matched.append((seq, filename_match.group(1).strip()))
        pos = prologue_end + len(text_tag)
    if not matched:
        return None
    matched.sort(key=lambda t: t[0])
    try:
        return matched[0][1].decode("ascii")
    except UnicodeDecodeError:
        return None


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

        # Prose forms (ADR-007).
        primary_name = self._select_prose_primary(index_items, filing.form_type)
        # Trigger Stage 2 SGML fallback when Stage 1 returned None OR when
        # Stage 1's pick is contested (multiple viable survivors with no
        # dominant size winner — the exhibit-larger-than-primary case the
        # ADR §"Why approach (c) over (a)-only" calls out). SGML is the
        # authoritative tiebreaker.
        if primary_name is None or _stage1_is_contested(index_items, primary_name):
            sgml_pick = await self._select_via_sgml_header(
                filing.cik, filing.accession_number, filing.form_type
            )
            if sgml_pick is not None:
                primary_name = sgml_pick
        if primary_name is None:
            raise _FetchError(
                "primary doc unresolved (heuristic + SGML fallback both failed) "
                f"for form_type={filing.form_type}"
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
        """ADR-007 Stage 1: heuristic primary-doc selection.

        Real EDGAR `index.json` `type` values are icon-name strings
        (`text.gif`, `compressed.gif`, …) — NOT form types. Selection is
        filename-driven: allow `.htm`/`.html`/`.xml`, exclude exhibit /
        XBRL-viewer / XBRL-accessory patterns, prefer filenames containing
        the form-type slug (soft hint), tie-break by largest size. Returns
        None when zero candidates survive — caller proceeds to Stage 2.
        """
        survivors: list[tuple[int, str]] = []
        for it in items:
            name = it.get("name", "")
            ext = Path(name).suffix.lower()
            if ext not in _STAGE1_ALLOW_EXTENSIONS:
                continue
            if any(p.search(name) for p in _STAGE1_EXCLUDE_PATTERNS):
                continue
            try:
                size = int(it.get("size", "0") or 0)
            except (TypeError, ValueError):
                size = 0
            survivors.append((size, name))
        if not survivors:
            return None
        survivors.sort(reverse=True)
        # ADR-007 §"Identification heuristic spec" (post-2026-05-03 patch):
        # form-type token is a SOFT positive hint that fires ONLY as a
        # tie-breaker when multiple candidates share the top size. It MUST
        # NOT displace the un-hinted size winner — exhibits commonly carry
        # the form-type token (e.g., Apple's `a10-kexhibit*.htm`) while the
        # primary often does not (e.g., `aapl-20240928.htm`), so a
        # hint-as-filter would invert the result.
        hints = _STAGE1_FORM_TYPE_HINTS.get(form_type.upper(), ())
        if hints:
            top_size = survivors[0][0]
            tied_at_top = [(s, n) for (s, n) in survivors if s == top_size]
            if len(tied_at_top) > 1:
                hinted = [
                    (s, n) for (s, n) in tied_at_top
                    if any(h in n.lower() for h in hints)
                ]
                if hinted:
                    return hinted[0][1]
        return survivors[0][1]

    async def _select_via_sgml_header(
        self, cik: int, accession_number: str, form_type: str
    ) -> str | None:
        """ADR-007 Stage 2: SGML-header fallback.

        Fetch `<accession>.txt`, parse `<DOCUMENT>` blocks in the header
        region, and return the FILENAME of the block whose TYPE matches
        `form_type` (preferring SEQUENCE=1). Returns None on any failure;
        the caller folds that into the partial-row error path.
        """
        url = _sgml_url(cik, accession_number)
        try:
            resp = await self._edgar.get(url)
        except EdgarUnavailable:
            return None
        if resp.status_code != 200:
            return None
        return _parse_sgml_filename_for_type(resp.content, form_type)

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
