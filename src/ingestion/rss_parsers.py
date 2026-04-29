"""EDGAR full-text RSS Atom parser (S3.2, ADR-002 §1).

The full-text RSS endpoints (`https://www.sec.gov/cgi-bin/browse-edgar`
with `action=getcurrent&output=atom&type=<F>`) return Atom 1.0 feeds.
Each `<entry>` carries:

  - `<id>` — `urn:tag:sec.gov,2008:accession-number=NNNNNNNNNN-YY-NNNNNN`
  - `<title>` — "<form> - <ISSUER NAME> (<10-digit CIK>)"
  - `<updated>` — RFC 3339 timestamp (treated as `filed_at`)
  - `<category term="<form-type>">`

Only what the discovery loop needs is extracted. Detail fetching (HTML,
XBRL, Form 4 XML) is S3.3's job.
"""

from __future__ import annotations

import datetime as dt
import re

from lxml import etree
from pydantic import BaseModel, ConfigDict

_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
_ACCESSION_RE = re.compile(r"accession-number=([\d-]+)")
_CIK_RE = re.compile(r"\((\d{4,10})\)")

# Defense-in-depth (carry-forward S3.2 review L-1): refuse external entity
# resolution and outbound DTD fetches. Practical risk against SEC EDGAR is
# near zero — the egress firewall (architecture §9.5) already pins us to
# *.sec.gov — but tightening parser defaults is free hardening.
_SAFE_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)


class RssEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accession_number: str
    cik: int
    form_type: str
    filed_at: dt.datetime  # tz-aware; preserves the EDGAR-emitted offset


def parse_atom(xml_bytes: bytes) -> list[RssEntry]:
    """Parse one EDGAR Atom feed payload into structured entries.

    Malformed individual entries (missing id / title / category) are
    skipped rather than failing the whole feed — a partial poll is still
    useful per FR-11's "drop early" posture, and the reconciler (S3.4)
    will catch anything we miss. Top-level XML errors surface as
    `lxml.etree.XMLSyntaxError`.
    """
    root = etree.fromstring(xml_bytes, parser=_SAFE_PARSER)
    out: list[RssEntry] = []
    for entry in root.findall("a:entry", _ATOM_NS):
        parsed = _parse_entry(entry)
        if parsed is not None:
            out.append(parsed)
    return out


def _parse_entry(entry: etree._Element) -> RssEntry | None:
    id_el = entry.find("a:id", _ATOM_NS)
    if id_el is None or not id_el.text:
        return None
    m = _ACCESSION_RE.search(id_el.text)
    if not m:
        return None
    accession = m.group(1)

    title_el = entry.find("a:title", _ATOM_NS)
    if title_el is None or not title_el.text:
        return None
    cik_match = _CIK_RE.search(title_el.text)
    if not cik_match:
        return None
    cik = int(cik_match.group(1))

    cat_el = entry.find("a:category", _ATOM_NS)
    if cat_el is None:
        return None
    form_type = cat_el.get("term")
    if not form_type:
        return None

    upd_el = entry.find("a:updated", _ATOM_NS)
    if upd_el is None or not upd_el.text:
        return None
    try:
        filed_at = dt.datetime.fromisoformat(upd_el.text.strip())
    except ValueError:
        return None
    if filed_at.tzinfo is None:
        # Atom 1.0 mandates timezone offsets; if upstream sends naive, treat as UTC.
        filed_at = filed_at.replace(tzinfo=dt.UTC)

    return RssEntry(
        accession_number=accession,
        cik=cik,
        form_type=form_type,
        filed_at=filed_at,
    )
