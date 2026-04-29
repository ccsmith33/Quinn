"""EDGAR submissions JSON parser (S3.4, ADR-002 §4).

Endpoint: `https://data.sec.gov/submissions/CIK{cik:010d}.json`. The
`filings.recent` block has parallel arrays describing the issuer's most
recent ~1,000 filings. Older filings hang off `filings.files[]`
("page" pointers) and are not consulted at v1 — the reconciler only
needs the recent window (default 7 days).

JSON parsing is via the stdlib only (no XML, no XXE risk here).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict


class SubmissionsEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cik: int
    accession_number: str
    form_type: str
    filed_at: dt.datetime
    primary_document: str | None


def parse_submissions(payload: dict[str, Any], *, cik: int) -> list[SubmissionsEntry]:
    """Extract `recent` array entries from a submissions JSON payload.

    Defensive: if the parallel arrays diverge in length, we take the
    minimum so an upstream shape change does not raise IndexError.
    """
    filings = payload.get("filings") or {}
    recent = filings.get("recent") or {}
    accessions = recent.get("accessionNumber") or []
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    primaries = recent.get("primaryDocument") or []

    n = min(len(accessions), len(forms), len(dates))
    out: list[SubmissionsEntry] = []
    for i in range(n):
        try:
            filed_at = dt.datetime.fromisoformat(dates[i]).replace(tzinfo=dt.UTC)
        except (TypeError, ValueError):
            continue
        primary = primaries[i] if i < len(primaries) else None
        out.append(
            SubmissionsEntry(
                cik=cik,
                accession_number=str(accessions[i]),
                form_type=str(forms[i]),
                filed_at=filed_at,
                primary_document=str(primary) if primary is not None else None,
            )
        )
    return out
