"""S3.4 — submissions JSON parser.

Endpoint: `https://data.sec.gov/submissions/CIK{cik:010d}.json`. Real
payload has a `filings.recent` object with parallel arrays. We extract
only what the reconciler diff needs.
"""

from __future__ import annotations

import datetime as dt

from ingestion.parsers.submissions import SubmissionsEntry, parse_submissions

_FAKE_PAYLOAD = {
    "cik": "1234567",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0001234567-26-000010",
                "0001234567-26-000011",
                "0001234567-26-000012",
            ],
            "form": ["10-K", "8-K", "4"],
            "filingDate": ["2026-04-01", "2026-04-15", "2026-04-29"],
            "primaryDocument": ["10k.htm", "8k.htm", "form4.xml"],
            "primaryDocDescription": [
                "10-K",
                "8-K",
                "FORM 4",
            ],
        },
        "files": [],
    },
}


def test_parse_submissions_extracts_recent_entries() -> None:
    entries = parse_submissions(_FAKE_PAYLOAD, cik=1234567)
    assert len(entries) == 3
    e = entries[0]
    assert isinstance(e, SubmissionsEntry)
    assert e.accession_number == "0001234567-26-000010"
    assert e.form_type == "10-K"
    assert e.filed_at.date() == dt.date(2026, 4, 1)
    assert e.cik == 1234567
    assert e.primary_document == "10k.htm"


def test_parse_submissions_handles_empty_recent() -> None:
    payload = {
        "cik": "9999999",
        "filings": {
            "recent": {
                "accessionNumber": [],
                "form": [],
                "filingDate": [],
                "primaryDocument": [],
            },
            "files": [],
        },
    }
    assert parse_submissions(payload, cik=9999999) == []


def test_parse_submissions_skips_misshaped_rows() -> None:
    # Defensive: if upstream arrays diverge in length, take the min.
    payload = {
        "cik": "1234567",
        "filings": {
            "recent": {
                "accessionNumber": ["a", "b", "c"],
                "form": ["8-K", "10-Q"],
                "filingDate": ["2026-04-01", "2026-04-15"],
                "primaryDocument": ["x.htm", "y.htm"],
            },
        },
    }
    entries = parse_submissions(payload, cik=1234567)
    assert len(entries) == 2
    assert [e.accession_number for e in entries] == ["a", "b"]
