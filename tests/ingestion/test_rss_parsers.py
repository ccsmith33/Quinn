"""S3.2 — EDGAR full-text RSS Atom parser.

ADR-002 §1: poll `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=<F>&output=atom`.
The Atom feed format is well-known; the parser extracts accession_number,
cik, form_type, filed_at from each entry.
"""

from __future__ import annotations

import datetime as dt

from ingestion.rss_parsers import RssEntry, parse_atom

# Real EDGAR Atom payload shape (one entry shown). Multiple entries in a real feed.
_ATOM_FIXTURE = b"""<?xml version="1.0" encoding="ISO-8859-1"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Latest Filings</title>
  <updated>2026-04-29T14:00:00-04:00</updated>
  <entry>
    <title>8-K - ACME MICROCAP CORP (0001234567) (Filer)</title>
    <link rel="alternate" type="text/html"
          href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0001234567&amp;type=8-K&amp;dateb=&amp;owner=include&amp;count=40"/>
    <summary type="html">
      &lt;b&gt;Form Type:&lt;/b&gt; 8-K&lt;br/&gt;
      &lt;b&gt;Filed:&lt;/b&gt; 2026-04-29&lt;br/&gt;
      &lt;b&gt;Accession Number:&lt;/b&gt; 0001234567-26-000099&lt;br/&gt;
    </summary>
    <updated>2026-04-29T13:55:00-04:00</updated>
    <category scheme="http://www.sec.gov/" label="form type" term="8-K"/>
    <id>urn:tag:sec.gov,2008:accession-number=0001234567-26-000099</id>
  </entry>
  <entry>
    <title>10-Q - WIDGET INC (0009876543) (Filer)</title>
    <link rel="alternate" type="text/html"
          href="https://www.sec.gov/Archives/edgar/data/9876543/000098765426000010/0000987654-26-000010-index.htm"/>
    <summary type="html">
      &lt;b&gt;Form Type:&lt;/b&gt; 10-Q&lt;br/&gt;
      &lt;b&gt;Filed:&lt;/b&gt; 2026-04-29&lt;br/&gt;
      &lt;b&gt;Accession Number:&lt;/b&gt; 0000987654-26-000010&lt;br/&gt;
    </summary>
    <updated>2026-04-29T13:50:30-04:00</updated>
    <category scheme="http://www.sec.gov/" label="form type" term="10-Q"/>
    <id>urn:tag:sec.gov,2008:accession-number=0000987654-26-000010</id>
  </entry>
</feed>
"""


def test_rss_entry_parsed() -> None:
    entries = parse_atom(_ATOM_FIXTURE)
    assert len(entries) == 2
    e0 = entries[0]
    assert isinstance(e0, RssEntry)
    assert e0.accession_number == "0001234567-26-000099"
    assert e0.cik == 1234567
    assert e0.form_type == "8-K"
    # filed_at carries timezone (we keep it tz-aware so latency math works
    # against any wall clock).
    assert e0.filed_at.tzinfo is not None
    assert e0.filed_at == dt.datetime.fromisoformat("2026-04-29T13:55:00-04:00")

    e1 = entries[1]
    assert e1.accession_number == "0000987654-26-000010"
    assert e1.cik == 9876543
    assert e1.form_type == "10-Q"


def test_empty_feed_returns_empty_list() -> None:
    payload = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Empty</title>
      <updated>2026-04-29T14:00:00-04:00</updated>
    </feed>
    """
    assert parse_atom(payload) == []


def test_skips_malformed_entry_without_id() -> None:
    payload = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>broken - no id</title>
    <updated>2026-04-29T13:55:00-04:00</updated>
    <category term="8-K"/>
  </entry>
  <entry>
    <title>8-K - ACME (0001234567)</title>
    <updated>2026-04-29T13:56:00-04:00</updated>
    <category term="8-K"/>
    <id>urn:tag:sec.gov,2008:accession-number=0001234567-26-000100</id>
  </entry>
</feed>
"""
    entries = parse_atom(payload)
    # A real EDGAR feed always carries id+title; we tolerate one missing id by
    # skipping it rather than crashing the whole poll.
    assert len(entries) == 1
    assert entries[0].accession_number == "0001234567-26-000100"
