"""S3.3 — HTML-to-plaintext normalizer for EDGAR prose forms.

Output is the source of truth for similarity prefilter (S4.2) and the LLM
analyzer (S5.3); behavior must be deterministic and order-stable.
"""

from __future__ import annotations

from ingestion.parsers.html_to_text import html_to_text


def test_html_to_text_collapses_whitespace_and_strips_tags() -> None:
    html = b"""<html>
      <head><title>ignored</title></head>
      <body>
        <h1>Quarterly Report</h1>
        <p>Revenue   was   <b>$10M</b>     this <i>quarter</i>.</p>
        <table><tr><td>EPS</td><td>$0.42</td></tr></table>
      </body>
    </html>"""
    out = html_to_text(html)
    assert "Quarterly Report" in out
    assert "$10M" in out
    assert "$0.42" in out
    # No HTML tags survive.
    assert "<" not in out
    assert ">" not in out
    # Whitespace is collapsed (no runs of >1 space).
    assert "  " not in out


def test_html_to_text_strips_script_and_style() -> None:
    html = b"""<html><body>
      <script>document.write('SECRET');</script>
      <style>body { color: red; }</style>
      <p>Hello world</p>
    </body></html>"""
    out = html_to_text(html)
    assert "Hello world" in out
    assert "SECRET" not in out
    assert "color: red" not in out


def test_html_to_text_handles_empty() -> None:
    assert html_to_text(b"") == ""


def test_html_to_text_handles_xbrl_inline_tags() -> None:
    # XBRL inline tags carry visible text in the inner content.
    html = b"""<html><body>
      <ix:nonNumeric contextRef="ctx0">Total revenue</ix:nonNumeric>
      <ix:nonFraction>15000000</ix:nonFraction>
    </body></html>"""
    out = html_to_text(html)
    assert "Total revenue" in out
    assert "15000000" in out
