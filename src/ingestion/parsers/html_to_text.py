"""HTML / inline-XBRL → plaintext for EDGAR prose forms (S3.3 AC-2).

Produces deterministic, whitespace-collapsed text used as the source of
truth for `filings.content_hash` and downstream similarity / LLM stages.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

_DROP_TAGS = ("script", "style")


def html_to_text(html_bytes: bytes) -> str:
    if not html_bytes:
        return ""
    soup = BeautifulSoup(html_bytes, "lxml")
    for tag_name in _DROP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse runs of whitespace to a single space; trim.
    return " ".join(text.split())
