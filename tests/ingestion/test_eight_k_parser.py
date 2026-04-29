"""S3.3 — 8-K item code extraction.

8-K filings declare which items they trigger via canonical "Item N.NN"
headings inside the primary document. The prefilter (S4.1) consumes these
to apply the §8.2 allow/deny list.
"""

from __future__ import annotations

from ingestion.parsers.eight_k import extract_item_codes


def test_8k_item_codes_extracted() -> None:
    plaintext = (
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n"
        "FORM 8-K\n\n"
        "Item 2.02 Results of Operations and Financial Condition\n"
        "On April 29 2026, ACME announced earnings ...\n\n"
        "Item 9.01 Financial Statements and Exhibits\n"
        "(d) Exhibits.\n"
    )
    codes = extract_item_codes(plaintext)
    assert codes == ["2.02", "9.01"]


def test_8k_item_codes_dedup_and_sorted() -> None:
    plaintext = "Item 8.01 ... see Item 1.01 ... Item 1.01 again ... Item 2.02"
    codes = extract_item_codes(plaintext)
    # Sorted ascending; deduped.
    assert codes == ["1.01", "2.02", "8.01"]


def test_8k_no_items_returns_empty() -> None:
    assert extract_item_codes("no item codes here") == []


def test_8k_ignores_section_numbers_in_body_text() -> None:
    # "Section 16" / "Rule 12b-2" must not collide with Item code regex.
    plaintext = "Section 16(a) ... Rule 12b-2 ... Item 5.07 was triggered."
    assert extract_item_codes(plaintext) == ["5.07"]
