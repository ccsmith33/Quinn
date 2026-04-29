"""8-K item-code extractor (S3.3 AC-4).

8-K filings declare which items they trigger via canonical "Item N.NN"
headings inside the primary document. The prefilter (S4.1) consumes the
extracted code list to apply the §8.2 allow/deny list.
"""

from __future__ import annotations

import re

# Match "Item " + one-or-two digits + dot + two digits, case-insensitive.
# Anchored to a word boundary so "Section 16" / "Rule 12b-2" / random
# "16.07" body text don't false-match.
_ITEM_RE = re.compile(r"\bItem\s+(\d{1,2}\.\d{2})\b", re.IGNORECASE)


def extract_item_codes(text: str) -> list[str]:
    """Return a deduped, ascending-sorted list of "N.NN" item codes.

    Sorting by string is fine for "N.NN" because both halves are
    fixed-width left-padded — matches numeric order for the 8-K item
    range (1.01 .. 9.01).
    """
    found = {m.group(1) for m in _ITEM_RE.finditer(text)}
    return sorted(found)
