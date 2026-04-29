"""S4.1 — 8-K item-code allow/deny prefilter (D-020 / PRD §8.2).

Pure logic: given the parsed `item_codes` from an 8-K, decide accept or
reject before any LLM cost is incurred. Unknown codes fail closed
(treated as deny) so newly-introduced item codes never silently pass
through the filter.

`9.01` ("Financial Statements and Exhibits") is procedural — accepted
only when paired with at least one allow-list item.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

ALLOW: frozenset[str] = frozenset(
    {
        "1.01", "1.02", "1.03",
        "2.01", "2.02", "2.03", "2.04", "2.05", "2.06",
        "3.01", "3.02", "3.03",
        "4.01", "4.02",
        "5.01", "5.02", "5.03", "5.07",
        "7.01",
        "8.01",
    }
)

DENY: frozenset[str] = frozenset(
    {"5.04", "5.05", "5.06", "5.08", "6.01", "6.02", "6.03", "6.04", "6.05"}
)


class ItemCodeDecision(NamedTuple):
    decision: Literal["accept", "reject"]
    reason: str


class ItemCodePrefilter:
    def evaluate(self, item_codes: list[str]) -> ItemCodeDecision:
        if not item_codes:
            return ItemCodeDecision("reject", "item_code_empty")

        if any(code in ALLOW for code in item_codes):
            return ItemCodeDecision("accept", "item_code_allow")

        if item_codes == ["9.01"]:
            return ItemCodeDecision("reject", "item_code_9.01_only")

        return ItemCodeDecision("reject", "item_code_deny")
