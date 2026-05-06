"""Routine 8-K rules — drop filings whose item-code structure makes them
~never tradeable BEFORE the LLM ever sees them. Pure logic on
(item_codes, raw_text); deterministic; no I/O.

Each rule has a unique `rule_fired` value so the journal's
`prefilter_decisions` table tracks which rule killed each filing.

Conservative drop list (task #1, day-4 cost-cuts):
  - {5.07} (with or without 9.01) — annual meeting voting.
  - {5.02} (with or without 9.01) AND no "appointed"/"elected" in body —
    officer retirement / departure with no incoming successor.
  - {8.01} (with or without 9.01) AND body has 'dividend'/'distribution'
    near a numeric amount AND no 'guidance'/'outlook'/'agreement' terms.
  - {7.01} (with or without 9.01) AND body has 'investor presentation' or
    'conference' AND no 'regulation fd' substance.

Substantive items (1.01, 2.01, 2.02, ...) — when present alongside any of
the above, the filter does NOT fire. The LLM still sees those filings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Item codes treated as exhibit-list metadata only — they don't change the
# routine nature of a filing on their own.
_METADATA_CODES: frozenset[str] = frozenset({"9.01"})

# Body-scan budget: only the leading window of `raw_text` is searched for
# routine-pattern keywords. Post-task-#3, raw_text can run to ~1 MB
# (512 KB body + 512 KB cumulative exhibits). The patterns we look for —
# dividend declarations, IR/conference posts, retirement announcements —
# always appear in the cover page or the headline of EX-99.1, never deep
# in a supplement. 16 KB comfortably covers a typical cover page plus the
# top of any attached press release. Bounding here keeps the prefilter
# O(constant) per filing instead of O(filing size).
_BODY_SCAN_BYTES = 16 * 1024

_APPOINTMENT_KEYWORDS = ("appointed", "elected")
_DIVIDEND_KEYWORDS = ("dividend", "distribution")
_DIVIDEND_DISQUALIFIERS = ("guidance", "outlook", "agreement")
_IR_KEYWORDS = ("investor presentation", "conference")
_REG_FD_DISQUALIFIER = "regulation fd"

# Numeric-amount near a dividend/distribution keyword: a dollar amount or a
# bare decimal within ~80 chars on either side. The window is generous
# because filings often say e.g. "declared a regular dividend ... of $0.42".
_AMOUNT_PATTERN = re.compile(r"\$\s?\d|[\d.]+\s*(?:per share|cents?)|\d+\.\d+")


@dataclass(frozen=True)
class RoutineDecision:
    rule_fired: str


def _substantive_codes(codes: list[str]) -> set[str]:
    """Codes that aren't metadata. The 'pattern' is `set(item_codes) - {9.01}`."""
    return {c for c in codes if c not in _METADATA_CODES}


def _has_amount_near_keyword(
    body: str, lower_body: str, keywords: tuple[str, ...]
) -> bool:
    """True if any of `keywords` appears in `lower_body` and a numeric
    amount appears within ~80 chars of it in `body`. Caller passes the
    already-lowercased view to avoid repeat allocations."""
    for kw in keywords:
        idx = lower_body.find(kw)
        while idx != -1:
            window = body[max(0, idx - 80) : idx + len(kw) + 80]
            if _AMOUNT_PATTERN.search(window):
                return True
            idx = lower_body.find(kw, idx + 1)
    return False


def _contains_any(lower_body: str, terms: tuple[str, ...]) -> bool:
    return any(t in lower_body for t in terms)


class RoutineEightKFilter:
    """Routine-8-K post-allow-list filter.

    `evaluate(codes, raw_text)` returns a `RoutineDecision` with
    `rule_fired` if the filing matches a routine pattern; otherwise
    `None`. Callers (the orchestrator) treat `None` as "not routine,
    proceed". Body scans are bounded to the leading
    `_BODY_SCAN_BYTES` chars of `raw_text` (see module docstring).
    """

    def evaluate(
        self, item_codes: list[str], *, raw_text: str
    ) -> RoutineDecision | None:
        substantive = _substantive_codes(item_codes)
        if not substantive:
            return None

        # 5.07 voting needs no body scan — pure code match.
        if substantive == {"5.07"}:
            return RoutineDecision(rule_fired="routine_5_07_voting")

        # Bounded body scan: slice once, lower once, reuse everywhere.
        head = raw_text[:_BODY_SCAN_BYTES]
        lower_head = head.lower()

        if substantive == {"5.02"}:
            if not _contains_any(lower_head, _APPOINTMENT_KEYWORDS):
                return RoutineDecision(rule_fired="routine_5_02_retirement")
            return None

        if substantive == {"8.01"}:
            if _contains_any(lower_head, _DIVIDEND_DISQUALIFIERS):
                return None
            if _has_amount_near_keyword(head, lower_head, _DIVIDEND_KEYWORDS):
                return RoutineDecision(rule_fired="routine_8_01_dividend")
            return None

        if substantive == {"7.01"}:
            if _REG_FD_DISQUALIFIER in lower_head:
                return None
            if _contains_any(lower_head, _IR_KEYWORDS):
                return RoutineDecision(rule_fired="routine_7_01_ir_conference")
            return None

        return None
