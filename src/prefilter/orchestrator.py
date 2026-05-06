"""S4.3 — Prefilter orchestrator.

Single entry-point the analyzer caller (S5.3 wiring) uses to decide whether a
filing reaches the LLM. Composes the universe gate, the 8-K item-code
prefilter (S4.1), and the similarity prefilter (S4.2) per ADR-003 / PRD §8.

Decision pipeline (story-04-03 AC-2):
  1. Universe gate (defense-in-depth) — out-of-universe → reject.
  2. 8-K only: item-code prefilter (S4.1). Reject → record `item_code_*`.
  3. Material 8-K bypass (FR-13) — 8-K + ANY allow-list code → accept WITHOUT
     running similarity.
  4. Form 4 — universe-only per PRD §8.3 / D-014; skip similarity.
  5. Otherwise (10-K, 10-Q, S-1, DEF 14A, non-bypass 8-Ks): run similarity.

Every evaluation writes one `prefilter_decisions` row (FR-14). Idempotent on
`filing_id` — replay returns the persisted decision without re-running the
pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from journal.models import FilingRow, PrefilterDecisionRow
from journal.repo import (
    get_prefilter_decision_by_filing,
    insert_prefilter_decision,
)
from observability.log_port import get_logger
from prefilter.item_codes import ALLOW as ITEM_ALLOW
from prefilter.item_codes import ItemCodePrefilter
from prefilter.routine_8k import RoutineEightKFilter
from prefilter.similarity import SimilarityChecker
from universe.api import Universe

log = get_logger(__name__)


@dataclass(frozen=True)
class PrefilterDecision:
    decision: Literal["accept", "reject"]
    rule_fired: str
    similarity_score: float | None = None
    reason_detail: str | None = None


def _parse_item_codes(raw: str | None) -> list[str]:
    """Decode `FilingRow.item_codes` (JSON list) into a Python list."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("invalid item_codes JSON; treating as empty", extra={"raw": raw})
        return []
    return [str(c) for c in parsed] if isinstance(parsed, list) else []


def _has_allow_item(codes: list[str]) -> bool:
    return any(c in ITEM_ALLOW for c in codes)


class Prefilter:
    """Composed prefilter for the analyzer caller.

    Constructor parameters:
      db_path:    SQLite journal path; used only to persist decisions.
      universe:   loaded `Universe` for membership checks (cheap lookup).
      similarity: an instance of `SimilarityChecker` (S4.2). Allowed to be a
                  `MagicMock` in tests where similarity must not be invoked.
    """

    def __init__(
        self,
        *,
        db_path: str,
        universe: Universe,
        similarity: SimilarityChecker,
        form_4_enabled: bool = True,
    ) -> None:
        self.db_path = db_path
        self.universe = universe
        self.similarity = similarity
        self.form_4_enabled = form_4_enabled
        self._item_code = ItemCodePrefilter()
        self._routine_8k = RoutineEightKFilter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, filing: FilingRow, raw_text: str) -> PrefilterDecision:
        if filing.id is None:
            raise ValueError(
                "FilingRow.id is required to persist prefilter decisions; "
                "callers must persist the filing before evaluation."
            )

        # Idempotency — replay returns the persisted decision unchanged.
        persisted = get_prefilter_decision_by_filing(self.db_path, filing.id)
        if persisted is not None:
            return PrefilterDecision(
                decision=persisted.decision,  # type: ignore[arg-type]
                rule_fired=persisted.rule_fired,
                similarity_score=persisted.similarity_score,
                reason_detail=persisted.reason_detail,
            )

        decision = self._decide(filing, raw_text)
        self._persist(filing.id, decision)
        return decision

    # ------------------------------------------------------------------
    # Internal — pipeline
    # ------------------------------------------------------------------

    def _decide(self, filing: FilingRow, raw_text: str) -> PrefilterDecision:
        # Stage 1 — universe (defense-in-depth).
        if not self.universe.is_in_universe_by_cik(filing.cik):
            return PrefilterDecision(
                decision="reject",
                rule_fired="universe",
                reason_detail=f"cik {filing.cik} not in universe",
            )

        # Stage 2 — 8-K item-code prefilter + routine-8-K filter +
        # material-8-K bypass.
        if filing.form_type == "8-K":
            codes = _parse_item_codes(filing.item_codes)
            ic_decision = self._item_code.evaluate(codes)
            if ic_decision.decision == "reject":
                return PrefilterDecision(
                    decision="reject",
                    rule_fired=ic_decision.reason,
                    reason_detail=f"item_codes={codes}",
                )
            # Routine-8-K filter (task #1): codes are in ALLOW but the
            # combination is provably non-tradeable (e.g., {5.07} voting,
            # {8.01} routine dividend). Drops before LLM cost is incurred.
            routine = self._routine_8k.evaluate(codes, raw_text=raw_text)
            if routine is not None:
                return PrefilterDecision(
                    decision="reject",
                    rule_fired=routine.rule_fired,
                    reason_detail=f"item_codes={codes}",
                )
            # Item-code prefilter accepted → at least one allow-list code is
            # present (S4.1's accept branch). Material 8-K bypass: skip
            # similarity entirely per FR-13.
            if _has_allow_item(codes):
                return PrefilterDecision(
                    decision="accept",
                    rule_fired="material_8k_bypass",
                    reason_detail=f"item_codes={codes}",
                )
            # Defensive fallthrough — shouldn't happen given S4.1's logic, but
            # if it ever does, fall through to similarity.

        # Stage 3 — Form 4 universe-only per PRD §8.3 / D-014.
        if filing.form_type == "4":
            if not self.form_4_enabled:
                return PrefilterDecision(
                    decision="reject",
                    rule_fired="form_4_disabled",
                )
            return PrefilterDecision(
                decision="accept",
                rule_fired="form_4_universe_only",
            )

        # Stage 4 — similarity (10-K, 10-Q, S-1, DEF 14A, non-bypass 8-Ks).
        sim = self.similarity.check(filing, raw_text)
        if sim.decision == "accept":
            return PrefilterDecision(
                decision="accept",
                rule_fired="pass",
                similarity_score=sim.score,
            )
        rule = (
            "similarity_minhash"
            if sim.reason == "minhash"
            else "similarity_tfidf"
        )
        return PrefilterDecision(
            decision="reject",
            rule_fired=rule,
            similarity_score=sim.score,
        )

    def _persist(self, filing_id: int, decision: PrefilterDecision) -> None:
        insert_prefilter_decision(
            self.db_path,
            PrefilterDecisionRow(
                filing_id=filing_id,
                decision=decision.decision,
                rule_fired=decision.rule_fired,
                similarity_score=decision.similarity_score,
                reason_detail=decision.reason_detail,
            ),
        )
