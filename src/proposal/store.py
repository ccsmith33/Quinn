"""S5.5 — Proposal store: schema-validated, append-only persistence.

Architecture references: §2.4 (Proposal store), §3.2 (TradeProposal schema),
§3.3 (NoTradeRecord schema), FR-19, FR-27, FR-28, FR-29, FR-30, NFR-16.

Carry-forward note (D-038 partial-snapshot-read pattern from S2.3):
Per the lessons-learned register, any store that writes a "header" row plus
N "member" rows under per-row commits inherits the partial-snapshot read
hazard. **The proposal store does NOT inherit this hazard**:

- `proposals` is one row per LLM-call-result (no children).
- `proposal_reviews` is `UNIQUE(proposal_id)` — at most one review per
  proposal (architecture §8.1 line 614).
- `source_filings` is denormalized into the `raw_response` JSON, not a
  separate child table.

Read-side count-validation is therefore N/A here. If a v2 story adds child
rows (e.g., per-leg order tracking on `proposals`), that story must
re-evaluate.
"""

from __future__ import annotations

import json
from typing import Any

from analyzer.results import AnalyzerMalformed, AnalyzerResult, NoTrade, ProposalEmitted
from analyzer.telemetry import CallTelemetry
from journal.models import FilingRow, ProposalReviewRow, ProposalRow
from journal.repo import (
    get_proposal_by_decision_id,
    get_proposal_by_id,
    insert_proposal,
    insert_proposal_review,
)
from observability.log_port import get_logger

from .schemas import (
    validate_no_trade_record,
    validate_trade_proposal,
)

log = get_logger(__name__)

ProposalId = int

# Opus review result types — imported lazily inside store_review to avoid
# a hard cycle (S5.4 imports the proposal-store Protocol from `analyzer.opus`).
# We don't need the symbols at import time; only the runtime isinstance check.


class ProposalStore:
    """Validates and persists analyzer + reviewer outputs to the journal.

    Single source of truth for the journal (FR-30); every write is
    append-only (NFR-16). The store enriches LLM-emitted JSON with two
    system-injected fields before writing: `prompt_version` and
    `source_filings` (architecture §3.2 — both marked `system_injected`).
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Proposals (FR-19, FR-27)
    # ------------------------------------------------------------------

    def store(
        self,
        result: AnalyzerResult,
        *,
        filing: FilingRow,
        model_id: str,
        prompt_version: str,
        decision_id: str,
        raw_response: str,
        telemetry: CallTelemetry,
    ) -> ProposalId:
        """Validate `result` and persist a row to `proposals`.

        AC-4 idempotency: re-stores with the same `decision_id` return the
        existing `proposal_id` without writing a new row. The journal's
        `decision_id` UNIQUE constraint backs this — we look up first to
        avoid the IntegrityError round-trip on the hot path.
        """
        existing = get_proposal_by_decision_id(self._db_path, decision_id)
        if existing is not None:
            assert existing.id is not None
            log.debug(
                "proposal.store.idempotent",
                extra={
                    "event": "proposal.store.idempotent",
                    "decision_id": decision_id,
                    "existing_proposal_id": existing.id,
                },
            )
            return existing.id

        kind, validated_payload, denormalized = _validate_and_extract(result)
        enriched_response = _enrich_response(
            result=result,
            validated_payload=validated_payload,
            prompt_version=prompt_version,
            filing=filing,
            raw_response=raw_response,
        )
        row = ProposalRow(
            filing_id=filing.id,  # type: ignore[arg-type]
            decision_id=decision_id,
            model_id=model_id,
            prompt_version=prompt_version,
            raw_response=enriched_response,
            kind=kind,
            **denormalized,
            input_tokens=telemetry.input_tokens,
            output_tokens=telemetry.output_tokens,
            cache_read_tokens=telemetry.cache_read_tokens,
            cache_creation_tokens=telemetry.cache_creation_tokens,
            latency_ms=telemetry.latency_ms,
            cost_usd=telemetry.cost_usd,
            reasoning_notes=_reasoning_notes(result),
        )
        pid = insert_proposal(self._db_path, row)
        log.info(
            "proposal.store",
            extra={
                "event": "proposal.store",
                "proposal_id": pid,
                "decision_id": decision_id,
                "kind": kind,
                "model_id": model_id,
                "prompt_version": prompt_version,
            },
        )
        return pid

    def fetch(self, proposal_id: ProposalId) -> ProposalRow:
        """Return the proposal row by id. Raises `KeyError` if not found."""
        row = get_proposal_by_id(self._db_path, proposal_id)
        if row is None:
            raise KeyError(f"no proposal with id={proposal_id}")
        return row

    # ------------------------------------------------------------------
    # Proposal reviews (FR-17, FR-28)
    # ------------------------------------------------------------------

    def store_review(
        self,
        proposal_id: ProposalId,
        review: Any,  # OpusReviewResult — runtime-checked below to avoid import cycle
        *,
        model_id: str,
        prompt_version: str,
        raw_response: str,
        telemetry: CallTelemetry,
    ) -> None:
        """Persist an Opus review to `proposal_reviews`.

        The journal's `UNIQUE(proposal_id)` constraint enforces 0..1 reviews
        per proposal (architecture §8.1).
        """
        # Local import to avoid a cycle: `analyzer.opus` imports the
        # store Protocol; we only need runtime isinstance discrimination.
        from analyzer.opus import (
            OpusMalformed,
            OpusModified,
            OpusRatified,
            OpusRejected,
        )

        if isinstance(review, OpusRatified):
            decision = "ratify"
            rationale = review.rationale
            modifications_json: str | None = None
        elif isinstance(review, OpusModified):
            decision = "modify"
            rationale = review.rationale
            modifications_json = json.dumps(review.modifications, sort_keys=True)
        elif isinstance(review, OpusRejected):
            decision = "reject"
            rationale = review.rationale
            modifications_json = None
        elif isinstance(review, OpusMalformed):
            decision = "malformed"
            rationale = f"[parse error] {review.error}"
            modifications_json = None
        else:
            raise TypeError(f"unrecognized review type: {type(review).__name__}")

        row = ProposalReviewRow(
            proposal_id=proposal_id,
            model_id=model_id,
            prompt_version=prompt_version,
            decision=decision,
            raw_response=raw_response,
            rationale=rationale,
            modifications_json=modifications_json,
            input_tokens=telemetry.input_tokens,
            output_tokens=telemetry.output_tokens,
            cache_read_tokens=telemetry.cache_read_tokens,
            cache_creation_tokens=telemetry.cache_creation_tokens,
            latency_ms=telemetry.latency_ms,
            cost_usd=telemetry.cost_usd,
        )
        insert_proposal_review(self._db_path, row)
        log.info(
            "proposal.store_review",
            extra={
                "event": "proposal.store_review",
                "proposal_id": proposal_id,
                "decision": decision,
                "model_id": model_id,
                "prompt_version": prompt_version,
            },
        )


# ---------------------------------------------------------------------------
# Validation + enrichment helpers
# ---------------------------------------------------------------------------

def _validate_and_extract(
    result: AnalyzerResult,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    """Validate the LLM payload against its schema and return:
    (`kind`, validated_payload-or-None, denormalized-row-fields).

    - `ProposalEmitted` → kind=`trade_proposal`, validated TradeProposal,
      denormalized symbol/direction/size/conviction/thesis.
    - `NoTrade` → kind=`no_trade`, validated NoTradeRecord, no denormalized
      trade fields.
    - `AnalyzerMalformed` → kind=`no_trade`, validated_payload=None, no
      denormalized fields. (Per S5.3 AC-6 — malformed gets a journal row
      so the auditor sees the failure.)
    """
    if isinstance(result, ProposalEmitted):
        validated = validate_trade_proposal(result.payload).model_dump(
            exclude_none=True
        )
        denorm = {
            "symbol": validated["symbol"],
            "direction": validated["direction"],
            "size_pct_requested": validated["size_pct_of_capital"],
            "conviction": validated["conviction"],
            "thesis": validated["thesis"],
        }
        return "trade_proposal", validated, denorm
    if isinstance(result, NoTrade):
        validate_no_trade_record(result.payload)  # raise on invalid
        return "no_trade", result.payload, {}
    if isinstance(result, AnalyzerMalformed):
        return "no_trade", None, {}
    raise TypeError(f"unrecognized AnalyzerResult: {type(result).__name__}")


def _enrich_response(
    *,
    result: AnalyzerResult,
    validated_payload: dict[str, Any] | None,
    prompt_version: str,
    filing: FilingRow,
    raw_response: str,
) -> str:
    """Inject system fields (`prompt_version`, `source_filings`) into the
    response JSON before persistence (architecture §3.2 — both fields
    `system_injected`). Returns the enriched JSON string.

    For `AnalyzerMalformed`, the raw response is preserved verbatim — the
    LLM didn't emit a parseable object so we have nowhere to inject. The
    parse error lands in `reasoning_notes` instead (see `_reasoning_notes`).
    """
    if isinstance(result, AnalyzerMalformed):
        return raw_response
    assert validated_payload is not None
    enriched = dict(validated_payload)
    enriched["prompt_version"] = prompt_version
    enriched["source_filings"] = [_filing_to_source_entry(filing)]
    return json.dumps(enriched, sort_keys=True)


def _filing_to_source_entry(filing: FilingRow) -> dict[str, Any]:
    """Convert a `FilingRow` to the `source_filings[]` entry shape per
    architecture §3.2."""
    entry: dict[str, Any] = {
        "accession_number": filing.accession_number,
        "filing_type": filing.form_type,
    }
    if filing.item_codes:
        try:
            entry["item_codes"] = json.loads(filing.item_codes)
        except (json.JSONDecodeError, TypeError):
            # FilingRow.item_codes is JSON-encoded TEXT (per §8.1); if it
            # isn't valid JSON, surface the raw string rather than failing
            # the whole proposal write.
            entry["item_codes"] = [filing.item_codes]
    return entry


def _reasoning_notes(result: AnalyzerResult) -> str | None:
    """Per S5.3 AC-6: malformed analyzer output gets `reasoning_notes`
    capturing the parse failure so it remains discoverable in the journal."""
    if isinstance(result, AnalyzerMalformed):
        return f"[analyzer_malformed] {result.error}"
    return None


__all__ = [
    "ProposalId",
    "ProposalStore",
]
