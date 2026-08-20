"""S5.4 — Opus high-conviction proposal reviewer.

Architecture references: §2.3 (LLM Analyzer Opus path), §3.4 (Opus review
schema), FR-17, FR-18, FR-28, ADR-005.

Per FR-17 / FR-18: proposals at conviction ≥ threshold (default 7) route to
Opus 4.7 for review. Opus emits structured `ratify | modify | reject` per
architecture §3.4. The result is persisted to `proposal_reviews` (FR-28).
Modifications produce a *working copy* of the proposal for execution; the
original journal row is not mutated (NFR-16, append-only).

Note on caching (carry-forward from reviewer-e5's S5.1 review, advisory
A-2): Opus's block-2 (proposal-payload context) is per-call rather than
daily-stable — every reviewed proposal has a different `decision_id`,
`model_id`, and `prompt_version`. NFR-12 cache-hit on Opus calls is
therefore limited to block 1 (role / rules / schema). This is by design
for v1; reviewer-e5 may revisit if Opus volume grows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from journal.models import FilingRow, ProposalRow
from journal.repo import get_llm_calls_by_decision_id
from observability.log_port import get_logger
from prompts.loader import ApiRequest

from .sonnet import cap_filing_text, truncation_marker
from .telemetry import CallTelemetry

log = get_logger(__name__)

# Belt-and-braces ceiling for the filing text handed to the reviewer when the
# analyzer input cap is not wired (`max_input_chars=0`, feature off). Every
# call path already caps via `cap_filing_text` before handing raw_text over
# (sonnet day-0 `_apply_input_cap`, `AgentLoop._review_deferred_entry`,
# `RetroFillCoordinator.run_tick`), so this only fires on a future call site
# that forgets to. Generous by design: the reviewer must see the same case
# file as the analyst, never a re-truncated one.
_UNWIRED_SAFETY_CAP_CHARS = 400_000


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpusRatified:
    """Opus accepted the proposal as-is. Execution sees the original."""

    proposal: ProposalRow
    rationale: str


@dataclass(frozen=True)
class OpusModified:
    """Opus accepted with overlays. `working_proposal` is a non-persisted
    copy of the original with modifications applied; the original journal
    row is unchanged (NFR-16)."""

    proposal: ProposalRow
    working_proposal: ProposalRow
    rationale: str
    modifications: dict[str, Any]


@dataclass(frozen=True)
class OpusRejected:
    """Opus rejected the proposal. Execution is not invoked."""

    proposal: ProposalRow
    rationale: str


@dataclass(frozen=True)
class OpusMalformed:
    """Opus response could not be parsed as a valid review even after
    retry. Execution is not invoked; the proposal is held."""

    proposal: ProposalRow
    raw_response: str
    error: str


OpusReviewResult = OpusRatified | OpusModified | OpusRejected | OpusMalformed


# ---------------------------------------------------------------------------
# Protocols (boundaries to S5.5 + S5.2)
# ---------------------------------------------------------------------------

class _ProposalStorePort(Protocol):
    """Subset of S5.5's `ProposalStore` used here. S5.5 will provide the
    real implementation; we depend on the contract, not the class."""

    def store_review(
        self,
        proposal_id: int,
        review: OpusReviewResult,
        *,
        model_id: str,
        prompt_version: str,
        raw_response: str,
        telemetry: CallTelemetry,
    ) -> None: ...


class _AnthropicClientPort(Protocol):
    async def call(
        self,
        request: ApiRequest,
        *,
        model_id: str,
        purpose: Literal["analyze", "review", "replay", "audit"],
        decision_id: str,
        max_tokens: int | None = None,
    ) -> str: ...


class _PromptBuilderPort(Protocol):
    def build_opus_proposal_review(
        self, proposal: ProposalRow, source_filing_text: str
    ) -> ApiRequest: ...


# ---------------------------------------------------------------------------
# Reviewer
# ---------------------------------------------------------------------------

class OpusReviewer:
    """Runs the Opus review path for high-conviction proposals."""

    def __init__(
        self,
        *,
        client: _AnthropicClientPort,
        store: _ProposalStorePort,
        prompt_builder: _PromptBuilderPort,
        opus_model_id: str,
        ks4_pct_cap: float,
        db_path: str,
        max_output_tokens: int | None = None,
        max_input_chars: int = 0,
    ) -> None:
        self._client = client
        self._store = store
        self._builder = prompt_builder
        self._opus_model_id = opus_model_id
        self._ks4_pct_cap = ks4_pct_cap
        self._db_path = db_path
        # S5.6 carry-fwd S5.4 reviewer / D-052 — Opus per-call token cap.
        self._max_output_tokens = max_output_tokens
        # Evidence-symmetry fix — same filing-body ceiling the day-0
        # analyzer applies (`analyzer.analyzer_max_input_chars`). Callers
        # already cap; this is the defensive backstop. 0 = unwired, fall
        # back to `_UNWIRED_SAFETY_CAP_CHARS`.
        self._max_input_chars = max_input_chars

    async def review(
        self,
        proposal: ProposalRow,
        source_filing: FilingRow,
        raw_text: str,
    ) -> OpusReviewResult:
        raw_text = self._defensive_cap(source_filing, raw_text)
        request = self._builder.build_opus_proposal_review(
            proposal, _filing_context(source_filing, raw_text)
        )
        raw = await self._client.call(
            request,
            model_id=self._opus_model_id,
            purpose="review",
            decision_id=proposal.decision_id,
            max_tokens=self._max_output_tokens,
        )
        result = _parse_review(proposal, raw, ks4_pct_cap=self._ks4_pct_cap)
        # AC-3: retry once on malformed with stricter instruction (mirrors
        # S5.3's pattern). The retry uses an augmented request that adds a
        # corrective instruction; the second response's telemetry is what
        # we persist (the first attempt is recorded in `llm_calls` for audit).
        if isinstance(result, OpusMalformed):
            log.warning(
                "opus.review.malformed",
                extra={
                    "event": "opus.review.malformed",
                    "decision_id": proposal.decision_id,
                    "error": result.error,
                    "attempt": 1,
                },
            )
            retry_request = _augment_request_for_retry(request, prior_error=result.error)
            raw = await self._client.call(
                retry_request,
                model_id=self._opus_model_id,
                purpose="review",
                decision_id=proposal.decision_id,
                max_tokens=self._max_output_tokens,
            )
            result = _parse_review(proposal, raw, ks4_pct_cap=self._ks4_pct_cap)
            if isinstance(result, OpusMalformed):
                log.error(
                    "opus.review.malformed",
                    extra={
                        "event": "opus.review.malformed",
                        "decision_id": proposal.decision_id,
                        "error": result.error,
                        "attempt": 2,
                    },
                )
        telemetry = self._read_telemetry(proposal.decision_id)
        self._store.store_review(
            proposal.id,  # type: ignore[arg-type]
            result,
            model_id=self._opus_model_id,
            prompt_version=request.prompt_version,
            raw_response=raw,
            telemetry=telemetry,
        )
        return result

    def _defensive_cap(self, filing: FilingRow, raw_text: str) -> str:
        """Backstop cap on the filing text — callers cap upstream.

        Every call path applies `cap_filing_text` at the analyzer's cap
        before calling `review()`, so a properly capped body (at most
        cap + truncation-marker length) passes through byte-identical —
        re-capping it would truncate the upstream marker and stack a
        second one. Only a body that exceeds cap + marker (an uncapped
        future call site, or an unwired cap with a pathological filing)
        gets trimmed here.
        """
        cap = self._max_input_chars or _UNWIRED_SAFETY_CAP_CHARS
        if len(raw_text) <= cap + len(truncation_marker(cap)):
            return raw_text
        log.warning(
            "opus.review.uncapped_input",
            extra={
                "event": "opus.review.uncapped_input",
                "filing_id": filing.id,
                "chars": len(raw_text),
                "cap": cap,
            },
        )
        return cap_filing_text(raw_text, cap=cap, filing_id=filing.id)

    def _read_telemetry(self, decision_id: str) -> CallTelemetry:
        rows = get_llm_calls_by_decision_id(self._db_path, decision_id)
        review_rows = [r for r in rows if r.purpose == "review"]
        if not review_rows:
            raise RuntimeError(
                f"no llm_calls row for decision_id={decision_id!r}, purpose='review' "
                f"— AnthropicClient should have written one"
            )
        last = review_rows[-1]
        return CallTelemetry(
            input_tokens=last.input_tokens,
            output_tokens=last.output_tokens,
            cache_read_tokens=last.cache_read_tokens,
            cache_creation_tokens=last.cache_creation_tokens,
            latency_ms=last.latency_ms,
            cost_usd=last.cost_usd,
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_review(
    proposal: ProposalRow, raw: str, *, ks4_pct_cap: float
) -> OpusReviewResult:
    """Parse Opus's raw response into one of the four result types.

    Performs structural validation against the §3.4 schema inline (no
    `jsonschema` dep needed at v1). Defense-in-depth — S5.5 will re-validate
    at the persistence boundary.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        return OpusMalformed(proposal=proposal, raw_response=raw, error=f"json: {e}")
    err = _structural_validate(payload)
    if err is not None:
        return OpusMalformed(proposal=proposal, raw_response=raw, error=err)
    decision = payload["decision"]
    rationale = payload["rationale"]
    if decision == "ratify":
        return OpusRatified(proposal=proposal, rationale=rationale)
    if decision == "reject":
        return OpusRejected(proposal=proposal, rationale=rationale)
    # decision == "modify"
    mods = payload.get("modifications") or {}
    overlay_err = _validate_modifications(mods, ks4_pct_cap=ks4_pct_cap)
    if overlay_err is not None:
        # AC-5: invalid overlay — treat as a reject with rationale that
        # explains why. The original Opus rationale is preserved alongside
        # the overlay-failure note.
        return OpusRejected(
            proposal=proposal,
            rationale=(
                f"[opus modify treated as reject — invalid overlay: {overlay_err}] "
                f"{rationale}"
            ),
        )
    working = _apply_overlay(proposal, mods)
    return OpusModified(
        proposal=proposal,
        working_proposal=working,
        rationale=rationale,
        modifications=dict(mods),
    )


def _structural_validate(payload: Any) -> str | None:
    """Lightweight JSON-Schema-equivalent structural check on the §3.4
    schema. Returns an error message string or None if valid."""
    if not isinstance(payload, dict):
        return "top-level must be an object"
    decision = payload.get("decision")
    if decision not in ("ratify", "modify", "reject"):
        return f"decision must be ratify|modify|reject, got {decision!r}"
    rationale = payload.get("rationale")
    if not isinstance(rationale, str):
        return "rationale must be a string"
    if not (50 <= len(rationale) <= 4000):
        return f"rationale length {len(rationale)} outside [50, 4000]"
    if "modifications" in payload and payload["modifications"] is not None:
        if not isinstance(payload["modifications"], dict):
            return "modifications must be an object"
        allowed = {
            "size_pct_of_capital",
            "stop_loss_price",
            "take_profit_price",
            "exit_conditions",
        }
        extra = set(payload["modifications"]) - allowed
        if extra:
            return f"modifications has unknown keys: {sorted(extra)}"
    return None


def _validate_modifications(
    mods: dict[str, Any], *, ks4_pct_cap: float
) -> str | None:
    """Validate modification overlay values are realistic. Returns an error
    message if any constraint is violated, else None."""
    if "size_pct_of_capital" in mods:
        v = mods["size_pct_of_capital"]
        if not isinstance(v, (int, float)) or v <= 0 or v > ks4_pct_cap:
            return f"size_pct_of_capital {v!r} not in (0, {ks4_pct_cap}]"
    for px_field in ("stop_loss_price", "take_profit_price"):
        if px_field in mods:
            v = mods[px_field]
            if not isinstance(v, (int, float)) or v <= 0:
                return f"{px_field} {v!r} must be a positive number"
    if "exit_conditions" in mods:
        v = mods["exit_conditions"]
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            return "exit_conditions must be a list of strings"
    return None


def _apply_overlay(proposal: ProposalRow, mods: dict[str, Any]) -> ProposalRow:
    """Return a working copy of `proposal` with `mods` applied. The
    original is not mutated (NFR-16)."""
    update: dict[str, Any] = {}
    if "size_pct_of_capital" in mods:
        update["size_pct_requested"] = float(mods["size_pct_of_capital"])
    # stop_loss_price / take_profit_price / exit_conditions live on the
    # execution-side trade plan, not on `proposals` row columns; they ride
    # along on the OpusModified.modifications dict (journaled to
    # `proposal_reviews.modifications_json`) and are applied to the
    # in-memory working trade by the agent loop's reviewer overlay
    # (`app.loop._reviewer_trade_overlay`) before validation / sizing /
    # order construction. We keep them out of the working_proposal
    # row-copy to preserve the schema invariant.
    return proposal.model_copy(update=update)


def _augment_request_for_retry(request: ApiRequest, *, prior_error: str) -> ApiRequest:
    """Build a retry request with a corrective instruction appended to the
    user message (block 3). System blocks (1, 2) are unchanged so the
    prompt-version hash and the prefix cache still match; block 3 is
    per-call already, so adding corrective text there is on-protocol.

    Per S5.3 AC-4 / S5.4 AC-3: 'retry once with stricter system instruction'.
    """
    from prompts.loader import Block, Message

    correction = (
        "\n\n[SYSTEM RETRY]: your previous response could not be parsed as a "
        "valid OpusProposalReview. Error: "
        f"{prior_error}. Respond with valid JSON only — no markdown fences, "
        "no commentary outside the JSON object. The JSON must conform to the "
        "OpusProposalReview schema in your system prompt."
    )
    new_messages: list[Message] = []
    for msg in request.messages:
        new_content: list[Block] = []
        for block in msg.content:
            new_content.append(Block(text=block.text, cache_control=block.cache_control))
        # Append correction to the LAST block of the LAST user message.
        if msg is request.messages[-1] and new_content:
            last = new_content[-1]
            new_content[-1] = Block(
                text=last.text + correction,
                cache_control=last.cache_control,
            )
        new_messages.append(Message(role=msg.role, content=new_content))
    return ApiRequest(
        system=list(request.system),
        messages=new_messages,
        prompt_version=request.prompt_version,
        fragment_versions=list(request.fragment_versions),
    )


def _filing_context(filing: FilingRow, raw_text: str) -> str:
    """Filing context fed into Opus's block-3 (review payload).

    Evidence symmetry: the reviewer sees the SAME capped filing body the
    analyzer saw (`cap_filing_text` at `analyzer_max_input_chars`), not a
    2,000-char cover-page slice. For 8-Ks the substance usually lives in
    Exhibit 99.1, well past 2k — a reviewer fed only the cover page rejects
    sound proposals as "asserted, not corroborated" (NPK 2026-08-14, AGPU
    2026-08-17). The judge must see the analyst's case file.
    """
    return (
        f"# Source filing\n"
        f"accession_number: {filing.accession_number}\n"
        f"cik: {filing.cik}\n"
        f"form_type: {filing.form_type}\n"
        f"filed_at: {filing.filed_at.isoformat()}\n"
        f"item_codes: {filing.item_codes or '[]'}\n"
        f"---\n"
        f"{raw_text.rstrip()}\n"
    )


__all__ = [
    "OpusMalformed",
    "OpusModified",
    "OpusRatified",
    "OpusRejected",
    "OpusReviewResult",
    "OpusReviewer",
]
