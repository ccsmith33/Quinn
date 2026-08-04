"""S5.3 — Sonnet filing analyzer.

Architecture references: §2.3 (LLM Analyzer), §3.2 (TradeProposal),
§3.3 (NoTradeRecord), FR-15, FR-16, FR-18, FR-19, NFR-1, ADR-005.

Flow per AC-1..AC-7:
1. Build the request via `PromptBuilder.build_sonnet_filing_analysis`
2. Compute deterministic `decision_id` (AC-9)
3. Send via `AnthropicClient.call` with `purpose="analyze"`
4. Parse response (strip ```json fences) and validate against schemas
5. On schema/parse failure: retry once with stricter user-message instruction
   (mirrors S5.4's `_augment_request_for_retry` pattern; system blocks stay
   byte-stable so cache prefix and prompt-version hash are preserved)
6. Persist via `ProposalStore.store(...)` — including `AnalyzerMalformed`
7. For accepted `ProposalEmitted` with `conviction >= threshold`: route to
   Opus reviewer (FR-17 gate). Below threshold: do not (FR-18).

Carry-forward notes:
- D-2 (S5.1): the no-trade schema is now embedded in the prompt body, so
  `_PROMPT_DEFS["sonnet_filing_analysis_v1"]` includes
  `("schema", "no_trade_record")`. Editing that schema rolls the
  prompt-version hash, preserving ADR-005's "modify any included artifact
  rolls the version" invariant.
- C-1 (S5.2): `max_tokens` is NOT plumbed here; `AnthropicClient` still
  hardcodes 16000. S5.6 wiring owns the fix.
- A-1 (S5.4): the conviction-threshold gate that decides "Opus or not"
  lives HERE, in the analyzer (per AC-7 and FR-18). The wrapper above
  this — S5.6 — owns no per-call gate; it just supplies the threshold
  via constructor injection. S5.4's `OpusReviewer` does not re-check the
  threshold (it trusts the caller). This separation is by design: a
  defensive re-check in S5.4 would force callers to re-justify the gate
  decision, defeating composability for replay/audit code paths that
  legitimately re-call Opus on lower-conviction proposals.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Literal, Protocol

from app.memory_context import MemoryContextAssembler, MemoryQuery
from journal.models import FilingRow, ProposalRow
from journal.repo import get_llm_calls_by_decision_id, get_proposal_by_id
from observability.log_port import get_logger
from prompts.loader import AnalyzerContext, ApiRequest

from .parse import ResponseParseError, extract_json_object
from .results import AnalyzerMalformed, AnalyzerResult, NoTrade, ProposalEmitted
from .telemetry import CallTelemetry

log = get_logger(__name__)

_DECISION_ID_HEX_LEN = 32


# ---------------------------------------------------------------------------
# Ports (boundaries to S5.1 / S5.2 / S5.4 / S5.5)
# ---------------------------------------------------------------------------

class _PromptBuilderPort(Protocol):
    def build_sonnet_filing_analysis(
        self,
        filing: FilingRow,
        raw_text: str,
        ctx: AnalyzerContext,
        *,
        memory_block: str | None = None,
    ) -> ApiRequest: ...

    def prompt_version(self, name: str) -> str: ...


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


class _ProposalStorePort(Protocol):
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
    ) -> int: ...


class _OpusReviewerPort(Protocol):
    async def review(
        self, proposal: ProposalRow, source_filing: FilingRow, raw_text: str
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class SonnetAnalyzer:
    """Primary LLM-analysis component (FR-15, FR-16).

    Single-call path:
      build → call → parse → validate → store → (gate→opus)

    All retries on transient transport errors are owned by `AnthropicClient`
    (per S5.2 AC-3). This module owns retry-once-on-malformed only.
    """

    def __init__(
        self,
        *,
        client: _AnthropicClientPort,
        store: _ProposalStorePort,
        prompt_builder: _PromptBuilderPort,
        opus_reviewer: _OpusReviewerPort,
        sonnet_model_id: str,
        opus_review_conviction_threshold: int,
        db_path: str,
        max_output_tokens: int | None = None,
        ks5_max_concurrent: int | None = None,
        open_positions_counter: Callable[[], int] | None = None,
        memory_assembler: MemoryContextAssembler | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._builder = prompt_builder
        self._opus = opus_reviewer
        self._sonnet_model_id = sonnet_model_id
        self._threshold = opus_review_conviction_threshold
        self._db_path = db_path
        # S5.6 carry-fwd S5.2 C-1: per-call cost ceiling (D-052). When
        # None (legacy test path), the SDK kwargs default applies.
        self._max_output_tokens = max_output_tokens
        # Feature B: skip Opus when KS5 is at capacity. None disables the
        # gate (legacy test path); production composition wires the value
        # from `config.execution.ks5_max_concurrent`. The counter must
        # share a source-of-truth with `SizingEngine.size()` (broker
        # truth) — otherwise Feature B can refuse to spend on a trade
        # that would have actually fit. Composition wires this to
        # `lambda: len(broker.get_positions())`.
        self._ks5_max_concurrent = ks5_max_concurrent
        self._open_positions_counter = open_positions_counter
        # LLM memory layer. None (default, and whenever the memory master
        # gate is off) → no MemoryQuery is built and the request is
        # byte-identical to the pre-memory analyzer.
        self._memory_assembler = memory_assembler

    async def analyze(
        self,
        filing: FilingRow,
        raw_text: str,
        ctx: AnalyzerContext,
    ) -> AnalyzerResult:
        memory_block = self._assemble_memory(filing)
        request = self._builder.build_sonnet_filing_analysis(
            filing, raw_text, ctx, memory_block=memory_block
        )
        prompt_version = request.prompt_version
        decision_id = compute_decision_id(
            filing_id=filing.id,
            model_id=self._sonnet_model_id,
            prompt_version=prompt_version,
        )

        raw = await self._client.call(
            request,
            model_id=self._sonnet_model_id,
            purpose="analyze",
            decision_id=decision_id,
            max_tokens=self._max_output_tokens,
        )
        result = _parse_and_validate(raw)
        if isinstance(result, AnalyzerMalformed):
            log.warning(
                "analyzer.malformed",
                extra={
                    "event": "analyzer.malformed",
                    "decision_id": decision_id,
                    "error": result.error,
                    "attempt": 1,
                },
            )
            retry_request = augment_request_for_retry(request, prior_error=result.error)
            raw = await self._client.call(
                retry_request,
                model_id=self._sonnet_model_id,
                purpose="analyze",
                decision_id=decision_id,
                max_tokens=self._max_output_tokens,
            )
            result = _parse_and_validate(raw)
            if isinstance(result, AnalyzerMalformed):
                log.error(
                    "analyzer.malformed",
                    extra={
                        "event": "analyzer.malformed",
                        "decision_id": decision_id,
                        "error": result.error,
                        "attempt": 2,
                    },
                )

        telemetry = self._read_telemetry(decision_id)
        proposal_id = self._store.store(
            result,
            filing=filing,
            model_id=self._sonnet_model_id,
            prompt_version=prompt_version,
            decision_id=decision_id,
            raw_response=raw,
            telemetry=telemetry,
        )

        # AC-7: conviction-gated routing to Opus reviewer (FR-17 / FR-18).
        if isinstance(result, ProposalEmitted):
            conviction = int(result.payload.get("conviction", 0))
            if conviction >= self._threshold:
                # Feature B: capacity gate. If KS5 is at cap, skip Opus —
                # the proposal can't be actioned and would only be rejected
                # by the sizer anyway. Absence of a `proposal_reviews` row
                # is the "pending_capacity" signal Feature C keys on.
                #
                # Source-of-truth: the count MUST come from the same place
                # the sizer's KS-5 gate consults (broker truth, see
                # `SizingEngine.size()`). If we read journal-truth here
                # the two gates can disagree and Feature B will refuse to
                # spend on a trade that would actually have fit. The
                # counter is callable so the call is lazy — invoked only
                # when conviction has already cleared the threshold.
                #
                # Fallback: if no counter is wired (legacy test path),
                # fall back to the AnalyzerContext's journal-derived
                # count. Production composition always wires the broker-
                # backed counter.
                if self._ks5_max_concurrent is not None:
                    if self._open_positions_counter is not None:
                        try:
                            current_open = int(self._open_positions_counter())
                        except Exception as e:  # noqa: BLE001 — broker outage
                            # If we can't ask the broker (transient outage),
                            # fall back to the journal-derived count to
                            # stay consistent with the rest of the pipeline.
                            log.warning(
                                "agent.opus_capacity_counter_failed",
                                extra={
                                    "event": "agent.opus_capacity_counter_failed",
                                    "error": str(e),
                                },
                            )
                            current_open = ctx.open_positions_count
                    else:
                        current_open = ctx.open_positions_count
                    if current_open >= self._ks5_max_concurrent:
                        log.info(
                            "agent.opus_skipped_at_capacity",
                            extra={
                                "event": "agent.opus_skipped_at_capacity",
                                "filing_id": filing.id,
                                "proposal_id": proposal_id,
                                "conviction": conviction,
                                "current_open_positions": current_open,
                                "ks5_max": self._ks5_max_concurrent,
                            },
                        )
                        return result

                proposal_row = get_proposal_by_id(self._db_path, proposal_id)
                if proposal_row is None:
                    raise RuntimeError(
                        f"proposal {proposal_id} disappeared between "
                        "store and read"
                    )
                await self._opus.review(proposal_row, filing, raw_text)

        return result

    def _assemble_memory(self, filing: FilingRow) -> str | None:
        """Build the memory block for this filing analysis, or None when the
        memory layer is disabled / contributes nothing. Provider failures
        are absorbed inside the assembler, so this never raises into the
        analysis path."""
        if self._memory_assembler is None:
            return None
        return self._memory_assembler.assemble(
            MemoryQuery(
                symbol=filing.issuer_ticker,
                purpose="analyze",
                execution_id=None,
                conviction=None,
            )
        )

    def _read_telemetry(self, decision_id: str) -> CallTelemetry:
        rows = get_llm_calls_by_decision_id(self._db_path, decision_id)
        analyze_rows = [r for r in rows if r.purpose == "analyze"]
        if not analyze_rows:
            raise RuntimeError(
                f"no llm_calls row for decision_id={decision_id!r}, "
                "purpose='analyze' — AnthropicClient should have written one"
            )
        last = analyze_rows[-1]
        return CallTelemetry(
            input_tokens=last.input_tokens,
            output_tokens=last.output_tokens,
            cache_read_tokens=last.cache_read_tokens,
            cache_creation_tokens=last.cache_creation_tokens,
            latency_ms=last.latency_ms,
            cost_usd=last.cost_usd,
        )


# ---------------------------------------------------------------------------
# Pure helpers (testable in isolation)
# ---------------------------------------------------------------------------

def compute_decision_id(*, filing_id: int | None, model_id: str, prompt_version: str) -> str:
    """AC-9: deterministic decision_id from (filing.id, model_id, prompt_version)."""
    if filing_id is None:
        raise ValueError("filing.id is required to compute decision_id")
    src = f"{filing_id}|{model_id}|{prompt_version}"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:_DECISION_ID_HEX_LEN]


def _parse_and_validate(raw: str) -> AnalyzerResult:
    """Parse `raw` and discriminate TradeProposal vs NoTradeRecord vs malformed.

    Per AC-4 dev note: discriminate by presence of `direction`/`symbol`
    (proposal) vs. `decision == "no_trade"` (no-trade). On either schema
    failure, return `AnalyzerMalformed` so the caller can retry once.
    """
    try:
        payload = extract_json_object(raw)
    except ResponseParseError as e:
        return AnalyzerMalformed(raw_response=raw, error=str(e))

    is_no_trade = payload.get("decision") == "no_trade"
    is_proposal = "symbol" in payload and "direction" in payload

    # Lazy import — keeps this module cheap to import.
    from proposal.schemas import (
        ProposalSchemaError,
        validate_no_trade_record,
        validate_trade_proposal,
    )

    if is_no_trade:
        try:
            validate_no_trade_record(payload)
        except ProposalSchemaError as e:
            return AnalyzerMalformed(raw_response=raw, error=f"no_trade_record: {e}")
        return NoTrade(payload=payload, raw_response=raw)
    if is_proposal:
        try:
            validate_trade_proposal(payload)
        except ProposalSchemaError as e:
            return AnalyzerMalformed(raw_response=raw, error=f"trade_proposal: {e}")
        return ProposalEmitted(payload=payload, raw_response=raw)
    return AnalyzerMalformed(
        raw_response=raw,
        error=(
            "discrimination failed: missing both `decision: \"no_trade\"` and "
            "`symbol`/`direction` keys"
        ),
    )


def augment_request_for_retry(request: ApiRequest, *, prior_error: str) -> ApiRequest:
    """Build a retry request with a corrective instruction appended to the
    last block of the last user message.

    System blocks (1, 2) are NOT modified — this preserves byte-equality of
    the cacheable prefix (NFR-12) and keeps the prompt-version hash stable.
    Block 3 is per-call already, so adding corrective text there is on-protocol.

    Mirrors S5.4's `_augment_request_for_retry` for parity (per S5.3 AC-4
    and S5.4 review T-1 carry-forward).
    """
    from prompts.loader import Block, Message

    correction = (
        "\n\n[SYSTEM RETRY]: your previous response was not valid JSON or did "
        "not match the schema. Error: "
        f"{prior_error}. Respond with valid JSON only — no markdown fences, "
        "no commentary outside the JSON object. Emit exactly one JSON object: "
        "either a TradeProposal or a NoTradeRecord conforming to the schemas "
        "in your system prompt."
    )
    new_messages: list[Message] = []
    for msg in request.messages:
        new_content: list[Block] = [
            Block(text=b.text, cache_control=b.cache_control) for b in msg.content
        ]
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


__all__ = [
    "SonnetAnalyzer",
    "augment_request_for_retry",
    "compute_decision_id",
]
