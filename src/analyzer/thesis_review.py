"""Feature A — Opus thesis-review on open positions.

When a position's declared time-horizon elapses without stop or
take-profit firing, run an Opus review of the original thesis given the
current price + filings since entry. Opus emits `hold | close |
adjust_stop`. The reconciler-driven coordinator applies the decision.

This module owns the LLM call + parse + persist seam. The action layer
(submit close order, adjust stop, reschedule) is in
`src/app/thesis_coordinator.py` so the LLM-bound logic stays unit-
testable without the broker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.memory_context import MemoryContextAssembler, MemoryQuery
from journal.models import ProposalRow, ThesisReviewRow
from journal.repo import get_llm_calls_by_decision_id, insert_thesis_review
from observability.log_port import get_logger
from prompts.loader import ApiRequest

from .telemetry import CallTelemetry

log = get_logger(__name__)


_DECISION_ID_PREFIX = "thesis-"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

# Expendability (chopping block, daily-sweep reviews only): every decision
# type carries an OPTIONAL 1..5 sacrifice-priority score plus a one-line
# reason. Absent (None) on non-sweep reviews and on any response that
# omits or malforms the field — the decision itself is never lost to a
# missing/bad expendability. 1 = last to sacrifice, 5 = first.


@dataclass(frozen=True)
class ThesisHold:
    rationale: str
    expendability: int | None = None
    expendability_reason: str | None = None


@dataclass(frozen=True)
class ThesisClose:
    rationale: str
    expendability: int | None = None
    expendability_reason: str | None = None


@dataclass(frozen=True)
class ThesisAdjustStop:
    rationale: str
    new_stop_price: float
    expendability: int | None = None
    expendability_reason: str | None = None


@dataclass(frozen=True)
class ThesisAdjustTakeProfit:
    """D-079 §3.4 — raise-only TP actuator (let-winners-run). Lowering a
    TP is expressible as `close`; a lowerable TP would reintroduce the
    tighten-only pressure S-4 diagnosed. Raise-only is enforced at apply
    time by the coordinator (new_tp must clear the current quote)."""

    rationale: str
    new_tp_price: float
    expendability: int | None = None
    expendability_reason: str | None = None


@dataclass(frozen=True)
class ThesisReviewMalformed:
    raw_response: str
    error: str


ThesisReviewResult = (
    ThesisHold
    | ThesisClose
    | ThesisAdjustStop
    | ThesisAdjustTakeProfit
    | ThesisReviewMalformed
)


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------

class _AnthropicClientPort(Protocol):
    async def call(
        self,
        request: ApiRequest,
        *,
        model_id: str,
        purpose: Literal["analyze", "review", "thesis_review", "replay", "audit"],
        decision_id: str,
        max_tokens: int | None = None,
    ) -> str: ...


class _PromptBuilderPort(Protocol):
    def build_opus_thesis_review(
        self,
        *,
        proposal: ProposalRow,
        execution_id: int,
        days_held: int,
        current_price: float,
        pct_change_since_entry: float,
        realized_fill_price: float | None,
        realized_dollar_size: float | None,
        time_horizon_days: int,
        stop_loss_price: float,
        take_profit_price: float | None,
        filings_since_entry_summary: str,
        position_zone: int | None = None,
        trail_armed: bool | None = None,
        hwm_gain_pct: float | None = None,
        trigger_reason: str | None = None,
        trigger_detail: str | None = None,
        capacity_pressure_block: str | None = None,
        memory_block: str | None = None,
    ) -> ApiRequest: ...


# ---------------------------------------------------------------------------
# Reviewer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThesisReviewContext:
    """All inputs needed to drive one thesis-review LLM call."""

    proposal: ProposalRow
    execution_id: int
    schedule_id: int
    days_held: int
    current_price: float
    pct_change_since_entry: float
    realized_fill_price: float | None
    realized_dollar_size: float | None
    time_horizon_days: int
    stop_loss_price: float
    take_profit_price: float | None
    filings_since_entry_summary: str
    # Zone-aware review context (event-reviews doctrine). Populated by
    # the coordinator for every review — calendar AND event — so the
    # prompt's jurisdiction-zones section has the state it governs by.
    # Defaults keep legacy construction sites valid.
    position_zone: int | None = None
    trail_armed: bool | None = None
    hwm_gain_pct: float | None = None
    # The schedule row's scheduled_reason verbatim ('entry', 'hold',
    # 'event:filing:...', ...) plus a human-readable detail line for
    # event triggers (filing form/accession, headline text).
    trigger_reason: str | None = None
    trigger_detail: str | None = None
    # Capacity-pressure sweep context (feature off / not a daily sweep /
    # not under pressure → None). Set by the coordinator only on
    # 'event:daily_sweep' reviews; appended to the per-call block 3 by the
    # prompt builder so it never perturbs the cached system blocks. See
    # EventReviewsConfig.capacity_target_slots.
    capacity_pressure_block: str | None = None


class ThesisReviewer:
    """Calls Opus with the thesis-review prompt and persists the outcome."""

    def __init__(
        self,
        *,
        client: _AnthropicClientPort,
        prompt_builder: _PromptBuilderPort,
        opus_model_id: str,
        db_path: str,
        max_output_tokens: int | None = None,
        memory_assembler: MemoryContextAssembler | None = None,
    ) -> None:
        self._client = client
        self._builder = prompt_builder
        self._opus_model_id = opus_model_id
        self._db_path = db_path
        self._max_output_tokens = max_output_tokens
        # LLM memory layer. None (default, and whenever the memory master
        # gate is off) → no MemoryQuery is built and the request is
        # byte-identical to the pre-memory reviewer.
        self._memory_assembler = memory_assembler

    async def review(self, ctx: ThesisReviewContext) -> ThesisReviewResult:
        memory_block = self._assemble_memory(ctx)
        request = self._builder.build_opus_thesis_review(
            proposal=ctx.proposal,
            execution_id=ctx.execution_id,
            days_held=ctx.days_held,
            current_price=ctx.current_price,
            pct_change_since_entry=ctx.pct_change_since_entry,
            realized_fill_price=ctx.realized_fill_price,
            realized_dollar_size=ctx.realized_dollar_size,
            time_horizon_days=ctx.time_horizon_days,
            stop_loss_price=ctx.stop_loss_price,
            take_profit_price=ctx.take_profit_price,
            filings_since_entry_summary=ctx.filings_since_entry_summary,
            position_zone=ctx.position_zone,
            trail_armed=ctx.trail_armed,
            hwm_gain_pct=ctx.hwm_gain_pct,
            trigger_reason=ctx.trigger_reason,
            trigger_detail=ctx.trigger_detail,
            capacity_pressure_block=ctx.capacity_pressure_block,
            memory_block=memory_block,
        )
        # Build a unique decision_id per thesis-review schedule so llm_calls
        # rows stay distinct from proposal-review rows for the same proposal.
        # purpose="thesis_review" (2026-08-04 cost-ledger fix): the event-
        # review layer journals under its own tag — riding "review" made it
        # indistinguishable from Opus proposal reviews, so every purpose-
        # keyed spend rollup undercounted the layer to zero.
        decision_id = f"{_DECISION_ID_PREFIX}{ctx.schedule_id:08d}"
        raw = await self._client.call(
            request,
            model_id=self._opus_model_id,
            purpose="thesis_review",
            decision_id=decision_id,
            max_tokens=self._max_output_tokens,
        )
        result = _parse(raw)
        if isinstance(result, ThesisReviewMalformed):
            log.warning(
                "thesis_review.malformed",
                extra={
                    "event": "thesis_review.malformed",
                    "schedule_id": ctx.schedule_id,
                    "execution_id": ctx.execution_id,
                    "error": result.error,
                },
            )

        telemetry = self._read_telemetry(decision_id)
        # Persist the outcome row. UNIQUE(schedule_id) ensures we never
        # double-fire the same schedule.
        insert_thesis_review(
            self._db_path,
            ThesisReviewRow(
                execution_id=ctx.execution_id,
                schedule_id=ctx.schedule_id,
                model_id=self._opus_model_id,
                prompt_version=request.prompt_version,
                decision=_decision_string(result),
                raw_response=raw,
                rationale=_rationale(result),
                modifications_json=_modifications_json(result),
                input_tokens=telemetry.input_tokens,
                output_tokens=telemetry.output_tokens,
                cache_read_tokens=telemetry.cache_read_tokens,
                cache_creation_tokens=telemetry.cache_creation_tokens,
                latency_ms=telemetry.latency_ms,
                cost_usd=telemetry.cost_usd,
            ),
        )
        return result

    def _assemble_memory(self, ctx: ThesisReviewContext) -> str | None:
        """Build the memory block for this thesis review, or None when the
        memory layer is disabled / contributes nothing. Provider failures
        are absorbed inside the assembler, so this never raises into the
        review path."""
        if self._memory_assembler is None:
            return None
        return self._memory_assembler.assemble(
            MemoryQuery(
                symbol=ctx.proposal.symbol,
                purpose="thesis_review",
                execution_id=ctx.execution_id,
                conviction=ctx.proposal.conviction,
            )
        )

    def _read_telemetry(self, decision_id: str) -> CallTelemetry:
        rows = get_llm_calls_by_decision_id(self._db_path, decision_id)
        review_rows = [r for r in rows if r.purpose == "thesis_review"]
        if not review_rows:
            raise RuntimeError(
                f"no llm_calls row for decision_id={decision_id!r} "
                "purpose=thesis_review"
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


def _parse(raw: str) -> ThesisReviewResult:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        return ThesisReviewMalformed(raw_response=raw, error=f"json: {e}")
    if not isinstance(payload, dict):
        return ThesisReviewMalformed(raw_response=raw, error="top-level must be object")
    decision = payload.get("decision")
    if decision not in ("hold", "close", "adjust_stop", "adjust_take_profit"):
        return ThesisReviewMalformed(
            raw_response=raw,
            error=(
                "decision must be hold|close|adjust_stop|adjust_take_profit, "
                f"got {decision!r}"
            ),
        )
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not (50 <= len(rationale) <= 4000):
        return ThesisReviewMalformed(
            raw_response=raw, error="rationale missing or length outside [50, 4000]"
        )
    exp, exp_reason = _parse_expendability(payload)
    if decision == "hold":
        return ThesisHold(
            rationale=rationale, expendability=exp, expendability_reason=exp_reason
        )
    if decision == "close":
        return ThesisClose(
            rationale=rationale, expendability=exp, expendability_reason=exp_reason
        )
    # adjust_stop / adjust_take_profit both require a modifications object
    # carrying their own price key — the actions must not cross wires.
    mods = payload.get("modifications")
    if not isinstance(mods, dict):
        return ThesisReviewMalformed(
            raw_response=raw,
            error=f"{decision} requires modifications object",
        )
    if decision == "adjust_stop":
        new_stop = mods.get("new_stop_price")
        if not isinstance(new_stop, (int, float)) or new_stop <= 0:
            return ThesisReviewMalformed(
                raw_response=raw,
                error=f"new_stop_price must be positive number, got {new_stop!r}",
            )
        return ThesisAdjustStop(
            rationale=rationale,
            new_stop_price=float(new_stop),
            expendability=exp,
            expendability_reason=exp_reason,
        )
    # adjust_take_profit (D-079 §3.4)
    new_tp = mods.get("new_tp_price")
    if not isinstance(new_tp, (int, float)) or new_tp <= 0:
        return ThesisReviewMalformed(
            raw_response=raw,
            error=f"new_tp_price must be positive number, got {new_tp!r}",
        )
    return ThesisAdjustTakeProfit(
        rationale=rationale,
        new_tp_price=float(new_tp),
        expendability=exp,
        expendability_reason=exp_reason,
    )


def _parse_expendability(payload: dict[str, Any]) -> tuple[int | None, str | None]:
    """Extract the optional chopping-block expendability (1..5) + reason.

    Tolerant by design: the field is present only on daily-sweep reviews
    and absent on everything else and on older responses. A missing,
    non-integer, out-of-range, or otherwise malformed value yields
    `(None, None)` — the decision must never be lost because expendability
    was bad. `bool` is rejected explicitly (a stray `true` is not a 1)."""
    raw = payload.get("expendability")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None, None
    if not (1 <= raw <= 5):
        return None, None
    reason = payload.get("expendability_reason")
    reason = reason if isinstance(reason, str) and reason.strip() else None
    return raw, reason


def _decision_string(result: ThesisReviewResult) -> str:
    if isinstance(result, ThesisHold):
        return "hold"
    if isinstance(result, ThesisClose):
        return "close"
    if isinstance(result, ThesisAdjustStop):
        return "adjust_stop"
    if isinstance(result, ThesisAdjustTakeProfit):
        return "adjust_take_profit"
    return "malformed"


def _rationale(result: ThesisReviewResult) -> str:
    if isinstance(
        result,
        (ThesisHold, ThesisClose, ThesisAdjustStop, ThesisAdjustTakeProfit),
    ):
        return result.rationale
    return f"[malformed: {result.error}]"  # type: ignore[union-attr]


def _modifications_json(result: ThesisReviewResult) -> str | None:
    if isinstance(result, ThesisAdjustStop):
        return json.dumps({"new_stop_price": result.new_stop_price})
    if isinstance(result, ThesisAdjustTakeProfit):
        return json.dumps({"new_tp_price": result.new_tp_price})
    return None


__all__ = [
    "ThesisAdjustStop",
    "ThesisAdjustTakeProfit",
    "ThesisClose",
    "ThesisHold",
    "ThesisReviewContext",
    "ThesisReviewMalformed",
    "ThesisReviewResult",
    "ThesisReviewer",
]
