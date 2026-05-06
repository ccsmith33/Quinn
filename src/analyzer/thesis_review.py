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

@dataclass(frozen=True)
class ThesisHold:
    rationale: str


@dataclass(frozen=True)
class ThesisClose:
    rationale: str


@dataclass(frozen=True)
class ThesisAdjustStop:
    rationale: str
    new_stop_price: float


@dataclass(frozen=True)
class ThesisReviewMalformed:
    raw_response: str
    error: str


ThesisReviewResult = (
    ThesisHold | ThesisClose | ThesisAdjustStop | ThesisReviewMalformed
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
        purpose: Literal["analyze", "review", "replay", "audit"],
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
    ) -> None:
        self._client = client
        self._builder = prompt_builder
        self._opus_model_id = opus_model_id
        self._db_path = db_path
        self._max_output_tokens = max_output_tokens

    async def review(self, ctx: ThesisReviewContext) -> ThesisReviewResult:
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
        )
        # Build a unique decision_id per thesis-review schedule so llm_calls
        # rows stay distinct from proposal-review rows for the same proposal.
        decision_id = f"{_DECISION_ID_PREFIX}{ctx.schedule_id:08d}"
        raw = await self._client.call(
            request,
            model_id=self._opus_model_id,
            purpose="review",
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

    def _read_telemetry(self, decision_id: str) -> CallTelemetry:
        rows = get_llm_calls_by_decision_id(self._db_path, decision_id)
        review_rows = [r for r in rows if r.purpose == "review"]
        if not review_rows:
            raise RuntimeError(
                f"no llm_calls row for decision_id={decision_id!r} purpose=review"
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
    if decision not in ("hold", "close", "adjust_stop"):
        return ThesisReviewMalformed(
            raw_response=raw,
            error=f"decision must be hold|close|adjust_stop, got {decision!r}",
        )
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not (50 <= len(rationale) <= 4000):
        return ThesisReviewMalformed(
            raw_response=raw, error="rationale missing or length outside [50, 4000]"
        )
    if decision == "hold":
        return ThesisHold(rationale=rationale)
    if decision == "close":
        return ThesisClose(rationale=rationale)
    # adjust_stop
    mods = payload.get("modifications")
    if not isinstance(mods, dict):
        return ThesisReviewMalformed(
            raw_response=raw,
            error="adjust_stop requires modifications object",
        )
    new_stop = mods.get("new_stop_price")
    if not isinstance(new_stop, (int, float)) or new_stop <= 0:
        return ThesisReviewMalformed(
            raw_response=raw,
            error=f"new_stop_price must be positive number, got {new_stop!r}",
        )
    return ThesisAdjustStop(rationale=rationale, new_stop_price=float(new_stop))


def _decision_string(result: ThesisReviewResult) -> str:
    if isinstance(result, ThesisHold):
        return "hold"
    if isinstance(result, ThesisClose):
        return "close"
    if isinstance(result, ThesisAdjustStop):
        return "adjust_stop"
    return "malformed"


def _rationale(result: ThesisReviewResult) -> str:
    if isinstance(result, (ThesisHold, ThesisClose, ThesisAdjustStop)):
        return result.rationale
    return f"[malformed: {result.error}]"  # type: ignore[union-attr]


def _modifications_json(result: ThesisReviewResult) -> str | None:
    if isinstance(result, ThesisAdjustStop):
        return json.dumps({"new_stop_price": result.new_stop_price})
    return None


__all__ = [
    "ThesisAdjustStop",
    "ThesisClose",
    "ThesisHold",
    "ThesisReviewContext",
    "ThesisReviewMalformed",
    "ThesisReviewResult",
    "ThesisReviewer",
]
