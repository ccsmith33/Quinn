"""Pydantic models mirroring `src/prompts/schemas/{trade_proposal,no_trade_record}.json`
for defense-in-depth validation at the persistence boundary (S5.5 AC-2).

S5.3 already validates LLM output against these schemas before handing the
result to S5.5; this module is the second-line check the proposal store
runs before any row is written. A `ProposalSchemaError` here is operational
evidence that S5.3 let an invalid payload through — it should not happen
in production.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class ProposalSchemaError(Exception):
    """Raised when a payload fails schema validation at the persistence
    boundary. Wraps pydantic's `ValidationError` with a stable surface."""

    def __init__(self, kind: str, errors: Any) -> None:
        super().__init__(f"{kind} schema validation failed: {errors}")
        self.kind = kind
        self.errors = errors


# ---------------------------------------------------------------------------
# TradeProposal (architecture §3.2)
# ---------------------------------------------------------------------------

class _SourceFiling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accession_number: str
    filing_type: str
    item_codes: list[str] | None = None


class TradeProposal(BaseModel):
    """Pydantic mirror of `src/prompts/schemas/trade_proposal.json`."""

    model_config = ConfigDict(extra="allow")  # tolerate forward-compatible LLM extensions

    symbol: str
    direction: Literal["long"]
    size_pct_of_capital: Annotated[float, Field(gt=0.0, le=0.20)]
    entry_style: Literal["market_open", "limit"]
    entry_limit_price: Annotated[float, Field(gt=0.0)] | None = None
    stop_loss_price: Annotated[float, Field(gt=0.0)]
    take_profit_price: Annotated[float, Field(gt=0.0)] | None = None
    # D-079 §3.5 / ADR-011 — analyzer-proposed trailing-stop distance as a
    # percent of price. Optional: when absent, the trailing ratchet defaults
    # to the proposal's own initial risk distance clamped to [1, 15].
    trail_distance_pct: Annotated[float, Field(ge=0.5, le=20.0)] | None = None
    time_horizon_days: Annotated[int, Field(ge=1, le=60)]
    conviction: Annotated[int, Field(ge=1, le=10)]
    thesis: Annotated[str, Field(min_length=50, max_length=4000)]
    signals: Annotated[list[str], Field(min_length=1)]
    exit_conditions: Annotated[list[str], Field(min_length=1)]
    risk_factors: Annotated[list[str], Field(min_length=1)]

    # System-injected fields (optional at validation time — populated by
    # the proposal store before the row is written).
    prompt_version: str | None = None
    source_filings: list[_SourceFiling] | None = None

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, v: str) -> str:
        if not _SYMBOL_RE.match(v):
            raise ValueError(f"symbol {v!r} does not match {_SYMBOL_RE.pattern}")
        return v

    @field_validator("signals", "exit_conditions", "risk_factors")
    @classmethod
    def _check_string_list_lengths(cls, v: list[str]) -> list[str]:
        for s in v:
            if not (10 <= len(s) <= 800):
                raise ValueError(f"list item length {len(s)} outside [10, 800]")
        return v

    @model_validator(mode="after")
    def _check_limit_entry_requires_price(self) -> TradeProposal:
        """Mirror the JSON Schema's `allOf` conditional: when
        `entry_style="limit"`, `entry_limit_price` is required.

        Closes S5.5 review D-2 (parity defect): the JSON Schema enforces
        this via `if/then`; pydantic doesn't read JSON-Schema conditionals,
        so the rule is hand-coded here. Without this, the proposal store
        would persist a limit-entry proposal with no price, and the
        execution layer (S6.x) would have to either re-validate or fail
        at order construction time.
        """
        if self.entry_style == "limit" and self.entry_limit_price is None:
            raise ValueError(
                "entry_limit_price is required when entry_style='limit' "
                "(architecture §3.2 allOf clause)"
            )
        return self


# ---------------------------------------------------------------------------
# NoTradeRecord (architecture §3.3)
# ---------------------------------------------------------------------------

class NoTradeRecord(BaseModel):
    """Pydantic mirror of `src/prompts/schemas/no_trade_record.json`."""

    model_config = ConfigDict(extra="allow")

    decision: Literal["no_trade"]
    thesis_or_reason: Annotated[str, Field(min_length=50, max_length=2000)]
    signals_considered: list[str]
    prompt_version: str | None = None
    source_filings: list[_SourceFiling] | None = None


# ---------------------------------------------------------------------------
# Public validators (raise ProposalSchemaError on failure)
# ---------------------------------------------------------------------------

def validate_trade_proposal(payload: dict[str, Any]) -> TradeProposal:
    """Validate `payload` against the TradeProposal schema. Defense-in-depth
    at the persistence boundary; `S5.3` is the primary validator."""
    try:
        return TradeProposal.model_validate(payload)
    except ValidationError as e:
        raise ProposalSchemaError("trade_proposal", e.errors()) from e


def validate_no_trade_record(payload: dict[str, Any]) -> NoTradeRecord:
    try:
        return NoTradeRecord.model_validate(payload)
    except ValidationError as e:
        raise ProposalSchemaError("no_trade_record", e.errors()) from e


__all__ = [
    "NoTradeRecord",
    "ProposalSchemaError",
    "TradeProposal",
    "validate_no_trade_record",
    "validate_trade_proposal",
]
