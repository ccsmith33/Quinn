"""Shared analyzer result types — discriminated union consumed by S5.5
(`ProposalStore.store`) and produced by S5.3 (`SonnetAnalyzer.analyze`).

Lives in `src/analyzer/` because S5.3 owns the producer surface; S5.5
(persistence) imports it. Keeping it here avoids a circular import: S5.5
already imports `CallTelemetry` from `analyzer.telemetry`, so the proposal
store stays a downstream consumer of analyzer types, not a peer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProposalEmitted:
    """The Sonnet analyzer parsed the LLM response into a valid
    `TradeProposal` (architecture §3.2)."""

    payload: dict[str, Any]
    raw_response: str


@dataclass(frozen=True)
class NoTrade:
    """The Sonnet analyzer parsed the LLM response into a valid
    `NoTradeRecord` (architecture §3.3) — sub-threshold conviction or
    explicit no-trade signal."""

    payload: dict[str, Any]
    raw_response: str


@dataclass(frozen=True)
class AnalyzerMalformed:
    """The Sonnet analyzer's retry-once-on-malformed exhausted; the LLM
    response could not be parsed as either a `TradeProposal` or a
    `NoTradeRecord`. Per architecture §2.3 failure modes, the journal
    still records the failure (kind=`no_trade`, reasoning_notes captures
    parse error)."""

    raw_response: str
    error: str


AnalyzerResult = ProposalEmitted | NoTrade | AnalyzerMalformed


__all__ = [
    "AnalyzerMalformed",
    "AnalyzerResult",
    "NoTrade",
    "ProposalEmitted",
]
