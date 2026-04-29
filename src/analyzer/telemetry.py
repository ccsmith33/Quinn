"""Per-call telemetry value object — shared between S5.4 (Opus reviewer)
and S5.5 (proposal store).

The journal `llm_calls` table (NFR-8) is the source of truth; this dataclass
is the in-memory shape that producers (analyzer, reviewer) hand to the
proposal store so it can populate the same telemetry columns on
`proposals` / `proposal_reviews` rows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallTelemetry:
    """Token / latency / cost snapshot from one Anthropic API call.

    Field names match the columns on `llm_calls` / `proposals` /
    `proposal_reviews` (architecture §8.1).
    """

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    latency_ms: int
    cost_usd: float
