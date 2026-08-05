"""Per-model Anthropic pricing table — used by `AnthropicClient` to compute
`cost_usd` from token counts (S5.2 AC-2).

Values match Anthropic's public pricing as of 2026-08 (claude-api skill
cached snapshot; see `shared/models.md`). Prices are in USD per token
(divided from the public per-1M-token rates).

REFRESH ANNUALLY. Anthropic does not version model IDs by pricing change,
so a price update on an existing model ID will only land here if a human
edits this table. The reviewer-e5 backstop should diff this file against
Anthropic's public pricing page during S5.2 review.

Cache economics (per `shared/prompt-caching.md`):
- cache reads cost ~0.1× base input price
- cache writes (5-minute TTL — what Quinn uses per ADR-005) cost ~1.25×
  base input price.
"""

from __future__ import annotations

from dataclasses import dataclass

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPricing:
    """Per-token rates for one model.

    All values are USD/token. `cache_read_per_token` and
    `cache_write_per_token` are derived from `input_per_token` per the cache
    economics in `shared/prompt-caching.md`.
    """

    input_per_token: float
    output_per_token: float
    cache_read_per_token: float
    cache_write_per_token: float


# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens → divided to per-token below)
# ---------------------------------------------------------------------------

# Per Anthropic public pricing (cached 2026-08):
#   Opus 5 / 4.8 / 4.7 / 4.6: $5 input / $25 output per 1M
#   Sonnet 4.6:               $3 input / $15 output per 1M
#   Haiku 4.5:                $1 input / $5  output per 1M
_PUBLIC_RATES_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Dated Haiku 4.5 alias — same model, snapshot id used by config.
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


def _build_pricing_table() -> dict[str, ModelPricing]:
    table: dict[str, ModelPricing] = {}
    for model_id, (input_rate, output_rate) in _PUBLIC_RATES_PER_MILLION.items():
        input_per_token = input_rate / _PER_MILLION
        output_per_token = output_rate / _PER_MILLION
        table[model_id] = ModelPricing(
            input_per_token=input_per_token,
            output_per_token=output_per_token,
            cache_read_per_token=input_per_token * 0.1,
            cache_write_per_token=input_per_token * 1.25,
        )
    return table


PRICING: dict[str, ModelPricing] = _build_pricing_table()


class UnknownModel(Exception):
    """Raised when `compute_cost` is called with a model id not in the
    pricing table. Operationally this means the table needs an update."""


def compute_cost(
    model_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
) -> float:
    """Compute USD cost for one Anthropic API call.

    Per architecture §10.1: cost lands in `llm_calls.cost_usd` and rolls up
    into the daily report. Uncached input tokens are billed at the base rate;
    cache reads at ~0.1× and cache writes at ~1.25× per `shared/prompt-caching.md`.
    """
    if model_id not in PRICING:
        raise UnknownModel(f"no pricing entry for model_id: {model_id!r}")
    p = PRICING[model_id]
    return (
        input_tokens * p.input_per_token
        + output_tokens * p.output_per_token
        + cache_read_tokens * p.cache_read_per_token
        + cache_creation_tokens * p.cache_write_per_token
    )
