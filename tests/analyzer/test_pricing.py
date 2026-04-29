"""S5.2 — Pricing table tests (AC-2 cost computation)."""

from __future__ import annotations

import pytest

from analyzer.pricing import PRICING, UnknownModel, compute_cost


def test_known_models_present() -> None:
    """Sanity-check that all model ids the rest of v1 may use are priced.

    If a new model id is added to `config.analyzer` (sonnet_model_id /
    opus_model_id) and missing here, `compute_cost` will raise.
    """
    for model_id in (
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ):
        assert model_id in PRICING


def test_compute_cost_basic_math() -> None:
    """Sonnet 4.6 = $3 in / $15 out per 1M; cache_read = 0.1×, cache_write = 1.25×."""
    cost = compute_cost(
        "claude-sonnet-4-6",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
    )
    expected = 3.00 + 15.00 + (3.00 * 0.1) + (3.00 * 1.25)
    assert cost == pytest.approx(expected, rel=1e-9)


def test_compute_cost_zero_tokens() -> None:
    assert compute_cost(
        "claude-opus-4-7",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    ) == 0.0


def test_compute_cost_unknown_model_raises() -> None:
    with pytest.raises(UnknownModel):
        compute_cost(
            "claude-fictional-7-0",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )


def test_opus_more_expensive_than_haiku() -> None:
    """Sanity check: Opus 4.7 input rate is strictly higher than Haiku 4.5."""
    opus = PRICING["claude-opus-4-7"]
    haiku = PRICING["claude-haiku-4-5"]
    assert opus.input_per_token > haiku.input_per_token
    assert opus.output_per_token > haiku.output_per_token


def test_cache_economics_proportional_to_input() -> None:
    """Per shared/prompt-caching.md: cache_read = 0.1× input; cache_write = 1.25× input."""
    for p in PRICING.values():
        assert p.cache_read_per_token == pytest.approx(p.input_per_token * 0.1)
        assert p.cache_write_per_token == pytest.approx(p.input_per_token * 1.25)
