"""S5.2 — Pricing table tests (AC-2 cost computation)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from analyzer.pricing import PRICING, UnknownModel, compute_cost

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_known_models_present() -> None:
    """Sanity-check that all model ids the rest of v1 may use are priced.

    If a new model id is added to `config.analyzer` (sonnet_model_id /
    opus_model_id) and missing here, `compute_cost` will raise.
    """
    for model_id in (
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
    ):
        assert model_id in PRICING


def test_opus_5_tier_pricing() -> None:
    """claude-opus-5 and claude-opus-4-8 are priced at the same $5/$25 per-1M
    tier as claude-opus-4-7 — the operator can flip `opus_model_id` between
    any of the three with no cost-model change."""
    reference = PRICING["claude-opus-4-7"]
    for model_id in ("claude-opus-5", "claude-opus-4-8"):
        assert PRICING[model_id] == reference


def test_compute_cost_resolves_for_new_opus_ids() -> None:
    """compute_cost must not raise UnknownModel for the Opus 5 review tier
    or the 4.8 fallback (the config-flip failure mode this table guards)."""
    for model_id in ("claude-opus-5", "claude-opus-4-8"):
        cost = compute_cost(
            model_id,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )
        assert cost == pytest.approx(5.00 + 25.00, rel=1e-9)


def _example_toml_model_ids() -> set[str]:
    """Every `*model_id` value configured anywhere in the example config."""
    with open(_REPO_ROOT / "config" / "quinn.example.toml", "rb") as f:
        cfg = tomllib.load(f)
    ids: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("model_id") and isinstance(value, str):
                    ids.add(value)
                else:
                    walk(value)

    walk(cfg)
    return ids


def test_all_configured_model_ids_have_pricing() -> None:
    """Guard against the UnknownModel-at-runtime failure mode on model swaps.

    Every model id reachable from default configuration — the example toml
    (the template the droplet config is derived from) plus the desk-journal
    Haiku default baked into code — must resolve in the pricing table, or
    the first live call after a config flip raises UnknownModel.
    """
    from app.desk_journal import HAIKU_MODEL_ID

    configured = _example_toml_model_ids() | {HAIKU_MODEL_ID}
    assert configured, "expected at least one *model_id in quinn.example.toml"
    missing = sorted(m for m in configured if m not in PRICING)
    assert not missing, f"model ids without pricing entries: {missing}"
    for model_id in configured:
        # Must not raise UnknownModel.
        compute_cost(
            model_id,
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=1,
            cache_creation_tokens=1,
        )


def test_haiku_dated_alias_matches_undated_pricing() -> None:
    """The dated Haiku 4.5 id (`claude-haiku-4-5-20251001`) must be priced
    identically to the undated `claude-haiku-4-5` entry — they reference
    the same underlying model. This guards against drift if the table
    is hand-edited.
    """
    undated = PRICING["claude-haiku-4-5"]
    dated = PRICING["claude-haiku-4-5-20251001"]
    assert dated == undated


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
