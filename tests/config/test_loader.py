"""S1.4 — config loader tests (architecture §10.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from config.loader import AppConfig, ConfigError, ExecutionConfig, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE_TOML = REPO_ROOT / "config" / "quinn.example.toml"


def _execution_kwargs(**overrides: object) -> dict[str, object]:
    """Minimal valid ExecutionConfig kwargs; overridable per-test."""
    base: dict[str, object] = {
        "broker_mode": "paper",
        "ks4_pct_cap": 0.15,
        "ks4_absolute_cap_usd": 1000.0,
        "ks5_max_concurrent": 10,
        "ks7_cash_reserve_pct": 0.03,
        "sizing_mid_pct": 0.07,
        "sizing_high_pct": 0.10,
    }
    base.update(overrides)
    return base


def test_load_example_config_builds_appconfig() -> None:
    cfg = load_config(str(EXAMPLE_TOML))
    assert isinstance(cfg, AppConfig)
    # spot-check a few values from §10.2
    assert cfg.ingestion.rss_poll_seconds_market == 60
    assert cfg.prefilter.similarity_threshold == 0.97
    assert cfg.analyzer.opus_review_conviction_threshold == 5
    assert cfg.execution.broker_mode in ("paper", "live")
    assert cfg.killswitch.ks1_daily_loss_pct == 0.03
    assert cfg.observability.log_level == "INFO"
    # S9.1 D-063 — dashboard config defaults to port 8444 when omitted.
    assert cfg.dashboard.port == 8444
    assert cfg.dashboard.bind_host == "127.0.0.1"


def test_invalid_broker_mode_rejected(tmp_path: Path) -> None:
    bad = EXAMPLE_TOML.read_text(encoding="utf-8").replace(
        'broker_mode = "paper"', 'broker_mode = "demo"'
    )
    p = tmp_path / "bad.toml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(p))


def test_load_config_default_path_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    # When path is None, loader looks at config/quinn.toml or QUINN_CONFIG_PATH.
    monkeypatch.setenv("QUINN_CONFIG_PATH", str(EXAMPLE_TOML))
    cfg = load_config()
    assert isinstance(cfg, AppConfig)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    minimal = """
[ingestion]
rss_poll_seconds_market = 60
"""
    p = tmp_path / "minimal.toml"
    p.write_text(minimal, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(p))


# ---------------------------------------------------------------------------
# KS-5 tiered dynamic-cap (D-087 fix #3) — backward-compatible, opt-in curve.
# ---------------------------------------------------------------------------

def test_ks5_tiers_default_to_empty_scaling_disabled() -> None:
    """A config carrying only `ks5_max_concurrent` (the legacy prod shape)
    must still construct, with the tier list defaulting to empty so the
    cap stays flat at `ks5_max_concurrent` (zero behavior change).
    """
    cfg = ExecutionConfig(**_execution_kwargs())  # type: ignore[arg-type]
    assert cfg.ks5_tiers == []


def test_ks5_tiers_parse_array_of_tables() -> None:
    cfg = ExecutionConfig(
        **_execution_kwargs(
            ks5_max_concurrent=30,
            ks5_tiers=[
                {"equity_max": 5000.0, "max_positions": 10},
                {"equity_max": 15000.0, "max_positions": 15},
            ],
        )  # type: ignore[arg-type]
    )
    assert len(cfg.ks5_tiers) == 2
    assert cfg.ks5_tiers[0].equity_max == pytest.approx(5000.0)
    assert cfg.ks5_tiers[0].max_positions == 10
    assert cfg.ks5_tiers[1].max_positions == 15


def test_ks5_tiers_reject_non_monotonic_thresholds() -> None:
    """Breakpoints must be strictly ascending in equity_max so the band
    lookup is unambiguous; a misordered list is an operator error.
    """
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(
                ks5_tiers=[
                    {"equity_max": 15000.0, "max_positions": 15},
                    {"equity_max": 5000.0, "max_positions": 10},
                ]
            )  # type: ignore[arg-type]
        )


def test_ks5_tier_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(
                ks5_tiers=[{"equity_max": 0.0, "max_positions": 10}]
            )  # type: ignore[arg-type]
        )


def test_ks5_tier_rejects_negative_max_positions() -> None:
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(
                ks5_tiers=[{"equity_max": 5000.0, "max_positions": -1}]
            )  # type: ignore[arg-type]
        )


def test_example_toml_disables_scaling_by_default() -> None:
    """The shipped example config must boot with the curve OFF so operators
    upgrade with zero behavior change until they explicitly opt in.
    """
    cfg = load_config(str(EXAMPLE_TOML))
    assert cfg.execution.ks5_tiers == []
    assert cfg.execution.ks5_max_concurrent == 10
