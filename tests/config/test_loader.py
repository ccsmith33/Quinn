"""S1.4 — config loader tests (architecture §10.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.loader import AppConfig, ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE_TOML = REPO_ROOT / "config" / "quinn.example.toml"


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
