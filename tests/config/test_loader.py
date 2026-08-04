"""S1.4 — config loader tests (architecture §10.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from config.loader import (
    AppConfig,
    ConfigError,
    EventReviewsConfig,
    ExecutionConfig,
    MemoryConfig,
    load_config,
)

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
# Capacity-pressure sweep — [event_reviews] capacity_target_slots.
# ---------------------------------------------------------------------------


def test_event_reviews_defaults_to_off() -> None:
    """No [event_reviews] section → the feature is off (target 0)."""
    cfg = EventReviewsConfig()
    assert cfg.capacity_target_slots == 0


def test_example_config_event_reviews_off_by_default() -> None:
    """The example ships [event_reviews] commented out, so a config built
    from it has the whole feature off — capacity_target_slots 0 included."""
    cfg = load_config(str(EXAMPLE_TOML))
    assert cfg.event_reviews.enabled is False
    assert cfg.event_reviews.capacity_target_slots == 0


def test_event_reviews_capacity_target_slots_parses_from_toml(tmp_path: Path) -> None:
    """A live [event_reviews] table with capacity_target_slots parses onto
    the existing section alongside the sweep flags."""
    text = EXAMPLE_TOML.read_text(encoding="utf-8")
    text += (
        "\n[event_reviews]\n"
        "enabled = true\n"
        "daily_sweep = true\n"
        "capacity_target_slots = 2\n"
    )
    p = tmp_path / "with_capacity.toml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.event_reviews.capacity_target_slots == 2
    assert cfg.event_reviews.daily_sweep is True


# ---------------------------------------------------------------------------
# [memory] — LLM memory layer (master gate + per-provider bools)
# ---------------------------------------------------------------------------


def test_memory_section_absent_defaults_off() -> None:
    """The example config carries no [memory] table — back-compat: the
    section defaults in with the master gate OFF and every provider bool
    ON (the master gate is what actually suppresses them)."""
    cfg = load_config(str(EXAMPLE_TOML))
    assert cfg.memory.enabled is False
    assert cfg.memory.doctrine_enabled is True
    assert cfg.memory.symbol_history_enabled is True
    assert cfg.memory.desk_journal_enabled is True
    assert cfg.memory.calibration_enabled is True


def test_memory_section_parses_from_toml(tmp_path: Path) -> None:
    text = EXAMPLE_TOML.read_text(encoding="utf-8")
    text += (
        "\n[memory]\n"
        "enabled = true\n"
        "symbol_history_enabled = false\n"
    )
    p = tmp_path / "with_memory.toml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.memory.enabled is True
    # explicit override sticks; the untouched bools keep their default
    assert cfg.memory.symbol_history_enabled is False
    assert cfg.memory.doctrine_enabled is True


def test_memory_forbids_unknown_key() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(bogus_provider_enabled=True)  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [-1, 6, 100])
def test_event_reviews_range_rejected(bad: int) -> None:
    with pytest.raises(ValidationError):
        EventReviewsConfig(capacity_target_slots=bad)


@pytest.mark.parametrize("good", [0, 1, 2, 5])
def test_event_reviews_range_accepted(good: int) -> None:
    assert EventReviewsConfig(capacity_target_slots=good).capacity_target_slots == good


def test_event_reviews_forbids_unknown_key() -> None:
    with pytest.raises(ValidationError):
        EventReviewsConfig(capacity_target_slots=2, bogus=1)  # type: ignore[call-arg]


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


def test_trail_stages_default_to_empty() -> None:
    """A config with no `trail_stages` (every existing prod shape) must
    construct with an empty list — staged tightening OFF, flat base-width
    trailing exactly as before."""
    cfg = ExecutionConfig(**_execution_kwargs())  # type: ignore[arg-type]
    assert cfg.trail_stages == []


def test_trail_stages_parse_array_of_tables() -> None:
    cfg = ExecutionConfig(
        **_execution_kwargs(
            trail_stages=[
                {"gain_pct": 20.0, "trail_pct": 8.0},
                {"gain_pct": 35.0, "trail_pct": 5.0},
            ],
        )  # type: ignore[arg-type]
    )
    assert [(s.gain_pct, s.trail_pct) for s in cfg.trail_stages] == [
        (20.0, 8.0),
        (35.0, 5.0),
    ]


def test_trail_stages_reject_non_ascending_gain_pct() -> None:
    """Milestones must be strictly ascending in gain_pct; a misordered
    (or duplicated) list is an operator error, rejected at load time."""
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(
                trail_stages=[
                    {"gain_pct": 35.0, "trail_pct": 8.0},
                    {"gain_pct": 20.0, "trail_pct": 5.0},
                ],
            )  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(
                trail_stages=[
                    {"gain_pct": 20.0, "trail_pct": 8.0},
                    {"gain_pct": 20.0, "trail_pct": 5.0},
                ],
            )  # type: ignore[arg-type]
        )


def test_trail_stages_reject_widening_trail_pct() -> None:
    """trail_pct must be non-increasing across stages — the trail may
    only tighten as gain milestones are crossed, never widen."""
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(
                trail_stages=[
                    {"gain_pct": 20.0, "trail_pct": 5.0},
                    {"gain_pct": 35.0, "trail_pct": 8.0},
                ],
            )  # type: ignore[arg-type]
        )


def test_trail_stage_rejects_non_positive_fields() -> None:
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(
                trail_stages=[{"gain_pct": 0.0, "trail_pct": 8.0}],
            )  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(
                trail_stages=[{"gain_pct": 20.0, "trail_pct": 0.0}],
            )  # type: ignore[arg-type]
        )


def test_breakeven_floor_defaults_off() -> None:
    """A config with no `breakeven_floor_gain_pct` (every existing prod
    shape) must default to 0.0 — the breakeven floor is OFF, byte-identical
    to pre-feature trailing behavior."""
    cfg = ExecutionConfig(**_execution_kwargs())  # type: ignore[arg-type]
    assert cfg.breakeven_floor_gain_pct == 0.0


def test_breakeven_floor_parses_configured_value() -> None:
    cfg = ExecutionConfig(
        **_execution_kwargs(breakeven_floor_gain_pct=12.0)  # type: ignore[arg-type]
    )
    assert cfg.breakeven_floor_gain_pct == pytest.approx(12.0)


def test_breakeven_floor_rejects_out_of_range() -> None:
    """Valid band is 0–50; negative or >50 is an operator error rejected
    at load time."""
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(breakeven_floor_gain_pct=-1.0)  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(breakeven_floor_gain_pct=50.1)  # type: ignore[arg-type]
        )


def test_example_toml_disables_scaling_by_default() -> None:
    """The shipped example config must boot with the curve OFF so operators
    upgrade with zero behavior change until they explicitly opt in.
    """
    cfg = load_config(str(EXAMPLE_TOML))
    assert cfg.execution.ks5_tiers == []
    assert cfg.execution.ks5_max_concurrent == 10


def test_max_initial_review_days_defaults_to_weekly() -> None:
    """Feature A — a config omitting the key (the legacy prod shape, and the
    shipped example) defaults the first-review cap to 7 days so fresh
    positions get a weekly re-look without operator action.
    """
    cfg = ExecutionConfig(**_execution_kwargs())  # type: ignore[arg-type]
    assert cfg.max_initial_review_days == 7
    example = load_config(str(EXAMPLE_TOML))
    assert example.execution.max_initial_review_days == 7


def test_max_initial_review_days_override_parses() -> None:
    """An operator may tune the first-review cap; a positive override parses
    and supersedes the default."""
    cfg = ExecutionConfig(
        **_execution_kwargs(max_initial_review_days=3)  # type: ignore[arg-type]
    )
    assert cfg.max_initial_review_days == 3


def test_max_initial_review_days_rejects_non_positive() -> None:
    """Zero or negative makes no sense as a review interval and is an
    operator error rejected at load time (Field gt=0)."""
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(max_initial_review_days=0)  # type: ignore[arg-type]
        )


def test_sizing_high_conviction_min_defaults_to_9() -> None:
    """A config omitting the key (every existing prod shape, and the shipped
    example) defaults the high-tier threshold to 9 — zero behavior change."""
    cfg = ExecutionConfig(**_execution_kwargs())  # type: ignore[arg-type]
    assert cfg.sizing_high_conviction_min == 9
    example = load_config(str(EXAMPLE_TOML))
    assert example.execution.sizing_high_conviction_min == 9


def test_sizing_high_conviction_min_override_parses() -> None:
    """An operator may lower the high-tier threshold within [6, 10]."""
    cfg = ExecutionConfig(
        **_execution_kwargs(sizing_high_conviction_min=8)  # type: ignore[arg-type]
    )
    assert cfg.sizing_high_conviction_min == 8


def test_sizing_high_conviction_min_rejects_out_of_range() -> None:
    """Values outside [6, 10] are operator errors rejected at load time."""
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(sizing_high_conviction_min=5)  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        ExecutionConfig(
            **_execution_kwargs(sizing_high_conviction_min=11)  # type: ignore[arg-type]
        )


def test_event_reviews_defaults_off_when_section_absent() -> None:
    """[event_reviews] — the section is optional and defaults OFF so a
    legacy prod quinn.toml (and the shipped example, which only carries a
    commented block) keeps booting with byte-identical behavior."""
    cfg = load_config(str(EXAMPLE_TOML))
    assert cfg.event_reviews.enabled is False
    assert cfg.event_reviews.anomaly_move_pct == 7.0
    assert cfg.event_reviews.cooldown_hours == 24.0
    assert cfg.event_reviews.gain_thresholds_pct == []
    assert cfg.event_reviews.daily_sweep is False
    assert cfg.event_reviews.sweep_after_utc_hour == 11


def test_event_reviews_section_parses(tmp_path: Path) -> None:
    toml = EXAMPLE_TOML.read_text(encoding="utf-8") + (
        "\n[event_reviews]\n"
        "enabled = true\n"
        "anomaly_move_pct = 5.0\n"
        "cooldown_hours = 12.0\n"
        "gain_thresholds_pct = [10.0, 20.0, 35.0]\n"
        "daily_sweep = true\n"
        "sweep_after_utc_hour = 12\n"
    )
    p = tmp_path / "event_reviews.toml"
    p.write_text(toml, encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.event_reviews.enabled is True
    assert cfg.event_reviews.anomaly_move_pct == 5.0
    assert cfg.event_reviews.cooldown_hours == 12.0
    assert cfg.event_reviews.gain_thresholds_pct == [10.0, 20.0, 35.0]
    assert cfg.event_reviews.daily_sweep is True
    assert cfg.event_reviews.sweep_after_utc_hour == 12


def test_event_reviews_rejects_out_of_range_sweep_hour() -> None:
    """`sweep_after_utc_hour` is an hour-of-day (0-23); anything else is
    an operator error rejected at load time."""
    from config.loader import EventReviewsConfig

    with pytest.raises(ValidationError):
        EventReviewsConfig(sweep_after_utc_hour=24)
    with pytest.raises(ValidationError):
        EventReviewsConfig(sweep_after_utc_hour=-1)


def test_event_reviews_rejects_non_ascending_gain_thresholds() -> None:
    from config.loader import EventReviewsConfig

    with pytest.raises(ValidationError):
        EventReviewsConfig(gain_thresholds_pct=[20.0, 10.0])
    with pytest.raises(ValidationError):
        EventReviewsConfig(gain_thresholds_pct=[-5.0, 10.0])


# ---------------------------------------------------------------------------
# Watchlist / deferred entry — [execution] watchlist_* keys.
# ---------------------------------------------------------------------------


def test_watchlist_defaults_off() -> None:
    """Keys absent → feature OFF (min_conviction 0) with documented
    defaults for the other two knobs — legacy configs parse and behave
    byte-identically."""
    cfg = ExecutionConfig(**_execution_kwargs())
    assert cfg.watchlist_min_conviction == 0
    assert cfg.watchlist_expiry_trading_days == 3
    assert cfg.watchlist_max_chase_pct == 8.0


def test_example_toml_watchlist_off_by_default() -> None:
    """The example ships the watchlist keys commented out → OFF."""
    cfg = load_config(str(EXAMPLE_TOML))
    assert cfg.execution.watchlist_min_conviction == 0
    assert cfg.execution.watchlist_expiry_trading_days == 3
    assert cfg.execution.watchlist_max_chase_pct == 8.0


def test_watchlist_parses_from_toml(tmp_path: Path) -> None:
    toml = EXAMPLE_TOML.read_text(encoding="utf-8").replace(
        "sizing_high_pct = 0.10",
        "sizing_high_pct = 0.10\n"
        "watchlist_min_conviction = 6\n"
        "watchlist_expiry_trading_days = 2\n"
        "watchlist_max_chase_pct = 5.5\n",
    )
    p = tmp_path / "watchlist.toml"
    p.write_text(toml, encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.execution.watchlist_min_conviction == 6
    assert cfg.execution.watchlist_expiry_trading_days == 2
    assert cfg.execution.watchlist_max_chase_pct == 5.5


@pytest.mark.parametrize(
    "field,bad",
    [
        ("watchlist_min_conviction", -1),
        ("watchlist_min_conviction", 11),
        ("watchlist_expiry_trading_days", 0),
        ("watchlist_expiry_trading_days", 11),
        ("watchlist_max_chase_pct", -0.1),
        ("watchlist_max_chase_pct", 25.1),
    ],
)
def test_watchlist_rejects_out_of_range(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        ExecutionConfig(**_execution_kwargs(**{field: bad}))


@pytest.mark.parametrize(
    "field,good",
    [
        ("watchlist_min_conviction", 0),
        ("watchlist_min_conviction", 10),
        ("watchlist_expiry_trading_days", 1),
        ("watchlist_expiry_trading_days", 10),
        ("watchlist_max_chase_pct", 0.0),
        ("watchlist_max_chase_pct", 25.0),
    ],
)
def test_watchlist_accepts_bounds(field: str, good: object) -> None:
    cfg = ExecutionConfig(**_execution_kwargs(**{field: good}))
    assert getattr(cfg, field) == good
