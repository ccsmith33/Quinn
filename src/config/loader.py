"""Non-secret configuration loader (architecture §10.2).

Loads `/opt/quinn/config/quinn.toml` (production) or a path provided by the
caller. Returns a typed Pydantic v2 `AppConfig`. Secrets are NOT read here —
see `config.secrets`.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(Exception):
    """Raised on invalid configuration content or file errors."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestionConfig(_Section):
    rss_poll_seconds_market: int = Field(gt=0)
    rss_poll_seconds_offhours: int = Field(gt=0)
    reconciler_interval_seconds: int = Field(gt=0)
    edgar_user_agent: str
    rss_cursor_path: str = Field(default="/var/lib/quinn/state/rss_cursor.json")
    raw_filings_root: str = Field(default="/var/lib/quinn/raw")
    # Hotfix 2026-05-04 — local persistence of SEC `company_tickers.json`
    # so the resolver survives SEC outages with a warm cache.
    ticker_cache_path: str = Field(default="/var/lib/quinn/cache/cik_ticker_map.json")


class PrefilterConfig(_Section):
    similarity_threshold: float = Field(ge=0.0, le=1.0)
    minhash_perms: int = Field(gt=0)
    form_4_enabled: bool = Field(default=True)


class AnalyzerConfig(_Section):
    sonnet_model_id: str
    opus_model_id: str
    opus_review_conviction_threshold: int = Field(ge=0, le=10)
    sonnet_max_output_tokens: int = Field(gt=0)
    # Per S5.6 carry-forward (S5.4 reviewer / S5.2 reviewer C-1): the
    # SDK's `max_tokens` was hardcoded at 16000 prior to S5.6; both
    # Sonnet and Opus must pull from typed config to honour the
    # per-call cost ceiling. See D-052.
    opus_max_output_tokens: int = Field(default=4096, gt=0)


class Ks5Tier(_Section):
    """One breakpoint of the KS-5 dynamic concurrent-position curve (D-087
    fix #3). `max_positions` applies while equity is `<= equity_max`.

    Tiers are an *opt-in* sublinear curve: small accounts trade a narrow
    book, larger accounts widen it with diminishing returns rather than
    linearly. The list is ordered, strictly ascending in `equity_max`, and
    always clamped by `ExecutionConfig.ks5_max_concurrent` (the hard
    ceiling) so the curve can never widen the book past the configured cap.
    """

    equity_max: float = Field(gt=0.0)
    max_positions: int = Field(ge=0)


class TrailStage(_Section):
    """One milestone of the staged trail-tightening curve. When the
    position's high-water gain vs entry (as a percent of entry) reaches
    `gain_pct`, the effective trail width becomes min(current width,
    `trail_pct`) — stages only ever TIGHTEN the trail on top of the
    position's base width (proposal `trail_distance_pct` or the clamped
    default), never widen it.

    The list is ordered, strictly ascending in `gain_pct`, with
    `trail_pct` non-increasing. Empty (the default) means the flat
    base-width behavior is unchanged.
    """

    gain_pct: float = Field(gt=0.0)
    trail_pct: float = Field(gt=0.0, lt=100.0)


class ExecutionConfig(_Section):
    broker_mode: Literal["paper", "live"]
    ks4_pct_cap: float = Field(ge=0.0, le=1.0)
    ks4_absolute_cap_usd: float = Field(ge=0.0)
    # KS-5 concurrent-position cap. `ks5_max_concurrent` is the hard ceiling
    # (and the *only* required knob — a legacy config carrying just this key
    # keeps booting and behaving flat). D-087 fix #3 adds an opt-in tiered
    # curve via `ks5_tiers`: when non-empty, the effective cap scales with
    # equity along those breakpoints, clamped by the ceiling. Empty (the
    # default) means the cap stays flat at `ks5_max_concurrent`.
    ks5_max_concurrent: int = Field(ge=0)
    ks5_tiers: list[Ks5Tier] = Field(default_factory=list)
    ks7_cash_reserve_pct: float = Field(ge=0.0, le=1.0)
    sizing_mid_pct: float = Field(ge=0.0, le=1.0)
    sizing_high_pct: float = Field(ge=0.0, le=1.0)
    # Conviction at/above which a proposal earns `sizing_high_pct` (below it,
    # `sizing_mid_pct`). Default 9 preserves the original PRD §6.1 tiering
    # exactly; bounded to [6, 10] so the high tier can't swallow the mid band.
    sizing_high_conviction_min: int = Field(default=9, ge=6, le=10)
    # PDT-SUNSET-2026-06-04: ADR-009 §8 — operator escape hatch for the
    # PDT day-trade budget feature. Default True so the feature is on by
    # default; flip to False to short-circuit `PDTState.is_active()` and
    # revert to broker-side pre-placement of stops/TPs.
    pdt_enabled: bool = Field(default=True)
    # D-079 §3.2 — exit-geometry floor: minimum reward:risk a proposal
    # with a take-profit must clear. Tunable when H3 prod data lands.
    min_reward_risk: float = Field(default=1.5, gt=0.0)
    # Trade-time hard price floor (FR-8 / ADR-006 §3). The daily universe
    # snapshot screens on previous-close >= $5; this is the re-check at
    # submission time. Configurable since 2026-07-17: a $946M-cap
    # conviction-8 (ABUS at $4.79) was rejected for dipping cents under
    # $5 intraday after passing the >=$5 universe screen. Default
    # preserves behavior exactly.
    price_floor_usd: float = Field(default=5.0, gt=0.0)
    # D-079 §3.5 / ADR-011 — trailing ratchet: engage once the winner has
    # earned `trail_activation_r` multiples of its initial risk distance;
    # raise the broker stop only when the new target clears the current
    # stop by `min_ratchet_step_pct` percent (bounds PATCH chatter).
    trail_activation_r: float = Field(default=1.0, gt=0.0)
    min_ratchet_step_pct: float = Field(default=0.25, ge=0.0)
    # Staged trail tightening — opt-in curve of {gain_pct, trail_pct}
    # milestones (see `TrailStage`). Derived at runtime from config +
    # current high-water mark each tick, so a config change applies to
    # already-open positions on restart. Empty (the default) = the flat
    # base-width trail, byte-identical to pre-feature behavior.
    trail_stages: list[TrailStage] = Field(default_factory=list)
    # Breakeven floor (study policy E) — once a position's high-water gain
    # vs entry clears this percent, the trailing stop is never allowed to
    # sit below the entry price (small confirmed winners can no longer
    # round-trip into losers). HWM-based, applied as a lower bound AFTER
    # the width/stage math so it NEVER narrows the trail band. Derived at
    # runtime each tick, so a config change applies to already-open
    # positions on restart. 0.0 (the default) = feature OFF, byte-identical
    # to the pre-feature trail. Bounded [0, 50].
    breakeven_floor_gain_pct: float = Field(default=0.0, ge=0.0, le=50.0)
    # Feature A — cap the FIRST thesis-review interval. The analyzer's
    # declared catalyst horizon can run ~30d, which leaves a fresh position
    # un-re-evaluated for a month. This bounds the entry-time review so a
    # position is looked at again at least this often; the first review
    # fires at min(declared horizon, this cap). Subsequent reviews already
    # recur weekly via thesis_coordinator.HOLD_RESCHEDULE_DAYS. Default 7
    # = weekly first look.
    max_initial_review_days: int = Field(default=7, gt=0)
    # DISPLACEMENT — when a HIGH-conviction proposal is rejected by the
    # KS-7 cash path, deterministically evict the weakest qualifying open
    # position (trail NOT armed, held >= 5 trading days, not entered
    # today, unrealized gain below the cutoff) and fund the new entry
    # from the same-open sale proceeds. Default OFF: with
    # `displacement_enabled = false` behavior is byte-identical. Hard cap
    # of ONE displacement per trading day is enforced in code (not
    # configurable). See execution/displacement.py.
    displacement_enabled: bool = Field(default=False)
    # Only proposals at/above this conviction may displace. Same [6, 10]
    # band as sizing_high_conviction_min so the knob can't reach into the
    # mid tier.
    displacement_min_conviction: int = Field(default=8, ge=6, le=10)
    # Fraction of the victim's current market value credited as fundable
    # cash when sizing the new entry (the haircut absorbs slippage between
    # the sizing decision and the victim sale's fill).
    displacement_proceeds_haircut: float = Field(default=0.90, gt=0.0, le=1.0)
    # Victim gain cutoff (percent unrealized gain vs cost): positions
    # at/above this are not evictable at the base tier. GRADUATED
    # eligibility: proposals with conviction >= 9 ignore this cutoff —
    # armed positions stay untouchable at every conviction.
    displacement_victim_max_gain_pct: float = Field(default=5.0)
    # CHOPPING BLOCK — consume the daily pre-market sweep's pre-ranked
    # sacrifice list for victim selection. When on (default) and a block
    # exists, displacement iterates victims in thesis-sanctioned block-rank
    # order instead of the deterministic weakest-price order; with no block
    # (feature not producing one) behavior is byte-identical to before.
    displacement_use_chopping_block: bool = Field(default=True)
    # AGE OVERRIDE — when a proposal's conviction is at/above this, and NO
    # normally age-eligible (>= 5 trading days) victim exists, displacement
    # may evict a YOUNGER victim, but ONLY one on the chopping block
    # (thesis-sanctioned), still unarmed-only, weakest-rank-first. 0 =
    # DISABLED (the default — the young-victim age gate is never relaxed).
    # Any other value must be in [6, 10]; the operator sets 8 or 9
    # explicitly. Requires displacement_use_chopping_block.
    displacement_young_victim_min_conviction: int = Field(default=0)
    # WATCHLIST / deferred entry (default OFF). An approved proposal
    # whose execution was rejected for CAPITAL reasons only
    # (ks5_concurrent_limit / ks7_cash_reserve / insufficient_capital)
    # with conviction >= watchlist_min_conviction is parked in the
    # `watchlist` table and retried through the NORMAL execution path
    # (validator + sizing + every KS gate) each agent tick during market
    # hours once capital frees. The 33k-event day-1 study
    # (big_winner_day1_study.md) shows day+1/+2 entry retains ~98% of
    # eventual big-winner P&L, so nearly all the day-0 value survives a
    # short deferral. 0 (the default) disables the feature entirely: no
    # table writes, no tick work — byte-identical to pre-feature.
    watchlist_min_conviction: int = Field(default=0, ge=0, le=10)
    # How long a parked proposal stays retryable, in TRADING days (the
    # study's edge decays fast — a name not entered within a couple of
    # sessions has usually either run or gone stale).
    watchlist_expiry_trading_days: int = Field(default=3, ge=1, le=10)
    # Chase guard (the study's slippage/momentum caveat): never pay more
    # than reference_price * (1 + this/100). Exactly AT the ceiling is
    # still allowed; strictly above resolves the row `skipped_chase`.
    watchlist_max_chase_pct: float = Field(default=8.0, ge=0.0, le=25.0)

    @model_validator(mode="after")
    def _check_young_victim_conviction_band(self) -> ExecutionConfig:
        """0 disables the age override; any other value must sit in the
        same [6, 10] conviction band as the other displacement knobs so it
        can't reach into the mid tier. Rejected at load time otherwise."""
        v = self.displacement_young_victim_min_conviction
        if v != 0 and not (6 <= v <= 10):
            raise ValueError(
                "displacement_young_victim_min_conviction must be 0 "
                f"(disabled) or in [6, 10]; got {v}"
            )
        return self

    @model_validator(mode="after")
    def _check_ks5_tiers_monotonic(self) -> ExecutionConfig:
        """Breakpoints must be strictly ascending in `equity_max` so the band
        lookup is unambiguous. A misordered list is an operator error and is
        rejected at load time rather than silently mis-banding in prod.
        """
        thresholds = [t.equity_max for t in self.ks5_tiers]
        if any(b <= a for a, b in zip(thresholds, thresholds[1:], strict=False)):
            raise ValueError(
                "ks5_tiers equity_max values must be strictly ascending; "
                f"got {thresholds}"
            )
        return self

    @model_validator(mode="after")
    def _check_trail_stages_monotonic(self) -> ExecutionConfig:
        """Milestones must be strictly ascending in `gain_pct` (unambiguous
        crossing order) with `trail_pct` non-increasing (the trail may only
        tighten as milestones are crossed). A misordered list is an operator
        error and is rejected at load time.
        """
        gains = [s.gain_pct for s in self.trail_stages]
        if any(b <= a for a, b in zip(gains, gains[1:], strict=False)):
            raise ValueError(
                "trail_stages gain_pct values must be strictly ascending; "
                f"got {gains}"
            )
        trails = [s.trail_pct for s in self.trail_stages]
        if any(b > a for a, b in zip(trails, trails[1:], strict=False)):
            raise ValueError(
                "trail_stages trail_pct values must be non-increasing; "
                f"got {trails}"
            )
        return self


class EventReviewsConfig(_Section):
    """Event-triggered, zone-aware thesis reviews (operator A/B feature).

    Default OFF: with `enabled = false` (or the section absent) no event
    trigger ever runs and behavior is byte-identical to the pure
    calendar-review system. The operator enables it on one box only as
    an A/B.

    Triggers (all schedule an immediate thesis review through the
    existing `thesis_review_schedule` seam — `due_at = now`,
    `scheduled_reason = 'event:...'`):
      - gain-cross: the position's high-water gain crosses a threshold
        (default thresholds: the `trail_stages` gain_pcts plus +10%
        arming; override via `gain_thresholds_pct`);
      - held-name filing: a newly ingested filing matches a held symbol;
      - tape anomaly: the quote moved >= `anomaly_move_pct` vs prev
        close intraday (price-only);
      - news headline (when the news client is available);
      - daily pre-market sweep (opt-in via `daily_sweep`): once per UTC
        day, on the first reconciler tick at/after
        `sweep_after_utc_hour` (default 11 ~= 7am ET), EVERY held
        position gets a review — the doctrine's fresh-buy test every
        morning. Once per day per position, restart-safe (keyed on a
        schedule row dated today).

    Anti-churn: per-position-per-trigger-type cooldown of
    `cooldown_hours` (default 24 ~= once per trading day), and no
    trigger fires while a review for the execution is already pending
    or due.
    """

    enabled: bool = False
    anomaly_move_pct: float = Field(default=7.0, gt=0.0)
    cooldown_hours: float = Field(default=24.0, gt=0.0)
    daily_sweep: bool = False
    # UTC hour-of-day after which the daily sweep may run (11 UTC ~= 7am
    # ET — inside the pre-market window, before the open).
    sweep_after_utc_hour: int = Field(default=11, ge=0, le=23)
    # Optional override of the gain-cross threshold list (percent gain
    # vs entry). Empty (default) = derive from execution.trail_stages
    # gain_pcts plus the +10% arming line.
    gain_thresholds_pct: list[float] = Field(default_factory=list)
    # Capacity-pressure sweep: when the daily pre-market sweep fires and
    # the portfolio is within `capacity_target_slots` of its KS-5
    # effective concurrent-position cap, every `event:daily_sweep` review
    # that morning gets an extra context block ranking the open positions
    # by weakness and biasing the reviewer (under the prompt's dead-money
    # doctrine) toward freeing the weakest NON-EXEMPT slots — so Quinn
    # keeps the ability to open new positions at the open. It is a nudge
    # to the existing LLM review, NOT a mechanical evictor, and the
    # doctrine's exemptions (armed trail, dated catalyst ahead, <5 trading
    # days) remain binding. Only meaningful when `daily_sweep = true`.
    # Range 0–5; 0 (the default) turns the capacity block off entirely.
    capacity_target_slots: int = Field(default=0, ge=0, le=5)

    @model_validator(mode="after")
    def _check_gain_thresholds_monotonic(self) -> EventReviewsConfig:
        """Threshold list must be strictly ascending so crossing order is
        unambiguous (same operator-error posture as ks5_tiers)."""
        t = self.gain_thresholds_pct
        if any(b <= a for a, b in zip(t, t[1:], strict=False)):
            raise ValueError(
                "gain_thresholds_pct values must be strictly ascending; "
                f"got {t}"
            )
        if any(v <= 0 for v in t):
            raise ValueError(
                f"gain_thresholds_pct values must be positive; got {t}"
            )
        return self


class MemoryConfig(_Section):
    """LLM memory layer (shared rail for the doctrine / symbol_history /
    desk_journal / calibration providers).

    `enabled` is the master gate — with it false (or the section absent,
    the default) no assembler is constructed and every LLM context build
    is byte-identical to the pre-memory system. The per-provider bools
    default true so that flipping the master on lights up whichever
    providers are registered; an operator silences one provider by
    setting its bool false without touching the others.
    """

    enabled: bool = False
    doctrine_enabled: bool = True
    symbol_history_enabled: bool = True
    desk_journal_enabled: bool = True
    calibration_enabled: bool = True


class ReconcilerConfig(_Section):
    interval_seconds_market: int = Field(gt=0)
    # RETIRED — WS1 (D-078, delta §2.2). Diff explanation is now keyed to
    # order lifecycle (pending fill, or filled within the last 2 ticks),
    # never to a sliding submitted_at window, so there is nothing left for
    # this knob to widen (the RC-2 time-bomb class is gone). The field is
    # still parsed (ignored) so a prod quinn.toml carrying the key keeps
    # booting; the cutover runbook removes the key, then this field can be
    # deleted.
    expected_fill_window_minutes: int = Field(default=10_080, gt=0)


class KillSwitchConfig(_Section):
    ks1_daily_loss_pct: float = Field(ge=0.0, le=1.0)
    ks2_trailing_dd_pct: float = Field(ge=0.0, le=1.0)
    ks3_consecutive_losses: int = Field(ge=0)
    # KS-2 acknowledged-drawdown watermark (2026-08 live-money daily-resume
    # ritual). After an operator resume of an auto:KS-2 halt, KS-2 stays
    # suppressed until the trailing drawdown deepens to acked_dd plus this
    # margin, expressed in PERCENTAGE POINTS (3.0 → re-fire at acked_dd +
    # 0.03) — unlike ks2_trailing_dd_pct, which is a 0–1 fraction. A new
    # 30-day equity peak above the acked peak expires the watermark. 0.0
    # (default) turns the feature OFF: same-day suppression only, behavior
    # identical to configs without this key.
    ks2_reack_margin_pct: float = Field(default=0.0, ge=0.0, le=10.0)
    # S7.3 — webhook fallback transport (ADR-004).
    webhook_port: int = Field(default=8443, ge=1, le=65535)
    webhook_counter_path: str = Field(default="/var/lib/quinn/state/webhook_counter")
    # WS1 (D-078, delta §2.3): while an identical fingerprinted halt
    # persists unresolved, re-page the operator at most once per this
    # many minutes (the kill-switch dedupes the rest — kills O-5).
    halt_repage_minutes: int = Field(default=240, gt=0)


class ObservabilityConfig(_Section):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    betterstack_endpoint: str


class DashboardConfig(_Section):
    """S9.1 D-063 — operator dashboard listener.

    Bind to localhost by default so the only public path is through Caddy
    (which terminates TLS + the operator-domain hostname check). Operators
    who want the dashboard on a private LAN without Caddy can override
    `bind_host` to `0.0.0.0` in `quinn.toml`.
    """

    port: int = Field(default=8444, ge=1, le=65535)
    bind_host: str = Field(default="127.0.0.1")
    auto_refresh_seconds: int = Field(default=30, gt=0)


class UniverseConfig(_Section):
    """D-065 — daily universe-refresh tunables.

    `yfinance_min_interval_seconds` enforces a per-call floor on the
    yfinance provider to stay below Yahoo's ~5 req/sec residential-IP
    throttle threshold. Default 0.2s = 5 req/sec. Lower values risk
    high failure rates; higher values lengthen daily refresh runtime.
    """

    yfinance_min_interval_seconds: float = Field(default=0.2, ge=0.0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ingestion: IngestionConfig
    prefilter: PrefilterConfig
    analyzer: AnalyzerConfig
    execution: ExecutionConfig
    # Event-triggered thesis reviews — optional section, default OFF so
    # legacy configs keep booting with byte-identical behavior.
    event_reviews: EventReviewsConfig = Field(default_factory=EventReviewsConfig)
    # LLM memory layer — optional section, default OFF (master gate) so
    # legacy configs keep booting with byte-identical LLM context.
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    reconciler: ReconcilerConfig
    killswitch: KillSwitchConfig
    observability: ObservabilityConfig
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)


_DEFAULT_PATHS = [
    "/opt/quinn/config/quinn.toml",
    "config/quinn.toml",
]


def _resolve_path(path: str | None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.environ.get("QUINN_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    for candidate in _DEFAULT_PATHS:
        p = Path(candidate)
        if p.exists():
            return p
    raise ConfigError(
        "no config path provided and no default config file found "
        f"(checked QUINN_CONFIG_PATH and {_DEFAULT_PATHS})"
    )


def load_config(path: str | None = None) -> AppConfig:
    cfg_path = _resolve_path(path)
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    try:
        with cfg_path.open("rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {cfg_path}: {e}") from e
    try:
        return AppConfig(**raw)
    except ValidationError as e:
        raise ConfigError(f"invalid config in {cfg_path}: {e}") from e
