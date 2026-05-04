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

from pydantic import BaseModel, ConfigDict, Field, ValidationError


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


class ExecutionConfig(_Section):
    broker_mode: Literal["paper", "live"]
    ks4_pct_cap: float = Field(ge=0.0, le=1.0)
    ks4_absolute_cap_usd: float = Field(ge=0.0)
    ks5_max_concurrent: int = Field(ge=0)
    ks7_cash_reserve_pct: float = Field(ge=0.0, le=1.0)
    sizing_mid_pct: float = Field(ge=0.0, le=1.0)
    sizing_high_pct: float = Field(ge=0.0, le=1.0)


class ReconcilerConfig(_Section):
    interval_seconds_market: int = Field(gt=0)


class KillSwitchConfig(_Section):
    ks1_daily_loss_pct: float = Field(ge=0.0, le=1.0)
    ks2_trailing_dd_pct: float = Field(ge=0.0, le=1.0)
    ks3_consecutive_losses: int = Field(ge=0)
    # S7.3 — webhook fallback transport (ADR-004).
    webhook_port: int = Field(default=8443, ge=1, le=65535)
    webhook_counter_path: str = Field(default="/var/lib/quinn/state/webhook_counter")


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
