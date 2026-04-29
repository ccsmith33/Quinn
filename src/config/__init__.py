"""Config package — non-secret config (loader) and secrets (env-var loader).

Architecture §9.4 (secrets), §10.2 (config). NFR-15: secrets never appear in
code, prompts, logs, or the journal.
"""

from .loader import (
    AnalyzerConfig,
    AppConfig,
    ConfigError,
    ExecutionConfig,
    IngestionConfig,
    KillSwitchConfig,
    ObservabilityConfig,
    PrefilterConfig,
    ReconcilerConfig,
    load_config,
)
from .secrets import MissingSecret, Secrets, load_secrets, redact

__all__ = [
    "AnalyzerConfig",
    "AppConfig",
    "ConfigError",
    "ExecutionConfig",
    "IngestionConfig",
    "KillSwitchConfig",
    "MissingSecret",
    "ObservabilityConfig",
    "PrefilterConfig",
    "ReconcilerConfig",
    "Secrets",
    "load_config",
    "load_secrets",
    "redact",
]
