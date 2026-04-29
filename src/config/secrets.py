"""Secrets loader (architecture §9.4, NFR-15).

Reads required secrets from environment variables. Values are wrapped in
`SecretStr` so `repr`/`str` redact automatically and accidental logging is
blocked at the model boundary. Other modules call `load_secrets()` once at
process start and pass the model around — they do NOT read env vars directly.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, SecretStr

_REQUIRED_KEYS: tuple[tuple[str, str], ...] = (
    ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    ("alpaca_api_key_id", "ALPACA_API_KEY_ID"),
    ("alpaca_api_secret_key", "ALPACA_API_SECRET_KEY"),
    ("alpaca_endpoint", "ALPACA_ENDPOINT"),
    ("telegram_bot_token", "TELEGRAM_BOT_TOKEN"),
    ("telegram_operator_chat_id", "TELEGRAM_OPERATOR_CHAT_ID"),
    ("kill_switch_hmac_key", "KILL_SWITCH_HMAC_KEY"),
    ("backup_b2_key_id", "BACKUP_B2_KEY_ID"),
    ("backup_b2_application_key", "BACKUP_B2_APPLICATION_KEY"),
    ("backup_b2_bucket", "BACKUP_B2_BUCKET"),
    ("log_sink_token", "LOG_SINK_TOKEN"),
    # Operator dashboard (S9.1 D-063). Required because the dashboard
    # systemd unit refuses to boot without credentials — there is no
    # "anonymous" mode.
    ("dashboard_user", "DASHBOARD_USER"),
    ("dashboard_password", "DASHBOARD_PASSWORD"),
)


class MissingSecret(Exception):
    """Raised when one or more required secret env vars are absent or empty."""

    def __init__(self, keys: list[str]) -> None:
        super().__init__(f"missing required secrets: {', '.join(keys)}")
        self.keys = keys


class Secrets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anthropic_api_key: SecretStr
    alpaca_api_key_id: SecretStr
    alpaca_api_secret_key: SecretStr
    alpaca_endpoint: SecretStr
    telegram_bot_token: SecretStr
    telegram_operator_chat_id: SecretStr
    kill_switch_hmac_key: SecretStr
    backup_b2_key_id: SecretStr
    backup_b2_application_key: SecretStr
    backup_b2_bucket: SecretStr
    log_sink_token: SecretStr
    operator_notify_email: SecretStr | None = None
    # S9.1 D-063 — operator dashboard creds; constant-time-compared in
    # `src/dashboard/auth.py` against the inbound HTTP basic header.
    dashboard_user: SecretStr
    dashboard_password: SecretStr
    # S9.1 D-063 — DuckDNS pair is only needed when running the dashboard
    # behind Caddy on a public-internet droplet with auto-DNS; absent on
    # a LAN-only / development box.
    duckdns_token: SecretStr | None = None
    duckdns_domain: SecretStr | None = None


def load_secrets() -> Secrets:
    missing: list[str] = []
    values: dict[str, Any] = {}
    for field_name, env_name in _REQUIRED_KEYS:
        v = os.environ.get(env_name)
        if v is None or v == "":
            missing.append(env_name)
        else:
            values[field_name] = SecretStr(v)
    if missing:
        raise MissingSecret(missing)
    optional_email = os.environ.get("OPERATOR_NOTIFY_EMAIL")
    values["operator_notify_email"] = SecretStr(optional_email) if optional_email else None
    duckdns_token = os.environ.get("DUCKDNS_TOKEN")
    values["duckdns_token"] = SecretStr(duckdns_token) if duckdns_token else None
    duckdns_domain = os.environ.get("DUCKDNS_DOMAIN")
    values["duckdns_domain"] = SecretStr(duckdns_domain) if duckdns_domain else None
    return Secrets(**values)


def _all_secret_values(secrets: Secrets) -> list[str]:
    out: list[str] = []
    for field_name in secrets.__class__.model_fields:
        v = getattr(secrets, field_name)
        if v is None:
            continue
        raw = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        if raw:
            out.append(raw)
    return out


def redact(text: str, secrets: Secrets) -> str:
    """Replace any secret value substring in `text` with `[REDACTED]`.

    Sort longest-first so a short secret that happens to be a substring of a
    longer one is not redacted prematurely. Empty / falsy values are skipped.
    """
    if not text:
        return text
    values = sorted(_all_secret_values(secrets), key=len, reverse=True)
    out = text
    for v in values:
        if v and v in out:
            out = out.replace(v, "[REDACTED]")
    return out
