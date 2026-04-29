"""S1.4 — secrets loader tests (NFR-15, architecture §9.4)."""

from __future__ import annotations

import pytest

from config.secrets import MissingSecret, Secrets, load_secrets, redact

_REQUIRED = [
    "ANTHROPIC_API_KEY",
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "ALPACA_ENDPOINT",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_OPERATOR_CHAT_ID",
    "KILL_SWITCH_HMAC_KEY",
    "BACKUP_B2_KEY_ID",
    "BACKUP_B2_APPLICATION_KEY",
    "BACKUP_B2_BUCKET",
    "LOG_SINK_TOKEN",
    "DASHBOARD_USER",
    "DASHBOARD_PASSWORD",
]


@pytest.fixture
def filled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _REQUIRED:
        monkeypatch.setenv(k, f"test-value-for-{k.lower()}")
    monkeypatch.delenv("OPERATOR_NOTIFY_EMAIL", raising=False)
    monkeypatch.delenv("DUCKDNS_TOKEN", raising=False)
    monkeypatch.delenv("DUCKDNS_DOMAIN", raising=False)


def test_load_secrets_happy_path(filled_env: None) -> None:
    s = load_secrets()
    assert isinstance(s, Secrets)
    # SecretStr doesn't expose value via attribute access without get_secret_value
    assert s.anthropic_api_key.get_secret_value() == "test-value-for-anthropic_api_key"
    assert s.alpaca_endpoint.get_secret_value() == "test-value-for-alpaca_endpoint"
    assert s.operator_notify_email is None  # optional
    # Dashboard creds are required (S9.1 D-063)
    assert s.dashboard_user.get_secret_value() == "test-value-for-dashboard_user"
    assert s.dashboard_password.get_secret_value() == "test-value-for-dashboard_password"
    # DuckDNS pair is optional (only set on a public-internet droplet)
    assert s.duckdns_token is None
    assert s.duckdns_domain is None


def test_load_secrets_with_optional_duckdns(
    filled_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DUCKDNS_TOKEN", "ddns-token-abc")
    monkeypatch.setenv("DUCKDNS_DOMAIN", "myquinn.duckdns.org")
    s = load_secrets()
    assert s.duckdns_token is not None
    assert s.duckdns_domain is not None
    assert s.duckdns_token.get_secret_value() == "ddns-token-abc"
    assert s.duckdns_domain.get_secret_value() == "myquinn.duckdns.org"


def test_missing_required_raises_listing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _REQUIRED:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("OPERATOR_NOTIFY_EMAIL", raising=False)
    with pytest.raises(MissingSecret) as exc_info:
        load_secrets()
    missing = exc_info.value.keys
    for k in _REQUIRED:
        assert k in missing


def test_missing_only_one_raises_with_just_that_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _REQUIRED:
        monkeypatch.setenv(k, "x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingSecret) as exc_info:
        load_secrets()
    assert exc_info.value.keys == ["ANTHROPIC_API_KEY"]


def test_secret_repr_redacts(filled_env: None) -> None:
    s = load_secrets()
    text_repr = repr(s)
    text_str = str(s)
    for k in _REQUIRED:
        raw = f"test-value-for-{k.lower()}"
        assert raw not in text_repr, f"raw secret leaked in repr: {raw}"
        assert raw not in text_str, f"raw secret leaked in str: {raw}"


def test_redact_helper(filled_env: None) -> None:
    s = load_secrets()
    raw = s.anthropic_api_key.get_secret_value()
    line = f"calling Anthropic with key={raw} for request 42"
    out = redact(line, s)
    assert raw not in out
    assert "[REDACTED]" in out
    assert "for request 42" in out


def test_redact_handles_empty_and_none_secrets(filled_env: None) -> None:
    s = load_secrets()
    # empty string input must not blow up
    assert redact("", s) == ""
    # text with no secret substring is unchanged
    assert redact("nothing sensitive here", s) == "nothing sensitive here"
