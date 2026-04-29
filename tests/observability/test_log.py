"""S8.1 — structured logging tests (NFR-9, NFR-15, FR-33).

TDD. Tests cover ACs 1-3, 5, 6, 7 of story-08-01-structured-logging.md.
The Vector config validation lives in `test_vector_config.py` (AC-4).
"""

from __future__ import annotations

import io
import json
import logging
import socket
import threading
import time

import pytest

from config.secrets import load_secrets
from observability import log as obs_log
from observability.log_port import get_logger

_REQUIRED_ENV = [
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
    for k in _REQUIRED_ENV:
        monkeypatch.setenv(k, f"test-value-for-{k.lower()}")
    monkeypatch.delenv("OPERATOR_NOTIFY_EMAIL", raising=False)


@pytest.fixture
def captured_stream(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Configure logging into an in-memory stream and return it.

    Resets structlog config + the stdlib root after to avoid bleed
    between tests.
    """
    buf = io.StringIO()
    obs_log.configure_logging(stream=buf, level=logging.DEBUG, secrets=None)
    yield buf
    obs_log.reset_logging()


# ---------------------------------------------------------------------------
# AC-1 — default fields injected on every event
# ---------------------------------------------------------------------------


def test_log_includes_default_fields(captured_stream: io.StringIO) -> None:
    log = get_logger("test.module")
    log.info("hello world", foo=1)

    line = captured_stream.getvalue().strip().splitlines()[-1]
    record = json.loads(line)

    assert record["service"] == "quinn"
    assert record["pid"] > 0
    assert record["host"] == socket.gethostname()
    assert record["level"] == "info"
    assert record["module"] == "test.module"
    assert record["event"] == "hello world"
    assert record["foo"] == 1
    # ISO-8601 timestamp present
    assert "timestamp" in record
    # very loose ISO check: year, dash, "T"
    assert record["timestamp"][4] == "-" and "T" in record["timestamp"]


# ---------------------------------------------------------------------------
# AC-2 — correlation_id contextvar propagation
# ---------------------------------------------------------------------------


def test_correlation_id_propagates_within_iteration(captured_stream: io.StringIO) -> None:
    log = get_logger("agent.loop")
    with obs_log.correlation_id_scope("iter-abc123"):
        log.info("step one")
        log.info("step two", extra_field="x")

    lines = captured_stream.getvalue().strip().splitlines()
    assert len(lines) == 2
    for raw in lines:
        rec = json.loads(raw)
        assert rec["correlation_id"] == "iter-abc123"


def test_correlation_id_absent_outside_scope(captured_stream: io.StringIO) -> None:
    log = get_logger("agent.loop")
    log.info("no scope")
    rec = json.loads(captured_stream.getvalue().strip().splitlines()[-1])
    assert "correlation_id" not in rec


def test_correlation_id_resets_after_scope(captured_stream: io.StringIO) -> None:
    log = get_logger("agent.loop")
    with obs_log.correlation_id_scope("iter-1"):
        log.info("inside")
    log.info("outside")

    inside, outside = (json.loads(line) for line in captured_stream.getvalue().strip().splitlines())
    assert inside["correlation_id"] == "iter-1"
    assert "correlation_id" not in outside


# ---------------------------------------------------------------------------
# AC-3 + AC-7 — secret redaction
# ---------------------------------------------------------------------------


def test_secret_redacted_in_log_output(monkeypatch: pytest.MonkeyPatch, filled_env: None) -> None:
    secrets = load_secrets()
    buf = io.StringIO()
    obs_log.configure_logging(stream=buf, level=logging.DEBUG, secrets=secrets)
    try:
        log = get_logger("test")
        raw = secrets.anthropic_api_key.get_secret_value()
        log.info("calling api", api_key=raw)
        rec = json.loads(buf.getvalue().strip().splitlines()[-1])
        assert raw not in buf.getvalue()
        assert rec["api_key"] == "[REDACTED]"
    finally:
        obs_log.reset_logging()


def test_secret_in_event_message_is_redacted(
    monkeypatch: pytest.MonkeyPatch, filled_env: None
) -> None:
    secrets = load_secrets()
    buf = io.StringIO()
    obs_log.configure_logging(stream=buf, level=logging.DEBUG, secrets=secrets)
    try:
        log = get_logger("test")
        raw = secrets.alpaca_api_secret_key.get_secret_value()
        log.info(f"using secret={raw} now")
        full = buf.getvalue()
        assert raw not in full
        assert "[REDACTED]" in full
    finally:
        obs_log.reset_logging()


def test_redaction_walks_nested_dicts_and_lists(
    monkeypatch: pytest.MonkeyPatch, filled_env: None
) -> None:
    secrets = load_secrets()
    buf = io.StringIO()
    obs_log.configure_logging(stream=buf, level=logging.DEBUG, secrets=secrets)
    try:
        log = get_logger("test")
        raw = secrets.telegram_bot_token.get_secret_value()
        log.info(
            "deep",
            payload={"headers": {"Authorization": f"Bearer {raw}"}},
            tokens=[raw, "safe"],
        )
        full = buf.getvalue()
        assert raw not in full
    finally:
        obs_log.reset_logging()


# ---------------------------------------------------------------------------
# H-1 fixes (reviewer-e8 retry 1/2): redaction must cover bytes,
# BaseException, and traceback strings produced by `format_exc_info`.
# ---------------------------------------------------------------------------


def test_secret_in_exception_arg_is_redacted(
    monkeypatch: pytest.MonkeyPatch, filled_env: None
) -> None:
    """`log.info("e", err=BoomError(f"401: {anthropic_key}"))` must NOT echo
    the key into the JSON output."""
    secrets = load_secrets()
    buf = io.StringIO()
    obs_log.configure_logging(stream=buf, level=logging.DEBUG, secrets=secrets)
    try:
        log = get_logger("test")
        raw = secrets.anthropic_api_key.get_secret_value()
        err = RuntimeError(f"401 unauthorized: api_key={raw}")
        log.info("upstream failed", err=err)
        full = buf.getvalue()
        assert raw not in full, (
            f"raw secret leaked when passed as exception arg; output was: {full}"
        )
        assert "[REDACTED]" in full
    finally:
        obs_log.reset_logging()


def test_secret_in_traceback_is_redacted(
    monkeypatch: pytest.MonkeyPatch, filled_env: None
) -> None:
    """When an exception is raised whose message contains a secret and
    `log.exception(...)` is called, the formatted traceback string must
    NOT echo the secret. This is the canonical H-1 leak path: an Anthropic
    error whose .args carries the request body / headers."""
    secrets = load_secrets()
    buf = io.StringIO()
    obs_log.configure_logging(stream=buf, level=logging.DEBUG, secrets=secrets)
    try:
        log = get_logger("test")
        raw = secrets.alpaca_api_secret_key.get_secret_value()
        try:
            raise RuntimeError(f"alpaca call failed with secret_key={raw}")
        except RuntimeError:
            log.exception("upstream broke")
        full = buf.getvalue()
        assert raw not in full, (
            "raw secret leaked into the formatted traceback; "
            f"output (truncated): {full[:1500]}"
        )
        assert "[REDACTED]" in full
    finally:
        obs_log.reset_logging()


def test_secret_in_bytes_payload_is_redacted(
    monkeypatch: pytest.MonkeyPatch, filled_env: None
) -> None:
    """A raw `bytes` value containing a secret (e.g., a captured HTTP body)
    must be redacted before serialization."""
    secrets = load_secrets()
    buf = io.StringIO()
    obs_log.configure_logging(stream=buf, level=logging.DEBUG, secrets=secrets)
    try:
        log = get_logger("test")
        raw = secrets.kill_switch_hmac_key.get_secret_value()
        log.info("body capture", body=f"key={raw}".encode())
        full = buf.getvalue()
        assert raw not in full, (
            f"raw secret leaked when passed as bytes; output was: {full}"
        )
    finally:
        obs_log.reset_logging()


# ---------------------------------------------------------------------------
# AC-5 — non-blocking on sink outage
# ---------------------------------------------------------------------------


def test_logger_does_not_block_on_sink_outage() -> None:
    """A handler that hangs in emit must not block the calling code path.

    Vector handles off-box delivery + replay; the application must always
    return immediately from `log.info(...)`. We model "sink outage" with a
    handler whose `emit` blocks until released, attach it via
    `attach_offbox_handler` (which wires it behind a non-blocking
    QueueHandler), then assert the application call returns promptly even
    while the downstream is hung.
    """
    release = threading.Event()

    class HangingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Block until the test releases us — proves the listener
            # thread is busy, but the calling thread is not.
            release.wait(timeout=10)

    buf = io.StringIO()
    obs_log.configure_logging(stream=buf, level=logging.DEBUG, secrets=None)
    try:
        hanging = HangingHandler()
        obs_log.attach_offbox_handler(hanging)

        log = get_logger("test")
        start = time.monotonic()
        log.info("should return fast")
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"log call blocked for {elapsed}s"
    finally:
        # Release the listener thread before tearing down so logging.shutdown
        # (atexit) doesn't deadlock waiting on the handler's lock.
        release.set()
        obs_log.reset_logging()


# ---------------------------------------------------------------------------
# AC-6 — log_port shim is backed by structlog (compat surface preserved)
# ---------------------------------------------------------------------------


def test_log_port_returns_structlog_backed_logger(captured_stream: io.StringIO) -> None:
    """`from observability.log_port import get_logger` continues to work and
    now produces JSON output with kwargs supported."""
    log = get_logger("compat.callsite")
    log.warning("legacy message %s", "with arg")
    log.info("kwargs message", k="v")

    lines = captured_stream.getvalue().strip().splitlines()
    assert len(lines) == 2
    rec_legacy = json.loads(lines[0])
    rec_kwargs = json.loads(lines[1])
    assert rec_legacy["level"] == "warning"
    assert "legacy message with arg" == rec_legacy["event"]
    assert rec_kwargs["k"] == "v"


def test_log_port_logger_supports_extra_kwarg(captured_stream: io.StringIO) -> None:
    """`log.debug("msg", extra={...})` was the stdlib-shim style — still works."""
    log = get_logger("compat.extra")
    log.debug("msg", extra={"filing_id": 42})
    rec = json.loads(captured_stream.getvalue().strip().splitlines()[-1])
    assert rec["filing_id"] == 42


def test_log_port_logger_exception_emits_traceback(captured_stream: io.StringIO) -> None:
    log = get_logger("compat.exc")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("caught")

    rec = json.loads(captured_stream.getvalue().strip().splitlines()[-1])
    assert rec["event"] == "caught"
    assert rec["level"] == "error"
    # structlog's exc_info processor injects an "exception" key with the traceback text
    assert "RuntimeError: boom" in rec.get("exception", "")
