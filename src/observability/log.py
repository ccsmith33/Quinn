"""Structured logging configuration (NFR-9, FR-33, NFR-15, NFR-17).

Architecture references: §10.1 Observability, §9.4 Secrets.

This module is the single point of truth for log configuration. The thin
shim in `log_port.py` (introduced in S1.3) delegates to `get_logger()`
here so every existing callsite — `from observability.log_port import
get_logger` — gains structured JSON output without touching its imports.

Design:

- `structlog` with stdlib integration: `structlog.stdlib.get_logger()`
  returns a BoundLogger that supports both kwargs (`log.info("event",
  k=v)`) and stdlib percent-format (`log.warning("rss: %s", err)`).
- A `correlation_id` `contextvars.ContextVar` is bound onto every event
  via `structlog.contextvars.merge_contextvars`; agent-loop iterations
  enter `correlation_id_scope(...)` to bind it for the duration.
- Default fields (`service`, `pid`, `host`) are added by a small
  processor.
- Secrets are scrubbed by `make_redact_processor(secrets)` which walks
  the event dict (str/dict/list/tuple) and applies `config.secrets.redact`
  to any string. This is defense in depth — the SecretStr wrappers
  already prevent accidental `repr`/`str` leaks at the model boundary.
- Off-box delivery is Vector's job (see `ops/vector/quinn.toml`); the
  application logs to stdout/journald and never blocks on sink delivery.
  When an extra (e.g. test) handler is attached via
  `attach_offbox_handler(...)`, it is wrapped in a non-blocking
  `QueueHandler` + `QueueListener` pair so a slow downstream cannot
  stall the agent loop (NFR-9 + AC-5).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import queue
import socket
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TextIO

import structlog
from structlog.stdlib import BoundLogger as _StructBoundLogger
from structlog.types import EventDict, Processor

from config.secrets import Secrets, redact

_correlation_id: ContextVar[str | None] = ContextVar("quinn_correlation_id", default=None)

_listener: logging.handlers.QueueListener | None = None
_queue_handler: logging.handlers.QueueHandler | None = None
_DEFAULT_SERVICE = "quinn"


class BoundLogger(_StructBoundLogger):
    """structlog stdlib BoundLogger with stdlib `extra=` propagation.

    Pre-S8.1 callsites use `log.info("msg", extra={"k": v})` — the stdlib
    idiom that sets attributes on the LogRecord. structlog's BoundLogger
    treats `extra` as just another kwarg, so the LogRecord never carried
    those attrs and tests that did `caplog.records[0].k` lost data.

    This subclass strips `extra` before processing, lets the structlog
    processor chain render the rest of the event dict, then re-injects
    `extra=` on the final stdlib call so the LogRecord has its attrs and
    JSON output also contains the keys (via _flatten_extra processor on
    the rendering side).
    """

    def _proxy_to_logger(
        self,
        method_name: str,
        event: str | None = None,
        *event_args: str,
        **event_kw: Any,
    ) -> Any:
        # Pull stdlib-style extra out so it doesn't get rendered as a
        # nested dict in the JSON. Keep its keys in the structlog
        # event_dict too (via _flatten_extra) so they appear at top level.
        extra: dict[str, Any] | None = None
        if "extra" in event_kw and isinstance(event_kw["extra"], dict):
            extra = event_kw["extra"]
            # Mirror into kwargs at top level so structlog renderers see them.
            for k, v in extra.items():
                event_kw.setdefault(k, v)
            # Pop so structlog doesn't also emit it as a nested key.
            event_kw.pop("extra")

        if event_args:
            event_kw["positional_args"] = event_args

        try:
            args, kw = self._process_event(method_name, event, event_kw)
        except structlog.DropEvent:
            return None

        # Re-inject extra into the stdlib call so LogRecord attrs are set
        # — preserves the pre-S8.1 idiom for caplog-based tests.
        if extra:
            kw_dict: dict[str, Any] = dict(kw)
            stdlib_extra: dict[str, Any] = dict(kw_dict.get("extra") or {})
            for k, v in extra.items():
                if k not in ("message", "asctime"):  # reserved by stdlib
                    stdlib_extra.setdefault(k, v)
            kw_dict["extra"] = stdlib_extra
            kw = kw_dict

        return getattr(self._logger, method_name)(*args, **kw)


class _DaemonQueueListener(logging.handlers.QueueListener):
    """QueueListener whose worker thread is a daemon.

    Daemon ensures process exit cannot be blocked by a hung downstream
    handler — critical for AC-5 (sink outage must not stall the app).
    """

    def start(self) -> None:
        # `_monitor` is the private monitor loop on the parent class.
        monitor = self._monitor  # type: ignore[attr-defined]
        self._thread = t = threading.Thread(target=monitor, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


def _add_default_fields(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    event_dict.setdefault("service", _DEFAULT_SERVICE)
    event_dict.setdefault("pid", os.getpid())
    event_dict.setdefault("host", socket.gethostname())
    return event_dict


def _add_correlation_id(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    cid = _correlation_id.get()
    if cid is not None:
        event_dict["correlation_id"] = cid
    return event_dict


def _rename_logger_to_module(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    """structlog's stdlib integration emits `logger=<name>`; our schema names
    that field `module` per architecture §10.1."""
    if "logger" in event_dict and "module" not in event_dict:
        event_dict["module"] = event_dict.pop("logger")
    return event_dict


def _flatten_extra(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    """Backward-compat with stdlib `log.debug("msg", extra={"k": v})` callsites.

    structlog passes kwargs straight through, so `extra={...}` would land as
    a nested key. Pre-S8.1 callsites (S1.3 stdlib shim) used `extra={...}`
    expecting flattening. Hoist its keys into the event dict so those
    callsites continue to surface fields at the top level.
    """
    extra = event_dict.pop("extra", None)
    if isinstance(extra, dict):
        for k, v in extra.items():
            event_dict.setdefault(k, v)
    elif extra is not None:
        # Non-dict extra: keep as-is rather than dropping.
        event_dict["extra"] = extra
    return event_dict


def make_redact_processor(secrets: Secrets | None) -> Processor:
    """Return a structlog processor that scrubs known secret substrings.

    When `secrets` is None (e.g. in unit tests that don't load env), the
    processor is a no-op pass-through so tests don't need to provide a
    full Secrets fixture.
    """
    if secrets is None:

        def _noop(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
            return event_dict

        return _noop

    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            return redact(value, secrets)
        if isinstance(value, bytes):
            # Decode best-effort, redact, re-encode. `errors="replace"` keeps
            # binary blobs from raising — the goal is "no secret leaks", not
            # round-trip fidelity. The output stays bytes so JSON renders it
            # consistently; structlog's JSONRenderer handles bytes via
            # `default=str` which then yields the redacted string.
            try:
                decoded = value.decode("utf-8", errors="replace")
            except Exception:
                return value
            return redact(decoded, secrets).encode("utf-8", errors="replace")
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_walk(v) for v in value)
        if isinstance(value, BaseException):
            # Exception objects render via str() in JSON output; redact the
            # rendered form. Wrapping in a placeholder object keeps the
            # downstream format_exc_info processor happy if this exception
            # is also referenced by exc_info — that path is covered by
            # running redaction AFTER format_exc_info has expanded the
            # traceback into a `exception` string field.
            return redact(str(value), secrets)
        return value

    def _redact(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
        for key in list(event_dict.keys()):
            event_dict[key] = _walk(event_dict[key])
        return event_dict

    return _redact


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure_logging(
    *,
    stream: TextIO | None = None,
    level: int = logging.INFO,
    secrets: Secrets | None = None,
) -> None:
    """Idempotently configure structlog + the stdlib root logger.

    `stream` defaults to stdout (journald captures stdout under systemd).
    `secrets` enables the redaction processor; pass `None` in tests that
    don't need it.
    """
    global _listener, _queue_handler

    # Tear down any previous configuration so repeat calls (tests) work.
    reset_logging()

    target_stream = stream if stream is not None else sys.stdout

    handler = logging.StreamHandler(target_stream)
    handler.setLevel(level)
    # The structlog ProcessorFormatter renders the event dict to JSON.
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            _rename_logger_to_module,
            _flatten_extra,
            _add_default_fields,
            _add_correlation_id,
            # `format_exc_info` MUST run BEFORE the redaction processor so
            # that exception tracebacks (which can echo upstream API error
            # messages containing secret fragments) are scrubbed before
            # serialization (H-1 fix).
            structlog.processors.format_exc_info,
            make_redact_processor(secrets),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ],
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            _rename_logger_to_module,
            _flatten_extra,
            _add_default_fields,
            _add_correlation_id,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            # `format_exc_info` MUST run BEFORE the redaction processor so
            # that exception tracebacks (which can echo upstream API error
            # messages containing secret fragments) are scrubbed before
            # serialization (H-1 fix).
            structlog.processors.format_exc_info,
            make_redact_processor(secrets),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def reset_logging() -> None:
    """Restore the stdlib logging root + tear down any queue listener.

    Used by tests for isolation; safe to call when nothing is configured.
    Listener teardown is best-effort and non-blocking — if a downstream
    handler is hung in `emit`, we detach the queue handler and abandon
    the listener thread rather than wait. This is the right semantics
    for AC-5: a slow sink must not stall callers, including teardown.

    structlog itself is re-configured to our default wrapper so
    callsites that don't go through `configure_logging` (other test
    files in the suite) still see `BoundLogger` semantics — including
    stdlib `extra=` propagation onto the LogRecord.
    """
    global _listener, _queue_handler

    root = logging.getLogger()

    if _queue_handler is not None and _queue_handler in root.handlers:
        root.removeHandler(_queue_handler)
    _queue_handler = None

    if _listener is not None:
        # Signal stop without joining the worker thread. If the worker is
        # blocked inside an emit() call, joining would deadlock the caller.
        listener = _listener
        _listener = None
        try:
            listener.queue.put_nowait(None)  # _sentinel for the listener loop
        except Exception:
            pass

    root.handlers.clear()
    _install_default_wrapper()


def attach_offbox_handler(handler: logging.Handler) -> None:
    """Attach an additional handler (e.g. a syslog/HTTP sink for direct
    off-box shipment) without blocking application code on its delivery.

    The handler is wired behind a `QueueHandler` + `QueueListener` so the
    application thread never blocks on `emit` — the listener thread drains
    the queue. This is AC-5: even if the sink hangs, `log.info(...)` returns
    immediately. (In production, Vector handles off-box delivery; this hook
    exists for tests and for any direct-ship adapter.)
    """
    global _listener, _queue_handler

    q: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    qh = logging.handlers.QueueHandler(q)
    listener = _DaemonQueueListener(q, handler, respect_handler_level=False)
    listener.start()

    root = logging.getLogger()
    root.addHandler(qh)
    _queue_handler = qh
    _listener = listener


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


@contextmanager
def correlation_id_scope(cid: str) -> Iterator[None]:
    """Bind `correlation_id=cid` on every log event emitted while in scope.

    The agent-loop iteration entrypoint should wrap each iteration in this
    scope (S5.6 — currently long-tail blocked).
    """
    token = _correlation_id.set(cid)
    try:
        yield
    finally:
        _correlation_id.reset(token)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------


def get_logger(name: str) -> BoundLogger:
    """Return a structlog `BoundLogger` for `name`.

    Backed by `structlog.stdlib`, so callers can use either:
      - kwargs:        `log.info("event", k=v)`
      - stdlib-style:  `log.warning("rss: %s", err)`
      - extra=...:     `log.debug("msg", extra={"k": v})`

    All three styles end up emitting a structured JSON event with the
    extra keys flattened to the top level. `extra=` is also propagated
    to the stdlib LogRecord so `caplog.records[i].<key>` continues to
    work for tests that introspect via the stdlib logging API.
    """
    return structlog.get_logger(name)


__all__ = [
    "BoundLogger",
    "attach_offbox_handler",
    "configure_logging",
    "correlation_id_scope",
    "get_logger",
    "make_redact_processor",
    "reset_logging",
]


def _install_default_wrapper() -> None:
    """Install our `BoundLogger` so `get_logger` returns it even before
    `configure_logging` runs.

    This matters for tests that construct a logger without going through
    `configure_logging` — they still get the stdlib `extra=` propagation
    that pre-S8.1 callsites depend on.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            _rename_logger_to_module,
            _flatten_extra,
            _add_default_fields,
            _add_correlation_id,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


_install_default_wrapper()
