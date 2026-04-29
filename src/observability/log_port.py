"""Logging port — stable import surface, structlog-backed implementation.

S1.3 introduced this shim around stdlib logging so downstream code could
adopt the import surface (`from observability.log_port import get_logger`)
ahead of the real structured-logging implementation. S8.1 swaps the
backing to `structlog` (see `observability.log` for configuration).

The import surface — `get_logger(name) -> Logger` — is unchanged, so
every existing callsite keeps working without edits. Returned loggers
are `structlog.stdlib.BoundLogger` instances which support both
kwarg-style (`log.info("event", k=v)`) and stdlib-style
(`log.warning("rss: %s", err)`) calls.
"""

from __future__ import annotations

from observability.log import BoundLogger as Logger
from observability.log import get_logger

__all__ = ["Logger", "get_logger"]
