"""S5.6 AC-6 — SIGTERM / SIGINT handling.

Installed once at process boot. Both signals flip the loop's
`shutdown_requested` flag and return. They do NOT call `sys.exit` or
`loop.stop()` — the consumer task observes the flag at well-defined
yield points so the in-flight pipeline always finishes cleanly (no
torn writes, no orphan proposals without their executions row).
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any

from observability.log_port import get_logger

log = get_logger(__name__)


def install_shutdown_signal_handlers(loop_obj: Any) -> None:
    """Wire SIGTERM and SIGINT to flip `loop_obj.shutdown_requested`.

    The handler runs in the asyncio event loop where possible (so the
    flag flip is observed promptly by the consumer). When asyncio's
    `add_signal_handler` is unavailable (e.g., Windows), falls back to
    `signal.signal` — sufficient for tests and OS-level shutdown.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    def _handle(sig_name: str) -> None:
        log.warning(
            "shutdown signal received",
            extra={"event": "shutdown_signal_received", "signal": sig_name},
        )
        loop_obj.shutdown_requested = True

    if running is not None:
        for sig, name in (
            (signal.SIGTERM, "SIGTERM"),
            (signal.SIGINT, "SIGINT"),
        ):
            try:
                running.add_signal_handler(sig, _handle, name)
            except (NotImplementedError, ValueError):
                # `add_signal_handler` raises NotImplementedError on
                # platforms without it (Windows). ValueError fires when
                # called from a non-main thread; both fall back below.
                signal.signal(sig, lambda *_a, _name=name: _handle(_name))
        return

    for sig, name in (
        (signal.SIGTERM, "SIGTERM"),
        (signal.SIGINT, "SIGINT"),
    ):
        signal.signal(sig, lambda *_a, _name=name: _handle(_name))


__all__ = ["install_shutdown_signal_handlers"]
