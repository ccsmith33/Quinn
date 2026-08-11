"""S5.6 AC-6 — SIGTERM / SIGINT handling.

Installed once at process boot. Both signals request a graceful
shutdown and return. They do NOT call `sys.exit` or `loop.stop()` —
the consumer task observes the shutdown flag at well-defined yield
points so the in-flight pipeline always finishes cleanly (no torn
writes, no orphan proposals without their executions row).

The handler prefers `loop_obj.request_shutdown()` (flag + a `None`
sentinel on the ingestion queue) so an IDLE consumer blocked on
`queue.get()` wakes immediately instead of hanging until the next
filing arrives (systemd would otherwise escalate SIGTERM → SIGKILL
after its stop timeout). Objects without that method — e.g. test
stubs — get the legacy flag-only behavior.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any

from observability.log_port import get_logger

log = get_logger(__name__)


def install_shutdown_signal_handlers(loop_obj: Any) -> None:
    """Wire SIGTERM and SIGINT to `loop_obj.request_shutdown()` (flag +
    queue sentinel), falling back to flipping
    `loop_obj.shutdown_requested` on objects without that method.

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
        request = getattr(loop_obj, "request_shutdown", None)
        if request is None:
            # Legacy / stub loop objects without the sentinel wake:
            # fall back to flag-only, exactly the pre-fix behavior.
            loop_obj.shutdown_requested = True
            return
        request()

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
