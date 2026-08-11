"""SIGTERM/SIGINT handler dispatch — flag + sentinel wake (restart-hang fix).

The handler must call `loop_obj.request_shutdown()` when the target
exposes it (production `AgentLoop`: sets the flag AND enqueues the
`None` sentinel that wakes an idle consumer), and must fall back to
flag-only on objects without the method (legacy test stubs), so the
pre-fix contract is preserved for them.
"""

from __future__ import annotations

import asyncio
import os
import signal

import pytest

from app.signals import install_shutdown_signal_handlers


class _StubWithRequestShutdown:
    def __init__(self) -> None:
        self.shutdown_requested = False
        self.request_shutdown_calls = 0

    def request_shutdown(self) -> None:
        self.request_shutdown_calls += 1
        self.shutdown_requested = True


class _FlagOnlyStub:
    """Mimics legacy loop stubs: has the flag, lacks request_shutdown."""

    def __init__(self) -> None:
        self.shutdown_requested = False


def _restore(originals: dict[int, object]) -> None:
    for sig, prev in originals.items():
        signal.signal(sig, prev)  # type: ignore[arg-type]


def test_sigterm_calls_request_shutdown_when_available() -> None:
    originals = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }
    stub = _StubWithRequestShutdown()
    try:
        # No running event loop here → the signal.signal fallback path.
        install_shutdown_signal_handlers(stub)
        os.kill(os.getpid(), signal.SIGTERM)
        # CPython delivers the Python-level handler at the next bytecode
        # boundary — i.e. before the assertions below execute.
        assert stub.request_shutdown_calls == 1
        assert stub.shutdown_requested is True
    finally:
        _restore(originals)


def test_sigterm_falls_back_to_flag_only_on_stub() -> None:
    originals = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }
    stub = _FlagOnlyStub()
    try:
        install_shutdown_signal_handlers(stub)
        os.kill(os.getpid(), signal.SIGTERM)
        assert stub.shutdown_requested is True
    finally:
        _restore(originals)


@pytest.mark.asyncio
async def test_sigterm_event_loop_path_calls_request_shutdown() -> None:
    """The production path: handler installed via `add_signal_handler`
    runs on the event-loop thread and calls request_shutdown()."""
    stub = _StubWithRequestShutdown()
    running = asyncio.get_running_loop()
    install_shutdown_signal_handlers(stub)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(100):
            if stub.request_shutdown_calls:
                break
            await asyncio.sleep(0.01)
        assert stub.request_shutdown_calls == 1
        assert stub.shutdown_requested is True
    finally:
        running.remove_signal_handler(signal.SIGTERM)
        running.remove_signal_handler(signal.SIGINT)
