"""Shared EDGAR HTTP client (S3.1, ADR-002 §6).

A single process-wide async client used by RSS discovery (S3.2), the detail
fetcher (S3.3), and the submissions-API reconciler (S3.4). Enforces SEC
fair-access policy (FR-6, NFR-17): 10 req/sec global ceiling, declared
User-Agent, exponential backoff with full jitter on 429/5xx, max 5 attempts,
then a circuit breaker (NFR-5) so a sustained outage does not stall the
agent loop.

Tests inject `httpx.MockTransport` via the `transport` constructor argument;
production callers leave it `None` and a real `httpx.AsyncClient` is built.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import httpx

EventCallback = Callable[[str, Mapping[str, Any]], None]

_RATE_LIMIT_PER_SEC = 10
_MIN_INTERVAL_SECONDS = 1.0 / _RATE_LIMIT_PER_SEC  # 0.1s — see ADR-002 §6
_DEFAULT_TIMEOUT_SECONDS = 15.0
_MAX_ATTEMPTS = 5
_DEFAULT_RETRY_BASE_SECONDS = 1.0
_DEFAULT_RETRY_CAP_SECONDS = 60.0
_DEFAULT_CIRCUIT_THRESHOLD = 3
_DEFAULT_CIRCUIT_WINDOW_SECONDS = 300.0  # 5 min
_DEFAULT_CIRCUIT_COOLDOWN_SECONDS = 300.0  # 5 min


class EdgarUnavailable(Exception):
    """Raised when retries are exhausted on 429/5xx responses or transport errors."""


class CircuitOpen(Exception):
    """Raised when the circuit is open and the call is short-circuited."""


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


class _RateLimiter:
    """Spacing-based limiter: every request start is at least 1/N seconds
    after the previous one. Equivalent to a 1-token bucket refilling at N/s,
    which prevents an initial burst above the ceiling.

    Using an `asyncio.Lock` makes this safe across concurrent coroutines
    sharing one client instance (ADR-002 §6: one shared client per process).
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


class _CircuitBreaker:
    """Threshold-in-window breaker.

    States:
      - closed: normal operation; failures recorded; if N failures land in
        the rolling window, transition to open.
      - open: every call short-circuits with `CircuitOpen` until the cooldown
        elapses; after that, the next call is admitted as a half-open probe.
      - half-open: the probe call runs through normally; on success → closed,
        on failure → open again with a fresh cooldown.

    Times are taken from `time.monotonic()` so wall-clock changes don't
    perturb behavior. `notify` is called with state transitions so the
    enclosing client can emit telemetry (AC-6).
    """

    _STATE_CLOSED = "closed"
    _STATE_OPEN = "open"
    _STATE_HALF_OPEN = "half_open"

    def __init__(
        self,
        threshold: int,
        window_seconds: float,
        cooldown_seconds: float,
        notify: Callable[[str, Mapping[str, Any]], None],
    ) -> None:
        self._threshold = threshold
        self._window = window_seconds
        self._cooldown = cooldown_seconds
        self._notify = notify
        self._failures: deque[float] = deque()
        self._state = self._STATE_CLOSED
        self._reopen_at = 0.0

    def before_call(self) -> None:
        now = time.monotonic()
        if self._state == self._STATE_OPEN:
            if now >= self._reopen_at:
                self._state = self._STATE_HALF_OPEN
            else:
                raise CircuitOpen(
                    f"edgar circuit open; retry after "
                    f"{self._reopen_at - now:.1f}s"
                )

    def record_success(self) -> None:
        if self._state in (self._STATE_HALF_OPEN, self._STATE_OPEN):
            self._state = self._STATE_CLOSED
            self._failures.clear()
            self._notify("circuit_close", {})
        # Closed-state successes don't clear the rolling window — failures
        # outside the window age out on their own in `record_failure`.

    def record_failure(self) -> None:
        now = time.monotonic()
        # Drop expired failures so the window is rolling, not cumulative.
        cutoff = now - self._window
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()
        self._failures.append(now)
        if (
            self._state in (self._STATE_CLOSED, self._STATE_HALF_OPEN)
            and len(self._failures) >= self._threshold
        ):
            self._state = self._STATE_OPEN
            self._reopen_at = now + self._cooldown
            self._notify(
                "circuit_open",
                {"failures": len(self._failures), "cooldown_s": self._cooldown},
            )


class EdgarClient:
    """Shared, rate-limited async HTTP client for SEC EDGAR endpoints."""

    def __init__(
        self,
        *,
        user_agent: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        retry_base_seconds: float = _DEFAULT_RETRY_BASE_SECONDS,
        retry_cap_seconds: float = _DEFAULT_RETRY_CAP_SECONDS,
        max_attempts: int = _MAX_ATTEMPTS,
        circuit_threshold: int = _DEFAULT_CIRCUIT_THRESHOLD,
        circuit_window_seconds: float = _DEFAULT_CIRCUIT_WINDOW_SECONDS,
        circuit_cooldown_seconds: float = _DEFAULT_CIRCUIT_COOLDOWN_SECONDS,
        on_event: EventCallback | None = None,
    ) -> None:
        if not user_agent or not isinstance(user_agent, str):
            raise ValueError(
                "EdgarClient requires a non-empty user_agent "
                "(SEC fair-access policy / FR-6)"
            )
        self._user_agent = user_agent
        self._on_event = on_event or (lambda _name, _ctx: None)
        self._retry_base = retry_base_seconds
        self._retry_cap = retry_cap_seconds
        self._max_attempts = max_attempts
        self._limiter = _RateLimiter(_MIN_INTERVAL_SECONDS)
        self._breaker = _CircuitBreaker(
            threshold=circuit_threshold,
            window_seconds=circuit_window_seconds,
            cooldown_seconds=circuit_cooldown_seconds,
            notify=self._on_event,
        )
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip"},
        )
        self.retry_count = 0  # cumulative retry attempts; surfaced for AC-6 / S8.1

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> EdgarClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def get(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        return await self._request("GET", url, params=params)

    async def get_bytes(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
    ) -> bytes:
        resp = await self._request("GET", url, params=params)
        return resp.content

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None,
    ) -> httpx.Response:
        self._breaker.before_call()
        try:
            resp = await self._send_with_retries(method, url, params=params)
        except EdgarUnavailable:
            self._breaker.record_failure()
            raise
        self._breaker.record_success()
        return resp

    async def _send_with_retries(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None,
    ) -> httpx.Response:
        last_status: int | None = None
        last_exc: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            await self._limiter.acquire()
            try:
                resp = await self._client.request(method, url, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                last_status = None
            else:
                if not _is_retryable_status(resp.status_code):
                    return resp
                last_status = resp.status_code
                last_exc = None
            if attempt >= self._max_attempts:
                break
            self.retry_count += 1
            self._on_event(
                "retry",
                {
                    "attempt": attempt,
                    "url": url,
                    "status": last_status,
                    "error": type(last_exc).__name__ if last_exc else None,
                },
            )
            await asyncio.sleep(self._backoff_seconds(attempt))
        raise EdgarUnavailable(
            f"EDGAR call failed after {self._max_attempts} attempts: "
            f"url={url} last_status={last_status} last_error={last_exc!r}"
        )

    def _backoff_seconds(self, attempt: int) -> float:
        # Full-jitter exponential backoff (AWS Architecture Blog formulation):
        # sleep = uniform(0, min(cap, base * 2 ** (attempt - 1))).
        if self._retry_base <= 0.0:
            return 0.0
        ceiling = min(self._retry_cap, self._retry_base * (2 ** (attempt - 1)))
        return random.uniform(0.0, ceiling)
