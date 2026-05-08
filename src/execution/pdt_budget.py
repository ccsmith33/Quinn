# PDT-SUNSET-2026-06-04: entire module is part of ADR-009. Delete on
# FINRA PDT-rule retirement (see ADR-009 §"Sunset / deprecation plan").
"""PDT day-trade budget — activation gate, budget arithmetic, 403 classifier.

ADR-009 §3.1 / pdt-budget-architecture.md §3.1.

`PDTState` is process-shared (one instance per agent). It is mutated by
the reconciler tick (`refresh()`) and read by consumers via
`is_active()`. Single-attribute reads are atomic on CPython (same
threadsafety pattern as `KillSwitchState`); no lock is required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from alpaca.common.exceptions import APIError

from broker.alpaca import BrokerUnavailable
from broker.protocol import AccountSnapshot
from observability.log_port import get_logger

log = get_logger(__name__)

_PDT_EQUITY_THRESHOLD = 25_000.0
_PDT_REJECTION_CODE = 40310100  # Alpaca's documented PDT rejection code.
_DAY_TRADE_RE = re.compile(r"day[- ]?trade", re.IGNORECASE)


@dataclass
class PDTState:
    """Process-shared activation flag holder.

    `is_active()` returns the flag atomically (single-attribute read).
    Mutated by the reconciler tick via `refresh()` after a fresh
    `AccountSnapshot` has been fetched.
    """

    _active: bool
    _last_equity: float

    @classmethod
    def from_account(
        cls, account: AccountSnapshot, *, pdt_enabled: bool
    ) -> PDTState:
        """Construct at boot from the first `broker.get_account()` snapshot.

        `pdt_enabled=False` short-circuits the gate to inactive
        regardless of `last_equity` (operator escape hatch per
        ADR-009 §8).
        """
        active = pdt_enabled and account.last_equity < _PDT_EQUITY_THRESHOLD
        return cls(_active=active, _last_equity=account.last_equity)

    def is_active(self) -> bool:
        return self._active

    def refresh(
        self, account: AccountSnapshot, *, pdt_enabled: bool
    ) -> bool:
        """Re-evaluate from a fresh `AccountSnapshot`. Returns the new state.

        Called from the reconciler tick after the broker `get_account()`
        result is in hand. Emits `pdt.activation.changed` at INFO when
        the flag flips. No-flip ticks emit no log.
        """
        was_active = self._active
        new_active = pdt_enabled and account.last_equity < _PDT_EQUITY_THRESHOLD
        self._active = new_active
        self._last_equity = account.last_equity
        if was_active != new_active:
            log.info(
                "pdt.activation.changed",
                extra={
                    "event": "pdt.activation.changed",
                    "was_active": was_active,
                    "now_active": new_active,
                    "last_equity": account.last_equity,
                },
            )
        return new_active


def compute_budget_remaining(account: AccountSnapshot, pending: int) -> int:
    """Return the day-trade budget remaining for the current rolling window.

    `max(0, 3 - account.daytrade_count - pending)`. `pending` is the
    count of in-flight sells the scanner has submitted in the current
    session that have not yet been observed in `account.daytrade_count`
    (counted locally to predict 403s before Alpaca's count catches up).
    """
    if pending < 0:
        raise ValueError(f"pending must be non-negative, got {pending}")
    return max(0, 3 - account.daytrade_count - pending)


class PDTBudgetExceeded(BrokerUnavailable):
    """Raised by `AlpacaBroker.submit_order` when a 403 day-trade error fires.

    Subclasses `BrokerUnavailable` so existing callers degrade safely
    (the entry-leg failure path already writes `executions.decision=
    "submission_failed"`).
    """


def classify_pdt_403(exc: Exception) -> bool:
    """Return True iff `exc` is an Alpaca PDT-rejection.

    Per ADR-009 §5.1: ALL of (a) `isinstance(exc, APIError)`, (b)
    `status_code == 403`, (c) `code == 40310100` OR message matches
    `r"day[- ]?trade"` (case-insensitive). The substring check is the
    safety net — Alpaca has historically tweaked error codes; the
    human-readable message has been stable.
    """
    if not isinstance(exc, APIError):
        return False
    if getattr(exc, "status_code", None) != 403:
        return False
    if getattr(exc, "code", None) == _PDT_REJECTION_CODE:
        return True
    return bool(_DAY_TRADE_RE.search(str(exc)))


__all__ = [
    "PDTBudgetExceeded",
    "PDTState",
    "classify_pdt_403",
    "compute_budget_remaining",
]
