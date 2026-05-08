# PDT-SUNSET-2026-06-04: tests for ADR-009 activation gate + budget arithmetic.
"""S-PDT-2 — `src/execution/pdt_budget.py` unit tests.

References: story S-PDT-2 AC-9; pdt-budget-architecture.md §10.1.
"""

from __future__ import annotations

import datetime as dt
import logging

import pytest
from alpaca.common.exceptions import APIError

from broker.protocol import AccountSnapshot
from execution.pdt_budget import (
    PDTBudgetExceeded,
    PDTState,
    classify_pdt_403,
    compute_budget_remaining,
)


def _account(
    *, last_equity: float = 22_300.0, daytrade_count: int = 0,
    equity: float | None = None,
) -> AccountSnapshot:
    """Test helper — minimal AccountSnapshot fixture."""
    eq = equity if equity is not None else last_equity
    return AccountSnapshot(
        equity=eq,
        cash=eq,
        buying_power=eq * 2,
        long_market_value=0.0,
        daypl=0.0,
        snapshot_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
        last_equity=last_equity,
        daytrade_count=daytrade_count,
    )


# ---------------------------------------------------------------------------
# compute_budget_remaining (AC-9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("daytrade_count", [0, 1, 2, 3])
@pytest.mark.parametrize("pending", [0, 1, 2, 3])
def test_compute_budget_remaining_matrix(
    daytrade_count: int, pending: int
) -> None:
    """For (daytrade_count, pending) ∈ {0..3}², returns max(0, 3 - dc - p)."""
    acct = _account(daytrade_count=daytrade_count)
    expected = max(0, 3 - daytrade_count - pending)
    assert compute_budget_remaining(acct, pending) == expected


def test_compute_budget_remaining_negative_pending_raises() -> None:
    """ADR-009 §3.1 — pending must be non-negative."""
    acct = _account(daytrade_count=0)
    with pytest.raises(ValueError, match="non-negative"):
        compute_budget_remaining(acct, -1)


# ---------------------------------------------------------------------------
# PDTState.from_account (AC-9)
# ---------------------------------------------------------------------------


def test_pdt_state_from_account_below_threshold() -> None:
    """last_equity=24999 → active; 25000 → inactive (strict <); 25001 → inactive."""
    state_below = PDTState.from_account(
        _account(last_equity=24_999.0), pdt_enabled=True
    )
    assert state_below.is_active() is True

    state_at = PDTState.from_account(
        _account(last_equity=25_000.0), pdt_enabled=True
    )
    assert state_at.is_active() is False

    state_above = PDTState.from_account(
        _account(last_equity=25_001.0), pdt_enabled=True
    )
    assert state_above.is_active() is False


def test_pdt_state_intraday_equity_irrelevant() -> None:
    """`equity=22000, last_equity=27500` → inactive: last_equity wins."""
    state = PDTState.from_account(
        _account(last_equity=27_500.0, equity=22_000.0), pdt_enabled=True
    )
    assert state.is_active() is False


def test_pdt_state_pdt_enabled_false_short_circuits() -> None:
    """`last_equity=22000, pdt_enabled=False` → inactive (operator escape hatch)."""
    state = PDTState.from_account(
        _account(last_equity=22_000.0), pdt_enabled=False
    )
    assert state.is_active() is False


# ---------------------------------------------------------------------------
# PDTState.refresh (AC-9)
# ---------------------------------------------------------------------------


def test_pdt_state_refresh_flips_state(caplog: pytest.LogCaptureFixture) -> None:
    """Start active; refresh with last_equity=27000 → inactive; flip log."""
    state = PDTState.from_account(
        _account(last_equity=22_000.0), pdt_enabled=True
    )
    assert state.is_active() is True

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="execution.pdt_budget"):
        result = state.refresh(
            _account(last_equity=27_000.0), pdt_enabled=True
        )
    assert result is False
    assert state.is_active() is False
    flips = [
        r for r in caplog.records
        if getattr(r, "event", None) == "pdt.activation.changed"
    ]
    assert len(flips) == 1
    rec = flips[0]
    assert rec.was_active is True
    assert rec.now_active is False
    assert rec.last_equity == 27_000.0


def test_pdt_state_refresh_no_flip_emits_no_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Idle ticks (no flip) emit no log."""
    state = PDTState.from_account(
        _account(last_equity=22_000.0), pdt_enabled=True
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="execution.pdt_budget"):
        state.refresh(_account(last_equity=22_500.0), pdt_enabled=True)
    flips = [
        r for r in caplog.records
        if getattr(r, "event", None) == "pdt.activation.changed"
    ]
    assert flips == []


def test_pdt_state_refresh_flip_to_active_emits_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Crossing threshold the other way also fires the changed log."""
    state = PDTState.from_account(
        _account(last_equity=27_000.0), pdt_enabled=True
    )
    assert state.is_active() is False
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="execution.pdt_budget"):
        state.refresh(_account(last_equity=22_000.0), pdt_enabled=True)
    flips = [
        r for r in caplog.records
        if getattr(r, "event", None) == "pdt.activation.changed"
    ]
    assert len(flips) == 1
    assert flips[0].was_active is False
    assert flips[0].now_active is True


# ---------------------------------------------------------------------------
# classify_pdt_403 (AC-9, ADR-009 §5.1, §5.3)
# ---------------------------------------------------------------------------


class _StubAPIError(APIError):
    """APIError doesn't have a public no-arg constructor in alpaca-py
    — its __init__ takes a Response. For unit tests we bypass __init__
    and set attributes directly. This stand-in keeps `isinstance` true.
    """

    def __init__(
        self, *, status_code: int | None = None, code: int | None = None,
        message: str = "",
    ) -> None:
        # Skip APIError.__init__ — we only need the attributes
        # classify_pdt_403 inspects.
        Exception.__init__(self, message)
        self._status_code = status_code
        self._code = code

    @property  # type: ignore[override]
    def status_code(self) -> int | None:  # type: ignore[override]
        return self._status_code

    @property  # type: ignore[override]
    def code(self) -> int | None:  # type: ignore[override]
        return self._code


def test_classify_pdt_403_status_code_match() -> None:
    """APIError(status_code=403, code=40310100) → True."""
    exc = _StubAPIError(
        status_code=403, code=40310100, message="client cannot day-trade"
    )
    assert classify_pdt_403(exc) is True


def test_classify_pdt_403_message_match() -> None:
    """APIError(status_code=403, msg='...day-trade...') → True even if
    the numeric code is absent or different."""
    exc = _StubAPIError(
        status_code=403, code=99999999,
        message="client cannot day-trade more than 3 times in 5 days",
    )
    assert classify_pdt_403(exc) is True


def test_classify_pdt_403_message_match_alt_punctuation() -> None:
    """`day trade`, `day-trade`, `Day-Trade` all match."""
    for msg in ["day trade", "day-trade", "Day Trade", "DAY-TRADE"]:
        exc = _StubAPIError(status_code=403, message=msg)
        assert classify_pdt_403(exc) is True, f"failed for {msg!r}"


def test_classify_pdt_403_unrelated_403() -> None:
    """APIError(status_code=403, 'insufficient buying power') → False."""
    exc = _StubAPIError(
        status_code=403, code=40010001, message="insufficient buying power"
    )
    assert classify_pdt_403(exc) is False


def test_classify_pdt_403_non_403_status() -> None:
    """APIError with non-403 status_code → False even if message mentions day-trade."""
    exc = _StubAPIError(
        status_code=500, message="internal error involving day-trade tracking"
    )
    assert classify_pdt_403(exc) is False


def test_classify_pdt_403_non_api_error() -> None:
    """Raw Exception → False."""
    assert classify_pdt_403(Exception("client cannot day-trade")) is False
    assert classify_pdt_403(ValueError("403 day-trade")) is False


# ---------------------------------------------------------------------------
# PDTBudgetExceeded subclasses BrokerUnavailable (story dev-notes)
# ---------------------------------------------------------------------------


def test_pdt_budget_exceeded_subclasses_broker_unavailable() -> None:
    """Existing callers that catch `BrokerUnavailable` degrade gracefully."""
    from broker.alpaca import BrokerUnavailable

    assert issubclass(PDTBudgetExceeded, BrokerUnavailable)
    exc = PDTBudgetExceeded("test")
    assert isinstance(exc, BrokerUnavailable)
