"""S6.3 — Sizing engine + KS-4..KS-7 pre-trade caps tests.

PRD §6.1 (sizing function) and §5.2 (KS-4..KS-7) specify the exact thresholds.
Pure logic — every collaborator (broker, universe, ks) is an injected value
or fake. No I/O.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from broker.protocol import AccountSnapshot, Position, Quote
from config.loader import ExecutionConfig
from execution.sizing import (
    SizingAccepted,
    SizingEngine,
    SizingRejected,
)
from proposal.schemas import TradeProposal

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _proposal(
    *,
    symbol: str = "ACME",
    size_pct: float = 0.10,
    conviction: int = 8,
) -> TradeProposal:
    payload: dict[str, Any] = {
        "symbol": symbol,
        "direction": "long",
        "size_pct_of_capital": size_pct,
        "entry_style": "market_open",
        "stop_loss_price": 4.50,
        "time_horizon_days": 10,
        "conviction": conviction,
        "thesis": (
            "Strong fundamentals with material 8-K disclosure indicating "
            "near-term catalyst per filing analysis."
        ),
        "signals": ["8-K item 2.02 strong earnings beat"],
        "exit_conditions": ["thesis breaks; stop hit; 30 days elapsed"],
        "risk_factors": ["macro risk; sector rotation; news risk"],
    }
    return TradeProposal.model_validate(payload)


def _account(*, equity: float, cash: float | None = None) -> AccountSnapshot:
    c = cash if cash is not None else equity
    return AccountSnapshot(
        equity=equity,
        cash=c,
        buying_power=c,
        long_market_value=equity - c,
        daypl=0.0,
        snapshot_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
    )


def _quote(*, last: float = 10.0) -> Quote:
    return Quote(
        symbol="ACME",
        bid=last - 0.02,
        ask=last + 0.02,
        last=last,
        ts=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
    )


def _position(symbol: str, qty: int = 10) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        avg_entry_price=10.0,
        market_value=10.0 * qty,
        unrealized_pnl=0.0,
    )


def _cfg(
    *,
    ks4_pct_cap: float = 0.20,
    ks4_absolute_cap_usd: float = 1000.0,
    ks5_max_concurrent: int = 5,
    ks7_cash_reserve_pct: float = 0.05,
    sizing_mid_pct: float = 0.05,
    sizing_high_pct: float = 0.10,
) -> ExecutionConfig:
    return ExecutionConfig(
        broker_mode="paper",
        ks4_pct_cap=ks4_pct_cap,
        ks4_absolute_cap_usd=ks4_absolute_cap_usd,
        ks5_max_concurrent=ks5_max_concurrent,
        ks7_cash_reserve_pct=ks7_cash_reserve_pct,
        sizing_mid_pct=sizing_mid_pct,
        sizing_high_pct=sizing_high_pct,
    )


# ---------------------------------------------------------------------------
# Tests (story §test plan)
# ---------------------------------------------------------------------------

def test_mid_conviction_5pct_sizing() -> None:
    """Test #1 (first failing): cv 8 → mid tier (5%); equity $5000 → $250."""
    p = _proposal(conviction=8, size_pct=0.10)  # request ignored when over tier
    account = _account(equity=5000.0)
    engine = SizingEngine()

    result = engine.size(p, account, open_positions=[], quote=_quote(last=10.0), cfg=_cfg())

    assert isinstance(result, SizingAccepted)
    assert result.realized_pct == pytest.approx(0.05)
    assert result.realized_dollar_size == pytest.approx(250.0)
    assert result.qty == 25  # floor(250 / 10)


def test_high_conviction_10pct_sizing() -> None:
    """Test #2: cv 9 → high tier (10%); equity $1000 → $100."""
    p = _proposal(conviction=9, size_pct=0.05)
    account = _account(equity=1000.0)
    engine = SizingEngine()

    result = engine.size(p, account, open_positions=[], quote=_quote(last=10.0), cfg=_cfg())

    assert isinstance(result, SizingAccepted)
    assert result.realized_pct == pytest.approx(0.10)
    assert result.realized_dollar_size == pytest.approx(100.0)


def test_ks4_absolute_floor_caps_size() -> None:
    """Test #3: equity $20K, cv 10 → 10% = $2000 → ks4_absolute_cap caps to $1000."""
    p = _proposal(conviction=10, size_pct=0.10)
    account = _account(equity=20_000.0)
    engine = SizingEngine()

    result = engine.size(p, account, open_positions=[], quote=_quote(last=10.0), cfg=_cfg())

    assert isinstance(result, SizingAccepted)
    assert result.realized_dollar_size == pytest.approx(1000.0)
    # Realized pct reflects the actual capped fraction, not the tier rate.
    assert result.realized_pct == pytest.approx(1000.0 / 20_000.0)


def test_ks4_pct_floor_dominates_at_low_equity() -> None:
    """Test #4: equity $1K, cv 10 → 10% = $100; absolute cap ($1000) inactive."""
    p = _proposal(conviction=10, size_pct=0.10)
    account = _account(equity=1000.0)
    engine = SizingEngine()

    result = engine.size(p, account, open_positions=[], quote=_quote(last=10.0), cfg=_cfg())

    assert isinstance(result, SizingAccepted)
    assert result.realized_dollar_size == pytest.approx(100.0)


def test_ks4_pct_cap_applies_when_tier_exceeds_pct_cap() -> None:
    """KS-4 §5.2: cap is min(ks4_pct_cap * equity, ks4_absolute_cap_usd).

    With sizing_high_pct=0.30 (hypothetical) and ks4_pct_cap=0.20, the pct
    cap fires first below the absolute floor.
    """
    p = _proposal(conviction=10)
    account = _account(equity=2000.0)
    engine = SizingEngine()

    cfg = _cfg(sizing_high_pct=0.30)  # tier wants $600, pct cap = $400
    result = engine.size(p, account, open_positions=[], quote=_quote(last=10.0), cfg=cfg)

    assert isinstance(result, SizingAccepted)
    assert result.realized_dollar_size == pytest.approx(400.0)


def test_ks5_rejects_when_5_open_positions() -> None:
    """Test #5: ks5_max_concurrent=5; 5 open → reject ks5_concurrent_limit."""
    p = _proposal(conviction=8)
    account = _account(equity=5000.0)
    open_positions = [_position(f"SYM{i}") for i in range(5)]
    engine = SizingEngine()

    result = engine.size(p, account, open_positions=open_positions, quote=_quote(), cfg=_cfg())

    assert isinstance(result, SizingRejected)
    assert result.reason == "ks5_concurrent_limit"


def test_ks5_accepts_at_4_open_positions() -> None:
    """Boundary: 4 open positions allowed; 5th proposal → accepted."""
    p = _proposal(conviction=8)
    account = _account(equity=5000.0)
    open_positions = [_position(f"SYM{i}") for i in range(4)]
    engine = SizingEngine()

    result = engine.size(p, account, open_positions=open_positions, quote=_quote(), cfg=_cfg())

    assert isinstance(result, SizingAccepted)


def test_ks6_rejects_re_entry_of_held_name() -> None:
    """Test #6: existing position in proposal.symbol → reject ks6_already_held."""
    p = _proposal(symbol="ACME", conviction=8)
    account = _account(equity=5000.0)
    open_positions = [_position("OTHER"), _position("ACME")]
    engine = SizingEngine()

    result = engine.size(p, account, open_positions=open_positions, quote=_quote(), cfg=_cfg())

    assert isinstance(result, SizingRejected)
    assert result.reason == "ks6_already_held"


def test_ks7_reduces_size_to_maintain_reserve() -> None:
    """Test #7: tier size violates reserve → realized reduced to keep reserve.

    Equity $5000, ks7_cash_reserve_pct=0.05 → reserve floor = $250.
    Cash = $300. Mid-tier (cv 8) wants $250. After spend, cash would be
    $50, below the $250 reserve. Reduce realized so cash_after >= $250:
    realized <= $300 - $250 = $50.
    """
    p = _proposal(conviction=8, size_pct=0.10)
    account = _account(equity=5000.0, cash=300.0)
    engine = SizingEngine()

    result = engine.size(p, account, open_positions=[], quote=_quote(last=10.0), cfg=_cfg())

    assert isinstance(result, SizingAccepted)
    assert result.realized_dollar_size == pytest.approx(50.0)
    # Sanity: cash after = 300 - 50 = 250 = reserve floor.
    assert result.qty == 5


def test_ks7_rejects_when_reserve_unreachable() -> None:
    """Test #8: cash floor ≥ cash → cannot place even one share → reject."""
    p = _proposal(conviction=8)
    # Reserve floor = 5% of $5000 = $250. Cash = $200 already below floor;
    # any spend makes it worse and 1 share at $10 = $10 leaves $190 < $250.
    account = _account(equity=5000.0, cash=200.0)
    engine = SizingEngine()

    result = engine.size(p, account, open_positions=[], quote=_quote(last=10.0), cfg=_cfg())

    assert isinstance(result, SizingRejected)
    assert result.reason == "ks7_cash_reserve"


def test_size_below_one_share_rejected() -> None:
    """Test #9: realized_dollar_size / quote.last < 1 → reject."""
    # Equity $100, mid tier $5 (5%); price $10 → qty = floor(5/10) = 0.
    p = _proposal(conviction=8)
    account = _account(equity=100.0)
    engine = SizingEngine()

    result = engine.size(p, account, open_positions=[], quote=_quote(last=10.0), cfg=_cfg())

    assert isinstance(result, SizingRejected)
    assert result.reason == "size_too_small_for_one_share"


def test_realized_dollar_size_request_logged_for_journaling() -> None:
    """Test #10 / AC-8: request and realized are both surfaced.

    Request = size_pct_of_capital * equity (the LLM's ask).
    Realized = post-cap, post-reserve actual.
    """
    p = _proposal(conviction=10, size_pct=0.15)  # LLM asks 15% (pre-schema-cap)
    # Note: schema's max is 0.20; 0.15 is within. cv 10 → 10% tier.
    account = _account(equity=20_000.0)
    engine = SizingEngine()

    result = engine.size(p, account, open_positions=[], quote=_quote(last=10.0), cfg=_cfg())

    assert isinstance(result, SizingAccepted)
    # LLM's request: 0.15 * 20000 = $3000.
    assert result.realized_dollar_size_request == pytest.approx(3000.0)
    # Realized: tier 10% = $2000, capped by ks4_absolute = $1000.
    assert result.realized_dollar_size == pytest.approx(1000.0)


def test_low_conviction_branch_is_defensive_assertion() -> None:
    """AC-2 defensive: cv < 7 should not reach sizing (analyzer routes elsewhere).

    The engine asserts; this catches an upstream wiring bug.
    """
    p = _proposal(conviction=5)
    account = _account(equity=5000.0)
    engine = SizingEngine()

    with pytest.raises(AssertionError):
        engine.size(p, account, open_positions=[], quote=_quote(), cfg=_cfg())
