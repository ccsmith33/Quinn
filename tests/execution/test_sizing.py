"""S6.3 — Sizing engine + KS-4..KS-7 pre-trade caps tests.

PRD §6.1 (sizing function) and §5.2 (KS-4..KS-7) specify the exact thresholds.
Pure logic — every collaborator (broker, universe, ks) is an injected value
or fake. No I/O.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from broker.protocol import AccountSnapshot, OpenOrder, Position, Quote
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


def _open_buy(symbol: str, qty: int = 10) -> OpenOrder:
    """Pending entry-leg BUY order, simulating the pre-market queue."""
    return OpenOrder(
        symbol=symbol,
        side="buy",
        qty=qty,
        order_type="market",
        status="accepted",
        broker_order_id=f"ord-{symbol}",
        client_order_id=f"prop-{symbol}-entry",
    )


def _open_sell(symbol: str, qty: int = 10) -> OpenOrder:
    """Pending protective SELL leg (stop or TP from a prior day).

    These should NOT count toward KS-5 capacity — the underlying long is
    already in `open_positions`.
    """
    return OpenOrder(
        symbol=symbol,
        side="sell",
        qty=qty,
        order_type="stop",
        status="new",
        broker_order_id=f"ord-{symbol}-stop",
        client_order_id=f"prop-{symbol}-stop",
    )


def _cfg(
    *,
    ks4_pct_cap: float = 0.20,
    ks4_absolute_cap_usd: float = 1000.0,
    ks5_max_concurrent: int = 5,
    ks5_tiers: list[dict[str, float]] | None = None,
    ks7_cash_reserve_pct: float = 0.05,
    sizing_mid_pct: float = 0.05,
    sizing_high_pct: float = 0.10,
    sizing_high_conviction_min: int = 9,
) -> ExecutionConfig:
    return ExecutionConfig(
        broker_mode="paper",
        ks4_pct_cap=ks4_pct_cap,
        ks4_absolute_cap_usd=ks4_absolute_cap_usd,
        ks5_max_concurrent=ks5_max_concurrent,
        ks5_tiers=ks5_tiers or [],  # type: ignore[arg-type]
        ks7_cash_reserve_pct=ks7_cash_reserve_pct,
        sizing_mid_pct=sizing_mid_pct,
        sizing_high_pct=sizing_high_pct,
        sizing_high_conviction_min=sizing_high_conviction_min,
    )


# Default tier curve the operator opts into (D-087 fix #3). Mirrors the
# breakpoints documented in the completion report / example config.
_DEFAULT_TIERS: list[dict[str, float]] = [
    {"equity_max": 5_000.0, "max_positions": 10},
    {"equity_max": 15_000.0, "max_positions": 15},
    {"equity_max": 35_000.0, "max_positions": 20},
    {"equity_max": 75_000.0, "max_positions": 25},
]


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


def test_default_high_tier_threshold_boundary() -> None:
    """Regression: with the default config (sizing_high_conviction_min=9),
    cv 8 sizes at the mid rate and cv 9 at the high rate — byte-identical
    to the pre-config-knob `_TIER_HIGH_THRESHOLD = 9` behavior."""
    account = _account(equity=5000.0)
    engine = SizingEngine()

    mid = engine.size(
        _proposal(conviction=8), account, open_positions=[], quote=_quote(last=10.0), cfg=_cfg()
    )
    high = engine.size(
        _proposal(conviction=9), account, open_positions=[], quote=_quote(last=10.0), cfg=_cfg()
    )

    assert isinstance(mid, SizingAccepted)
    assert mid.realized_pct == pytest.approx(0.05)
    assert isinstance(high, SizingAccepted)
    assert high.realized_pct == pytest.approx(0.10)


def test_configured_high_tier_threshold_lowers_boundary() -> None:
    """With sizing_high_conviction_min=8, cv 8 now earns the high rate and
    cv 7 stays at the mid rate."""
    account = _account(equity=5000.0)
    engine = SizingEngine()
    cfg = _cfg(sizing_high_conviction_min=8)

    high = engine.size(
        _proposal(conviction=8), account, open_positions=[], quote=_quote(last=10.0), cfg=cfg
    )
    mid = engine.size(
        _proposal(conviction=7), account, open_positions=[], quote=_quote(last=10.0), cfg=cfg
    )

    assert isinstance(high, SizingAccepted)
    assert high.realized_pct == pytest.approx(0.10)
    assert isinstance(mid, SizingAccepted)
    assert mid.realized_pct == pytest.approx(0.05)


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


def test_below_floor_conviction_returns_sizing_rejected() -> None:
    """Conviction below the floor (5) is rejected as a journaled outcome
    rather than raising — defense-in-depth so a stray low-cv proposal
    can't crash the agent loop.
    """
    p = _proposal(conviction=4)
    account = _account(equity=5000.0)
    engine = SizingEngine()

    result = engine.size(
        p, account, open_positions=[], quote=_quote(), cfg=_cfg()
    )
    assert isinstance(result, SizingRejected)
    assert result.reason == "conviction_too_low"


def test_ks5_trips_with_only_pending_buys_and_zero_filled() -> None:
    """Hotfix 2026-05-07 (the production bug): pre-market market BUYs
    queue at Alpaca and only fill at 9:30 ET. With ks5_max_concurrent=10
    and 10 pending buys but 0 filled, KS-5 must still trip — pre-fix it
    saw `len(open_positions)=0` and let through what eventually became
    27 unprotected positions.
    """
    p = _proposal(symbol="ACME", conviction=8)
    account = _account(equity=5000.0)
    pending = [_open_buy(f"SYM{i}") for i in range(10)]
    engine = SizingEngine()

    result = engine.size(
        p,
        account,
        open_positions=[],
        quote=_quote(),
        cfg=_cfg(ks5_max_concurrent=10),
        pending_buys=pending,
    )
    assert isinstance(result, SizingRejected)
    assert result.reason == "ks5_concurrent_limit"


def test_ks5_trips_with_mixed_filled_and_pending() -> None:
    """KS-5 counts UNIQUE symbols across (filled ∪ pending entry buys).

    5 filled positions + 5 distinct pending buys = 10 effective slots
    consumed; the next proposal must be rejected at ks5_max_concurrent=10.
    """
    p = _proposal(symbol="NEWCO", conviction=8)
    account = _account(equity=5000.0)
    filled = [_position(f"FILLED{i}") for i in range(5)]
    pending = [_open_buy(f"PEND{i}") for i in range(5)]
    engine = SizingEngine()

    result = engine.size(
        p,
        account,
        open_positions=filled,
        quote=_quote(),
        cfg=_cfg(ks5_max_concurrent=10),
        pending_buys=pending,
    )
    assert isinstance(result, SizingRejected)
    assert result.reason == "ks5_concurrent_limit"


def test_ks6_trips_when_pending_buy_exists_for_symbol() -> None:
    """A second entry on the same symbol while the first is still
    pending fill must be rejected as `ks6_already_held`. This both
    matches the long-only mandate and avoids the wash-trade rejection
    cascade observed pre-fix (Alpaca rejects the second back-to-back
    submission with broker code 40310000).
    """
    p = _proposal(symbol="ACME", conviction=8)
    account = _account(equity=5000.0)
    pending = [_open_buy("ACME")]
    engine = SizingEngine()

    result = engine.size(
        p,
        account,
        open_positions=[],
        quote=_quote(),
        cfg=_cfg(),
        pending_buys=pending,
    )
    assert isinstance(result, SizingRejected)
    assert result.reason == "ks6_already_held"


def test_ks5_does_not_count_pending_sells() -> None:
    """Protective SELL legs (stops / TPs from prior days) surface in
    `get_open_orders()` too but must NOT consume KS-5 capacity — the
    underlying long position is already in `open_positions`. Without
    this, a portfolio with N positions + N stops would falsely
    appear to be at 2N capacity.
    """
    p = _proposal(symbol="NEWCO", conviction=8)
    account = _account(equity=5000.0)
    # 4 filled positions plus their 4 stops = 8 open orders, but only
    # 4 effective KS-5 slots consumed. cfg max=5 → accepted.
    filled = [_position(f"SYM{i}") for i in range(4)]
    pending_sells = [_open_sell(f"SYM{i}") for i in range(4)]
    engine = SizingEngine()

    result = engine.size(
        p,
        account,
        open_positions=filled,
        quote=_quote(),
        cfg=_cfg(ks5_max_concurrent=5),
        pending_buys=pending_sells,
    )
    assert isinstance(result, SizingAccepted)


def test_pending_buys_defaults_to_empty_when_omitted() -> None:
    """Backward-compat: legacy callers that don't pass `pending_buys`
    behave as if there are no pending entries — preserves the prior
    semantics for tests / paths not yet wired to the new broker
    surface.
    """
    p = _proposal(symbol="NEWCO", conviction=8)
    account = _account(equity=5000.0)
    engine = SizingEngine()

    # No `pending_buys=` kwarg → defaults to None → treated as empty.
    result = engine.size(p, account, open_positions=[], quote=_quote(), cfg=_cfg())
    assert isinstance(result, SizingAccepted)


def test_at_floor_conviction_proceeds_through_sizing() -> None:
    """Conviction == floor (5) is accepted and routed through normal sizing
    using the mid-tier rate (cv 5..6 share the mid-tier rate; cv >= 9 hits
    the high-tier rate).
    """
    p = _proposal(conviction=5, size_pct=0.05)
    account = _account(equity=5000.0)
    engine = SizingEngine()

    result = engine.size(
        p, account, open_positions=[], quote=_quote(), cfg=_cfg()
    )
    assert isinstance(result, SizingAccepted)
    # mid-tier rate is 5% per ExecutionConfig default → realized_pct ~ 0.05.
    assert result.realized_pct == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# KS-5 dynamic (tiered) cap (D-087 fix #3).
#
# The effective concurrent cap scales with equity along a tunable, *sublinear*
# breakpoint curve. We exercise it through `engine.size()`: with N positions
# already open at a given equity, the (N+1)-th proposal is accepted iff
# N < effective_cap. We drive the boundary by setting N = cap-1 (accept) and
# N = cap (reject). Equity is always supplied with ample cash so KS-7 / KS-4
# never interfere with the KS-5 assertion.
# ---------------------------------------------------------------------------

def _size_with_n_open(*, equity: float, n_open: int, cfg: ExecutionConfig) -> object:
    """Run sizing for a fresh symbol with `n_open` distinct held positions.

    Cash == equity so KS-7 never trips; price low + tier small so KS-4 and
    the one-share floor never trip. Isolates the KS-5 decision.
    """
    p = _proposal(symbol="NEWCO", conviction=8)
    account = _account(equity=equity, cash=equity)
    open_positions = [_position(f"HELD{i}") for i in range(n_open)]
    return SizingEngine().size(
        p, account, open_positions=open_positions, quote=_quote(last=1.0), cfg=cfg
    )


def test_ks5_flat_when_tiers_unset_preserves_legacy_behavior() -> None:
    """With no tier curve, the cap is exactly `ks5_max_concurrent` at any
    equity — the legacy flat behavior prod relies on today.
    """
    cfg = _cfg(ks5_max_concurrent=10)  # no tiers
    # 10 held at $100k still rejects the 11th (flat, equity-independent).
    rej = _size_with_n_open(equity=100_000.0, n_open=10, cfg=cfg)
    assert isinstance(rej, SizingRejected)
    assert rej.reason == "ks5_concurrent_limit"
    # 9 held → 10th accepted.
    acc = _size_with_n_open(equity=100_000.0, n_open=9, cfg=cfg)
    assert isinstance(acc, SizingAccepted)


@pytest.mark.parametrize(
    ("equity", "expected_cap"),
    [
        (3_500.0, 10),    # anchor: ~10 positions at $3.5k (band <= $5k)
        (5_000.0, 10),    # exact lower-band edge → still 10
        (5_000.01, 15),   # just over $5k → next band
        (10_000.0, 15),   # anchor: ~15 positions at $10k (band <= $15k)
        (15_000.0, 15),   # exact band edge → still 15
        (15_000.01, 20),  # just over $15k → next band
        (35_000.0, 20),   # band edge
        (75_000.0, 25),   # band edge
        (100_000.0, 30),  # anchor: flattens to the hard ceiling, no blow-up
    ],
)
def test_ks5_tiered_cap_band_boundaries(equity: float, expected_cap: int) -> None:
    """The effective cap equals the documented band value at each anchor and
    boundary. ks5_max_concurrent=30 is the hard ceiling for the top band.
    """
    cfg = _cfg(ks5_max_concurrent=30, ks5_tiers=_DEFAULT_TIERS)

    # cap-1 held → accepted; cap held → rejected. This pins the exact cap.
    acc = _size_with_n_open(equity=equity, n_open=expected_cap - 1, cfg=cfg)
    assert isinstance(acc, SizingAccepted), (
        f"equity={equity}: expected accept at {expected_cap - 1} held"
    )
    rej = _size_with_n_open(equity=equity, n_open=expected_cap, cfg=cfg)
    assert isinstance(rej, SizingRejected), (
        f"equity={equity}: expected reject at {expected_cap} held"
    )
    assert rej.reason == "ks5_concurrent_limit"


def test_ks5_tiered_cap_never_exceeds_hard_ceiling() -> None:
    """Even if a tier's max_positions is above ks5_max_concurrent, the
    ceiling is authoritative — the curve can never widen the book past it.
    """
    cfg = _cfg(
        ks5_max_concurrent=12,
        ks5_tiers=[{"equity_max": 5_000.0, "max_positions": 100}],
    )
    rej = _size_with_n_open(equity=1_000.0, n_open=12, cfg=cfg)
    assert isinstance(rej, SizingRejected)
    assert rej.reason == "ks5_concurrent_limit"
    acc = _size_with_n_open(equity=1_000.0, n_open=11, cfg=cfg)
    assert isinstance(acc, SizingAccepted)


def test_ks5_tiered_cap_above_all_tiers_uses_ceiling() -> None:
    """Equity beyond the last breakpoint uses ks5_max_concurrent as the
    final (flat) hard cap — the curve flattens, it does not blow up.
    """
    cfg = _cfg(ks5_max_concurrent=30, ks5_tiers=_DEFAULT_TIERS)
    # $1M: deep above $75k → ceiling 30, identical to the $100k anchor.
    rej = _size_with_n_open(equity=1_000_000.0, n_open=30, cfg=cfg)
    assert isinstance(rej, SizingRejected)
    acc = _size_with_n_open(equity=1_000_000.0, n_open=29, cfg=cfg)
    assert isinstance(acc, SizingAccepted)


# ---------------------------------------------------------------------------
# KS-7 pending-entry-spend race (hotfix 2026-07-08).
#
# Broker-reported cash does NOT decrease while an entry buy is queued
# (Alpaca holds buying_power, not cash, for open orders), so back-to-back
# decisions each saw the same untouched cash and KS-7 approved spends that
# summed past it — paper cash went to −$17 on 2026-07-08. The sizing engine
# now subtracts `pending_entry_spend` (the committed-but-unfilled dollar
# value of open entry buys, computed by the agent loop from journal order
# prices) from `account.cash` before the reserve check.
# ---------------------------------------------------------------------------

def test_ks7_pending_entry_spend_reduces_available_cash() -> None:
    """(a) A pending entry buy's committed spend shrinks what KS-7 will
    let the next proposal spend.

    Equity $5000 → reserve floor $250. Broker cash $1000, but $600 is
    committed to a queued entry buy → available $400 → max spend $150.
    Mid tier (cv 8) wants $250 → downsized to $150 (15 shares @ $10).
    Pre-fix this sized to the full $250 against the untouched cash.
    """
    p = _proposal(conviction=8)
    account = _account(equity=5000.0, cash=1000.0)
    engine = SizingEngine()

    result = engine.size(
        p,
        account,
        open_positions=[],
        quote=_quote(last=10.0),
        cfg=_cfg(),
        pending_buys=[_open_buy("PEND1", qty=24)],
        pending_entry_spend=600.0,
    )

    assert isinstance(result, SizingAccepted)
    assert result.realized_dollar_size == pytest.approx(150.0)
    assert result.qty == 15


def test_ks7_pending_entry_spend_rejects_when_floor_unfundable() -> None:
    """(a) When pending spend pushes available cash below the reserve
    floor, even one share is unfundable → the existing `ks7_cash_reserve`
    rejection fires (same reject reason as before).

    Equity $5000 → floor $250. Cash $500 with $300 pending → available
    $200; max spend −$50 < 1 share @ $10 → reject. Pre-fix this accepted
    a $250 spend.
    """
    p = _proposal(conviction=8)
    account = _account(equity=5000.0, cash=500.0)
    engine = SizingEngine()

    result = engine.size(
        p,
        account,
        open_positions=[],
        quote=_quote(last=10.0),
        cfg=_cfg(),
        pending_buys=[_open_buy("PEND1", qty=30)],
        pending_entry_spend=300.0,
    )

    assert isinstance(result, SizingRejected)
    assert result.reason == "ks7_cash_reserve"


def test_ks7_eols_incident_reconstruction() -> None:
    """(b) The 2026-07-08 EOLS slip, reconstructed.

    MCRB's ~$538 entry was committed but not yet reflected in broker cash
    when EOLS was sized at 13:07 UTC: cash read $1062.44 while true
    available cash was ~$524. Pre-fix KS-7 computed
    max_spend = 1062.44 − 500 (5% of $10k equity) = $562.44 and approved
    10 shares @ $54.10 ≈ $541 — which drove cash to −$17 when both
    entries filled at the bell. Post-fix, the $538.36 pending spend is
    subtracted first: 524.08 − 500 = $24.08 < one share → reject
    `ks7_cash_reserve`.
    """
    p = _proposal(symbol="EOLS", conviction=9)
    account = _account(equity=10_000.0, cash=1062.44)
    engine = SizingEngine()

    result = engine.size(
        p,
        account,
        open_positions=[],
        quote=_quote(last=54.10),
        cfg=_cfg(),
        pending_buys=[_open_buy("MCRB", qty=10)],
        pending_entry_spend=538.36,
    )

    assert isinstance(result, SizingRejected)
    assert result.reason == "ks7_cash_reserve"


def test_ks7_no_pending_spend_behavior_unchanged() -> None:
    """(c) Regression: with no pending orders (spend 0 / param omitted)
    the engine behaves byte-for-byte as before — same downsizing math as
    the original KS-7 test (#7)."""
    p = _proposal(conviction=8, size_pct=0.10)
    account = _account(equity=5000.0, cash=300.0)
    engine = SizingEngine()

    legacy = engine.size(
        p, account, open_positions=[], quote=_quote(last=10.0), cfg=_cfg()
    )
    explicit_zero = engine.size(
        p,
        account,
        open_positions=[],
        quote=_quote(last=10.0),
        cfg=_cfg(),
        pending_buys=[],
        pending_entry_spend=0.0,
    )

    assert isinstance(legacy, SizingAccepted)
    assert isinstance(explicit_zero, SizingAccepted)
    assert explicit_zero == legacy
    assert explicit_zero.realized_dollar_size == pytest.approx(50.0)
    assert explicit_zero.qty == 5
