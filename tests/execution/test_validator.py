"""S6.2 — Execution validation pipeline tests.

Order of checks (deterministic, fixed by AC-3): schema → kill-switch →
universe → price floor → capital. Earlier checks short-circuit later ones,
so a kill-switch reject must NOT call into universe / broker quote / broker
account paths. Tests verify each rejection reason in isolation and the
short-circuit invariant.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from broker.protocol import AccountSnapshot, OrderRequest, Position, Quote, SubmittedOrder
from execution.validator import Accepted, ProposalValidator, Rejected
from killswitch.api import KillSwitchUninitialized
from proposal.schemas import TradeProposal

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _RecordingBroker:
    """Minimal BrokerAdapter fake that records which methods were called."""

    def __init__(
        self,
        *,
        quote: Quote | None = None,
        account: AccountSnapshot | None = None,
    ) -> None:
        self._quote = quote
        self._account = account
        self.calls: list[str] = []

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:  # pragma: no cover
        self.calls.append("submit_order")
        raise AssertionError("submit_order must not be called from validator")

    def cancel_order(self, broker_order_id: str) -> None:  # pragma: no cover
        self.calls.append("cancel_order")
        raise AssertionError("cancel_order must not be called from validator")

    def get_account(self) -> AccountSnapshot:
        self.calls.append("get_account")
        if self._account is None:
            raise AssertionError("get_account called but no account configured")
        return self._account

    def get_positions(self) -> list[Position]:  # pragma: no cover
        self.calls.append("get_positions")
        return []

    def get_quote(self, symbol: str) -> Quote:
        self.calls.append("get_quote")
        if self._quote is None:
            raise AssertionError(f"get_quote({symbol!r}) called but no quote configured")
        return self._quote


class _FakeUniverse:
    def __init__(self, members: set[str]) -> None:
        self._members = members
        self.calls: list[str] = []

    def is_in_universe(self, ticker: str) -> bool:
        self.calls.append(ticker)
        return ticker in self._members


class _FakeKillSwitch:
    def __init__(self, *, halted: bool = False, raise_uninit: bool = False) -> None:
        self._halted = halted
        self._raise = raise_uninit
        self.calls: list[str] = []

    def is_halted(self) -> bool:
        self.calls.append("is_halted")
        if self._raise:
            raise KillSwitchUninitialized("seed row missing")
        return self._halted


class _FakeJournal:
    """Placeholder — S6.2 does not write to the journal (AC-7)."""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _proposal(
    *,
    symbol: str = "ACME",
    direction: str = "long",
    size_pct: float = 0.05,
    entry_style: str = "market_open",
    entry_limit_price: float | None = None,
    stop_loss_price: float = 4.50,
    take_profit_price: float | None = None,
) -> TradeProposal:
    payload: dict[str, Any] = {
        "symbol": symbol,
        "direction": direction,
        "size_pct_of_capital": size_pct,
        "entry_style": entry_style,
        "stop_loss_price": stop_loss_price,
        "time_horizon_days": 10,
        "conviction": 8,
        "thesis": (
            "Strong fundamentals with material 8-K disclosure indicating "
            "near-term catalyst per filing analysis."
        ),
        "signals": ["8-K item 2.02 strong earnings beat"],
        "exit_conditions": ["thesis breaks; stop hit; 30 days elapsed"],
        "risk_factors": ["macro risk; sector rotation; news risk"],
    }
    if entry_limit_price is not None:
        payload["entry_limit_price"] = entry_limit_price
    if take_profit_price is not None:
        payload["take_profit_price"] = take_profit_price
    return TradeProposal.model_validate(payload)


def _quote(*, bid: float, ask: float | None = None, last: float | None = None) -> Quote:
    return Quote(
        symbol="ACME",
        bid=bid,
        ask=ask if ask is not None else bid + 0.05,
        last=last if last is not None else bid + 0.02,
        ts=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
    )


def _account(*, buying_power: float = 100_000.0, equity: float = 100_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        cash=buying_power,
        buying_power=buying_power,
        long_market_value=0.0,
        daypl=0.0,
        snapshot_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
    )


# ---------------------------------------------------------------------------
# Tests (story §test plan)
# ---------------------------------------------------------------------------

def test_kill_switch_halted_rejects() -> None:
    """First failing test (story §test plan #1): KS halted → Rejected('kill_switch')."""
    broker = _RecordingBroker()
    universe = _FakeUniverse({"ACME"})
    ks = _FakeKillSwitch(halted=True)
    validator = ProposalValidator()

    result = validator.validate(_proposal(), broker, universe, ks, _FakeJournal())

    assert isinstance(result, Rejected)
    assert result.reason == "kill_switch"


def test_kill_switch_uninitialized_treated_as_halted() -> None:
    """Fail-closed per S7.1 contract: missing seed row → reject as kill_switch."""
    broker = _RecordingBroker()
    universe = _FakeUniverse({"ACME"})
    ks = _FakeKillSwitch(raise_uninit=True)
    validator = ProposalValidator()

    result = validator.validate(_proposal(), broker, universe, ks, _FakeJournal())

    assert isinstance(result, Rejected)
    assert result.reason == "kill_switch"


def test_universe_miss_rejects() -> None:
    """AC-2: symbol not in universe → Rejected('universe')."""
    broker = _RecordingBroker()
    universe = _FakeUniverse({"OTHER"})
    ks = _FakeKillSwitch()
    validator = ProposalValidator()

    result = validator.validate(_proposal(symbol="ACME"), broker, universe, ks, _FakeJournal())

    assert isinstance(result, Rejected)
    assert result.reason == "universe"


def test_price_floor_rejects_on_bid_below_5() -> None:
    """AC-4 / ADR-006: bid drops below $5.00 by trade time → Rejected('price_floor')."""
    broker = _RecordingBroker(
        quote=_quote(bid=4.80, ask=4.85, last=4.82),
        account=_account(),
    )
    universe = _FakeUniverse({"ACME"})
    ks = _FakeKillSwitch()
    validator = ProposalValidator()

    result = validator.validate(_proposal(), broker, universe, ks, _FakeJournal())

    assert isinstance(result, Rejected)
    assert result.reason == "price_floor"


def test_price_floor_rejects_on_last_below_5() -> None:
    """AC-2 'price_floor': last < $5.00 also triggers rejection."""
    broker = _RecordingBroker(
        quote=_quote(bid=5.10, ask=5.15, last=4.99),
        account=_account(),
    )
    universe = _FakeUniverse({"ACME"})
    ks = _FakeKillSwitch()
    validator = ProposalValidator()

    result = validator.validate(_proposal(), broker, universe, ks, _FakeJournal())

    assert isinstance(result, Rejected)
    assert result.reason == "price_floor"


def test_price_floor_accepts_at_5_exactly() -> None:
    """Boundary: bid == $5.00 and last == $5.00 → not rejected by price_floor."""
    broker = _RecordingBroker(
        quote=_quote(bid=5.00, ask=5.05, last=5.00),
        account=_account(),
    )
    universe = _FakeUniverse({"ACME"})
    ks = _FakeKillSwitch()
    validator = ProposalValidator()

    result = validator.validate(_proposal(), broker, universe, ks, _FakeJournal())

    assert isinstance(result, Accepted)


def test_short_direction_rejected() -> None:
    """AC-2: direction != 'long' → Rejected('direction_unsupported').

    `TradeProposal.direction` is `Literal['long']`; the validator must still
    defend against payloads that bypass pydantic (e.g., a dict promoted via
    a mutable wrapper). We simulate by mutating the model's underlying dict.
    """
    p = _proposal()
    # Bypass pydantic's runtime validation to simulate an out-of-band write.
    object.__setattr__(p, "direction", "short")

    broker = _RecordingBroker()
    universe = _FakeUniverse({"ACME"})
    ks = _FakeKillSwitch()
    validator = ProposalValidator()

    result = validator.validate(p, broker, universe, ks, _FakeJournal())

    assert isinstance(result, Rejected)
    assert result.reason == "direction_unsupported"


def test_insufficient_capital_rejects() -> None:
    """AC-5: proposed dollar size > buying_power → Rejected('insufficient_capital')."""
    # 5% of $1000 equity = $50 dollar size; buying_power $40 < $50.
    broker = _RecordingBroker(
        quote=_quote(bid=10.00, ask=10.05, last=10.00),
        account=_account(buying_power=40.0, equity=1000.0),
    )
    universe = _FakeUniverse({"ACME"})
    ks = _FakeKillSwitch()
    validator = ProposalValidator()

    result = validator.validate(_proposal(size_pct=0.05), broker, universe, ks, _FakeJournal())

    assert isinstance(result, Rejected)
    assert result.reason == "insufficient_capital"


def test_acceptance_returns_augmented_proposal() -> None:
    """AC-6: accepted proposal carries `realized_dollar_size`.

    S6.2 stubs sizing with the requested size (size_pct * equity); S6.3
    replaces the stub with the real sizing engine.
    """
    broker = _RecordingBroker(
        quote=_quote(bid=10.00, ask=10.05, last=10.00),
        account=_account(buying_power=10_000.0, equity=10_000.0),
    )
    universe = _FakeUniverse({"ACME"})
    ks = _FakeKillSwitch()
    validator = ProposalValidator()

    result = validator.validate(_proposal(size_pct=0.05), broker, universe, ks, _FakeJournal())

    assert isinstance(result, Accepted)
    # 5% of $10K equity = $500.
    assert result.realized_dollar_size == pytest.approx(500.0)
    assert result.proposal.symbol == "ACME"


def test_check_order_short_circuits_on_kill_switch() -> None:
    """AC-3: KS halted → universe lookup and broker quote/account NOT called."""
    broker = _RecordingBroker()
    universe = _FakeUniverse({"ACME"})
    ks = _FakeKillSwitch(halted=True)
    validator = ProposalValidator()

    validator.validate(_proposal(), broker, universe, ks, _FakeJournal())

    assert ks.calls == ["is_halted"]
    assert universe.calls == []
    assert broker.calls == []


def test_check_order_short_circuits_on_universe_miss() -> None:
    """AC-3: universe miss → broker quote/account NOT called."""
    broker = _RecordingBroker()
    universe = _FakeUniverse({"OTHER"})
    ks = _FakeKillSwitch()
    validator = ProposalValidator()

    validator.validate(_proposal(), broker, universe, ks, _FakeJournal())

    assert universe.calls == ["ACME"]
    assert broker.calls == []


def test_check_order_short_circuits_on_price_floor() -> None:
    """AC-3: price_floor reject → broker.get_account NOT called."""
    broker = _RecordingBroker(
        quote=_quote(bid=4.80, ask=4.85, last=4.82),
        account=_account(),
    )
    universe = _FakeUniverse({"ACME"})
    ks = _FakeKillSwitch()
    validator = ProposalValidator()

    validator.validate(_proposal(), broker, universe, ks, _FakeJournal())

    # get_quote is called for the price floor check; get_account should NOT be.
    assert "get_quote" in broker.calls
    assert "get_account" not in broker.calls


# ---------------------------------------------------------------------------
# D-079 §3.2 — exit-geometry floor (kills S-4's unconstrained half).
# Deterministic, runs after the price floor (the quote is already in hand)
# and before the capital check. Longs only in v1.
# Entry reference: entry_limit_price for limit entries, quote.last otherwise
# (the same pre-submission reference the executor's inverted-stop guard uses).
# ---------------------------------------------------------------------------


def test_exit_geometry_rejects_tp_at_or_below_entry() -> None:
    """The AORT-class inversion on the TP side: tp <= entry can only
    harvest an instant loss-or-zero. quote.last=10.02; tp=9.00."""
    p = _proposal(stop_loss_price=9.00, take_profit_price=9.00)
    broker = _RecordingBroker(quote=_quote(bid=10.00), account=_account())
    result = ProposalValidator().validate(
        p, broker, _FakeUniverse({"ACME"}), _FakeKillSwitch(), _FakeJournal()
    )
    assert isinstance(result, Rejected)
    assert result.reason == "exit_geometry"


def test_exit_geometry_rejects_reward_risk_below_floor() -> None:
    """entry=10.02 (quote.last), stop=9.02 → risk 1.00; tp=10.52 →
    reward 0.50; R:R 0.5 < default floor 1.5 → reject."""
    p = _proposal(stop_loss_price=9.02, take_profit_price=10.52)
    broker = _RecordingBroker(quote=_quote(bid=10.00), account=_account())
    result = ProposalValidator().validate(
        p, broker, _FakeUniverse({"ACME"}), _FakeKillSwitch(), _FakeJournal()
    )
    assert isinstance(result, Rejected)
    assert result.reason == "exit_geometry"


def test_exit_geometry_accepts_reward_risk_at_floor() -> None:
    """R:R exactly at the floor passes (>= contract): entry=10.02,
    stop=9.02 (risk 1.00), tp=11.52 (reward 1.50) → 1.5 >= 1.5."""
    p = _proposal(stop_loss_price=9.02, take_profit_price=11.52)
    broker = _RecordingBroker(quote=_quote(bid=10.00), account=_account())
    result = ProposalValidator().validate(
        p, broker, _FakeUniverse({"ACME"}), _FakeKillSwitch(), _FakeJournal()
    )
    assert isinstance(result, Accepted)


def test_exit_geometry_skipped_when_no_take_profit() -> None:
    """TP remains optional (D-009 discretion preserved): a no-TP proposal
    relies on stop + trailing + thesis review and passes the floor."""
    p = _proposal(stop_loss_price=9.02, take_profit_price=None)
    broker = _RecordingBroker(quote=_quote(bid=10.00), account=_account())
    result = ProposalValidator().validate(
        p, broker, _FakeUniverse({"ACME"}), _FakeKillSwitch(), _FakeJournal()
    )
    assert isinstance(result, Accepted)


def test_exit_geometry_rejects_inverted_stop_when_tp_present() -> None:
    """stop >= entry makes the risk denominator non-positive — geometry
    is meaningless. Reject here (defense in depth; the executor's
    inverted-stop guard would also catch it at submission time)."""
    p = _proposal(stop_loss_price=10.50, take_profit_price=14.00)
    broker = _RecordingBroker(quote=_quote(bid=10.00), account=_account())
    result = ProposalValidator().validate(
        p, broker, _FakeUniverse({"ACME"}), _FakeKillSwitch(), _FakeJournal()
    )
    assert isinstance(result, Rejected)
    assert result.reason == "exit_geometry"


def test_exit_geometry_uses_limit_price_as_entry_reference() -> None:
    """For entry_style=limit the intended fill is the limit price:
    entry=9.00, stop=8.00 (risk 1.00), tp=10.60 (reward 1.60) → passes
    even though quote.last=10.02 would make the reward only 0.58."""
    p = _proposal(
        entry_style="limit",
        entry_limit_price=9.00,
        stop_loss_price=8.00,
        take_profit_price=10.60,
    )
    broker = _RecordingBroker(quote=_quote(bid=10.00), account=_account())
    result = ProposalValidator().validate(
        p, broker, _FakeUniverse({"ACME"}), _FakeKillSwitch(), _FakeJournal()
    )
    assert isinstance(result, Accepted)


def test_exit_geometry_floor_is_configurable() -> None:
    """`min_reward_risk` is a ctor param wired from config (tunable when
    H3 prod data lands, D-080). R:R=0.5 passes a 0.4 floor."""
    p = _proposal(stop_loss_price=9.02, take_profit_price=10.52)
    broker = _RecordingBroker(quote=_quote(bid=10.00), account=_account())
    result = ProposalValidator(min_reward_risk=0.4).validate(
        p, broker, _FakeUniverse({"ACME"}), _FakeKillSwitch(), _FakeJournal()
    )
    assert isinstance(result, Accepted)


def test_exit_geometry_short_circuits_before_capital_check() -> None:
    """Check order: geometry runs before the capital check — a geometry
    reject must not call get_account."""
    p = _proposal(stop_loss_price=9.02, take_profit_price=10.52)
    broker = _RecordingBroker(quote=_quote(bid=10.00), account=_account())
    result = ProposalValidator().validate(
        p, broker, _FakeUniverse({"ACME"}), _FakeKillSwitch(), _FakeJournal()
    )
    assert isinstance(result, Rejected)
    assert "get_account" not in broker.calls
