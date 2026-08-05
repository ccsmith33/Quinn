"""Full-book Opus-review gate — pure-logic tests (API cost lever).

Pins the two properties the feature rests on: the bar rises ONLY when the
book is full and KS-7 headroom cannot fund a single cheapest-legal share,
and the gate is stateless (every call is a fresh read, so a freed slot or
returned cash reverts it with no reset).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from broker.protocol import AccountSnapshot, OpenOrder, Position
from config.loader import ExecutionConfig, Ks5Tier
from execution.review_gate import (
    NO_DISPLACEMENT_FLOOR,
    BookPressure,
    compute_book_pressure,
    displacement_viable_conviction,
    effective_opus_review_threshold,
)


def _cfg(**overrides: Any) -> ExecutionConfig:
    base: dict[str, Any] = {
        "broker_mode": "paper",
        "ks4_pct_cap": 0.15,
        "ks4_absolute_cap_usd": 1000.0,
        "ks5_max_concurrent": 10,
        "ks7_cash_reserve_pct": 0.03,
        "sizing_mid_pct": 0.07,
        "sizing_high_pct": 0.10,
    }
    base.update(overrides)
    return ExecutionConfig(**base)


class _FakeBroker:
    def __init__(
        self,
        *,
        equity: float = 10_000.0,
        cash: float = 5_000.0,
        symbols: list[str] | None = None,
        pending: list[str] | None = None,
        fail: bool = False,
    ) -> None:
        self._equity = equity
        self._cash = cash
        self._symbols = symbols or []
        self._pending = pending or []
        self._fail = fail

    def get_account(self) -> AccountSnapshot:
        if self._fail:
            raise RuntimeError("broker down")
        return AccountSnapshot(
            equity=self._equity,
            cash=self._cash,
            buying_power=self._cash,
            long_market_value=self._equity - self._cash,
            daypl=0.0,
            snapshot_at=dt.datetime.now(dt.UTC),
        )

    def get_positions(self) -> list[Position]:
        return [
            Position(
                symbol=s,
                qty=10,
                avg_entry_price=50.0,
                market_value=500.0,
                unrealized_pnl=0.0,
            )
            for s in self._symbols
        ]

    def get_open_orders(self) -> list[OpenOrder]:
        return [
            OpenOrder(
                broker_order_id=f"o-{s}",
                client_order_id=f"prop-{s}-entry",
                symbol=s,
                side="buy",
                qty=10,
                order_type="market",
                status="accepted",
            )
            for s in self._pending
        ]


def _no_spend(_orders: list[Any]) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# displacement_viable_conviction
# ---------------------------------------------------------------------------

def test_viable_conviction_without_displacement_is_the_fixed_floor() -> None:
    assert displacement_viable_conviction(_cfg()) == NO_DISPLACEMENT_FLOOR


def test_viable_conviction_tracks_displacement_min_when_enabled() -> None:
    cfg = _cfg(displacement_enabled=True, displacement_min_conviction=9)
    assert displacement_viable_conviction(cfg) == 9


def test_young_victim_override_lowers_the_viable_bar() -> None:
    """A young-victim knob BELOW the base floor must not cause a proposal
    displacement would act on to be skipped."""
    cfg = _cfg(
        displacement_enabled=True,
        displacement_min_conviction=9,
        displacement_young_victim_min_conviction=8,
    )
    assert displacement_viable_conviction(cfg) == 8


def test_young_victim_disabled_zero_does_not_lower_the_bar() -> None:
    cfg = _cfg(
        displacement_enabled=True,
        displacement_min_conviction=8,
        displacement_young_victim_min_conviction=0,
    )
    assert displacement_viable_conviction(cfg) == 8


# ---------------------------------------------------------------------------
# compute_book_pressure
# ---------------------------------------------------------------------------

def test_pressure_counts_held_union_pending_against_the_tiered_cap() -> None:
    cfg = _cfg(
        ks5_max_concurrent=10,
        ks5_tiers=[Ks5Tier(equity_max=15_000.0, max_positions=4)],
    )
    broker = _FakeBroker(equity=10_000.0, symbols=["A", "B"], pending=["B", "C"])
    pressure = compute_book_pressure(
        broker=broker, cfg=cfg, pending_entry_spend_fn=_no_spend
    )
    assert pressure is not None
    # Tiered cap 4, used = |{A,B} ∪ {B,C}| = 3 → one slot left.
    assert pressure.open_slots == 1


def test_pressure_headroom_subtracts_reserve_and_committed_spend() -> None:
    cfg = _cfg(ks7_cash_reserve_pct=0.10)
    broker = _FakeBroker(equity=10_000.0, cash=3_000.0, pending=["X"])
    pressure = compute_book_pressure(
        broker=broker, cfg=cfg, pending_entry_spend_fn=lambda orders: 500.0
    )
    assert pressure is not None
    # 3000 cash − 500 committed − 1000 reserve floor.
    assert pressure.cash_headroom == 1_500.0


def test_broker_outage_yields_no_pressure() -> None:
    pressure = compute_book_pressure(
        broker=_FakeBroker(fail=True), cfg=_cfg(), pending_entry_spend_fn=_no_spend
    )
    assert pressure is None


def test_unpriced_pending_spend_degrades_to_zero() -> None:
    def _boom(_orders: list[Any]) -> float:
        raise RuntimeError("no journal row")

    pressure = compute_book_pressure(
        broker=_FakeBroker(equity=10_000.0, cash=1_000.0, pending=["X"]),
        cfg=_cfg(ks7_cash_reserve_pct=0.0),
        pending_entry_spend_fn=_boom,
    )
    assert pressure is not None
    assert pressure.cash_headroom == 1_000.0


# ---------------------------------------------------------------------------
# effective_opus_review_threshold
# ---------------------------------------------------------------------------

def _threshold(pressure: BookPressure | None, **kw: Any) -> int:
    params: dict[str, Any] = {
        "configured_threshold": 5,
        "cfg": _cfg(),
        "pressure": pressure,
        "gate_enabled": True,
        "price_floor_usd": 5.0,
    }
    params.update(kw)
    return effective_opus_review_threshold(**params)


def test_gate_off_never_moves_the_bar() -> None:
    locked = BookPressure(open_slots=0, cash_headroom=-100.0)
    assert _threshold(locked, gate_enabled=False) == 5


def test_full_book_and_dry_cash_raises_the_bar() -> None:
    locked = BookPressure(open_slots=0, cash_headroom=1.0)
    assert _threshold(locked) == NO_DISPLACEMENT_FLOOR


def test_a_free_slot_keeps_the_configured_bar() -> None:
    assert _threshold(BookPressure(open_slots=1, cash_headroom=-100.0)) == 5


def test_cash_that_can_fund_one_share_keeps_the_configured_bar() -> None:
    assert _threshold(BookPressure(open_slots=0, cash_headroom=5.0)) == 5


def test_missing_pressure_keeps_the_configured_bar() -> None:
    assert _threshold(None) == 5


def test_overfull_book_negative_slots_raises_the_bar() -> None:
    assert _threshold(BookPressure(open_slots=-2, cash_headroom=0.0)) == 8


def test_configured_bar_above_the_viable_floor_wins() -> None:
    locked = BookPressure(open_slots=0, cash_headroom=0.0)
    assert _threshold(locked, configured_threshold=9) == 9


def test_displacement_config_moves_the_raised_bar() -> None:
    locked = BookPressure(open_slots=0, cash_headroom=0.0)
    cfg = _cfg(displacement_enabled=True, displacement_min_conviction=10)
    assert _threshold(locked, cfg=cfg) == 10
