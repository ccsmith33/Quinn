"""S6.1 — BrokerAdapter Protocol + domain types (architecture §2.6)."""

from __future__ import annotations

import datetime as dt
import inspect

import pytest
from pydantic import ValidationError

from broker.protocol import (
    AccountSnapshot,
    BrokerAdapter,
    OrderRequest,
    Position,
    Quote,
    SubmittedOrder,
)


def test_protocol_methods_present() -> None:
    """AC-1: BrokerAdapter protocol exposes the required methods.

    Hotfix 2026-05-07: `get_open_orders` added so the KS-5 gate sees
    pre-market queued entries; `submit_bracket_order` added so the
    submitter can issue atomic complex orders that bypass Alpaca's
    wash-trade detector on back-to-back stops.
    """
    expected = {
        "submit_order",
        "submit_bracket_order",
        "cancel_order",
        "get_account",
        "get_positions",
        "get_open_orders",
        "get_quote",
    }
    actual = {name for name in dir(BrokerAdapter) if not name.startswith("_")}
    assert expected.issubset(actual), f"missing: {expected - actual}"


def test_order_request_is_pydantic_model() -> None:
    req = OrderRequest(
        symbol="AAPL",
        side="buy",
        qty=10,
        order_type="market",
        tif="day",
        client_order_id="cid-1",
    )
    assert req.symbol == "AAPL"
    assert req.qty == 10


def test_order_request_validates_side() -> None:
    with pytest.raises(ValidationError):
        OrderRequest(
            symbol="AAPL",
            side="sideways",  # type: ignore[arg-type]
            qty=10,
            order_type="market",
            tif="day",
            client_order_id="cid-1",
        )


def test_order_request_limit_requires_limit_price() -> None:
    with pytest.raises(ValidationError):
        OrderRequest(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="limit",
            tif="day",
            client_order_id="cid-1",
        )


def test_submitted_order_has_required_fields() -> None:
    so = SubmittedOrder(
        broker_order_id="abc-123",
        client_order_id="cid-1",
        symbol="AAPL",
        side="buy",
        qty=10,
        order_type="market",
        status="accepted",
        submitted_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
    )
    assert so.broker_order_id == "abc-123"


def test_account_snapshot_shape() -> None:
    snap = AccountSnapshot(
        equity=100_000.0,
        cash=50_000.0,
        buying_power=200_000.0,
        long_market_value=50_000.0,
        daypl=123.45,
        snapshot_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
    )
    assert snap.equity == 100_000.0


def test_position_shape() -> None:
    p = Position(
        symbol="AAPL",
        qty=10,
        avg_entry_price=180.5,
        market_value=1900.0,
        unrealized_pnl=95.0,
    )
    assert p.symbol == "AAPL"


def test_quote_shape_includes_bid_ask_last_ts() -> None:
    """AC-6: Quote returns {bid, ask, last, ts} snapshot."""
    q = Quote(
        symbol="AAPL",
        bid=180.0,
        ask=180.05,
        last=180.02,
        ts=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
    )
    assert q.bid == 180.0
    assert q.ask == 180.05
    assert q.last == 180.02
    assert q.ts.tzinfo is not None


def test_protocol_signatures() -> None:
    """AC-1: signatures match the specified shapes."""
    sigs = {
        name: inspect.signature(getattr(BrokerAdapter, name))
        for name in (
            "submit_order",
            "cancel_order",
            "get_account",
            "get_positions",
            "get_quote",
        )
    }
    # submit_order(self, req: OrderRequest) -> SubmittedOrder
    params = list(sigs["submit_order"].parameters)
    assert params == ["self", "req"]
    # cancel_order(self, broker_order_id: str) -> None
    assert list(sigs["cancel_order"].parameters) == ["self", "broker_order_id"]
    # get_account(self) -> AccountSnapshot
    assert list(sigs["get_account"].parameters) == ["self"]
    # get_positions(self) -> list[Position]
    assert list(sigs["get_positions"].parameters) == ["self"]
    # get_quote(self, symbol: str) -> Quote
    assert list(sigs["get_quote"].parameters) == ["self", "symbol"]
