"""WS1 — `BrokerAdapter.get_order_by_id` + SubmittedOrder fill fields
(architecture-option-b-2026-06-09.md §7.1, ADR-010).

The FillIngestor polls GET /v2/orders/{id} for every journaled order with
`final_status IS NULL`. Self-contained fakes (not test_alpaca's) so WS2's
parallel edits to test_alpaca.py never collide with this file.
"""

from __future__ import annotations

import datetime as dt
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

import broker.alpaca as alpaca_mod
from broker.alpaca import AlpacaBroker
from broker.protocol import BrokerAdapter, SubmittedOrder

PAPER_ENDPOINT = "https://paper-api.alpaca.markets"


class _FakeAPIError(Exception):
    def __init__(self, status_code: int, message: str = "fake api error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self._error = {"message": message}


def _filled_order(
    *,
    order_id: str,
    symbol: str = "AAPL",
    status: str = "filled",
    filled_avg_price: str | None = "10.55",
    filled_qty: str = "10",
    filled_at: dt.datetime | None = dt.datetime(2026, 6, 9, 13, 31, tzinfo=dt.UTC),
) -> Any:
    return SimpleNamespace(
        id=order_id,
        client_order_id=f"cid-{symbol}",
        symbol=symbol,
        side=SimpleNamespace(value="sell"),
        qty=filled_qty,
        order_type=SimpleNamespace(value="market"),
        type=SimpleNamespace(value="market"),
        status=SimpleNamespace(value=status),
        submitted_at=dt.datetime(2026, 6, 9, 13, 30, tzinfo=dt.UTC),
        limit_price=None,
        stop_price=None,
        filled_avg_price=filled_avg_price,
        filled_qty=filled_qty,
        filled_at=filled_at,
    )


class _FakeTradingClient:
    def __init__(self, **_: Any) -> None:
        self.get_by_id_calls: list[str] = []
        self.orders_by_id: dict[str, Any] = {}
        self.raise_seq: list[Exception] = []

    def get_order_by_id(self, order_id: str) -> Any:
        self.get_by_id_calls.append(str(order_id))
        if self.raise_seq:
            raise self.raise_seq.pop(0)
        if str(order_id) not in self.orders_by_id:
            raise _FakeAPIError(404, "order not found")
        return self.orders_by_id[str(order_id)]


class _FakeDataClient:
    def __init__(self, **_: Any) -> None:
        pass


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> AlpacaBroker:
    monkeypatch.setattr(alpaca_mod, "TradingClient", _FakeTradingClient)
    monkeypatch.setattr(alpaca_mod, "StockHistoricalDataClient", _FakeDataClient)
    monkeypatch.setattr(alpaca_mod, "APIError", _FakeAPIError)
    monkeypatch.setattr(alpaca_mod.time, "sleep", lambda *_: None)
    return AlpacaBroker(
        mode="paper",
        api_key_id=SecretStr("k"),
        api_secret=SecretStr("s"),
        endpoint=PAPER_ENDPOINT,
    )


def _client(broker: AlpacaBroker) -> _FakeTradingClient:
    return broker._trading  # type: ignore[attr-defined,return-value]


# ---------------------------------------------------------------------------
# SubmittedOrder fill fields (§7.1)
# ---------------------------------------------------------------------------

def test_submitted_order_fill_fields_default_backward_compatible() -> None:
    """Pre-WS1 constructors (no fill kwargs) must keep working."""
    o = SubmittedOrder(
        broker_order_id="b-1",
        client_order_id="c-1",
        symbol="AAPL",
        side="buy",
        qty=10,
        order_type="market",
        status="accepted",
        submitted_at=dt.datetime(2026, 6, 9, 13, 30, tzinfo=dt.UTC),
    )
    assert o.filled_avg_price is None
    assert o.filled_qty == 0
    assert o.filled_at is None


def test_protocol_includes_get_order_by_id(broker: AlpacaBroker) -> None:
    assert hasattr(BrokerAdapter, "get_order_by_id")
    assert isinstance(broker, BrokerAdapter)


# ---------------------------------------------------------------------------
# AlpacaBroker.get_order_by_id
# ---------------------------------------------------------------------------

def test_get_order_by_id_normalizes_fill_fields(broker: AlpacaBroker) -> None:
    oid = str(uuid.uuid4())
    _client(broker).orders_by_id[oid] = _filled_order(order_id=oid)

    got = broker.get_order_by_id(oid)

    assert got is not None
    assert got.broker_order_id == oid
    assert got.status == "filled"
    assert got.filled_avg_price == 10.55
    assert got.filled_qty == 10
    assert got.filled_at == dt.datetime(2026, 6, 9, 13, 31, tzinfo=dt.UTC)


def test_get_order_by_id_404_returns_none(broker: AlpacaBroker) -> None:
    assert broker.get_order_by_id("no-such-order") is None


def test_get_order_by_id_unfilled_order_has_empty_fill_fields(
    broker: AlpacaBroker,
) -> None:
    oid = str(uuid.uuid4())
    _client(broker).orders_by_id[oid] = _filled_order(
        order_id=oid,
        status="new",
        filled_avg_price=None,
        filled_qty="0",
        filled_at=None,
    )

    got = broker.get_order_by_id(oid)

    assert got is not None
    assert got.status == "new"
    assert got.filled_avg_price is None
    assert got.filled_qty == 0
    assert got.filled_at is None


def test_get_order_by_id_retries_transient_then_succeeds(
    broker: AlpacaBroker,
) -> None:
    oid = str(uuid.uuid4())
    client = _client(broker)
    client.orders_by_id[oid] = _filled_order(order_id=oid)
    client.raise_seq = [_FakeAPIError(429, "rate limited")]

    got = broker.get_order_by_id(oid)

    assert got is not None
    assert got.filled_qty == 10
    assert len(client.get_by_id_calls) == 2
