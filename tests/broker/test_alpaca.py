"""S6.1 — AlpacaBroker concrete implementation (architecture §2.6, ADR-001).

Tests use a fake replacement for the alpaca-py SDK clients so that no real
network calls are made. The goal is to verify:
- protocol conformance,
- endpoint/mode validation (FR-23, D-007),
- response normalization (alpaca-py models → our domain types),
- NFR-5 retry behaviour,
- FR-23 logical equivalence: paper and live adapters expose the same surface.
"""

from __future__ import annotations

import datetime as dt
import inspect
import logging
import statistics
import time
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

import broker.alpaca as alpaca_mod
from broker.alpaca import AlpacaBroker, BrokerUnavailable
from broker.protocol import (
    AccountSnapshot,
    BracketOrderRequest,
    BrokerAdapter,
    BrokerRejected,
    OpenOrder,
    OrderRequest,
    Position,
    Quote,
    SubmittedOrder,
)

PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
LIVE_ENDPOINT = "https://api.alpaca.markets"


# ---------------------------------------------------------------------------
# Fake alpaca-py SDK
# ---------------------------------------------------------------------------


class _FakeAPIError(Exception):
    """Mirrors alpaca.common.exceptions.APIError; carries a status_code."""

    def __init__(self, status_code: int, message: str = "fake api error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self._error = {"message": message}


class _FakeTradingClient:
    """Stand-in for alpaca.trading.client.TradingClient."""

    last_init_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        url_override: str | None = None,
        **_: Any,
    ) -> None:
        type(self).last_init_kwargs = {
            "api_key": api_key,
            "secret_key": secret_key,
            "paper": paper,
            "url_override": url_override,
        }
        self.submit_calls: list[Any] = []
        self.cancel_calls: list[str] = []
        # D-079 §7.2: replace surface — (order_id, ReplaceOrderRequest) pairs.
        self.replace_calls: list[tuple[str, Any]] = []
        # Configurable behaviour; tests mutate via the instance attached to broker.
        self._submit_responses: list[Any] = []
        self._submit_exceptions: list[Exception | None] = []
        self._replace_responses: list[Any] = []
        self._replace_exceptions: list[Exception | None] = []
        # Hotfix 2026-05-07: open-orders surface (KS-5 pending-buy gate).
        self._open_orders_response: list[Any] = []

    # ---- helpers used by tests ----
    def queue_submit_response(self, resp: Any) -> None:
        self._submit_responses.append(resp)
        self._submit_exceptions.append(None)

    def queue_submit_exception(self, exc: Exception) -> None:
        self._submit_responses.append(None)
        self._submit_exceptions.append(exc)

    def queue_replace_response(self, resp: Any) -> None:
        self._replace_responses.append(resp)
        self._replace_exceptions.append(None)

    def queue_replace_exception(self, exc: Exception) -> None:
        self._replace_responses.append(None)
        self._replace_exceptions.append(exc)

    # ---- SDK surface ----
    def submit_order(self, order_data: Any) -> Any:
        self.submit_calls.append(order_data)
        if self._submit_exceptions:
            exc = self._submit_exceptions.pop(0)
            resp = self._submit_responses.pop(0)
            if exc is not None:
                raise exc
            return resp
        return _make_fake_alpaca_order(symbol=order_data.symbol, qty=order_data.qty)

    def cancel_order_by_id(self, order_id: str) -> None:
        self.cancel_calls.append(order_id)

    def replace_order_by_id(self, order_id: str, order_data: Any) -> Any:
        self.replace_calls.append((order_id, order_data))
        if self._replace_exceptions:
            exc = self._replace_exceptions.pop(0)
            resp = self._replace_responses.pop(0)
            if exc is not None:
                raise exc
            return resp
        return _make_fake_alpaca_order(symbol="AAPL", qty=1)

    def get_account(self) -> Any:
        return SimpleNamespace(
            equity="100000.00",
            cash="50000.00",
            buying_power="200000.00",
            long_market_value="50000.00",
            last_equity="99877.00",  # prev trading-day close; used for daypl
        )

    def get_all_positions(self) -> list[Any]:
        return [
            SimpleNamespace(
                symbol="AAPL",
                qty="10",
                avg_entry_price="180.50",
                market_value="1900.00",
                unrealized_pl="95.00",
                side="long",
            )
        ]

    def get_orders(self, filter: Any = None) -> list[Any]:  # noqa: A002
        # Tests can mutate `self._open_orders_response` to simulate
        # pre-market queued orders. Hotfix 2026-05-07.
        return list(self._open_orders_response)


class _FakeDataClient:
    last_init_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        url_override: str | None = None,
        **_: Any,
    ) -> None:
        type(self).last_init_kwargs = {
            "api_key": api_key,
            "secret_key": secret_key,
            "url_override": url_override,
        }
        self._latest_trade_value = 180.02

    def get_stock_latest_quote(self, request: Any) -> Any:
        symbol = request.symbol_or_symbols
        if isinstance(symbol, list):
            symbol = symbol[0]
        return {
            symbol: SimpleNamespace(
                bid_price=180.0,
                ask_price=180.05,
                timestamp=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
            )
        }

    def get_stock_latest_trade(self, request: Any) -> Any:
        symbol = request.symbol_or_symbols
        if isinstance(symbol, list):
            symbol = symbol[0]
        return {
            symbol: SimpleNamespace(
                price=self._latest_trade_value,
                timestamp=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
            )
        }


def _make_fake_alpaca_order(symbol: str, qty: int) -> Any:
    return SimpleNamespace(
        id=uuid.uuid4(),
        client_order_id=f"cid-{symbol}",
        symbol=symbol,
        side=SimpleNamespace(value="buy"),
        qty=str(qty),
        order_type=SimpleNamespace(value="market"),
        type=SimpleNamespace(value="market"),
        status=SimpleNamespace(value="accepted"),
        submitted_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
        limit_price=None,
        stop_price=None,
    )


def _make_fake_bracket_response(
    *,
    symbol: str,
    qty: int,
    entry_order_type: str = "market",
    entry_limit_price: float | None = None,
    stop_price: float = 8.5,
    take_profit_price: float | None = None,
) -> Any:
    """Build a fake alpaca-py parent order with `legs` populated.

    Mirrors what the real SDK returns from a BRACKET / OTO submission:
    the parent (entry) carries the child orders in `.legs`. Tests use
    this to drive the broker's response normalizer without calling out
    over the wire.
    """
    legs: list[Any] = []
    legs.append(
        SimpleNamespace(
            id=uuid.uuid4(),
            client_order_id=f"cid-{symbol}-stop",
            symbol=symbol,
            side=SimpleNamespace(value="sell"),
            qty=str(qty),
            order_type=SimpleNamespace(value="stop"),
            type=SimpleNamespace(value="stop"),
            status=SimpleNamespace(value="held"),
            submitted_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
            limit_price=None,
            stop_price=str(stop_price),
        )
    )
    if take_profit_price is not None:
        legs.append(
            SimpleNamespace(
                id=uuid.uuid4(),
                client_order_id=f"cid-{symbol}-tp",
                symbol=symbol,
                side=SimpleNamespace(value="sell"),
                qty=str(qty),
                order_type=SimpleNamespace(value="limit"),
                type=SimpleNamespace(value="limit"),
                status=SimpleNamespace(value="held"),
                submitted_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
                limit_price=str(take_profit_price),
                stop_price=None,
            )
        )
    return SimpleNamespace(
        id=uuid.uuid4(),
        client_order_id=f"cid-{symbol}",
        symbol=symbol,
        side=SimpleNamespace(value="buy"),
        qty=str(qty),
        order_type=SimpleNamespace(value=entry_order_type),
        type=SimpleNamespace(value=entry_order_type),
        status=SimpleNamespace(value="accepted"),
        submitted_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
        limit_price=str(entry_limit_price) if entry_limit_price is not None else None,
        stop_price=None,
        legs=legs,
    )


def _make_fake_oco_response(
    *,
    symbol: str,
    qty: int,
    take_profit_price: float,
    stop_price: float,
) -> Any:
    """Fake alpaca-py response for a sell-side OCO submission (D-079 §7.3).

    Alpaca returns the take-profit LIMIT order as the parent with the
    stop child in `.legs` — the mirror of the bracket response shape,
    minus the entry."""
    stop_leg = SimpleNamespace(
        id=uuid.uuid4(),
        client_order_id=f"cid-{symbol}-oco-stop",
        symbol=symbol,
        side=SimpleNamespace(value="sell"),
        qty=str(qty),
        order_type=SimpleNamespace(value="stop"),
        type=SimpleNamespace(value="stop"),
        status=SimpleNamespace(value="held"),
        submitted_at=dt.datetime(2026, 6, 9, 14, 30, tzinfo=dt.UTC),
        limit_price=None,
        stop_price=str(stop_price),
    )
    return SimpleNamespace(
        id=uuid.uuid4(),
        client_order_id=f"cid-{symbol}-oco",
        symbol=symbol,
        side=SimpleNamespace(value="sell"),
        qty=str(qty),
        order_type=SimpleNamespace(value="limit"),
        type=SimpleNamespace(value="limit"),
        status=SimpleNamespace(value="accepted"),
        submitted_at=dt.datetime(2026, 6, 9, 14, 30, tzinfo=dt.UTC),
        limit_price=str(take_profit_price),
        stop_price=None,
        legs=[stop_leg],
    )


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the alpaca-py clients + APIError inside broker.alpaca."""
    monkeypatch.setattr(alpaca_mod, "TradingClient", _FakeTradingClient)
    monkeypatch.setattr(alpaca_mod, "StockHistoricalDataClient", _FakeDataClient)
    monkeypatch.setattr(alpaca_mod, "APIError", _FakeAPIError)
    # Make the retry sleep a no-op so tests don't actually wait seconds.
    monkeypatch.setattr(alpaca_mod.time, "sleep", lambda *_: None)


@pytest.fixture
def paper_broker(fake_sdk: None) -> AlpacaBroker:
    return AlpacaBroker(
        mode="paper",
        api_key_id=SecretStr("k"),
        api_secret=SecretStr("s"),
        endpoint=PAPER_ENDPOINT,
    )


@pytest.fixture
def live_broker(fake_sdk: None) -> AlpacaBroker:
    return AlpacaBroker(
        mode="live",
        api_key_id=SecretStr("k"),
        api_secret=SecretStr("s"),
        endpoint=LIVE_ENDPOINT,
    )


# ---------------------------------------------------------------------------
# AC-1: protocol conformance
# ---------------------------------------------------------------------------


def test_protocol_methods_present_on_alpaca(paper_broker: AlpacaBroker) -> None:
    """First failing test: AlpacaBroker satisfies the BrokerAdapter protocol."""
    assert isinstance(paper_broker, BrokerAdapter)
    for method in (
        "submit_order",
        "submit_bracket_order",
        "cancel_order",
        "get_account",
        "get_positions",
        "get_open_orders",
        "get_quote",
    ):
        assert callable(getattr(paper_broker, method))


# ---------------------------------------------------------------------------
# AC-2: endpoint/mode validation (FR-23)
# ---------------------------------------------------------------------------


def test_endpoint_mode_mismatch_rejected_at_init(fake_sdk: None) -> None:
    with pytest.raises(ValueError, match="endpoint.*mode"):
        AlpacaBroker(
            mode="paper",
            api_key_id=SecretStr("k"),
            api_secret=SecretStr("s"),
            endpoint=LIVE_ENDPOINT,
        )
    with pytest.raises(ValueError, match="endpoint.*mode"):
        AlpacaBroker(
            mode="live",
            api_key_id=SecretStr("k"),
            api_secret=SecretStr("s"),
            endpoint=PAPER_ENDPOINT,
        )


def test_paper_endpoint_with_paper_mode_accepted(fake_sdk: None) -> None:
    broker = AlpacaBroker(
        mode="paper",
        api_key_id=SecretStr("k"),
        api_secret=SecretStr("s"),
        endpoint=PAPER_ENDPOINT,
    )
    assert broker.mode == "paper"
    assert _FakeTradingClient.last_init_kwargs["paper"] is True


def test_live_endpoint_with_live_mode_accepted(fake_sdk: None) -> None:
    broker = AlpacaBroker(
        mode="live",
        api_key_id=SecretStr("k"),
        api_secret=SecretStr("s"),
        endpoint=LIVE_ENDPOINT,
    )
    assert broker.mode == "live"
    assert _FakeTradingClient.last_init_kwargs["paper"] is False


# ---------------------------------------------------------------------------
# AC-2: response normalization
# ---------------------------------------------------------------------------


def test_submit_order_response_shape(paper_broker: AlpacaBroker) -> None:
    req = OrderRequest(
        symbol="AAPL",
        side="buy",
        qty=10,
        order_type="market",
        tif="day",
        client_order_id="cid-AAPL",
    )
    out = paper_broker.submit_order(req)
    assert isinstance(out, SubmittedOrder)
    assert out.symbol == "AAPL"
    assert out.qty == 10
    assert out.side == "buy"
    assert out.order_type == "market"
    assert out.status == "accepted"
    assert out.broker_order_id  # non-empty string
    assert out.submitted_at.tzinfo is not None


def test_get_account_normalizes_to_account_snapshot(paper_broker: AlpacaBroker) -> None:
    snap = paper_broker.get_account()
    assert isinstance(snap, AccountSnapshot)
    assert snap.equity == 100_000.0
    assert snap.cash == 50_000.0
    assert snap.buying_power == 200_000.0
    assert snap.long_market_value == 50_000.0
    assert snap.daypl == 123.0  # equity - last_equity
    assert snap.snapshot_at.tzinfo is not None


def test_get_positions_normalizes(paper_broker: AlpacaBroker) -> None:
    positions = paper_broker.get_positions()
    assert len(positions) == 1
    p = positions[0]
    assert isinstance(p, Position)
    assert p.symbol == "AAPL"
    assert p.qty == 10
    assert p.avg_entry_price == 180.5


def test_get_open_orders_normalizes_to_open_order_list(
    paper_broker: AlpacaBroker,
) -> None:
    """Hotfix 2026-05-07: the new `get_open_orders()` adapter method
    must surface broker-queued orders so the KS-5 capacity gate counts
    pre-market entries that haven't filled yet.
    """
    paper_broker._trading._open_orders_response = [
        SimpleNamespace(
            id="ord-1",
            client_order_id="prop-1-entry",
            symbol="ACME",
            side=SimpleNamespace(value="buy"),
            qty="10",
            order_type=SimpleNamespace(value="market"),
            type=SimpleNamespace(value="market"),
            status=SimpleNamespace(value="accepted"),
        ),
        SimpleNamespace(
            id="ord-2",
            client_order_id="prop-2-stop",
            symbol="OLDCO",
            side=SimpleNamespace(value="sell"),
            qty="5",
            order_type=SimpleNamespace(value="stop"),
            type=SimpleNamespace(value="stop"),
            status=SimpleNamespace(value="new"),
        ),
    ]

    orders = paper_broker.get_open_orders()
    assert len(orders) == 2
    assert all(isinstance(o, OpenOrder) for o in orders)

    entry, stop = orders
    assert entry.symbol == "ACME"
    assert entry.side == "buy"
    assert entry.is_entry is True
    assert entry.qty == 10
    assert stop.side == "sell"
    # SELL legs (stops / TPs) are NOT entries — the underlying long is
    # already counted in `get_positions()`.
    assert stop.is_entry is False


def test_quote_returns_bid_ask_last_ts(paper_broker: AlpacaBroker) -> None:
    """AC-6: Quote contains bid, ask, last, ts."""
    q = paper_broker.get_quote("AAPL")
    assert isinstance(q, Quote)
    assert q.symbol == "AAPL"
    assert q.bid == 180.0
    assert q.ask == 180.05
    assert q.last == 180.02
    assert q.ts.tzinfo is not None


def test_cancel_order_passes_id(paper_broker: AlpacaBroker) -> None:
    paper_broker.cancel_order("order-xyz")
    # The fake records the id; reach into the underlying client.
    assert paper_broker._trading.cancel_calls == ["order-xyz"]


# ---------------------------------------------------------------------------
# AC-3: NFR-5 retry on 429/503/network with exponential backoff + jitter
# ---------------------------------------------------------------------------


def test_429_retried_then_succeeds(paper_broker: AlpacaBroker) -> None:
    fake_trading = paper_broker._trading
    fake_trading.queue_submit_exception(_FakeAPIError(429))
    fake_trading.queue_submit_exception(_FakeAPIError(429))
    fake_trading.queue_submit_response(_make_fake_alpaca_order("AAPL", 10))

    req = OrderRequest(
        symbol="AAPL",
        side="buy",
        qty=10,
        order_type="market",
        tif="day",
        client_order_id="cid-AAPL",
    )
    out = paper_broker.submit_order(req)
    assert out.symbol == "AAPL"
    # 2 retries + 1 success = 3 calls total
    assert len(fake_trading.submit_calls) == 3


def test_503_retried_then_succeeds(paper_broker: AlpacaBroker) -> None:
    fake_trading = paper_broker._trading
    fake_trading.queue_submit_exception(_FakeAPIError(503))
    fake_trading.queue_submit_response(_make_fake_alpaca_order("MSFT", 5))

    req = OrderRequest(
        symbol="MSFT",
        side="buy",
        qty=5,
        order_type="market",
        tif="day",
        client_order_id="cid-MSFT",
    )
    out = paper_broker.submit_order(req)
    assert out.symbol == "MSFT"


def test_network_error_retried_then_succeeds(paper_broker: AlpacaBroker) -> None:
    fake_trading = paper_broker._trading
    fake_trading.queue_submit_exception(ConnectionError("network down"))
    fake_trading.queue_submit_response(_make_fake_alpaca_order("AAPL", 1))

    req = OrderRequest(
        symbol="AAPL",
        side="buy",
        qty=1,
        order_type="market",
        tif="day",
        client_order_id="cid-AAPL",
    )
    out = paper_broker.submit_order(req)
    assert out.symbol == "AAPL"


def test_retry_exhausts_to_broker_unavailable(paper_broker: AlpacaBroker) -> None:
    fake_trading = paper_broker._trading
    for _ in range(10):
        fake_trading.queue_submit_exception(_FakeAPIError(429))

    req = OrderRequest(
        symbol="AAPL",
        side="buy",
        qty=1,
        order_type="market",
        tif="day",
        client_order_id="cid-AAPL",
    )
    with pytest.raises(BrokerUnavailable):
        paper_broker.submit_order(req)
    # spec: max 5 attempts (1 initial + 4 retries)
    assert len(fake_trading.submit_calls) == 5


def test_non_retryable_api_error_wrapped_as_broker_rejected(
    paper_broker: AlpacaBroker,
) -> None:
    """Hotfix 2026-05-07: a non-retryable APIError (e.g. 422 insufficient
    buying power, or the wash-trade 40310000 that escaped the production
    handler) must be wrapped as `BrokerRejected` at the broker boundary.

    Raw `APIError` propagation was the proximate cause of the 27
    unprotected positions on first live trading day — the execution
    layer's `BrokerUnavailable`-only `except` let raw rejections fall
    through unjournaled. Wrapping here forces a single domain-typed
    failure mode that callers can handle.
    """
    fake_trading = paper_broker._trading
    fake_trading.queue_submit_exception(_FakeAPIError(422, "insufficient buying power"))

    req = OrderRequest(
        symbol="AAPL",
        side="buy",
        qty=10_000_000,
        order_type="market",
        tif="day",
        client_order_id="cid-AAPL",
    )
    with pytest.raises(BrokerRejected) as ei:
        paper_broker.submit_order(req)
    assert ei.value.status_code == 422
    # Single call — non-retryable should not loop.
    assert len(fake_trading.submit_calls) == 1


def test_wash_trade_apierror_wrapped_as_broker_rejected(
    paper_broker: AlpacaBroker,
) -> None:
    """The exact production failure mode: Alpaca rejects a back-to-back
    stop submission with broker_code=40310000 ("potential wash trade
    detected"). This must surface as `BrokerRejected` so the execution
    layer can route it to the `submission_partial_no_stop` killswitch
    halt (incident 2026-05-07).
    """
    fake_trading = paper_broker._trading
    err = _FakeAPIError(
        422, "potential wash trade detected — opposite side market/stop order exists"
    )
    err._error = {"code": 40310000, "message": str(err)}
    fake_trading.queue_submit_exception(err)

    req = OrderRequest(
        symbol="ACME",
        side="sell",
        qty=10,
        order_type="stop",
        tif="gtc",
        client_order_id="prop-1-stop",
        stop_price=8.50,
    )
    with pytest.raises(BrokerRejected) as ei:
        paper_broker.submit_order(req)
    assert ei.value.status_code == 422
    assert ei.value.broker_code == 40310000
    assert "wash trade" in str(ei.value)


def test_submit_bracket_order_translates_correctly(
    paper_broker: AlpacaBroker,
) -> None:
    """Hotfix 2026-05-07: a `BracketOrderRequest` with both stop and TP
    must translate to a `MarketOrderRequest` with `order_class=BRACKET`,
    `stop_loss=StopLossRequest(...)`, and `take_profit=TakeProfitRequest(...)`.
    The adapter splits the parent's `legs` into stop + tp `SubmittedOrder`s.
    """
    from alpaca.trading.enums import OrderClass
    from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

    fake_trading = paper_broker._trading
    fake_trading.queue_submit_response(
        _make_fake_bracket_response(
            symbol="ACME",
            qty=10,
            entry_order_type="market",
            stop_price=8.5,
            take_profit_price=12.0,
        )
    )

    req = BracketOrderRequest(
        entry_symbol="ACME",
        entry_side="buy",
        entry_qty=10,
        entry_order_type="market",
        entry_tif="day",
        entry_client_order_id="prop-1-entry",
        stop_loss_price=8.5,
        take_profit_price=12.0,
    )
    entry, stop, tp = paper_broker.submit_bracket_order(req)

    # The adapter built a single MarketOrderRequest with BRACKET class +
    # StopLoss + TakeProfit children.
    assert len(fake_trading.submit_calls) == 1
    sent = fake_trading.submit_calls[0]
    assert isinstance(sent, MarketOrderRequest)
    assert sent.order_class == OrderClass.BRACKET
    assert isinstance(sent.stop_loss, StopLossRequest)
    assert sent.stop_loss.stop_price == 8.5
    assert isinstance(sent.take_profit, TakeProfitRequest)
    assert sent.take_profit.limit_price == 12.0
    assert sent.client_order_id == "prop-1-entry"

    # Response normalization: parent → entry; legs → stop + tp.
    assert entry.symbol == "ACME"
    assert entry.side == "buy"
    assert entry.order_type == "market"
    assert stop.side == "sell"
    assert stop.order_type == "stop"
    assert stop.stop_price == 8.5
    assert tp is not None
    assert tp.side == "sell"
    assert tp.order_type == "limit"
    assert tp.limit_price == 12.0


def test_submit_oto_order_when_take_profit_absent(
    paper_broker: AlpacaBroker,
) -> None:
    """Hotfix 2026-05-07: when TP is not supplied, the adapter must use
    `OrderClass.OTO` (entry + stop only) — Alpaca's BRACKET requires
    BOTH legs and rejects requests without a take-profit. The returned
    tuple's third element is `None`.
    """
    from alpaca.trading.enums import OrderClass
    from alpaca.trading.requests import MarketOrderRequest, StopLossRequest

    fake_trading = paper_broker._trading
    fake_trading.queue_submit_response(
        _make_fake_bracket_response(
            symbol="ACME",
            qty=10,
            entry_order_type="market",
            stop_price=8.5,
            take_profit_price=None,
        )
    )

    req = BracketOrderRequest(
        entry_symbol="ACME",
        entry_side="buy",
        entry_qty=10,
        entry_order_type="market",
        entry_tif="day",
        entry_client_order_id="prop-2-entry",
        stop_loss_price=8.5,
        take_profit_price=None,
    )
    entry, stop, tp = paper_broker.submit_bracket_order(req)

    sent = fake_trading.submit_calls[0]
    assert isinstance(sent, MarketOrderRequest)
    assert sent.order_class == OrderClass.OTO
    assert isinstance(sent.stop_loss, StopLossRequest)
    assert sent.stop_loss.stop_price == 8.5
    # OTO carries no take-profit on the wire.
    assert sent.take_profit is None

    assert entry.symbol == "ACME"
    assert stop.order_type == "stop"
    assert tp is None


def test_submit_bracket_order_with_limit_entry(
    paper_broker: AlpacaBroker,
) -> None:
    """Bracket entry can also be a limit (when proposal.entry_style='limit').
    The adapter should emit a `LimitOrderRequest` with the same BRACKET
    class + protective leg shape.
    """
    from alpaca.trading.enums import OrderClass
    from alpaca.trading.requests import LimitOrderRequest

    fake_trading = paper_broker._trading
    fake_trading.queue_submit_response(
        _make_fake_bracket_response(
            symbol="ACME",
            qty=10,
            entry_order_type="limit",
            entry_limit_price=9.95,
            stop_price=8.5,
            take_profit_price=12.0,
        )
    )

    req = BracketOrderRequest(
        entry_symbol="ACME",
        entry_side="buy",
        entry_qty=10,
        entry_order_type="limit",
        entry_tif="day",
        entry_client_order_id="prop-3-entry",
        entry_limit_price=9.95,
        stop_loss_price=8.5,
        take_profit_price=12.0,
    )
    paper_broker.submit_bracket_order(req)

    sent = fake_trading.submit_calls[0]
    assert isinstance(sent, LimitOrderRequest)
    assert sent.order_class == OrderClass.BRACKET
    assert sent.limit_price == 9.95


def test_submit_bracket_order_propagates_broker_rejected(
    paper_broker: AlpacaBroker,
) -> None:
    """If Alpaca rejects the whole bracket (e.g. 422 insufficient buying
    power), the adapter must surface `BrokerRejected` — there is no
    partial state to recover.
    """
    fake_trading = paper_broker._trading
    fake_trading.queue_submit_exception(_FakeAPIError(422, "insufficient buying power"))

    req = BracketOrderRequest(
        entry_symbol="ACME",
        entry_side="buy",
        entry_qty=10,
        entry_order_type="market",
        entry_tif="day",
        entry_client_order_id="prop-4-entry",
        stop_loss_price=8.5,
        take_profit_price=12.0,
    )
    with pytest.raises(BrokerRejected) as ei:
        paper_broker.submit_bracket_order(req)
    assert ei.value.status_code == 422


def test_retryable_api_error_still_retries(paper_broker: AlpacaBroker) -> None:
    """Wrapping non-retryable APIErrors must NOT alter the retry path:
    429s still get backoff + retry, eventually succeeding.
    """
    fake_trading = paper_broker._trading
    fake_trading.queue_submit_exception(_FakeAPIError(429))
    fake_trading.queue_submit_exception(_FakeAPIError(429))
    fake_trading.queue_submit_response(_make_fake_alpaca_order("AAPL", 1))

    req = OrderRequest(
        symbol="AAPL",
        side="buy",
        qty=1,
        order_type="market",
        tif="day",
        client_order_id="cid-AAPL",
    )
    out = paper_broker.submit_order(req)
    assert out.symbol == "AAPL"
    assert len(fake_trading.submit_calls) == 3


def test_backoff_uses_full_jitter(
    monkeypatch: pytest.MonkeyPatch, paper_broker: AlpacaBroker
) -> None:
    """The retry helper must call sleep with values bounded by exponential cap."""
    sleep_args: list[float] = []
    monkeypatch.setattr(alpaca_mod.time, "sleep", lambda s: sleep_args.append(s))

    fake_trading = paper_broker._trading
    for _ in range(4):
        fake_trading.queue_submit_exception(_FakeAPIError(429))
    fake_trading.queue_submit_response(_make_fake_alpaca_order("AAPL", 1))

    req = OrderRequest(
        symbol="AAPL",
        side="buy",
        qty=1,
        order_type="market",
        tif="day",
        client_order_id="cid-AAPL",
    )
    paper_broker.submit_order(req)
    # We expect 4 sleeps (one per failed attempt before each retry).
    assert len(sleep_args) == 4
    # Full jitter: each sleep is in [0, min(cap, base * 2**n)].
    for n, s in enumerate(sleep_args):
        upper = min(60.0, 1.0 * (2**n))
        assert 0.0 <= s <= upper, f"sleep[{n}]={s} not in [0, {upper}]"


# ---------------------------------------------------------------------------
# AC-4: NFR-2 budget — submit_order p95 under 10s with fake SDK
# ---------------------------------------------------------------------------


def test_p95_under_10s_with_fake_sdk(paper_broker: AlpacaBroker) -> None:
    req = OrderRequest(
        symbol="AAPL",
        side="buy",
        qty=1,
        order_type="market",
        tif="day",
        client_order_id="cid-AAPL",
    )
    durations: list[float] = []
    for _ in range(20):
        # Re-queue a fresh response each iteration.
        paper_broker._trading.queue_submit_response(_make_fake_alpaca_order("AAPL", 1))
        t0 = time.perf_counter()
        paper_broker.submit_order(req)
        durations.append(time.perf_counter() - t0)
    p95 = statistics.quantiles(durations, n=20)[-1]
    assert p95 < 10.0, f"p95={p95}s exceeds 10s budget"


# ---------------------------------------------------------------------------
# AC-5: FR-23 logical equivalence — paper and live adapters share the surface
# ---------------------------------------------------------------------------


def test_paper_and_live_have_identical_method_signatures(
    paper_broker: AlpacaBroker, live_broker: AlpacaBroker
) -> None:
    for name in (
        "submit_order",
        "submit_bracket_order",
        "cancel_order",
        "get_account",
        "get_positions",
        "get_open_orders",
        "get_quote",
    ):
        assert inspect.signature(getattr(paper_broker, name)) == inspect.signature(
            getattr(live_broker, name)
        ), f"signature divergence on {name}"


def test_paper_and_live_share_underlying_class(
    paper_broker: AlpacaBroker, live_broker: AlpacaBroker
) -> None:
    """FR-23 / D-007: same code path for paper vs. live; only the SDK 'paper' flag differs."""
    assert type(paper_broker) is type(live_broker)
    # The only init delta is the `paper` flag.
    # (Both used the same fake TradingClient — last_init_kwargs is class-level.)
    assert _FakeTradingClient.last_init_kwargs["paper"] is False  # last init was live


# ---------------------------------------------------------------------------
# Sanity: no `if mode == "paper"` branch in submit / cancel / get_*
# ---------------------------------------------------------------------------


def test_no_mode_branch_in_runtime_methods() -> None:
    """D-007 sacred: no per-mode branching inside the broker request methods."""
    src = inspect.getsource(AlpacaBroker)
    # Mode is read only inside __init__; runtime methods must not check it.
    for needle in ("self.mode ==", 'self.mode == "paper"', 'self.mode == "live"'):
        # Allow occurrences inside __init__ only.
        # Strip the __init__ body from the source and search the rest.
        init_idx = src.find("def __init__")
        next_def = src.find("def ", init_idx + 1)
        body_after_init = src[:init_idx] + src[next_def:]
        assert needle not in body_after_init, f"forbidden mode branch found: {needle!r}"


# ---------------------------------------------------------------------------
# PDT-SUNSET-2026-06-04: ADR-009 §3.4 — defensive 403 handler at the chokepoint.
# Tests use a `_FakeAPIError` subclass that ALSO subclasses the real
# `alpaca.common.exceptions.APIError` so `classify_pdt_403`'s isinstance
# check fires. The existing fake-SDK tests above use `_FakeAPIError` (a
# bare Exception) only because their `submit_order` test-paths predate
# the wrap; those tests still pass because non-APIError exceptions
# bypass the wrap's `except APIError` clause and propagate unchanged.
# ---------------------------------------------------------------------------


from alpaca.common.exceptions import APIError as _RealAPIError  # noqa: E402


class _PDTFakeAPIError(_RealAPIError):
    """Real-APIError subclass for the PDT 403 wrap tests."""

    def __init__(
        self, *, status_code: int, code: int | None = None,
        message: str = "fake api error",
    ) -> None:
        Exception.__init__(self, message)
        self._status_code = status_code
        self._code = code

    @property  # type: ignore[override]
    def status_code(self) -> int | None:  # type: ignore[override]
        return self._status_code

    @property  # type: ignore[override]
    def code(self) -> int | None:  # type: ignore[override]
        return self._code


def test_alpaca_submit_order_classifies_pdt_403(
    paper_broker: AlpacaBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-009 §3.4: a 403 day-trade APIError is reraised as PDTBudgetExceeded."""
    from execution.pdt_budget import PDTBudgetExceeded

    # The wrap's `except APIError` clause resolves the name at runtime
    # against `broker.alpaca.APIError`. The fake_sdk fixture replaces
    # that binding with `_FakeAPIError` (bare Exception); restore it to
    # the real class so our PDT subclass is matched.
    monkeypatch.setattr(alpaca_mod, "APIError", _RealAPIError)

    fake_trading = paper_broker._trading
    fake_trading.queue_submit_exception(
        _PDTFakeAPIError(
            status_code=403, code=40310100,
            message="forbidden: client cannot day-trade",
        )
    )
    req = OrderRequest(
        symbol="AAPL", side="sell", qty=5, order_type="market",
        tif="day", client_order_id="cid-pdt-1",
    )
    with pytest.raises(PDTBudgetExceeded):
        paper_broker.submit_order(req)
    assert len(fake_trading.submit_calls) == 1


def test_alpaca_submit_order_passes_through_non_pdt_403(
    paper_broker: AlpacaBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-PDT 403 (e.g. insufficient buying power) does NOT classify as
    PDTBudgetExceeded — it surfaces as `BrokerRejected` (the wash-trade
    hotfix domain type for non-retryable APIErrors). The PDT classifier
    must not false-positive on non-PDT 403s."""
    from broker.protocol import BrokerRejected
    from execution.pdt_budget import PDTBudgetExceeded

    monkeypatch.setattr(alpaca_mod, "APIError", _RealAPIError)

    fake_trading = paper_broker._trading
    fake_trading.queue_submit_exception(
        _PDTFakeAPIError(
            status_code=403, code=40010001,
            message="insufficient buying power",
        )
    )
    req = OrderRequest(
        symbol="AAPL", side="sell", qty=5, order_type="market",
        tif="day", client_order_id="cid-pdt-2",
    )
    with pytest.raises(BrokerRejected) as excinfo:
        paper_broker.submit_order(req)
    assert not isinstance(excinfo.value, PDTBudgetExceeded)


def test_alpaca_normalize_account_reads_pdt_fields(
    paper_broker: AlpacaBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-009 §3.3: `_normalize_account` reads `last_equity` +
    `daytrade_count` defensively from the alpaca account object."""
    fake_acct = SimpleNamespace(
        equity="22300.0",
        cash="5000.0",
        buying_power="44600.0",
        long_market_value="17300.0",
        last_equity="22000.0",
        daytrade_count=2,
    )
    monkeypatch.setattr(
        paper_broker._trading, "get_account", lambda: fake_acct
    )
    snap = paper_broker.get_account()
    assert snap.last_equity == 22_000.0
    assert snap.daytrade_count == 2


def test_alpaca_normalize_account_pdt_fields_defaults(
    paper_broker: AlpacaBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the alpaca account object lacks `last_equity` /
    `daytrade_count` (post-2026-07-06 API removal), defaults survive:
    last_equity falls back to current equity, daytrade_count → 0
    (feature naturally inerts).
    """
    fake_acct = SimpleNamespace(
        equity="100000.0",
        cash="50000.0",
        buying_power="200000.0",
        long_market_value="50000.0",
    )  # no last_equity, no daytrade_count
    monkeypatch.setattr(
        paper_broker._trading, "get_account", lambda: fake_acct
    )
    snap = paper_broker.get_account()
    assert snap.last_equity == 100_000.0  # fell back to equity
    assert snap.daytrade_count == 0


def test_normalize_account_daypl_from_last_equity() -> None:
    """daypl = equity - last_equity (Alpaca's previous trading-day close).

    Regression (2026-07-08 incident): the old code computed daypl from a
    nonexistent `equity_previous_close` attribute, whose getattr fallback
    made daypl structurally 0.0 — KS-1 could never fire.
    """
    fake_acct = SimpleNamespace(
        equity="3475.0",
        cash="1200.0",
        buying_power="6950.0",
        long_market_value="2275.0",
        last_equity="3600.0",
        daytrade_count=0,
    )
    snap = alpaca_mod._normalize_account(fake_acct)
    assert snap.daypl == -125.0
    assert snap.equity == 3475.0
    assert snap.cash == 1200.0
    assert snap.buying_power == 6950.0
    assert snap.long_market_value == 2275.0
    assert snap.last_equity == 3600.0


def test_normalize_account_missing_last_equity_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No `last_equity` on the account object → daypl degrades to 0.0
    AND a WARNING is emitted, so the degradation cannot hide silently."""
    monkeypatch.setattr(alpaca_mod, "_warned_missing_last_equity", False)
    fake_acct = SimpleNamespace(
        equity="3475.0",
        cash="1200.0",
        buying_power="6950.0",
        long_market_value="2275.0",
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="broker.alpaca"):
        snap = alpaca_mod._normalize_account(fake_acct)
    assert snap.daypl == 0.0
    assert snap.last_equity == 3475.0
    warnings = [
        r for r in caplog.records
        if getattr(r, "event", None) == "broker.account.last_equity_missing"
    ]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# D-079 §7.2 — replace_limit_order: atomic PATCH of a live TP limit order.
# Same safety contract as replace_stop_order: raise on failure, original
# order stands. Wire-level assertions per quality-standards (actuator tests
# assert broker-call arguments, not just normalized returns).
# ---------------------------------------------------------------------------


def test_replace_limit_order_sends_patch_with_new_limit_price(
    paper_broker: AlpacaBroker,
) -> None:
    from alpaca.trading.requests import ReplaceOrderRequest

    fake_trading = paper_broker._trading
    fake_trading.queue_replace_response(
        _make_fake_alpaca_order(symbol="ACME", qty=10)
    )

    out = paper_broker.replace_limit_order(
        "broker-tp-001",
        new_limit_price=14.25,
        client_order_id="thesis-adjtp-42",
    )

    assert len(fake_trading.replace_calls) == 1
    order_id, sent = fake_trading.replace_calls[0]
    assert order_id == "broker-tp-001"
    assert isinstance(sent, ReplaceOrderRequest)
    assert sent.limit_price == 14.25
    assert sent.stop_price is None  # limit replace must not carry a stop price
    assert sent.client_order_id == "thesis-adjtp-42"
    assert isinstance(out, SubmittedOrder)


def test_replace_limit_order_propagates_rejection(
    paper_broker: AlpacaBroker,
) -> None:
    """On a definitive rejection the adapter raises BrokerRejected — the
    caller's contract is 'original TP still live, do not proceed'."""
    fake_trading = paper_broker._trading
    fake_trading.queue_replace_exception(
        _FakeAPIError(422, "order is not replaceable")
    )

    with pytest.raises(BrokerRejected):
        paper_broker.replace_limit_order(
            "broker-tp-001",
            new_limit_price=14.25,
            client_order_id="thesis-adjtp-43",
        )


def test_replace_stop_order_sends_patch_with_new_stop_price(
    paper_broker: AlpacaBroker,
) -> None:
    """Wire-level regression for the existing stop replace (previously
    only covered at the adapter-fake level in integration tests)."""
    from alpaca.trading.requests import ReplaceOrderRequest

    fake_trading = paper_broker._trading
    fake_trading.queue_replace_response(
        _make_fake_alpaca_order(symbol="ACME", qty=10)
    )

    paper_broker.replace_stop_order(
        "broker-stop-001",
        new_stop_price=11.80,
        client_order_id="thesis-adjstop-42",
    )

    order_id, sent = fake_trading.replace_calls[0]
    assert order_id == "broker-stop-001"
    assert isinstance(sent, ReplaceOrderRequest)
    assert sent.stop_price == 11.80
    assert sent.limit_price is None
    assert sent.client_order_id == "thesis-adjstop-42"


# ---------------------------------------------------------------------------
# D-079 §7.3 — submit_oco_sell: protective sell pair for an existing
# position. limit_price set → Alpaca OCO (TP limit parent + stop child,
# both GTC, linked). limit_price None → plain GTC stop sell. Consumed by
# WS3's PDTTransitionConverter — the signature is load-bearing.
# ---------------------------------------------------------------------------


def test_submit_oco_sell_sends_oco_class_with_both_legs(
    paper_broker: AlpacaBroker,
) -> None:
    from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
    from alpaca.trading.requests import (
        LimitOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
    )

    fake_trading = paper_broker._trading
    fake_trading.queue_submit_response(
        _make_fake_oco_response(
            symbol="ACME", qty=13, take_profit_price=14.0, stop_price=9.5
        )
    )

    stop, tp = paper_broker.submit_oco_sell(
        symbol="ACME",
        qty=13,
        stop_price=9.5,
        limit_price=14.0,
        client_order_id="pdt-convert-7",
    )

    assert len(fake_trading.submit_calls) == 1
    sent = fake_trading.submit_calls[0]
    assert isinstance(sent, LimitOrderRequest)
    assert sent.order_class == OrderClass.OCO
    assert sent.side == OrderSide.SELL
    assert sent.time_in_force == TimeInForce.GTC
    assert sent.symbol == "ACME"
    assert sent.qty == 13
    assert sent.client_order_id == "pdt-convert-7"
    assert isinstance(sent.take_profit, TakeProfitRequest)
    assert sent.take_profit.limit_price == 14.0
    assert isinstance(sent.stop_loss, StopLossRequest)
    assert sent.stop_loss.stop_price == 9.5

    # Response split: stop child + TP parent, both normalized.
    assert stop.order_type == "stop"
    assert stop.stop_price == 9.5
    assert tp is not None
    assert tp.order_type == "limit"
    assert tp.limit_price == 14.0


def test_submit_oco_sell_without_limit_price_sends_plain_gtc_stop(
    paper_broker: AlpacaBroker,
) -> None:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import StopOrderRequest

    fake_trading = paper_broker._trading
    fake_trading.queue_submit_response(
        SimpleNamespace(
            id=uuid.uuid4(),
            client_order_id="pdt-convert-8",
            symbol="ACME",
            side=SimpleNamespace(value="sell"),
            qty="13",
            order_type=SimpleNamespace(value="stop"),
            type=SimpleNamespace(value="stop"),
            status=SimpleNamespace(value="accepted"),
            submitted_at=dt.datetime(2026, 6, 9, 14, 30, tzinfo=dt.UTC),
            limit_price=None,
            stop_price="9.5",
        )
    )

    stop, tp = paper_broker.submit_oco_sell(
        symbol="ACME",
        qty=13,
        stop_price=9.5,
        limit_price=None,
        client_order_id="pdt-convert-8",
    )

    sent = fake_trading.submit_calls[0]
    assert isinstance(sent, StopOrderRequest)
    assert sent.side == OrderSide.SELL
    assert sent.time_in_force == TimeInForce.GTC
    assert sent.stop_price == 9.5
    assert sent.client_order_id == "pdt-convert-8"
    assert stop.order_type == "stop"
    assert stop.stop_price == 9.5
    assert tp is None


def test_submit_oco_sell_missing_stop_leg_raises(
    paper_broker: AlpacaBroker,
) -> None:
    """Defense in depth mirroring submit_bracket_order: an OCO response
    without the stop child means the broker state is unknown — surface
    BrokerRejected rather than claim success without a stop."""
    fake_trading = paper_broker._trading
    resp = _make_fake_oco_response(
        symbol="ACME", qty=13, take_profit_price=14.0, stop_price=9.5
    )
    resp.legs = []  # broker returned the TP parent but no stop child
    fake_trading.queue_submit_response(resp)

    with pytest.raises(BrokerRejected, match="stop leg missing"):
        paper_broker.submit_oco_sell(
            symbol="ACME",
            qty=13,
            stop_price=9.5,
            limit_price=14.0,
            client_order_id="pdt-convert-9",
        )


def test_submit_oco_sell_propagates_rejection(
    paper_broker: AlpacaBroker,
) -> None:
    fake_trading = paper_broker._trading
    fake_trading.queue_submit_exception(
        _FakeAPIError(422, "insufficient qty available")
    )

    with pytest.raises(BrokerRejected):
        paper_broker.submit_oco_sell(
            symbol="ACME",
            qty=13,
            stop_price=9.5,
            limit_price=14.0,
            client_order_id="pdt-convert-10",
        )
