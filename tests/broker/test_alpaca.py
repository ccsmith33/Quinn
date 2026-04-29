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
    BrokerAdapter,
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
        # Configurable behaviour; tests mutate via the instance attached to broker.
        self._submit_responses: list[Any] = []
        self._submit_exceptions: list[Exception | None] = []

    # ---- helpers used by tests ----
    def queue_submit_response(self, resp: Any) -> None:
        self._submit_responses.append(resp)
        self._submit_exceptions.append(None)

    def queue_submit_exception(self, exc: Exception) -> None:
        self._submit_responses.append(None)
        self._submit_exceptions.append(exc)

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

    def get_account(self) -> Any:
        return SimpleNamespace(
            equity="100000.00",
            cash="50000.00",
            buying_power="200000.00",
            long_market_value="50000.00",
            equity_previous_close="99877.00",  # used for daypl computation
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
    for method in ("submit_order", "cancel_order", "get_account", "get_positions", "get_quote"):
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
    assert snap.snapshot_at.tzinfo is not None


def test_get_positions_normalizes(paper_broker: AlpacaBroker) -> None:
    positions = paper_broker.get_positions()
    assert len(positions) == 1
    p = positions[0]
    assert isinstance(p, Position)
    assert p.symbol == "AAPL"
    assert p.qty == 10
    assert p.avg_entry_price == 180.5


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


def test_non_retryable_error_propagates_unwrapped(paper_broker: AlpacaBroker) -> None:
    """A 4xx that isn't 429 (e.g. 422 insufficient buying power) must NOT be retried."""
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
    with pytest.raises(_FakeAPIError):
        paper_broker.submit_order(req)
    assert len(fake_trading.submit_calls) == 1


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
    for name in ("submit_order", "cancel_order", "get_account", "get_positions", "get_quote"):
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
