"""AlpacaBroker — concrete BrokerAdapter wrapping the alpaca-py SDK.

Implements architecture §2.6 and ADR-001 (no in-process simulator). Single
code path for paper and live (D-007, FR-23): mode and endpoint pair must be
consistent at construction; runtime methods do not branch on mode.
"""

from __future__ import annotations

import datetime as dt
import random
import time
from typing import Any, Literal, TypeVar

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
)
from pydantic import SecretStr

from observability.log_port import get_logger

from .protocol import (
    AccountSnapshot,
    OrderRequest,
    Position,
    Quote,
    SubmittedOrder,
)

log = get_logger(__name__)

PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
LIVE_ENDPOINT = "https://api.alpaca.markets"

_RETRY_BASE_SECONDS = 1.0
_RETRY_CAP_SECONDS = 60.0
_RETRY_MAX_ATTEMPTS = 5

T = TypeVar("T")


class BrokerUnavailable(Exception):
    """Raised when retries are exhausted on a transient broker failure (NFR-5)."""


class _EndpointModeMismatch(ValueError):
    """Raised at init when the configured endpoint does not match mode."""


def _is_retryable(exc: BaseException) -> bool:
    """429 / 503 / network errors are retryable; everything else propagates."""
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        return status in (429, 503)
    if isinstance(exc, OSError):
        # ConnectionError, TimeoutError, socket errors, etc.
        return True
    return False


def _retry(callable_: Any, *args: Any, **kwargs: Any) -> Any:
    """Run `callable_` with exponential backoff + full jitter on transient errors.

    Spec: base 1s, cap 60s, max 5 attempts. On exhaustion, raise BrokerUnavailable
    chained from the last error (NFR-5).
    """
    last_exc: BaseException | None = None
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            return callable_(*args, **kwargs)
        except BaseException as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc
            if attempt == _RETRY_MAX_ATTEMPTS - 1:
                break
            upper = min(_RETRY_CAP_SECONDS, _RETRY_BASE_SECONDS * (2**attempt))
            time.sleep(random.uniform(0.0, upper))
    raise BrokerUnavailable(
        f"alpaca request failed after {_RETRY_MAX_ATTEMPTS} attempts: {last_exc}"
    ) from last_exc


_ORDER_SIDE_MAP = {"buy": OrderSide.BUY, "sell": OrderSide.SELL}
_TIF_MAP = {
    "day": TimeInForce.DAY,
    "gtc": TimeInForce.GTC,
    "ioc": TimeInForce.IOC,
    "fok": TimeInForce.FOK,
    "opg": TimeInForce.OPG,
    "cls": TimeInForce.CLS,
}


def _to_alpaca_request(req: OrderRequest) -> Any:
    side = _ORDER_SIDE_MAP[req.side]
    tif = _TIF_MAP[req.tif]
    common = {
        "symbol": req.symbol,
        "qty": req.qty,
        "side": side,
        "time_in_force": tif,
        "client_order_id": req.client_order_id,
        "extended_hours": req.extended_hours,
    }
    if req.order_type == "market":
        return MarketOrderRequest(**common)
    if req.order_type == "limit":
        return LimitOrderRequest(limit_price=req.limit_price, **common)
    if req.order_type == "stop":
        return StopOrderRequest(stop_price=req.stop_price, **common)
    if req.order_type == "stop_limit":
        return StopLimitOrderRequest(
            limit_price=req.limit_price, stop_price=req.stop_price, **common
        )
    raise ValueError(f"unknown order_type: {req.order_type}")


def _enum_value(v: Any) -> Any:
    return v.value if hasattr(v, "value") else v


def _normalize_submitted(alp: Any) -> SubmittedOrder:
    raw_type = _enum_value(getattr(alp, "order_type", None) or getattr(alp, "type", None))
    return SubmittedOrder(
        broker_order_id=str(alp.id),
        client_order_id=str(alp.client_order_id),
        symbol=str(alp.symbol),
        side=_enum_value(alp.side),
        qty=int(float(alp.qty)),
        order_type=raw_type,
        status=_enum_value(alp.status),
        submitted_at=alp.submitted_at,
        limit_price=float(alp.limit_price) if alp.limit_price is not None else None,
        stop_price=float(alp.stop_price) if alp.stop_price is not None else None,
    )


def _normalize_account(acct: Any) -> AccountSnapshot:
    equity = float(acct.equity)
    prev_close = float(getattr(acct, "equity_previous_close", equity) or equity)
    return AccountSnapshot(
        equity=equity,
        cash=float(acct.cash),
        buying_power=float(acct.buying_power),
        long_market_value=float(acct.long_market_value),
        daypl=equity - prev_close,
        snapshot_at=dt.datetime.now(dt.UTC),
    )


def _normalize_position(pos: Any) -> Position:
    return Position(
        symbol=str(pos.symbol),
        qty=int(float(pos.qty)),
        avg_entry_price=float(pos.avg_entry_price),
        market_value=float(pos.market_value),
        unrealized_pnl=float(pos.unrealized_pl),
    )


class AlpacaBroker:
    """BrokerAdapter implementation backed by alpaca-py.

    Construction validates the mode↔endpoint pair (FR-23). Runtime methods do
    not inspect `self.mode` — paper and live are interchangeable behind the
    same code path (D-007).
    """

    def __init__(
        self,
        *,
        mode: Literal["paper", "live"],
        api_key_id: SecretStr,
        api_secret: SecretStr,
        endpoint: str,
    ) -> None:
        if mode == "paper" and endpoint != PAPER_ENDPOINT:
            raise _EndpointModeMismatch(
                f"endpoint {endpoint!r} does not match mode={mode!r} "
                f"(expected {PAPER_ENDPOINT!r})"
            )
        if mode == "live" and endpoint != LIVE_ENDPOINT:
            raise _EndpointModeMismatch(
                f"endpoint {endpoint!r} does not match mode={mode!r} "
                f"(expected {LIVE_ENDPOINT!r})"
            )

        self.mode: Literal["paper", "live"] = mode
        self.endpoint: str = endpoint
        self._trading = TradingClient(
            api_key=api_key_id.get_secret_value(),
            secret_key=api_secret.get_secret_value(),
            paper=(mode == "paper"),
        )
        self._data = StockHistoricalDataClient(
            api_key=api_key_id.get_secret_value(),
            secret_key=api_secret.get_secret_value(),
        )

    # ------------------------------------------------------------------
    # BrokerAdapter surface
    # ------------------------------------------------------------------

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        alpaca_req = _to_alpaca_request(req)
        raw = _retry(self._trading.submit_order, alpaca_req)
        return _normalize_submitted(raw)

    def cancel_order(self, broker_order_id: str) -> None:
        _retry(self._trading.cancel_order_by_id, broker_order_id)

    def get_account(self) -> AccountSnapshot:
        raw = _retry(self._trading.get_account)
        return _normalize_account(raw)

    def get_positions(self) -> list[Position]:
        raw = _retry(self._trading.get_all_positions)
        return [_normalize_position(p) for p in raw]

    def get_quote(self, symbol: str) -> Quote:
        quote_req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        trade_req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        quotes = _retry(self._data.get_stock_latest_quote, quote_req)
        trades = _retry(self._data.get_stock_latest_trade, trade_req)
        q = quotes[symbol]
        t = trades[symbol]
        return Quote(
            symbol=symbol,
            bid=float(q.bid_price),
            ask=float(q.ask_price),
            last=float(t.price),
            ts=q.timestamp,
        )

    def get_order_by_client_id(
        self, client_order_id: str
    ) -> SubmittedOrder | None:
        """S5.6 carry-fwd S6.4 reviewer M-4 — orphan-order lookup.

        Alpaca's `get_order_by_client_id` returns the order if any was
        ever submitted with that id (deterministic per-proposal in v1
        — `prop-{id}-{role}`); 404s when none exists. Treat 404 as
        None per the BrokerAdapter contract.
        """
        try:
            raw = _retry(self._trading.get_order_by_client_id, client_order_id)
        except APIError as e:
            if getattr(e, "status_code", None) == 404:
                return None
            raise
        if raw is None:
            return None
        return _normalize_submitted(raw)


__all__ = [
    "AlpacaBroker",
    "BrokerUnavailable",
    "LIVE_ENDPOINT",
    "PAPER_ENDPOINT",
]
