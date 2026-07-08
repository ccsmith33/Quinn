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
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
)
from pydantic import SecretStr

from observability.log_port import get_logger

from .protocol import (
    AccountSnapshot,
    BracketOrderRequest,
    BrokerRejected,
    OpenOrder,
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


def _wrap_api_error(exc: APIError) -> BrokerRejected:
    """Translate a non-retryable alpaca-py `APIError` into our domain
    exception so callers depend only on `broker.protocol`.

    Hotfix 2026-05-07 (incident: 27 unprotected positions): a wash-trade
    rejection (status 422, broker code 40310000) on a back-to-back stop
    submission was propagating raw out of the broker abstraction and
    escaping the `OrderSubmitter`'s `BrokerUnavailable` handler entirely
    — the position stayed live without protection AND without a journal
    row. Wrapping at the broker boundary lets execution catch a single
    union (`BrokerRejected | BrokerUnavailable`) and route both to the
    `submission_partial_no_stop` killswitch halt + journal write.
    """
    status = getattr(exc, "status_code", None)
    # alpaca-py packs the broker-specific code in the `_error` dict
    # (e.g. {"code": 40310000, "message": "..."}); fall back to None.
    err_payload = getattr(exc, "_error", None)
    broker_code: int | str | None = None
    if isinstance(err_payload, dict):
        broker_code = err_payload.get("code")
    return BrokerRejected(
        str(exc),
        status_code=status,
        broker_code=broker_code,
    )


def _retry(callable_: Any, *args: Any, **kwargs: Any) -> Any:
    """Run `callable_` with exponential backoff + full jitter on transient errors.

    Spec: base 1s, cap 60s, max 5 attempts. On exhaustion, raise BrokerUnavailable
    chained from the last error (NFR-5).

    Non-retryable `APIError`s (anything that isn't 429/503) are wrapped as
    `BrokerRejected` so they cross the broker boundary in a domain type
    rather than the raw alpaca-py SDK exception (incident 2026-05-07).
    Non-`APIError` exceptions still propagate raw.
    """
    last_exc: BaseException | None = None
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            return callable_(*args, **kwargs)
        except BaseException as exc:
            if not _is_retryable(exc):
                if isinstance(exc, APIError):
                    raise _wrap_api_error(exc) from exc
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


def _to_alpaca_bracket_request(req: BracketOrderRequest) -> Any:
    """Translate a `BracketOrderRequest` into an alpaca-py order request
    with the appropriate `order_class` (BRACKET or OTO).

    Hotfix 2026-05-07 (incident: 27 unprotected positions): bracket /
    OTO orders eliminate the wash-trade race that occurred when entry
    + stop were submitted as separate calls pre-market. Alpaca treats
    the bundle as a single transaction, so the back-to-back wash-trade
    detector (40310000) does not fire on the protective leg.

    Class selection:
      - take_profit_price set    → BRACKET (entry parent + stop child + tp child)
      - take_profit_price absent → OTO     (entry parent + stop child)
    """
    side = _ORDER_SIDE_MAP[req.entry_side]
    tif = _TIF_MAP[req.entry_tif]
    if req.take_profit_price is not None:
        order_class = OrderClass.BRACKET
        take_profit = TakeProfitRequest(limit_price=req.take_profit_price)
        stop_loss = StopLossRequest(stop_price=req.stop_loss_price)
    else:
        order_class = OrderClass.OTO
        take_profit = None
        stop_loss = StopLossRequest(stop_price=req.stop_loss_price)

    common: dict[str, Any] = {
        "symbol": req.entry_symbol,
        "qty": req.entry_qty,
        "side": side,
        "time_in_force": tif,
        "client_order_id": req.entry_client_order_id,
        "extended_hours": req.entry_extended_hours,
        "order_class": order_class,
        "stop_loss": stop_loss,
    }
    if take_profit is not None:
        common["take_profit"] = take_profit

    if req.entry_order_type == "market":
        return MarketOrderRequest(**common)
    if req.entry_order_type == "limit":
        return LimitOrderRequest(limit_price=req.entry_limit_price, **common)
    raise ValueError(
        f"bracket entry order_type must be market or limit, got {req.entry_order_type!r}"
    )


def _enum_value(v: Any) -> Any:
    return v.value if hasattr(v, "value") else v


def _normalize_submitted(alp: Any) -> SubmittedOrder:
    raw_type = _enum_value(getattr(alp, "order_type", None) or getattr(alp, "type", None))
    # WS1 (D-078): fill detail for the FillIngestor's get_order_by_id poll.
    # getattr defaults keep submission-time responses (and older fakes)
    # working — those orders simply haven't filled yet.
    filled_avg_price = getattr(alp, "filled_avg_price", None)
    filled_qty = getattr(alp, "filled_qty", None)
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
        filled_avg_price=(
            float(filled_avg_price) if filled_avg_price is not None else None
        ),
        filled_qty=int(float(filled_qty)) if filled_qty is not None else 0,
        filled_at=getattr(alp, "filled_at", None),
    )


_warned_missing_last_equity = False


def _normalize_account(acct: Any) -> AccountSnapshot:
    global _warned_missing_last_equity
    equity = float(acct.equity)
    # PDT-SUNSET-2026-06-04: ADR-009 §3.3 — defensive reads with
    # defaults that survive Alpaca's planned 2026-07-06 field removal.
    # `last_equity` (equity at previous trading-day close) falls back to
    # current equity (keeps `< 25k` predicate functioning on intraday
    # equity in the fallback case) — but that degrades daypl to 0.0 and
    # blinds KS-1, so warn once instead of failing silently;
    # `daytrade_count` falls back to 0 (which makes `budget_remaining=3`
    # always — feature naturally inert post-API-removal).
    raw_last_equity = getattr(acct, "last_equity", None)
    if raw_last_equity:
        last_equity = float(raw_last_equity)
    else:
        last_equity = equity
        if not _warned_missing_last_equity:
            _warned_missing_last_equity = True
            log.warning(
                "broker.account.last_equity_missing",
                extra={
                    "event": "broker.account.last_equity_missing",
                    "detail": "daypl degraded to 0.0; KS-1 daily-loss "
                    "halt is blind until last_equity is restored",
                },
            )
    daytrade_count = int(getattr(acct, "daytrade_count", 0) or 0)
    return AccountSnapshot(
        equity=equity,
        cash=float(acct.cash),
        buying_power=float(acct.buying_power),
        long_market_value=float(acct.long_market_value),
        daypl=equity - last_equity,
        snapshot_at=dt.datetime.now(dt.UTC),
        last_equity=last_equity,
        daytrade_count=daytrade_count,
    )


def _normalize_position(pos: Any) -> Position:
    return Position(
        symbol=str(pos.symbol),
        qty=int(float(pos.qty)),
        avg_entry_price=float(pos.avg_entry_price),
        market_value=float(pos.market_value),
        unrealized_pnl=float(pos.unrealized_pl),
    )


def _normalize_open_order(o: Any) -> OpenOrder:
    raw_type = _enum_value(getattr(o, "order_type", None) or getattr(o, "type", None))
    return OpenOrder(
        symbol=str(o.symbol),
        side=_enum_value(o.side),
        qty=int(float(o.qty)),
        order_type=raw_type,
        status=_enum_value(o.status),
        broker_order_id=str(o.id),
        client_order_id=str(o.client_order_id),
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
        # PDT-SUNSET-2026-06-04: ADR-009 §3.4 — classify Alpaca's 403
        # PDT rejection at the chokepoint and raise PDTBudgetExceeded
        # so VirtualExitScanner can route the failed sell to
        # `deferred_sells`. Non-PDT errors propagate as `BrokerRejected`
        # (wash-trade hotfix 2026-05-07) — `_retry` already wraps
        # non-retryable APIErrors before they reach this catch, so we
        # inspect `__cause__` to recover the original APIError for
        # classification.
        # Local import to avoid a module-level cycle: pdt_budget imports
        # `BrokerUnavailable` from this module.
        from execution.pdt_budget import PDTBudgetExceeded, classify_pdt_403

        try:
            raw = _retry(self._trading.submit_order, alpaca_req)
        except BrokerRejected as e:
            cause = e.__cause__
            if isinstance(cause, APIError) and classify_pdt_403(cause):
                raise PDTBudgetExceeded(str(cause)) from cause
            raise
        except APIError as e:
            # Defense-in-depth: should not occur post-hotfix (`_retry`
            # wraps APIError → BrokerRejected), but a future change to
            # `_retry` could re-expose APIError here. Keep the PDT
            # classifier reachable on the raw form too.
            if classify_pdt_403(e):
                raise PDTBudgetExceeded(str(e)) from e
            raise
        return _normalize_submitted(raw)

    def submit_bracket_order(
        self, req: BracketOrderRequest
    ) -> tuple[SubmittedOrder, SubmittedOrder, SubmittedOrder | None]:
        """Submit entry + stop (+ optional TP) as a single atomic complex
        order (BRACKET / OTO).

        Alpaca returns the parent (entry) with its `legs` field populated
        with the child orders. We split them by side / order_type:
          - the SELL stop child  → stop leg
          - the SELL limit child → take-profit leg

        Hotfix 2026-05-07: a partial state ("entry placed, stop rejected")
        is impossible by the broker's contract — either all legs are
        accepted or the whole submission is rejected. Failure surfaces as
        `BrokerRejected` / `BrokerUnavailable` and the caller treats it as
        a clean `submission_failed` (no exposure, no journal order rows).
        """
        alpaca_req = _to_alpaca_bracket_request(req)
        raw = _retry(self._trading.submit_order, alpaca_req)
        entry = _normalize_submitted(raw)

        legs = list(getattr(raw, "legs", None) or [])
        stop_raw = next(
            (
                leg
                for leg in legs
                if _enum_value(leg.side) == "sell"
                and _enum_value(
                    getattr(leg, "order_type", None) or getattr(leg, "type", None)
                )
                == "stop"
            ),
            None,
        )
        if stop_raw is None:
            # Defense in depth: alpaca-py contract says BRACKET/OTO always
            # populates the stop leg. If it's missing the parent is in an
            # unknown state — better to surface as BrokerRejected than to
            # claim success with no stop.
            raise BrokerRejected(
                "bracket submission accepted but stop leg missing from response",
                status_code=None,
                broker_code=None,
            )
        stop = _normalize_submitted(stop_raw)

        tp: SubmittedOrder | None = None
        if req.take_profit_price is not None:
            tp_raw = next(
                (
                    leg
                    for leg in legs
                    if _enum_value(leg.side) == "sell"
                    and _enum_value(
                        getattr(leg, "order_type", None) or getattr(leg, "type", None)
                    )
                    == "limit"
                ),
                None,
            )
            if tp_raw is None:
                raise BrokerRejected(
                    "bracket submission accepted but take-profit leg missing from response",
                    status_code=None,
                    broker_code=None,
                )
            tp = _normalize_submitted(tp_raw)

        return entry, stop, tp

    def cancel_order(self, broker_order_id: str) -> None:
        _retry(self._trading.cancel_order_by_id, broker_order_id)

    def get_account(self) -> AccountSnapshot:
        raw = _retry(self._trading.get_account)
        return _normalize_account(raw)

    def get_positions(self) -> list[Position]:
        raw = _retry(self._trading.get_all_positions)
        return [_normalize_position(p) for p in raw]

    def get_open_orders(self) -> list[OpenOrder]:
        """Return broker orders that are currently open / pending fill.

        Alpaca's `QueryOrderStatus.OPEN` covers the lifecycle states that
        haven't reached a terminal disposition (filled / canceled /
        expired / rejected): accepted, new, partially_filled, pending_new,
        accepted_for_bidding. We rely on the broker's own classification
        rather than filtering client-side, both because the SDK does the
        filtering for free and because the set of "open" statuses is the
        broker's contract, not ours.

        Hotfix 2026-05-07: production saw 28 entries opened in one
        pre-market session because each new sizing check saw 0 open
        positions (broker hadn't filled them yet). KS-5 now consults
        `len(positions ∪ open_buy_orders by symbol)`.
        """
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        raw = _retry(self._trading.get_orders, filter=req)
        return [_normalize_open_order(o) for o in raw]

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
        except BrokerRejected as e:
            # 404 is "no order with that client_id" — the v1 orphan-order
            # contract treats this as None, not an error. Other rejections
            # (auth failure, malformed id) still propagate.
            if e.status_code == 404:
                return None
            raise
        if raw is None:
            return None
        return _normalize_submitted(raw)

    def get_order_by_id(self, broker_order_id: str) -> SubmittedOrder | None:
        """WS1 (D-078, delta §7.1, ADR-010) — FillIngestor poll target.

        GET /v2/orders/{id}; the response carries `filled_avg_price` /
        `filled_qty` / `filled_at`, normalized into the SubmittedOrder
        fill fields. 404 → None per the BrokerAdapter contract (an order
        the broker has purged is not a transport error).
        """
        try:
            raw = _retry(self._trading.get_order_by_id, broker_order_id)
        except BrokerRejected as e:
            if e.status_code == 404:
                return None
            raise
        if raw is None:
            return None
        return _normalize_submitted(raw)

    def replace_stop_order(
        self,
        broker_order_id: str,
        *,
        new_stop_price: float,
        client_order_id: str,
    ) -> SubmittedOrder:
        """Feature A — atomic stop-price replacement via Alpaca's
        PATCH /v2/orders/{id}. The broker either accepts the new
        stop_price (returning a new order with the same protective leg
        intact) or rejects, leaving the original stop live. There is
        no uncovered window."""
        replace_req = ReplaceOrderRequest(
            stop_price=new_stop_price,
            client_order_id=client_order_id,
        )
        raw = _retry(
            self._trading.replace_order_by_id, broker_order_id, replace_req
        )
        return _normalize_submitted(raw)

    def replace_limit_order(
        self,
        broker_order_id: str,
        *,
        new_limit_price: float,
        client_order_id: str,
    ) -> SubmittedOrder:
        """D-079 §7.2 — atomic limit-price replacement (the TP-raise
        actuator). Same PATCH semantics as `replace_stop_order`: accept
        means a new order id with no uncovered window; reject leaves
        the original take-profit live."""
        replace_req = ReplaceOrderRequest(
            limit_price=new_limit_price,
            client_order_id=client_order_id,
        )
        raw = _retry(
            self._trading.replace_order_by_id, broker_order_id, replace_req
        )
        return _normalize_submitted(raw)

    def submit_oco_sell(
        self,
        *,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float | None,
        client_order_id: str,
    ) -> tuple[SubmittedOrder, SubmittedOrder | None]:
        """D-079 §7.3 — protective sell pair for an existing position.

        limit_price set  → Alpaca OCO: the take-profit LIMIT order is the
                           parent, the stop child arrives in `legs`. One
                           leg filling cancels the other broker-side.
        limit_price None → plain GTC stop sell (no OCO wrapper needed).

        Wire shape per Alpaca's OCO contract: `order_class=oco`,
        `type=limit`, `take_profit.limit_price` + `stop_loss.stop_price`;
        no top-level limit_price (the take_profit object drives the TP
        leg). Mirrors `submit_bracket_order`'s defense-in-depth: a
        response missing the stop child surfaces as BrokerRejected
        rather than claiming success without a stop.
        """
        if limit_price is None:
            stop_req = OrderRequest(
                symbol=symbol,
                side="sell",
                qty=qty,
                order_type="stop",
                tif="gtc",
                client_order_id=client_order_id,
                stop_price=stop_price,
            )
            return self.submit_order(stop_req), None

        oco_req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.OCO,
            take_profit=TakeProfitRequest(limit_price=limit_price),
            stop_loss=StopLossRequest(stop_price=stop_price),
            client_order_id=client_order_id,
        )
        raw = _retry(self._trading.submit_order, oco_req)
        tp = _normalize_submitted(raw)

        legs = list(getattr(raw, "legs", None) or [])
        stop_raw = next(
            (
                leg
                for leg in legs
                if _enum_value(leg.side) == "sell"
                and _enum_value(
                    getattr(leg, "order_type", None) or getattr(leg, "type", None)
                )
                == "stop"
            ),
            None,
        )
        if stop_raw is None:
            raise BrokerRejected(
                "oco submission accepted but stop leg missing from response",
                status_code=None,
                broker_code=None,
            )
        stop = _normalize_submitted(stop_raw)
        return stop, tp


__all__ = [
    "AlpacaBroker",
    "BrokerRejected",
    "BrokerUnavailable",
    "LIVE_ENDPOINT",
    "PAPER_ENDPOINT",
]
