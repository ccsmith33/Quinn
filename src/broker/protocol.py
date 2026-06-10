"""BrokerAdapter Protocol + domain types (architecture §2.6, FR-23/FR-25/FR-26).

Single code path for paper and live (D-007). Concrete implementations differ
only in credentials and endpoint; all callers depend on this Protocol.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TIF = Literal["day", "gtc", "ioc", "fok", "opg", "cls"]
OrderStatus = Literal[
    "accepted",
    "new",
    "partially_filled",
    "filled",
    "done_for_day",
    "canceled",
    "expired",
    "replaced",
    "pending_cancel",
    "pending_replace",
    "rejected",
    "suspended",
    "pending_new",
    "calculated",
    "stopped",
    "held",
]


class _Domain(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrderRequest(_Domain):
    """Domain-level order spec submitted via the broker adapter."""

    symbol: str
    side: Side
    qty: int = Field(gt=0)
    order_type: OrderType
    tif: TIF
    client_order_id: str
    limit_price: float | None = None
    stop_price: float | None = None
    extended_hours: bool = False

    @model_validator(mode="after")
    def _check_prices(self) -> OrderRequest:
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit_price required for order_type=limit")
        if self.order_type == "stop" and self.stop_price is None:
            raise ValueError("stop_price required for order_type=stop")
        if self.order_type == "stop_limit" and (
            self.limit_price is None or self.stop_price is None
        ):
            raise ValueError("limit_price and stop_price required for order_type=stop_limit")
        return self


class BracketOrderRequest(_Domain):
    """Domain-level bracket / OTO order spec.

    Hotfix 2026-05-07 (incident: 27 unprotected positions): Alpaca rejects a
    sell-stop submitted back-to-back with a still-pending entry buy as a
    wash trade (broker code 40310000). Alpaca's recommended remedy in the
    error message itself is "use complex orders": entry + protective legs
    submitted as a single atomic order with the broker creating the
    children when the entry fills. This eliminates the partial-state
    failure mode entirely — either the broker accepts the bracket (entry +
    stop + optional TP all created) or rejects it (no exposure).

    `take_profit_price` is optional. Alpaca's BRACKET class requires both
    legs; when TP is omitted we fall back to OTO (one-triggers-other,
    entry + stop only). The broker adapter chooses the wire-level class
    from the presence/absence of `take_profit_price`.

    Only the entry leg carries a `client_order_id`: Alpaca generates child
    cids for the protective legs (BRACKET / OTO leg requests don't accept
    one). Child broker order ids surface in the response and are recorded
    on the journal `OrderRow`s.
    """

    entry_symbol: str
    entry_side: Side
    entry_qty: int = Field(gt=0)
    entry_order_type: OrderType
    entry_tif: TIF
    entry_client_order_id: str
    entry_limit_price: float | None = None
    entry_extended_hours: bool = False

    stop_loss_price: float
    take_profit_price: float | None = None

    @model_validator(mode="after")
    def _check(self) -> BracketOrderRequest:
        if self.entry_order_type == "limit" and self.entry_limit_price is None:
            raise ValueError("entry_limit_price required for entry_order_type=limit")
        if self.entry_order_type not in ("market", "limit"):
            raise ValueError(
                "bracket entries must be order_type 'market' or 'limit'"
            )
        return self


class SubmittedOrder(_Domain):
    """Broker's acknowledgement after submit; canonical fields the journal records.

    WS1 (D-078, delta §7.1): also the response shape of `get_order_by_id`,
    so it carries the broker's fill detail. Fill fields default to empty
    for submission-time acknowledgements (nothing has filled yet).
    """

    broker_order_id: str
    client_order_id: str
    symbol: str
    side: Side
    qty: int
    order_type: OrderType
    status: OrderStatus
    submitted_at: dt.datetime
    limit_price: float | None = None
    stop_price: float | None = None
    filled_avg_price: float | None = None
    filled_qty: int = 0
    filled_at: dt.datetime | None = None


class AccountSnapshot(_Domain):
    equity: float
    cash: float
    buying_power: float
    long_market_value: float
    daypl: float
    snapshot_at: dt.datetime
    # PDT-SUNSET-2026-06-04: ADR-009 §3.3 — `last_equity` is Alpaca's
    # previous-day-close equity (drives the `< 25k` activation gate);
    # `daytrade_count` is the rolling 5-business-day count. Defaults are
    # backward-compatible for tests/code paths that don't yet pass them
    # and survive Alpaca's planned 2026-07-06 field removal (last_equity
    # falls back to current equity at the boundary; daytrade_count → 0
    # naturally inerts the budget arithmetic).
    last_equity: float = 0.0
    daytrade_count: int = 0


class Position(_Domain):
    symbol: str
    qty: int
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float


class OpenOrder(_Domain):
    """Snapshot of a broker order that is currently open / pending fill.

    Used by the Feature B / KS-5 capacity gate to count pre-market queued
    entries that have not yet filled (incident 2026-05-07: positions only
    show up in `get_positions()` after fill, so back-to-back pre-market
    entries can each see "0 open positions" and bypass KS-5).

    `is_entry` distinguishes BUY entries from protective SELL legs (stops
    / TPs from prior days) so KS-5 only counts capacity that's actually
    being consumed by new entries.
    """

    symbol: str
    side: Side
    qty: int
    order_type: OrderType
    status: OrderStatus
    broker_order_id: str
    client_order_id: str

    @property
    def is_entry(self) -> bool:
        """A BUY-side open order is an entry leg (Quinn is long-only).

        SELL stops / TPs from prior days surface as open orders too, but
        they don't consume KS-5 capacity — the underlying long position
        already counts there.
        """
        return self.side == "buy"


class Quote(_Domain):
    """NBBO snapshot used by execution at submission time (pre_submission_nbbo)."""

    symbol: str
    bid: float
    ask: float
    last: float
    ts: dt.datetime


class BrokerRejected(Exception):
    """Non-retryable broker rejection — distinct from `BrokerUnavailable`.

    Raised when the broker returns a definitive 4xx that is NOT 429 (e.g.
    Alpaca's 40310000 wash-trade detection on a back-to-back stop after a
    pending entry, or a 422 insufficient buying power). Retrying these is
    pointless — the broker has made a deterministic decision — but the
    failure must still flow through the same execution-layer handlers as
    a transient outage so the kill-switch halts on `submission_partial_no_stop`
    and the journal records the partial state. See incident 2026-05-07.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        broker_code: int | str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.broker_code = broker_code


@runtime_checkable
class BrokerAdapter(Protocol):
    def submit_order(self, req: OrderRequest) -> SubmittedOrder: ...
    def submit_bracket_order(
        self, req: BracketOrderRequest
    ) -> tuple[SubmittedOrder, SubmittedOrder, SubmittedOrder | None]:
        """Submit an entry + protective stop (+ optional take-profit) as
        a single atomic complex order.

        Returns `(entry, stop, take_profit_or_None)`. Either ALL legs are
        created at the broker or NONE are — partial state is impossible
        by the broker's contract. On any failure the implementation
        raises `BrokerRejected` (non-retryable) or `BrokerUnavailable`
        (retries exhausted) and the caller treats this as a clean
        `submission_failed`: no exposure, no journal order rows.

        Wire-level class:
          - take_profit_price set    → BRACKET (entry, stop, tp)
          - take_profit_price absent → OTO     (entry, stop only)

        Hotfix 2026-05-07: replaces the prior three-sequential-`submit_order`
        flow that could leave a position unprotected when Alpaca's wash-
        trade detector (40310000) rejected the back-to-back stop while
        the entry buy was still pending pre-market.
        """
        ...

    def cancel_order(self, broker_order_id: str) -> None: ...
    def get_account(self) -> AccountSnapshot: ...
    def get_positions(self) -> list[Position]: ...
    def get_open_orders(self) -> list[OpenOrder]:
        """Return broker orders that are currently open / pending fill.

        Filters to lifecycle statuses that are not yet terminal:
        accepted, new, partially_filled, pending_new, accepted_for_bidding.
        Used by the KS-5 capacity gate to count pre-market queued entries
        that haven't filled yet (incident 2026-05-07).
        """
        ...

    def get_quote(self, symbol: str) -> Quote: ...

    def get_order_by_client_id(
        self, client_order_id: str
    ) -> SubmittedOrder | None:
        """Look up a previously-submitted order by deterministic
        `client_order_id` (S5.6 carry-fwd S6.4 reviewer M-4).

        On crash-recovery, the agent loop queries this with the
        `prop-{proposal_id}-{role}` ids before re-submitting any leg
        whose journal row was lost mid-pipeline; a hit means the
        broker already accepted the order and we must NOT submit
        again. Returns `None` when the broker has no record.
        Implementations must treat 404 as None — do not raise.
        """
        ...

    def get_order_by_id(self, broker_order_id: str) -> SubmittedOrder | None:
        """GET /v2/orders/{id} — current status + fill detail for a
        previously-submitted order (WS1, D-078, delta §7.1, ADR-010).

        The FillIngestor polls this for every journaled order with
        `final_status IS NULL` on each reconcile tick. Returns None on
        404 (mirrors `get_order_by_client_id` — do not raise for a
        missing order).
        """
        ...

    def replace_stop_order(
        self,
        broker_order_id: str,
        *,
        new_stop_price: float,
        client_order_id: str,
    ) -> SubmittedOrder:
        """Atomically replace a live stop order's `stop_price` (Feature A).

        Used by the thesis-review `adjust_stop` path so the position is
        never momentarily uncovered. Alpaca's PATCH /v2/orders/{id} is
        atomic: the broker either accepts the replacement (returning a
        new order id) and leaves NO gap, or rejects it and the original
        stop remains live.

        On any failure (network, invalid price, order already filled),
        implementations MUST raise — the caller's safety contract is
        "don't proceed unless the replacement succeeded; the original
        stop is still live."
        """
        ...


__all__ = [
    "AccountSnapshot",
    "BracketOrderRequest",
    "BrokerAdapter",
    "BrokerRejected",
    "OpenOrder",
    "OrderRequest",
    "OrderStatus",
    "OrderType",
    "Position",
    "Quote",
    "Side",
    "SubmittedOrder",
    "TIF",
]
