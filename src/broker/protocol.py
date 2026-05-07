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


class SubmittedOrder(_Domain):
    """Broker's acknowledgement after submit; canonical fields the journal records."""

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


class AccountSnapshot(_Domain):
    equity: float
    cash: float
    buying_power: float
    long_market_value: float
    daypl: float
    snapshot_at: dt.datetime


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
