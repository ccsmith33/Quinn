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


class Quote(_Domain):
    """NBBO snapshot used by execution at submission time (pre_submission_nbbo)."""

    symbol: str
    bid: float
    ask: float
    last: float
    ts: dt.datetime


@runtime_checkable
class BrokerAdapter(Protocol):
    def submit_order(self, req: OrderRequest) -> SubmittedOrder: ...
    def cancel_order(self, broker_order_id: str) -> None: ...
    def get_account(self) -> AccountSnapshot: ...
    def get_positions(self) -> list[Position]: ...
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
    "OrderRequest",
    "OrderStatus",
    "OrderType",
    "Position",
    "Quote",
    "Side",
    "SubmittedOrder",
    "TIF",
]
