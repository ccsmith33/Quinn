"""Hotfix 2026-07-08 — `_pending_entry_spend` (agent loop → KS-7 seam).

Broker-reported cash does not decrease while an entry buy is queued, so
the agent loop prices each pending entry buy from its journal `orders`
row and hands the committed-but-unfilled total to `SizingEngine.size()`
as `pending_entry_spend`:

  - limit entries:  qty × limit_price
  - market entries: qty × pre_submission_last (NBBO captured at submit)
  - unknown / unpriced orders (e.g. operator-manual): skipped with a
    WARNING — same blind spot as pre-fix, but visible.
"""

from __future__ import annotations

import datetime as dt

import pytest

from broker.protocol import OpenOrder
from journal.models import OrderRow


def _open_buy(symbol: str, broker_order_id: str, qty: int) -> OpenOrder:
    return OpenOrder(
        symbol=symbol,
        side="buy",
        qty=qty,
        order_type="market",
        status="accepted",
        broker_order_id=broker_order_id,
        client_order_id=f"prop-{symbol}-entry",
    )


def _order_row(
    *,
    symbol: str,
    broker_order_id: str,
    qty: int,
    limit_price: float | None = None,
    pre_submission_last: float | None = None,
) -> OrderRow:
    return OrderRow(
        execution_id=1,
        role="entry",
        symbol=symbol,
        side="buy",
        order_type="limit" if limit_price is not None else "market",
        qty=qty,
        limit_price=limit_price,
        tif="gtc",
        broker_order_id=broker_order_id,
        submitted_at=dt.datetime(2026, 7, 8, 12, 1, tzinfo=dt.UTC),
        pre_submission_last=pre_submission_last,
        final_status=None,
    )


def test_pending_entry_spend_prices_from_journal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import journal.repo as repo
    from app.loop import _pending_entry_spend

    rows = {
        # Limit entry: qty × limit_price.
        "b-lim": _order_row(
            symbol="LIMCO", broker_order_id="b-lim", qty=10, limit_price=25.0
        ),
        # Market entry (the pre-market-queue shape): qty × pre_submission_last.
        "b-mkt": _order_row(
            symbol="MCRB",
            broker_order_id="b-mkt",
            qty=10,
            pre_submission_last=53.84,
        ),
    }
    monkeypatch.setattr(
        repo, "get_order_by_broker_id", lambda _db, bid: rows.get(bid)
    )

    pending = [
        _open_buy("LIMCO", "b-lim", qty=10),
        _open_buy("MCRB", "b-mkt", qty=10),
        # No journal row (operator-manual order) → skipped, not a crash.
        _open_buy("MANUAL", "b-unknown", qty=5),
    ]
    spend = _pending_entry_spend("unused.db", pending)

    assert spend == pytest.approx(10 * 25.0 + 10 * 53.84)


def test_pending_entry_spend_empty_is_zero() -> None:
    from app.loop import _pending_entry_spend

    assert _pending_entry_spend("unused.db", []) == 0.0
