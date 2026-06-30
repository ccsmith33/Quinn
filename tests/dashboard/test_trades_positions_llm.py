"""S9.1 — trades, positions, llm pages (AC-5, AC-6, AC-7)."""

from __future__ import annotations

import datetime as _dt

from fastapi.testclient import TestClient

from broker.protocol import AccountSnapshot, Position, Quote
from journal.models import PositionRow

from .conftest import (
    auth_headers,
    seed_execution_with_orders,
    seed_filings,
    seed_llm_calls,
    seed_prompt,
    seed_proposals,
)

# ---------------------------------------------------------------------------
# AC-5 — /trades
# ---------------------------------------------------------------------------


def test_trades_lists_recent_executions(
    client: TestClient, journal, db_path: str
) -> None:
    seed_prompt(db_path)
    fids = seed_filings(db_path, n=2)
    pids = seed_proposals(journal, filing_ids=fids)
    for i, pid in enumerate(pids):
        seed_execution_with_orders(
            journal, pid, symbol=f"AB{i}", qty=100 + i, fill_price=10.0 + i
        )
    resp = client.get("/trades", headers=auth_headers())
    assert resp.status_code == 200
    body = resp.text
    for i in range(len(pids)):
        assert f"AB{i}" in body
        # decision_id link to proposal page
        assert f"/proposals?symbol=AB{i}" in body or f"d-{i:03d}" in body


def test_trades_shows_slippage_vs_pre_submission_nbbo(
    client: TestClient, journal, db_path: str
) -> None:
    seed_prompt(db_path)
    fids = seed_filings(db_path, n=1)
    pids = seed_proposals(journal, filing_ids=fids)
    seed_execution_with_orders(journal, pids[0], symbol="AB0", qty=10, fill_price=12.34)
    resp = client.get("/trades", headers=auth_headers())
    body = resp.text
    # slippage = fill - mid_nbbo, rendered as a number; the test seeds
    # bid=fill-0.02, ask=fill+0.02, so mid==fill, slippage==0.
    assert "slippage" in body.lower()


# ---------------------------------------------------------------------------
# AC-6 — /positions (broker quote, 5s in-process cache)
# ---------------------------------------------------------------------------


class _FakeBroker:
    def __init__(self) -> None:
        self.positions = [
            Position(
                symbol="AB0",
                qty=100,
                avg_entry_price=10.0,
                market_value=1100.0,
                unrealized_pnl=100.0,
            ),
            Position(
                symbol="AB1",
                qty=50,
                avg_entry_price=20.0,
                market_value=950.0,
                unrealized_pnl=-50.0,
            ),
        ]
        self.quote_calls = 0

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=10000.0,
            cash=5000.0,
            buying_power=20000.0,
            long_market_value=2050.0,
            daypl=12.5,
            snapshot_at=_dt.datetime.now(_dt.UTC),
        )

    def get_positions(self) -> list[Position]:
        return list(self.positions)

    def get_quote(self, symbol: str) -> Quote:
        self.quote_calls += 1
        prices = {"AB0": 11.0, "AB1": 19.0}
        p = prices.get(symbol, 1.0)
        return Quote(
            symbol=symbol,
            bid=p - 0.01,
            ask=p + 0.01,
            last=p,
            ts=_dt.datetime.now(_dt.UTC),
        )

    def submit_order(self, req):  # pragma: no cover — dashboard never writes
        raise NotImplementedError

    def cancel_order(self, _bid: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_order_by_client_id(self, _cid: str):  # pragma: no cover
        raise NotImplementedError


def _build_with_broker(journal):
    from dashboard.app import build_app

    return build_app(
        journal=journal,
        dashboard_user="operator",
        dashboard_password="correcthorsebatterystaple",
        broker=_FakeBroker(),
        auto_refresh_seconds=30,
    )


def test_positions_renders_open_positions(journal) -> None:
    journal.insert_position(
        PositionRow(
            snapshot_at=_dt.datetime.now(_dt.UTC),
            source="broker",
            symbol="AB0",
            qty=100,
            avg_entry_price=10.0,
            market_value=1100.0,
            unrealized_pnl=100.0,
            notes=None,
        )
    )
    app = _build_with_broker(journal)
    client = TestClient(app)
    resp = client.get("/positions", headers=auth_headers())
    assert resp.status_code == 200
    body = resp.text
    assert "AB0" in body
    # current quote (live broker, not stored snapshot) appears
    assert "11" in body  # AB0 quote price 11.0


def test_positions_quote_cache_5s(journal) -> None:
    """AC-6: broker quotes are cached 5s — repeat hits within the window
    must not re-call the broker.
    """
    journal.insert_position(
        PositionRow(
            snapshot_at=_dt.datetime.now(_dt.UTC),
            source="broker",
            symbol="AB0",
            qty=100,
            avg_entry_price=10.0,
            market_value=1100.0,
            unrealized_pnl=100.0,
            notes=None,
        )
    )
    from dashboard.app import build_app

    fake = _FakeBroker()
    app = build_app(
        journal=journal,
        dashboard_user="operator",
        dashboard_password="correcthorsebatterystaple",
        broker=fake,
        auto_refresh_seconds=30,
    )
    client = TestClient(app)
    client.get("/positions", headers=auth_headers())
    first = fake.quote_calls
    client.get("/positions", headers=auth_headers())
    second = fake.quote_calls
    assert second == first, "second hit within 5s should use cached quote"


def test_positions_renders_thesis_and_stop_levels(
    client: TestClient, journal, db_path: str
) -> None:
    """The /positions page joins entry thesis + stop/take-profit via
    get_position_entry_context (D-063 follow-up)."""
    from journal.models import OrderRow

    seed_prompt(db_path)
    fids = seed_filings(db_path, n=1)
    pids = seed_proposals(journal, filing_ids=fids)
    eid = seed_execution_with_orders(journal, pids[0], symbol="AB0", qty=100, fill_price=10.0)
    journal.insert_order(
        OrderRow(
            execution_id=eid,
            role="stop",
            symbol="AB0",
            side="sell",
            order_type="stop",
            qty=100,
            stop_price=9.25,
            tif="day",
            broker_order_id="alpaca-stop",
            submitted_at=_dt.datetime.now(_dt.UTC),
            final_status=None,
        )
    )
    journal.insert_position(
        PositionRow(
            snapshot_at=_dt.datetime.now(_dt.UTC),
            source="broker",
            symbol="AB0",
            qty=100,
            avg_entry_price=10.0,
            market_value=1100.0,
            unrealized_pnl=100.0,
            notes=None,
        )
    )
    resp = client.get("/positions", headers=auth_headers())
    assert resp.status_code == 200
    body = resp.text
    assert "9.25" in body  # stop level rendered
    assert "alpha thesis" in body  # entry thesis joined from the proposal


def test_positions_without_broker_renders_journal_only(
    client: TestClient, journal
) -> None:
    """No broker injected (default) — page still renders from `positions` table."""
    journal.insert_position(
        PositionRow(
            snapshot_at=_dt.datetime.now(_dt.UTC),
            source="broker",
            symbol="AB0",
            qty=100,
            avg_entry_price=10.0,
            market_value=1100.0,
            unrealized_pnl=100.0,
            notes=None,
        )
    )
    resp = client.get("/positions", headers=auth_headers())
    assert resp.status_code == 200
    assert "AB0" in resp.text


# ---------------------------------------------------------------------------
# AC-7 — /llm (model spend / cache hit / latency)
# ---------------------------------------------------------------------------


def test_llm_renders_mtd_total_and_breakdown(
    client: TestClient, journal, db_path: str
) -> None:
    seed_llm_calls(journal, n=4)
    resp = client.get("/llm", headers=auth_headers())
    assert resp.status_code == 200
    body = resp.text
    # MTD total cost line
    assert "MTD" in body or "month" in body.lower()
    # Per-model breakdown — both models must appear.
    assert "claude-sonnet-4-6" in body
    assert "claude-opus-4-7" in body


def test_llm_renders_cache_hit_pct(
    client: TestClient, journal, db_path: str
) -> None:
    seed_llm_calls(journal, n=2)
    resp = client.get("/llm", headers=auth_headers())
    body = resp.text.lower()
    assert "cache" in body and ("%" in body or "pct" in body)


def test_llm_renders_p50_p95_latency(
    client: TestClient, journal, db_path: str
) -> None:
    seed_llm_calls(journal, n=10)
    resp = client.get("/llm", headers=auth_headers())
    body = resp.text.lower()
    assert "p50" in body
    assert "p95" in body
