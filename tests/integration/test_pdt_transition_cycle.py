# PDT-TRANSITION-D-077: converter end-to-end against the real journal.
# Survives P2 with the converter (removed post-soak). Deliberately
# imports NO PDT-SUNSET module and seeds `virtual_exits` /
# `deferred_sells` via raw SQL — after P2 the tables persist as
# read-only history but their writer code is gone.
"""ADR-012 §4.2 — converter cycle against a real SQLite journal.

Proves the repo facade methods the converter depends on
(`list_active_virtual_exits`, `mark_virtual_exit_submitted`,
`insert_order`, `list_unreplayed_deferred_sells`,
`mark_deferred_skipped`) compose correctly: conversion orders land in
`orders` with GTC tif and D-077 notes, virtual exits leave the active
set, the deferred queue is invalidated, and a second run is a no-op.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from broker.protocol import Position, Quote, SubmittedOrder
from execution.pdt_transition import PDTTransitionConverter
from execution.virtual_exits import DeferredSellReplayer
from journal.migrate import apply_migrations
from journal.models import (
    ExecutionRow,
    FilingRow,
    OrderRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    JournalRepo,
    insert_execution,
    insert_filing,
    insert_prompt,
    insert_proposal,
)

_NOW = dt.datetime(2026, 6, 9, 12, 0, tzinfo=dt.UTC)


class _StubBroker:
    def __init__(self) -> None:
        self.oco_calls: list[dict[str, object]] = []
        self.submitted: list[object] = []
        self.lookups: list[str] = []
        self._seq = 0

    def get_positions(self) -> list[Position]:
        return [Position(
            symbol="AAPL", qty=5, avg_entry_price=100.0,
            market_value=500.0, unrealized_pnl=0.0,
        )]

    def get_quote(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, bid=99.95, ask=100.05, last=100.0, ts=_NOW)

    def get_order_by_client_id(self, client_order_id: str) -> SubmittedOrder | None:
        self.lookups.append(client_order_id)
        return None

    def submit_order(self, req: object) -> SubmittedOrder:
        raise AssertionError("plain submit_order must not be used for an OCO pair")

    def submit_oco_sell(
        self, *, symbol: str, qty: int, stop_price: float,
        limit_price: float | None, client_order_id: str,
    ) -> tuple[SubmittedOrder, SubmittedOrder | None]:
        self.oco_calls.append({
            "symbol": symbol, "qty": qty, "stop_price": stop_price,
            "limit_price": limit_price, "client_order_id": client_order_id,
        })

        def _so(order_type: str, seq: int) -> SubmittedOrder:
            return SubmittedOrder(
                broker_order_id=f"alp-conv-{seq}",
                client_order_id=client_order_id,
                symbol=symbol, side="sell", qty=qty,
                order_type=order_type,  # type: ignore[arg-type]
                status="accepted", submitted_at=_NOW,
                limit_price=limit_price if order_type == "limit" else None,
                stop_price=stop_price if order_type == "stop" else None,
            )

        self._seq += 2
        stop_so = _so("stop", self._seq - 1)
        tp_so = _so("limit", self._seq) if limit_price is not None else None
        return stop_so, tp_so


def _seed(db_path: str) -> int:
    """Prompt → filing → proposal → execution; returns execution id."""
    insert_prompt(db_path, PromptRow(
        prompt_version="sonnet_v1@deadbeef", name="sonnet_v1",
        file_path="src/prompts/sonnet_v1.txt", content_hash="deadbeef",
    ))
    fid = insert_filing(db_path, FilingRow(
        accession_number="0001234567-26-000999", cik=320193, form_type="8-K",
        filed_at=dt.datetime(2026, 6, 8, 14, 0),
        fetched_at=dt.datetime(2026, 6, 8, 14, 1),
        raw_text_path="/tmp/raw.txt", content_hash="abc",
        item_codes='["2.02"]', issuer_ticker="AAPL",
    ))
    pid = insert_proposal(db_path, ProposalRow(
        filing_id=fid, decision_id="dec-pdt-conv-1",
        model_id="claude-sonnet-4-6",
        prompt_version="sonnet_v1@deadbeef",
        raw_response="{}", kind="trade_proposal",
        symbol="AAPL", direction="long", size_pct_requested=0.05,
        conviction=8, thesis="thesis text",
        input_tokens=10, output_tokens=20, latency_ms=100, cost_usd=0.001,
    ))
    eid = insert_execution(db_path, ExecutionRow(
        proposal_id=pid, decision="accepted",
        realized_size_pct=0.05, realized_dollar_size=500.0,
        submitted_orders_json="[]",
    ))
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO virtual_exits (execution_id, proposal_id, symbol, "
            "qty, role, entry_price, stop_price, state) "
            "VALUES (?, ?, 'AAPL', 5, 'stop', 100.0, 95.0, 'active')",
            (eid, pid),
        )
        stop_vid = conn.execute(
            "SELECT id FROM virtual_exits ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO virtual_exits (execution_id, proposal_id, symbol, "
            "qty, role, entry_price, tp_price, state) "
            "VALUES (?, ?, 'AAPL', 5, 'tp', 100.0, 110.0, 'active')",
            (eid, pid),
        )
        conn.execute(
            "INSERT INTO deferred_sells (virtual_exit_id, execution_id, "
            "proposal_id, symbol, qty, role, trigger_price, ev_at_defer, "
            "deferred_reason) "
            "VALUES (?, ?, ?, 'AAPL', 5, 'stop', 94.0, -5.0, 'ev_lost')",
            (stop_vid, eid, pid),
        )
        conn.commit()
    return eid


def test_converter_cycle_real_journal_then_second_run_noop(
    tmp_path: Path,
) -> None:
    db_path = str(tmp_path / "journal.db")
    apply_migrations(db_path)
    _seed(db_path)

    journal = JournalRepo(db_path)
    broker = _StubBroker()
    converter = PDTTransitionConverter(broker=broker, journal=journal)
    report = converter.run()

    assert report.converted_oco == 1
    assert report.invalidated_deferred == 1
    assert report.errors == 0
    assert len(broker.oco_calls) == 1
    assert broker.oco_calls[0]["stop_price"] == 95.0
    assert broker.oco_calls[0]["limit_price"] == 110.0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        vexits = conn.execute(
            "SELECT role, state, submitted_broker_order_id, submitted_at "
            "FROM virtual_exits ORDER BY id ASC"
        ).fetchall()
        assert [v["state"] for v in vexits] == ["submitted", "submitted"]
        assert all(v["submitted_broker_order_id"] is not None for v in vexits)
        assert all(v["submitted_at"] is not None for v in vexits)

        sells = conn.execute(
            "SELECT role, side, tif, notes, broker_order_id, final_status "
            "FROM orders WHERE side='sell' ORDER BY id ASC"
        ).fetchall()
        assert [s["role"] for s in sells] == ["stop", "take_profit"]
        assert all(s["tif"] == "gtc" for s in sells)
        assert all(s["notes"] == "D-077 conversion" for s in sells)
        # D-078 §7.4 lifecycle contract (W3-H1): final_status is NULL at
        # submission so get_orders_pending_fill (final_status IS NULL)
        # surfaces the converted GTC legs to WS1's FillIngestor — a row
        # born 'accepted' would make its fill permanently unexplainable.
        assert all(s["final_status"] is None for s in sells)
        # The journaled broker ids are exactly the vexit linkage —
        # the §4.4(d) operator audit chain.
        assert {s["broker_order_id"] for s in sells} == {
            v["submitted_broker_order_id"] for v in vexits
        }

        deferred = conn.execute(
            "SELECT replayed_at, replay_broker_order_id, notes "
            "FROM deferred_sells"
        ).fetchall()
        assert len(deferred) == 1
        assert deferred[0]["replayed_at"] is not None
        assert deferred[0]["replay_broker_order_id"] is None
        assert "invalidated: superseded by D-077 conversion" in deferred[0]["notes"]

    # Second boot: zero active rows, zero unreplayed rows -> pure no-op.
    report2 = converter.run()
    assert report2.converted_oco == 0
    assert report2.invalidated_deferred == 0
    assert len(broker.oco_calls) == 1  # unchanged


class _HealStubBroker(_StubBroker):
    """Pre-check hit: a prior boot already placed the conversion order."""

    def __init__(self, prior: SubmittedOrder) -> None:
        super().__init__()
        self._prior = prior

    def get_order_by_client_id(self, client_order_id: str) -> SubmittedOrder | None:
        self.lookups.append(client_order_id)
        if client_order_id == self._prior.client_order_id:
            return self._prior
        return None

    def submit_oco_sell(
        self, *, symbol: str, qty: int, stop_price: float,
        limit_price: float | None, client_order_id: str,
    ) -> tuple[SubmittedOrder, SubmittedOrder | None]:
        raise AssertionError("heal path must not submit a new order")


def test_heal_after_journal_crash_real_journal(tmp_path: Path) -> None:
    """W3-M1 crash window against the real UNIQUE constraint: a prior
    boot journaled the conversion order but crashed before
    mark_virtual_exit_submitted. The re-insert trips
    orders.broker_order_id UNIQUE; the converter must consume it as
    already-journaled and still flip the vexits, with zero group errors
    (a per-boot error here pins the rows 'active' and blocks the §4.4
    operator gates until manual surgery)."""
    db_path = str(tmp_path / "journal.db")
    apply_migrations(db_path)
    eid = _seed(db_path)

    journal = JournalRepo(db_path)
    with sqlite3.connect(db_path) as conn:
        stop_vid = conn.execute(
            "SELECT id FROM virtual_exits WHERE role='stop'"
        ).fetchone()[0]
    prior = SubmittedOrder(
        broker_order_id="alp-prior-1",
        client_order_id=f"pdt-convert-{stop_vid}",
        symbol="AAPL", side="sell", qty=5,
        order_type="stop", status="accepted", submitted_at=_NOW,
        stop_price=95.0,
    )
    # The prior boot's journal write survived the crash.
    journal.insert_order(OrderRow(
        execution_id=eid, role="stop", symbol="AAPL", side="sell",
        order_type="stop", qty=5, stop_price=95.0, tif="gtc",
        broker_order_id="alp-prior-1", submitted_at=_NOW,
        notes="D-077 conversion",
    ))

    broker = _HealStubBroker(prior)
    converter = PDTTransitionConverter(broker=broker, journal=journal)
    report = converter.run()

    assert report.healed == 1
    assert report.errors == 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        vexits = conn.execute(
            "SELECT state, submitted_broker_order_id FROM virtual_exits "
            "ORDER BY id ASC"
        ).fetchall()
        assert [v["state"] for v in vexits] == ["submitted", "submitted"]
        assert all(
            v["submitted_broker_order_id"] == "alp-prior-1" for v in vexits
        )
        sells = conn.execute(
            "SELECT broker_order_id, final_status FROM orders "
            "WHERE side='sell'"
        ).fetchall()
        # No duplicate row, and still pending-fill (D-078 §7.4).
        assert [s["broker_order_id"] for s in sells] == ["alp-prior-1"]
        assert sells[0]["final_status"] is None

    # Next boot: the active set is drained; nothing left to heal.
    report2 = converter.run()
    assert report2.healed == 0
    assert report2.errors == 0


class _ComposedStubBroker(_StubBroker):
    """Boot-shaped stub: AAPL converts cleanly; MSFT's quote is missing
    so its group errors and stays active; the replayer's market submit
    works (the path the converted group must never reach)."""

    def __init__(self) -> None:
        super().__init__()
        self.replay_submits: list[object] = []

    def get_positions(self) -> list[Position]:
        return [
            Position(symbol="AAPL", qty=5, avg_entry_price=100.0,
                     market_value=500.0, unrealized_pnl=0.0),
            Position(symbol="MSFT", qty=3, avg_entry_price=200.0,
                     market_value=600.0, unrealized_pnl=0.0),
        ]

    def get_quote(self, symbol: str) -> Quote:
        if symbol == "MSFT":
            raise KeyError("MSFT quote unavailable")
        return super().get_quote(symbol)

    def submit_order(self, req: object) -> SubmittedOrder:
        self.replay_submits.append(req)
        self._seq += 1
        return SubmittedOrder(
            broker_order_id=f"alp-replay-{self._seq}",
            client_order_id=req.client_order_id,  # type: ignore[attr-defined]
            symbol=req.symbol,  # type: ignore[attr-defined]
            side="sell", qty=req.qty,  # type: ignore[attr-defined]
            order_type="market", status="accepted", submitted_at=_NOW,
        )


def test_boot_composed_converter_then_real_replayer(tmp_path: Path) -> None:
    """W3-M2: the real _boot composition — converter first, then the
    REAL DeferredSellReplayer against the same journal. The converted
    group's deferred row must never reach the broker (drained by the
    converter in the same boot); a yesterday-row the converter
    legitimately left behind (its group errored, rows still active)
    still replays normally."""
    db_path = str(tmp_path / "journal.db")
    apply_migrations(db_path)
    _seed(db_path)

    # Group B: MSFT stop vexit + a 3-days-old deferred row. The stub
    # broker has no MSFT quote, so the converter errors the group and
    # leaves it for the replayer path.
    three_days_ago = (
        dt.datetime.now(dt.UTC) - dt.timedelta(days=3)
    ).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        fid = conn.execute("SELECT id FROM filings LIMIT 1").fetchone()[0]
        pid_b = insert_proposal(db_path, ProposalRow(
            filing_id=fid, decision_id="dec-pdt-conv-2",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet_v1@deadbeef",
            raw_response="{}", kind="trade_proposal",
            symbol="MSFT", direction="long", size_pct_requested=0.05,
            conviction=8, thesis="thesis text",
            input_tokens=10, output_tokens=20, latency_ms=100,
            cost_usd=0.001,
        ))
        eid_b = insert_execution(db_path, ExecutionRow(
            proposal_id=pid_b, decision="accepted",
            realized_size_pct=0.05, realized_dollar_size=600.0,
            submitted_orders_json="[]",
        ))
        conn.execute(
            "INSERT INTO virtual_exits (execution_id, proposal_id, symbol, "
            "qty, role, entry_price, stop_price, state) "
            "VALUES (?, ?, 'MSFT', 3, 'stop', 200.0, 190.0, 'active')",
            (eid_b, pid_b),
        )
        vid_b = conn.execute(
            "SELECT id FROM virtual_exits WHERE symbol='MSFT'"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO deferred_sells (virtual_exit_id, execution_id, "
            "proposal_id, symbol, qty, role, trigger_price, ev_at_defer, "
            "deferred_reason, deferred_at) "
            "VALUES (?, ?, ?, 'MSFT', 3, 'stop', 189.0, -5.0, 'ev_lost', ?)",
            (vid_b, eid_b, pid_b, three_days_ago),
        )
        conn.commit()

    journal = JournalRepo(db_path)
    broker = _ComposedStubBroker()

    # _boot ordering (load-bearing): converter, THEN replayer.
    converter_report = PDTTransitionConverter(
        broker=broker, journal=journal
    ).run()
    replay_report = DeferredSellReplayer(
        broker=broker, journal=journal  # type: ignore[arg-type]
    ).run()

    assert converter_report.converted_oco == 1  # AAPL
    assert converter_report.errors == 1  # MSFT quote missing
    assert converter_report.invalidated_deferred == 1  # AAPL's row drained

    # The converted group submitted NOTHING through the replayer; the
    # leftover yesterday-row reached the broker exactly once.
    assert replay_report.replayed == 1
    assert replay_report.skipped_superseded == 0
    assert len(broker.replay_submits) == 1
    replayed_req = broker.replay_submits[0]
    assert replayed_req.symbol == "MSFT"  # type: ignore[attr-defined]
    assert replayed_req.client_order_id.startswith(  # type: ignore[attr-defined]
        "pdt-replay-"
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT symbol, replayed_at, replay_broker_order_id, notes "
            "FROM deferred_sells ORDER BY id ASC"
        ).fetchall()
        assert len(rows) == 2
        by_symbol = {r["symbol"]: r for r in rows}
        # AAPL: drained by the converter, no broker order.
        assert by_symbol["AAPL"]["replayed_at"] is not None
        assert by_symbol["AAPL"]["replay_broker_order_id"] is None
        assert (
            "invalidated: superseded by D-077 conversion"
            in by_symbol["AAPL"]["notes"]
        )
        # MSFT: genuinely replayed with a broker order id.
        assert by_symbol["MSFT"]["replayed_at"] is not None
        assert by_symbol["MSFT"]["replay_broker_order_id"] is not None
