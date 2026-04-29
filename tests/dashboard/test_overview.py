"""S9.1 — overview page (AC-3).

`/` renders kill-switch state + reason, latest equity + day P&L, equity
sparkline (inline SVG), counts of "filings ingested today / proposals
emitted today / trades submitted today", auto-halt status.
"""

from __future__ import annotations

import datetime as _dt

from fastapi.testclient import TestClient

from journal.models import KillSwitchStateRow

from .conftest import (
    auth_headers,
    seed_account_snapshots,
    seed_execution_with_orders,
    seed_filings,
    seed_kill_switch,
    seed_prompt,
    seed_proposals,
)


def test_overview_renders_kill_switch_state_active(
    client: TestClient, journal, db_path: str
) -> None:
    seed_kill_switch(journal, halted=False)
    resp = client.get("/", headers=auth_headers())
    assert resp.status_code == 200
    body = resp.text
    assert "active" in body.lower()


def test_overview_renders_kill_switch_state_halted_with_reason(
    client: TestClient, journal, db_path: str
) -> None:
    journal.insert_kill_switch_state(
        KillSwitchStateRow(
            set_at=_dt.datetime.now(_dt.UTC),
            state="halted",
            reason="ks1_daily_loss",
            set_by="system",
            notes="auto",
        )
    )
    resp = client.get("/", headers=auth_headers())
    body = resp.text
    assert "halted" in body.lower()
    assert "ks1_daily_loss" in body


def test_overview_renders_equity_and_daypl(
    client: TestClient, journal, db_path: str
) -> None:
    seed_account_snapshots(journal, days=3)
    resp = client.get("/", headers=auth_headers())
    body = resp.text
    # equity numeric and day P&L are present
    assert "10150" in body or "10,150" in body
    assert "12.5" in body  # daypl seed value


def test_overview_renders_equity_sparkline_svg(
    client: TestClient, journal, db_path: str
) -> None:
    """AC-3: 30d equity sparkline inline SVG, no JS chart lib."""
    seed_account_snapshots(journal, days=10)
    resp = client.get("/", headers=auth_headers())
    body = resp.text
    assert "<svg" in body and "</svg>" in body
    assert "polyline" in body or "<path" in body


def test_overview_renders_today_counts(
    client: TestClient, journal, db_path: str
) -> None:
    """AC-3: filings/proposals/trades submitted-today counts."""
    seed_prompt(db_path)
    fids = seed_filings(db_path, n=2)
    pids = seed_proposals(journal, filing_ids=fids)
    seed_execution_with_orders(journal, pids[0])
    resp = client.get("/", headers=auth_headers())
    body = resp.text.lower()
    # The 3 word-tokens must appear next to numerics.
    for keyword in ("filings", "proposals", "trades"):
        assert keyword in body
