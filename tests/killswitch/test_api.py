"""S7.1 — KillSwitch read/write API tests.

Architecture references: ADR-004 (kill-switch transport / state model),
§2.8 kill-switch, NFR-16 (append-only).
"""

from __future__ import annotations

import inspect
import threading
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.repo import JournalRepo, connect
from killswitch.api import KillSwitch, KillSwitchUninitialized


@pytest.fixture
def journal(tmp_path: Path) -> JournalRepo:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return JournalRepo(str(db_path))


def test_default_state_active_after_seed(journal: JournalRepo) -> None:
    """AC-3: seed row from 001_init.sql means state() returns active+system+boot."""
    ks = KillSwitch(journal)
    state = ks.state()
    assert state.state == "active"
    assert state.set_by == "system"
    assert state.reason == "boot"
    assert ks.is_halted() is False


def test_halt_then_state_returns_halted(journal: JournalRepo) -> None:
    """AC-1/AC-2: halt() inserts a new row and state() reports halted."""
    ks = KillSwitch(journal)
    ks.halt(reason="manual:telegram", set_by="operator", notes="op halted from phone")
    state = ks.state()
    assert state.state == "halted"
    assert state.reason == "manual:telegram"
    assert state.set_by == "operator"
    assert state.notes == "op halted from phone"
    assert ks.is_halted() is True


def test_resume_then_state_returns_active(journal: JournalRepo) -> None:
    """AC-1: resume() inserts an active row that supersedes the halt."""
    ks = KillSwitch(journal)
    ks.halt(reason="auto:KS-1", set_by="system")
    assert ks.is_halted() is True
    ks.resume(set_by="operator", notes="halted resolved")
    state = ks.state()
    assert state.state == "active"
    assert state.set_by == "operator"
    assert state.notes == "halted resolved"
    assert ks.is_halted() is False


def test_history_preserved_via_appends(journal: JournalRepo) -> None:
    """AC-2: halt/resume cycles never UPDATE/DELETE — every transition is a row."""
    ks = KillSwitch(journal)
    ks.halt(reason="auto:KS-2", set_by="system")
    ks.resume(set_by="operator")
    ks.halt(reason="manual:webhook", set_by="operator")
    with connect(journal.db_path) as conn:
        rows = conn.execute(
            "SELECT state, reason, set_by FROM kill_switch_state ORDER BY set_at ASC"
        ).fetchall()
    # seed row + 3 transitions
    assert len(rows) == 4
    assert rows[0]["reason"] == "boot"
    assert rows[1]["state"] == "halted" and rows[1]["reason"] == "auto:KS-2"
    assert rows[2]["state"] == "active"
    assert rows[3]["state"] == "halted" and rows[3]["reason"] == "manual:webhook"


def test_concurrent_halts_both_persist(journal: JournalRepo) -> None:
    """AC-5: simulated bot + auto-evaluator halts → both rows persist; latest wins."""
    ks = KillSwitch(journal)
    errors: list[Exception] = []

    def bot_halt() -> None:
        try:
            ks.halt(reason="manual:telegram", set_by="operator")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def auto_halt() -> None:
        try:
            ks.halt(reason="auto:KS-1", set_by="system")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=bot_halt), threading.Thread(target=auto_halt)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with connect(journal.db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM kill_switch_state WHERE state='halted'"
        ).fetchone()[0]
    assert n == 2
    # the latest halt is whichever ran last; both reasons are valid per AC-4
    assert ks.state().reason in ("manual:telegram", "auto:KS-1")
    assert ks.is_halted() is True


def test_no_update_or_delete_methods_exposed() -> None:
    """AC-2: KillSwitch has no public mutator beyond halt/resume; no update/delete in src."""
    public = {
        name
        for name, _ in inspect.getmembers(KillSwitch, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert public == {"state", "halt", "resume", "is_halted"}, (
        f"unexpected public methods: {public}"
    )
    src = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "killswitch"
        / "api.py"
    ).read_text(encoding="utf-8")
    upper = src.upper()
    assert "UPDATE KILL_SWITCH_STATE" not in upper
    assert "DELETE FROM KILL_SWITCH_STATE" not in upper


def test_state_raises_when_table_empty(tmp_path: Path) -> None:
    """AC-3: KillSwitchUninitialized when table is empty (defensive against seed-loss)."""
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    # purge the seed row to simulate an uninitialized table
    with connect(str(db_path)) as conn:
        conn.execute("DELETE FROM kill_switch_state")
    ks = KillSwitch(JournalRepo(str(db_path)))
    with pytest.raises(KillSwitchUninitialized):
        ks.state()


def test_reason_validation_accepts_adr_004_conventions(journal: JournalRepo) -> None:
    """AC-4: reason strings in ADR-004 §kill-switch state model conventions are accepted."""
    ks = KillSwitch(journal)
    accepted = [
        "manual:telegram",
        "manual:webhook",
        "auto:KS-1",
        "auto:KS-2",
        "auto:KS-3",
        "reconciler:discrepancy",
        "submission_partial_no_stop",
    ]
    for reason in accepted:
        ks.halt(reason=reason, set_by="system")
        ks.resume(set_by="system")
    assert ks.state().state == "active"
