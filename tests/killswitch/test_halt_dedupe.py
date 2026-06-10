"""WS1 — halt dedupe + pager throttle (D-078, delta §2.3; kills O-5).

`KillSwitch.halt()` gains an optional `fingerprint`. While an UNRESOLVED
halt row with the same reason+fingerprint exists, repeat calls insert no
row and return False (caller skips the page). Re-page at most once per
`halt_repage_minutes` (default 240) while the fingerprint persists. A
resumed halt whose fingerprint recurs is a fresh halt — resumes are not
free passes. `auto:KS-*` dedupe in auto_halts.py is unchanged.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.repo import JournalRepo, connect
from killswitch.api import KillSwitch


@pytest.fixture
def journal(tmp_path: Path) -> JournalRepo:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return JournalRepo(str(db_path))


class _Clock:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += dt.timedelta(**kwargs)


@pytest.fixture
def clock() -> _Clock:
    return _Clock(dt.datetime(2026, 6, 9, 14, 0, tzinfo=dt.UTC))


def _halt_rows(journal: JournalRepo) -> list[dict]:
    with connect(journal.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM kill_switch_state WHERE state='halted' ORDER BY set_at ASC"
        ).fetchall()
    return [dict(r) for r in rows]


_FP = "ACET:-30|BALY:-19"


def test_halt_without_fingerprint_always_inserts(
    journal: JournalRepo, clock: _Clock
) -> None:
    """Legacy callers (manual halts, auto:KS-*) are untouched: every call
    appends a row and reports fired."""
    ks = KillSwitch(journal, now_fn=clock)
    assert ks.halt(reason="manual:telegram", set_by="operator") is True
    clock.advance(seconds=1)  # set_at is the table PK; wall clocks never tie
    assert ks.halt(reason="manual:telegram", set_by="operator") is True
    assert len(_halt_rows(journal)) == 2


def test_first_fingerprinted_halt_fires(journal: JournalRepo, clock: _Clock) -> None:
    ks = KillSwitch(journal, now_fn=clock)
    fired = ks.halt(
        reason="reconciler:discrepancy",
        set_by="system",
        notes='[{"symbol": "ACET"}]',
        fingerprint=_FP,
    )
    assert fired is True
    assert ks.is_halted() is True
    assert len(_halt_rows(journal)) == 1


def test_identical_fingerprint_deduped_within_window(
    journal: JournalRepo, clock: _Clock
) -> None:
    """The O-5 amplifier: a persisting diff re-halts every 300 s tick.
    Deduped — one row, one page, regardless of tick count."""
    ks = KillSwitch(journal, now_fn=clock, repage_minutes=240)
    assert ks.halt(
        reason="reconciler:discrepancy", set_by="system", fingerprint=_FP
    ) is True
    for _ in range(10):  # ten 5-minute ticks
        clock.advance(minutes=5)
        assert ks.halt(
            reason="reconciler:discrepancy", set_by="system", fingerprint=_FP
        ) is False
    assert len(_halt_rows(journal)) == 1
    assert ks.is_halted() is True


def test_repage_after_window_elapses(journal: JournalRepo, clock: _Clock) -> None:
    ks = KillSwitch(journal, now_fn=clock, repage_minutes=240)
    assert ks.halt(
        reason="reconciler:discrepancy", set_by="system", fingerprint=_FP
    ) is True
    clock.advance(minutes=241)
    assert ks.halt(
        reason="reconciler:discrepancy", set_by="system", fingerprint=_FP
    ) is True
    assert len(_halt_rows(journal)) == 2


def test_resume_then_same_fingerprint_is_fresh_halt(
    journal: JournalRepo, clock: _Clock
) -> None:
    """Resumes are not free passes: a recurring fingerprint after resume
    pages again immediately."""
    ks = KillSwitch(journal, now_fn=clock, repage_minutes=240)
    ks.halt(reason="reconciler:discrepancy", set_by="system", fingerprint=_FP)
    clock.advance(minutes=5)
    ks.resume(set_by="operator", notes="investigated")
    clock.advance(minutes=5)
    fired = ks.halt(
        reason="reconciler:discrepancy", set_by="system", fingerprint=_FP
    )
    assert fired is True
    assert len(_halt_rows(journal)) == 2


def test_different_fingerprint_fires_while_halted(
    journal: JournalRepo, clock: _Clock
) -> None:
    """A NEW diff set while already halted is new information — page."""
    ks = KillSwitch(journal, now_fn=clock)
    ks.halt(reason="reconciler:discrepancy", set_by="system", fingerprint=_FP)
    clock.advance(minutes=5)
    fired = ks.halt(
        reason="reconciler:discrepancy",
        set_by="system",
        fingerprint="QNST:-19",
    )
    assert fired is True
    assert len(_halt_rows(journal)) == 2


def test_different_reason_same_fingerprint_fires(
    journal: JournalRepo, clock: _Clock
) -> None:
    ks = KillSwitch(journal, now_fn=clock)
    ks.halt(reason="reconciler:discrepancy", set_by="system", fingerprint=_FP)
    clock.advance(minutes=5)
    assert ks.halt(reason="auto:KS-9", set_by="system", fingerprint=_FP) is True


def test_fingerprinted_notes_envelope_preserves_detail(
    journal: JournalRepo, clock: _Clock
) -> None:
    """kill_switch_state has no fingerprint column (migration 005 is
    indexes-only), so the fingerprint rides in a notes JSON envelope.
    The operator detail stays retrievable."""
    ks = KillSwitch(journal, now_fn=clock)
    detail = '[{"symbol": "ACET", "broker_qty": 0, "journal_qty": 30}]'
    ks.halt(
        reason="reconciler:discrepancy",
        set_by="system",
        notes=detail,
        fingerprint=_FP,
    )
    [row] = _halt_rows(journal)
    envelope = json.loads(row["notes"])
    assert envelope["fingerprint"] == _FP
    assert envelope["detail"] == detail


def test_unfingerprinted_notes_stored_raw(journal: JournalRepo, clock: _Clock) -> None:
    ks = KillSwitch(journal, now_fn=clock)
    ks.halt(reason="manual:telegram", set_by="operator", notes="via /halt")
    [row] = _halt_rows(journal)
    assert row["notes"] == "via /halt"


def test_dedupe_survives_process_restart(journal: JournalRepo, clock: _Clock) -> None:
    """Dedupe state lives in the journal, not in memory — a new KillSwitch
    instance (agent restart) still dedupes the persisting fingerprint."""
    ks1 = KillSwitch(journal, now_fn=clock)
    ks1.halt(reason="reconciler:discrepancy", set_by="system", fingerprint=_FP)
    clock.advance(minutes=5)
    ks2 = KillSwitch(journal, now_fn=clock)  # fresh instance, same journal
    assert ks2.halt(
        reason="reconciler:discrepancy", set_by="system", fingerprint=_FP
    ) is False
    assert len(_halt_rows(journal)) == 1
