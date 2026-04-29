"""S8.2 — Alert watcher tests (NFR-11, ACs 4-5 + 7).

TDD. Covers:
  AC-4: Telegram alerts on KS flips, monthly cost ≥$150, backup failure,
        NFR-1 latency p95 > 90s/1h, log sink unavailable >10min.
  AC-5: same alert type fires at most once per hour (debounce).
  Test-7: NFR-1 latency alert when synthetic proposals.latency_ms produces
          p95 > 90s over the last hour.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.models import (
    BackupRow,
    FilingRow,
    KillSwitchStateRow,
    PromptRow,
)
from journal.repo import (
    JournalRepo,
    insert_backup,
    insert_filing,
    insert_kill_switch_state,
    insert_prompt,
)
from observability.alerts import AlertWatcher, LogSinkStatus


@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "journal.db")
    apply_migrations(db_path)
    return db_path


@pytest.fixture
def journal(db: str) -> JournalRepo:
    return JournalRepo(db)


@pytest.fixture
def now() -> dt.datetime:
    return dt.datetime(2026, 4, 28, 18, 0, 0, tzinfo=dt.UTC)


class _RecordingNotifier:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, text: str) -> None:
        self.sent.append(text)


def _seed_filing(db: str) -> int:
    return insert_filing(
        db,
        FilingRow(
            accession_number="0001-01",
            cik=1,
            form_type="8-K",
            filed_at=dt.datetime.now(dt.UTC),
            fetched_at=dt.datetime.now(dt.UTC),
            raw_text_path="/dev/null",
            content_hash="x" * 64,
        ),
    )


def _seed_prompt(db: str) -> None:
    insert_prompt(
        db,
        PromptRow(
            prompt_version="p_v1",
            name="t",
            file_path="/dev/null",
            content_hash="x" * 64,
        ),
    )


def _insert_llm_call_at(
    db: str,
    *,
    decision_id: str,
    cost_usd: float,
    called_at: dt.datetime,
    prompt_version: str = "p_v1",
) -> None:
    """Direct insert so the test can pin `called_at`; the repo's helper
    ignores the row's `called_at` and uses SQL DEFAULT CURRENT_TIMESTAMP."""
    from journal.repo import connect

    with connect(db) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO llm_calls "
            "(decision_id, purpose, model_id, prompt_version, "
            "input_tokens, output_tokens, latency_ms, cost_usd, called_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                decision_id,
                "analyze",
                "claude-sonnet-4-6",
                prompt_version,
                1,
                1,
                1,
                cost_usd,
                called_at,
            ),
        )
        conn.execute("COMMIT")


# ---------------------------------------------------------------------------
# AC-4 — kill-switch flip alert
# ---------------------------------------------------------------------------


def test_alert_on_kill_switch_flip(db: str, journal: JournalRepo, now: dt.datetime) -> None:
    """A new `kill_switch_state` row inserted since the last poll fires a
    Telegram alert."""
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(journal=journal, notifier=notifier)
    # First poll: only the boot seed row exists; not an alertable flip
    # (the watcher tracks the most recent set_at it has already seen).
    watcher.poll(now=now)
    assert len(notifier.sent) == 0

    # Operator halts via Telegram → new row inserted.
    insert_kill_switch_state(
        db,
        KillSwitchStateRow(
            set_at=now,
            state="halted",
            reason="manual:telegram",
            set_by="operator",
            notes=None,
        ),
    )
    watcher.poll(now=now + dt.timedelta(seconds=60))
    assert len(notifier.sent) == 1
    assert "kill" in notifier.sent[0].lower() or "halted" in notifier.sent[0].lower()


def test_alert_kill_switch_does_not_double_fire(
    db: str, journal: JournalRepo, now: dt.datetime
) -> None:
    """Repeated polls with the same latest row do NOT re-emit."""
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(journal=journal, notifier=notifier)
    insert_kill_switch_state(
        db,
        KillSwitchStateRow(
            set_at=now,
            state="halted",
            reason="manual:telegram",
            set_by="operator",
            notes=None,
        ),
    )
    watcher.poll(now=now + dt.timedelta(seconds=60))
    watcher.poll(now=now + dt.timedelta(seconds=120))
    assert len(notifier.sent) == 1


# ---------------------------------------------------------------------------
# AC-4 — monthly cost overrun
# ---------------------------------------------------------------------------


def test_alert_on_monthly_cost_overrun(db: str, journal: JournalRepo, now: dt.datetime) -> None:
    """When sum(llm_calls.cost_usd) for the current month ≥ $150, fire."""
    _seed_prompt(db)
    month_start = dt.datetime(now.year, now.month, 1, tzinfo=dt.UTC)
    # 10 calls at $15.01 = $150.10
    for i in range(10):
        _insert_llm_call_at(
            db,
            decision_id=f"d-{i}",
            cost_usd=15.01,
            called_at=month_start + dt.timedelta(days=i, hours=1),
        )
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(journal=journal, notifier=notifier)
    watcher.poll(now=now)
    matches = [s for s in notifier.sent if "$150" in s or "150.0" in s or "cost" in s.lower()]
    assert matches, f"expected cost-overrun alert; got: {notifier.sent}"


def test_no_alert_when_monthly_cost_below_threshold(
    db: str, journal: JournalRepo, now: dt.datetime
) -> None:
    _seed_prompt(db)
    month_start = dt.datetime(now.year, now.month, 1, tzinfo=dt.UTC)
    _insert_llm_call_at(
        db,
        decision_id="d-1",
        cost_usd=149.99,
        called_at=month_start + dt.timedelta(days=1),
    )
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(journal=journal, notifier=notifier)
    watcher.poll(now=now)
    assert all("cost" not in s.lower() for s in notifier.sent)


# ---------------------------------------------------------------------------
# AC-4 — backup failure
# ---------------------------------------------------------------------------


def test_alert_on_backup_verify_failed(db: str, journal: JournalRepo, now: dt.datetime) -> None:
    insert_backup(
        db,
        BackupRow(
            started_at=now - dt.timedelta(hours=2),
            completed_at=now - dt.timedelta(hours=1),
            backup_path="/var/backups/quinn-2026-04-28.db",
            db_size_bytes=1024,
            sha256="x" * 64,
            verified_at=now - dt.timedelta(minutes=30),
            verify_result="failed:checksum_mismatch",
            notes=None,
        ),
    )
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(journal=journal, notifier=notifier)
    watcher.poll(now=now)
    assert any("backup" in s.lower() for s in notifier.sent)


def test_alert_on_backup_missing_for_26h(db: str, journal: JournalRepo, now: dt.datetime) -> None:
    """No backup row inserted in the last 26h → fire."""
    insert_backup(
        db,
        BackupRow(
            started_at=now - dt.timedelta(hours=27),
            completed_at=now - dt.timedelta(hours=27, minutes=-30),
            backup_path="/var/backups/quinn-2026-04-27.db",
            db_size_bytes=1024,
            sha256="x" * 64,
            verified_at=now - dt.timedelta(hours=26, minutes=30),
            verify_result="ok",
            notes=None,
        ),
    )
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(journal=journal, notifier=notifier)
    watcher.poll(now=now)
    assert any("backup" in s.lower() for s in notifier.sent)


def test_no_alert_when_recent_backup_succeeded(
    db: str, journal: JournalRepo, now: dt.datetime
) -> None:
    insert_backup(
        db,
        BackupRow(
            started_at=now - dt.timedelta(hours=2),
            completed_at=now - dt.timedelta(hours=1),
            backup_path="/var/backups/quinn-2026-04-28.db",
            db_size_bytes=1024,
            sha256="x" * 64,
            verified_at=now - dt.timedelta(minutes=30),
            verify_result="ok",
            notes=None,
        ),
    )
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(journal=journal, notifier=notifier)
    watcher.poll(now=now)
    assert all("backup" not in s.lower() for s in notifier.sent)


# ---------------------------------------------------------------------------
# Test-7 — NFR-1 latency p95 alert
# ---------------------------------------------------------------------------


def _insert_proposal_with_latency(
    db: str,
    *,
    filing_id: int,
    decision_id: str,
    latency_ms: int,
    created_at: dt.datetime,
) -> None:
    """Helper that bypasses `insert_proposal` so tests can pin `created_at`
    inside the latency window. The repo's `insert_proposal` lets SQL
    default `created_at = CURRENT_TIMESTAMP`, which doesn't match a fixed
    test `now` — so we write the row directly."""
    from journal.repo import connect

    with connect(db) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO proposals "
            "(filing_id, decision_id, model_id, prompt_version, raw_response, "
            "kind, input_tokens, output_tokens, latency_ms, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                filing_id,
                decision_id,
                "claude-sonnet-4-6",
                "p_v1",
                "{}",
                "trade",
                1,
                1,
                latency_ms,
                0.01,
                created_at,
            ),
        )
        conn.execute("COMMIT")


def test_nfr1_latency_alert(db: str, journal: JournalRepo, now: dt.datetime) -> None:
    """Synthetic latency rows in `proposals` create > 90s p95 over last hour
    → alert fires."""
    _seed_prompt(db)
    f = _seed_filing(db)
    inside_window = now - dt.timedelta(minutes=30)
    # 100 proposals in the last hour. 90 fast (1000ms), 10 slow (95_000ms).
    # With nearest-rank p95 on n=100, the index is ceil(0.95*100)-1 = 94,
    # which falls in the slow tail (indices 90..99) → alert fires.
    for i in range(90):
        _insert_proposal_with_latency(
            db,
            filing_id=f,
            decision_id=f"fast-{i}",
            latency_ms=1000,
            created_at=inside_window,
        )
    for i in range(10):
        _insert_proposal_with_latency(
            db,
            filing_id=f,
            decision_id=f"slow-{i}",
            latency_ms=95_000,
            created_at=inside_window,
        )
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(journal=journal, notifier=notifier)
    watcher.poll(now=now)
    matches = [s for s in notifier.sent if "latency" in s.lower() or "p95" in s.lower()]
    assert matches, f"expected NFR-1 latency alert; got: {notifier.sent}"


def test_no_latency_alert_when_p95_under_90s(
    db: str, journal: JournalRepo, now: dt.datetime
) -> None:
    _seed_prompt(db)
    f = _seed_filing(db)
    inside_window = now - dt.timedelta(minutes=30)
    for i in range(100):
        _insert_proposal_with_latency(
            db,
            filing_id=f,
            decision_id=f"ok-{i}",
            latency_ms=2000,
            created_at=inside_window,
        )
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(journal=journal, notifier=notifier)
    watcher.poll(now=now)
    assert all("latency" not in s.lower() for s in notifier.sent)


# ---------------------------------------------------------------------------
# AC-4 — log sink unavailable >10 min
# ---------------------------------------------------------------------------


def test_alert_on_log_sink_unavailable(db: str, journal: JournalRepo, now: dt.datetime) -> None:
    """Log-sink status is reported externally (Vector emits a metric); the
    AlertWatcher accepts a `log_sink_status_fn` that returns the duration
    of the current outage in seconds (or None if healthy)."""
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(
        journal=journal,
        notifier=notifier,
        log_sink_status_fn=lambda: LogSinkStatus(unavailable_since=now - dt.timedelta(minutes=15)),
    )
    watcher.poll(now=now)
    assert any("log sink" in s.lower() or "log-sink" in s.lower() for s in notifier.sent)


def test_no_alert_when_log_sink_outage_under_10min(
    db: str, journal: JournalRepo, now: dt.datetime
) -> None:
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(
        journal=journal,
        notifier=notifier,
        log_sink_status_fn=lambda: LogSinkStatus(unavailable_since=now - dt.timedelta(minutes=5)),
    )
    watcher.poll(now=now)
    assert all("log sink" not in s.lower() for s in notifier.sent)


# ---------------------------------------------------------------------------
# AC-5 — debounce: same alert type once per hour max
# ---------------------------------------------------------------------------


def test_alert_debounced_within_hour(db: str, journal: JournalRepo, now: dt.datetime) -> None:
    """A backup-failure alert that just fired must NOT fire again until 1h
    has elapsed."""
    insert_backup(
        db,
        BackupRow(
            started_at=now - dt.timedelta(hours=2),
            completed_at=now - dt.timedelta(hours=1),
            backup_path="/var/backups/quinn-2026-04-28.db",
            db_size_bytes=1024,
            sha256="x" * 64,
            verified_at=now - dt.timedelta(minutes=30),
            verify_result="failed:checksum_mismatch",
            notes=None,
        ),
    )
    notifier = _RecordingNotifier()
    watcher = AlertWatcher(journal=journal, notifier=notifier)
    watcher.poll(now=now)
    sent_first = list(notifier.sent)
    assert any("backup" in s.lower() for s in sent_first)

    # 30 minutes later — same condition still true; MUST not fire again.
    watcher.poll(now=now + dt.timedelta(minutes=30))
    assert notifier.sent == sent_first

    # 65 minutes after the first alert — debounce window passed.
    watcher.poll(now=now + dt.timedelta(minutes=65))
    new_alerts = [s for s in notifier.sent[len(sent_first) :] if "backup" in s.lower()]
    assert len(new_alerts) >= 1


def test_alert_debounce_is_per_alert_type(db: str, journal: JournalRepo, now: dt.datetime) -> None:
    """Distinct alert kinds (backup vs. cost-overrun) debounce independently."""
    _seed_prompt(db)
    insert_backup(
        db,
        BackupRow(
            started_at=now - dt.timedelta(hours=2),
            completed_at=now - dt.timedelta(hours=1),
            backup_path="/var/backups/quinn-2026-04-28.db",
            db_size_bytes=1024,
            sha256="x" * 64,
            verified_at=now - dt.timedelta(minutes=30),
            verify_result="failed:checksum_mismatch",
            notes=None,
        ),
    )
    month_start = dt.datetime(now.year, now.month, 1, tzinfo=dt.UTC)
    _insert_llm_call_at(
        db,
        decision_id="d-1",
        cost_usd=200.0,
        called_at=month_start + dt.timedelta(days=1),
    )

    notifier = _RecordingNotifier()
    watcher = AlertWatcher(journal=journal, notifier=notifier)
    watcher.poll(now=now)
    backup_alerts = [s for s in notifier.sent if "backup" in s.lower()]
    cost_alerts = [s for s in notifier.sent if "cost" in s.lower() or "$150" in s or "$200" in s]
    # Both alert kinds fired in the same poll despite both being active —
    # debouncing is per kind, not global.
    assert len(backup_alerts) == 1
    assert len(cost_alerts) == 1
