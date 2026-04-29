"""S8.2 — Daily reporter tests (NFR-10, FR-34, ACs 1-3 + 6).

TDD. Covers:
  AC-1: `compose_and_send(now)` returns a Report
  AC-2: report aggregates from journal correctly (filings, prefilter rules,
        proposals/kind, executions accepted, fills, day-end equity, MTD
        inference cost FROM `llm_calls`).
  AC-3: report is sent via Telegram and optionally email.
  AC-6 (per dev-note): KS-1 cleared at next session start (09:30 ET on
        next trading day).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from jobs.daily_report import (
    DailyReporter,
    Report,
    clear_auto_ks1_at_session_start,
)
from journal.migrate import apply_migrations
from journal.models import (
    AccountSnapshotRow,
    FilingRow,
    KillSwitchStateRow,
    OrderRow,
    PrefilterDecisionRow,
    ProposalRow,
)
from journal.repo import (
    JournalRepo,
    insert_account_snapshot,
    insert_execution,
    insert_filing,
    insert_kill_switch_state,
    insert_order,
    insert_prefilter_decision,
    insert_prompt,
    insert_proposal,
)

# ---------------------------------------------------------------------------
# Fixtures — db with a freshly applied migration; a JournalRepo over it.
# ---------------------------------------------------------------------------


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
    # 2026-04-28 16:30 ET == 20:30 UTC (DST in effect)
    return dt.datetime(2026, 4, 28, 20, 30, 0, tzinfo=dt.UTC)


def _seed_prompt(db: str, version: str = "p_v1") -> None:
    from journal.models import PromptRow

    insert_prompt(
        db,
        PromptRow(
            prompt_version=version,
            name="test",
            file_path="/dev/null",
            content_hash="x" * 64,
        ),
    )


def _insert_filing(db: str, accession: str, filed_at: dt.datetime, fetched_at: dt.datetime) -> int:
    return insert_filing(
        db,
        FilingRow(
            accession_number=accession,
            cik=320193,
            form_type="8-K",
            filed_at=filed_at,
            fetched_at=fetched_at,
            raw_text_path=f"/tmp/{accession}.txt",
            content_hash=accession.ljust(64, "x"),
        ),
    )


def _insert_proposal(
    db: str,
    filing_id: int,
    decision_id: str,
    kind: str = "trade",
    cost_usd: float = 0.0,
    latency_ms: int = 1000,
    prompt_version: str = "p_v1",
) -> int:
    return insert_proposal(
        db,
        ProposalRow(
            filing_id=filing_id,
            decision_id=decision_id,
            model_id="claude-sonnet-4-6",
            prompt_version=prompt_version,
            raw_response="{}",
            kind=kind,
            symbol="AAPL",
            direction="long",
            size_pct_requested=0.05,
            conviction=7,
            thesis="t",
            input_tokens=100,
            output_tokens=50,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        ),
    )


def _insert_llm_call(
    db: str,
    decision_id: str,
    cost_usd: float,
    purpose: str = "analyze",
    called_at: dt.datetime | None = None,
    prompt_version: str = "p_v1",
) -> None:
    """Direct INSERT so tests can pin `called_at` (the repo's
    `insert_llm_call` ignores `row.called_at` and lets SQL DEFAULT
    CURRENT_TIMESTAMP fill it — fine for production since calls happen
    "now", but breaks tests that backdate rows to test month-boundary
    logic).
    """
    from journal.repo import connect

    with connect(db) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if called_at is not None:
            conn.execute(
                "INSERT INTO llm_calls "
                "(decision_id, purpose, model_id, prompt_version, "
                "input_tokens, output_tokens, latency_ms, cost_usd, called_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision_id,
                    purpose,
                    "claude-sonnet-4-6",
                    prompt_version,
                    100,
                    50,
                    1000,
                    cost_usd,
                    called_at,
                ),
            )
        else:
            conn.execute(
                "INSERT INTO llm_calls "
                "(decision_id, purpose, model_id, prompt_version, "
                "input_tokens, output_tokens, latency_ms, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision_id,
                    purpose,
                    "claude-sonnet-4-6",
                    prompt_version,
                    100,
                    50,
                    1000,
                    cost_usd,
                ),
            )
        conn.execute("COMMIT")


# ---------------------------------------------------------------------------
# AC-2: report data sources
# ---------------------------------------------------------------------------


def test_report_aggregates_journal_correctly(
    db: str, journal: JournalRepo, now: dt.datetime
) -> None:
    """End-to-end: seed every counted entity and assert the report sums."""
    _seed_prompt(db)

    today_start = now.replace(hour=13, minute=30, tzinfo=dt.UTC)  # 09:30 ET
    yesterday = now - dt.timedelta(days=1)

    # 3 filings ingested today, 1 yesterday (must NOT count).
    f1 = _insert_filing(db, "0001-01", today_start, today_start)
    f2 = _insert_filing(db, "0001-02", today_start, today_start + dt.timedelta(minutes=5))
    f3 = _insert_filing(db, "0001-03", today_start, today_start + dt.timedelta(hours=2))
    _insert_filing(db, "0001-04", yesterday, yesterday)

    # Prefilter decisions: 2 accept (rule_fired=item_code), 1 reject (rule=duplicate)
    insert_prefilter_decision(
        db,
        PrefilterDecisionRow(filing_id=f1, decision="accept", rule_fired="item_code"),
    )
    insert_prefilter_decision(
        db,
        PrefilterDecisionRow(filing_id=f2, decision="accept", rule_fired="item_code"),
    )
    insert_prefilter_decision(
        db,
        PrefilterDecisionRow(filing_id=f3, decision="reject", rule_fired="duplicate"),
    )

    # 2 proposals (kind=trade), 1 (kind=no_trade).
    p1 = _insert_proposal(db, f1, "dec-1", kind="trade", cost_usd=0.42)
    p2 = _insert_proposal(db, f2, "dec-2", kind="trade", cost_usd=0.31)
    _insert_proposal(db, f3, "dec-3", kind="no_trade", cost_usd=0.05)

    # llm_calls — these are what S8.2 reads for spend (NOT proposal_reviews).
    month_start = dt.datetime(now.year, now.month, 1, tzinfo=dt.UTC)
    _insert_llm_call(db, "dec-1", 0.42, called_at=month_start + dt.timedelta(days=1))
    _insert_llm_call(db, "dec-2", 0.31, called_at=month_start + dt.timedelta(days=2))
    _insert_llm_call(db, "dec-3", 0.05, called_at=month_start + dt.timedelta(days=3))
    # A call from PRIOR month — must NOT count.
    prior_month = (month_start - dt.timedelta(days=1)).replace(day=1)
    _insert_llm_call(db, "old-dec", 99.0, called_at=prior_month)

    # 2 executions accepted, 1 rejected.
    e1 = insert_execution(
        db,
        __exec_row(p1, "accepted"),
    )
    e2 = insert_execution(
        db,
        __exec_row(p2, "accepted"),
    )
    insert_execution(
        db,
        __exec_row(_insert_proposal(db, f3, "dec-4", kind="trade"), "rejected"),
    )

    # 2 fills (filled), 1 partially_filled, 1 cancelled.
    insert_order(db, __order_row(e1, "AAPL", "filled"))
    insert_order(db, __order_row(e2, "AAPL", "partially_filled"))
    insert_order(db, __order_row(e2, "AAPL", "cancelled"))

    # Day-end equity snapshot.
    insert_account_snapshot(
        db,
        AccountSnapshotRow(
            snapshot_at=now - dt.timedelta(minutes=2),
            equity=10_750.50,
            cash=5000.0,
            buying_power=10_000.0,
            long_market_value=5750.50,
            daypl=125.50,
        ),
    )

    reporter = DailyReporter(journal=journal, telegram=_RecordingNotifier())
    report = reporter.compose(now=now)

    assert report.filings_ingested == 3
    assert report.prefilter_counts == {
        ("accept", "item_code"): 2,
        ("reject", "duplicate"): 1,
    }
    assert report.proposals_by_kind == {"trade": 3, "no_trade": 1}
    assert report.proposals_executed == 2
    assert report.fills == 2  # filled + partially_filled
    assert report.day_end_equity == pytest.approx(10_750.50)
    # Spend MUST come from llm_calls, NOT proposal_reviews (D-049+ carry-fwd).
    assert report.mtd_inference_cost_usd == pytest.approx(0.42 + 0.31 + 0.05)


def test_report_uses_llm_calls_not_proposal_reviews_for_spend(
    db: str, journal: JournalRepo, now: dt.datetime
) -> None:
    """Critical S5.4 + S5.5 carry-forward: the journal's `llm_calls` is the
    canonical source for inference spend (NFR-8). `proposal_reviews.cost_usd`
    exists for analyzer telemetry but is NOT the spend source-of-truth.
    """
    _seed_prompt(db)
    f = _insert_filing(db, "0001-01", now, now)
    _insert_proposal(db, f, "dec-1", kind="trade", cost_usd=99.0)  # NOT counted

    month_start = dt.datetime(now.year, now.month, 1, tzinfo=dt.UTC)
    _insert_llm_call(db, "dec-1", 0.42, called_at=month_start + dt.timedelta(days=1))

    reporter = DailyReporter(journal=journal, telegram=_RecordingNotifier())
    report = reporter.compose(now=now)
    # only the llm_calls row counts; the proposal cost is not summed.
    assert report.mtd_inference_cost_usd == pytest.approx(0.42)


def test_report_includes_all_required_fields(
    db: str, journal: JournalRepo, now: dt.datetime
) -> None:
    """AC-2 summary: every required field is present in the Report dataclass
    (smoke check; numeric defaults to 0 when journal is empty)."""
    reporter = DailyReporter(journal=journal, telegram=_RecordingNotifier())
    report = reporter.compose(now=now)
    assert isinstance(report, Report)
    assert report.filings_ingested == 0
    assert report.proposals_executed == 0
    assert report.fills == 0
    assert report.day_end_equity is None  # no snapshot yet
    assert report.mtd_inference_cost_usd == 0.0


# ---------------------------------------------------------------------------
# AC-3: send via Telegram, optional email
# ---------------------------------------------------------------------------


class _RecordingNotifier:
    """Test double for `Notifier`."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, text: str) -> None:
        self.sent.append(text)


def test_compose_and_send_dispatches_to_telegram(
    db: str, journal: JournalRepo, now: dt.datetime
) -> None:
    tg = _RecordingNotifier()
    reporter = DailyReporter(journal=journal, telegram=tg)
    report = reporter.compose_and_send(now=now)

    assert isinstance(report, Report)
    assert len(tg.sent) == 1
    assert "Daily report" in tg.sent[0] or "filings" in tg.sent[0].lower()


def test_compose_and_send_dispatches_to_email_when_configured(
    db: str, journal: JournalRepo, now: dt.datetime
) -> None:
    tg = _RecordingNotifier()
    email = _RecordingNotifier()
    reporter = DailyReporter(journal=journal, telegram=tg, email=email)
    reporter.compose_and_send(now=now)
    assert len(tg.sent) == 1
    assert len(email.sent) == 1


def test_compose_and_send_skips_email_when_not_configured(
    db: str, journal: JournalRepo, now: dt.datetime
) -> None:
    tg = _RecordingNotifier()
    reporter = DailyReporter(journal=journal, telegram=tg, email=None)
    reporter.compose_and_send(now=now)
    assert len(tg.sent) == 1


def test_telegram_is_required(db: str, journal: JournalRepo) -> None:
    """The reporter MUST have a Telegram notifier — it's the operator's
    primary alert channel (S7.2)."""
    with pytest.raises(TypeError):
        DailyReporter(journal=journal)  # type: ignore[call-arg]


def test_notifier_protocol_signature() -> None:
    """`Notifier` is a Protocol with `send(text) -> None`."""
    from jobs.daily_report import Notifier as _N

    assert hasattr(_N, "send")


# ---------------------------------------------------------------------------
# AC-6: KS-1 cleared at next session start
# ---------------------------------------------------------------------------


def test_ks1_cleared_at_next_session_start(db: str, journal: JournalRepo) -> None:
    """Per S7.4 AC-2 + S8.2 dev-note: at 09:30 ET on the next trading day,
    any active `auto:KS-1` halt is automatically cleared."""
    # Friday 16:00 ET → Monday 09:30 ET (Sat/Sun skipped).
    fri_close = dt.datetime(2026, 4, 24, 20, 0, 0, tzinfo=dt.UTC)  # 16:00 ET
    insert_kill_switch_state(
        db,
        KillSwitchStateRow(
            set_at=fri_close,
            state="halted",
            reason="auto:KS-1",
            set_by="system",
            notes="daily-loss",
        ),
    )

    next_session = dt.datetime(2026, 4, 27, 13, 30, 0, tzinfo=dt.UTC)  # Mon 09:30 ET
    cleared = clear_auto_ks1_at_session_start(now=next_session, journal=journal)

    assert cleared is True
    state = journal.get_latest_kill_switch_state()
    assert state is not None
    assert state.state == "active"
    assert state.reason == "resume"
    assert state.set_by == "system"


def test_ks1_clear_does_nothing_when_halt_is_manual(db: str, journal: JournalRepo) -> None:
    """Manual halts (`manual:telegram`) are NEVER auto-cleared."""
    insert_kill_switch_state(
        db,
        KillSwitchStateRow(
            set_at=dt.datetime(2026, 4, 24, 20, 0, 0, tzinfo=dt.UTC),
            state="halted",
            reason="manual:telegram",
            set_by="operator",
            notes=None,
        ),
    )
    next_session = dt.datetime(2026, 4, 27, 13, 30, 0, tzinfo=dt.UTC)
    cleared = clear_auto_ks1_at_session_start(now=next_session, journal=journal)
    assert cleared is False
    assert journal.get_latest_kill_switch_state().state == "halted"


def test_ks1_clear_does_nothing_when_halt_is_ks2_or_ks3(db: str, journal: JournalRepo) -> None:
    """Only KS-1 (daily-loss) auto-clears at next session — KS-2 / KS-3
    persist until operator intervention."""
    insert_kill_switch_state(
        db,
        KillSwitchStateRow(
            set_at=dt.datetime(2026, 4, 24, 20, 0, 0, tzinfo=dt.UTC),
            state="halted",
            reason="auto:KS-2",
            set_by="system",
            notes=None,
        ),
    )
    next_session = dt.datetime(2026, 4, 27, 13, 30, 0, tzinfo=dt.UTC)
    cleared = clear_auto_ks1_at_session_start(now=next_session, journal=journal)
    assert cleared is False
    assert journal.get_latest_kill_switch_state().state == "halted"


def test_ks1_clear_only_fires_at_session_start_window(db: str, journal: JournalRepo) -> None:
    """The clear function is idempotent across a 5-minute window starting at
    09:30 ET — calling it at 14:00 UTC on a market day MUST NOT clear."""
    insert_kill_switch_state(
        db,
        KillSwitchStateRow(
            set_at=dt.datetime(2026, 4, 24, 20, 0, 0, tzinfo=dt.UTC),
            state="halted",
            reason="auto:KS-1",
            set_by="system",
            notes=None,
        ),
    )
    mid_session = dt.datetime(2026, 4, 27, 18, 0, 0, tzinfo=dt.UTC)  # 14:00 ET
    cleared = clear_auto_ks1_at_session_start(now=mid_session, journal=journal)
    assert cleared is False


# ---------------------------------------------------------------------------
# Helpers — direct row constructors for terse seeding
# ---------------------------------------------------------------------------


def __exec_row(proposal_id: int, decision: str) -> object:
    from journal.models import ExecutionRow

    return ExecutionRow(proposal_id=proposal_id, decision=decision)


def __order_row(execution_id: int, symbol: str, status: str) -> OrderRow:
    return OrderRow(
        execution_id=execution_id,
        role="entry",
        symbol=symbol,
        side="buy",
        order_type="market",
        qty=10,
        tif="DAY",
        broker_order_id=f"bro-{execution_id}-{symbol}-{status}",
        submitted_at=dt.datetime.now(dt.UTC),
        final_status=status,
        realized_fill_price=100.0 if status in ("filled", "partially_filled") else None,
        realized_fill_qty=10
        if status == "filled"
        else (5 if status == "partially_filled" else None),
        realized_fill_at=dt.datetime.now(dt.UTC)
        if status in ("filled", "partially_filled")
        else None,
    )
