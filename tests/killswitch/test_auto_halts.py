"""S7.4 — Automatic halt evaluator tests (KS-1, KS-2, KS-3).

Architecture references: PRD §5 (KS thresholds), §2.8 kill-switch,
ADR-004 (auto + manual write the same kill_switch_state shape).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from config.loader import KillSwitchConfig
from journal.migrate import apply_migrations
from journal.models import (
    AccountSnapshotRow,
    ExecutionRow,
    FilingRow,
    OrderRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    JournalRepo,
    insert_account_snapshot,
    insert_execution,
    insert_filing,
    insert_order,
    insert_prompt,
    insert_proposal,
)
from killswitch.api import KillSwitch
from killswitch.auto_halts import AutoHaltEvaluator, Halt


def _cfg(
    *,
    ks1: float = 0.03,
    ks2: float = 0.10,
    ks3: int = 6,
) -> KillSwitchConfig:
    return KillSwitchConfig(
        ks1_daily_loss_pct=ks1,
        ks2_trailing_dd_pct=ks2,
        ks3_consecutive_losses=ks3,
    )


@pytest.fixture
def journal(tmp_path: Path) -> JournalRepo:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return JournalRepo(str(db_path))


@pytest.fixture
def ks(journal: JournalRepo) -> KillSwitch:
    return KillSwitch(journal)


def _snapshot(journal: JournalRepo, *, at: dt.datetime, equity: float, daypl: float) -> None:
    insert_account_snapshot(
        journal.db_path,
        AccountSnapshotRow(
            snapshot_at=at,
            equity=equity,
            cash=equity,
            buying_power=equity,
            long_market_value=0.0,
            daypl=daypl,
        ),
    )


def _seed_proposal(journal: JournalRepo, decision_id: str) -> int:
    """Create a filing+prompt+proposal so executions FK can resolve."""
    db = journal.db_path
    fid = insert_filing(
        db,
        FilingRow(
            accession_number=f"ACC-{decision_id}",
            cik=1234567,
            form_type="8-K",
            filed_at=dt.datetime(2026, 4, 1, 10, 0, 0),
            fetched_at=dt.datetime(2026, 4, 1, 10, 0, 0),
            raw_text_path=f"/var/lib/quinn/raw/{decision_id}.txt",
            content_hash=f"h-{decision_id}",
        ),
    )
    insert_prompt(
        db,
        PromptRow(
            prompt_version=f"sonnet@{decision_id}",
            name="sonnet",
            file_path="src/prompts/sonnet.txt",
            content_hash=f"h-{decision_id}",
        ),
    )
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id=decision_id,
            model_id="claude-sonnet-4-6",
            prompt_version=f"sonnet@{decision_id}",
            raw_response="{}",
            kind="trade_proposal",
            input_tokens=10,
            output_tokens=10,
            latency_ms=1,
            cost_usd=0.0,
        ),
    )
    return pid


def _closed_trade(
    journal: JournalRepo,
    *,
    decision_id: str,
    symbol: str,
    entry_price: float,
    exit_price: float,
    qty: int,
    closed_at: dt.datetime,
) -> None:
    """Create a long round-trip: BUY at entry → SELL (stop or take_profit) at exit."""
    pid = _seed_proposal(journal, decision_id)
    eid = insert_execution(
        journal.db_path,
        ExecutionRow(proposal_id=pid, decision="accept", realized_size_pct=0.05),
    )
    # entry order (buy)
    insert_order(
        journal.db_path,
        OrderRow(
            execution_id=eid,
            role="entry",
            symbol=symbol,
            side="buy",
            order_type="market",
            qty=qty,
            tif="day",
            broker_order_id=f"ord-{decision_id}-entry",
            submitted_at=closed_at - dt.timedelta(hours=1),
            final_status="filled",
            realized_fill_price=entry_price,
            realized_fill_qty=qty,
            realized_fill_at=closed_at - dt.timedelta(hours=1),
            realized_fee=0.0,
        ),
    )
    # exit order (sell — categorized as 'stop' for losses, 'take_profit' for wins;
    # the evaluator should not care which)
    role = "take_profit" if exit_price > entry_price else "stop"
    insert_order(
        journal.db_path,
        OrderRow(
            execution_id=eid,
            role=role,
            symbol=symbol,
            side="sell",
            order_type="market",
            qty=qty,
            tif="day",
            broker_order_id=f"ord-{decision_id}-exit",
            submitted_at=closed_at,
            final_status="filled",
            realized_fill_price=exit_price,
            realized_fill_qty=qty,
            realized_fill_at=closed_at,
            realized_fee=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# KS-1 — daily loss
# ---------------------------------------------------------------------------


def test_ks1_fires_when_daily_loss_breach(journal: JournalRepo, ks: KillSwitch) -> None:
    now = dt.datetime(2026, 4, 28, 15, 30, 0, tzinfo=dt.UTC)
    # SOD equity = 1000; daypl = -40 → -4% > -3% threshold
    _snapshot(journal, at=now - dt.timedelta(minutes=5), equity=960.0, daypl=-40.0)
    halts = AutoHaltEvaluator().evaluate(now=now, journal=journal, ks=ks, cfg=_cfg(ks1=0.03))
    assert any(h.rule == "KS-1" for h in halts)
    assert ks.is_halted() is True
    assert ks.state().reason == "auto:KS-1"


def test_ks1_does_not_fire_below_threshold(journal: JournalRepo, ks: KillSwitch) -> None:
    now = dt.datetime(2026, 4, 28, 15, 30, 0, tzinfo=dt.UTC)
    # daypl = -20 on SOD equity 1000 → -2% (below 3% threshold)
    _snapshot(journal, at=now - dt.timedelta(minutes=5), equity=980.0, daypl=-20.0)
    halts = AutoHaltEvaluator().evaluate(now=now, journal=journal, ks=ks, cfg=_cfg(ks1=0.03))
    assert all(h.rule != "KS-1" for h in halts)
    assert ks.is_halted() is False


def test_ks1_clears_at_session_start(journal: JournalRepo, ks: KillSwitch) -> None:
    """PRD §5.3: KS-1 halts persist until next session start.

    Implementation: KillSwitch.resume(set_by="system", notes="session_start")
    at session start clears the halt; the evaluator on the new day's snapshot
    (daypl reset to ~0) does not re-fire.
    """
    day1 = dt.datetime(2026, 4, 28, 15, 30, 0, tzinfo=dt.UTC)
    _snapshot(journal, at=day1, equity=960.0, daypl=-40.0)
    ev = AutoHaltEvaluator()
    ev.evaluate(now=day1, journal=journal, ks=ks, cfg=_cfg(ks1=0.03))
    assert ks.is_halted() is True
    # Simulate next session start: operator/system resume + new day's snapshot
    day2 = dt.datetime(2026, 4, 29, 9, 30, 0, tzinfo=dt.UTC)
    ks.resume(set_by="system", notes="session_start")
    _snapshot(journal, at=day2, equity=960.0, daypl=0.0)
    halts = ev.evaluate(now=day2, journal=journal, ks=ks, cfg=_cfg(ks1=0.03))
    assert all(h.rule != "KS-1" for h in halts)
    assert ks.is_halted() is False


def test_ks1_incident_2026_07_08_daypl_blindness_cured(
    journal: JournalRepo, ks: KillSwitch
) -> None:
    """Regression (2026-07-08 incident): the broker normalizer computed
    daypl from a nonexistent `equity_previous_close` field, so every
    snapshot carried daypl=0.0 and KS-1 was structurally blind.

    Reconstruct: SOD equity 3600 → intraday equity 3475 (loss 3.47%)
    against ks1_daily_loss_pct=0.03. With the old broken daypl=0.0 the
    evaluator returns []; with the fixed daypl=-125.0 it returns KS-1.
    """
    ev = AutoHaltEvaluator()
    cfg = _cfg(ks1=0.03)
    now = dt.datetime(2026, 7, 8, 15, 30, 0, tzinfo=dt.UTC)

    # OLD behavior: daypl always 0.0 → KS-1 never fires (the blindness).
    _snapshot(journal, at=now - dt.timedelta(minutes=10), equity=3475.0, daypl=0.0)
    assert ev._eval_ks1(now, journal, cfg) == []

    # FIXED behavior: daypl = equity - last_equity = 3475 - 3600 = -125.
    _snapshot(journal, at=now - dt.timedelta(minutes=5), equity=3475.0, daypl=-125.0)
    halts = ev._eval_ks1(now, journal, cfg)
    assert [h.rule for h in halts] == ["KS-1"]
    assert halts[0].reason == "auto:KS-1"


# ---------------------------------------------------------------------------
# KS-2 — 30d trailing drawdown
# ---------------------------------------------------------------------------


def test_ks2_fires_on_trailing_drawdown(journal: JournalRepo, ks: KillSwitch) -> None:
    now = dt.datetime(2026, 4, 28, 15, 30, 0, tzinfo=dt.UTC)
    # 30d peak = 1200; current = 1080 → 10% drawdown exactly
    _snapshot(journal, at=now - dt.timedelta(days=20), equity=1200.0, daypl=0.0)
    _snapshot(journal, at=now - dt.timedelta(minutes=5), equity=1079.0, daypl=-1.0)
    halts = AutoHaltEvaluator().evaluate(now=now, journal=journal, ks=ks, cfg=_cfg(ks2=0.10))
    assert any(h.rule == "KS-2" for h in halts)
    assert ks.is_halted() is True
    assert ks.state().reason == "auto:KS-2"


def test_ks2_ignores_pre_30d_peak(journal: JournalRepo, ks: KillSwitch) -> None:
    """A peak older than 30d does not count toward trailing drawdown."""
    now = dt.datetime(2026, 4, 28, 15, 30, 0, tzinfo=dt.UTC)
    _snapshot(journal, at=now - dt.timedelta(days=45), equity=2000.0, daypl=0.0)
    _snapshot(journal, at=now - dt.timedelta(days=20), equity=1100.0, daypl=0.0)
    _snapshot(journal, at=now - dt.timedelta(minutes=5), equity=1050.0, daypl=-50.0)
    halts = AutoHaltEvaluator().evaluate(now=now, journal=journal, ks=ks, cfg=_cfg(ks2=0.10))
    # Within-30d peak is 1100 → drawdown = (1100-1050)/1100 ≈ 4.5% < 10%
    assert all(h.rule != "KS-2" for h in halts)


def test_ks2_persists_until_operator_resume(journal: JournalRepo, ks: KillSwitch) -> None:
    now = dt.datetime(2026, 4, 28, 15, 30, 0, tzinfo=dt.UTC)
    _snapshot(journal, at=now - dt.timedelta(days=10), equity=2000.0, daypl=0.0)
    _snapshot(journal, at=now, equity=1700.0, daypl=-100.0)  # 15% DD
    ev = AutoHaltEvaluator()
    ev.evaluate(now=now, journal=journal, ks=ks, cfg=_cfg(ks2=0.10))
    assert ks.is_halted() is True
    # Re-evaluate the next minute → still halted, no spam (idempotent)
    halts = ev.evaluate(
        now=now + dt.timedelta(minutes=1), journal=journal, ks=ks, cfg=_cfg()
    )
    # Already halted by KS-2 → no new auto:KS-* row inserted
    for h in halts:
        if isinstance(h, Halt) and h.rule == "KS-2":
            assert h.fired is False


# ---------------------------------------------------------------------------
# KS-3 — consecutive losing trades
# ---------------------------------------------------------------------------


def test_ks3_fires_on_6_consecutive_losses(journal: JournalRepo, ks: KillSwitch) -> None:
    base = dt.datetime(2026, 4, 28, 10, 0, 0, tzinfo=dt.UTC)
    for i in range(6):
        _closed_trade(
            journal,
            decision_id=f"d{i}",
            symbol=f"SYM{i}",
            entry_price=10.0,
            exit_price=9.0,  # loss
            qty=10,
            closed_at=base + dt.timedelta(hours=i),
        )
    now = base + dt.timedelta(hours=10)
    halts = AutoHaltEvaluator().evaluate(now=now, journal=journal, ks=ks, cfg=_cfg(ks3=6))
    assert any(h.rule == "KS-3" for h in halts)
    assert ks.is_halted() is True
    assert ks.state().reason == "auto:KS-3"


def test_ks3_does_not_fire_with_intervening_win(journal: JournalRepo, ks: KillSwitch) -> None:
    base = dt.datetime(2026, 4, 28, 10, 0, 0, tzinfo=dt.UTC)
    # 5 losses, then 1 win, then 1 loss = recent 6 = [loss,win,loss,loss,loss,loss]
    seq = [9.0] * 5 + [11.0] + [9.0]
    for i, exit_price in enumerate(seq):
        _closed_trade(
            journal,
            decision_id=f"d{i}",
            symbol=f"SYM{i}",
            entry_price=10.0,
            exit_price=exit_price,
            qty=10,
            closed_at=base + dt.timedelta(hours=i),
        )
    now = base + dt.timedelta(hours=10)
    halts = AutoHaltEvaluator().evaluate(now=now, journal=journal, ks=ks, cfg=_cfg(ks3=6))
    assert all(h.rule != "KS-3" for h in halts)
    assert ks.is_halted() is False


def test_ks3_does_not_fire_with_fewer_than_threshold_trades(
    journal: JournalRepo, ks: KillSwitch
) -> None:
    base = dt.datetime(2026, 4, 28, 10, 0, 0, tzinfo=dt.UTC)
    for i in range(5):  # only 5 losses, threshold is 6
        _closed_trade(
            journal,
            decision_id=f"d{i}",
            symbol=f"SYM{i}",
            entry_price=10.0,
            exit_price=9.0,
            qty=10,
            closed_at=base + dt.timedelta(hours=i),
        )
    halts = AutoHaltEvaluator().evaluate(
        now=base + dt.timedelta(hours=10), journal=journal, ks=ks, cfg=_cfg(ks3=6)
    )
    assert all(h.rule != "KS-3" for h in halts)


# ---------------------------------------------------------------------------
# Cross-cutting: idempotency, callability after closed trade
# ---------------------------------------------------------------------------


def test_already_halted_no_spam(journal: JournalRepo, ks: KillSwitch) -> None:
    """AC-6: re-evaluating while halted by an auto:KS-* reason adds no new rows."""
    now = dt.datetime(2026, 4, 28, 15, 30, 0, tzinfo=dt.UTC)
    _snapshot(journal, at=now, equity=960.0, daypl=-40.0)
    ev = AutoHaltEvaluator()
    ev.evaluate(now=now, journal=journal, ks=ks, cfg=_cfg())
    # count rows after first halt
    from journal.repo import connect

    with connect(journal.db_path) as conn:
        n_before = conn.execute("SELECT COUNT(*) FROM kill_switch_state").fetchone()[0]
    # Re-evaluate three times; no new rows appended
    for _ in range(3):
        ev.evaluate(now=now + dt.timedelta(seconds=60), journal=journal, ks=ks, cfg=_cfg())
    with connect(journal.db_path) as conn:
        n_after = conn.execute("SELECT COUNT(*) FROM kill_switch_state").fetchone()[0]
    assert n_after == n_before


def test_evaluator_called_after_closed_trade(journal: JournalRepo, ks: KillSwitch) -> None:
    """AC-5: the evaluator can be triggered explicitly after a close — tested by
    constructing a 6th losing trade then calling evaluate exactly once."""
    base = dt.datetime(2026, 4, 28, 10, 0, 0, tzinfo=dt.UTC)
    for i in range(5):
        _closed_trade(
            journal,
            decision_id=f"d{i}",
            symbol=f"SYM{i}",
            entry_price=10.0,
            exit_price=9.0,
            qty=10,
            closed_at=base + dt.timedelta(hours=i),
        )
    ev = AutoHaltEvaluator()
    halts_before = ev.evaluate(
        now=base + dt.timedelta(hours=6), journal=journal, ks=ks, cfg=_cfg(ks3=6)
    )
    assert all(h.rule != "KS-3" for h in halts_before)
    # Reconciler reports the 6th close
    _closed_trade(
        journal,
        decision_id="d5",
        symbol="SYM5",
        entry_price=10.0,
        exit_price=9.0,
        qty=10,
        closed_at=base + dt.timedelta(hours=6),
    )
    halts_after = ev.evaluate(
        now=base + dt.timedelta(hours=7), journal=journal, ks=ks, cfg=_cfg(ks3=6)
    )
    assert any(h.rule == "KS-3" for h in halts_after)
    assert ks.is_halted() is True


# ---------------------------------------------------------------------------
# Operator-resume re-fire suppression (2026-07-14 clay incident: KS-1
# halted 17:47:41, operator resumed 17:48:10, evaluator re-fired 17:48:42
# because the daily-loss condition was still true; box sat halted 3 days
# while the operator believed they had resumed).
# ---------------------------------------------------------------------------


def _ks_at(journal: JournalRepo, at: dt.datetime) -> KillSwitch:
    """KillSwitch whose writes carry a controlled timestamp (the default
    now_fn stamps the REAL clock, which would defeat the same-ET-day
    comparison in tests)."""
    return KillSwitch(journal, now_fn=lambda: at)


def test_operator_resume_suppresses_same_day_ks1_refire(
    journal: JournalRepo, ks: KillSwitch
) -> None:
    """(1) halt → operator resume → same-day condition still true → NO
    re-halt, and the suppression is logged as
    `killswitch.refire_suppressed`."""
    import logging

    now = dt.datetime(2026, 7, 14, 21, 48, 42, tzinfo=dt.UTC)  # 17:48:42 ET
    _snapshot(journal, at=now - dt.timedelta(minutes=5), equity=960.0, daypl=-40.0)

    # 17:47:41 ET — KS-1 fires.
    halt_at = dt.datetime(2026, 7, 14, 21, 47, 41, tzinfo=dt.UTC)
    _ks_at(journal, halt_at).halt(
        reason="auto:KS-1", set_by="system", notes="daypl breach"
    )
    # 17:48:10 ET — the operator resumes: they have SEEN the condition.
    resume_at = dt.datetime(2026, 7, 14, 21, 48, 10, tzinfo=dt.UTC)
    _ks_at(journal, resume_at).resume(set_by="operator")
    assert ks.is_halted() is False

    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture(level=logging.INFO)
    named_logger = logging.getLogger("killswitch.auto_halts")
    prior_level = named_logger.level
    named_logger.setLevel(logging.INFO)
    named_logger.addHandler(handler)
    try:
        halts = AutoHaltEvaluator().evaluate(
            now=now, journal=journal, ks=ks, cfg=_cfg(ks1=0.03)
        )
    finally:
        named_logger.removeHandler(handler)
        named_logger.setLevel(prior_level)

    # The condition is still surfaced (fired=False) but NOT re-halted.
    ks1 = [h for h in halts if h.rule == "KS-1"]
    assert len(ks1) == 1
    assert ks1[0].fired is False
    assert ks.is_halted() is False
    suppressions = [
        r for r in captured
        if getattr(r, "event", None) == "killswitch.refire_suppressed"
    ]
    assert len(suppressions) == 1
    assert suppressions[0].rule == "KS-1"


def test_next_et_day_breach_fires_normally_after_resume(
    journal: JournalRepo, ks: KillSwitch
) -> None:
    """(2) The suppression dies at the ET date rollover: a fresh breach
    the NEXT day halts normally."""
    resume_day = dt.datetime(2026, 7, 14, 21, 48, 10, tzinfo=dt.UTC)
    _ks_at(
        journal, resume_day - dt.timedelta(seconds=30)
    ).halt(reason="auto:KS-1", set_by="system", notes="daypl breach")
    _ks_at(journal, resume_day).resume(set_by="operator")

    # Next ET day, fresh breach.
    now = dt.datetime(2026, 7, 15, 15, 30, 0, tzinfo=dt.UTC)
    _snapshot(journal, at=now - dt.timedelta(minutes=5), equity=960.0, daypl=-40.0)

    halts = AutoHaltEvaluator().evaluate(
        now=now, journal=journal, ks=ks, cfg=_cfg(ks1=0.03)
    )

    assert any(h.rule == "KS-1" and h.fired for h in halts)
    assert ks.is_halted() is True
    assert ks.state().reason == "auto:KS-1"


def test_ks2_still_fires_same_day_after_ks1_resume(
    journal: JournalRepo, ks: KillSwitch
) -> None:
    """(3) Suppression is scoped PER RULE: a same-day KS-1 resume does
    not shield a KS-2 breach."""
    now = dt.datetime(2026, 7, 14, 21, 48, 42, tzinfo=dt.UTC)
    # 30d peak 1200 vs current 960 → 20% drawdown > 10% threshold; also
    # keep the KS-1 condition true so the test proves per-rule scoping.
    _snapshot(
        journal,
        at=now - dt.timedelta(days=3),
        equity=1200.0,
        daypl=0.0,
    )
    _snapshot(journal, at=now - dt.timedelta(minutes=5), equity=960.0, daypl=-40.0)

    _ks_at(
        journal, now - dt.timedelta(seconds=61)
    ).halt(reason="auto:KS-1", set_by="system", notes="daypl breach")
    _ks_at(journal, now - dt.timedelta(seconds=32)).resume(set_by="operator")

    halts = AutoHaltEvaluator().evaluate(
        now=now, journal=journal, ks=ks, cfg=_cfg(ks1=0.03, ks2=0.10)
    )

    # KS-1 suppressed; KS-2 fired.
    assert any(h.rule == "KS-1" and not h.fired for h in halts)
    assert any(h.rule == "KS-2" and h.fired for h in halts)
    assert ks.is_halted() is True
    assert ks.state().reason == "auto:KS-2"


def test_system_resume_does_not_suppress_refire(
    journal: JournalRepo, ks: KillSwitch
) -> None:
    """A SYSTEM resume (session_start) is not an operator acceptance —
    the same-day re-fire contract is unchanged for it."""
    now = dt.datetime(2026, 7, 14, 21, 48, 42, tzinfo=dt.UTC)
    _snapshot(journal, at=now - dt.timedelta(minutes=5), equity=960.0, daypl=-40.0)
    _ks_at(
        journal, now - dt.timedelta(seconds=61)
    ).halt(reason="auto:KS-1", set_by="system", notes="daypl breach")
    _ks_at(journal, now - dt.timedelta(seconds=32)).resume(
        set_by="system", notes="session_start"
    )

    halts = AutoHaltEvaluator().evaluate(
        now=now, journal=journal, ks=ks, cfg=_cfg(ks1=0.03)
    )

    assert any(h.rule == "KS-1" and h.fired for h in halts)
    assert ks.is_halted() is True
