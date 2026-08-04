"""KS-2 acknowledged-drawdown watermark tests.

Live-money problem: the 30-day trailing peak decays slowly, so after a
drawdown the KS-2 condition stays true for DAYS and the rule re-fires
every morning; the operator must /resume daily (a ritual carrying zero
new information — and a forgotten resume kills the day's sweep).

Fix under test: an operator resume of an auto:KS-2 halt persists an
acknowledgment watermark (drawdown pct + 30d peak at ack time) in the
resume row's `kill_switch_state.notes` JSON. While the feature is on
(`killswitch.ks2_reack_margin_pct` > 0) KS-2 is suppressed until the
drawdown deepens to `acked_dd + margin` (percentage points); a genuine
new equity high above the acked peak expires the watermark.

Same-day suppression (commit 5133cbd) remains untouched underneath.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pytest

from config.loader import KillSwitchConfig
from journal.migrate import apply_migrations
from journal.models import AccountSnapshotRow
from journal.repo import JournalRepo, insert_account_snapshot
from killswitch.api import KillSwitch
from killswitch.auto_halts import AutoHaltEvaluator, Halt
from killswitch.telegram_bot import BotCore

CHAT_ID = 1


def _cfg(*, reack: float = 0.0) -> KillSwitchConfig:
    return KillSwitchConfig(
        ks1_daily_loss_pct=0.03,
        ks2_trailing_dd_pct=0.10,
        ks3_consecutive_losses=6,
        ks2_reack_margin_pct=reack,
    )


@pytest.fixture
def journal(tmp_path: Path) -> JournalRepo:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return JournalRepo(str(db_path))


def _ks_at(journal: JournalRepo, at: dt.datetime) -> KillSwitch:
    """KillSwitch whose writes carry a controlled timestamp — the default
    now_fn stamps the REAL clock, which would break set_at ordering
    against this file's 2026 timeline."""
    return KillSwitch(journal, now_fn=lambda: at)


def _snapshot(
    journal: JournalRepo, *, at: dt.datetime, equity: float, daypl: float = 0.0
) -> None:
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


def _evaluate(
    journal: JournalRepo, now: dt.datetime, cfg: KillSwitchConfig
) -> tuple[list[Halt], KillSwitch]:
    ks = _ks_at(journal, now)
    halts = AutoHaltEvaluator().evaluate(now=now, journal=journal, ks=ks, cfg=cfg)
    return halts, ks


def _bot_resume(journal: JournalRepo, at: dt.datetime) -> None:
    """Operator resume through the real Telegram-bot path (two-step
    /resume + yes) with a controlled clock, so the resume row carries
    whatever notes the bot writes."""
    bot = BotCore(
        ks=_ks_at(journal, at), journal=journal, allowed_chat_id=CHAT_ID, clock=lambda: at
    )
    assert bot.handle_command(CHAT_ID, "/resume") is not None
    reply = bot.handle_command(CHAT_ID, "yes")
    assert reply is not None and "resumed" in reply.text


# Timeline anchors: 16:00 UTC = 12:00 ET, so consecutive UTC days are
# consecutive ET days (same-day suppression never masks the watermark
# behavior under test on day 2+).
PEAK_AT = dt.datetime(2026, 6, 20, 16, 0, 0, tzinfo=dt.UTC)
HALT_DAY = dt.datetime(2026, 6, 25, 16, 0, 0, tzinfo=dt.UTC)


def _halt_and_ack(
    journal: JournalRepo,
    *,
    reack: float,
    peak: float = 100_000.0,
    dd_equity: float = 87_500.0,
) -> None:
    """Seed peak → 12.5% drawdown → KS-2 auto-halt → operator bot resume."""
    _snapshot(journal, at=PEAK_AT, equity=peak)
    _snapshot(journal, at=HALT_DAY - dt.timedelta(minutes=5), equity=dd_equity)
    halts, ks = _evaluate(journal, HALT_DAY, _cfg(reack=reack))
    assert any(h.rule == "KS-2" and h.fired for h in halts)
    assert ks.state().reason == "auto:KS-2"
    _bot_resume(journal, at=HALT_DAY + dt.timedelta(minutes=30))
    assert ks.is_halted() is False


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _evaluate_capturing(
    journal: JournalRepo, now: dt.datetime, cfg: KillSwitchConfig
) -> tuple[list[Halt], KillSwitch, list[logging.LogRecord]]:
    handler = _Capture()
    logger = logging.getLogger("killswitch.auto_halts")
    prior = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        halts, ks = _evaluate(journal, now, cfg)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior)
    return halts, ks, handler.records


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_off_and_bounds() -> None:
    """Old config (no new key) parses with margin 0.0 = feature OFF;
    valid range is 0–10 percentage points."""
    cfg = KillSwitchConfig(
        ks1_daily_loss_pct=0.03, ks2_trailing_dd_pct=0.10, ks3_consecutive_losses=6
    )
    assert cfg.ks2_reack_margin_pct == 0.0
    assert _cfg(reack=10.0).ks2_reack_margin_pct == 10.0
    with pytest.raises(ValueError):
        _cfg(reack=10.5)
    with pytest.raises(ValueError):
        _cfg(reack=-0.1)


# ---------------------------------------------------------------------------
# Default-off back-compat
# ---------------------------------------------------------------------------


def test_feature_off_still_refires_next_day(journal: JournalRepo) -> None:
    """margin=0.0 (default): behavior identical to today — the standing
    drawdown re-fires on the next ET day even though the bot wrote an
    acknowledgment watermark on the resume row."""
    _halt_and_ack(journal, reack=0.0)
    day2 = HALT_DAY + dt.timedelta(days=1)
    _snapshot(journal, at=day2 - dt.timedelta(minutes=5), equity=87_500.0)
    halts, ks = _evaluate(journal, day2, _cfg(reack=0.0))
    assert any(h.rule == "KS-2" and h.fired for h in halts)
    assert ks.state().reason == "auto:KS-2"


# ---------------------------------------------------------------------------
# Watermark persistence (bot writes it on the resume row)
# ---------------------------------------------------------------------------


def test_bot_resume_after_ks2_halt_writes_watermark(journal: JournalRepo) -> None:
    _halt_and_ack(journal, reack=3.0)
    row = journal.get_latest_kill_switch_state()
    assert row is not None
    assert row.state == "active" and row.set_by == "operator"
    assert row.notes is not None
    payload = json.loads(row.notes)
    assert payload["detail"] == "via /resume yes-confirm"
    ack = payload["ks2_ack"]
    assert ack["dd_pct"] == pytest.approx(0.125)
    assert ack["peak"] == pytest.approx(100_000.0)


def test_bot_resume_after_manual_halt_writes_plain_notes(journal: JournalRepo) -> None:
    """No auto:KS-2 halt in the unresolved window → no watermark JSON;
    resume notes stay byte-identical to today."""
    t0 = dt.datetime(2026, 6, 25, 16, 0, 0, tzinfo=dt.UTC)
    ticks = iter(range(1, 100))

    def clock() -> dt.datetime:  # strictly increasing: set_at is the PK
        return t0 + dt.timedelta(seconds=next(ticks))

    bot = BotCore(
        ks=KillSwitch(journal, now_fn=clock),
        journal=journal,
        allowed_chat_id=CHAT_ID,
        clock=clock,
    )
    bot.handle_command(CHAT_ID, "/halt")
    bot.handle_command(CHAT_ID, "/resume")
    bot.handle_command(CHAT_ID, "yes")
    row = journal.get_latest_kill_switch_state()
    assert row is not None
    assert row.state == "active"
    assert row.notes == "via /resume yes-confirm"


# ---------------------------------------------------------------------------
# Suppression while inside the acked margin
# ---------------------------------------------------------------------------


def test_acked_drawdown_suppressed_next_day_with_distinct_log(
    journal: JournalRepo,
) -> None:
    """halt → resume → same drawdown NEXT day: suppressed (fired=False),
    switch stays active, and the suppression log carries the distinct
    acked-watermark detail."""
    _halt_and_ack(journal, reack=3.0)
    day2 = HALT_DAY + dt.timedelta(days=1)
    _snapshot(journal, at=day2 - dt.timedelta(minutes=5), equity=87_500.0)

    halts, ks, records = _evaluate_capturing(journal, day2, _cfg(reack=3.0))

    ks2 = [h for h in halts if h.rule == "KS-2"]
    assert len(ks2) == 1
    assert ks2[0].fired is False
    assert ks2[0].suppressed_by == "acked_watermark"
    assert ks.is_halted() is False

    suppressions = [
        r
        for r in records
        if getattr(r, "event", None) == "killswitch.refire_suppressed"
        and getattr(r, "mode", None) == "acked_watermark"
    ]
    assert len(suppressions) == 1
    assert suppressions[0].rule == "KS-2"
    assert suppressions[0].acked_dd == pytest.approx(0.125)
    assert suppressions[0].current_dd == pytest.approx(0.125)


def test_fires_at_exact_acked_plus_margin_boundary(journal: JournalRepo) -> None:
    """>= fires: acked_dd=0.125, margin=6.25pp → re-fire threshold is
    exactly dd=0.1875 (all values binary-exact — no epsilon fudging)."""
    _halt_and_ack(journal, reack=6.25)

    # Day 2: dd 0.187 — just inside the margin → suppressed.
    day2 = HALT_DAY + dt.timedelta(days=1)
    _snapshot(journal, at=day2 - dt.timedelta(minutes=5), equity=81_300.0)
    halts, ks = _evaluate(journal, day2, _cfg(reack=6.25))
    assert any(h.rule == "KS-2" and not h.fired for h in halts)
    assert ks.is_halted() is False

    # Day 3: dd exactly 0.1875 == acked + margin → FIRES.
    day3 = HALT_DAY + dt.timedelta(days=2)
    _snapshot(journal, at=day3 - dt.timedelta(minutes=5), equity=81_250.0)
    halts, ks = _evaluate(journal, day3, _cfg(reack=6.25))
    assert any(h.rule == "KS-2" and h.fired for h in halts)
    assert ks.state().reason == "auto:KS-2"


# ---------------------------------------------------------------------------
# Expiry: a genuine new high resets the regime
# ---------------------------------------------------------------------------


def test_new_peak_expires_watermark_and_fresh_drawdown_fires(
    journal: JournalRepo,
) -> None:
    """Equity makes a new high above the acked peak → watermark expires;
    a later fresh >=10% drawdown from the NEW peak halts normally even
    though its dd (10.9%) is below acked_dd + margin (15.5%)."""
    _halt_and_ack(journal, reack=3.0)

    day3 = HALT_DAY + dt.timedelta(days=2)
    _snapshot(journal, at=day3, equity=101_000.0)  # new high > acked peak

    day5 = HALT_DAY + dt.timedelta(days=4)
    _snapshot(journal, at=day5 - dt.timedelta(minutes=5), equity=90_000.0)
    halts, ks = _evaluate(journal, day5, _cfg(reack=3.0))
    assert any(h.rule == "KS-2" and h.fired for h in halts)
    assert ks.state().reason == "auto:KS-2"


def test_decayed_peak_does_not_expire_watermark(journal: JournalRepo) -> None:
    """The acked 100k peak rolls OUT of the 30-day window (decayed
    rolling peak = 87.5k) — the ack must still stand: a >=10% drawdown
    vs the decayed peak stays suppressed while inside the margin."""
    _halt_and_ack(journal, reack=3.0)
    # Equity drifts sideways below the acked peak (does NOT expire the ack).
    _snapshot(journal, at=HALT_DAY + dt.timedelta(days=3), equity=87_500.0)

    later = HALT_DAY + dt.timedelta(days=32)  # PEAK_AT now outside 30d window
    _snapshot(journal, at=later - dt.timedelta(minutes=5), equity=77_000.0)
    # rolling peak is now 87_500 → dd = 10500/87500 = 12% >= 10% threshold,
    # and 0.12 < acked 0.125 + 0.03 → still acknowledged pain.
    halts, ks = _evaluate(journal, later, _cfg(reack=3.0))
    assert any(h.rule == "KS-2" and not h.fired for h in halts)
    assert ks.is_halted() is False


# ---------------------------------------------------------------------------
# Durability + isolation
# ---------------------------------------------------------------------------


def test_watermark_survives_process_restart(journal: JournalRepo) -> None:
    """Fresh JournalRepo/KillSwitch/evaluator objects over the same DB
    file (simulated restart) still see the watermark."""
    _halt_and_ack(journal, reack=3.0)
    day2 = HALT_DAY + dt.timedelta(days=1)
    _snapshot(journal, at=day2 - dt.timedelta(minutes=5), equity=87_500.0)

    fresh_journal = JournalRepo(journal.db_path)
    fresh_ks = _ks_at(fresh_journal, day2)
    halts = AutoHaltEvaluator().evaluate(
        now=day2, journal=fresh_journal, ks=fresh_ks, cfg=_cfg(reack=3.0)
    )
    assert any(h.rule == "KS-2" and not h.fired for h in halts)
    assert fresh_ks.is_halted() is False


def test_ks1_unaffected_by_active_ks2_watermark(journal: JournalRepo) -> None:
    """An active KS-2 watermark never shields KS-1: a next-day daily-loss
    breach halts normally while KS-2 stays suppressed."""
    _halt_and_ack(journal, reack=3.0)
    day2 = HALT_DAY + dt.timedelta(days=1)
    # daypl -4000 on SOD 91_500 → 4.37% daily loss >= 3%; equity 87_500
    # keeps the KS-2 condition true (dd 12.5%, inside the acked margin).
    _snapshot(
        journal, at=day2 - dt.timedelta(minutes=5), equity=87_500.0, daypl=-4_000.0
    )
    halts, ks = _evaluate(journal, day2, _cfg(reack=3.0))
    assert any(h.rule == "KS-1" and h.fired for h in halts)
    assert any(h.rule == "KS-2" and not h.fired for h in halts)
    assert ks.state().reason == "auto:KS-1"
