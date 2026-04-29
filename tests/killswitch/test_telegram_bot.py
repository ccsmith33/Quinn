"""S7.2 — Telegram bot core tests.

The bot core (`src/killswitch/telegram_bot.py`) is pure logic over a
`KillSwitch` + `JournalRepo` + injected clock; it does not import or touch
the python-telegram-bot library. The thin process entry point at
`src/bot.py` is the only place that talks to Telegram.

Architecture references: ADR-004 §Telegram bot, FR-32 (60s confirmation
budget), FR-34 (status surface), NFR-15 (no secrets in logs), NFR-17
(Telegram-only egress).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.models import (
    AccountSnapshotRow,
    FilingRow,
    LlmCallRow,
    PositionRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    JournalRepo,
    insert_account_snapshot,
    insert_filing,
    insert_llm_call,
    insert_position,
    insert_prompt,
    insert_proposal,
)
from killswitch.api import KillSwitch
from killswitch.telegram_bot import BotCore, BotReply


@pytest.fixture
def journal(tmp_path: Path) -> JournalRepo:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return JournalRepo(str(db_path))


@pytest.fixture
def ks(journal: JournalRepo) -> KillSwitch:
    return KillSwitch(journal)


class _Clock:
    """Manually-advanced clock for deterministic time-based tests."""

    def __init__(self, t0: dt.datetime) -> None:
        self.t = t0

    def __call__(self) -> dt.datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = self.t + dt.timedelta(seconds=seconds)


def _bot(
    ks: KillSwitch,
    journal: JournalRepo,
    *,
    allowed_chat_id: int = 42,
    clock: _Clock | None = None,
) -> BotCore:
    return BotCore(
        ks=ks,
        journal=journal,
        allowed_chat_id=allowed_chat_id,
        clock=clock or _Clock(dt.datetime(2026, 4, 28, 12, 0, 0, tzinfo=dt.UTC)),
    )


def _seed_status_data(journal: JournalRepo) -> None:
    """Populate journal with rows the /status query reads."""
    db = journal.db_path
    insert_account_snapshot(
        db,
        AccountSnapshotRow(
            snapshot_at=dt.datetime(2026, 4, 28, 11, 55, 0),
            equity=12_345.67,
            cash=5_000.00,
            buying_power=15_000.00,
            long_market_value=7_345.67,
            daypl=123.45,
        ),
    )
    insert_position(
        journal.db_path,
        PositionRow(
            snapshot_at=dt.datetime(2026, 4, 28, 11, 55, 0),
            source="broker",
            symbol="ACME",
            qty=100,
            avg_entry_price=12.34,
            market_value=1234.0,
            unrealized_pnl=10.0,
        ),
    )
    insert_position(
        journal.db_path,
        PositionRow(
            snapshot_at=dt.datetime(2026, 4, 28, 11, 55, 0),
            source="broker",
            symbol="WIDG",
            qty=50,
            avg_entry_price=20.0,
            market_value=1000.0,
            unrealized_pnl=-5.0,
        ),
    )
    fid = insert_filing(
        db,
        FilingRow(
            accession_number="0001234567-26-000001",
            cik=1234567,
            form_type="8-K",
            filed_at=dt.datetime(2026, 4, 28, 11, 30, 0),
            fetched_at=dt.datetime(2026, 4, 28, 11, 45, 0),
            raw_text_path="/var/lib/quinn/raw/x.txt",
            content_hash="h1",
        ),
    )
    insert_prompt(
        db,
        PromptRow(
            prompt_version="sonnet@x",
            name="sonnet",
            file_path="src/prompts/sonnet.txt",
            content_hash="x",
        ),
    )
    insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id="dec-1",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet@x",
            raw_response="{}",
            kind="trade_proposal",
            input_tokens=100,
            output_tokens=20,
            latency_ms=300,
            cost_usd=0.01,
        ),
    )
    insert_llm_call(
        db,
        LlmCallRow(
            decision_id="dec-1",
            purpose="analyzer",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet@x",
            input_tokens=100,
            output_tokens=20,
            latency_ms=300,
            cost_usd=1.50,
        ),
    )
    insert_llm_call(
        db,
        LlmCallRow(
            decision_id="dec-1",
            purpose="reviewer",
            model_id="claude-opus-4-7",
            prompt_version="sonnet@x",
            input_tokens=100,
            output_tokens=20,
            latency_ms=300,
            cost_usd=2.50,
        ),
    )


# ---------------------------------------------------------------------------
# AC-2: chat ID allow-list
# ---------------------------------------------------------------------------


def test_unauthorized_chat_id_ignored(
    ks: KillSwitch, journal: JournalRepo, caplog: pytest.LogCaptureFixture
) -> None:
    bot = _bot(ks, journal, allowed_chat_id=42)
    reply = bot.handle_command(chat_id=99, text="/halt")
    assert reply is None  # silent ignore (no message sent back)
    assert ks.is_halted() is False  # state unchanged
    # WARN-level audit log per AC-2
    matched = [
        r
        for r in caplog.records
        if "telegram.unauthorized" in r.getMessage()
        or getattr(r, "event", None) == "telegram.unauthorized"
    ]
    assert matched, "expected telegram.unauthorized log entry"


# ---------------------------------------------------------------------------
# AC-2 / AC-4: /halt
# ---------------------------------------------------------------------------


def test_halt_command_flips_state(ks: KillSwitch, journal: JournalRepo) -> None:
    bot = _bot(ks, journal)
    reply = bot.handle_command(chat_id=42, text="/halt")
    assert isinstance(reply, BotReply)
    assert reply.chat_id == 42
    assert "halted" in reply.text.lower()
    state = ks.state()
    assert state.state == "halted"
    assert state.reason == "manual:telegram"
    assert state.set_by == "operator"


# ---------------------------------------------------------------------------
# AC-2: /resume soft-confirm + 60s window
# ---------------------------------------------------------------------------


def test_resume_requires_yes_confirm(ks: KillSwitch, journal: JournalRepo) -> None:
    ks.halt(reason="manual:telegram", set_by="operator")
    bot = _bot(ks, journal)
    first = bot.handle_command(chat_id=42, text="/resume")
    assert first is not None
    assert "yes" in first.text.lower()
    # state still halted until 'yes' arrives
    assert ks.is_halted() is True
    confirm = bot.handle_command(chat_id=42, text="yes")
    assert confirm is not None
    assert "resumed" in confirm.text.lower() or "active" in confirm.text.lower()
    assert ks.is_halted() is False


def test_resume_confirm_window_expires_at_60s(ks: KillSwitch, journal: JournalRepo) -> None:
    ks.halt(reason="manual:telegram", set_by="operator")
    clock = _Clock(dt.datetime(2026, 4, 28, 12, 0, 0, tzinfo=dt.UTC))
    bot = _bot(ks, journal, clock=clock)
    bot.handle_command(chat_id=42, text="/resume")
    clock.advance(61)
    reply = bot.handle_command(chat_id=42, text="yes")
    # confirm window expired → not a resume; state still halted
    assert ks.is_halted() is True
    # bot should respond explaining expiry
    assert reply is not None
    assert "expired" in reply.text.lower() or "no pending" in reply.text.lower()


def test_resume_unprompted_yes_is_ignored(ks: KillSwitch, journal: JournalRepo) -> None:
    """A `yes` with no prior /resume must not flip state."""
    ks.halt(reason="manual:telegram", set_by="operator")
    bot = _bot(ks, journal)
    reply = bot.handle_command(chat_id=42, text="yes")
    assert ks.is_halted() is True
    # any reply is fine but state must not change
    if reply is not None:
        assert "resumed" not in reply.text.lower()


# ---------------------------------------------------------------------------
# AC-2: /status (FR-34)
# ---------------------------------------------------------------------------


def test_status_command_returns_expected_fields(ks: KillSwitch, journal: JournalRepo) -> None:
    _seed_status_data(journal)
    bot = _bot(ks, journal)
    reply = bot.handle_command(chat_id=42, text="/status")
    assert reply is not None
    text = reply.text
    # FR-34 fields per AC-2
    assert "kill-switch" in text.lower() or "kill switch" in text.lower()
    assert "active" in text.lower()  # ks state
    assert "123.45" in text  # daypl
    assert "ACME" in text and "WIDG" in text  # open positions
    assert "2" in text  # position count
    # MTD inference cost = 1.50 + 2.50 = 4.00
    assert "4.00" in text or "$4.00" in text


def test_status_works_with_agent_down(ks: KillSwitch, journal: JournalRepo) -> None:
    """ADR-004: bot answers /status from journal alone, no agent process needed."""
    _seed_status_data(journal)
    # bot has no reference to any agent loop; only journal + ks
    bot = _bot(ks, journal)
    reply = bot.handle_command(chat_id=42, text="/status")
    assert reply is not None
    assert "active" in reply.text.lower()


def test_status_handles_empty_journal_gracefully(ks: KillSwitch, journal: JournalRepo) -> None:
    """Fresh journal (no snapshots/positions/filings) → /status still replies."""
    bot = _bot(ks, journal)
    reply = bot.handle_command(chat_id=42, text="/status")
    assert reply is not None
    # should mention "no data" somehow per missing fields, but must not crash
    assert "kill-switch" in reply.text.lower() or "kill switch" in reply.text.lower()


# ---------------------------------------------------------------------------
# AC-3: latency
# ---------------------------------------------------------------------------


def test_halt_latency_under_60s_with_5s_poll(ks: KillSwitch, journal: JournalRepo) -> None:
    """AC-3: handler is sub-second; poll cadence ≤ 5s satisfies 60s budget.

    We assert handler completes under 1s; the 5s poll cadence is enforced by
    the entry-point process (bot.py) and is asserted by integration in CI.
    """
    import time

    bot = _bot(ks, journal)
    t0 = time.perf_counter()
    reply = bot.handle_command(chat_id=42, text="/halt")
    elapsed = time.perf_counter() - t0
    assert reply is not None
    assert elapsed < 1.0, f"halt handler took {elapsed:.3f}s (>1s)"
    assert ks.is_halted() is True
    # The PTB long-poll cadence is configured in src/bot.py; assert the constant
    from killswitch.telegram_bot import LONG_POLL_INTERVAL_S
    assert LONG_POLL_INTERVAL_S <= 5
