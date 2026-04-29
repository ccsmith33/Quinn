"""Telegram bot core (S7.2, ADR-004 §Telegram bot).

Pure-logic command dispatcher. Does not import `python-telegram-bot`; the
process entry point at `src/bot.py` wires PTB to `BotCore.handle_command`.
This separation keeps the dispatcher testable without spinning up Telegram
and lets the agent loop crash without affecting this process (ADR-004).

Recognized commands (operator chat ID only, allow-list of one):
- `/halt`     → KillSwitch.halt("manual:telegram", set_by="operator")
- `/resume`   → soft-confirm flow; user must reply `yes` within 60s
- `/status`   → FR-34 snapshot from journal (kill-switch state, P&L,
                positions, last filing/proposal time, MTD inference cost)
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from dataclasses import dataclass

from journal.repo import JournalRepo
from observability.log_port import get_logger

from .api import KillSwitch, KillSwitchUninitialized

log = get_logger(__name__)

# AC-3: long-poll cadence ≤ 5s leaves ample headroom inside FR-32's 60s budget.
LONG_POLL_INTERVAL_S: int = 3

# AC-2: /resume requires `yes` within this window.
RESUME_CONFIRM_WINDOW_S: int = 60


@dataclass(frozen=True)
class BotReply:
    chat_id: int
    text: str


class BotCore:
    def __init__(
        self,
        ks: KillSwitch,
        journal: JournalRepo,
        allowed_chat_id: int,
        clock: Callable[[], _dt.datetime] | None = None,
    ) -> None:
        self._ks = ks
        self._journal = journal
        self._allowed_chat_id = int(allowed_chat_id)
        self._clock: Callable[[], _dt.datetime] = clock or (
            lambda: _dt.datetime.now(_dt.UTC)
        )
        # In-memory pending-resume state. The bot is a single OS process per
        # ADR-004, so process-local memory is fine; if the bot crashes mid-
        # confirm the operator simply re-issues /resume.
        self._pending_resume_at: _dt.datetime | None = None

    def handle_command(self, chat_id: int, text: str) -> BotReply | None:
        """Dispatch a single inbound message. Returns the reply or None.

        Returning None means "no reply" — used for unauthorized senders so we
        do not leak the bot's existence to non-operator chat IDs.
        """
        if chat_id != self._allowed_chat_id:
            log.warning(
                "ignoring command from unauthorized chat",
                extra={"event": "telegram.unauthorized", "chat_id": chat_id},
            )
            return None

        cmd = (text or "").strip()
        if cmd == "/halt":
            return self._cmd_halt(chat_id)
        if cmd == "/resume":
            return self._cmd_resume_request(chat_id)
        if cmd.lower() == "yes":
            return self._cmd_resume_confirm(chat_id)
        if cmd == "/status":
            return self._cmd_status(chat_id)
        return BotReply(
            chat_id=chat_id,
            text="Unknown command. Use /halt, /resume, or /status.",
        )

    # -- /halt ------------------------------------------------------------

    def _cmd_halt(self, chat_id: int) -> BotReply:
        self._ks.halt(reason="manual:telegram", set_by="operator", notes="via /halt")
        now = self._clock().isoformat(timespec="seconds")
        return BotReply(
            chat_id=chat_id,
            text=f"Kill-switch HALTED at {now}. New entries will be rejected.",
        )

    # -- /resume (two-step) ----------------------------------------------

    def _cmd_resume_request(self, chat_id: int) -> BotReply:
        self._pending_resume_at = self._clock()
        return BotReply(
            chat_id=chat_id,
            text=(
                "Reply 'yes' within 60s to confirm resume. "
                "Kill-switch remains HALTED until confirmed."
            ),
        )

    def _cmd_resume_confirm(self, chat_id: int) -> BotReply:
        if self._pending_resume_at is None:
            return BotReply(
                chat_id=chat_id,
                text="No pending /resume to confirm.",
            )
        elapsed = (self._clock() - self._pending_resume_at).total_seconds()
        if elapsed > RESUME_CONFIRM_WINDOW_S:
            self._pending_resume_at = None
            return BotReply(
                chat_id=chat_id,
                text="Resume confirm window expired. Send /resume again.",
            )
        self._ks.resume(set_by="operator", notes="via /resume yes-confirm")
        self._pending_resume_at = None
        now = self._clock().isoformat(timespec="seconds")
        return BotReply(
            chat_id=chat_id,
            text=f"Kill-switch resumed at {now}. State is now ACTIVE.",
        )

    # -- /status ---------------------------------------------------------

    def _cmd_status(self, chat_id: int) -> BotReply:
        try:
            state = self._ks.state().state
        except KillSwitchUninitialized:
            state = "UNINITIALIZED (fail-closed: treat as HALTED)"

        snap = self._journal.get_latest_account_snapshot()
        positions = self._journal.get_open_positions()
        last_filing = self._journal.get_last_filing_fetched_at()
        last_proposal = self._journal.get_last_proposal_created_at()
        now = self._clock()
        mtd_cost = self._journal.get_mtd_inference_cost_usd(now)

        daypl_line = (
            f"Day P&L: ${snap.daypl:+.2f} (equity ${snap.equity:,.2f})"
            if snap is not None
            else "Day P&L: no account snapshot yet"
        )
        if positions:
            symbols = ", ".join(p.symbol for p in positions)
            pos_line = f"Open positions ({len(positions)}): {symbols}"
        else:
            pos_line = "Open positions (0): none"

        last_filing_line = (
            f"Last EDGAR poll: {_age(last_filing, now)} ago"
            if last_filing is not None
            else "Last EDGAR poll: no filings ingested yet"
        )
        last_proposal_line = (
            f"Last proposal: {_age(last_proposal, now)} ago"
            if last_proposal is not None
            else "Last proposal: none yet"
        )
        mtd_line = f"MTD inference cost: ${mtd_cost:.2f}"

        return BotReply(
            chat_id=chat_id,
            text="\n".join(
                [
                    f"Kill-switch: {state.upper()}",
                    daypl_line,
                    pos_line,
                    last_filing_line,
                    last_proposal_line,
                    mtd_line,
                ]
            ),
        )


def _age(then: _dt.datetime, now: _dt.datetime) -> str:
    if then.tzinfo is None and now.tzinfo is not None:
        then = then.replace(tzinfo=now.tzinfo)
    elif now.tzinfo is None and then.tzinfo is not None:
        now = now.replace(tzinfo=then.tzinfo)
    seconds = max(0, int((now - then).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600}h"
