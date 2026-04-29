"""Kill-switch state read/write API (S7.1, ADR-004).

Single source of truth for "are we halted?" — Telegram bot (S7.2), webhook
listener (S7.3), and auto-evaluator (S7.4) all write through `KillSwitch`;
the execution layer (S6.2) reads through it on every entry-validation call
(FR-20). Backed by the append-only `kill_switch_state` table (NFR-16); no row
is ever updated or deleted.
"""

from __future__ import annotations

import datetime as _dt
from typing import Literal

from journal.models import KillSwitchStateRow
from journal.repo import JournalRepo

SetBy = Literal["operator", "system"]


class KillSwitchUninitialized(Exception):
    """Raised when `kill_switch_state` is empty.

    The seed row inserted by 001_init.sql guarantees this never fires in normal
    operation; callers must fail closed (refuse to execute) if it does.
    """


class KillSwitch:
    def __init__(self, journal: JournalRepo) -> None:
        self._journal = journal

    def state(self) -> KillSwitchStateRow:
        row = self._journal.get_latest_kill_switch_state()
        if row is None:
            raise KillSwitchUninitialized(
                "kill_switch_state has no rows; seed row missing"
            )
        return row

    def is_halted(self) -> bool:
        return self.state().state == "halted"

    def halt(self, reason: str, set_by: SetBy, notes: str = "") -> None:
        self._journal.insert_kill_switch_state(
            KillSwitchStateRow(
                set_at=_dt.datetime.now(_dt.UTC),
                state="halted",
                reason=reason,
                set_by=set_by,
                notes=notes or None,
            )
        )

    def resume(self, set_by: SetBy, notes: str = "") -> None:
        self._journal.insert_kill_switch_state(
            KillSwitchStateRow(
                set_at=_dt.datetime.now(_dt.UTC),
                state="active",
                reason="resume",
                set_by=set_by,
                notes=notes or None,
            )
        )
