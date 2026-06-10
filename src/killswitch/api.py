"""Kill-switch state read/write API (S7.1, ADR-004).

Single source of truth for "are we halted?" — Telegram bot (S7.2), webhook
listener (S7.3), and auto-evaluator (S7.4) all write through `KillSwitch`;
the execution layer (S6.2) reads through it on every entry-validation call
(FR-20). Backed by the append-only `kill_switch_state` table (NFR-16); no row
is ever updated or deleted.

WS1 (D-078, delta §2.3 — kills O-5): `halt()` accepts an optional
`fingerprint`. While an UNRESOLVED halt row with the same reason+fingerprint
exists, repeat calls append no row and return False so the caller skips the
page; a re-page row is appended at most once per `repage_minutes`. A resumed
halt whose fingerprint recurs is a fresh halt — resumes are not free passes.
The table has no fingerprint column (migration 005 is indexes-only), so
fingerprinted rows store a notes JSON envelope
`{"fingerprint": ..., "detail": <caller notes>}`; dedupe state therefore
lives in the journal and survives process restarts.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Callable
from typing import Literal

from journal.models import KillSwitchStateRow
from journal.repo import JournalRepo

SetBy = Literal["operator", "system"]

# Delta §2.3: re-page at most once per this many minutes while the same
# fingerprint persists. Config `killswitch.halt_repage_minutes` overrides
# at composition time.
DEFAULT_REPAGE_MINUTES = 240


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _fingerprint_of(row: KillSwitchStateRow) -> str | None:
    """Recover the fingerprint from a row's notes envelope; None for
    legacy / unfingerprinted rows (free-text or diff-array notes)."""
    if not row.notes:
        return None
    try:
        payload = json.loads(row.notes)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        fp = payload.get("fingerprint")
        return fp if isinstance(fp, str) else None
    return None


class KillSwitchUninitialized(Exception):
    """Raised when `kill_switch_state` is empty.

    The seed row inserted by 001_init.sql guarantees this never fires in normal
    operation; callers must fail closed (refuse to execute) if it does.
    """


class KillSwitch:
    def __init__(
        self,
        journal: JournalRepo,
        *,
        repage_minutes: int = DEFAULT_REPAGE_MINUTES,
        now_fn: Callable[[], _dt.datetime] = _utcnow,
    ) -> None:
        self._journal = journal
        self._repage_minutes = repage_minutes
        self._now_fn = now_fn

    def state(self) -> KillSwitchStateRow:
        row = self._journal.get_latest_kill_switch_state()
        if row is None:
            raise KillSwitchUninitialized(
                "kill_switch_state has no rows; seed row missing"
            )
        return row

    def is_halted(self) -> bool:
        return self.state().state == "halted"

    def halt(
        self,
        reason: str,
        set_by: SetBy,
        notes: str = "",
        fingerprint: str | None = None,
    ) -> bool:
        """Append a halted row. Returns True when a row was appended (the
        caller should page); False when an unresolved halt with the same
        reason+fingerprint already exists inside the re-page window.

        `fingerprint=None` preserves the pre-WS1 contract exactly: every
        call appends and fires (manual halts, auto:KS-* — those have
        their own idempotency in auto_halts.py).
        """
        now = self._now_fn()
        stored_notes = notes or None
        if fingerprint is not None:
            unresolved = self._journal.get_kill_switch_halts_since_last_resume()
            matching = [
                r
                for r in unresolved
                if r.reason == reason and _fingerprint_of(r) == fingerprint
            ]
            if matching:
                newest = max(self._as_utc(r.set_at) for r in matching)
                if now - newest < _dt.timedelta(minutes=self._repage_minutes):
                    return False
            stored_notes = json.dumps(
                {"fingerprint": fingerprint, "detail": notes}
            )
        self._journal.insert_kill_switch_state(
            KillSwitchStateRow(
                set_at=now,
                state="halted",
                reason=reason,
                set_by=set_by,
                notes=stored_notes,
            )
        )
        return True

    def resume(self, set_by: SetBy, notes: str = "") -> None:
        self._journal.insert_kill_switch_state(
            KillSwitchStateRow(
                set_at=self._now_fn(),
                state="active",
                reason="resume",
                set_by=set_by,
                notes=notes or None,
            )
        )

    @staticmethod
    def _as_utc(ts: _dt.datetime) -> _dt.datetime:
        """Rows written by this class are tz-aware; tolerate naive rows
        (operator SQL, legacy fixtures) by assuming UTC."""
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=_dt.UTC)
