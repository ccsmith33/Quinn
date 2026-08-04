"""Automatic halt evaluator for KS-1, KS-2, KS-3 (S7.4, PRD §5).

The evaluator is a stateless object that, given a snapshot of "now" plus the
journal + kill-switch + config, returns the set of halt rules that fired and
applies any new halts via `KillSwitch.halt(...)`.

Idempotency (AC-6): if the kill-switch is already halted with an `auto:KS-*`
reason we do not re-halt — the evaluator returns the rules that *would* fire
but `Halt.fired=False` so the caller can still surface them in `/status` /
logs without inserting redundant rows. Manual halts (`manual:telegram`,
`manual:webhook`) are never overwritten by the evaluator either.

Schedule (AC-5): callers run this from the agent loop's periodic task (every
60s in market hours) AND explicitly after the position reconciler confirms
a closed trade. Both call sites use the same `evaluate(...)` entry point.

Operator-resume re-fire suppression (2026-07-14 clay incident): KS-1
halted 17:47:41, the operator resumed 17:48:10, and the evaluator
re-fired KS-1 at 17:48:42 because the daily-loss condition was — of
course — still true; the box then sat halted for three days while the
operator believed they had resumed. An OPERATOR resume of an auto rule
now suppresses re-firing of the SAME rule (KS-1 / KS-2 — condition-style
rules whose predicate stays true all day) until the next ET calendar
day: the human has seen the condition and accepted it. A NEW day's fresh
breach fires normally, a DIFFERENT rule still fires the same day, and
KS-3 (event-style: consecutive losses) is unaffected. Suppressions are
logged as `killswitch.refire_suppressed` and still returned with
`fired=False` so /status can surface the standing condition.

KS-2 acknowledged-drawdown watermark (2026-08, live money): the 30-day
peak decays slowly, so after a drawdown the KS-2 predicate stays true
for DAYS — same-day suppression alone still forces a daily /resume
ritual (and a forgotten resume kills the day's entries). When
`killswitch.ks2_reack_margin_pct` > 0, an operator resume of an
auto:KS-2 halt carries a durable watermark (drawdown pct + 30d peak at
ack time, written by the Telegram bot into the resume row's notes JSON
— see telegram_bot._resume_notes). KS-2 is then suppressed (logged as
`killswitch.refire_suppressed` with mode=acked_watermark) until the
drawdown deepens to acked_dd + margin (>= fires — new pain beyond what
was acknowledged). A genuine new equity high above the acked peak
expires the watermark permanently; a decaying rolling peak does not.
Default margin 0.0 = feature off, behavior identical to the paragraph
above. KS-1/KS-3 are untouched by the watermark.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass

from config.calendar import ET
from config.loader import KillSwitchConfig
from journal.repo import JournalRepo
from observability.log_port import get_logger

from .api import KillSwitch, KillSwitchUninitialized

log = get_logger(__name__)

KS_TRAILING_WINDOW_DAYS = 30

# Auto rules whose predicate is a standing CONDITION (true all day once
# breached) — these are the re-fire-suppression candidates. KS-3 is
# event-driven (a NEW consecutive loss re-arms it legitimately) and is
# deliberately absent.
_RESUME_SUPPRESSIBLE_REASONS = frozenset({"auto:KS-1", "auto:KS-2"})


@dataclass(frozen=True)
class Halt:
    rule: str  # "KS-1" | "KS-2" | "KS-3"
    reason: str  # "auto:KS-1" etc — written into kill_switch_state.reason
    detail: str
    fired: bool  # True if this evaluation appended a new halt row
    # "acked_watermark" when the KS-2 acked-drawdown watermark suppressed
    # this halt (feature `killswitch.ks2_reack_margin_pct`); None otherwise.
    suppressed_by: str | None = None


@dataclass(frozen=True)
class _Ks2Ack:
    """Parsed KS-2 acknowledgment watermark (see telegram_bot._resume_notes)."""

    set_at: _dt.datetime
    dd_pct: float
    peak: float


class AutoHaltEvaluator:
    def evaluate(
        self,
        now: _dt.datetime,
        journal: JournalRepo,
        ks: KillSwitch,
        cfg: KillSwitchConfig,
    ) -> list[Halt]:
        rules: list[Halt] = []
        rules.extend(self._eval_ks1(now, journal, cfg))
        rules.extend(self._eval_ks2(now, journal, cfg))
        rules.extend(self._eval_ks3(journal, cfg))

        already_auto_halted = self._is_auto_halted(ks)
        suppressed_reasons = self._operator_resumed_today(now, journal)
        out: list[Halt] = []
        for r in rules:
            if already_auto_halted:
                out.append(
                    Halt(
                        rule=r.rule,
                        reason=r.reason,
                        detail=r.detail,
                        fired=False,
                        suppressed_by=r.suppressed_by,
                    )
                )
                continue
            if r.suppressed_by is not None:
                # KS-2 acked-drawdown watermark (checked + logged in
                # _eval_ks2): the operator has acknowledged this depth of
                # pain and the drawdown has not deepened past the re-ack
                # margin — surface the condition but do not halt.
                out.append(
                    Halt(
                        rule=r.rule,
                        reason=r.reason,
                        detail=r.detail,
                        fired=False,
                        suppressed_by=r.suppressed_by,
                    )
                )
                continue
            if r.reason in suppressed_reasons:
                # The operator resumed THIS rule earlier today with its
                # condition still standing — re-firing would silently
                # defeat the resume (2026-07-14: 32 seconds later, then
                # three days halted). Suppress until the ET day rolls.
                out.append(
                    Halt(rule=r.rule, reason=r.reason, detail=r.detail, fired=False)
                )
                log.info(
                    "killswitch.refire_suppressed",
                    extra={
                        "event": "killswitch.refire_suppressed",
                        "rule": r.rule,
                        "reason": r.reason,
                        "detail": r.detail,
                    },
                )
                continue
            ks.halt(reason=r.reason, set_by="system", notes=r.detail)
            already_auto_halted = True
            out.append(Halt(rule=r.rule, reason=r.reason, detail=r.detail, fired=True))
            log.warning(
                "auto-halt fired",
                extra={"event": "killswitch.auto_halt", "rule": r.rule, "detail": r.detail},
            )
        return out

    # -- KS-1 -------------------------------------------------------------

    def _eval_ks1(
        self, now: _dt.datetime, journal: JournalRepo, cfg: KillSwitchConfig
    ) -> list[Halt]:
        snap = journal.get_latest_account_snapshot()
        if snap is None:
            return []
        sod_equity = snap.equity - snap.daypl
        if sod_equity <= 0:
            return []
        loss_pct = -snap.daypl / sod_equity if snap.daypl < 0 else 0.0
        if loss_pct >= cfg.ks1_daily_loss_pct:
            detail = (
                f"daypl={snap.daypl:.2f} sod_equity={sod_equity:.2f} "
                f"loss_pct={loss_pct:.4f} threshold={cfg.ks1_daily_loss_pct}"
            )
            return [Halt(rule="KS-1", reason="auto:KS-1", detail=detail, fired=False)]
        return []

    # -- KS-2 -------------------------------------------------------------

    def _eval_ks2(
        self, now: _dt.datetime, journal: JournalRepo, cfg: KillSwitchConfig
    ) -> list[Halt]:
        since = now - _dt.timedelta(days=KS_TRAILING_WINDOW_DAYS)
        peak = journal.get_peak_equity_since(since)
        snap = journal.get_latest_account_snapshot()
        if peak is None or snap is None or peak <= 0:
            return []
        threshold_equity = peak * (1.0 - cfg.ks2_trailing_dd_pct)
        if snap.equity <= threshold_equity:
            dd_pct = (peak - snap.equity) / peak
            detail = (
                f"peak30d={peak:.2f} current={snap.equity:.2f} "
                f"dd_pct={dd_pct:.4f} threshold={cfg.ks2_trailing_dd_pct}"
            )
            suppressed_by = None
            ack = (
                self._active_ks2_ack(journal)
                if cfg.ks2_reack_margin_pct > 0.0
                else None
            )
            if ack is not None:
                # The margin is in PERCENTAGE POINTS (see KillSwitchConfig).
                # Fire at exactly acked_dd + margin (>= fires): only a
                # drawdown strictly inside the margin is acknowledged pain.
                refire_dd = ack.dd_pct + cfg.ks2_reack_margin_pct / 100.0
                if dd_pct < refire_dd:
                    suppressed_by = "acked_watermark"
                    log.info(
                        "killswitch.refire_suppressed",
                        extra={
                            "event": "killswitch.refire_suppressed",
                            "rule": "KS-2",
                            "reason": "auto:KS-2",
                            "mode": "acked_watermark",
                            "acked_dd": ack.dd_pct,
                            "current_dd": dd_pct,
                            "refire_dd": refire_dd,
                            "acked_peak": ack.peak,
                        },
                    )
            return [
                Halt(
                    rule="KS-2",
                    reason="auto:KS-2",
                    detail=detail,
                    fired=False,
                    suppressed_by=suppressed_by,
                )
            ]
        return []

    # -- KS-3 -------------------------------------------------------------

    def _eval_ks3(self, journal: JournalRepo, cfg: KillSwitchConfig) -> list[Halt]:
        n = cfg.ks3_consecutive_losses
        if n <= 0:
            return []
        closes = journal.get_closed_trade_pnls_chronological()
        if len(closes) < n:
            return []
        recent = closes[-n:]
        if all(pnl < 0 for _, pnl, _ in recent):
            detail = (
                f"last_{n}_pnls=" + ", ".join(f"{pnl:.2f}" for _, pnl, _ in recent)
            )
            return [Halt(rule="KS-3", reason="auto:KS-3", detail=detail, fired=False)]
        return []

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _active_ks2_ack(journal: JournalRepo) -> _Ks2Ack | None:
        """The newest KS-2 acknowledgment watermark, or None when absent,
        malformed, or EXPIRED.

        Expiry: the ack dies once equity has made a genuine new high above
        the peak stored at ack time (recovery resets the regime — a later
        fresh drawdown from the new peak must halt normally). Implemented
        as max(equity since ack) > acked_peak, which is sticky: the new
        high expires the ack permanently even after it rolls out of the
        30-day window. A decaying rolling peak never expires the ack.
        """
        get_row = getattr(journal, "get_latest_ks2_ack_row", None)
        if get_row is None:  # legacy fake — feature inert
            return None
        row = get_row()
        if row is None or not row.notes:
            return None
        try:
            payload = json.loads(row.notes)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        ack = payload.get("ks2_ack")
        if not isinstance(ack, dict):
            return None
        dd_pct = ack.get("dd_pct")
        peak = ack.get("peak")
        if not isinstance(dd_pct, int | float) or not isinstance(peak, int | float):
            return None
        set_at = row.set_at
        if set_at.tzinfo is None:
            set_at = set_at.replace(tzinfo=_dt.UTC)
        peak_since_ack = journal.get_peak_equity_since(set_at)
        if peak_since_ack is not None and peak_since_ack > float(peak):
            return None  # expired: genuine new high above the acked peak
        return _Ks2Ack(set_at=set_at, dd_pct=float(dd_pct), peak=float(peak))

    @staticmethod
    def _operator_resumed_today(
        now: _dt.datetime, journal: JournalRepo
    ) -> frozenset[str]:
        """Reasons in `_RESUME_SUPPRESSIBLE_REASONS` whose halts the
        newest OPERATOR resume resolved, when that resume happened on
        the same ET calendar day as `now` — i.e. the same-day re-fire
        suppression set. Empty otherwise (no operator resume, a resume
        from a prior day, or a system resume like session_start)."""
        get_rows = getattr(journal, "get_recent_kill_switch_rows", None)
        if get_rows is None:  # legacy fake — feature inert
            return frozenset()
        rows = get_rows()
        resume_idx = next(
            (
                i
                for i, r in enumerate(rows)
                if r.state == "active"
                and r.set_by == "operator"
                and r.reason == "resume"
            ),
            None,
        )
        if resume_idx is None:
            return frozenset()
        resume = rows[resume_idx]
        if _as_et_date(resume.set_at) != _as_et_date(now):
            return frozenset()
        # The halts that resume resolved: the contiguous 'halted' run
        # directly below it (newest-first order → walk forward until
        # the previous 'active' row).
        suppressed: set[str] = set()
        for r in rows[resume_idx + 1 :]:
            if r.state == "active":
                break
            if r.reason in _RESUME_SUPPRESSIBLE_REASONS:
                suppressed.add(r.reason)
        return frozenset(suppressed)

    @staticmethod
    def _is_auto_halted(ks: KillSwitch) -> bool:
        try:
            state = ks.state()
        except KillSwitchUninitialized:
            # fail-closed; treat as halted so we do not insert redundant rows
            return True
        return state.state == "halted" and state.reason.startswith("auto:")


def _as_et_date(ts: _dt.datetime) -> _dt.date:
    """ET calendar date of a journal timestamp; naive rows (operator
    SQL, legacy fixtures) are treated as UTC — same convention as
    `KillSwitch._as_utc`."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.UTC)
    return ts.astimezone(ET).date()
