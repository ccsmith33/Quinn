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
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from config.loader import KillSwitchConfig
from journal.repo import JournalRepo
from observability.log_port import get_logger

from .api import KillSwitch, KillSwitchUninitialized

log = get_logger(__name__)

KS_TRAILING_WINDOW_DAYS = 30


@dataclass(frozen=True)
class Halt:
    rule: str  # "KS-1" | "KS-2" | "KS-3"
    reason: str  # "auto:KS-1" etc — written into kill_switch_state.reason
    detail: str
    fired: bool  # True if this evaluation appended a new halt row


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
        out: list[Halt] = []
        for r in rules:
            if already_auto_halted:
                out.append(
                    Halt(rule=r.rule, reason=r.reason, detail=r.detail, fired=False)
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
            return [Halt(rule="KS-2", reason="auto:KS-2", detail=detail, fired=False)]
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
    def _is_auto_halted(ks: KillSwitch) -> bool:
        try:
            state = ks.state()
        except KillSwitchUninitialized:
            # fail-closed; treat as halted so we do not insert redundant rows
            return True
        return state.state == "halted" and state.reason.startswith("auto:")
