"""S6.5 — Position reconciler (FR-24, ADR-001).

Compares broker truth to the journal's latest position snapshot per
symbol; on divergence, soft-halts the kill-switch with reason
`reconciler:discrepancy` and notifies the operator. Runs:

  - periodically every `cfg.interval_seconds_market` seconds during
    market hours (off-hours suppressed — no fills happen overnight);
  - at startup (the agent loop's first tick is on boot);
  - after every successful order submission (`trigger_after_submission`).

Single code path for paper / live (D-007 sacred): the broker adapter is
the only seam.

Transient broker outages are suppressed: `BrokerUnavailable` from the
broker counts as a deferred reconcile (no halt, no diff). After 3
consecutive transient failures we log WARN — but we still do not halt
on broker outage alone, because the kill-switch is the *application*'s
view of safety, not the broker's connectivity status.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from broker.alpaca import BrokerUnavailable
from broker.protocol import AccountSnapshot, BrokerAdapter, Position
from config.calendar import is_market_hours
from config.loader import ReconcilerConfig
from journal.models import AccountSnapshotRow, PositionRow
from observability.log_port import get_logger

# Forward-only protocol for the Feature C retro coordinator. We don't
# import the concrete class here to avoid a circular dependency with
# `app.retro_fill` (which already imports from `journal.repo`).

log = get_logger(__name__)

# Threshold above which transient broker failures are surfaced as a WARN
# log line (still no halt — broker outage ≠ position discrepancy).
_PERSISTENT_FAILURE_THRESHOLD = 3


@dataclass(frozen=True)
class PositionDiff:
    symbol: str
    broker_qty: int
    journal_qty: int
    broker_avg_entry: float | None
    journal_avg_entry: float | None


@dataclass(frozen=True)
class ReconcileReport:
    matched: bool
    diffs: list[PositionDiff]
    deferred: bool = False
    suppressed: bool = False


class _JournalLike(Protocol):
    def get_open_positions(self) -> list: ...
    def insert_position(self, row: PositionRow) -> int: ...
    def insert_account_snapshot(self, row: AccountSnapshotRow) -> int: ...


class _KillSwitchLike(Protocol):
    def halt(self, reason: str, set_by: str, notes: str = "") -> None: ...


class _AlerterLike(Protocol):
    def notify(self, message: str) -> None: ...


class _RetroCoordinatorLike(Protocol):
    async def run_tick(self) -> None: ...


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Reconciler:
    """Long-running async component that reconciles broker truth with the
    journal's position view. Owns the periodic asyncio task; the surface
    `reconcile_now()` is synchronous so tests and the post-submission
    trigger can call it directly without an event loop.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        journal: _JournalLike,
        ks: _KillSwitchLike,
        cfg: ReconcilerConfig,
        *,
        alerter: _AlerterLike | None = None,
        now_fn: Callable[[], dt.datetime] = _utcnow,
        retro_coordinator: _RetroCoordinatorLike | None = None,
        thesis_coordinator: _RetroCoordinatorLike | None = None,
    ) -> None:
        self._broker = broker
        self._journal = journal
        self._ks = ks
        self._cfg = cfg
        self._alerter = alerter
        self._now_fn = now_fn
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._consecutive_failures = 0
        # Feature C — invoked at the end of every successful reconcile
        # tick. None disables retro fill (legacy test path).
        self._retro = retro_coordinator
        # Feature A — open-position thesis-review coordinator. Same
        # lifecycle gating as `_retro`: skipped on suppressed/deferred
        # ticks; errors logged and not propagated.
        self._thesis = thesis_coordinator

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the periodic reconcile task. Idempotent: a second call
        before stop() is a no-op."""
        if self._task is not None and not self._task.done():
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run_loop(), name="reconciler-loop")

    async def stop(self) -> None:
        """Cancel the periodic task and await its termination."""
        if self._task is None:
            return
        self._stopping.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run_loop(self) -> None:
        # Run reconcile immediately on start (boot-time AC) and then on the
        # configured cadence.
        while not self._stopping.is_set():
            try:
                report = self.reconcile_now()
            except Exception as e:  # noqa: BLE001 — defensive; failures must not kill loop
                log.error(
                    "reconciler.loop_error",
                    extra={
                        "event": "reconciler.loop_error",
                        "error": str(e),
                    },
                )
                report = None

            # Feature C — opportunistic retro-fill on slot-open. Runs only
            # after a successful, non-suppressed reconcile so the journal's
            # view of open positions is fresh. Off-hours suppression is
            # inherited via `report.suppressed`. Errors are caught here so
            # a retro hiccup never stops the reconciler tick cadence.
            if (
                self._retro is not None
                and report is not None
                and not report.suppressed
                and not report.deferred
            ):
                try:
                    await self._retro.run_tick()
                except Exception as e:  # noqa: BLE001
                    log.error(
                        "reconciler.retro_tick_error",
                        extra={
                            "event": "reconciler.retro_tick_error",
                            "error": str(e),
                            "error_class": type(e).__name__,
                        },
                    )

            # Feature A — open-position thesis-review tick. Same gating
            # as the retro tick. Runs AFTER retro because a retro fill
            # opening a new position should still get its first thesis
            # review scheduled at entry — that scheduling happens in the
            # agent loop's submission path, not here.
            if (
                self._thesis is not None
                and report is not None
                and not report.suppressed
                and not report.deferred
            ):
                try:
                    await self._thesis.run_tick()
                except Exception as e:  # noqa: BLE001
                    log.error(
                        "reconciler.thesis_tick_error",
                        extra={
                            "event": "reconciler.thesis_tick_error",
                            "error": str(e),
                            "error_class": type(e).__name__,
                        },
                    )
            try:
                await asyncio.wait_for(
                    self._stopping.wait(),
                    timeout=self._cfg.interval_seconds_market,
                )
            except TimeoutError:
                continue

    # ------------------------------------------------------------------
    # Reconcile entry points
    # ------------------------------------------------------------------

    def trigger_after_submission(self) -> ReconcileReport:
        """Force a reconcile regardless of market clock. Called by the
        order submitter after a successful submission so the operator's
        view of state catches up immediately."""
        return self._reconcile(respect_market_hours=False)

    def reconcile_now(self) -> ReconcileReport:
        """Run a single reconcile cycle. Suppressed off-hours per AC-2."""
        return self._reconcile(respect_market_hours=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reconcile(self, *, respect_market_hours: bool) -> ReconcileReport:
        now = self._now_fn()
        if respect_market_hours and not is_market_hours(now):
            return ReconcileReport(matched=True, diffs=[], suppressed=True)

        try:
            broker_positions = self._broker.get_positions()
            broker_account = self._broker.get_account()
        except BrokerUnavailable as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _PERSISTENT_FAILURE_THRESHOLD:
                log.warning(
                    "reconciler.broker_unavailable_persistent",
                    extra={
                        "event": "reconciler.broker_unavailable_persistent",
                        "consecutive_failures": self._consecutive_failures,
                        "error": str(e),
                    },
                )
            else:
                log.info(
                    "reconciler.broker_unavailable_transient",
                    extra={
                        "event": "reconciler.broker_unavailable_transient",
                        "consecutive_failures": self._consecutive_failures,
                    },
                )
            return ReconcileReport(matched=True, diffs=[], deferred=True)

        # Successful broker call — reset the transient counter.
        self._consecutive_failures = 0

        journal_positions = self._journal.get_open_positions()
        diffs = self._compute_diffs(broker_positions, journal_positions)

        # Persist the broker's truth as a `positions` snapshot regardless of
        # match/divergence — the journal is append-only and snapshots are
        # the audit record of state at this instant.
        for pos in broker_positions:
            self._journal.insert_position(
                PositionRow(
                    snapshot_at=now,
                    source="reconciler",
                    symbol=pos.symbol,
                    qty=pos.qty,
                    avg_entry_price=pos.avg_entry_price,
                    market_value=pos.market_value,
                    unrealized_pnl=pos.unrealized_pnl,
                )
            )
        # Account snapshot for the daily-loss / equity-tracking paths
        # (KS-1 / KS-2 / /status).
        self._journal.insert_account_snapshot(
            AccountSnapshotRow(
                snapshot_at=now,
                equity=broker_account.equity,
                cash=broker_account.cash,
                buying_power=broker_account.buying_power,
                long_market_value=broker_account.long_market_value,
                daypl=broker_account.daypl,
            )
        )

        if not diffs:
            log.info(
                "reconciler.match",
                extra={
                    "event": "reconciler.match",
                    "broker_position_count": len(broker_positions),
                },
            )
            return ReconcileReport(matched=True, diffs=[])

        # Divergence: soft halt + alert.
        notes = json.dumps([
            {
                "symbol": d.symbol,
                "broker_qty": d.broker_qty,
                "journal_qty": d.journal_qty,
                "broker_avg_entry": d.broker_avg_entry,
                "journal_avg_entry": d.journal_avg_entry,
            }
            for d in diffs
        ])
        log.error(
            "reconciler.discrepancy",
            extra={
                "event": "reconciler.discrepancy",
                "diff_count": len(diffs),
                "diff_summary": notes,
            },
        )
        self._ks.halt(
            reason="reconciler:discrepancy",
            set_by="system",
            notes=notes,
        )
        if self._alerter is not None:
            self._alerter.notify(
                f"Reconciler: {len(diffs)} position discrepancy(ies); kill-switch halted. {notes}"
            )
        return ReconcileReport(matched=False, diffs=diffs)

    @staticmethod
    def _compute_diffs(
        broker_positions: list[Position],
        journal_positions: list,
    ) -> list[PositionDiff]:
        """Compare broker (truth) and journal (expected) per-symbol.

        Treats journal_positions as duck-typed: any object exposing
        `symbol`, `qty`, `avg_entry_price` works (lets tests pass a thin
        stand-in instead of a full `PositionRow`).
        """
        broker_by = {p.symbol: p for p in broker_positions}
        journal_by = {p.symbol: p for p in journal_positions}
        all_symbols = set(broker_by) | set(journal_by)

        diffs: list[PositionDiff] = []
        for sym in sorted(all_symbols):
            b = broker_by.get(sym)
            j = journal_by.get(sym)
            b_qty = b.qty if b is not None else 0
            j_qty = j.qty if j is not None else 0
            b_avg = b.avg_entry_price if b is not None else None
            j_avg = j.avg_entry_price if j is not None else None
            qty_mismatch = b_qty != j_qty
            # Compare avg_entry only when both sides report a position.
            avg_mismatch = (
                b is not None
                and j is not None
                and abs(b.avg_entry_price - j.avg_entry_price) > 1e-6
            )
            if qty_mismatch or avg_mismatch:
                diffs.append(
                    PositionDiff(
                        symbol=sym,
                        broker_qty=b_qty,
                        journal_qty=j_qty,
                        broker_avg_entry=b_avg,
                        journal_avg_entry=j_avg,
                    )
                )
        return diffs


# Compatibility export for `AccountSnapshot` → AccountSnapshotRow row builder
# is internal; AccountSnapshot from broker.protocol is the *input* type.
_AccountSnapshot = AccountSnapshot  # silence unused-import; referenced in helpers


__all__ = [
    "PositionDiff",
    "ReconcileReport",
    "Reconciler",
]
