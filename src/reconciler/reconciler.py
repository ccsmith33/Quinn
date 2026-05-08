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
from dataclasses import dataclass, field
from typing import Any, Protocol

from broker.alpaca import BrokerUnavailable
from broker.protocol import AccountSnapshot, BrokerAdapter, Position
from config.calendar import is_market_hours
from config.loader import ReconcilerConfig
from journal.models import AccountSnapshotRow, OrderRow, PositionRow
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
class ExplainedDiff:
    """A `PositionDiff` whose qty mismatch is fully accounted for by recent
    rows in the journal's `orders` table (post-bracket-submission window).
    Carried in `ReconcileReport.explained_diffs` for caller introspection
    and structured-log payloads.
    """

    diff: PositionDiff
    explained_by_order_ids: tuple[int, ...]


@dataclass(frozen=True)
class ReconcileReport:
    matched: bool
    diffs: list[PositionDiff]
    deferred: bool = False
    suppressed: bool = False
    # Hotfix 2026-05-07 — diffs that the recent-orders classifier explained
    # away as expected (bracket entry just filled, stop/TP just filled).
    # `matched` is True and `diffs` is empty whenever every diff is
    # explained; the explained list is preserved here for the operator.
    explained_diffs: list[ExplainedDiff] = field(default_factory=list)


class _JournalLike(Protocol):
    def get_open_positions(self) -> list: ...
    def insert_position(self, row: PositionRow) -> int: ...
    def insert_account_snapshot(self, row: AccountSnapshotRow) -> int: ...
    def get_orders_since(
        self, symbol: str, since: dt.datetime
    ) -> list[OrderRow]: ...


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
        # PDT-SUNSET-2026-06-04: ADR-009 §3.1 — activation flag holder.
        # `None` disables the refresh hook (legacy test paths). When
        # provided, the reconciler refreshes it after every successful
        # tick using the just-reconciled `broker_account`.
        pdt_state: Any | None = None,
        pdt_enabled: bool = True,
        # PDT-SUNSET-2026-06-04: ADR-009 §"Scanner hook" — invoked
        # after retro/thesis on every successful, non-suppressed,
        # non-deferred tick. None disables the hook.
        virtual_exits_scanner: Any | None = None,
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
        # PDT-SUNSET-2026-06-04: ADR-009 wiring.
        self._pdt_state = pdt_state
        self._pdt_enabled = pdt_enabled
        self._virtual_exits_scanner = virtual_exits_scanner

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

            # PDT-SUNSET-2026-06-04: ADR-009 §"Scanner hook" — virtual
            # exit scanner runs once per successful, non-suppressed,
            # non-deferred tick. Same protective shape as retro/thesis:
            # errors logged, never propagated; runs synchronously
            # (scanner is sync).
            if (
                self._virtual_exits_scanner is not None
                and report is not None
                and not report.suppressed
                and not report.deferred
            ):
                try:
                    self._virtual_exits_scanner.run_tick()
                except Exception as e:  # noqa: BLE001
                    log.error(
                        "reconciler.scanner_tick_error",
                        extra={
                            "event": "reconciler.scanner_tick_error",
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

        # PDT-SUNSET-2026-06-04: ADR-009 §3.1 — refresh activation flag
        # AFTER the position-reconcile body succeeded (broker_account in
        # hand) and BEFORE retro/thesis/scanner hooks fire. The refresh
        # is idempotent (one comparison + two attribute writes) and
        # makes no extra broker call. Errors are caught defensively so
        # a refresh hiccup never aborts the reconcile tick.
        if self._pdt_state is not None:
            try:
                self._pdt_state.refresh(
                    broker_account, pdt_enabled=self._pdt_enabled
                )
            except Exception as e:  # noqa: BLE001
                log.error(
                    "reconciler.pdt_refresh_error",
                    extra={
                        "event": "reconciler.pdt_refresh_error",
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
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

        # Hotfix 2026-05-07 — classify each diff against the journal's
        # recent `orders` rows. Diffs whose qty delta is fully accounted
        # for by a recent bracket-leg submission (entry-buy fill increases
        # broker qty; stop/TP-sell fill decreases it) are "expected" and
        # do not halt. Avg_entry mismatches are always unexpected.
        window_minutes = self._cfg.expected_fill_window_minutes
        since = now - dt.timedelta(minutes=window_minutes)
        expected: list[ExplainedDiff] = []
        unexpected: list[PositionDiff] = []
        # Per-diff explanation (always populated for unexpected diffs too,
        # so the structured-log payload can show "no orders explained
        # this divergence" alongside the raw qty numbers).
        per_diff_orders: dict[str, list[int]] = {}
        for d in diffs:
            recent = self._journal.get_orders_since(d.symbol, since)
            classification, order_ids = self._classify_diff(d, recent)
            per_diff_orders[d.symbol] = list(order_ids)
            if classification == "expected":
                expected.append(
                    ExplainedDiff(diff=d, explained_by_order_ids=order_ids)
                )
            else:
                unexpected.append(d)

        # All diffs explained → no halt. Log the pending-fill match event.
        if not unexpected:
            log.info(
                "reconciler.match_with_pending_fills",
                extra={
                    "event": "reconciler.match_with_pending_fills",
                    "broker_position_count": len(broker_positions),
                    "explained_diffs": json.dumps([
                        {
                            "symbol": ed.diff.symbol,
                            "broker_qty": ed.diff.broker_qty,
                            "journal_qty": ed.diff.journal_qty,
                            "explained_by_order_ids": list(
                                ed.explained_by_order_ids
                            ),
                        }
                        for ed in expected
                    ]),
                },
            )
            return ReconcileReport(
                matched=True, diffs=[], explained_diffs=expected
            )

        # At least one unexpected diff → halt. Enrich the diff_summary
        # payload so each row carries its classification (and which order
        # ids explained it, when any).
        expected_symbols = {ed.diff.symbol for ed in expected}
        notes = json.dumps([
            {
                "symbol": d.symbol,
                "broker_qty": d.broker_qty,
                "journal_qty": d.journal_qty,
                "broker_avg_entry": d.broker_avg_entry,
                "journal_avg_entry": d.journal_avg_entry,
                "classification": (
                    "expected" if d.symbol in expected_symbols else "unexpected"
                ),
                "explained_by_order_ids": per_diff_orders.get(d.symbol, []),
            }
            for d in diffs
        ])
        log.error(
            "reconciler.discrepancy",
            extra={
                "event": "reconciler.discrepancy",
                "diff_count": len(diffs),
                "unexpected_count": len(unexpected),
                "expected_count": len(expected),
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
                f"Reconciler: {len(unexpected)} unexpected position "
                f"discrepancy(ies) (of {len(diffs)} total); kill-switch "
                f"halted. {notes}"
            )
        # `diffs` carries only the unexpected (halt-triggering) rows;
        # explained ones move to `explained_diffs`.
        return ReconcileReport(
            matched=False, diffs=unexpected, explained_diffs=expected
        )

    @staticmethod
    def _classify_diff(
        diff: PositionDiff, recent_orders: list[OrderRow]
    ) -> tuple[str, tuple[int, ...]]:
        """Classify a `PositionDiff` against recent `orders` rows for the
        same symbol. Returns ('expected'|'unexpected', explaining_order_ids).

        Rules (hotfix 2026-05-07):
          - broker_qty > journal_qty: an "expected" increase requires
            recent role='entry', side='buy' orders whose summed qty covers
            the delta.
          - broker_qty < journal_qty: an "expected" decrease requires
            recent role IN ('stop','take_profit'), side='sell' orders
            whose summed qty covers the delta.
          - qtys equal but avg_entry differs: ALWAYS unexpected — recent
            orders cannot explain a price drift on the same share count.
          - mixed qty diff AND avg_entry diff: ALWAYS unexpected — the
            avg_entry mismatch is the giveaway, even if the qty delta
            would otherwise be expected.

        `recent_orders` should already be filtered to the time window
        (caller-side) and to this diff's symbol; we only filter by
        role/side here.
        """
        # Avg_entry mismatch alone (or in combination) → always unexpected.
        # The reconciler computes `avg_mismatch` only when both sides have
        # a position; if either side is None, b_avg/j_avg might be None.
        if (
            diff.broker_avg_entry is not None
            and diff.journal_avg_entry is not None
            and abs(diff.broker_avg_entry - diff.journal_avg_entry) > 1e-6
        ):
            return ("unexpected", ())

        delta = diff.broker_qty - diff.journal_qty
        if delta == 0:
            # No qty diff and we already cleared the avg_entry path above
            # (either both sides agree or one side is missing avg). This
            # branch should be unreachable — _compute_diffs only emits a
            # PositionDiff when qty OR avg differs — but be defensive.
            return ("unexpected", ())

        if delta > 0:
            wanted_roles = {"entry"}
            wanted_side = "buy"
        else:
            wanted_roles = {"stop", "take_profit"}
            wanted_side = "sell"

        explaining_qty = 0
        explaining_ids: list[int] = []
        for o in recent_orders:
            if o.role not in wanted_roles or o.side != wanted_side:
                continue
            explaining_qty += o.qty
            if o.id is not None:
                explaining_ids.append(o.id)

        if explaining_qty >= abs(delta):
            return ("expected", tuple(explaining_ids))
        return ("unexpected", tuple(explaining_ids))

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
    "ExplainedDiff",
    "PositionDiff",
    "ReconcileReport",
    "Reconciler",
]
