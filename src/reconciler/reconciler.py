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

WS1 (D-078, architecture-option-b-2026-06-09.md §2.2): the reconciler is
verification-only. Fill truth arrives via the FillIngestor invoked at the
top of every tick (ADR-010); diff classification is keyed to order
LIFECYCLE — a sell explains a decrease while it is pending fill, or for 2
ticks after its recorded fill — instead of a sliding submitted_at window
(kills RC-2), accepts every sell role incl. thesis_close (kills RC-3),
treats avg-entry drift on matching qty as informational (kills RC-5), and
absorbs externally-closed positions with a tombstone plus a single alert
instead of a permanent halt loop (kills RC-4). Halts carry a diff-set
fingerprint so the kill-switch dedupes the O-5 pager amplifier.
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

# WS1 delta §2.2 external-close absorption: a journal-open symbol absent
# from the broker book with no explaining sell order is tombstoned after
# this many CONSECUTIVE ticks of confirmed absence (mirrors the scanner's
# _POSITION_MISS_THRESHOLD pattern from the 2026-05-11 hotfix). In-memory
# counter — a restart re-counts, which only delays absorption by ≤3 ticks.
_EXTERNAL_CLOSE_MISS_THRESHOLD = 3

# A recorded fill keeps explaining its position diff for this many
# reconcile ticks after realized_fill_at (delta §2.2).
_FILLED_EXPLANATION_TICKS = 2


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
    # WS1 delta §2.2 external-close absorption (RC-4): journal-open symbols
    # absent at the broker with no explaining sell. `external_closes` =
    # symbols tombstoned this tick (absence confirmed 3 consecutive ticks);
    # `external_close_pending` = diffs still inside the grace window.
    # Neither halts — an absent position is zero exposure.
    external_closes: list[str] = field(default_factory=list)
    external_close_pending: list[PositionDiff] = field(default_factory=list)


class _JournalLike(Protocol):
    def get_open_positions(self) -> list: ...
    def insert_position(self, row: PositionRow) -> int: ...
    def insert_account_snapshot(self, row: AccountSnapshotRow) -> int: ...
    def get_lifecycle_orders_for_symbol(
        self, symbol: str, filled_since: dt.datetime
    ) -> list[OrderRow]: ...
    def insert_position_tombstone(
        self,
        symbol: str,
        source: str,
        notes: str,
        *,
        snapshot_at: dt.datetime | None = None,
    ) -> int: ...


class _KillSwitchLike(Protocol):
    def halt(
        self,
        reason: str,
        set_by: str,
        notes: str = "",
        fingerprint: str | None = None,
    ) -> bool: ...


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
        # WS1 (D-078, ADR-010): FillIngestor invoked at the TOP of every
        # reconcile body, before snapshotting and classification, so the
        # tick that detects a fill also explains it. None disables
        # (legacy test paths).
        fill_ingestor: Any | None = None,
        # WS1 seam for WS2 (delta §3.5): ExitPolicyTicker hook — invoked
        # AFTER the thesis hook on every successful, non-suppressed,
        # non-deferred tick (the slot the PDT scanner vacates in P2), so
        # a thesis-driven stop replacement lands before the ratchet reads
        # the live stop. None disables; WS2 wires the real ticker.
        exit_policy_ticker: Any | None = None,
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
        # WS1 wiring + external-close consecutive-miss counters (§2.2).
        self._fill_ingestor = fill_ingestor
        self._exit_policy_ticker = exit_policy_ticker
        self._external_miss: dict[str, int] = {}
        # Negative-cash tripwire edge detector (hotfix 2026-07-08): True
        # while the last observed broker cash was < 0, so a sustained
        # episode alerts once (per ≥0 → <0 crossing), not every tick.
        # In-memory — a restart mid-episode re-alerts at most once.
        self._cash_was_negative = False

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

            # WS1 seam for WS2 (delta §3.5) — exit-policy ticker runs
            # right after the thesis hook so a thesis-driven stop
            # replacement is already journaled when the ratchet looks up
            # the live stop. Same protective shape as the other hooks:
            # errors logged, never propagated; sync like the scanner.
            if (
                self._exit_policy_ticker is not None
                and report is not None
                and not report.suppressed
                and not report.deferred
            ):
                try:
                    self._exit_policy_ticker.run_tick()
                except Exception as e:  # noqa: BLE001
                    log.error(
                        "reconciler.exit_policy_tick_error",
                        extra={
                            "event": "reconciler.exit_policy_tick_error",
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

        # WS1 (ADR-010): ingest fill outcomes FIRST so the journal view
        # the diff pass reads already reflects recorded fills and their
        # tombstones. Errors must never abort the reconcile body — the
        # ingestor is bookkeeping; the reconcile is the safety net.
        if self._fill_ingestor is not None:
            try:
                self._fill_ingestor.run_tick()
            except Exception as e:  # noqa: BLE001
                log.error(
                    "reconciler.fill_ingest_error",
                    extra={
                        "event": "reconciler.fill_ingest_error",
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )

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

        # Negative-cash tripwire (hotfix 2026-07-08): broker cash < 0
        # means an entry buy overshot available cash (a KS-7 slip). This
        # is the seam where account truth is already observed every tick,
        # so check here: WARNING journal-log + one operator alert per
        # crossing. Observational only — no halt, no early return.
        self._check_negative_cash(broker_account)

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
            # All journal-open symbols matched at the broker — any
            # external-close grace counters are stale (symbol reappeared
            # or got tombstoned); clear them so absence must be
            # CONSECUTIVE to absorb.
            self._external_miss.clear()
            log.info(
                "reconciler.match",
                extra={
                    "event": "reconciler.match",
                    "broker_position_count": len(broker_positions),
                },
            )
            return ReconcileReport(matched=True, diffs=[])

        # WS1 (delta §2.2) — lifecycle classification. A diff is explained
        # by orders that are pending fill (any age — kills the RC-2
        # submitted_at time bomb) or filled within the last 2 ticks.
        # Sell-side: ALL roles qualify (kills RC-3). Avg-entry drift on
        # matching qty is informational (kills RC-5). Journal-open symbols
        # absent at the broker with no explaining sell are absorbed as
        # external closes after 3 consecutive ticks (kills RC-4) — an
        # absent position is zero exposure; halting gates entries only
        # and protects nothing there.
        filled_since = now - dt.timedelta(
            seconds=_FILLED_EXPLANATION_TICKS * self._cfg.interval_seconds_market
        )
        expected: list[ExplainedDiff] = []
        unexpected: list[PositionDiff] = []
        ext_pending: list[PositionDiff] = []
        ext_closed: list[str] = []
        # Symbols whose absence counter is live this tick (reported
        # pending OR masked-but-counted, see W1-M1 below) — counters for
        # anything else are cleared after the loop.
        grace_symbols: set[str] = set()
        classifications: dict[str, str] = {}
        per_diff_orders: dict[str, list[int]] = {}
        for d in diffs:
            lifecycle = self._journal.get_lifecycle_orders_for_symbol(
                d.symbol, filled_since
            )
            classification, order_ids = self._classify_diff(d, lifecycle)
            per_diff_orders[d.symbol] = list(order_ids)
            if classification == "drift":
                # Qty agrees; price-only drift (add-on weighted avg,
                # partial-fill drift, splits). Fill truth arrives via the
                # ingestor — this is informational, never halt-worthy.
                log.info(
                    "reconciler.avg_entry_drift",
                    extra={
                        "event": "reconciler.avg_entry_drift",
                        "symbol": d.symbol,
                        "broker_avg_entry": d.broker_avg_entry,
                        "journal_avg_entry": d.journal_avg_entry,
                        "qty": d.broker_qty,
                    },
                )
                classifications[d.symbol] = "expected"
                expected.append(
                    ExplainedDiff(diff=d, explained_by_order_ids=order_ids)
                )
            elif (
                d.broker_qty == 0
                and d.journal_qty > 0
                and not self._fills_cover_absence(d, lifecycle)
            ):
                # External-close absorption: confirmed-absent for N
                # consecutive ticks → tombstone + ONE alert, no halt.
                #
                # W1-M1 (review-option-b-ws1-2026-06-10): the counter runs
                # even when PENDING sells cover the qty — only a RECORDED
                # fill suppresses it. The FillIngestor polls those same
                # pending orders at the top of this very tick, so a sell
                # still pending while the position is absent across the
                # whole grace window cannot be fill latency: it is an
                # operator/broker flatten with the protective order left
                # live (Alpaca's position-close does not cancel open
                # orders). During the grace window an order-covered diff
                # stays explained (the INSG-style fill-visibility race);
                # an uncovered one is reported external_close_pending.
                misses = self._external_miss.get(d.symbol, 0) + 1
                if misses >= _EXTERNAL_CLOSE_MISS_THRESHOLD:
                    live_orders = [
                        o for o in lifecycle
                        if o.side == "sell" and o.final_status is None
                    ]
                    self._absorb_external_close(d, now, live_orders=live_orders)
                    ext_closed.append(d.symbol)
                    classifications[d.symbol] = "external_close"
                else:
                    self._external_miss[d.symbol] = misses
                    grace_symbols.add(d.symbol)
                    if classification == "expected":
                        classifications[d.symbol] = "expected"
                        expected.append(
                            ExplainedDiff(diff=d, explained_by_order_ids=order_ids)
                        )
                    else:
                        ext_pending.append(d)
                        classifications[d.symbol] = "external_close_pending"
                    log.info(
                        "reconciler.external_close_pending",
                        extra={
                            "event": "reconciler.external_close_pending",
                            "symbol": d.symbol,
                            "journal_qty": d.journal_qty,
                            "consecutive_misses": misses,
                            "threshold": _EXTERNAL_CLOSE_MISS_THRESHOLD,
                            "masked_by_live_orders": (
                                classification == "expected"
                            ),
                        },
                    )
            elif classification == "expected":
                classifications[d.symbol] = "expected"
                expected.append(
                    ExplainedDiff(diff=d, explained_by_order_ids=order_ids)
                )
            else:
                classifications[d.symbol] = "unexpected"
                unexpected.append(d)

        # Counters survive only for symbols still inside the grace window
        # this tick — anything explained, reappeared, or tombstoned resets.
        self._external_miss = {
            sym: n for sym, n in self._external_miss.items()
            if sym in grace_symbols
        }

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
                    "external_closes": json.dumps(ext_closed),
                    "external_close_pending": json.dumps(
                        sorted(d.symbol for d in ext_pending)
                    ),
                },
            )
            return ReconcileReport(
                matched=True,
                diffs=[],
                explained_diffs=expected,
                external_closes=ext_closed,
                external_close_pending=ext_pending,
            )

        # At least one unexpected diff → halt. Enrich the diff_summary
        # payload so each row carries its classification (and which order
        # ids explained it, when any).
        notes = json.dumps([
            {
                "symbol": d.symbol,
                "broker_qty": d.broker_qty,
                "journal_qty": d.journal_qty,
                "broker_avg_entry": d.broker_avg_entry,
                "journal_avg_entry": d.journal_avg_entry,
                "classification": classifications[d.symbol],
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
        # WS1 delta §2.3 — deterministic fingerprint over the unexpected
        # diff set; the kill-switch dedupes repeat halts and throttles
        # re-pages while the identical diff set persists.
        fingerprint = "|".join(
            f"{d.symbol}:{d.broker_qty - d.journal_qty}"
            for d in sorted(unexpected, key=lambda d: d.symbol)
        )
        fired = self._ks.halt(
            reason="reconciler:discrepancy",
            set_by="system",
            notes=notes,
            fingerprint=fingerprint,
        )
        if fired and self._alerter is not None:
            self._alerter.notify(
                f"Reconciler: {len(unexpected)} unexpected position "
                f"discrepancy(ies) (of {len(diffs)} total); kill-switch "
                f"halted. {notes}"
            )
        # `diffs` carries only the unexpected (halt-triggering) rows;
        # explained ones move to `explained_diffs`.
        return ReconcileReport(
            matched=False,
            diffs=unexpected,
            explained_diffs=expected,
            external_closes=ext_closed,
            external_close_pending=ext_pending,
        )

    def _check_negative_cash(self, account: AccountSnapshot) -> None:
        """Hotfix 2026-07-08 — alert the operator once per ≥0 → <0 cash
        crossing. Dedupe is a simple in-memory edge detector: consecutive
        negative ticks inside one episode stay silent; recovery to ≥0
        re-arms it. Chosen over a per-day cap for simplicity — it also
        catches a second distinct crossing on the same day."""
        if account.cash >= 0:
            self._cash_was_negative = False
            return
        if self._cash_was_negative:
            return
        self._cash_was_negative = True
        log.warning(
            "account.cash_negative",
            extra={
                "event": "account.cash_negative",
                "cash": account.cash,
                "equity": account.equity,
                "buying_power": account.buying_power,
            },
        )
        if self._alerter is not None:
            self._alerter.notify(
                f"Account cash NEGATIVE: cash=${account.cash:.2f}, "
                f"equity=${account.equity:.2f}. An entry buy overshot "
                f"available cash (KS-7 gap). Check open orders and "
                f"recent fills at the broker."
            )

    @staticmethod
    def _fills_cover_absence(
        d: PositionDiff, lifecycle_orders: list[OrderRow]
    ) -> bool:
        """True when RECORDED sell fills (already window-filtered by the
        lifecycle query) cover the full journal qty of a broker-absent
        symbol — genuine fill latency, never a masked external close
        (W1-M1). Pending rows deliberately do not count here."""
        recorded = sum(
            o.realized_fill_qty or 0
            for o in lifecycle_orders
            if o.side == "sell"
            and o.final_status in ("filled", "partially_filled_closed")
        )
        return recorded >= d.journal_qty

    def _absorb_external_close(
        self,
        d: PositionDiff,
        now: dt.datetime,
        *,
        live_orders: list[OrderRow] | None = None,
    ) -> None:
        """Delta §2.2 RC-4 absorption: tombstone + single alert, no halt.
        Broker liquidations and operator manual flattens both land here,
        visibly, once. W1-M1: when live (pending) sell orders still cover
        the vanished position, the alert names them — they need operator
        cancellation at the broker, or their eventual fill would sell
        shares we no longer hold."""
        live = live_orders or []
        live_desc = ", ".join(
            f"journal order {o.id} (broker {o.broker_order_id}, role {o.role})"
            for o in live
        )
        notes = (
            f"externally closed at broker (journal qty {d.journal_qty}); "
            f"confirmed absent {_EXTERNAL_CLOSE_MISS_THRESHOLD} "
            f"consecutive ticks; D-078"
        )
        if live:
            notes += f"; live sell orders left open: {live_desc}"
        self._journal.insert_position_tombstone(
            d.symbol,
            "reconciler_external_close",
            notes,
            snapshot_at=now,
        )
        event = (
            "position.closed_externally_live_orders"
            if live
            else "position.closed_externally"
        )
        log.warning(
            event,
            extra={
                "event": event,
                "symbol": d.symbol,
                "journal_qty": d.journal_qty,
                "live_order_ids": [o.id for o in live],
            },
        )
        if self._alerter is not None:
            if live:
                self._alerter.notify(
                    f"Position closed externally at broker: {d.symbol} "
                    f"(journal qty {d.journal_qty}) while sell order(s) "
                    f"are STILL LIVE: {live_desc}. Cancel them at the "
                    f"broker — a later fill would sell shares we no "
                    f"longer hold. Tombstoned; no halt — zero exposure."
                )
            else:
                self._alerter.notify(
                    f"Position closed externally at broker: {d.symbol} "
                    f"(journal qty {d.journal_qty}, no journaled sell). "
                    f"Tombstoned; no halt — zero exposure. Verify at broker "
                    f"if this wasn't you."
                )

    @staticmethod
    def _classify_diff(
        diff: PositionDiff, lifecycle_orders: list[OrderRow]
    ) -> tuple[str, tuple[int, ...]]:
        """Classify a `PositionDiff` against the symbol's LIFECYCLE orders
        (pending fill, or filled within the caller's 2-tick window).
        Returns ('expected'|'unexpected'|'drift', explaining_order_ids).

        Rules (WS1, D-078 delta §2.2 — replaces the hotfix-2026-05-07
        role whitelist + submitted_at window):
          - qtys equal, avg_entry drifted → 'drift' (informational; the
            caller logs and treats as expected — kills RC-5).
          - broker_qty > journal_qty: explained by role='entry', side='buy'
            rows whose qty covers the delta.
          - broker_qty < journal_qty: explained by side='sell' rows of ANY
            role — stop, take_profit, thesis_close, trailing_stop, future
            roles — whose qty covers the delta (kills RC-3; §7.5 closes
            the vocabulary so a new role can never reintroduce it).
          - a filled row contributes its realized_fill_qty (ground truth);
            a pending row contributes its submitted qty.
          - partial coverage stays 'unexpected' — never silently accept a
            partial explanation as full.

        `lifecycle_orders` is already filtered to this symbol and the
        lifecycle window (journal query); we only filter side/role here.
        """
        delta = diff.broker_qty - diff.journal_qty
        if delta == 0:
            # _compute_diffs only emits equal-qty diffs for avg mismatch.
            return ("drift", ())

        wanted_side = "buy" if delta > 0 else "sell"
        explaining_qty = 0
        explaining_ids: list[int] = []
        for o in lifecycle_orders:
            if o.side != wanted_side:
                continue
            if delta > 0 and o.role != "entry":
                continue
            if (
                o.final_status in ("filled", "partially_filled_closed")
                and o.realized_fill_qty is not None
            ):
                explaining_qty += o.realized_fill_qty
            else:
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
