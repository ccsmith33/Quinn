# PDT-SUNSET-2026-06-04: ADR-009 §3.2 / §"Order construction branch".
"""Virtual exit scanner + EV allocator (S-PDT-4).

Per-tick scan path:
  1. Read all `state='active'` virtual exits from the journal.
  2. Fetch one quote per unique symbol.
  3. Mark "ready-to-sell" any virtual exit whose threshold is crossed
     (stop ↓, tp ↑).
  4. Compute EV per ready exit (D-071: stop = (current − stop) × qty,
     tp = (tp − entry) × qty).
  5. Sort by EV descending, ties by `id` ascending (deterministic).
  6. Compute `budget = max(0, 3 − daytrade_count − pending)`.
  7. Submit the top-`budget` ready exits via `broker.submit_order`.
  8. Defer the rest into `deferred_sells` with `deferred_reason='ev_lost'`.

The defensive 403 handler routes a `PDTBudgetExceeded` from the broker
during submit into a `deferred_sells` row with `deferred_reason='pdt_403'`
and continues — does NOT propagate. Other `BrokerUnavailable` failures
(transient 503/network) skip the row without deferring; the next tick
will retry.

Position closed externally — when a virtual_exit row references a
symbol not present in `broker.get_positions()`, the row is marked
`obsolete` ("position_closed_externally") and skipped.

The whole module short-circuits to a no-op when `pdt_state.is_active()`
is False (e.g., `last_equity >= 25k` or operator escape hatch flipped).

Sunset: drop this module on FINRA PDT-rule retirement (2026-06-04).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from alpaca.common.exceptions import APIError

from broker.alpaca import BrokerUnavailable
from broker.protocol import BrokerAdapter, OrderRequest, SubmittedOrder
from config.calendar import ET
from execution.pdt_budget import (
    PDTBudgetExceeded,
    PDTState,
    compute_budget_remaining,
)
from journal.models import DeferredSellRow, OrderRow, VirtualExitRow
from observability.log_port import get_logger

log = get_logger(__name__)


class _JournalLike(Protocol):
    def list_active_virtual_exits(self) -> list[VirtualExitRow]: ...
    def mark_virtual_exit_submitted(
        self, vid: int, broker_order_id: str
    ) -> None: ...
    def mark_virtual_exit_obsolete(self, vid: int, reason: str) -> None: ...
    def insert_deferred_sell(self, row: DeferredSellRow) -> int: ...
    # PDT-SUNSET-2026-06-04: S-PDT-5 — replayer reads + drains.
    def list_unreplayed_deferred_sells(self) -> list[DeferredSellRow]: ...
    def mark_deferred_replayed(
        self, did: int, broker_order_id: str
    ) -> None: ...
    def mark_deferred_skipped(self, did: int, reason: str) -> None: ...
    # Supersede-race check (replayer-side guard).
    def get_virtual_exit_state(self, vid: int) -> str | None: ...
    # HOTFIX-2026-05-08: PDT sells must be journaled for reconciler
    # tolerance to explain the broker-position decrease.
    def insert_order(self, row: OrderRow) -> int: ...


@dataclass(frozen=True)
class ScannerReport:
    """Counts emitted from one `run_tick` invocation. For logging only;
    callers MUST NOT branch on its values (the scanner's behavior is
    self-contained — D-070)."""

    ready: int
    submitted: int
    deferred_ev: int
    deferred_403: int


def compute_ev(exit: VirtualExitRow, current_price: float) -> float:
    """ADR-009 §"EV computation contract" / D-071.

    - role='stop': `(current - stop_price) * qty` — loss avoided
      (negative when stop is breached and selling crystallizes the loss;
      ranked vs other negatives by less-negative-first).
    - role='tp':   `(tp_price - entry_price) * qty` — gain locked
      (always positive given a sane proposal where `tp > entry`).

    `thesis_close` is NOT a virtual_exits role (only deferred_sells);
    raises ValueError if encountered.
    """
    if exit.role == "stop":
        if exit.stop_price is None:
            raise ValueError(f"stop virtual_exit id={exit.id} missing stop_price")
        return (current_price - exit.stop_price) * exit.qty
    if exit.role == "tp":
        if exit.tp_price is None:
            raise ValueError(f"tp virtual_exit id={exit.id} missing tp_price")
        return (exit.tp_price - exit.entry_price) * exit.qty
    raise ValueError(f"unknown virtual_exit role: {exit.role!r}")


def _is_threshold_crossed(e: VirtualExitRow, current_price: float) -> bool:
    """ADR-009 §3 — `stop` fires at `current <= stop_price`,
    `tp` fires at `current >= tp_price`. Inclusive on the boundary so
    a quote exactly at threshold is a fire (matches broker stop-order
    semantics)."""
    if e.role == "stop":
        if e.stop_price is None:
            return False
        return current_price <= e.stop_price
    if e.role == "tp":
        if e.tp_price is None:
            return False
        return current_price >= e.tp_price
    return False


def _to_sell_request(r: VirtualExitRow) -> OrderRequest:
    """Build the OrderRequest the scanner submits when a threshold is
    crossed.

    `client_order_id='pdt-vexit-{id}'` is deterministic per virtual exit
    so a duplicate submit (e.g., crash mid-tick) hits Alpaca's idempotent
    dedup on client_order_id.
    """
    if r.role == "stop":
        # Threshold already crossed; submit a market sell (fastest exit
        # at-or-near current price). Submitting a `stop` order would
        # consume the same day-trade budget and would just fire
        # immediately — market is the simpler choice.
        return OrderRequest(
            symbol=r.symbol,
            side="sell",
            qty=r.qty,
            order_type="market",
            tif="day",
            client_order_id=f"pdt-vexit-{r.id}",
        )
    if r.role == "tp":
        if r.tp_price is None:
            raise ValueError(f"tp virtual_exit id={r.id} missing tp_price")
        return OrderRequest(
            symbol=r.symbol,
            side="sell",
            qty=r.qty,
            order_type="limit",
            tif="day",
            limit_price=r.tp_price,
            client_order_id=f"pdt-vexit-{r.id}",
        )
    raise ValueError(f"unsupported virtual_exit role: {r.role!r}")


# HOTFIX-2026-05-11: consecutive-miss threshold for marking a virtual
# exit obsolete via the position_closed_externally path. The 2026-05-11
# open-of-day incident: scanner ran during the partial-fill window
# right after 9:30 ET, broker.get_positions() returned a stale view
# (some symbols still queued), and one tick was enough to nuke
# virtual_exits for not-yet-filled symbols — leaving them unprotected
# once the fills landed seconds later. Requiring N consecutive misses
# before the obsolete-mark fires absorbs the fill burst without
# eternally postponing the legitimate manual-close case.
_POSITION_MISS_THRESHOLD = 3


class VirtualExitScanner:
    """Once-per-reconciler-tick scanner. Constructed in composition,
    invoked from `reconciler._run_loop` after the retro/thesis hooks.

    Re-reads the journal, account, and per-symbol quotes every tick.
    The PDTState reference is shared with the agent loop and reconciler
    (single source of truth).

    HOTFIX-2026-05-11: maintains a small in-memory miss counter per
    virtual_exit id so the position_closed_externally check can require
    N consecutive missing ticks before firing. Counter resets the
    instant the position reappears, so an open-of-day fill-burst race
    can't drift past the grace window indefinitely.
    """

    def __init__(
        self,
        *,
        broker: BrokerAdapter,
        journal: _JournalLike,
        pdt_state: PDTState,
    ) -> None:
        self._broker = broker
        self._journal = journal
        self._pdt_state = pdt_state
        # HOTFIX-2026-05-11: virtual_exit.id -> consecutive missing ticks.
        # Bounded by the active virtual_exit count; trimmed on every tick.
        self._position_miss_counts: dict[int, int] = {}

    def run_tick(self) -> ScannerReport:
        if not self._pdt_state.is_active():
            return ScannerReport(0, 0, 0, 0)

        exits = self._journal.list_active_virtual_exits()
        if not exits:
            # No active virtual exits — skip account/positions calls
            # entirely (quiet-tick fast path).
            return ScannerReport(0, 0, 0, 0)

        # Fresh broker state. `daytrade_count` for budget arithmetic;
        # positions for the position-closed-externally check.
        account = self._broker.get_account()
        broker_positions = self._broker.get_positions()
        open_symbols = {p.symbol for p in broker_positions if p.qty != 0}

        # Mark virtual exits whose underlying position has been closed
        # externally (operator manual exit) — ADR-009 / S-PDT-4 AC-7.
        # HOTFIX-2026-05-11: require `_POSITION_MISS_THRESHOLD` consecutive
        # missing ticks before marking obsolete. The 2026-05-11 open-of-day
        # incident: scanner ran ~1s after a halt during the 9:30 partial-fill
        # window; broker.get_positions() did not yet contain the fresh
        # Monday-open buys, and the scanner nuked virtual_exits for those
        # symbols — they were unprotected by the time their fills landed
        # seconds later. The counter resets the tick a position reappears,
        # so a transient absence cannot accumulate across unrelated days.
        live_exits: list[VirtualExitRow] = []
        seen_ids: set[int] = set()
        for e in exits:
            if e.id is None:
                # Defensive — DB always assigns ids; never gate live_exits
                # on a row with no id.
                continue
            seen_ids.add(e.id)
            if e.symbol not in open_symbols:
                miss_count = self._position_miss_counts.get(e.id, 0) + 1
                self._position_miss_counts[e.id] = miss_count
                if miss_count >= _POSITION_MISS_THRESHOLD:
                    self._journal.mark_virtual_exit_obsolete(
                        e.id, reason="position_closed_externally"
                    )
                    log.info(
                        "pdt.virtual_exit.obsolete",
                        extra={
                            "event": "pdt.virtual_exit.obsolete",
                            "virtual_exit_id": e.id,
                            "symbol": e.symbol,
                            "reason": "position_closed_externally",
                            "consecutive_misses": miss_count,
                        },
                    )
                    # Drop from the in-memory grace map; row is now
                    # obsolete in the journal so it won't return next
                    # tick from list_active_virtual_exits().
                    self._position_miss_counts.pop(e.id, None)
                else:
                    log.info(
                        "pdt.virtual_exit.position_missing_grace",
                        extra={
                            "event": "pdt.virtual_exit.position_missing_grace",
                            "virtual_exit_id": e.id,
                            "symbol": e.symbol,
                            "consecutive_misses": miss_count,
                            "threshold": _POSITION_MISS_THRESHOLD,
                        },
                    )
                continue
            # Position present — reset the grace counter (if any). A
            # missing-then-present sequence shouldn't accumulate; only
            # truly consecutive misses count.
            if e.id in self._position_miss_counts:
                self._position_miss_counts.pop(e.id, None)
            live_exits.append(e)

        # Garbage-collect counters for ids that have left the active set
        # entirely (row moved to 'submitted' / 'obsolete' on a prior tick).
        # Bounds the dict to the active virtual_exits at any moment.
        for stale_id in [vid for vid in self._position_miss_counts if vid not in seen_ids]:
            self._position_miss_counts.pop(stale_id, None)

        # Quote per unique symbol (one fetch per symbol, regardless of
        # how many virtual_exit rows reference it).
        symbols = {e.symbol for e in live_exits}
        current_prices: dict[str, float] = {}
        for sym in symbols:
            quote = self._broker.get_quote(sym)
            current_prices[sym] = quote.last

        ready: list[tuple[VirtualExitRow, float]] = []
        for e in live_exits:
            cp = current_prices[e.symbol]
            if _is_threshold_crossed(e, cp):
                ev = compute_ev(e, cp)
                ready.append((e, ev))

        # Sort by (-ev, id ASC) — descending EV, FIFO tie-break.
        ready.sort(key=lambda re: (-re[1], re[0].id or 0))

        # PDT-SUNSET-2026-06-04: ADR-009 §"Why pending_dt_orders is
        # tracked locally" / FINDING-2 — increment a local pending
        # counter on each successful submit and re-evaluate the budget
        # per-row before deciding submit-vs-defer. This prevents
        # within-tick over-allocation when an external (e.g. operator)
        # day-trade landed between our `get_account()` snapshot and
        # the scanner's per-row submit. Pending is incremented ONLY on
        # successful submit (not on transient broker failures, which
        # don't burn an Alpaca-side slot, and not on PDTBudgetExceeded,
        # which Alpaca already counted against us).
        initial_budget = compute_budget_remaining(account, pending=0)
        pending = 0
        submitted = 0
        deferred_ev = 0
        deferred_403 = 0

        for r, ev in ready:
            if r.id is None:  # defensive — DB always assigns ids
                continue
            current_budget = compute_budget_remaining(account, pending=pending)
            if current_budget <= 0:
                # Local budget exhausted (initial slice consumed OR
                # external Alpaca day-trade bumped our prediction).
                # Defer the rest.
                self._defer(r, ev, current_prices[r.symbol], reason="ev_lost")
                deferred_ev += 1
                continue

            req = _to_sell_request(r)
            try:
                resp = self._broker.submit_order(req)
            except PDTBudgetExceeded as e:
                # Local budget prediction diverged from Alpaca's truth.
                # Route to deferred_sells with deferred_reason='pdt_403'.
                # Do NOT increment `pending`: Alpaca rejected, so no
                # broker-side slot was consumed beyond what their
                # `daytrade_count` already reports.
                log.warning(
                    "pdt.local_budget_diverged",
                    extra={
                        "event": "pdt.local_budget_diverged",
                        "expected_budget": current_budget,
                        "account_daytrade_count": account.daytrade_count,
                        "pending": pending,
                        "symbol": r.symbol,
                        "attempted_role": r.role,
                        "error": str(e),
                    },
                )
                self._defer(r, ev, current_prices[r.symbol], reason="pdt_403")
                deferred_403 += 1
                continue
            except BrokerUnavailable as e:
                # Transient broker failure — do NOT defer (would
                # over-defer for routine 503/network blips). Next tick
                # retries this row from `list_active_virtual_exits`.
                # Do NOT increment `pending`: no slot was burned.
                log.warning(
                    "pdt.scanner.broker_unavailable_transient",
                    extra={
                        "event": "pdt.scanner.broker_unavailable_transient",
                        "symbol": r.symbol,
                        "virtual_exit_id": r.id,
                        "error": str(e),
                    },
                )
                continue
            # HOTFIX-2026-05-08: PDT sells must be journaled for reconciler
            # tolerance to explain the broker-position decrease.
            self._journal.insert_order(
                _build_sell_order_row(
                    execution_id=r.execution_id,
                    role=r.role,
                    req=req,
                    resp=resp,
                )
            )
            self._journal.mark_virtual_exit_submitted(
                r.id, broker_order_id=resp.broker_order_id
            )
            log.info(
                "pdt.virtual_exit.submitted",
                extra={
                    "event": "pdt.virtual_exit.submitted",
                    "virtual_exit_id": r.id,
                    "broker_order_id": resp.broker_order_id,
                    "ev": ev,
                    "symbol": r.symbol,
                    "role": r.role,
                },
            )
            pending += 1
            submitted += 1

        # `budget` for the tick log: report the initial snapshot value
        # so operator/log readers can correlate with `daytrade_count`.
        budget = initial_budget

        report = ScannerReport(
            ready=len(ready),
            submitted=submitted,
            deferred_ev=deferred_ev,
            deferred_403=deferred_403,
        )
        log.info(
            "pdt.scanner.tick",
            extra={
                "event": "pdt.scanner.tick",
                "ready_count": report.ready,
                "submitted": report.submitted,
                "deferred_ev": report.deferred_ev,
                "deferred_403": report.deferred_403,
                "budget": budget,
                "account_daytrade_count": account.daytrade_count,
            },
        )
        return report

    def _defer(
        self,
        r: VirtualExitRow,
        ev: float,
        trigger_price: float,
        *,
        reason: str,
    ) -> None:
        """Insert a `deferred_sells` row for the given virtual exit.

        `reason` is `'ev_lost'` (lost EV ranking) or `'pdt_403'` (broker
        rejected; local prediction diverged). The source virtual_exit
        row is left `state='active'` so the next tick can re-evaluate
        — the deferred row is the carry-forward; the active row is the
        live signal source.
        """
        if r.id is None:
            return
        self._journal.insert_deferred_sell(
            DeferredSellRow(
                virtual_exit_id=r.id,
                execution_id=r.execution_id,
                proposal_id=r.proposal_id,
                symbol=r.symbol,
                qty=r.qty,
                role=r.role,  # 'stop' or 'tp' on the active row.
                trigger_price=trigger_price,
                ev_at_defer=ev,
                deferred_reason=reason,  # type: ignore[arg-type]
            )
        )


# ---------------------------------------------------------------------------
# PDT-SUNSET-2026-06-04: S-PDT-5 — pre-market deferred-sell replayer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayReport:
    """Counts emitted from one `DeferredSellReplayer.run` invocation."""

    replayed: int
    skipped_no_position: int
    skipped_error: int
    skipped_today: int
    # PDT-SUNSET-2026-06-04: deferred_sells row was superseded — its
    # source virtual_exit transitioned to 'submitted' (won the EV race
    # on a later same-session tick) or 'obsolete' (position closed
    # externally) before this replayer run. Marked replayed_at with a
    # SUPERSEDED:* note so the row exits the unreplayed queue without
    # a broker submit.
    skipped_superseded: int = 0


def _today_et(now_fn: Callable[[], dt.datetime]) -> dt.date:
    """ET trading-day boundary. DST-safe via zoneinfo."""
    now = now_fn()
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.UTC)
    return now.astimezone(ET).date()


def _deferred_at_to_date_et(
    deferred_at: dt.datetime | str | None,
) -> dt.date | None:
    """Coerce a `deferred_at` value (TIMESTAMP from SQLite — may be a
    str like '2026-05-07 14:30:00' or a datetime) to a date in ET.
    Returns None if unparseable.
    """
    if deferred_at is None:
        return None
    if isinstance(deferred_at, str):
        # SQLite stores TIMESTAMPs as ISO strings; treat as UTC.
        try:
            parsed = dt.datetime.fromisoformat(deferred_at)
        except ValueError:
            return None
    else:
        parsed = deferred_at
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(ET).date()


def _to_replay_request(d: DeferredSellRow) -> OrderRequest:
    """Build the OrderRequest the replayer submits for a deferred sell.

    `client_order_id='pdt-replay-{id}'` is deterministic per
    deferred_sells.id; if a previous boot crashed mid-replay, Alpaca's
    idempotent dedup on client_order_id rejects the second submit. AC-6
    handles that case by reconciling with the broker's existing order.
    """
    if d.role == "stop" or d.role == "thesis_close":
        return OrderRequest(
            symbol=d.symbol, side="sell", qty=d.qty,
            order_type="market", tif="day",
            client_order_id=f"pdt-replay-{d.id}",
        )
    if d.role == "tp":
        return OrderRequest(
            symbol=d.symbol, side="sell", qty=d.qty,
            order_type="limit", tif="day",
            limit_price=d.trigger_price,  # the TP threshold the row was deferred at
            client_order_id=f"pdt-replay-{d.id}",
        )
    raise ValueError(f"unsupported deferred_sells role: {d.role!r}")


# HOTFIX-2026-05-08: PDT sells must be journaled for reconciler tolerance
# to explain the broker-position decrease (otherwise the reconciler halts on
# every virtual-stop fire). Map virtual_exits / deferred_sells role to the
# journal `orders.role` convention used by the entry path
# (src/execution/orders.py): "stop" → "stop", "tp" → "take_profit",
# "thesis_close" → "thesis_close".
def _order_role_for_pdt_sell(role: str) -> str:
    if role == "tp":
        return "take_profit"
    return role


def _build_sell_order_row(
    *,
    execution_id: int,
    role: str,
    req: OrderRequest,
    resp: SubmittedOrder,
) -> OrderRow:
    return OrderRow(
        execution_id=execution_id,
        role=_order_role_for_pdt_sell(role),
        symbol=req.symbol,
        side=req.side,
        order_type=req.order_type,
        qty=req.qty,
        limit_price=req.limit_price,
        stop_price=req.stop_price,
        tif=req.tif,
        broker_order_id=resp.broker_order_id,
        submitted_at=resp.submitted_at,
        final_status=resp.status,
    )


def _is_duplicate_client_order_id(exc: BaseException) -> bool:
    """Alpaca returns a 422 with `client_order_id must be unique` when
    a deterministic client_order_id has already been used. Match
    defensively on status_code + substring.
    """
    if not isinstance(exc, APIError):
        return False
    if getattr(exc, "status_code", None) not in (409, 422):
        return False
    msg = str(exc).lower()
    return "client_order_id" in msg or "duplicate" in msg


class DeferredSellReplayer:
    """One-shot pre-market replayer for `deferred_sells` rows.

    Drains rows whose `deferred_at` is from a prior ET trading day,
    submits them as broker orders. Rows from today are skipped (they
    would still consume same-day PDT budget).

    Errors per row are logged and swallowed — a single bad row never
    aborts the run. This matches the `RetroFillCoordinator` pattern.
    """

    def __init__(
        self,
        *,
        broker: BrokerAdapter,
        journal: _JournalLike,
        now_fn: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    ) -> None:
        self._broker = broker
        self._journal = journal
        self._now_fn = now_fn

    def run(self) -> ReplayReport:
        unreplayed = self._journal.list_unreplayed_deferred_sells()
        if not unreplayed:
            log.info(
                "pdt.replay.run",
                extra={
                    "event": "pdt.replay.run",
                    "replayed_count": 0,
                    "skipped_no_position": 0,
                    "skipped_error": 0,
                    "skipped_today": 0,
                    "skipped_superseded": 0,
                },
            )
            return ReplayReport(0, 0, 0, 0, 0)

        today = _today_et(self._now_fn)
        replayed = 0
        skipped_no_position = 0
        skipped_error = 0
        skipped_today = 0
        skipped_superseded = 0

        # One positions snapshot for the whole replay run — pre-market
        # is quiet, no risk of mid-replay drift, and saves O(N) calls.
        try:
            positions = self._broker.get_positions()
        except Exception as e:  # noqa: BLE001
            log.error(
                "pdt.replay.positions_unavailable",
                extra={
                    "event": "pdt.replay.positions_unavailable",
                    "error": str(e),
                    "error_class": type(e).__name__,
                },
            )
            return ReplayReport(0, 0, 0, 0, 0)
        open_symbols = {p.symbol for p in positions if p.qty != 0}

        for d in unreplayed:
            try:
                deferred_date = _deferred_at_to_date_et(d.deferred_at)
                if deferred_date is not None and deferred_date >= today:
                    skipped_today += 1
                    continue

                # PDT-SUNSET-2026-06-04: supersede-race guard (D-073).
                # If the source virtual_exit transitioned out of 'active'
                # (scanner submitted on a later same-session tick, or
                # the position was closed externally and the scanner
                # marked obsolete), this deferred row is stale —
                # submitting it would short an unrelated position or
                # double-sell. Mark replayed_at with a SUPERSEDED:*
                # note so the row exits the queue without a broker
                # round-trip.
                source_state = self._journal.get_virtual_exit_state(
                    d.virtual_exit_id
                )
                if source_state in ("submitted", "obsolete"):
                    if d.id is not None:
                        self._journal.mark_deferred_skipped(
                            d.id, reason=f"SUPERSEDED:{source_state}"
                        )
                    log.info(
                        "pdt.replay.skipped_superseded",
                        extra={
                            "event": "pdt.replay.skipped_superseded",
                            "deferred_id": d.id,
                            "virtual_exit_id": d.virtual_exit_id,
                            "source_state": source_state,
                        },
                    )
                    skipped_superseded += 1
                    continue

                if d.symbol not in open_symbols:
                    if d.id is not None:
                        self._journal.mark_deferred_skipped(
                            d.id, reason="position_closed_externally"
                        )
                    log.info(
                        "pdt.replay.skipped_no_position",
                        extra={
                            "event": "pdt.replay.skipped_no_position",
                            "deferred_id": d.id,
                            "symbol": d.symbol,
                        },
                    )
                    skipped_no_position += 1
                    continue

                req = _to_replay_request(d)
                try:
                    resp = self._broker.submit_order(req)
                except PDTBudgetExceeded as e:
                    # Pre-market shouldn't 403; defense-in-depth — leave
                    # the row unreplayed so next session retries.
                    log.warning(
                        "pdt.replay.skipped",
                        extra={
                            "event": "pdt.replay.skipped",
                            "deferred_id": d.id,
                            "reason": "pdt_budget_exceeded",
                            "error": str(e),
                        },
                    )
                    skipped_error += 1
                    continue
                except BrokerUnavailable as e:
                    log.warning(
                        "pdt.replay.skipped",
                        extra={
                            "event": "pdt.replay.skipped",
                            "deferred_id": d.id,
                            "reason": "broker_unavailable",
                            "error": str(e),
                        },
                    )
                    skipped_error += 1
                    continue
                except APIError as e:
                    # Handle duplicate client_order_id from a prior
                    # boot's mid-replay crash (AC-6). Reconcile with the
                    # broker's existing order.
                    if _is_duplicate_client_order_id(e):
                        existing = self._broker.get_order_by_client_id(
                            req.client_order_id
                        )
                        if existing is not None and d.id is not None:
                            # HOTFIX-2026-05-08: PDT sells must be journaled
                            # for reconciler tolerance to explain the
                            # broker-position decrease. Use the existing
                            # (prior-boot) broker_order_id since this branch
                            # reconciles with an order already at the broker.
                            self._journal.insert_order(
                                _build_sell_order_row(
                                    execution_id=d.execution_id,
                                    role=d.role,
                                    req=req,
                                    resp=existing,
                                )
                            )
                            self._journal.mark_deferred_replayed(
                                d.id, broker_order_id=existing.broker_order_id
                            )
                            # PDT-SUNSET-2026-06-04: FINDING-3 (round-3
                            # review carry-forward closure) — transition
                            # the source virtual_exit to 'submitted' so
                            # the next scanner tick filters it out via
                            # list_active_virtual_exits().
                            self._journal.mark_virtual_exit_submitted(
                                d.virtual_exit_id,
                                broker_order_id=existing.broker_order_id,
                            )
                            replayed += 1
                            log.info(
                                "pdt.replay.duplicate_reconciled",
                                extra={
                                    "event": "pdt.replay.duplicate_reconciled",
                                    "deferred_id": d.id,
                                    "broker_order_id": existing.broker_order_id,
                                },
                            )
                            continue
                    # Unexpected APIError — log and skip.
                    log.warning(
                        "pdt.replay.skipped",
                        extra={
                            "event": "pdt.replay.skipped",
                            "deferred_id": d.id,
                            "reason": "api_error",
                            "error": str(e),
                        },
                    )
                    skipped_error += 1
                    continue

                if d.id is not None:
                    # HOTFIX-2026-05-08: PDT sells must be journaled for
                    # reconciler tolerance to explain the broker-position
                    # decrease.
                    self._journal.insert_order(
                        _build_sell_order_row(
                            execution_id=d.execution_id,
                            role=d.role,
                            req=req,
                            resp=resp,
                        )
                    )
                    self._journal.mark_deferred_replayed(
                        d.id, broker_order_id=resp.broker_order_id
                    )
                    # PDT-SUNSET-2026-06-04: FINDING-3 (round-3 review
                    # carry-forward closure) — transition the source
                    # virtual_exit to 'submitted' so the next scanner
                    # tick filters it out via list_active_virtual_exits().
                    self._journal.mark_virtual_exit_submitted(
                        d.virtual_exit_id,
                        broker_order_id=resp.broker_order_id,
                    )
                replayed += 1
            except Exception as e:  # noqa: BLE001 — per-row isolation (AC-4).
                log.error(
                    "pdt.replay.row_error",
                    extra={
                        "event": "pdt.replay.row_error",
                        "deferred_id": d.id,
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )
                skipped_error += 1

        report = ReplayReport(
            replayed=replayed,
            skipped_no_position=skipped_no_position,
            skipped_error=skipped_error,
            skipped_today=skipped_today,
            skipped_superseded=skipped_superseded,
        )
        log.info(
            "pdt.replay.run",
            extra={
                "event": "pdt.replay.run",
                "replayed_count": report.replayed,
                "skipped_no_position": report.skipped_no_position,
                "skipped_error": report.skipped_error,
                "skipped_today": report.skipped_today,
                "skipped_superseded": report.skipped_superseded,
            },
        )
        return report


__all__ = [
    "DeferredSellReplayer",
    "ReplayReport",
    "ScannerReport",
    "VirtualExitScanner",
    "compute_ev",
]
