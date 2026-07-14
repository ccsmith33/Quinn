"""Event-triggered thesis reviews — the trigger framework.

Runs once per reconciler tick (the same slot the thesis coordinator
consumes from), evaluates HELD positions, and schedules IMMEDIATE
thesis reviews through the existing Feature A seam: an "event trigger"
is nothing more than a `thesis_review_schedule` row with `due_at = now`
and a `scheduled_reason` of the form `event:<type>[:<detail>]`. The
reconciler's thesis hook runs right after this one, so an event row is
consumed on the same tick that scheduled it.

Because `find_due_thesis_reviews` keys on MAX(id) per execution, the
event row supersedes the pending future-dated calendar row; after the
event review resolves, the coordinator's normal reschedule (+7d on
hold, etc.) restores the calendar cadence. Nothing about the calendar
system changes.

Triggers (Milestone 1 + 3):
  - GAIN-CROSS  (`event:gain_cross:<pct>`): the position's high-water
    gain vs entry crossed a threshold. Thresholds default to the
    configured `trail_stages` gain_pcts plus the +10% arming line;
    each threshold fires at most once per position lifetime (the
    crossing IS the event).
  - HELD-NAME FILING (`event:filing:<accession>`): a newly ingested
    filing's issuer matches a held symbol. "Newly ingested" is tracked
    with an in-memory filings-table id watermark, primed on the first
    tick so historic filings never fire at boot.
  - TAPE ANOMALY (`event:anomaly`): the current quote moved >=
    `anomaly_move_pct` vs the previous close (absolute, price-only).
  - NEWS (`event:news:<id>:<headline>`): a new Alpaca news headline for
    a held symbol (only active when a news client is wired). The
    last-seen article id per symbol is cached in memory and primed on
    the first successful poll so stale articles never fire at boot; the
    headline rides in the reason string so the review context can cite
    it.
  - DAILY SWEEP (`event:daily_sweep`, opt-in via `daily_sweep`): once
    per UTC day, on the first tick at/after `sweep_after_utc_hour`
    (default 11 ~= 7am ET pre-market), every held position gets a
    review row so each one faces the doctrine's "would we buy this
    fresh?" test every morning. Idempotency is ledger-derived: a
    position is skipped when ANY schedule row for it is already dated
    today, so a mid-morning restart never double-sweeps.

Anti-churn (all triggers):
  - per-position-per-trigger-type cooldown (`cooldown_hours`, default
    24h ~= once per trading day), measured against the `scheduled_at`
    of the last `event:<type>...` row for the execution;
  - never schedule while a review for the execution is already pending
    AND due (latest schedule row un-fired with `due_at <= now`);
  - at most one event row per execution per tick.

Everything here is unreachable when `[event_reviews].enabled = false`:
composition does not construct the engine, and the engine itself
short-circuits `run_tick` as defense in depth.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from broker.protocol import BrokerAdapter
from config.loader import TrailStage
from journal.exit_policy import get_exit_policy_state
from journal.models import ThesisReviewScheduleRow
from journal.repo import (
    connect,
    get_latest_thesis_schedule_for_execution,
    get_orders_for_execution,
    insert_thesis_review_schedule,
)
from observability.log_port import get_logger

log = get_logger(__name__)

# The +10% arming line — always part of the default gain-cross
# threshold list alongside the trail_stages milestones.
_ARMING_GAIN_PCT = 10.0

# Truncation bound for headline text embedded in `event:news:...`
# reasons — the schedule ledger carries the evidence, but not a novel.
_NEWS_HEADLINE_REASON_CAP = 200


class _JournalLike(Protocol):
    db_path: str


class NewsClientPort(Protocol):
    """Minimal news surface the engine consumes. The real implementation
    is `broker.news.AlpacaNewsClient`; tests inject a fake. Must return
    articles sorted newest-first."""

    def latest_news(self, symbols: Sequence[str]) -> list[Any]: ...


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _ensure_aware(d: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes; treat them as UTC (same convention
    as the thesis coordinator / exit-policy ticker)."""
    if d.tzinfo is None:
        return d.replace(tzinfo=dt.UTC)
    return d


def _fmt_pct(value: float) -> str:
    """Stable threshold rendering for reason strings (10.0 -> '10')."""
    return f"{value:g}"


class EventTriggerEngine:
    """Per-tick evaluator that converts position events into immediate
    thesis-review schedule rows. Deterministic, broker-read-only: the
    ONLY write it ever performs is `insert_thesis_review_schedule`.
    """

    def __init__(
        self,
        *,
        journal: _JournalLike,
        broker: BrokerAdapter,
        enabled: bool,
        anomaly_move_pct: float = 7.0,
        cooldown_hours: float = 24.0,
        gain_thresholds_pct: Sequence[float] = (),
        trail_stages: Sequence[TrailStage] = (),
        prev_close_lookup: Callable[[str], float | None] | None = None,
        news_client: NewsClientPort | None = None,
        daily_sweep: bool = False,
        sweep_after_utc_hour: int = 11,
        now_fn: Callable[[], dt.datetime] = _utcnow,
    ) -> None:
        self._journal = journal
        self._broker = broker
        self._enabled = enabled
        self._anomaly_move_pct = anomaly_move_pct
        self._cooldown = dt.timedelta(hours=cooldown_hours)
        # Gain-cross thresholds: explicit config override wins; otherwise
        # the trail_stages milestones plus the +10% arming line.
        if gain_thresholds_pct:
            thresholds = list(gain_thresholds_pct)
        else:
            thresholds = sorted(
                {_ARMING_GAIN_PCT, *(s.gain_pct for s in trail_stages)}
            )
        self._gain_thresholds = tuple(sorted(thresholds))
        self._prev_close_lookup = prev_close_lookup
        self._news_client = news_client
        self._daily_sweep = daily_sweep
        self._sweep_after_utc_hour = sweep_after_utc_hour
        self._now_fn = now_fn
        # Date of the last completed sweep pass — a tick-loop
        # short-circuit only. The durable once-per-day guarantee is
        # ledger-derived (see `_check_daily_sweep`), so losing this on
        # restart is safe.
        self._sweep_done_for: dt.date | None = None
        # In-memory watermarks. A restart re-primes without firing, so a
        # process bounce can never replay historic filings/news as fresh
        # events (the calendar reviews backstop anything missed).
        self._filings_watermark: int | None = None
        # Per-symbol highest news article id already seen. A symbol's
        # first sighting primes its cursor from the current page without
        # firing (boot / just-entered-position safety).
        self._news_seen_max: dict[str, int] = {}

    def run_tick(self) -> None:
        """One trigger-evaluation pass. Errors on a single position /
        trigger are logged and never stop the rest of the tick (same
        protective shape as the sibling reconciler hooks)."""
        if not self._enabled:
            return
        now = self._now_fn()
        try:
            held = self._held_executions()
        except Exception as e:  # noqa: BLE001 — broker glitch: skip tick
            log.warning(
                "event_triggers.positions_unavailable",
                extra={
                    "event": "event_triggers.positions_unavailable",
                    "error": str(e),
                },
            )
            return
        # Executions that already received an event row this tick (or
        # arrive already pending/due) — one review per position per tick.
        scheduled: set[int] = set()

        # Filing + news triggers scan their own feeds and fan out to the
        # held map; price triggers walk the held positions directly.
        try:
            self._check_filings(held, now=now, scheduled=scheduled)
        except Exception as e:  # noqa: BLE001
            log.error(
                "event_triggers.filing_check_error",
                extra={
                    "event": "event_triggers.filing_check_error",
                    "error": str(e),
                    "error_class": type(e).__name__,
                },
            )
        if self._news_client is not None:
            try:
                self._check_news(held, now=now, scheduled=scheduled)
            except Exception as e:  # noqa: BLE001
                log.error(
                    "event_triggers.news_check_error",
                    extra={
                        "event": "event_triggers.news_check_error",
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )
        for symbol, execution_id in held.items():
            try:
                self._check_price_triggers(
                    execution_id=execution_id,
                    symbol=symbol,
                    now=now,
                    scheduled=scheduled,
                )
            except Exception as e:  # noqa: BLE001
                log.error(
                    "event_triggers.price_check_error",
                    extra={
                        "event": "event_triggers.price_check_error",
                        "execution_id": execution_id,
                        "symbol": symbol,
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )
        # Sweep runs LAST so real events keep their descriptive reasons;
        # the sweep only picks up positions nothing else touched today.
        if self._daily_sweep:
            try:
                self._check_daily_sweep(held, now=now, scheduled=scheduled)
            except Exception as e:  # noqa: BLE001
                log.error(
                    "event_triggers.daily_sweep_error",
                    extra={
                        "event": "event_triggers.daily_sweep_error",
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )

    # ------------------------------------------------------------------
    # Trigger evaluation
    # ------------------------------------------------------------------

    def _check_price_triggers(
        self,
        *,
        execution_id: int,
        symbol: str,
        now: dt.datetime,
        scheduled: set[int],
    ) -> None:
        """Gain-cross + tape anomaly — one quote fetch per symbol."""
        if execution_id in scheduled:
            return
        try:
            quote = self._broker.get_quote(symbol)
        except Exception as e:  # noqa: BLE001 — quote glitch: skip symbol
            log.warning(
                "event_triggers.quote_failed",
                extra={
                    "event": "event_triggers.quote_failed",
                    "symbol": symbol,
                    "error": str(e),
                },
            )
            return
        last = float(quote.last)
        if last <= 0:
            return

        # --- TAPE ANOMALY: |last vs prev close| >= anomaly_move_pct ---
        prev_close = (
            self._prev_close_lookup(symbol)
            if self._prev_close_lookup is not None
            else None
        )
        if prev_close is not None and prev_close > 0:
            move_pct = abs(last / prev_close - 1.0) * 100.0
            if move_pct >= self._anomaly_move_pct and self._try_schedule(
                execution_id=execution_id,
                symbol=symbol,
                trigger_type="anomaly",
                reason="event:anomaly",
                now=now,
                scheduled=scheduled,
                detail={
                    "move_pct": round(move_pct, 2),
                    "last": last,
                    "prev_close": prev_close,
                },
            ):
                return  # one event per execution per tick

        # --- GAIN-CROSS: HWM gain vs entry crossed a threshold --------
        entry_price = self._entry_price(execution_id)
        if entry_price is None:
            return
        state = get_exit_policy_state(
            self._journal.db_path, execution_id=execution_id
        )
        hwm = max(last, state.high_water_mark) if state is not None else last
        hwm_gain_pct = (hwm / entry_price - 1.0) * 100.0
        crossed = [t for t in self._gain_thresholds if hwm_gain_pct >= t]
        if not crossed:
            return
        # Fire the HIGHEST newly crossed threshold; each threshold fires
        # at most once per position lifetime (the ledger remembers).
        for threshold in sorted(crossed, reverse=True):
            reason = f"event:gain_cross:{_fmt_pct(threshold)}"
            if self._reason_already_fired(execution_id, reason):
                # Every threshold at/below an already-fired one is stale
                # news — the list is ascending and HWM never falls.
                return
            if self._try_schedule(
                execution_id=execution_id,
                symbol=symbol,
                trigger_type="gain_cross",
                reason=reason,
                now=now,
                scheduled=scheduled,
                detail={
                    "threshold_pct": threshold,
                    "hwm_gain_pct": round(hwm_gain_pct, 2),
                },
            ):
                return

    def _check_filings(
        self,
        held: dict[str, int],
        *,
        now: dt.datetime,
        scheduled: set[int],
    ) -> None:
        """HELD-NAME FILING — scan filings ingested since the watermark.
        First tick primes the watermark without firing (boot safety)."""
        with connect(self._journal.db_path) as conn:
            if self._filings_watermark is None:
                row = conn.execute("SELECT MAX(id) AS max_id FROM filings").fetchone()
                self._filings_watermark = int(row["max_id"] or 0)
                return
            rows = conn.execute(
                "SELECT id, accession_number, form_type, issuer_ticker "
                "FROM filings WHERE id > ? ORDER BY id ASC",
                (self._filings_watermark,),
            ).fetchall()
        for r in rows:
            self._filings_watermark = max(self._filings_watermark, int(r["id"]))
            ticker = r["issuer_ticker"]
            if ticker is None or ticker not in held:
                continue
            execution_id = held[ticker]
            accession = str(r["accession_number"])
            reason = f"event:filing:{accession}"
            if self._reason_already_fired(execution_id, reason):
                continue
            self._try_schedule(
                execution_id=execution_id,
                symbol=ticker,
                trigger_type="filing",
                reason=reason,
                now=now,
                scheduled=scheduled,
                detail={"accession": accession, "form_type": r["form_type"]},
            )

    def _check_news(
        self,
        held: dict[str, int],
        *,
        now: dt.datetime,
        scheduled: set[int],
    ) -> None:
        """NEWS — one batched poll of the newest headlines for all held
        symbols (a single API request per tick; well inside Alpaca's
        data rate limits).

        Per-symbol cursor semantics mirror the filings watermark: a
        symbol's FIRST sighting primes its cursor from the current page
        and never fires (so boot, restart, and a just-opened position
        cannot replay stale articles); afterwards an article fires only
        when its id is above the cursor, and the cursor advances even
        when cooldown suppresses the row — a suppressed headline does
        not come back later."""
        if not held:
            return
        articles = self._news_client.latest_news(sorted(held))  # type: ignore[union-attr]
        parsed: list[tuple[int, Any]] = []
        for article in articles:
            try:
                aid = int(str(article.article_id))
            except (TypeError, ValueError):
                continue  # non-numeric id: can't order it — skip
            parsed.append((aid, article))

        # Prime cursors for newly seen held symbols (no firing).
        fresh = {s for s in held if s not in self._news_seen_max}
        for symbol in fresh:
            self._news_seen_max[symbol] = max(
                (
                    aid
                    for aid, article in parsed
                    if symbol in (getattr(article, "symbols", ()) or ())
                ),
                default=0,
            )

        # Oldest-to-newest so multi-article gaps between polls resolve in
        # arrival order and the cursor lands on the newest id.
        for aid, article in sorted(parsed, key=lambda t: t[0]):
            headline = (getattr(article, "headline", "") or "")[
                :_NEWS_HEADLINE_REASON_CAP
            ]
            for symbol in getattr(article, "symbols", ()) or ():
                if symbol not in held or symbol in fresh:
                    continue
                if aid <= self._news_seen_max[symbol]:
                    continue
                self._news_seen_max[symbol] = aid
                self._try_schedule(
                    execution_id=held[symbol],
                    symbol=symbol,
                    trigger_type="news",
                    reason=f"event:news:{aid}:{headline}",
                    now=now,
                    scheduled=scheduled,
                    detail={"article_id": aid, "headline": headline},
                )

    def _check_daily_sweep(
        self,
        held: dict[str, int],
        *,
        now: dt.datetime,
        scheduled: set[int],
    ) -> None:
        """DAILY PRE-MARKET SWEEP — once per UTC day, on the first tick
        at/after `sweep_after_utc_hour`, put every held position in front
        of the reviewer (`event:daily_sweep`).

        Once-per-day-per-position is ledger-derived, not clock-derived: a
        position whose schedule ledger already carries ANY row dated
        today (an earlier sweep before a restart, or a real event review
        this morning) is skipped, so a mid-morning bounce cannot
        double-sweep. `_sweep_done_for` merely short-circuits the ticks
        after a completed pass; an errored pass leaves it unset so the
        next tick retries (already-swept positions still skip via the
        ledger). The sweep bypasses the per-trigger-type cooldown — the
        date key IS its anti-churn, and a rolling `cooldown_hours` window
        could swallow a next-day sweep whose tick lands minutes earlier
        than yesterday's."""
        today = now.date()
        if now.hour < self._sweep_after_utc_hour:
            return
        if self._sweep_done_for == today:
            return
        for symbol, execution_id in sorted(held.items()):
            if self._has_schedule_row_today(execution_id, now=now):
                continue
            self._try_schedule(
                execution_id=execution_id,
                symbol=symbol,
                trigger_type="daily_sweep",
                reason="event:daily_sweep",
                now=now,
                scheduled=scheduled,
                check_cooldown=False,
                detail={"sweep_date": today.isoformat()},
            )
        self._sweep_done_for = today

    # ------------------------------------------------------------------
    # Scheduling + anti-churn guards
    # ------------------------------------------------------------------

    def _try_schedule(
        self,
        *,
        execution_id: int,
        symbol: str,
        trigger_type: str,
        reason: str,
        now: dt.datetime,
        scheduled: set[int],
        detail: dict[str, Any],
        check_cooldown: bool = True,
    ) -> bool:
        """Apply the shared anti-churn guards and, if clear, insert the
        due-now schedule row. Returns True when a row was written."""
        if execution_id in scheduled:
            return False
        if self._review_pending_and_due(execution_id, now=now):
            log.debug(
                "event_triggers.skipped_review_already_due",
                extra={
                    "event": "event_triggers.skipped_review_already_due",
                    "execution_id": execution_id,
                    "trigger": trigger_type,
                },
            )
            return False
        if check_cooldown and self._in_cooldown(
            execution_id, trigger_type, now=now
        ):
            log.info(
                "event_triggers.skipped_cooldown",
                extra={
                    "event": "event_triggers.skipped_cooldown",
                    "execution_id": execution_id,
                    "symbol": symbol,
                    "trigger": trigger_type,
                },
            )
            return False
        insert_thesis_review_schedule(
            self._journal.db_path,
            ThesisReviewScheduleRow(
                execution_id=execution_id,
                due_at=now,
                scheduled_reason=reason,
            ),
        )
        scheduled.add(execution_id)
        log.info(
            "event_triggers.review_scheduled",
            extra={
                "event": "event_triggers.review_scheduled",
                "execution_id": execution_id,
                "symbol": symbol,
                "trigger": trigger_type,
                "reason": reason,
                **detail,
            },
        )
        return True

    def _review_pending_and_due(
        self, execution_id: int, *, now: dt.datetime
    ) -> bool:
        """True when the execution's LATEST schedule row is un-fired AND
        already due — i.e. the coordinator will review it this tick
        anyway, so an event row would be a duplicate. A future-dated
        pending row does NOT block: pulling that review forward is the
        whole point of an event trigger."""
        latest = get_latest_thesis_schedule_for_execution(
            self._journal.db_path, execution_id
        )
        if latest is None or latest.id is None:
            return False
        if _ensure_aware(latest.due_at) > now:
            return False
        with connect(self._journal.db_path) as conn:
            fired = conn.execute(
                "SELECT 1 FROM thesis_reviews WHERE schedule_id = ? LIMIT 1",
                (latest.id,),
            ).fetchone()
        return fired is None

    def _in_cooldown(
        self, execution_id: int, trigger_type: str, *, now: dt.datetime
    ) -> bool:
        """Per-position-per-trigger-type cooldown, measured from the
        `due_at` of the most recent `event:<type>...` row. Event rows
        are written with `due_at = now`, so `due_at` IS the trigger's
        fire time — and unlike the DB-defaulted `scheduled_at` it comes
        from the engine's own clock (single time source)."""
        with connect(self._journal.db_path) as conn:
            row = conn.execute(
                "SELECT due_at FROM thesis_review_schedule "
                "WHERE execution_id = ? AND scheduled_reason LIKE ? "
                "ORDER BY id DESC LIMIT 1",
                (execution_id, f"event:{trigger_type}%"),
            ).fetchone()
        if row is None or row["due_at"] is None:
            return False
        last_at = row["due_at"]
        if isinstance(last_at, str):  # defensive: driver-dependent shape
            last_at = dt.datetime.fromisoformat(last_at)
        return now - _ensure_aware(last_at) < self._cooldown

    def _has_schedule_row_today(
        self, execution_id: int, *, now: dt.datetime
    ) -> bool:
        """True when ANY schedule row for the execution has a `due_at`
        dated today (UTC) — the sweep's restart-proof once-per-day key.
        Calendar rows are future-dated at insert and event rows carry
        `due_at = now`, so 'row dated today' == 'review already put in
        front of the coordinator today'."""
        day_start = dt.datetime.combine(now.date(), dt.time.min, tzinfo=dt.UTC)
        with connect(self._journal.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM thesis_review_schedule "
                "WHERE execution_id = ? AND due_at >= ? AND due_at < ? "
                "LIMIT 1",
                (execution_id, day_start, day_start + dt.timedelta(days=1)),
            ).fetchone()
        return row is not None

    def _reason_already_fired(self, execution_id: int, reason: str) -> bool:
        with connect(self._journal.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM thesis_review_schedule "
                "WHERE execution_id = ? AND scheduled_reason = ? LIMIT 1",
                (execution_id, reason),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Journal lookups
    # ------------------------------------------------------------------

    def _held_executions(self) -> dict[str, int]:
        """symbol -> latest accepted execution id, for every position the
        BROKER currently holds (broker truth for 'held', journal for the
        execution mapping — same split as the boot re-arm sweep)."""
        out: dict[str, int] = {}
        symbols = [
            p.symbol for p in self._broker.get_positions() if p.qty > 0
        ]
        if not symbols:
            return out
        with connect(self._journal.db_path) as conn:
            for symbol in symbols:
                row = conn.execute(
                    "SELECT e.id AS execution_id FROM executions e "
                    "JOIN proposals p ON p.id = e.proposal_id "
                    "WHERE e.decision = 'accepted' AND p.symbol = ? "
                    "ORDER BY e.id DESC LIMIT 1",
                    (symbol,),
                ).fetchone()
                if row is not None:
                    out[symbol] = int(row["execution_id"])
        return out

    def _entry_price(self, execution_id: int) -> float | None:
        """Entry fill (or pre-submission last) — same derivation as the
        exit-policy ticker's gain math."""
        orders = get_orders_for_execution(self._journal.db_path, execution_id)
        entry = next((o for o in orders if o.role == "entry"), None)
        if entry is None:
            return None
        price = entry.realized_fill_price or entry.pre_submission_last
        if price is None or price <= 0:
            return None
        return float(price)


__all__ = ["EventTriggerEngine", "NewsClientPort"]
