"""S8.2 — Daily reporter (NFR-10, FR-34, architecture §10.1, §2.11).

Composes a roll-up of the trading session and dispatches it via Telegram
(and optionally email). Scheduled via systemd timer at 16:30 ET on every
trading day (S8.3 ships the unit + timer).

Design:

- `Report` is the structured payload — a frozen dataclass with the seven
  fields AC-2 enumerates.
- `DailyReporter.compose(now)` runs all journal queries and returns the
  Report.
- `DailyReporter.compose_and_send(now)` composes, formats as text, and
  dispatches.
- `Notifier` is a tiny `send(text) -> None` Protocol. The Telegram
  process (S7.2) wires its real Bot here; tests inject a recording
  double. Email is optional — wired only when `OPERATOR_NOTIFY_EMAIL`
  is set per S8.1's secrets contract.
- `clear_auto_ks1_at_session_start(now, journal)` is called once at
  09:30 ET on each trading day and lifts any active `auto:KS-1` halt
  (per S7.4 AC-2 + S8.2 dev-note). Manual halts (`manual:*`) and
  KS-2/KS-3 (`auto:KS-2`, `auto:KS-3`) are NEVER auto-cleared.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Protocol

from config.calendar import ET, is_market_open_day
from journal.models import KillSwitchStateRow
from journal.repo import JournalRepo, connect
from observability.log_port import get_logger

log = get_logger(__name__)


# Window in which the next-session-open clear-fn fires. The systemd timer
# is set to 09:30 ET; we accept a small tolerance so a ±5min jitter does
# not skip a day.
_SESSION_OPEN_TOLERANCE_S: int = 300


class Notifier(Protocol):
    """Send a text payload to a notification channel (Telegram, email)."""

    def send(self, text: str) -> None: ...


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    session_date: _dt.date
    filings_ingested: int
    prefilter_counts: dict[tuple[str, str], int]  # (decision, rule_fired) -> count
    proposals_by_kind: dict[str, int]
    proposals_executed: int  # executions.decision = 'accepted'
    fills: int  # orders.final_status in ('filled', 'partially_filled')
    day_end_equity: float | None
    mtd_inference_cost_usd: float

    def format_text(self) -> str:
        """Operator-facing summary, one section per metric."""
        prefilter_lines = (
            "\n".join(
                f"  {decision} ({rule}): {count}"
                for (decision, rule), count in sorted(self.prefilter_counts.items())
            )
            or "  none"
        )
        kind_lines = (
            "\n".join(
                f"  {kind}: {count}" for kind, count in sorted(self.proposals_by_kind.items())
            )
            or "  none"
        )
        equity_str = f"${self.day_end_equity:,.2f}" if self.day_end_equity is not None else "—"
        return "\n".join(
            [
                f"Daily report — {self.session_date.isoformat()}",
                f"Filings ingested: {self.filings_ingested}",
                "Prefilter:",
                prefilter_lines,
                "Proposals:",
                kind_lines,
                f"Proposals executed: {self.proposals_executed}",
                f"Fills (filled + partial): {self.fills}",
                f"Day-end equity: {equity_str}",
                f"MTD inference cost: ${self.mtd_inference_cost_usd:.2f}",
            ]
        )


# ---------------------------------------------------------------------------
# DailyReporter
# ---------------------------------------------------------------------------


class DailyReporter:
    def __init__(
        self,
        *,
        journal: JournalRepo,
        telegram: Notifier,
        email: Notifier | None = None,
    ) -> None:
        self._journal = journal
        self._telegram = telegram
        self._email = email

    def compose(self, *, now: _dt.datetime) -> Report:
        """Aggregate all journal data sources into a Report."""
        session_start, session_end = _session_bounds_utc(now)

        with connect(self._journal.db_path) as conn:
            filings_ingested = _scalar_int(
                conn,
                "SELECT COUNT(*) FROM filings WHERE fetched_at >= ? AND fetched_at < ?",
                (session_start, session_end),
            )
            prefilter_counts = _prefilter_counts(conn, session_start, session_end)
            proposals_by_kind = _proposals_by_kind(conn, session_start, session_end)
            proposals_executed = _scalar_int(
                conn,
                "SELECT COUNT(*) FROM executions e "
                "JOIN proposals p ON p.id = e.proposal_id "
                "JOIN filings f ON f.id = p.filing_id "
                "WHERE e.decision = 'accepted' "
                "AND f.fetched_at >= ? AND f.fetched_at < ?",
                (session_start, session_end),
            )
            fills = _scalar_int(
                conn,
                "SELECT COUNT(*) FROM orders o "
                "JOIN executions e ON e.id = o.execution_id "
                "JOIN proposals p ON p.id = e.proposal_id "
                "JOIN filings f ON f.id = p.filing_id "
                "WHERE o.final_status IN ('filled', 'partially_filled') "
                "AND f.fetched_at >= ? AND f.fetched_at < ?",
                (session_start, session_end),
            )

        snap = self._journal.get_latest_account_snapshot()
        day_end_equity = float(snap.equity) if snap is not None else None
        # NFR-8 / S5.4 D-1 + S5.5 D-1 carry-forward: spend reads `llm_calls`,
        # NEVER `proposal_reviews`. `llm_calls` is the canonical source.
        mtd_cost = self._journal.get_mtd_inference_cost_usd(now)

        return Report(
            session_date=_session_date_et(now),
            filings_ingested=filings_ingested,
            prefilter_counts=prefilter_counts,
            proposals_by_kind=proposals_by_kind,
            proposals_executed=proposals_executed,
            fills=fills,
            day_end_equity=day_end_equity,
            mtd_inference_cost_usd=mtd_cost,
        )

    def compose_and_send(self, *, now: _dt.datetime) -> Report:
        report = self.compose(now=now)
        text = report.format_text()
        self._telegram.send(text)
        if self._email is not None:
            self._email.send(text)
        log.info(
            "daily_report.sent",
            extra={
                "event": "daily_report.sent",
                "session_date": report.session_date.isoformat(),
                "filings_ingested": report.filings_ingested,
                "proposals_executed": report.proposals_executed,
                "fills": report.fills,
                "mtd_cost_usd": round(report.mtd_inference_cost_usd, 2),
                "email_sent": self._email is not None,
            },
        )
        return report


# ---------------------------------------------------------------------------
# KS-1 reset at next session start (per S7.4 AC-2 + S8.2 dev-note)
# ---------------------------------------------------------------------------


def clear_auto_ks1_at_session_start(*, now: _dt.datetime, journal: JournalRepo) -> bool:
    """If `now` is within the session-start window AND the latest kill-switch
    row is `auto:KS-1`, insert a `resume` row.

    Returns True iff a clear was inserted.

    Manual halts (`manual:telegram`, `manual:webhook`) and KS-2 / KS-3
    (`auto:KS-2`, `auto:KS-3`) are NEVER cleared by this function — those
    require operator intervention.
    """
    if not _is_session_open_window(now):
        return False
    state = journal.get_latest_kill_switch_state()
    if state is None:
        return False
    if state.state != "halted":
        return False
    if state.reason != "auto:KS-1":
        return False
    journal.insert_kill_switch_state(
        KillSwitchStateRow(
            set_at=now,
            state="active",
            reason="resume",
            set_by="system",
            notes="auto-cleared at next session open (KS-1)",
        )
    )
    log.info(
        "killswitch.auto_clear_ks1",
        extra={"event": "killswitch.auto_clear_ks1", "set_at": now.isoformat()},
    )
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_date_et(now: _dt.datetime) -> _dt.date:
    return _to_et(now).date()


def _session_bounds_utc(now: _dt.datetime) -> tuple[_dt.datetime, _dt.datetime]:
    """Return (session_start, session_end) as UTC-naive matching journal storage.

    Journal timestamps are written via Python's stdlib datetime ISO format
    without `Z`, and queries compare against the same shape. We define a
    "session" as the calendar ET day containing `now` — start at 00:00 ET,
    end at 24:00 ET. Wide enough to capture pre-market filings and any
    after-hours Form 4s the discovery loop catches before 16:30 ET.
    """
    et_now = _to_et(now)
    et_start = _dt.datetime.combine(et_now.date(), _dt.time(0, 0), tzinfo=ET)
    et_end = et_start + _dt.timedelta(days=1)
    # Strip tz to match how the journal stores timestamps (naive UTC).
    return (
        et_start.astimezone(_dt.UTC).replace(tzinfo=None),
        et_end.astimezone(_dt.UTC).replace(tzinfo=None),
    )


def _to_et(when: _dt.datetime) -> _dt.datetime:
    if when.tzinfo is None:
        return when.replace(tzinfo=ET)
    return when.astimezone(ET)


def _is_session_open_window(now: _dt.datetime) -> bool:
    """True iff `now` is within ±_SESSION_OPEN_TOLERANCE_S of 09:30 ET on a
    market-open day."""
    et_now = _to_et(now)
    if not is_market_open_day(et_now.date()):
        return False
    target = _dt.datetime.combine(et_now.date(), _dt.time(9, 30), tzinfo=ET)
    delta = abs((et_now - target).total_seconds())
    return delta <= _SESSION_OPEN_TOLERANCE_S


def _scalar_int(conn, sql: str, params: tuple) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


def _prefilter_counts(conn, start: _dt.datetime, end: _dt.datetime) -> dict[tuple[str, str], int]:
    """Counts of prefilter decisions for filings ingested in the session.

    Joining via filings.fetched_at (rather than prefilter_decisions.decided_at)
    bounds the report to filings the agent ingested during this session,
    even if the prefilter ran on a backfilled batch outside the window.
    Equivalent under steady-state operation, more correct under reconciler
    catch-up runs.
    """
    rows = conn.execute(
        "SELECT pd.decision, pd.rule_fired, COUNT(*) AS c "
        "FROM prefilter_decisions pd "
        "JOIN filings f ON f.id = pd.filing_id "
        "WHERE f.fetched_at >= ? AND f.fetched_at < ? "
        "GROUP BY pd.decision, pd.rule_fired",
        (start, end),
    ).fetchall()
    return {(r["decision"], r["rule_fired"]): int(r["c"]) for r in rows}


def _proposals_by_kind(conn, start: _dt.datetime, end: _dt.datetime) -> dict[str, int]:
    """Counts of proposals authored on filings ingested in the session.

    Same rationale as `_prefilter_counts`: bound by `filings.fetched_at`
    rather than `proposals.created_at` so reconciler-driven backfill of an
    older filing doesn't pollute today's report.
    """
    rows = conn.execute(
        "SELECT p.kind, COUNT(*) AS c FROM proposals p "
        "JOIN filings f ON f.id = p.filing_id "
        "WHERE f.fetched_at >= ? AND f.fetched_at < ? GROUP BY p.kind",
        (start, end),
    ).fetchall()
    return {r["kind"]: int(r["c"]) for r in rows}
