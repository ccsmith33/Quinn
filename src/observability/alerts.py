"""S8.2 — Alert watcher (NFR-11, NFR-1, architecture §10.1).

Periodic poll (every 60s, scheduled by S5.6 / S8.3) that evaluates a fixed
set of alert conditions against the journal + ambient state and dispatches
Telegram alerts via a `Notifier`. Each alert kind is debounced to at most
once per hour (AC-5) so a sustained condition doesn't spam the operator.

Alert kinds (architecture §10.1):
- `kill_switch_flip` — new `kill_switch_state` row since last poll.
- `monthly_cost_overrun` — sum(`llm_calls.cost_usd`) for current calendar
  month ≥ $150 (NFR-11). Note: spend is read from `llm_calls`, not
  `proposal_reviews` (S5.4 D-1 + S5.5 D-1 carry-forward).
- `backup_failure` — latest `backups` row has `verify_result LIKE 'failed%'`,
  OR no backup row at all in the last 26h.
- `nfr1_latency_breach` — p95 of `proposals.latency_ms` over the last hour
  exceeds 90s (NFR-1).
- `log_sink_unavailable` — Vector reports an off-box-sink outage > 10 min.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from journal.repo import JournalRepo, connect
from observability.log_port import get_logger

log = get_logger(__name__)

MONTHLY_COST_THRESHOLD_USD: float = 150.0
NFR1_LATENCY_P95_THRESHOLD_MS: int = 90_000
NFR1_LATENCY_WINDOW_S: int = 3600
LOG_SINK_OUTAGE_THRESHOLD_S: int = 600
BACKUP_MAX_AGE_H: int = 26
DEBOUNCE_S: int = 3600

AlertKind = Literal[
    "kill_switch_flip",
    "monthly_cost_overrun",
    "backup_failure",
    "nfr1_latency_breach",
    "log_sink_unavailable",
]


class Notifier(Protocol):
    def send(self, text: str) -> None: ...


class TelegramNotifier:
    """Minimal outbound Telegram sender (S5.6 carry-fwd S6.5 reviewer M-3
    + AlertWatcher wiring). One-shot HTTPS POST to `sendMessage`; never
    blocks the agent loop on delivery — a 5s connect timeout is the
    upper bound. Failures log WARN and return; the structured log is
    the secondary channel per NFR-9 (never block trading on a sink).

    Used by both `Reconciler` (via `_TelegramAlerterAdapter` which maps
    `notify(message)` → `send(text)`) and `AlertWatcher` (direct
    `Notifier` Protocol consumer).
    """

    def __init__(self, *, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send(self, text: str) -> None:
        # Lazy import so unit tests don't require httpx for fakes.
        import httpx

        try:
            resp = httpx.post(
                self._url,
                data={"chat_id": self._chat_id, "text": text},
                timeout=5.0,
            )
            if resp.status_code >= 400:
                log.warning(
                    "telegram.send_failed",
                    extra={
                        "event": "telegram.send_failed",
                        "status": resp.status_code,
                    },
                )
        except Exception as e:  # noqa: BLE001 — log + drop, never block
            log.warning(
                "telegram.send_error",
                extra={
                    "event": "telegram.send_error",
                    "error_class": type(e).__name__,
                },
            )


@dataclass(frozen=True)
class LogSinkStatus:
    """Reported externally (Vector tail / health endpoint).

    `unavailable_since` is None when the sink is healthy; otherwise it is
    the start of the current outage.
    """

    unavailable_since: _dt.datetime | None


class AlertWatcher:
    def __init__(
        self,
        *,
        journal: JournalRepo,
        notifier: Notifier,
        log_sink_status_fn: Callable[[], LogSinkStatus] | None = None,
    ) -> None:
        self._journal = journal
        self._notifier = notifier
        self._log_sink_status_fn = log_sink_status_fn or (lambda: LogSinkStatus(None))
        # Debounce state: when each alert kind last fired (or None if never).
        self._last_fired: dict[AlertKind, _dt.datetime | None] = {
            "kill_switch_flip": None,
            "monthly_cost_overrun": None,
            "backup_failure": None,
            "nfr1_latency_breach": None,
            "log_sink_unavailable": None,
        }
        # Snapshot the latest kill-switch row at construction time so the
        # boot seed row from 001_init.sql doesn't fire a false alert. Any
        # row inserted AFTER the watcher comes up is a real flip.
        latest_ks = self._journal.get_latest_kill_switch_state()
        self._last_seen_ks_set_at: _dt.datetime | None = (
            _ensure_aware(latest_ks.set_at) if latest_ks is not None else None
        )

    def poll(self, *, now: _dt.datetime) -> list[AlertKind]:
        """Evaluate all alert conditions; dispatch any that fire and are not
        currently debounced. Returns the list of alert kinds that fired."""
        fired: list[AlertKind] = []

        if self._check_ks_flip(now):
            fired.append("kill_switch_flip")
        if self._check_cost_overrun(now):
            fired.append("monthly_cost_overrun")
        if self._check_backup_failure(now):
            fired.append("backup_failure")
        if self._check_latency_breach(now):
            fired.append("nfr1_latency_breach")
        if self._check_log_sink(now):
            fired.append("log_sink_unavailable")

        return fired

    # -- Individual checks ------------------------------------------------

    def _check_ks_flip(self, now: _dt.datetime) -> bool:
        latest = self._journal.get_latest_kill_switch_state()
        if latest is None:
            return False
        latest_at = _ensure_aware(latest.set_at)
        last_seen = self._last_seen_ks_set_at
        if last_seen is not None and latest_at <= last_seen:
            return False
        # Genuinely new row.
        self._last_seen_ks_set_at = latest_at
        return self._dispatch(
            kind="kill_switch_flip",
            now=now,
            text=(
                f"Kill-switch flipped: state={latest.state} reason={latest.reason} "
                f"by={latest.set_by} at {latest_at.isoformat(timespec='seconds')}"
            ),
        )

    def _check_cost_overrun(self, now: _dt.datetime) -> bool:
        # NFR-11: spend reads `llm_calls`. The JournalRepo helper sums
        # `cost_usd` since the start of the current calendar month.
        cost = self._journal.get_mtd_inference_cost_usd(now)
        if cost < MONTHLY_COST_THRESHOLD_USD:
            return False
        return self._dispatch(
            kind="monthly_cost_overrun",
            now=now,
            text=(
                f"Monthly inference cost ${cost:.2f} ≥ "
                f"${MONTHLY_COST_THRESHOLD_USD:.0f} threshold (NFR-11)"
            ),
        )

    def _check_backup_failure(self, now: _dt.datetime) -> bool:
        latest = _get_latest_backup_summary(self._journal, now)
        if latest is None:
            return False
        kind, detail = latest
        if not kind:
            return False
        return self._dispatch(
            kind="backup_failure",
            now=now,
            text=f"Backup alert: {detail}",
        )

    def _check_latency_breach(self, now: _dt.datetime) -> bool:
        p95 = _proposals_latency_p95_ms(
            self._journal,
            since=now - _dt.timedelta(seconds=NFR1_LATENCY_WINDOW_S),
            until=now,
        )
        if p95 is None or p95 <= NFR1_LATENCY_P95_THRESHOLD_MS:
            return False
        return self._dispatch(
            kind="nfr1_latency_breach",
            now=now,
            text=(
                f"NFR-1 latency p95 over last 1h = {p95 / 1000:.1f}s "
                f"(threshold {NFR1_LATENCY_P95_THRESHOLD_MS / 1000:.0f}s)"
            ),
        )

    def _check_log_sink(self, now: _dt.datetime) -> bool:
        status = self._log_sink_status_fn()
        if status.unavailable_since is None:
            return False
        outage_s = (now - status.unavailable_since).total_seconds()
        if outage_s < LOG_SINK_OUTAGE_THRESHOLD_S:
            return False
        return self._dispatch(
            kind="log_sink_unavailable",
            now=now,
            text=(
                f"Log sink unavailable for {outage_s / 60:.0f}m "
                f"(threshold {LOG_SINK_OUTAGE_THRESHOLD_S // 60}m, NFR-9)"
            ),
        )

    # -- Debounce + dispatch ----------------------------------------------

    def _dispatch(self, *, kind: AlertKind, now: _dt.datetime, text: str) -> bool:
        last = self._last_fired[kind]
        if last is not None and (now - last).total_seconds() < DEBOUNCE_S:
            return False
        self._notifier.send(text)
        self._last_fired[kind] = now
        log.warning(
            "alert.fired",
            extra={"event": "alert.fired", "kind": kind, "text": text},
        )
        return True


def _ensure_aware(when: _dt.datetime) -> _dt.datetime:
    """Treat naive datetimes as UTC. Journal storage strips tzinfo, so reads
    can return naive datetimes that need normalizing for comparison with
    `now` (always aware)."""
    if when.tzinfo is None:
        return when.replace(tzinfo=_dt.UTC)
    return when


# ---------------------------------------------------------------------------
# Journal helpers — kept module-level to keep `JournalRepo` minimal.
# ---------------------------------------------------------------------------


def _get_latest_backup_summary(journal: JournalRepo, now: _dt.datetime) -> tuple[str, str] | None:
    """Return (kind, human-readable detail) describing a backup problem, or
    None if everything is fine.

    Conditions:
      - latest row's `verify_result` starts with `failed` → ("verify_failed", ...)
      - no row at all in the last `BACKUP_MAX_AGE_H` hours → ("stale", ...)
    """
    with connect(journal.db_path) as conn:
        row = conn.execute(
            "SELECT started_at, verify_result FROM backups ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        # No backup row at all — fresh deploy (S8.3 backup unit not yet
        # provisioned, or the first daily run hasn't fired). Don't false-alert.
        # Once at least one backup row exists, the stale-after-26h check
        # below takes over.
        return None

    started_at = row["started_at"]
    if not isinstance(started_at, _dt.datetime):
        started_at = _dt.datetime.fromisoformat(str(started_at).replace(" ", "T"))
    started_at = _ensure_aware(started_at)
    now_a = _ensure_aware(now)
    age_s = (now_a - started_at).total_seconds()
    if age_s > BACKUP_MAX_AGE_H * 3600:
        return (
            "stale",
            f"latest backup is {age_s / 3600:.1f}h old (max {BACKUP_MAX_AGE_H}h)",
        )

    verify = (row["verify_result"] or "").strip().lower()
    if verify.startswith("failed"):
        return ("verify_failed", f"latest backup verify_result={row['verify_result']!r}")

    return None


def _proposals_latency_p95_ms(
    journal: JournalRepo, *, since: _dt.datetime, until: _dt.datetime
) -> int | None:
    """Compute p95 of `proposals.latency_ms` for rows with
    `created_at` in [since, until). Returns None if no rows.

    Uses nearest-rank p95 for sub-second determinism in tests; fine at the
    journal volumes we expect (≤ 100 proposals / hour).

    Datetime params are passed through unchanged: the journal's
    `_adapt_datetime` ISO-formats both stored values and bind params, so
    aware-vs-naive lookups must match shape on both sides. Stripping tz
    here would produce a string-compare mismatch against rows stored with
    aware datetimes.
    """
    with connect(journal.db_path) as conn:
        rows = conn.execute(
            "SELECT latency_ms FROM proposals "
            "WHERE created_at >= ? AND created_at < ? "
            "ORDER BY latency_ms ASC",
            (since, until),
        ).fetchall()
    if not rows:
        return None
    n = len(rows)
    # Nearest-rank: index = ceil(0.95 * n) - 1, clamped to [0, n-1].
    import math

    idx = max(0, min(n - 1, math.ceil(0.95 * n) - 1))
    return int(rows[idx]["latency_ms"])
