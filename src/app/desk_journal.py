"""Desk journal — per-trade post-mortems + weekly synthesis (LLM memory).

Two write paths and one read path, all gated behind
`config.memory.enabled AND config.memory.desk_journal_enabled`:

  1. POST-MORTEMS (daily): once per UTC day, on the reconciler tick (the
     `desk_journal_ticker` hook, after all trading hooks), find accepted
     executions that CLOSED (filled entry + filled sell-side exit —
     journal truth for flat-at-broker) and still lack a
     `trade_postmortems` row. For each (capped at `daily_cap`/day so a
     backlog drains gradually) make ONE Haiku call with a compact fact
     block and parse `thesis_summary` / `outcome_summary` / `lesson`.
     `exit_quality_pct` (realized gain vs peak gain) is computed HERE in
     code, never by the LLM. Malformed output → log + skip; the missing
     row makes the trade a candidate again tomorrow. Nothing here ever
     raises into the reconciler tick.

  2. SYNTHESIS (weekly): on the first tick of a new ISO week (ET), if at
     least 3 post-mortems landed since the last synthesis, one Haiku
     call summarizes the most recent ~30 post-mortems into <=15 lines of
     patterns. Stored as `desk_memory(kind='synthesis')` with
     version = prev + 1 and the prior row deactivated.

  3. PROVIDER (`desk_journal`): serves the active synthesis to every LLM
     purpose via the shared memory rail. Deterministic — same DB state,
     byte-identical section.

CHARTER — DESCRIPTIVE ONLY. Post-mortems record what happened; the
synthesis summarizes patterns (with sample sizes). NEITHER invents new
trading rules: the live trade count is far too small, and any rule
distilled from it would be overfitting. Both prompt files
(`desk_postmortem_v1.txt`, `desk_synthesis_v1.txt`) state this charter
explicitly and forbid prescriptive language.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from typing import Any, Protocol

from analyzer.anthropic_client import Purpose
from app.memory_context import MemoryQuery, MemorySection, Provider
from config.calendar import ET
from journal.models import DeskMemoryRow, TradePostmortemRow
from journal.repo import (
    count_postmortems_since,
    deactivate_desk_memory,
    find_closed_executions_without_postmortem,
    get_active_desk_memory,
    get_recent_postmortems,
    insert_desk_memory,
    insert_trade_postmortem,
)
from observability.log_port import get_logger
from prompts.loader import ApiRequest

log = get_logger(__name__)

# Cheap journaling model — post-mortems and syntheses are summarization,
# not trading judgment. Cost still lands in `llm_calls` via the shared
# AnthropicClient (purposes 'postmortem' / 'synthesis').
HAIKU_MODEL_ID = "claude-haiku-4-5-20251001"

_POSTMORTEM_MAX_TOKENS = 1024
_SYNTHESIS_MAX_TOKENS = 2048

# Weekly synthesis gating.
_SYNTHESIS_MIN_NEW_POSTMORTEMS = 3
_SYNTHESIS_WINDOW_ROWS = 30
_SYNTHESIS_MAX_LINES = 15

# Post-mortem field hygiene: each parsed field is collapsed to one line
# and bounded — the journal stores observations, not essays.
_FIELD_CHAR_CAP = 300

# The section header the provider renders — pinned as a constant so the
# provider output is byte-stable and tests can assert on it.
DESK_JOURNAL_SECTION_TITLE = "Desk journal — recent trade patterns (descriptive)"

# Exit-order role → the mechanism vocabulary the post-mortem records.
# Unknown roles pass through verbatim (descriptive fallback — never guess).
_EXIT_MECHANISM_BY_ROLE = {
    "stop": "stop",
    "trailing_stop": "trail",
    "take_profit": "take_profit",
    "thesis_close": "thesis_close",
    "displacement_close": "displacement",
}


class _JournalLike(Protocol):
    db_path: str


class _AnthropicClientPort(Protocol):
    async def call(
        self,
        request: ApiRequest,
        *,
        model_id: str,
        purpose: Purpose,
        decision_id: str,
        max_tokens: int | None = None,
    ) -> str: ...


class _PromptBuilderPort(Protocol):
    def build_desk_postmortem(self, *, trade_context: str) -> ApiRequest: ...
    def build_desk_synthesis(self, *, postmortems_digest: str) -> ApiRequest: ...


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _ensure_aware(d: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes; treat them as UTC (same convention
    as the thesis coordinator / event-trigger engine)."""
    if d.tzinfo is None:
        return d.replace(tzinfo=dt.UTC)
    return d


# ---------------------------------------------------------------------------
# Pure computation — code owns the numbers, the LLM only narrates.
# ---------------------------------------------------------------------------


def compute_exit_quality_pct(
    *,
    entry_price: float | None,
    exit_price: float | None,
    peak_price: float | None,
) -> float | None:
    """Share of the peak gain the exit captured, in percent.

    peak = max(high-water mark, exit, entry) — the HWM can lag a gap exit
    and is None when the trade never came under trailing management.
    Returns None when prices are missing or the trade never traded above
    entry (no favorable move existed, so "captured vs peak" is undefined
    — the loss story lives in outcome_summary instead). Can exceed 0..100
    downward: a trade that round-tripped a big peak gain into a loss
    scores negative, which is exactly the honest number.
    """
    if entry_price is None or exit_price is None or entry_price <= 0:
        return None
    peak = max(
        p for p in (peak_price, exit_price, entry_price) if p is not None
    )
    available = peak - entry_price
    if available <= 0:
        return None
    return round((exit_price - entry_price) / available * 100.0, 1)


def _exit_mechanism(exit_role: str | None) -> str:
    if not exit_role:
        return "unknown"
    return _EXIT_MECHANISM_BY_ROLE.get(exit_role, exit_role)


def _days_held(
    entry_at: dt.datetime | None, exit_at: dt.datetime | None
) -> int | None:
    if entry_at is None or exit_at is None:
        return None
    delta = _ensure_aware(exit_at) - _ensure_aware(entry_at)
    return max(delta.days, 0)


def _one_line(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:_FIELD_CHAR_CAP]


def _parse_postmortem(raw: str) -> tuple[str, str, str] | None:
    """Parse the Haiku response into (thesis_summary, outcome_summary,
    lesson) — all single-line, char-capped — or None when malformed."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    fields: list[str] = []
    for key in ("thesis_summary", "outcome_summary", "lesson"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
        fields.append(_one_line(value))
    return (fields[0], fields[1], fields[2])


def _clean_synthesis(raw: str) -> str | None:
    """Normalize the synthesis text: strip, drop blank lines, hard-cap at
    `_SYNTHESIS_MAX_LINES` (the prompt asks; code enforces). None when
    the response is empty — treated as malformed by the caller."""
    lines = [ln.rstrip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    return "\n".join(lines[:_SYNTHESIS_MAX_LINES])


def _render_trade_context(
    cand: dict[str, Any],
    *,
    days_held: int | None,
    mechanism: str,
    exit_quality_pct: float | None,
) -> str:
    def _price(v: Any) -> str:
        return f"{float(v):.2f}" if v is not None else "unknown"

    quality = (
        f"{exit_quality_pct:.1f}"
        if exit_quality_pct is not None
        else "n/a (no favorable move recorded)"
    )
    conviction = cand.get("conviction")
    return (
        f"# Closed trade\n"
        f"symbol: {cand['symbol']}\n"
        f"conviction_at_entry: "
        f"{conviction if conviction is not None else 'unknown'}\n"
        f"entry_price: {_price(cand.get('entry_price'))}\n"
        f"exit_price: {_price(cand.get('exit_price'))}\n"
        f"peak_price: {_price(cand.get('high_water_mark'))}\n"
        f"days_held: {days_held if days_held is not None else 'unknown'}\n"
        f"exit_mechanism: {mechanism}\n"
        f"exit_quality_pct: {quality}\n"
        f"---\n"
        f"# Original entry thesis (verbatim)\n"
        f"{cand.get('thesis') or '(no thesis recorded)'}\n"
    )


def _render_postmortems_digest(rows: list[TradePostmortemRow]) -> str:
    lines = [
        "# Recent trade post-mortems (newest first)",
        f"count: {len(rows)}",
        "---",
    ]
    for r in rows:
        date = (
            r.created_at.date().isoformat()
            if r.created_at is not None
            else "unknown"
        )
        quality = (
            f"{r.exit_quality_pct:.1f}%"
            if r.exit_quality_pct is not None
            else "n/a"
        )
        lines.append(
            f"- {date} [{r.symbol}] exit_quality {quality} | "
            f"thesis: {r.thesis_summary or '-'} | "
            f"outcome: {r.outcome_summary or '-'} | "
            f"lesson: {r.lesson or '-'}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The reconciler-tick hook
# ---------------------------------------------------------------------------


class DeskJournalTicker:
    """Per-reconciler-tick driver for both desk-journal write paths.

    Daily post-mortem pass: gated by an in-memory date marker (tick-loop
    short-circuit only); the durable once-per-trade guarantee is the
    `trade_postmortems` row itself, so a restart can never double-write
    a post-mortem — the close-detection query skips rows that exist.

    Weekly synthesis pass: gated by an in-memory ISO-week marker PLUS the
    durable check that the active synthesis row was not already created
    this ISO week (ET) — a restart cannot double-synthesize.
    """

    def __init__(
        self,
        *,
        journal: _JournalLike,
        client: _AnthropicClientPort,
        prompt_builder: _PromptBuilderPort,
        haiku_model_id: str = HAIKU_MODEL_ID,
        daily_cap: int = 10,
        now_fn: Callable[[], dt.datetime] = _utcnow,
    ) -> None:
        self._journal = journal
        self._client = client
        self._builder = prompt_builder
        self._haiku_model_id = haiku_model_id
        self._daily_cap = daily_cap
        self._now_fn = now_fn
        self._postmortems_done_for: dt.date | None = None
        self._synthesis_checked_week: tuple[int, int] | None = None

    async def run_tick(self) -> None:
        """One desk-journal pass. Never raises past a stage boundary —
        the desk journal is an enrichment and must not perturb the
        reconciler cadence (same protective shape as sibling hooks)."""
        now = self._now_fn()
        if self._postmortems_done_for != now.date():
            try:
                await self._generate_postmortems()
                # Marked done only after a completed pass (per-trade
                # failures inside are absorbed); a failed QUERY leaves the
                # marker unset so the next tick retries.
                self._postmortems_done_for = now.date()
            except Exception as e:  # noqa: BLE001 — fail-open by design
                log.error(
                    "desk_journal.postmortem_pass_error",
                    extra={
                        "event": "desk_journal.postmortem_pass_error",
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )
        try:
            await self._maybe_synthesize(now)
        except Exception as e:  # noqa: BLE001 — fail-open by design
            log.error(
                "desk_journal.synthesis_pass_error",
                extra={
                    "event": "desk_journal.synthesis_pass_error",
                    "error": str(e),
                    "error_class": type(e).__name__,
                },
            )

    # ------------------------------------------------------------------
    # Post-mortems
    # ------------------------------------------------------------------

    async def _generate_postmortems(self) -> None:
        candidates = find_closed_executions_without_postmortem(
            self._journal.db_path, limit=self._daily_cap
        )
        for cand in candidates:
            try:
                await self._postmortem_one(cand)
            except Exception as e:  # noqa: BLE001 — one bad trade never
                # blocks the rest; no row was written, so tomorrow retries.
                log.warning(
                    "desk_journal.postmortem_failed",
                    extra={
                        "event": "desk_journal.postmortem_failed",
                        "execution_id": cand.get("execution_id"),
                        "symbol": cand.get("symbol"),
                        "error": str(e),
                        "error_class": type(e).__name__,
                    },
                )

    async def _postmortem_one(self, cand: dict[str, Any]) -> None:
        execution_id = int(cand["execution_id"])
        symbol = str(cand["symbol"])
        mechanism = _exit_mechanism(cand.get("exit_role"))
        days_held = _days_held(cand.get("entry_at"), cand.get("exit_at"))
        exit_quality_pct = compute_exit_quality_pct(
            entry_price=cand.get("entry_price"),
            exit_price=cand.get("exit_price"),
            peak_price=cand.get("high_water_mark"),
        )
        request = self._builder.build_desk_postmortem(
            trade_context=_render_trade_context(
                cand,
                days_held=days_held,
                mechanism=mechanism,
                exit_quality_pct=exit_quality_pct,
            )
        )
        raw = await self._client.call(
            request,
            model_id=self._haiku_model_id,
            purpose="postmortem",
            decision_id=f"postmortem-{execution_id:08d}",
            max_tokens=_POSTMORTEM_MAX_TOKENS,
        )
        parsed = _parse_postmortem(raw)
        if parsed is None:
            # Skip, don't crash: the missing trade_postmortems row keeps
            # the execution in tomorrow's candidate list.
            log.warning(
                "desk_journal.postmortem_malformed",
                extra={
                    "event": "desk_journal.postmortem_malformed",
                    "execution_id": execution_id,
                    "symbol": symbol,
                    "raw_head": raw[:200],
                },
            )
            return
        thesis_summary, outcome_summary, lesson = parsed
        insert_trade_postmortem(
            self._journal.db_path,
            TradePostmortemRow(
                execution_id=execution_id,
                symbol=symbol,
                thesis_summary=thesis_summary,
                outcome_summary=outcome_summary,
                exit_quality_pct=exit_quality_pct,
                lesson=lesson,
            ),
        )
        log.info(
            "desk_journal.postmortem_written",
            extra={
                "event": "desk_journal.postmortem_written",
                "execution_id": execution_id,
                "symbol": symbol,
                "exit_mechanism": mechanism,
                "exit_quality_pct": exit_quality_pct,
                "days_held": days_held,
            },
        )

    # ------------------------------------------------------------------
    # Weekly synthesis
    # ------------------------------------------------------------------

    async def _maybe_synthesize(self, now: dt.datetime) -> None:
        iso = _ensure_aware(now).astimezone(ET).isocalendar()
        week = (iso[0], iso[1])
        if self._synthesis_checked_week == week:
            return
        active = get_active_desk_memory(self._journal.db_path, "synthesis")
        if active is not None and active.created_at is not None:
            created_iso = (
                _ensure_aware(active.created_at).astimezone(ET).isocalendar()
            )
            if (created_iso[0], created_iso[1]) == week:
                # Already synthesized this ISO week (restart-proof check).
                self._synthesis_checked_week = week
                return
        new_count = count_postmortems_since(
            self._journal.db_path,
            active.created_at if active is not None else None,
        )
        if new_count < _SYNTHESIS_MIN_NEW_POSTMORTEMS:
            # Too little new evidence — check again next week. Marking the
            # week checked keeps this evaluation to (at most) once a week.
            self._synthesis_checked_week = week
            log.info(
                "desk_journal.synthesis_skipped_too_few",
                extra={
                    "event": "desk_journal.synthesis_skipped_too_few",
                    "new_postmortems": new_count,
                    "required": _SYNTHESIS_MIN_NEW_POSTMORTEMS,
                },
            )
            return
        rows = get_recent_postmortems(
            self._journal.db_path, limit=_SYNTHESIS_WINDOW_ROWS
        )
        request = self._builder.build_desk_synthesis(
            postmortems_digest=_render_postmortems_digest(rows)
        )
        raw = await self._client.call(
            request,
            model_id=self._haiku_model_id,
            purpose="synthesis",
            decision_id=f"synthesis-{week[0]}W{week[1]:02d}",
            max_tokens=_SYNTHESIS_MAX_TOKENS,
        )
        content = _clean_synthesis(raw)
        # One attempt per week either way — a malformed response is logged
        # and the week is marked checked (next week retries with more data).
        self._synthesis_checked_week = week
        if content is None:
            log.warning(
                "desk_journal.synthesis_malformed",
                extra={
                    "event": "desk_journal.synthesis_malformed",
                    "raw_head": raw[:200],
                },
            )
            return
        version = (active.version + 1) if active is not None else 1
        new_id = insert_desk_memory(
            self._journal.db_path,
            DeskMemoryRow(
                kind="synthesis", content=content, version=version, active=1
            ),
        )
        deactivate_desk_memory(
            self._journal.db_path, "synthesis", except_id=new_id
        )
        log.info(
            "desk_journal.synthesis_written",
            extra={
                "event": "desk_journal.synthesis_written",
                "desk_memory_id": new_id,
                "version": version,
                "postmortems_in_window": len(rows),
                "new_postmortems": new_count,
            },
        )


# ---------------------------------------------------------------------------
# The memory-rail provider
# ---------------------------------------------------------------------------


def make_desk_journal_provider(db_path: str) -> Provider:
    """Provider serving the active weekly synthesis to every LLM purpose.

    Deterministic: the section is a pure render of the active
    `desk_memory(kind='synthesis')` row — no clock, no per-query state.
    Returns None while no synthesis exists yet (first weeks of live
    trading), which the assembler treats as "stay silent".
    """

    def provide(query: MemoryQuery) -> MemorySection | None:
        row = get_active_desk_memory(db_path, "synthesis")
        if row is None:
            return None
        return MemorySection(
            title=DESK_JOURNAL_SECTION_TITLE,
            body=row.content,
            provider_name="desk_journal",
        )

    return provide


__all__ = [
    "DESK_JOURNAL_SECTION_TITLE",
    "HAIKU_MODEL_ID",
    "DeskJournalTicker",
    "compute_exit_quality_pct",
    "make_desk_journal_provider",
]
