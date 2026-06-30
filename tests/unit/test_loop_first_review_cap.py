"""Feature A — cap the FIRST thesis-review interval at the config ceiling.

`AgentLoop._schedule_thesis_review` writes the entry-time review-schedule
row at `now + review_days`, where `review_days = min(declared horizon,
ExecutionConfig.max_initial_review_days)`. This pins a fresh position to a
re-look at least as often as the configured cap (default weekly), even when
the analyzer declared a long catalyst horizon (which can run ~30d).

These tests isolate the cap arithmetic by stubbing the journal write — the
FK-constrained `thesis_review_schedule.execution_id` would otherwise force a
full executions row, which the integration suite already exercises.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any

import pytest

import journal.repo as journal_repo
from app.loop import AgentLoop


def _schedule_and_capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    horizon: int,
    cap: int | None,
) -> tuple[dt.datetime, dt.datetime]:
    """Build a minimal AgentLoop, run `_schedule_thesis_review`, and return
    `(due_at, before)` where `before` is captured just prior to the call.

    `cap is None` models legacy/test construction with no config wired.
    """
    captured: dict[str, Any] = {}

    def _fake_insert(db_path: str, row: Any) -> int:
        captured["row"] = row
        return 1

    monkeypatch.setattr(journal_repo, "insert_thesis_review_schedule", _fake_insert)

    config = (
        None
        if cap is None
        else SimpleNamespace(execution=SimpleNamespace(max_initial_review_days=cap))
    )
    components = SimpleNamespace(journal=SimpleNamespace(db_path="unused"))
    loop = AgentLoop(components=components, config=config)  # type: ignore[arg-type]

    proposal = SimpleNamespace(time_horizon_days=horizon)
    before = dt.datetime.now(dt.UTC)
    loop._schedule_thesis_review(execution_id=1, proposal=proposal)  # noqa: SLF001

    row = captured["row"]
    assert row.scheduled_reason == "entry"
    return row.due_at, before


def test_first_review_capped_when_horizon_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 30-day catalyst horizon under a 7-day cap fires the first review at
    7 days, not 30 — the whole point of the feature."""
    due, before = _schedule_and_capture(monkeypatch, horizon=30, cap=7)
    # due == now + 7d, with now microseconds after `before`, so the floored
    # day-delta is exactly 7.
    assert (due - before).days == 7


def test_first_review_uses_horizon_when_below_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short horizon must not be stretched up to the cap — the cap only
    shortens, never lengthens, the first interval."""
    due, before = _schedule_and_capture(monkeypatch, horizon=5, cap=7)
    assert (due - before).days == 5


def test_first_review_uncapped_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy/test construction with no config wired preserves prior
    behavior: the declared horizon is used uncapped. This guards the
    no-config integration path (AgentLoop built without `config=`)."""
    due, before = _schedule_and_capture(monkeypatch, horizon=30, cap=None)
    assert (due - before).days == 30
