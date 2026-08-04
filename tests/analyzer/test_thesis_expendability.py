"""Chopping block — the thesis-review parser's tolerance for the optional
`expendability` (1..5) + `expendability_reason` fields daily-sweep reviews
emit. The decision (hold/close/adjust_*) must NEVER be lost because the
expendability field is absent, out of range, wrong-typed, or otherwise
malformed — non-sweep and legacy responses simply omit it.
"""

from __future__ import annotations

import json

from analyzer.thesis_review import (
    ThesisAdjustStop,
    ThesisClose,
    ThesisHold,
    _parse,
)


def _hold(**extra: object) -> str:
    payload: dict[str, object] = {"decision": "hold", "rationale": "x" * 60}
    payload.update(extra)
    return json.dumps(payload)


def test_expendability_present_parses() -> None:
    r = _parse(_hold(expendability=4, expendability_reason="catalyst spent, drifting"))
    assert isinstance(r, ThesisHold)
    assert r.expendability == 4
    assert r.expendability_reason == "catalyst spent, drifting"


def test_expendability_absent_is_none() -> None:
    r = _parse(_hold())
    assert isinstance(r, ThesisHold)
    assert r.expendability is None
    assert r.expendability_reason is None


def test_expendability_out_of_range_ignored_decision_preserved() -> None:
    for bad in (0, 6, 9, -1):
        r = _parse(_hold(expendability=bad, expendability_reason="x"))
        assert isinstance(r, ThesisHold)  # decision survives
        assert r.expendability is None
        assert r.expendability_reason is None  # reason dropped with the score


def test_expendability_wrong_type_ignored() -> None:
    r = _parse(_hold(expendability="high"))
    assert isinstance(r, ThesisHold)
    assert r.expendability is None


def test_expendability_bool_rejected() -> None:
    # A stray JSON `true` is not the integer 1.
    r = _parse(_hold(expendability=True))
    assert isinstance(r, ThesisHold)
    assert r.expendability is None


def test_reason_without_score_is_dropped() -> None:
    r = _parse(_hold(expendability_reason="orphan reason, no score"))
    assert isinstance(r, ThesisHold)
    assert r.expendability is None
    assert r.expendability_reason is None


def test_expendability_on_close_and_adjust_stop() -> None:
    c = _parse(json.dumps({"decision": "close", "rationale": "y" * 60, "expendability": 5}))
    assert isinstance(c, ThesisClose)
    assert c.expendability == 5
    a = _parse(
        json.dumps(
            {
                "decision": "adjust_stop",
                "rationale": "z" * 60,
                "modifications": {"new_stop_price": 12.5},
                "expendability": 2,
                "expendability_reason": "working but no dated event",
            }
        )
    )
    assert isinstance(a, ThesisAdjustStop)
    assert a.new_stop_price == 12.5
    assert a.expendability == 2
    assert a.expendability_reason == "working but no dated event"
