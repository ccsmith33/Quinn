"""D-079 §3.4 — thesis-review decision parsing.

The decision space grows to `hold | close | adjust_stop |
adjust_take_profit` (raise-only TP actuator, operator-ratified
let-winners-run). Parser-level unit tests; the coordinator apply paths
are covered in tests/integration/test_agent_loop_thesis_review.py.
"""

from __future__ import annotations

import json

from analyzer.thesis_review import (
    ThesisAdjustStop,
    ThesisAdjustTakeProfit,
    ThesisClose,
    ThesisHold,
    ThesisReviewMalformed,
    _parse,
)


def _payload(**overrides: object) -> str:
    base: dict[str, object] = {
        "decision": "hold",
        "rationale": (
            "Catalyst still in regulatory-review window; no contradicting "
            "filing since entry; price between stop and take-profit."
        ),
    }
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# Existing decision space — regression
# ---------------------------------------------------------------------------


def test_parse_hold() -> None:
    result = _parse(_payload(decision="hold"))
    assert isinstance(result, ThesisHold)


def test_parse_close() -> None:
    result = _parse(_payload(decision="close"))
    assert isinstance(result, ThesisClose)


def test_parse_adjust_stop() -> None:
    result = _parse(
        _payload(
            decision="adjust_stop",
            modifications={"new_stop_price": 105.0},
        )
    )
    assert isinstance(result, ThesisAdjustStop)
    assert result.new_stop_price == 105.0


def test_parse_unknown_decision_malformed() -> None:
    result = _parse(_payload(decision="liquidate_everything"))
    assert isinstance(result, ThesisReviewMalformed)


# ---------------------------------------------------------------------------
# D-079 §3.4 — adjust_take_profit
# ---------------------------------------------------------------------------


def test_parse_adjust_take_profit() -> None:
    result = _parse(
        _payload(
            decision="adjust_take_profit",
            modifications={"new_tp_price": 140.0},
        )
    )
    assert isinstance(result, ThesisAdjustTakeProfit)
    assert result.new_tp_price == 140.0


def test_parse_adjust_take_profit_requires_modifications() -> None:
    result = _parse(_payload(decision="adjust_take_profit"))
    assert isinstance(result, ThesisReviewMalformed)


def test_parse_adjust_take_profit_requires_positive_price() -> None:
    result = _parse(
        _payload(
            decision="adjust_take_profit",
            modifications={"new_tp_price": 0},
        )
    )
    assert isinstance(result, ThesisReviewMalformed)


def test_parse_adjust_take_profit_rejects_wrong_key() -> None:
    """An adjust_take_profit carrying only new_stop_price is malformed —
    the actions must not silently cross wires."""
    result = _parse(
        _payload(
            decision="adjust_take_profit",
            modifications={"new_stop_price": 105.0},
        )
    )
    assert isinstance(result, ThesisReviewMalformed)


def test_modifications_json_round_trip_for_adjust_take_profit() -> None:
    from analyzer.thesis_review import _decision_string, _modifications_json

    result = _parse(
        _payload(
            decision="adjust_take_profit",
            modifications={"new_tp_price": 140.0},
        )
    )
    assert _decision_string(result) == "adjust_take_profit"
    mods = _modifications_json(result)
    assert mods is not None
    assert json.loads(mods) == {"new_tp_price": 140.0}
