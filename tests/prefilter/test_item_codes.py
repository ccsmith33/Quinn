"""S4.1 — 8-K item-code allow/deny prefilter (D-020 / PRD §8.2).

Pure-logic gate before any LLM call; fail-closed on unknown codes.
"""

from __future__ import annotations

from prefilter.item_codes import ItemCodeDecision, ItemCodePrefilter


def test_allow_list_single_item() -> None:
    decision = ItemCodePrefilter().evaluate(["2.02"])
    assert decision.decision == "accept"


def test_deny_list_only() -> None:
    decision = ItemCodePrefilter().evaluate(["5.04", "5.05"])
    assert decision.decision == "reject"
    assert decision.reason == "item_code_deny"


def test_9_01_alone_rejected() -> None:
    decision = ItemCodePrefilter().evaluate(["9.01"])
    assert decision.decision == "reject"
    assert decision.reason == "item_code_9.01_only"


def test_unknown_code_alone_rejected_fail_closed() -> None:
    decision = ItemCodePrefilter().evaluate(["99.99"])
    assert decision.decision == "reject"
    assert decision.reason == "item_code_deny"


def test_unknown_plus_allow_accepted() -> None:
    decision = ItemCodePrefilter().evaluate(["99.99", "2.02"])
    assert decision.decision == "accept"


def test_empty_list_rejected() -> None:
    decision = ItemCodePrefilter().evaluate([])
    assert decision.decision == "reject"
    assert decision.reason == "item_code_empty"


def test_mixed_allow_and_deny_accepted() -> None:
    decision = ItemCodePrefilter().evaluate(["2.02", "5.05"])
    assert decision.decision == "accept"


def test_decision_is_namedtuple_like() -> None:
    decision = ItemCodePrefilter().evaluate(["2.02"])
    assert isinstance(decision, ItemCodeDecision)
    assert decision.decision == "accept"
    assert isinstance(decision.reason, str)


def test_9_01_with_allow_list_item_accepted() -> None:
    # 9.01 is "Financial Statements and Exhibits" — fine when paired with a real item.
    decision = ItemCodePrefilter().evaluate(["2.02", "9.01"])
    assert decision.decision == "accept"


def test_9_01_with_deny_only_rejected() -> None:
    # 9.01 alongside only deny items still has no allow item → reject.
    decision = ItemCodePrefilter().evaluate(["5.05", "9.01"])
    assert decision.decision == "reject"
    assert decision.reason == "item_code_deny"
