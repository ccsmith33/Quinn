"""S5.5 — Schema-validation unit tests (pydantic mirrors of the JSON schemas)."""

from __future__ import annotations

import pytest

from proposal.schemas import (
    ProposalSchemaError,
    validate_no_trade_record,
    validate_trade_proposal,
)


def _trade_proposal_minimal() -> dict:
    return {
        "symbol": "ACME",
        "direction": "long",
        "size_pct_of_capital": 0.10,
        "entry_style": "market_open",
        "stop_loss_price": 9.50,
        "time_horizon_days": 14,
        "conviction": 8,
        "thesis": "X" * 60,
        "signals": ["Item 1.01 — Material Definitive Agreement"],
        "exit_conditions": ["Exit on contradicting filing within 5 days"],
        "risk_factors": ["Closing conditions not yet met"],
    }


def test_trade_proposal_minimum_valid() -> None:
    p = validate_trade_proposal(_trade_proposal_minimal())
    assert p.symbol == "ACME"
    assert p.conviction == 8


def test_trade_proposal_size_above_cap_rejected() -> None:
    bad = _trade_proposal_minimal()
    bad["size_pct_of_capital"] = 0.99
    with pytest.raises(ProposalSchemaError) as exc:
        validate_trade_proposal(bad)
    assert exc.value.kind == "trade_proposal"


def test_trade_proposal_size_zero_rejected() -> None:
    bad = _trade_proposal_minimal()
    bad["size_pct_of_capital"] = 0.0
    with pytest.raises(ProposalSchemaError):
        validate_trade_proposal(bad)


def test_trade_proposal_short_rejected() -> None:
    """v1 is long-only per architecture §3.2."""
    bad = _trade_proposal_minimal()
    bad["direction"] = "short"
    with pytest.raises(ProposalSchemaError):
        validate_trade_proposal(bad)


def test_trade_proposal_thesis_too_short_rejected() -> None:
    bad = _trade_proposal_minimal()
    bad["thesis"] = "too short"
    with pytest.raises(ProposalSchemaError):
        validate_trade_proposal(bad)


def test_trade_proposal_signal_too_short_rejected() -> None:
    bad = _trade_proposal_minimal()
    bad["signals"] = ["short"]  # < 10 chars
    with pytest.raises(ProposalSchemaError):
        validate_trade_proposal(bad)


def test_trade_proposal_invalid_symbol_rejected() -> None:
    bad = _trade_proposal_minimal()
    bad["symbol"] = "lowercase"
    with pytest.raises(ProposalSchemaError):
        validate_trade_proposal(bad)


def test_trade_proposal_conviction_out_of_range_rejected() -> None:
    bad = _trade_proposal_minimal()
    bad["conviction"] = 11
    with pytest.raises(ProposalSchemaError):
        validate_trade_proposal(bad)


def test_trade_proposal_horizon_above_60_rejected() -> None:
    bad = _trade_proposal_minimal()
    bad["time_horizon_days"] = 90
    with pytest.raises(ProposalSchemaError):
        validate_trade_proposal(bad)


def test_no_trade_record_minimum_valid() -> None:
    p = validate_no_trade_record(
        {
            "decision": "no_trade",
            "thesis_or_reason": "X" * 80,
            "signals_considered": ["routine governance"],
        }
    )
    assert p.decision == "no_trade"


def test_no_trade_record_wrong_decision_rejected() -> None:
    with pytest.raises(ProposalSchemaError):
        validate_no_trade_record(
            {
                "decision": "trade",
                "thesis_or_reason": "X" * 80,
                "signals_considered": [],
            }
        )


def test_limit_entry_requires_price() -> None:
    """S5.5 review D-2 (parity defect): the JSON Schema's `allOf`
    conditional requires `entry_limit_price` when `entry_style="limit"`.
    `validate_trade_proposal` must enforce the same rule.
    """
    bad = _trade_proposal_minimal()
    bad["entry_style"] = "limit"
    # entry_limit_price intentionally absent
    with pytest.raises(ProposalSchemaError):
        validate_trade_proposal(bad)

    # Adding the price makes it valid.
    bad["entry_limit_price"] = 12.50
    p = validate_trade_proposal(bad)
    assert p.entry_style == "limit"
    assert p.entry_limit_price == 12.50

    # market_open without entry_limit_price stays valid (the conditional
    # only fires for entry_style="limit").
    market = _trade_proposal_minimal()
    assert market["entry_style"] == "market_open"
    assert "entry_limit_price" not in market
    p2 = validate_trade_proposal(market)
    assert p2.entry_style == "market_open"
    assert p2.entry_limit_price is None


def test_no_trade_record_thesis_too_short_rejected() -> None:
    with pytest.raises(ProposalSchemaError):
        validate_no_trade_record(
            {
                "decision": "no_trade",
                "thesis_or_reason": "short",
                "signals_considered": [],
            }
        )
