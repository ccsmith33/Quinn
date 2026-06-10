"""S6.2 — Deterministic execution validation pipeline (FR-19, FR-20).

The pipeline gates every proposal before order construction (S6.4). Order
of checks (fixed; AC-3): schema → kill-switch → universe → price floor →
capital. Earlier checks short-circuit later ones so a kill-switch reject
costs zero broker calls. The LLM is not consulted here (D-015).

Single code path: there is no `if paper/live` branch in this module
(D-007 sacred). The `BrokerAdapter` Protocol is the only seam.

Sizing (S6.3) and KS-4..KS-7 caps (S6.3) are NOT in this story. AC-6
specifies that S6.2 stubs `realized_dollar_size` with the requested size
(`size_pct_of_capital * equity`) so the surface S6.4 consumes is
forward-compatible; S6.3 replaces this stub with the real engine cleanly.

Journal writes are NOT performed here (AC-7). S5.6 owns the `executions`
row write for the reject path; the rejection-reason strings defined here
are the values that S5.6's `AgentLoop._write_rejected_execution`
persists to `executions.reject_reason`. (Carry-fwd S6.4 reviewer M-1:
S6.4's `OrderSubmitter` only handles ACCEPTED proposals — rejections
never reach S6.4 at all.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from broker.protocol import BrokerAdapter, Quote
from killswitch.api import KillSwitchUninitialized
from observability.log_port import get_logger
from proposal.schemas import TradeProposal

log = get_logger(__name__)

# AC-2: rejection-reason vocabulary, persisted to `executions.reject_reason`
# by S5.6's agent loop (carry-fwd S6.4 reviewer M-1; S6.4's OrderSubmitter
# only writes the ACCEPTED branch). Strings are stable; do not rename
# without coordinating the write path and any downstream report queries.
#
# `pending_capacity` (Feature B) is the placeholder reason written by
# `AgentLoop._execute` when the analyzer's KS-5 capacity gate skipped
# Opus and the loop must short-circuit before broker submission. It is
# NOT emitted by the validator — but it shares the `executions.reject_reason`
# column, so the canonical vocabulary lives here next to the validator
# reasons for ops to read in one place.
RejectReason = Literal[
    "schema",
    "kill_switch",
    "universe",
    "price_floor",
    "exit_geometry",
    "insufficient_capital",
    "direction_unsupported",
    "pending_capacity",
]

# Trade-time price floor (FR-8 / ADR-006 §3): the daily snapshot enforces
# previous-close ≥ $5.00; this re-check covers names that fell below the
# floor between snapshot and order submission.
_PRICE_FLOOR_USD = 5.00


# ---------------------------------------------------------------------------
# Result types (Accepted | Rejected)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Accepted:
    """Validation passed. `realized_dollar_size` is the stub size for S6.2;
    S6.3 replaces this with the sized-and-capped value."""

    proposal: TradeProposal
    realized_dollar_size: float


@dataclass(frozen=True)
class Rejected:
    reason: RejectReason


ValidationResult = Accepted | Rejected


# ---------------------------------------------------------------------------
# Collaborator Protocols (structural, for unit-test fakes)
# ---------------------------------------------------------------------------

class _UniverseLike(Protocol):
    def is_in_universe(self, ticker: str) -> bool: ...


class _KillSwitchLike(Protocol):
    def is_halted(self) -> bool: ...


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ProposalValidator:
    """Pure validator — no I/O of its own; collaborators (broker, universe,
    kill-switch) own the side effects.

    The `journal` parameter on `validate()` is reserved for forward
    compatibility with S6.4's execution-row write path; this story does not
    persist anything (AC-7).

    `min_reward_risk` (D-079 §3.2, default 1.5): the exit-geometry floor.
    Wired from `config.execution.min_reward_risk` at composition; tunable
    when the H3 prod-data verdict lands (D-080).
    """

    def __init__(self, *, min_reward_risk: float = 1.5) -> None:
        self._min_reward_risk = min_reward_risk

    def validate(
        self,
        proposal: TradeProposal,
        broker: BrokerAdapter,
        universe: _UniverseLike,
        ks: _KillSwitchLike,
        journal: object,
    ) -> ValidationResult:
        # 1. Schema (defense-in-depth). S5.5 already validates against
        #    `proposal/schemas.py`; if the model fails here, the caller is
        #    handing us a wrongly-typed object.
        if not isinstance(proposal, TradeProposal):
            log.warning(
                "execution.validator.reject",
                extra={
                    "event": "execution.validator.reject",
                    "reason": "schema",
                },
            )
            return Rejected(reason="schema")

        # 2. Kill-switch (FR-20). Fail closed on `KillSwitchUninitialized`:
        #    the seed row is guaranteed by 001_init.sql, so a missing row
        #    indicates DB tampering or a bad migration; treat as halted.
        try:
            if ks.is_halted():
                return self._reject(proposal, "kill_switch")
        except KillSwitchUninitialized:
            log.error(
                "execution.validator.kill_switch_uninitialized",
                extra={
                    "event": "execution.validator.kill_switch_uninitialized",
                    "symbol": proposal.symbol,
                },
            )
            return self._reject(proposal, "kill_switch")

        # 3. Direction (FR-19: long-only in v1). The TradeProposal pydantic
        #    model declares `direction: Literal["long"]`, but we re-check
        #    here so a hand-mutated payload cannot bypass the rule.
        if proposal.direction != "long":
            return self._reject(proposal, "direction_unsupported")

        # 4. Universe membership.
        if not universe.is_in_universe(proposal.symbol):
            return self._reject(proposal, "universe")

        # 5. Trade-time price floor (FR-8 / ADR-006 §3). Reject if the
        #    current bid OR last has fallen below $5.00 since the daily
        #    snapshot. We require BOTH bid and last to be ≥ $5 because a
        #    crossed quote with bid < 5 still represents executable risk.
        quote: Quote = broker.get_quote(proposal.symbol)
        if quote.bid < _PRICE_FLOOR_USD or quote.last < _PRICE_FLOOR_USD:
            return self._reject(proposal, "price_floor")

        # 5.5 Exit-geometry floor (D-079 §3.2, kills S-4's unconstrained
        #     half). Longs only in v1. Entry reference: the intended fill —
        #     entry_limit_price for limit entries, quote.last otherwise
        #     (same reference the executor's inverted-stop guard uses).
        #     TP is optional (D-009 discretion preserved): a no-TP proposal
        #     relies on stop + trailing ratchet + thesis review and is not
        #     gated here.
        if proposal.take_profit_price is not None:
            entry_ref = (
                proposal.entry_limit_price
                if proposal.entry_style == "limit"
                and proposal.entry_limit_price is not None
                else quote.last
            )
            risk = entry_ref - proposal.stop_loss_price
            reward = proposal.take_profit_price - entry_ref
            # tp <= entry is the AORT-class inversion on the TP side;
            # stop >= entry makes the risk denominator non-positive —
            # both are geometry the floor cannot price.
            if reward <= 0 or risk <= 0:
                return self._reject(proposal, "exit_geometry")
            if reward / risk < self._min_reward_risk:
                log.info(
                    "execution.validator.exit_geometry_below_floor",
                    extra={
                        "event": "execution.validator.exit_geometry_below_floor",
                        "symbol": proposal.symbol,
                        "entry_ref": entry_ref,
                        "stop_loss_price": proposal.stop_loss_price,
                        "take_profit_price": proposal.take_profit_price,
                        "reward_risk": reward / risk,
                        "min_reward_risk": self._min_reward_risk,
                    },
                )
                return self._reject(proposal, "exit_geometry")

        # 6. Capital. AC-6 stub: realized_dollar_size = requested size as a
        #    fraction of equity. S6.3 replaces this with the real sizing
        #    engine (KS-4 caps, conviction tiers, KS-7 reserve). The shape
        #    of `Accepted` is stable so S6.4 won't change.
        account = broker.get_account()
        realized_dollar_size = proposal.size_pct_of_capital * account.equity
        if realized_dollar_size > account.buying_power:
            return self._reject(proposal, "insufficient_capital")

        log.info(
            "execution.validator.accept",
            extra={
                "event": "execution.validator.accept",
                "symbol": proposal.symbol,
                "realized_dollar_size": realized_dollar_size,
            },
        )
        return Accepted(proposal=proposal, realized_dollar_size=realized_dollar_size)

    @staticmethod
    def _reject(proposal: TradeProposal, reason: RejectReason) -> Rejected:
        log.info(
            "execution.validator.reject",
            extra={
                "event": "execution.validator.reject",
                "symbol": proposal.symbol,
                "reason": reason,
            },
        )
        return Rejected(reason=reason)


__all__ = [
    "Accepted",
    "ProposalValidator",
    "RejectReason",
    "Rejected",
    "ValidationResult",
]
