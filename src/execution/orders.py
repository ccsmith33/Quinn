"""S6.4 — Order construction + Alpaca submission + journal write (FR-21, FR-23).

For an `AcceptedProposal` (the validator + sizing combined output) the
submitter:

  1. captures the pre-submission NBBO (`broker.get_quote`) per ADR-001;
  2. submits the order(s) following ONE of two paths:
       - PDT-SUNSET-2026-06-04 (ADR-009): under PDT mode the entry
         is submitted alone (plain market/limit buy via
         `broker.submit_order`); the protective stop and optional TP
         are recorded as `virtual_exits` rows and managed in software.
         Bracket/OTO is unsuitable here because pre-placing protective
         legs at the broker is exactly what PDT mode exists to avoid
         on a sub-$25k account.
       - default flow: a single atomic complex order
           - BRACKET when the proposal carries a take-profit price
             (entry + stop-loss + take-profit, all created together);
           - OTO     when the proposal has no take-profit
             (entry + stop-loss only).
  3. journals the execution row + one order row per leg (or
     `virtual_exits` row(s) on the PDT path).

Single code path for paper / live (D-007 sacred): no `if mode==...`
branching here; the broker adapter is the seam. The PDT branch is
mode-agnostic with respect to paper/live — it depends only on
`pdt_state.is_active()`.

Hotfix 2026-05-07 (incident: 27 unprotected positions): the prior flow
submitted entry, stop, and TP as three sequential `submit_order` calls.
Alpaca's wash-trade detector (broker code 40310000) rejected the
back-to-back sell-stop while the entry buy was still pending pre-market,
and the position landed live without protection. Bracket / OTO orders
are atomic at the broker — either every leg is created or the whole
submission is rejected — so the `submission_partial_no_stop` failure
mode cannot occur on the normal flow. The historical reason value is
retained because crash-recovery / orphan adoption can still produce it
when a process dies between submission and journal write (the broker
has the entry; child legs from a bracket parent show up via
`parent_id`, but the orphan-adoption code path uses deterministic
client_order_ids that bracket children don't carry).

Failure handling:

- Bracket submission fails (entire complex order rejected, retries
  exhausted, etc.): write `executions` row with
  `decision="submission_failed"`. No legs at the broker; no kill-switch
  flip — there is no exposure.
- Bracket submission accepted: write `executions` row with
  `decision="accepted"` plus one `OrderRow` per leg (entry + stop +
  optional TP).

The journal write order matters: execution row first (parent), then
order rows (children).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol

from broker.alpaca import BrokerUnavailable
from broker.protocol import (
    BracketOrderRequest,
    BrokerAdapter,
    BrokerRejected,
    OrderRequest,
    Quote,
    SubmittedOrder,
)
from journal.models import ExecutionRow, OrderRow, VirtualExitRow
from observability.log_port import get_logger
from proposal.schemas import TradeProposal

log = get_logger(__name__)


SubmissionFailureReason = Literal[
    "submission_failed",
    "submission_partial_no_stop",
]


@dataclass(frozen=True)
class AcceptedProposal:
    """Combined validator + sizing output that S6.4 consumes.

    `proposal_id` ties this back to the `proposals` row so the execution
    row's FK can be set. `qty`, `realized_dollar_size`, `realized_pct`,
    and `realized_dollar_size_request` come from the sizing engine.
    """

    proposal: TradeProposal
    proposal_id: int
    qty: int
    realized_dollar_size: float
    realized_pct: float
    realized_dollar_size_request: float


@dataclass(frozen=True)
class SubmissionAccepted:
    execution_id: int
    entry_broker_order_id: str
    # PDT-SUNSET-2026-06-04: ADR-009 — `None` under PDT mode (the
    # protective leg lives in `virtual_exits`, not at the broker).
    stop_broker_order_id: str | None
    take_profit_broker_order_id: str | None


@dataclass(frozen=True)
class SubmissionFailed:
    reason: SubmissionFailureReason
    execution_id: int


SubmissionResult = SubmissionAccepted | SubmissionFailed


class _JournalLike(Protocol):
    def insert_execution(self, row: ExecutionRow) -> int: ...
    def insert_order(self, row: OrderRow) -> int: ...
    # PDT-SUNSET-2026-06-04: ADR-009 — only called from the
    # `pdt_state.is_active()` branch; non-PDT path doesn't touch this.
    def insert_virtual_exit(self, row: VirtualExitRow) -> int: ...


class _PDTStateLike(Protocol):
    """PDT-SUNSET-2026-06-04: structural protocol for `PDTState`.

    Defined as a Protocol (not an import) so this module remains
    decoupled from `execution.pdt_budget` (single-direction dependency:
    consumers import from `pdt_budget`, not the other way around).
    """

    def is_active(self) -> bool: ...


class _KillSwitchLike(Protocol):
    def halt(self, reason: str, set_by: str, notes: str = "") -> None: ...


class OrderSubmitter:
    """Translates an accepted proposal into broker orders and journal rows.

    Stateless; safe to construct once per process.
    """

    def submit(
        self,
        accepted_proposal: AcceptedProposal,
        broker: BrokerAdapter,
        journal: _JournalLike,
        ks: _KillSwitchLike,
        # PDT-SUNSET-2026-06-04: ADR-009 §"Order construction branch".
        # Optional so legacy call sites (and existing unit tests) keep
        # working byte-for-byte. When omitted or `is_active() is False`,
        # the entry-and-pre-place path runs unchanged.
        pdt_state: _PDTStateLike | None = None,
    ) -> SubmissionResult:
        proposal = accepted_proposal.proposal

        # --- 1. Pre-submission NBBO (ADR-001) ---------------------------
        nbbo = broker.get_quote(proposal.symbol)

        # PDT-SUNSET-2026-06-04: ADR-009 §"Order construction branch".
        # Under PDT mode, submit ONLY the entry (a plain market/limit
        # buy) and record stop/TP as `virtual_exits` rows. Bracket /
        # OTO is *not* used here — pre-placing protective legs at the
        # broker is precisely what PDT mode exists to avoid (a sub-$25k
        # account would burn day-trades on every stop hit). The wash-
        # trade race that motivated the bracket flow doesn't apply
        # because no protective leg is submitted.
        if pdt_state is not None and pdt_state.is_active():
            entry_req = self._build_entry(accepted_proposal)
            try:
                entry_resp = broker.submit_order(entry_req)
            except (BrokerRejected, BrokerUnavailable) as e:
                log.error(
                    "execution.submit.entry_failed_pdt_mode",
                    extra={
                        "event": "execution.submit.entry_failed_pdt_mode",
                        "symbol": proposal.symbol,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                execution_id = journal.insert_execution(
                    ExecutionRow(
                        proposal_id=accepted_proposal.proposal_id,
                        decision="submission_failed",
                        realized_size_pct=accepted_proposal.realized_pct,
                        realized_dollar_size=accepted_proposal.realized_dollar_size,
                        submitted_orders_json=json.dumps([]),
                    )
                )
                return SubmissionFailed(
                    reason="submission_failed", execution_id=execution_id
                )
            _ = ks  # not used on the PDT branch (no protective leg can fail)
            return self._submit_with_virtual_exits(
                accepted_proposal, entry_req, entry_resp, nbbo, journal
            )

        # --- 2. Build + submit the bracket / OTO order ----------------
        # Single atomic call: either every leg is created at the broker
        # (entry + stop + optional TP) or the whole submission is
        # rejected. The `submission_partial_no_stop` failure mode that
        # caused the 2026-05-07 incident cannot occur here — Alpaca
        # treats the bundle as one transaction so the wash-trade
        # detector (40310000) does not fire on a back-to-back stop.
        bracket_req = self._build_bracket(accepted_proposal)
        try:
            entry_resp, stop_resp, tp_resp = broker.submit_bracket_order(bracket_req)
        except (BrokerRejected, BrokerUnavailable) as e:
            log.error(
                "execution.submit.bracket_failed",
                extra={
                    "event": "execution.submit.bracket_failed",
                    "symbol": proposal.symbol,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            execution_id = journal.insert_execution(
                ExecutionRow(
                    proposal_id=accepted_proposal.proposal_id,
                    decision="submission_failed",
                    realized_size_pct=accepted_proposal.realized_pct,
                    realized_dollar_size=accepted_proposal.realized_dollar_size,
                    submitted_orders_json=json.dumps([]),
                )
            )
            return SubmissionFailed(reason="submission_failed", execution_id=execution_id)

        # --- 3. Journal: execution row + order rows -------------------
        submitted = [
            {"broker_order_id": entry_resp.broker_order_id, "role": "entry"},
            {"broker_order_id": stop_resp.broker_order_id, "role": "stop"},
        ]
        if tp_resp is not None:
            submitted.append(
                {"broker_order_id": tp_resp.broker_order_id, "role": "take_profit"}
            )

        execution_id = journal.insert_execution(
            ExecutionRow(
                proposal_id=accepted_proposal.proposal_id,
                decision="accepted",
                realized_size_pct=accepted_proposal.realized_pct,
                realized_dollar_size=accepted_proposal.realized_dollar_size,
                submitted_orders_json=json.dumps(submitted),
            )
        )
        journal.insert_order(
            self._entry_row_from_bracket(
                execution_id, accepted_proposal, bracket_req, entry_resp, nbbo
            )
        )
        journal.insert_order(
            self._stop_row_from_bracket(
                execution_id, accepted_proposal, bracket_req, stop_resp
            )
        )
        if tp_resp is not None:
            journal.insert_order(
                self._take_profit_row_from_bracket(
                    execution_id, accepted_proposal, bracket_req, tp_resp
                )
            )

        log.info(
            "execution.submit.accepted",
            extra={
                "event": "execution.submit.accepted",
                "symbol": proposal.symbol,
                "execution_id": execution_id,
                "entry_broker_order_id": entry_resp.broker_order_id,
                "stop_broker_order_id": stop_resp.broker_order_id,
                "take_profit_broker_order_id": tp_resp.broker_order_id if tp_resp else None,
                "order_class": "bracket" if tp_resp is not None else "oto",
            },
        )
        # Suppress unused-variable warning — `ks` is still in the
        # signature for backward compatibility with callers; the
        # submission_partial_no_stop kill-switch halt that consumed it
        # is dead code under bracket flow.
        _ = ks
        return SubmissionAccepted(
            execution_id=execution_id,
            entry_broker_order_id=entry_resp.broker_order_id,
            stop_broker_order_id=stop_resp.broker_order_id,
            take_profit_broker_order_id=tp_resp.broker_order_id if tp_resp else None,
        )

    # ------------------------------------------------------------------
    # Bracket / OTO order builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bracket(ap: AcceptedProposal) -> BracketOrderRequest:
        """Construct a bracket / OTO request from the accepted proposal.

        Class is selected by the broker adapter from `take_profit_price`:
          - take_profit_price set    → BRACKET
          - take_profit_price absent → OTO

        Hotfix 2026-05-07: this is the single submission point for
        normal flow. The historical separate `_build_entry` /
        `_build_stop` / `_build_take_profit` pathway has been retired.
        """
        p = ap.proposal
        entry_order_type: Literal["market", "limit"]
        entry_limit_price: float | None
        if p.entry_style == "limit":
            assert p.entry_limit_price is not None  # enforced by TradeProposal validator
            entry_order_type = "limit"
            entry_limit_price = p.entry_limit_price
        else:
            entry_order_type = "market"
            entry_limit_price = None

        return BracketOrderRequest(
            entry_symbol=p.symbol,
            entry_side="buy",
            entry_qty=ap.qty,
            entry_order_type=entry_order_type,
            entry_tif="day",
            entry_client_order_id=f"prop-{ap.proposal_id}-entry",
            entry_limit_price=entry_limit_price,
            stop_loss_price=p.stop_loss_price,
            take_profit_price=p.take_profit_price,
        )

    # ------------------------------------------------------------------
    # OrderRow constructors
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_row_from_bracket(
        execution_id: int,
        ap: AcceptedProposal,
        req: BracketOrderRequest,
        resp: SubmittedOrder,
        nbbo: Quote,
        notes: str | None = None,
    ) -> OrderRow:
        return OrderRow(
            execution_id=execution_id,
            role="entry",
            symbol=req.entry_symbol,
            side=req.entry_side,
            order_type=req.entry_order_type,
            qty=req.entry_qty,
            limit_price=req.entry_limit_price,
            stop_price=None,
            tif=req.entry_tif,
            broker_order_id=resp.broker_order_id,
            submitted_at=resp.submitted_at,
            pre_submission_bid=nbbo.bid,
            pre_submission_ask=nbbo.ask,
            pre_submission_last=nbbo.last,
            pre_submission_quote_at=nbbo.ts,
            final_status=resp.status,
            notes=notes,
        )

    @staticmethod
    def _stop_row_from_bracket(
        execution_id: int,
        ap: AcceptedProposal,
        req: BracketOrderRequest,
        resp: SubmittedOrder,
    ) -> OrderRow:
        return OrderRow(
            execution_id=execution_id,
            role="stop",
            symbol=req.entry_symbol,
            side="sell",
            order_type="stop",
            qty=req.entry_qty,
            limit_price=None,
            stop_price=req.stop_loss_price,
            tif="gtc",
            broker_order_id=resp.broker_order_id,
            submitted_at=resp.submitted_at,
            final_status=resp.status,
        )

    @staticmethod
    def _take_profit_row_from_bracket(
        execution_id: int,
        ap: AcceptedProposal,
        req: BracketOrderRequest,
        resp: SubmittedOrder,
    ) -> OrderRow:
        assert req.take_profit_price is not None  # caller-checked
        return OrderRow(
            execution_id=execution_id,
            role="take_profit",
            symbol=req.entry_symbol,
            side="sell",
            order_type="limit",
            qty=req.entry_qty,
            limit_price=req.take_profit_price,
            stop_price=None,
            tif="gtc",
            broker_order_id=resp.broker_order_id,
            submitted_at=resp.submitted_at,
            final_status=resp.status,
        )


    # ------------------------------------------------------------------
    # PDT-SUNSET-2026-06-04: ADR-009 §"Order construction branch".
    # The helpers below build plain (non-bracket) entry/stop/TP requests
    # and the entry OrderRow used by the virtual-exit path. They mirror
    # the pre-hotfix flow but are reachable ONLY from the PDT branch;
    # the bracket flow above does not call them.
    # ------------------------------------------------------------------

    @staticmethod
    def _build_entry(ap: AcceptedProposal) -> OrderRequest:
        p = ap.proposal
        if p.entry_style == "limit":
            assert p.entry_limit_price is not None  # enforced by TradeProposal validator
            return OrderRequest(
                symbol=p.symbol,
                side="buy",
                qty=ap.qty,
                order_type="limit",
                tif="day",
                client_order_id=f"prop-{ap.proposal_id}-entry",
                limit_price=p.entry_limit_price,
            )
        return OrderRequest(
            symbol=p.symbol,
            side="buy",
            qty=ap.qty,
            order_type="market",
            tif="day",
            client_order_id=f"prop-{ap.proposal_id}-entry",
        )

    @staticmethod
    def _build_stop(ap: AcceptedProposal) -> OrderRequest:
        p = ap.proposal
        return OrderRequest(
            symbol=p.symbol,
            side="sell",
            qty=ap.qty,
            order_type="stop",
            tif="gtc",
            client_order_id=f"prop-{ap.proposal_id}-stop",
            stop_price=p.stop_loss_price,
        )

    @staticmethod
    def _build_take_profit(ap: AcceptedProposal) -> OrderRequest | None:
        p = ap.proposal
        if p.take_profit_price is None:
            return None
        return OrderRequest(
            symbol=p.symbol,
            side="sell",
            qty=ap.qty,
            order_type="limit",
            tif="gtc",
            client_order_id=f"prop-{ap.proposal_id}-tp",
            limit_price=p.take_profit_price,
        )

    @staticmethod
    def _entry_row(
        execution_id: int,
        ap: AcceptedProposal,
        req: OrderRequest,
        resp: SubmittedOrder,
        nbbo: Quote,
        notes: str | None = None,
    ) -> OrderRow:
        return OrderRow(
            execution_id=execution_id,
            role="entry",
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            qty=req.qty,
            limit_price=req.limit_price,
            stop_price=req.stop_price,
            tif=req.tif,
            broker_order_id=resp.broker_order_id,
            submitted_at=resp.submitted_at,
            pre_submission_bid=nbbo.bid,
            pre_submission_ask=nbbo.ask,
            pre_submission_last=nbbo.last,
            pre_submission_quote_at=nbbo.ts,
            final_status=resp.status,
            notes=notes,
        )

    def _submit_with_virtual_exits(
        self,
        ap: AcceptedProposal,
        entry_req: OrderRequest,
        entry_resp: SubmittedOrder,
        nbbo: Quote,
        journal: _JournalLike,
    ) -> SubmissionAccepted:
        """Entry succeeded; record protective legs as `virtual_exits` rows
        instead of submitting them to the broker.

        Journal write order (AC-4): execution → entry order → virtual
        exits (stop, then optional TP). Crash window between any two
        writes leaves the position open with NO virtual exit; the
        reconciler discrepancy path (existing behavior) catches it.
        """
        p = ap.proposal
        stop_req = self._build_stop(ap)
        tp_req = self._build_take_profit(ap)

        # Build submitted_orders_json placeholder; virtual_exit_ids fill
        # in after the inserts (audit trail per AC-6).
        submitted: list[dict[str, object]] = [
            {"broker_order_id": entry_resp.broker_order_id, "role": "entry"},
        ]
        execution_id = journal.insert_execution(
            ExecutionRow(
                proposal_id=ap.proposal_id,
                decision="accepted",
                realized_size_pct=ap.realized_pct,
                realized_dollar_size=ap.realized_dollar_size,
                # Final json patched below after the virtual_exit ids are
                # known. Insert with the entry-only placeholder so the
                # FK chain is valid even if a later step crashes.
                submitted_orders_json=json.dumps(submitted),
            )
        )
        journal.insert_order(
            self._entry_row(execution_id, ap, entry_req, entry_resp, nbbo)
        )

        # ADR-009 §3.5 / §6 — entry_price = nbbo.last (closest-to-fill
        # pre-fill price). EV ranking later compares current_price
        # against this.
        stop_vid = journal.insert_virtual_exit(
            VirtualExitRow(
                execution_id=execution_id,
                proposal_id=ap.proposal_id,
                symbol=p.symbol,
                qty=ap.qty,
                role="stop",
                entry_price=nbbo.last,
                stop_price=stop_req.stop_price,
                state="active",
            )
        )
        submitted.append({
            "broker_order_id": None,
            "role": "stop",
            "virtual": True,
            "virtual_exit_id": stop_vid,
        })

        tp_vid: int | None = None
        if tp_req is not None:
            tp_vid = journal.insert_virtual_exit(
                VirtualExitRow(
                    execution_id=execution_id,
                    proposal_id=ap.proposal_id,
                    symbol=p.symbol,
                    qty=ap.qty,
                    role="tp",
                    entry_price=nbbo.last,
                    tp_price=tp_req.limit_price,
                    state="active",
                )
            )
            submitted.append({
                "broker_order_id": None,
                "role": "take_profit",
                "virtual": True,
                "virtual_exit_id": tp_vid,
            })

        log.info(
            "pdt.virtual_exit.created",
            extra={
                "event": "pdt.virtual_exit.created",
                "execution_id": execution_id,
                "symbol": p.symbol,
                "stop_virtual_exit_id": stop_vid,
                "stop_price": stop_req.stop_price,
                "tp_virtual_exit_id": tp_vid,
                "tp_price": tp_req.limit_price if tp_req is not None else None,
                "entry_price": nbbo.last,
                "qty": ap.qty,
            },
        )
        log.info(
            "execution.submit.accepted",
            extra={
                "event": "execution.submit.accepted",
                "symbol": p.symbol,
                "execution_id": execution_id,
                "entry_broker_order_id": entry_resp.broker_order_id,
                "stop_broker_order_id": None,
                "take_profit_broker_order_id": None,
                "pdt_mode": True,
            },
        )
        return SubmissionAccepted(
            execution_id=execution_id,
            entry_broker_order_id=entry_resp.broker_order_id,
            stop_broker_order_id=None,
            take_profit_broker_order_id=None,
        )


__all__ = [
    "AcceptedProposal",
    "OrderSubmitter",
    "SubmissionAccepted",
    "SubmissionFailed",
    "SubmissionFailureReason",
    "SubmissionResult",
]
