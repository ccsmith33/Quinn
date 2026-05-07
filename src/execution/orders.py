"""S6.4 — Order construction + Alpaca submission + journal write (FR-21, FR-23).

For an `AcceptedProposal` (the validator + sizing combined output) the
submitter:

  1. captures the pre-submission NBBO (`broker.get_quote`) per ADR-001;
  2. submits the entry order (market or limit BUY, TIF=day);
  3. submits the protective stop-loss (sell stop, TIF=GTC);
  4. submits the optional take-profit (sell limit, TIF=GTC).

Single code path for paper / live (D-007 sacred): no `if mode==...`
branching here; the broker adapter is the seam.

Failure handling (AC-8):

- Entry fails (broker exhausted retries before any order placed): write
  `executions` row with `decision="submission_failed"`. No protective
  legs attempted; no kill-switch flip — there is no exposure.
- Entry succeeds, stop fails: this is a critical state — the position is
  open without protection. Flip the kill-switch to `halted` with reason
  `submission_partial_no_stop`, set_by `system`. The execution row
  records `decision="submission_partial_no_stop"`. The take-profit leg
  is *not* attempted because the operator must triage; KS-halted state
  also blocks subsequent entries (FR-20).
- Entry + stop succeed, TP fails: protective leg is in place, so this is
  recoverable; the execution decision is still `accepted` and the TP
  failure is journaled in the entry/stop rows' `notes` for operator
  review.

The journal write order matters: execution row first (parent), then
order rows (children). On entry failure we still write the execution
row so the rejection is auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol

from broker.alpaca import BrokerUnavailable
from broker.protocol import BrokerAdapter, BrokerRejected, OrderRequest, Quote, SubmittedOrder
from journal.models import ExecutionRow, OrderRow
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
    stop_broker_order_id: str
    take_profit_broker_order_id: str | None


@dataclass(frozen=True)
class SubmissionFailed:
    reason: SubmissionFailureReason
    execution_id: int


SubmissionResult = SubmissionAccepted | SubmissionFailed


class _JournalLike(Protocol):
    def insert_execution(self, row: ExecutionRow) -> int: ...
    def insert_order(self, row: OrderRow) -> int: ...


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
    ) -> SubmissionResult:
        proposal = accepted_proposal.proposal

        # --- 1. Pre-submission NBBO (ADR-001) ---------------------------
        nbbo = broker.get_quote(proposal.symbol)

        # --- 2. Build + submit entry order -----------------------------
        # `BrokerRejected` covers non-retryable broker rejections (e.g.
        # 422 insufficient buying power); `BrokerUnavailable` covers the
        # transient-then-exhausted path. Both share a single failure
        # branch — there is no exposure on entry-leg failure (no fill, no
        # position), so we journal `submission_failed` and stop. (Hotfix
        # 2026-05-07: prior to this, raw APIErrors propagated past the
        # submitter unjournaled.)
        entry_req = self._build_entry(accepted_proposal)
        try:
            entry_resp = broker.submit_order(entry_req)
        except (BrokerRejected, BrokerUnavailable) as e:
            log.error(
                "execution.submit.entry_failed",
                extra={
                    "event": "execution.submit.entry_failed",
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

        # --- 3. Build + submit stop-loss (entry now exposed) ----------
        # Critical handler: the entry order is alive at the broker. ANY
        # failure (transient `BrokerUnavailable` OR non-retryable
        # `BrokerRejected` — e.g. Alpaca's 40310000 wash-trade detection
        # when the entry is still pending pre-market) leaves the position
        # unprotected. Both must trip the killswitch and journal the
        # partial state. Hotfix 2026-05-07: incident_2026_05_07_unprotected_positions
        # describes the wash-trade variant escaping the prior
        # `BrokerUnavailable`-only handler.
        stop_req = self._build_stop(accepted_proposal)
        try:
            stop_resp = broker.submit_order(stop_req)
        except (BrokerRejected, BrokerUnavailable) as e:
            log.critical(
                "execution.submit.stop_failed_position_unprotected",
                extra={
                    "event": "execution.submit.stop_failed_position_unprotected",
                    "symbol": proposal.symbol,
                    "entry_broker_order_id": entry_resp.broker_order_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # KS halt FIRST so subsequent entries are blocked, then journal
            # the partial state. (Order matters: a crash between these two
            # writes still leaves KS halted, the conservative state.)
            ks.halt(
                reason="submission_partial_no_stop",
                set_by="system",
                notes=(
                    f"entry {entry_resp.broker_order_id} for {proposal.symbol} "
                    f"submitted; stop submission failed: {e}"
                ),
            )
            submitted = [{"broker_order_id": entry_resp.broker_order_id, "role": "entry"}]
            execution_id = journal.insert_execution(
                ExecutionRow(
                    proposal_id=accepted_proposal.proposal_id,
                    decision="submission_partial_no_stop",
                    realized_size_pct=accepted_proposal.realized_pct,
                    realized_dollar_size=accepted_proposal.realized_dollar_size,
                    submitted_orders_json=json.dumps(submitted),
                )
            )
            journal.insert_order(
                self._entry_row(
                    execution_id, accepted_proposal, entry_req, entry_resp, nbbo
                )
            )
            return SubmissionFailed(
                reason="submission_partial_no_stop",
                execution_id=execution_id,
            )

        # --- 4. Optional take-profit ----------------------------------
        tp_resp: SubmittedOrder | None = None
        tp_note: str | None = None
        tp_req = self._build_take_profit(accepted_proposal)
        if tp_req is not None:
            try:
                tp_resp = broker.submit_order(tp_req)
            except (BrokerRejected, BrokerUnavailable) as e:
                # Position is protected by stop; TP failure is recoverable.
                tp_note = f"tp_submission_failed: {e}"
                log.warning(
                    "execution.submit.tp_failed",
                    extra={
                        "event": "execution.submit.tp_failed",
                        "symbol": proposal.symbol,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

        # --- 5. Journal: execution row + order rows -------------------
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
            self._entry_row(
                execution_id, accepted_proposal, entry_req, entry_resp, nbbo, notes=tp_note
            )
        )
        journal.insert_order(
            self._stop_row(execution_id, accepted_proposal, stop_req, stop_resp)
        )
        if tp_resp is not None and tp_req is not None:
            journal.insert_order(
                self._take_profit_row(execution_id, accepted_proposal, tp_req, tp_resp)
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
            },
        )
        return SubmissionAccepted(
            execution_id=execution_id,
            entry_broker_order_id=entry_resp.broker_order_id,
            stop_broker_order_id=stop_resp.broker_order_id,
            take_profit_broker_order_id=tp_resp.broker_order_id if tp_resp else None,
        )

    # ------------------------------------------------------------------
    # Order builders (AC-2, AC-3)
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

    # ------------------------------------------------------------------
    # OrderRow constructors
    # ------------------------------------------------------------------

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

    @staticmethod
    def _stop_row(
        execution_id: int,
        ap: AcceptedProposal,
        req: OrderRequest,
        resp: SubmittedOrder,
    ) -> OrderRow:
        return OrderRow(
            execution_id=execution_id,
            role="stop",
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            qty=req.qty,
            limit_price=req.limit_price,
            stop_price=req.stop_price,
            tif=req.tif,
            broker_order_id=resp.broker_order_id,
            submitted_at=resp.submitted_at,
            final_status=resp.status,
        )

    @staticmethod
    def _take_profit_row(
        execution_id: int,
        ap: AcceptedProposal,
        req: OrderRequest,
        resp: SubmittedOrder,
    ) -> OrderRow:
        return OrderRow(
            execution_id=execution_id,
            role="take_profit",
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            qty=req.qty,
            limit_price=req.limit_price,
            stop_price=req.stop_price,
            tif=req.tif,
            broker_order_id=resp.broker_order_id,
            submitted_at=resp.submitted_at,
            final_status=resp.status,
        )


__all__ = [
    "AcceptedProposal",
    "OrderSubmitter",
    "SubmissionAccepted",
    "SubmissionFailed",
    "SubmissionFailureReason",
    "SubmissionResult",
]
