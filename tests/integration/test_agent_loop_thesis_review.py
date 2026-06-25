"""Feature A — time/thesis review on open positions.

Covers the four primary AC paths plus the safety-critical adjust_stop
ordering (new-stop-first, then cancel old).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from broker.protocol import (
    AccountSnapshot,
    OrderRequest,
    Position,
    Quote,
    SubmittedOrder,
)
from journal.models import (
    ExecutionRow,
    FilingRow,
    OrderRow,
    PositionRow,
    PromptRow,
    ProposalRow,
    ThesisReviewScheduleRow,
)
from journal.repo import (
    connect,
    insert_execution,
    insert_filing,
    insert_order,
    insert_position,
    insert_prompt,
    insert_proposal,
    insert_thesis_review_schedule,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeBrokerWithCancelTracking:
    """Records submit + cancel + replace_stop order, returns canned
    quotes. Records the SEQUENCE of mutating calls in `call_log` so we
    can assert ordering invariants."""

    def __init__(self, *, last_price: float = 110.0) -> None:
        self.submitted: list[OrderRequest] = []
        self.canceled: list[str] = []
        self.replaced: list[dict[str, Any]] = []
        # Ordered log of (op_name, payload) tuples — primary mechanism
        # for asserting "atomic replace happened, no cancel of the old
        # stop, no submit of a new stop order."
        self.call_log: list[tuple[str, dict[str, Any]]] = []
        self.next_id = 0
        self._last_price = last_price
        self._submit_should_fail = False
        self._cancel_should_fail = False
        self._replace_should_fail = False
        self._replace_limit_should_fail = False
        self.replaced_limits: list[dict[str, Any]] = []
        # §3.3 step-5 fallback support: per-broker-order-id state the
        # coordinator probes via get_order_by_id (WS1 §7.1 — the fakes
        # grow it ahead of the real adapter so WS2 stays green
        # standalone), plus broker-side open positions and an OCO
        # recorder for the re-place path.
        self.orders_by_id: dict[str, SubmittedOrder] = {}
        self.positions: list[Position] = []
        self.submitted_oco: list[dict[str, Any]] = []

    def seed_order(self, order: SubmittedOrder) -> None:
        self.orders_by_id[order.broker_order_id] = order

    def get_order_by_id(self, broker_order_id: str) -> SubmittedOrder | None:
        self.call_log.append(("get_order_by_id", {"id": broker_order_id}))
        return self.orders_by_id.get(broker_order_id)

    def submit_oco_sell(
        self,
        *,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float | None,
        client_order_id: str,
    ) -> tuple[SubmittedOrder, SubmittedOrder | None]:
        self.submitted_oco.append(
            {
                "symbol": symbol,
                "qty": qty,
                "stop_price": stop_price,
                "limit_price": limit_price,
                "client_order_id": client_order_id,
            }
        )
        self.call_log.append(
            (
                "submit_oco",
                {
                    "symbol": symbol,
                    "stop_price": stop_price,
                    "limit_price": limit_price,
                },
            )
        )
        self.next_id += 1
        stop = SubmittedOrder(
            broker_order_id=f"thfake-oco-stop-{self.next_id:06d}",
            client_order_id=f"{client_order_id}-stop",
            symbol=symbol,
            side="sell",
            qty=qty,
            order_type="stop",
            status="accepted",
            submitted_at=dt.datetime.now(dt.UTC),
            stop_price=stop_price,
        )
        tp: SubmittedOrder | None = None
        if limit_price is not None:
            self.next_id += 1
            tp = SubmittedOrder(
                broker_order_id=f"thfake-oco-tp-{self.next_id:06d}",
                client_order_id=f"{client_order_id}-tp",
                symbol=symbol,
                side="sell",
                qty=qty,
                order_type="limit",
                status="accepted",
                submitted_at=dt.datetime.now(dt.UTC),
                limit_price=limit_price,
            )
        return stop, tp

    def fail_next_submit(self) -> None:
        self._submit_should_fail = True

    def fail_next_cancel(self) -> None:
        self._cancel_should_fail = True

    def fail_next_replace(self) -> None:
        self._replace_should_fail = True

    def fail_next_replace_limit(self) -> None:
        self._replace_limit_should_fail = True

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        if self._submit_should_fail:
            self._submit_should_fail = False
            self.call_log.append(("submit_failed", {"client_order_id": req.client_order_id}))
            raise RuntimeError("simulated submit failure")
        self.submitted.append(req)
        self.call_log.append(
            (
                "submit",
                {
                    "client_order_id": req.client_order_id,
                    "side": req.side,
                    "order_type": req.order_type,
                    "stop_price": req.stop_price,
                },
            )
        )
        self.next_id += 1
        return SubmittedOrder(
            broker_order_id=f"thfake-{self.next_id:06d}",
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            order_type=req.order_type,
            status="accepted",
            submitted_at=dt.datetime.now(dt.UTC),
            limit_price=req.limit_price,
            stop_price=req.stop_price,
        )

    def cancel_order(self, broker_order_id: str) -> None:
        if self._cancel_should_fail:
            self._cancel_should_fail = False
            self.call_log.append(("cancel_failed", {"id": broker_order_id}))
            raise RuntimeError("simulated cancel failure")
        self.canceled.append(broker_order_id)
        self.call_log.append(("cancel", {"id": broker_order_id}))

    def replace_stop_order(
        self,
        broker_order_id: str,
        *,
        new_stop_price: float,
        client_order_id: str,
    ) -> SubmittedOrder:
        if self._replace_should_fail:
            self._replace_should_fail = False
            self.call_log.append(
                (
                    "replace_failed",
                    {"id": broker_order_id, "new_stop_price": new_stop_price},
                )
            )
            raise RuntimeError("simulated replace failure")
        self.replaced.append(
            {
                "old_id": broker_order_id,
                "new_stop_price": new_stop_price,
                "client_order_id": client_order_id,
            }
        )
        self.next_id += 1
        new_id = f"thfake-replace-{self.next_id:06d}"
        self.call_log.append(
            (
                "replace",
                {
                    "old_id": broker_order_id,
                    "new_id": new_id,
                    "new_stop_price": new_stop_price,
                },
            )
        )
        return SubmittedOrder(
            broker_order_id=new_id,
            client_order_id=client_order_id,
            symbol="ACME",
            side="sell",
            qty=100,
            order_type="stop",
            status="accepted",
            submitted_at=dt.datetime.now(dt.UTC),
            stop_price=new_stop_price,
        )

    def replace_limit_order(
        self,
        broker_order_id: str,
        *,
        new_limit_price: float,
        client_order_id: str,
    ) -> SubmittedOrder:
        if self._replace_limit_should_fail:
            self._replace_limit_should_fail = False
            self.call_log.append(
                (
                    "replace_limit_failed",
                    {"id": broker_order_id, "new_limit_price": new_limit_price},
                )
            )
            raise RuntimeError("simulated replace_limit failure")
        self.replaced_limits.append(
            {
                "old_id": broker_order_id,
                "new_limit_price": new_limit_price,
                "client_order_id": client_order_id,
            }
        )
        self.next_id += 1
        new_id = f"thfake-replace-limit-{self.next_id:06d}"
        self.call_log.append(
            (
                "replace_limit",
                {
                    "old_id": broker_order_id,
                    "new_id": new_id,
                    "new_limit_price": new_limit_price,
                },
            )
        )
        return SubmittedOrder(
            broker_order_id=new_id,
            client_order_id=client_order_id,
            symbol="ACME",
            side="sell",
            qty=100,
            order_type="limit",
            status="accepted",
            submitted_at=dt.datetime.now(dt.UTC),
            limit_price=new_limit_price,
        )

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=100_000.0,
            cash=50_000.0,
            buying_power=50_000.0,
            long_market_value=50_000.0,
            daypl=0.0,
            snapshot_at=dt.datetime.now(dt.UTC),
        )

    def get_positions(self) -> list[Position]:
        return list(self.positions)

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            bid=self._last_price - 0.01,
            ask=self._last_price + 0.01,
            last=self._last_price,
            ts=dt.datetime.now(dt.UTC),
        )

    def get_order_by_client_id(self, client_order_id: str) -> SubmittedOrder | None:
        return None


class _FakeAnthropicForThesis:
    """Anthropic client stub — writes one llm_calls row per call so the
    ThesisReviewer's telemetry read works."""

    def __init__(self, db_path: str, response: str) -> None:
        self._db = db_path
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def call(
        self,
        request: Any,
        *,
        model_id: str,
        purpose: str,
        decision_id: str,
        max_tokens: int | None = None,
    ) -> str:
        from journal.models import LlmCallRow
        from journal.repo import insert_llm_call

        self.calls.append(
            {"model_id": model_id, "purpose": purpose, "decision_id": decision_id}
        )
        insert_llm_call(
            self._db,
            LlmCallRow(
                decision_id=decision_id,
                purpose=purpose,
                model_id=model_id,
                prompt_version=request.prompt_version,
                input_tokens=1500,
                output_tokens=400,
                cache_read_tokens=4000,
                cache_creation_tokens=0,
                latency_ms=2000,
                cost_usd=0.05,
                error_class=None,
            ),
        )
        return self._response


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _setup_open_position(
    db: str,
    *,
    symbol: str = "ACME",
    horizon_days: int = 14,
    stop_loss_price: float = 90.0,
    take_profit_price: float | None = 150.0,
    fill_price: float = 100.0,
    qty: int = 100,
    entry_at: dt.datetime,
    sonnet_pv: str,
) -> tuple[int, int]:
    """Insert a complete trade: filing → proposal → execution(accepted) →
    orders (entry/stop/take_profit) → positions snapshot. Returns
    (proposal_id, execution_id)."""
    fid = insert_filing(
        db,
        FilingRow(
            accession_number=f"thesis-acc-{symbol}",
            cik=320193,
            form_type="8-K",
            filed_at=entry_at - dt.timedelta(hours=2),
            fetched_at=entry_at - dt.timedelta(hours=1),
            raw_text_path=f"/tmp/thesis-{symbol}.txt",
            content_hash=f"hash-{symbol}",
            item_codes='["1.01"]',
            issuer_ticker=symbol,
        ),
    )
    payload = {
        "symbol": symbol,
        "direction": "long",
        "size_pct_of_capital": 0.10,
        "entry_style": "market_open",
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "time_horizon_days": horizon_days,
        "conviction": 8,
        "thesis": (
            f"{symbol} announced a material acquisition with concrete pricing "
            "terms; integration timeline plausible."
        ),
        "signals": ["Item 1.01 — Material Definitive Agreement"],
        "exit_conditions": ["Exit on contradicting filing within 5 trading days"],
        "risk_factors": ["Closing conditions not yet met"],
    }
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id=f"thesis-d-{symbol}",
            model_id="claude-sonnet-4-6",
            prompt_version=sonnet_pv,
            raw_response=json.dumps(payload),
            kind="trade_proposal",
            symbol=symbol,
            direction="long",
            size_pct_requested=0.10,
            conviction=8,
            thesis=payload["thesis"],
            input_tokens=1500,
            output_tokens=800,
            cache_read_tokens=4000,
            cache_creation_tokens=0,
            latency_ms=2000,
            cost_usd=0.02,
        ),
    )
    eid = insert_execution(
        db,
        ExecutionRow(
            proposal_id=pid,
            decision="accepted",
            realized_size_pct=0.10,
            realized_dollar_size=qty * fill_price,
            submitted_orders_json="[]",
        ),
    )
    insert_order(
        db,
        OrderRow(
            execution_id=eid,
            role="entry",
            symbol=symbol,
            side="buy",
            order_type="market",
            qty=qty,
            tif="day",
            broker_order_id=f"orig-entry-{symbol}",
            submitted_at=entry_at,
            realized_fill_price=fill_price,
            realized_fill_qty=qty,
            realized_fill_at=entry_at,
            final_status="filled",
        ),
    )
    # D-078 §7.4: live protective legs carry final_status NULL until a
    # terminal disposition is recorded — the §3.3/§3.4 live-order lookups
    # key on exactly that.
    insert_order(
        db,
        OrderRow(
            execution_id=eid,
            role="stop",
            symbol=symbol,
            side="sell",
            order_type="stop",
            qty=qty,
            tif="gtc",
            stop_price=stop_loss_price,
            broker_order_id=f"orig-stop-{symbol}",
            submitted_at=entry_at,
            final_status=None,
        ),
    )
    if take_profit_price is not None:
        insert_order(
            db,
            OrderRow(
                execution_id=eid,
                role="take_profit",
                symbol=symbol,
                side="sell",
                order_type="limit",
                qty=qty,
                tif="gtc",
                limit_price=take_profit_price,
                broker_order_id=f"orig-tp-{symbol}",
                submitted_at=entry_at,
                final_status=None,
            ),
        )
    insert_position(
        db,
        PositionRow(
            snapshot_at=entry_at,
            source="reconciler",
            symbol=symbol,
            qty=qty,
            avg_entry_price=fill_price,
            market_value=qty * fill_price,
            unrealized_pnl=0.0,
        ),
    )
    return pid, eid


def _seed_prompts(db: str, prompt_builder: Any) -> tuple[str, str]:
    """Register sonnet + opus_thesis_review prompt versions so FK holds.
    Returns (sonnet_pv, thesis_pv)."""
    sonnet_pv = "sonnet_filing_analysis@thesis-test"
    insert_prompt(
        db,
        PromptRow(
            prompt_version=sonnet_pv,
            name="sonnet_filing_analysis",
            file_path="src/prompts/sonnet_filing_analysis_v1.txt",
            content_hash="x" * 64,
        ),
    )
    # The thesis reviewer composes the ACTIVE prompt (v2 since D-079).
    from prompts.loader import ACTIVE_OPUS_THESIS_REVIEW_PROMPT

    thesis_pv = prompt_builder.prompt_version(ACTIVE_OPUS_THESIS_REVIEW_PROMPT)
    insert_prompt(
        db,
        PromptRow(
            prompt_version=thesis_pv,
            name=ACTIVE_OPUS_THESIS_REVIEW_PROMPT,
            file_path=f"src/prompts/{ACTIVE_OPUS_THESIS_REVIEW_PROMPT}.txt",
            content_hash="y" * 64,
        ),
    )
    return sonnet_pv, thesis_pv


def _hold_response() -> str:
    return json.dumps(
        {
            "decision": "hold",
            "rationale": (
                "Catalyst from Item 1.01 still in regulatory-review window; "
                "no contradicting filing has appeared since entry. Position "
                "is roughly flat from cost basis with stop intact — extending "
                "horizon is justified."
            ),
        }
    )


def _close_response() -> str:
    return json.dumps(
        {
            "decision": "close",
            "rationale": (
                "Acquisition closed and re-rating has played out; expected "
                "value of holding the slot is below cost. The thesis has "
                "resolved cleanly per the original time horizon."
            ),
        }
    )


def _adjust_tp_response(new_tp: float) -> str:
    return json.dumps(
        {
            "decision": "adjust_take_profit",
            "rationale": (
                "Catalyst has strengthened and the position is through the "
                "original target zone with momentum; raising the take-profit "
                "extends the run while the ratcheted stop protects the gain."
            ),
            "modifications": {"new_tp_price": new_tp},
        }
    )


def _adjust_stop_response(new_stop: float) -> str:
    return json.dumps(
        {
            "decision": "adjust_stop",
            "rationale": (
                "Position has moved in our favour by ~10% with the catalyst "
                "still in play. Tightening the stop locks in a meaningful "
                "portion of the realized gain while leaving room for the "
                "thesis to keep playing out."
            ),
            "modifications": {"new_stop_price": new_stop},
        }
    )


def _live_order(
    broker_order_id: str,
    *,
    order_type: str = "stop",
    status: str = "accepted",
    stop_price: float | None = None,
    limit_price: float | None = None,
    symbol: str = "ACME",
    qty: int = 100,
) -> SubmittedOrder:
    """SubmittedOrder snapshot for seeding the fake's get_order_by_id."""
    return SubmittedOrder(
        broker_order_id=broker_order_id,
        client_order_id=f"cid-{broker_order_id}",
        symbol=symbol,
        side="sell",
        qty=qty,
        order_type=order_type,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        submitted_at=dt.datetime.now(dt.UTC),
        stop_price=stop_price,
        limit_price=limit_price,
    )


def _broker_position(symbol: str, *, qty: int = 100) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        avg_entry_price=100.0,
        market_value=qty * 110.0,
        unrealized_pnl=qty * 10.0,
    )


class _OutcomeRecorder:
    """Fake of WS1's §7.4 `record_order_outcome` port — captures the
    (order_id, final_status) completions the coordinator emits."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def __call__(
        self,
        order_id: int,
        final_status: str,
        *,
        fill_price: float | None,
        fill_qty: int | None,
        fill_at: dt.datetime | None,
    ) -> None:
        self.calls.append((order_id, final_status))


def _make_coordinator(
    db_path: str,
    journal: Any,
    prompt_builder: Any,
    broker: Any,
    response: str,
    now: dt.datetime,
    *,
    recorder: _OutcomeRecorder | None = None,
) -> Any:
    from analyzer.thesis_review import ThesisReviewer
    from app.thesis_coordinator import ThesisReviewCoordinator

    reviewer = ThesisReviewer(
        client=_FakeAnthropicForThesis(db_path, response),
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        db_path=db_path,
    )
    kwargs: dict[str, Any] = {}
    if recorder is not None:
        kwargs["record_order_outcome"] = recorder
    return ThesisReviewCoordinator(
        journal=journal,
        broker=broker,
        reviewer=reviewer,
        filings_lookup=lambda *, issuer_ticker, since: "(no filings since entry)",
        now_fn=lambda: now,
        **kwargs,
    )


def _orders_for_execution(db_path: str, execution_id: int) -> list[OrderRow]:
    from journal.repo import get_orders_for_execution

    return get_orders_for_execution(db_path, execution_id)


def _due_schedule(db_path: str, execution_id: int, now: dt.datetime) -> None:
    insert_thesis_review_schedule(
        db_path,
        ThesisReviewScheduleRow(
            execution_id=execution_id,
            due_at=now - dt.timedelta(hours=1),
            scheduled_reason="entry",
        ),
    )


def _pending_schedules(db_path: str, execution_id: int, now: dt.datetime) -> list:
    """Schedules still due AFTER `now` — i.e. the follow-ups written by
    the decision handlers."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM thesis_review_schedule "
            "WHERE execution_id = ? AND due_at > ?",
            (execution_id, now),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thesis_review_hold_path_reschedules_plus_seven_days(
    db_path: str, journal, prompt_builder
) -> None:
    """AC: Opus says `hold` → no broker action → next schedule at +7d."""
    from analyzer.thesis_review import ThesisReviewer
    from app.thesis_coordinator import (
        HOLD_RESCHEDULE_DAYS,
        ThesisReviewCoordinator,
    )
    from journal.repo import (
        connect as _connect,
        get_latest_thesis_schedule_for_execution,
    )

    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    entry_at = now - dt.timedelta(days=15)
    _, eid = _setup_open_position(
        db_path, entry_at=entry_at, sonnet_pv=sonnet_pv
    )
    insert_thesis_review_schedule(
        db_path,
        ThesisReviewScheduleRow(
            execution_id=eid,
            due_at=now - dt.timedelta(hours=1),
            scheduled_reason="entry",
        ),
    )

    broker = _FakeBrokerWithCancelTracking(last_price=100.0)
    fake_anth = _FakeAnthropicForThesis(db_path, _hold_response())
    reviewer = ThesisReviewer(
        client=fake_anth,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        db_path=db_path,
    )
    coordinator = ThesisReviewCoordinator(
        journal=journal,
        broker=broker,
        reviewer=reviewer,
        filings_lookup=lambda *, issuer_ticker, since: "(no filings since entry)",
        now_fn=lambda: now,
    )

    await coordinator.run_tick()

    # No broker action.
    assert broker.submitted == []
    assert broker.canceled == []

    # New schedule row at now + 7d, reason=hold.
    sched = get_latest_thesis_schedule_for_execution(db_path, eid)
    assert sched is not None
    assert sched.scheduled_reason == "hold"
    expected_due = now + dt.timedelta(days=HOLD_RESCHEDULE_DAYS)
    # Allow tiny clock skew tolerance via SQLite's TIMESTAMP coercion.
    delta = abs((sched.due_at - expected_due).total_seconds())
    assert delta < 1.0

    # thesis_reviews row exists with decision=hold.
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT decision FROM thesis_reviews WHERE execution_id = ?",
            (eid,),
        ).fetchone()
    assert row is not None
    assert row["decision"] == "hold"


@pytest.mark.asyncio
async def test_thesis_review_close_path_submits_market_sell_and_cancels_gtcs(
    db_path: str, journal, prompt_builder
) -> None:
    """AC: Opus says `close` → market sell submitted → GTC stop + TP
    cancelled. No further schedule."""
    from analyzer.thesis_review import ThesisReviewer
    from app.thesis_coordinator import ThesisReviewCoordinator

    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    entry_at = now - dt.timedelta(days=15)
    _, eid = _setup_open_position(
        db_path, entry_at=entry_at, sonnet_pv=sonnet_pv
    )
    insert_thesis_review_schedule(
        db_path,
        ThesisReviewScheduleRow(
            execution_id=eid,
            due_at=now - dt.timedelta(hours=1),
            scheduled_reason="entry",
        ),
    )

    broker = _FakeBrokerWithCancelTracking(last_price=120.0)
    fake_anth = _FakeAnthropicForThesis(db_path, _close_response())
    reviewer = ThesisReviewer(
        client=fake_anth,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        db_path=db_path,
    )
    coordinator = ThesisReviewCoordinator(
        journal=journal,
        broker=broker,
        reviewer=reviewer,
        filings_lookup=lambda *, issuer_ticker, since: "(no filings since entry)",
        now_fn=lambda: now,
    )

    await coordinator.run_tick()

    # Market sell submitted for entry.qty=100.
    assert len(broker.submitted) == 1
    sell = broker.submitted[0]
    assert sell.side == "sell"
    assert sell.order_type == "market"
    assert sell.tif == "day"
    assert sell.qty == 100
    assert sell.symbol == "ACME"

    # Both GTC legs cancelled (stop + take_profit).
    assert sorted(broker.canceled) == sorted(["orig-stop-ACME", "orig-tp-ACME"])

    # SAFETY ORDERING: market sell MUST be submitted before either GTC
    # cancel. Two separate lists (`submitted` and `canceled`) couldn't
    # catch a bug that called cancel-then-submit. `call_log` records
    # mutating ops in invocation order — assert positional precedence.
    submit_idx = next(
        i for i, (op, payload) in enumerate(broker.call_log)
        if op == "submit"
        and payload.get("client_order_id") == f"thesis-close-exec-{eid}"
    )
    stop_cancel_idx = next(
        i for i, (op, payload) in enumerate(broker.call_log)
        if op == "cancel" and payload.get("id") == "orig-stop-ACME"
    )
    tp_cancel_idx = next(
        i for i, (op, payload) in enumerate(broker.call_log)
        if op == "cancel" and payload.get("id") == "orig-tp-ACME"
    )
    assert submit_idx < stop_cancel_idx, (
        "close path must submit market sell BEFORE cancelling the GTC stop"
    )
    assert submit_idx < tp_cancel_idx, (
        "close path must submit market sell BEFORE cancelling the GTC take-profit"
    )

    # No follow-up schedule (close → done).
    from journal.repo import get_latest_thesis_schedule_for_execution
    sched = get_latest_thesis_schedule_for_execution(db_path, eid)
    assert sched is not None
    assert sched.scheduled_reason == "entry"  # still the original


@pytest.mark.asyncio
async def test_thesis_review_adjust_stop_uses_atomic_replace_no_cancel_no_resubmit(
    db_path: str, journal, prompt_builder
) -> None:
    """SAFETY-CRITICAL: stop replacement must be ATOMIC via the broker's
    `replace_stop_order` call (Alpaca PATCH /v2/orders/{id}). The
    coordinator MUST NOT do a plain submit-then-cancel sequence — that
    leaves either a gap (uncovered) or two live stops.

    Asserts:
    - Exactly ONE `replace` call against the original stop's id.
    - ZERO `submit` calls (no new stop order created via POST).
    - ZERO `cancel` calls against the original stop (the replace makes
      cancel unnecessary; Alpaca handles the swap atomically).
    - The take_profit order is left alone."""
    from analyzer.thesis_review import ThesisReviewer
    from app.thesis_coordinator import ThesisReviewCoordinator

    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    entry_at = now - dt.timedelta(days=15)
    _, eid = _setup_open_position(
        db_path,
        entry_at=entry_at,
        stop_loss_price=90.0,
        sonnet_pv=sonnet_pv,
    )
    insert_thesis_review_schedule(
        db_path,
        ThesisReviewScheduleRow(
            execution_id=eid,
            due_at=now - dt.timedelta(hours=1),
            scheduled_reason="entry",
        ),
    )

    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    new_stop = 105.0
    fake_anth = _FakeAnthropicForThesis(db_path, _adjust_stop_response(new_stop))
    reviewer = ThesisReviewer(
        client=fake_anth,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        db_path=db_path,
    )
    coordinator = ThesisReviewCoordinator(
        journal=journal,
        broker=broker,
        reviewer=reviewer,
        filings_lookup=lambda *, issuer_ticker, since: "(no filings since entry)",
        now_fn=lambda: now,
    )

    await coordinator.run_tick()

    # ONE atomic replace against the original stop's id.
    assert len(broker.replaced) == 1
    rep = broker.replaced[0]
    assert rep["old_id"] == "orig-stop-ACME"
    assert rep["new_stop_price"] == new_stop

    # No fresh submit_order calls — replace is atomic, not submit-then-cancel.
    assert broker.submitted == [], (
        "adjust_stop must NOT POST a new stop order; replace is atomic"
    )
    # No cancel calls at all — atomic replace makes cancel unnecessary.
    assert broker.canceled == [], (
        "adjust_stop must NOT cancel the old stop; replace handles the swap"
    )
    # call_log shape: only one mutating op, the replace.
    mutating_ops = [
        op for op, _ in broker.call_log
        if op in ("submit", "cancel", "replace")
    ]
    assert mutating_ops == ["replace"]

    # SAFETY ORDERING (defense-in-depth): no cancel of the original
    # stop appears anywhere in the log. A buggy implementation that
    # did cancel(old) → submit(new) instead of atomic replace would
    # show a `cancel` op for orig-stop-ACME here. The earlier asserts
    # already cover this via `broker.canceled == []`, but pinning it
    # against `call_log` makes the regression test explicit.
    cancel_ops = [
        payload for op, payload in broker.call_log if op == "cancel"
    ]
    assert cancel_ops == [], (
        "atomic replace must NOT be implemented as cancel-then-submit"
    )


@pytest.mark.asyncio
async def test_thesis_review_adjust_stop_keeps_old_stop_when_replace_fails(
    db_path: str, journal, prompt_builder
) -> None:
    """SAFETY: if the atomic replace fails (broker rejects), the OLD
    stop is still live at the broker — Alpaca's PATCH semantics
    guarantee the original is unchanged on failure. The coordinator
    must NOT cancel the old stop. Test pins the never-uncovered
    invariant."""
    from analyzer.thesis_review import ThesisReviewer
    from app.thesis_coordinator import ThesisReviewCoordinator

    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    entry_at = now - dt.timedelta(days=15)
    _, eid = _setup_open_position(
        db_path, entry_at=entry_at, sonnet_pv=sonnet_pv
    )
    insert_thesis_review_schedule(
        db_path,
        ThesisReviewScheduleRow(
            execution_id=eid,
            due_at=now - dt.timedelta(hours=1),
            scheduled_reason="entry",
        ),
    )

    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    broker.fail_next_replace()
    # §3.3 step 5: after a failed PATCH the coordinator interrogates the
    # target via get_order_by_id. Seed the original stop as STILL LIVE
    # (the transient-failure case) plus an open broker position.
    broker.seed_order(_live_order("orig-stop-ACME", order_type="stop", stop_price=90.0))
    broker.positions = [_broker_position("ACME", qty=100)]
    fake_anth = _FakeAnthropicForThesis(db_path, _adjust_stop_response(105.0))
    reviewer = ThesisReviewer(
        client=fake_anth,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        db_path=db_path,
    )
    coordinator = ThesisReviewCoordinator(
        journal=journal,
        broker=broker,
        reviewer=reviewer,
        filings_lookup=lambda *, issuer_ticker, since: "(no filings since entry)",
        now_fn=lambda: now,
    )

    await coordinator.run_tick()

    # Replace was attempted and failed; coordinator did NOT compensate
    # by submitting a new stop or cancelling the old one. Original stop
    # remains the sole protective leg at the broker.
    assert broker.replaced == []
    assert broker.submitted == []
    assert broker.canceled == []
    assert broker.submitted_oco == []
    # The failure-path is recorded.
    assert any(op == "replace_failed" for op, _ in broker.call_log)


@pytest.mark.asyncio
async def test_thesis_review_skipped_when_position_already_closed(
    db_path: str, journal, prompt_builder
) -> None:
    """AC: stop or TP firing BEFORE the review date → review never runs.
    Modeled here by inserting a positions snapshot with qty=0 (the
    reconciler-truth state after a stop fire)."""
    from analyzer.thesis_review import ThesisReviewer
    from app.thesis_coordinator import ThesisReviewCoordinator

    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    entry_at = now - dt.timedelta(days=15)
    _, eid = _setup_open_position(
        db_path, entry_at=entry_at, sonnet_pv=sonnet_pv
    )
    # Simulate stop firing — latest snapshot is qty=0.
    insert_position(
        db_path,
        PositionRow(
            snapshot_at=entry_at + dt.timedelta(days=10),
            source="reconciler",
            symbol="ACME",
            qty=0,
            avg_entry_price=100.0,
            market_value=0.0,
            unrealized_pnl=0.0,
        ),
    )
    insert_thesis_review_schedule(
        db_path,
        ThesisReviewScheduleRow(
            execution_id=eid,
            due_at=now - dt.timedelta(hours=1),
            scheduled_reason="entry",
        ),
    )

    broker = _FakeBrokerWithCancelTracking(last_price=80.0)
    fake_anth = _FakeAnthropicForThesis(db_path, _hold_response())
    reviewer = ThesisReviewer(
        client=fake_anth,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        db_path=db_path,
    )
    coordinator = ThesisReviewCoordinator(
        journal=journal,
        broker=broker,
        reviewer=reviewer,
        filings_lookup=lambda *, issuer_ticker, since: "(no filings since entry)",
        now_fn=lambda: now,
    )

    await coordinator.run_tick()

    # No Opus call (we don't pay for a review on a closed position).
    assert fake_anth.calls == []
    # No broker action.
    assert broker.submitted == []
    assert broker.canceled == []
    # No thesis_reviews row written.
    with connect(db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM thesis_reviews WHERE execution_id = ?",
            (eid,),
        ).fetchone()[0]
    assert n == 0


@pytest.mark.asyncio
async def test_entry_schedules_thesis_review_at_horizon(
    db_path: str,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """When the agent loop submits an accepted trade, it must write a
    `thesis_review_schedule` row at `now + time_horizon_days`. This pins
    the contract the coordinator depends on.
    """
    import asyncio
    from app.loop import AgentLoop
    from app.state import AgentState
    from tests.integration.conftest import (
        _opus_ratify_json,
        _valid_proposal_json,
    )

    # Use the standard smoke fixture: one valid 8-K → analyzer → Opus →
    # accepted trade.
    f = make_filing(accession="0001234567-26-thesis-entry", cik=320193)
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()

    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=8)],
        "review": [_opus_ratify_json()],
    }
    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    runner = asyncio.create_task(loop.run())

    async def _wait(predicate, timeout: float = 10.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("predicate timed out")

    await _wait(lambda: queue.empty() and loop.state == AgentState.IDLE)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    await asyncio.wait_for(runner, timeout=5.0)

    # A thesis_review_schedule row was written; due_at = entry + 14d.
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT execution_id, due_at, scheduled_reason "
            "FROM thesis_review_schedule"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["scheduled_reason"] == "entry"
    # The proposal payload's time_horizon_days is 14 in `_valid_proposal_json`,
    # so the due date must be ~14 days out from the entry submission time.
    # SQLite returns either a string or a datetime depending on adapters.
    due_raw = rows[0]["due_at"]
    if isinstance(due_raw, str):
        due = dt.datetime.fromisoformat(due_raw.replace(" ", "T"))
    else:
        due = due_raw
    if due.tzinfo is None:
        due = due.replace(tzinfo=dt.UTC)
    delta_days = (due - dt.datetime.now(dt.UTC)).days
    assert 13 <= delta_days <= 14


@pytest.mark.asyncio
async def test_thesis_review_skipped_for_rejected_executions(
    db_path: str, journal, prompt_builder
) -> None:
    """A schedule attached to a `rejected` execution must not fire — the
    SQL filter `e.decision = 'accepted'` enforces this."""
    from analyzer.thesis_review import ThesisReviewer
    from app.thesis_coordinator import ThesisReviewCoordinator

    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    entry_at = now - dt.timedelta(days=15)
    # Create a separate proposal for the rejected execution. UNIQUE on
    # (proposal_id) means we can't pile multiple executions on one row.
    fid = insert_filing(
        db_path,
        FilingRow(
            accession_number="rejected-acc-X",
            cik=999999,
            form_type="8-K",
            filed_at=entry_at - dt.timedelta(hours=2),
            fetched_at=entry_at - dt.timedelta(hours=1),
            raw_text_path="/tmp/rejected.txt",
            content_hash="hash-rejected",
            item_codes='["1.01"]',
            issuer_ticker="REJX",
        ),
    )
    rej_pid = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=fid,
            decision_id="rejected-d-X",
            model_id="claude-sonnet-4-6",
            prompt_version=sonnet_pv,
            raw_response="{}",
            kind="trade_proposal",
            symbol="REJX",
            direction="long",
            size_pct_requested=0.10,
            conviction=8,
            thesis="x",
            input_tokens=10,
            output_tokens=10,
            latency_ms=10,
            cost_usd=0.001,
        ),
    )
    # Schedule attached to the rejected execution.
    rejected_eid = insert_execution(
        db_path,
        ExecutionRow(
            proposal_id=rej_pid,
            decision="rejected",
            reject_reason="ks5_concurrent_limit",
            submitted_orders_json="[]",
        ),
    )
    insert_thesis_review_schedule(
        db_path,
        ThesisReviewScheduleRow(
            execution_id=rejected_eid,
            due_at=now - dt.timedelta(hours=1),
            scheduled_reason="entry",
        ),
    )

    broker = _FakeBrokerWithCancelTracking(last_price=100.0)
    fake_anth = _FakeAnthropicForThesis(db_path, _hold_response())
    reviewer = ThesisReviewer(
        client=fake_anth,
        prompt_builder=prompt_builder,
        opus_model_id="claude-opus-4-7",
        db_path=db_path,
    )
    coordinator = ThesisReviewCoordinator(
        journal=journal,
        broker=broker,
        reviewer=reviewer,
        filings_lookup=lambda *, issuer_ticker, since: "(no filings since entry)",
        now_fn=lambda: now,
    )

    await coordinator.run_tick()

    assert fake_anth.calls == []
    assert broker.submitted == []


# ---------------------------------------------------------------------------
# D-079 §3.3 — adjust_stop journal chain + step-5 fallback branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adjust_stop_journals_replacement_chain(
    db_path: str, journal, prompt_builder
) -> None:
    """§3.3 steps 3-4: success journals a NEW live stop row (role
    preserved, final_status NULL) and completes the old row as
    'replaced' via the §7.4 port; the §3.5 state pointer rotates to the
    new journal row when trailing is engaged."""
    from journal.exit_policy import (
        ExitPolicyStateRow,
        get_exit_policy_state,
        upsert_exit_policy_state,
    )

    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path, entry_at=now - dt.timedelta(days=15), sonnet_pv=sonnet_pv
    )
    _due_schedule(db_path, eid, now)
    old_stop_row = next(
        o for o in _orders_for_execution(db_path, eid) if o.role == "stop"
    )
    # Trailing engaged for the execution → the pointer must rotate.
    upsert_exit_policy_state(
        db_path,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=5.0,
            trail_engaged=True,
            high_water_mark=110.0,
            stop_order_journal_id=old_stop_row.id,
        ),
    )

    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    recorder = _OutcomeRecorder()
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_stop_response(105.0), now, recorder=recorder,
    )

    await coordinator.run_tick()

    assert len(broker.replaced) == 1
    rows = _orders_for_execution(db_path, eid)
    new_stops = [
        o for o in rows
        if o.role == "stop" and o.id != old_stop_row.id
    ]
    assert len(new_stops) == 1, "replacement must journal a fresh stop row"
    new_stop = new_stops[0]
    assert new_stop.stop_price == 105.0
    assert new_stop.final_status is None, "replacement row is the live leg"
    assert new_stop.tif == "gtc"
    assert old_stop_row.broker_order_id in (new_stop.notes or "")
    # Old row completed as 'replaced' through the §7.4 single-writer.
    assert recorder.calls == [(old_stop_row.id, "replaced")]
    # §3.5 step 4: ratchet pointer follows the live row.
    state = get_exit_policy_state(db_path, execution_id=eid)
    assert state is not None
    assert state.stop_order_journal_id == new_stop.id


@pytest.mark.asyncio
async def test_adjust_stop_rejects_stop_at_or_above_current_price(
    db_path: str, journal, prompt_builder
) -> None:
    """A long's stop must stay strictly below the current price — a
    stop at/above it is a `close` in disguise and would fire on the
    next tick. No broker mutation; retry rescheduled."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path, entry_at=now - dt.timedelta(days=15), sonnet_pv=sonnet_pv
    )
    _due_schedule(db_path, eid, now)

    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_stop_response(112.0), now,  # above last=110
    )

    await coordinator.run_tick()

    assert broker.replaced == []
    assert broker.submitted == []
    assert broker.canceled == []
    follow_ups = _pending_schedules(db_path, eid, now)
    assert len(follow_ups) == 1  # +1d retry


@pytest.mark.asyncio
async def test_adjust_stop_lost_race_when_old_stop_filled(
    db_path: str, journal, prompt_builder
) -> None:
    """§3.3 step 5, branch 1: PATCH failed because the stop FILLED.
    Not an error — fill ingestion records the exit. No re-place, no
    cancel, no reschedule (the position is closing)."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path, entry_at=now - dt.timedelta(days=15), sonnet_pv=sonnet_pv
    )
    _due_schedule(db_path, eid, now)

    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    broker.fail_next_replace()
    broker.seed_order(
        _live_order("orig-stop-ACME", status="filled", stop_price=90.0)
    )
    broker.positions = [_broker_position("ACME")]
    recorder = _OutcomeRecorder()
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_stop_response(105.0), now, recorder=recorder,
    )

    await coordinator.run_tick()

    assert broker.replaced == []
    assert broker.submitted == []
    assert broker.canceled == []
    assert broker.submitted_oco == []
    # Fill ingestion owns the outcome — the coordinator records nothing.
    assert recorder.calls == []
    assert _pending_schedules(db_path, eid, now) == []


@pytest.mark.asyncio
async def test_adjust_stop_replaces_dead_stop_as_oco_with_surviving_tp(
    db_path: str, journal, prompt_builder
) -> None:
    """§3.3 step 5, branch 2a: stop alone dead, TP leg still live →
    cancel the surviving TP and re-place BOTH as one OCO (two
    independent full-qty sells are impossible — §4.2 qty constraint).
    Asserts the broker-call arguments end-to-end."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path,
        entry_at=now - dt.timedelta(days=15),
        take_profit_price=150.0,
        sonnet_pv=sonnet_pv,
    )
    _due_schedule(db_path, eid, now)
    rows = _orders_for_execution(db_path, eid)
    old_stop_row = next(o for o in rows if o.role == "stop")
    old_tp_row = next(o for o in rows if o.role == "take_profit")

    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    broker.fail_next_replace()
    broker.seed_order(
        _live_order("orig-stop-ACME", status="expired", stop_price=90.0)
    )
    broker.seed_order(
        _live_order(
            "orig-tp-ACME", order_type="limit", status="accepted",
            limit_price=150.0,
        )
    )
    broker.positions = [_broker_position("ACME")]
    recorder = _OutcomeRecorder()
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_stop_response(105.0), now, recorder=recorder,
    )

    await coordinator.run_tick()

    # Surviving TP canceled, then exactly one OCO with the LLM's new
    # stop and the original TP price.
    assert broker.canceled == ["orig-tp-ACME"]
    assert len(broker.submitted_oco) == 1
    oco = broker.submitted_oco[0]
    assert oco["symbol"] == "ACME"
    assert oco["qty"] == 100
    assert oco["stop_price"] == 105.0
    assert oco["limit_price"] == 150.0
    # No plain submit_order — protection is re-placed as a pair.
    assert broker.submitted == []

    # Journal chain: fresh live stop + TP rows; old rows completed with
    # their actual dispositions.
    rows = _orders_for_execution(db_path, eid)
    new_stop = next(
        o for o in rows if o.role == "stop" and o.id != old_stop_row.id
    )
    new_tp = next(
        o for o in rows if o.role == "take_profit" and o.id != old_tp_row.id
    )
    assert new_stop.stop_price == 105.0 and new_stop.final_status is None
    assert new_tp.limit_price == 150.0 and new_tp.final_status is None
    assert sorted(recorder.calls) == sorted(
        [(old_stop_row.id, "expired"), (old_tp_row.id, "canceled")]
    )
    # Re-protected → next review at +7d.
    assert len(_pending_schedules(db_path, eid, now)) == 1


@pytest.mark.asyncio
async def test_adjust_stop_replaces_both_dead_legs_with_fresh_oco(
    db_path: str, journal, prompt_builder
) -> None:
    """§3.3 step 5, branch 2b: both legs dead (the day-2+ DAY-bracket
    case — they expired together) → fresh OCO, no cancel needed."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path,
        entry_at=now - dt.timedelta(days=15),
        take_profit_price=150.0,
        sonnet_pv=sonnet_pv,
    )
    _due_schedule(db_path, eid, now)

    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    broker.fail_next_replace()
    broker.seed_order(
        _live_order("orig-stop-ACME", status="expired", stop_price=90.0)
    )
    broker.seed_order(
        _live_order(
            "orig-tp-ACME", order_type="limit", status="expired",
            limit_price=150.0,
        )
    )
    broker.positions = [_broker_position("ACME")]
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_stop_response(105.0), now,
    )

    await coordinator.run_tick()

    assert broker.canceled == []  # nothing left to cancel
    assert len(broker.submitted_oco) == 1
    oco = broker.submitted_oco[0]
    assert oco["stop_price"] == 105.0
    assert oco["limit_price"] == 150.0


@pytest.mark.asyncio
async def test_adjust_stop_replaces_dead_stop_plain_gtc_when_no_tp(
    db_path: str, journal, prompt_builder
) -> None:
    """§3.3 step 5, branch 2c: no-TP proposal whose stop died → plain
    GTC stop re-place (submit_oco_sell with limit_price=None)."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path,
        entry_at=now - dt.timedelta(days=15),
        take_profit_price=None,
        sonnet_pv=sonnet_pv,
    )
    _due_schedule(db_path, eid, now)

    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    broker.fail_next_replace()
    broker.seed_order(
        _live_order("orig-stop-ACME", status="canceled", stop_price=90.0)
    )
    broker.positions = [_broker_position("ACME")]
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_stop_response(105.0), now,
    )

    await coordinator.run_tick()

    assert broker.canceled == []
    assert len(broker.submitted_oco) == 1
    oco = broker.submitted_oco[0]
    assert oco["stop_price"] == 105.0
    assert oco["limit_price"] is None


@pytest.mark.asyncio
async def test_adjust_stop_noop_when_position_gone_at_broker(
    db_path: str, journal, prompt_builder
) -> None:
    """§3.3 step 5, branch 3: position absent at the broker → no-op;
    the §2.2 tombstone path owns it. No re-place, no reschedule."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path, entry_at=now - dt.timedelta(days=15), sonnet_pv=sonnet_pv
    )
    _due_schedule(db_path, eid, now)

    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    broker.fail_next_replace()
    broker.seed_order(
        _live_order("orig-stop-ACME", status="canceled", stop_price=90.0)
    )
    broker.positions = []  # gone at broker
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_stop_response(105.0), now,
    )

    await coordinator.run_tick()

    assert broker.submitted_oco == []
    assert broker.canceled == []
    assert broker.submitted == []
    assert _pending_schedules(db_path, eid, now) == []


# ---------------------------------------------------------------------------
# D-079 §3.4 — adjust_take_profit (raise-only, atomic replace_limit_order)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adjust_tp_uses_atomic_replace_limit_order(
    db_path: str, journal, prompt_builder
) -> None:
    """§3.4 happy path: ONE atomic replace_limit_order against the live
    TP's broker id with the new price; stop leg untouched; journal
    chain mirrors §3.3 (new live row + old row 'replaced')."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path,
        entry_at=now - dt.timedelta(days=15),
        take_profit_price=150.0,
        sonnet_pv=sonnet_pv,
    )
    _due_schedule(db_path, eid, now)
    old_tp_row = next(
        o for o in _orders_for_execution(db_path, eid) if o.role == "take_profit"
    )

    broker = _FakeBrokerWithCancelTracking(last_price=145.0)
    recorder = _OutcomeRecorder()
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_tp_response(180.0), now, recorder=recorder,
    )

    await coordinator.run_tick()

    # Exactly one atomic limit replace with the right arguments.
    assert len(broker.replaced_limits) == 1
    rep = broker.replaced_limits[0]
    assert rep["old_id"] == "orig-tp-ACME"
    assert rep["new_limit_price"] == 180.0
    # The stop leg and everything else untouched.
    assert broker.replaced == []
    assert broker.submitted == []
    assert broker.canceled == []
    mutating = [
        op for op, _ in broker.call_log
        if op in ("submit", "cancel", "replace", "replace_limit", "submit_oco")
    ]
    assert mutating == ["replace_limit"]

    # Journal chain: fresh live TP row, old completed as replaced.
    rows = _orders_for_execution(db_path, eid)
    new_tp = next(
        o for o in rows if o.role == "take_profit" and o.id != old_tp_row.id
    )
    assert new_tp.limit_price == 180.0
    assert new_tp.final_status is None
    assert new_tp.tif == "gtc"
    assert old_tp_row.broker_order_id in (new_tp.notes or "")
    assert recorder.calls == [(old_tp_row.id, "replaced")]
    # Next review at +7d.
    assert len(_pending_schedules(db_path, eid, now)) == 1


@pytest.mark.asyncio
async def test_adjust_tp_raise_only_rejects_tp_at_or_below_current_price(
    db_path: str, journal, prompt_builder
) -> None:
    """§3.4: raise-only — a new TP at/below the current quote is
    rejected with no broker mutation (lowering a TP is expressible as
    `close`; a lowerable TP reintroduces S-4's tighten-only pressure)."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path,
        entry_at=now - dt.timedelta(days=15),
        take_profit_price=150.0,
        sonnet_pv=sonnet_pv,
    )
    _due_schedule(db_path, eid, now)

    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_tp_response(105.0), now,  # below last=110
    )

    await coordinator.run_tick()

    assert broker.replaced_limits == []
    assert broker.replaced == []
    assert broker.submitted == []
    assert broker.canceled == []
    # +1d retry written.
    assert len(_pending_schedules(db_path, eid, now)) == 1


@pytest.mark.asyncio
async def test_adjust_tp_keeps_old_tp_when_replace_fails(
    db_path: str, journal, prompt_builder
) -> None:
    """§3.4 safety: PATCH failure leaves the original TP live (Alpaca
    contract). No compensation, no cancel; retry tomorrow."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path,
        entry_at=now - dt.timedelta(days=15),
        take_profit_price=150.0,
        sonnet_pv=sonnet_pv,
    )
    _due_schedule(db_path, eid, now)

    broker = _FakeBrokerWithCancelTracking(last_price=145.0)
    broker.fail_next_replace_limit()
    recorder = _OutcomeRecorder()
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_tp_response(180.0), now, recorder=recorder,
    )

    await coordinator.run_tick()

    assert broker.replaced_limits == []
    assert broker.submitted == []
    assert broker.canceled == []
    assert any(op == "replace_limit_failed" for op, _ in broker.call_log)
    assert recorder.calls == []
    # No new TP row journaled — the original row is still the live leg.
    tp_rows = [
        o for o in _orders_for_execution(db_path, eid)
        if o.role == "take_profit"
    ]
    assert len(tp_rows) == 1
    # +1d retry.
    assert len(_pending_schedules(db_path, eid, now)) == 1


@pytest.mark.asyncio
async def test_adjust_tp_skipped_when_no_live_tp_leg(
    db_path: str, journal, prompt_builder
) -> None:
    """§3.4: the raise actuator exists for the OCO TP leg — a no-TP
    position has nothing to raise. Treated as hold (+7d), no broker
    calls."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path,
        entry_at=now - dt.timedelta(days=15),
        take_profit_price=None,
        sonnet_pv=sonnet_pv,
    )
    _due_schedule(db_path, eid, now)

    broker = _FakeBrokerWithCancelTracking(last_price=145.0)
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_tp_response(180.0), now,
    )

    await coordinator.run_tick()

    assert broker.replaced_limits == []
    assert broker.submitted == []
    assert broker.canceled == []
    assert len(_pending_schedules(db_path, eid, now)) == 1


# ---------------------------------------------------------------------------
# D-087 — adjust_stop_skipped must NOT orphan an open position from
# thesis re-review, and a boot sweep re-arms any already-orphaned slot.
# ---------------------------------------------------------------------------


def _complete_live_stop(db_path: str, execution_id: int) -> None:
    """Mark the position's journaled stop terminal so the coordinator's
    `_get_live_protective_order(roles=('stop','trailing_stop'))` returns
    None — the precondition that drives the `adjust_stop_skipped`
    branch (no journaled live stop, e.g. a pre-WS3-conversion PDT-era
    position)."""
    from journal.repo import record_order_outcome

    stop = next(
        o for o in _orders_for_execution(db_path, execution_id) if o.role == "stop"
    )
    assert stop.id is not None
    record_order_outcome(
        db_path, stop.id, "expired", fill_price=None, fill_qty=None, fill_at=None
    )


@pytest.mark.asyncio
async def test_adjust_stop_skipped_reschedules_when_position_still_open(
    db_path: str, journal, prompt_builder
) -> None:
    """D-087 regression: when adjust_stop can't act because no live stop
    is journaled (PDT-era position pre-WS3 conversion) but the position
    is STILL OPEN, the coordinator must re-arm a +1d retry instead of
    silently returning — the prod orphaning bug. No broker mutation."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path, entry_at=now - dt.timedelta(days=15), sonnet_pv=sonnet_pv
    )
    _complete_live_stop(db_path, eid)  # no journaled live stop remains
    _due_schedule(db_path, eid, now)

    # Position is still open per the latest journal snapshot (qty=100
    # from _setup_open_position) — has_open_position() returns True.
    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_stop_response(105.0), now,
    )

    await coordinator.run_tick()

    # No broker action on the skipped branch.
    assert broker.replaced == []
    assert broker.submitted == []
    assert broker.canceled == []
    assert broker.submitted_oco == []
    # The position is re-armed: exactly one +1d follow-up, reason adjust_stop.
    follow_ups = _pending_schedules(db_path, eid, now)
    assert len(follow_ups) == 1
    assert follow_ups[0]["scheduled_reason"] == "adjust_stop"
    due = follow_ups[0]["due_at"]
    if isinstance(due, str):
        due = dt.datetime.fromisoformat(due.replace(" ", "T"))
    if due.tzinfo is None:
        due = due.replace(tzinfo=dt.UTC)
    assert abs((due - (now + dt.timedelta(days=1))).total_seconds()) < 1.0


@pytest.mark.asyncio
async def test_adjust_stop_skipped_stays_terminal_when_position_closed(
    db_path: str, journal, prompt_builder
) -> None:
    """D-087: the skipped re-arm fires ONLY for still-open positions. If
    the slot has since closed (latest snapshot qty=0), there is nothing
    left to review — stay terminal, write no follow-up schedule."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path, entry_at=now - dt.timedelta(days=15), sonnet_pv=sonnet_pv
    )
    _complete_live_stop(db_path, eid)
    _due_schedule(db_path, eid, now)
    # Slot closed AFTER entry: latest positions snapshot is qty=0. Note
    # this also clears the run_tick open-position guard, so the review
    # never runs — but the unit-level branch is still pinned because
    # has_open_position() is the same predicate either way.
    insert_position(
        db_path,
        PositionRow(
            snapshot_at=now - dt.timedelta(hours=1),
            source="reconciler",
            symbol="ACME",
            qty=0,
            avg_entry_price=100.0,
            market_value=0.0,
            unrealized_pnl=0.0,
        ),
    )

    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_stop_response(105.0), now,
    )

    await coordinator.run_tick()

    assert broker.replaced == []
    assert broker.submitted == []
    # No follow-up schedule: a closed slot must not be re-armed.
    assert _pending_schedules(db_path, eid, now) == []


@pytest.mark.asyncio
async def test_adjust_stop_lost_race_and_position_gone_stay_terminal(
    db_path: str, journal, prompt_builder
) -> None:
    """D-087 guard: the §3.3 step-5 fallback's terminal branches must
    NOT acquire a reschedule from the skipped-branch fix. A filled old
    stop (fill ingestion owns the exit) and an absent broker position
    (the tombstone path owns it) both stay terminal — no follow-up."""
    sonnet_pv, _ = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)

    # Branch 1 — lost race: PATCH failed because the stop FILLED.
    _, eid_race = _setup_open_position(
        db_path, entry_at=now - dt.timedelta(days=15), sonnet_pv=sonnet_pv
    )
    _due_schedule(db_path, eid_race, now)
    broker = _FakeBrokerWithCancelTracking(last_price=110.0)
    broker.fail_next_replace()
    broker.seed_order(
        _live_order("orig-stop-ACME", status="filled", stop_price=90.0)
    )
    broker.positions = [_broker_position("ACME")]
    coordinator = _make_coordinator(
        db_path, journal, prompt_builder, broker,
        _adjust_stop_response(105.0), now,
    )
    await coordinator.run_tick()
    assert _pending_schedules(db_path, eid_race, now) == []

    # Branch 3 — position gone at the broker. Use a distinct symbol so
    # the journal open-position guard (latest positions snapshot) and the
    # broker truth agree the slot is gone for THIS execution.
    fid = insert_filing(
        db_path,
        FilingRow(
            accession_number="gone-acc-X",
            cik=789019,
            form_type="8-K",
            filed_at=now - dt.timedelta(days=15, hours=2),
            fetched_at=now - dt.timedelta(days=15, hours=1),
            raw_text_path="/tmp/gone.txt",
            content_hash="hash-gone",
            item_codes='["1.01"]',
            issuer_ticker="WIDG",
        ),
    )
    payload = {
        "symbol": "WIDG", "direction": "long", "size_pct_of_capital": 0.10,
        "entry_style": "market_open", "stop_loss_price": 90.0,
        "take_profit_price": 150.0, "time_horizon_days": 14, "conviction": 8,
        "thesis": "WIDG catalyst.", "signals": ["Item 1.01"],
        "exit_conditions": ["x"], "risk_factors": ["y"],
    }
    pid = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=fid, decision_id="gone-d-X", model_id="claude-sonnet-4-6",
            prompt_version=sonnet_pv, raw_response=json.dumps(payload),
            kind="trade_proposal", symbol="WIDG", direction="long",
            size_pct_requested=0.10, conviction=8, thesis="WIDG catalyst.",
            input_tokens=10, output_tokens=10, latency_ms=10, cost_usd=0.001,
        ),
    )
    eid_gone = insert_execution(
        db_path,
        ExecutionRow(
            proposal_id=pid, decision="accepted", realized_size_pct=0.10,
            realized_dollar_size=10000.0, submitted_orders_json="[]",
        ),
    )
    insert_order(
        db_path,
        OrderRow(
            execution_id=eid_gone, role="entry", symbol="WIDG", side="buy",
            order_type="market", qty=100, tif="day",
            broker_order_id="orig-entry-WIDG", submitted_at=now - dt.timedelta(days=15),
            realized_fill_price=100.0, realized_fill_qty=100,
            realized_fill_at=now - dt.timedelta(days=15), final_status="filled",
        ),
    )
    insert_order(
        db_path,
        OrderRow(
            execution_id=eid_gone, role="stop", symbol="WIDG", side="sell",
            order_type="stop", qty=100, tif="gtc", stop_price=90.0,
            broker_order_id="orig-stop-WIDG",
            submitted_at=now - dt.timedelta(days=15), final_status=None,
        ),
    )
    # Open per journal so run_tick proceeds to the review.
    insert_position(
        db_path,
        PositionRow(
            snapshot_at=now - dt.timedelta(days=15), source="reconciler",
            symbol="WIDG", qty=100, avg_entry_price=100.0,
            market_value=11000.0, unrealized_pnl=1000.0,
        ),
    )
    _due_schedule(db_path, eid_gone, now)

    broker2 = _FakeBrokerWithCancelTracking(last_price=110.0)
    broker2.fail_next_replace()
    broker2.seed_order(
        _live_order("orig-stop-WIDG", status="canceled", stop_price=90.0, symbol="WIDG")
    )
    broker2.positions = []  # gone at broker
    coordinator2 = _make_coordinator(
        db_path, journal, prompt_builder, broker2,
        _adjust_stop_response(105.0), now,
    )
    await coordinator2.run_tick()
    assert _pending_schedules(db_path, eid_gone, now) == []


@pytest.mark.asyncio
async def test_boot_rearm_sweep_arms_orphaned_open_position_idempotently(
    db_path: str,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    prompt_builder,
) -> None:
    """D-087: the boot sweep re-arms a due-now thesis review for any open
    BROKER position whose latest schedule already fired without a
    successor (the orphaned state), and is idempotent across boots."""
    import asyncio

    from app.loop import AgentLoop
    from journal.models import ThesisReviewRow
    from journal.repo import insert_thesis_review

    sonnet_pv, thesis_pv = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path, entry_at=now - dt.timedelta(days=20), sonnet_pv=sonnet_pv
    )
    # Orphan it: the entry schedule fired (a thesis_reviews row exists)
    # but no successor schedule was written — exactly the adjust_stop_
    # skipped early-return outcome.
    sched_id = insert_thesis_review_schedule(
        db_path,
        ThesisReviewScheduleRow(
            execution_id=eid,
            due_at=now - dt.timedelta(days=6),
            scheduled_reason="entry",
        ),
    )
    insert_thesis_review(
        db_path,
        ThesisReviewRow(
            execution_id=eid,
            schedule_id=sched_id,
            model_id="claude-opus-4-7",
            prompt_version=thesis_pv,
            decision="adjust_stop",
            raw_response="{}",
            rationale="fired but orphaned",
            input_tokens=1, output_tokens=1, latency_ms=1, cost_usd=0.0,
        ),
    )
    # Sanity: no PENDING schedule remains for this execution.
    assert _pending_schedules(db_path, eid, now - dt.timedelta(days=30)) != []
    from journal.repo import (
        find_accepted_executions_without_pending_thesis_review,
    )
    assert (eid, "ACME") in find_accepted_executions_without_pending_thesis_review(
        db_path
    )

    # The broker reports the position open (source of truth for the sweep).
    fake_broker.positions = [_broker_position("ACME", qty=100)]

    queue: asyncio.Queue = asyncio.Queue()
    components = build_components(queue=queue)
    loop = AgentLoop(components=components)

    def _open_schedule_count() -> int:
        with connect(db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM thesis_review_schedule s "
                "WHERE s.execution_id = ? AND NOT EXISTS ("
                "  SELECT 1 FROM thesis_reviews tr WHERE tr.schedule_id = s.id)",
                (eid,),
            ).fetchone()[0]

    assert _open_schedule_count() == 0  # orphaned: zero pending

    await loop._boot()  # noqa: SLF001 — boot seam under test

    # Exactly one fresh, due-now schedule was armed.
    assert _open_schedule_count() == 1
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT scheduled_reason FROM thesis_review_schedule s "
            "WHERE s.execution_id = ? AND NOT EXISTS ("
            "  SELECT 1 FROM thesis_reviews tr WHERE tr.schedule_id = s.id) "
            "ORDER BY s.id DESC LIMIT 1",
            (eid,),
        ).fetchone()
    assert row["scheduled_reason"] == "rearm"

    # Idempotent: a second boot does NOT stack another schedule (the
    # execution now has a pending review, so the query excludes it).
    await loop._boot()  # noqa: SLF001
    assert _open_schedule_count() == 1


@pytest.mark.asyncio
async def test_boot_rearm_sweep_skips_closed_positions(
    db_path: str,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    prompt_builder,
) -> None:
    """D-087: the sweep is driven by BROKER truth — an orphaned
    execution whose position is NOT open at the broker is left alone, so
    a closed slot is never re-armed even if a stale journal row lingers."""
    import asyncio

    from app.loop import AgentLoop
    from journal.models import ThesisReviewRow
    from journal.repo import insert_thesis_review

    sonnet_pv, thesis_pv = _seed_prompts(db_path, prompt_builder)
    now = dt.datetime(2026, 5, 6, 14, 0, 0, tzinfo=dt.UTC)
    _, eid = _setup_open_position(
        db_path, entry_at=now - dt.timedelta(days=20), sonnet_pv=sonnet_pv
    )
    sched_id = insert_thesis_review_schedule(
        db_path,
        ThesisReviewScheduleRow(
            execution_id=eid,
            due_at=now - dt.timedelta(days=6),
            scheduled_reason="entry",
        ),
    )
    insert_thesis_review(
        db_path,
        ThesisReviewRow(
            execution_id=eid, schedule_id=sched_id, model_id="claude-opus-4-7",
            prompt_version=thesis_pv, decision="hold", raw_response="{}",
            rationale="fired", input_tokens=1, output_tokens=1,
            latency_ms=1, cost_usd=0.0,
        ),
    )

    # Broker has NO open position for ACME → must not be re-armed.
    fake_broker.positions = []

    queue: asyncio.Queue = asyncio.Queue()
    components = build_components(queue=queue)
    loop = AgentLoop(components=components)

    await loop._boot()  # noqa: SLF001

    with connect(db_path) as conn:
        pending = conn.execute(
            "SELECT COUNT(*) FROM thesis_review_schedule s "
            "WHERE s.execution_id = ? AND NOT EXISTS ("
            "  SELECT 1 FROM thesis_reviews tr WHERE tr.schedule_id = s.id)",
            (eid,),
        ).fetchone()[0]
    assert pending == 0
