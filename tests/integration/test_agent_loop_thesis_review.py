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

    def fail_next_submit(self) -> None:
        self._submit_should_fail = True

    def fail_next_cancel(self) -> None:
        self._cancel_should_fail = True

    def fail_next_replace(self) -> None:
        self._replace_should_fail = True

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
        return []

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
            final_status="accepted",
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
                final_status="accepted",
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
    thesis_pv = prompt_builder.prompt_version("opus_thesis_review_v1")
    insert_prompt(
        db,
        PromptRow(
            prompt_version=thesis_pv,
            name="opus_thesis_review_v1",
            file_path="src/prompts/opus_thesis_review_v1.txt",
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
