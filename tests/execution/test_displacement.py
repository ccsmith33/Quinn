"""DISPLACEMENT — evict the weakest unarmed position to fund a
high-conviction entry (execution/displacement.py).

Money-path tests: victim qualification (unarmed + aged + sub-cutoff, with
the graduated conviction-9 gain-cutoff exemption), deterministic
weakest-first pick, sell-before-buy ordering, the one-per-trading-day
hard cap, broker-failure abort, and config validation. The engine runs
against a real migrated journal DB; the broker is the only fake.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from broker.protocol import (
    AccountSnapshot,
    OrderRequest,
    Position,
    Quote,
    SubmittedOrder,
)
from config.loader import ExecutionConfig
from execution.displacement import (
    DISPLACEMENT_NOTES_PREFIX,
    DisplacementEngine,
    displaced_today,
    trading_days_held,
)
from execution.sizing import SizingAccepted, SizingEngine, SizingRejected
from journal.exit_policy import ExitPolicyStateRow, upsert_exit_policy_state
from journal.migrate import apply_migrations
from journal.models import (
    ExecutionRow,
    FilingRow,
    OrderRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    connect,
    insert_execution,
    insert_filing,
    insert_order,
    insert_prompt,
    insert_proposal,
)
from proposal.schemas import TradeProposal

# Tuesday 2026-06-09 11:00 ET, market hours. No NYSE holidays fall in the
# 14-calendar-day look-back used by "aged" victims (Memorial Day 05-25 is
# just outside it).
NOW = dt.datetime(2026, 6, 9, 15, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = str(tmp_path / "journal.db")
    apply_migrations(p)
    return p


class _RecordingBroker:
    """submit_order / cancel_order recorder with a one-shot failure arm."""

    def __init__(self) -> None:
        self.submitted: list[OrderRequest] = []
        self.canceled: list[str] = []
        self.fail_next_submit = False
        self._n = 0

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        if self.fail_next_submit:
            self.fail_next_submit = False
            raise RuntimeError("simulated broker outage")
        self.submitted.append(req)
        self._n += 1
        return SubmittedOrder(
            broker_order_id=f"disp-{self._n:04d}",
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            order_type=req.order_type,
            status="accepted",
            submitted_at=NOW,
        )

    def cancel_order(self, broker_order_id: str) -> None:
        self.canceled.append(broker_order_id)


def _cfg(**overrides: Any) -> ExecutionConfig:
    base: dict[str, Any] = {
        "broker_mode": "paper",
        "ks4_pct_cap": 0.20,
        "ks4_absolute_cap_usd": 1000.0,
        "ks5_max_concurrent": 10,
        "ks7_cash_reserve_pct": 0.03,
        "sizing_mid_pct": 0.07,
        "sizing_high_pct": 0.10,
        "displacement_enabled": True,
        "displacement_min_conviction": 8,
        "displacement_proceeds_haircut": 0.90,
        "displacement_victim_max_gain_pct": 5.0,
    }
    base.update(overrides)
    return ExecutionConfig(**base)


def _proposal(*, symbol: str = "ACME", conviction: int = 8) -> TradeProposal:
    return TradeProposal.model_validate(
        {
            "symbol": symbol,
            "direction": "long",
            "size_pct_of_capital": 0.10,
            "entry_style": "market_open",
            "stop_loss_price": 4.50,
            "time_horizon_days": 10,
            "conviction": conviction,
            "thesis": (
                "Strong fundamentals with material 8-K disclosure "
                "indicating a near-term catalyst per filing analysis."
            ),
            "signals": ["8-K item 2.02 strong earnings beat"],
            "exit_conditions": ["thesis breaks; stop hit; 30 days elapsed"],
            "risk_factors": ["macro risk; sector rotation; news risk"],
        }
    )


def _account(*, equity: float = 10_000.0, cash: float = 350.0) -> AccountSnapshot:
    # cash 350 vs reserve floor 300 (3% of 10k): 50 of headroom < one
    # share at 60 → the ks7 path rejects without a displacement credit.
    return AccountSnapshot(
        equity=equity,
        cash=cash,
        buying_power=cash,
        long_market_value=equity - cash,
        daypl=0.0,
        snapshot_at=NOW,
    )


def _quote(*, last: float = 60.0) -> Quote:
    return Quote(symbol="ACME", bid=last - 0.02, ask=last + 0.02, last=last, ts=NOW)


def _position(
    symbol: str, *, qty: int = 20, avg_entry: float = 100.0, unrealized_pct: float
) -> Position:
    cost = avg_entry * qty
    pnl = cost * unrealized_pct / 100.0
    return Position(
        symbol=symbol,
        qty=qty,
        avg_entry_price=avg_entry,
        market_value=cost + pnl,
        unrealized_pnl=pnl,
    )


def _seed_victim(
    db: str,
    *,
    symbol: str,
    entry_at: dt.datetime,
    conviction: int = 7,
    armed: bool = False,
    live_stop: bool = True,
) -> int:
    """filing → prompt → proposal → accepted execution → filled entry
    (+ optional live GTC stop, + optional armed trail state). Returns the
    execution id."""
    accession = f"disp-acc-{symbol}"
    fid = insert_filing(
        db,
        FilingRow(
            accession_number=accession,
            cik=hash(symbol) % 999_999,
            form_type="8-K",
            filed_at=entry_at - dt.timedelta(hours=2),
            fetched_at=entry_at - dt.timedelta(hours=1),
            raw_text_path="/tmp/disp.txt",
            content_hash=f"h-{accession}",
            item_codes='["1.01"]',
            issuer_ticker=symbol,
        ),
    )
    try:
        insert_prompt(
            db,
            PromptRow(
                prompt_version="sonnet@disptest",
                name="sonnet",
                file_path="src/prompts/sonnet.txt",
                content_hash="x" * 64,
            ),
        )
    except Exception:  # noqa: BLE001 — registered by a prior seed
        pass
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id=f"disp-d-{accession}",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet@disptest",
            raw_response=json.dumps({"symbol": symbol}),
            kind="trade_proposal",
            symbol=symbol,
            direction="long",
            size_pct_requested=0.10,
            conviction=conviction,
            thesis="seeded victim thesis",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            cost_usd=0.0,
        ),
    )
    eid = insert_execution(
        db,
        ExecutionRow(
            proposal_id=pid,
            decision="accepted",
            realized_size_pct=0.10,
            realized_dollar_size=2000.0,
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
            qty=20,
            tif="gtc",
            broker_order_id=f"entry-{symbol}",
            submitted_at=entry_at,
            realized_fill_price=100.0,
            realized_fill_qty=20,
            realized_fill_at=entry_at,
            final_status="filled",
        ),
    )
    if live_stop:
        insert_order(
            db,
            OrderRow(
                execution_id=eid,
                role="stop",
                symbol=symbol,
                side="sell",
                order_type="stop",
                qty=20,
                tif="gtc",
                stop_price=90.0,
                broker_order_id=f"stop-{symbol}",
                submitted_at=entry_at,
                final_status=None,
            ),
        )
    if armed:
        upsert_exit_policy_state(
            db,
            ExitPolicyStateRow(
                execution_id=eid,
                symbol=symbol,
                trail_distance_pct=8.0,
                trail_engaged=True,
                high_water_mark=125.0,
            ),
        )
    return eid


def _engine(db: str, broker: _RecordingBroker) -> DisplacementEngine:
    return DisplacementEngine(db_path=db, broker=broker, now_fn=lambda: NOW)


def _attempt(
    engine: DisplacementEngine,
    *,
    conviction: int = 8,
    positions: list[Position],
    cfg: ExecutionConfig,
) -> SizingAccepted | None:
    return engine.attempt(
        proposal=_proposal(conviction=conviction),
        proposal_id=999,
        account=_account(),
        open_positions=positions,
        quote=_quote(),
        cfg=cfg,
        sizer=SizingEngine(),
    )


AGED = NOW - dt.timedelta(days=14)  # ~10 ET trading days held
YOUNG = NOW - dt.timedelta(days=3)  # 2 ET trading days held


# ---------------------------------------------------------------------------
# (g) config validation
# ---------------------------------------------------------------------------


def test_config_defaults_are_off_and_conservative() -> None:
    cfg = _cfg(
        displacement_enabled=False,
    )
    # Remove the explicit test overrides to see the model defaults.
    cfg = ExecutionConfig(
        broker_mode="paper",
        ks4_pct_cap=0.20,
        ks4_absolute_cap_usd=1000.0,
        ks5_max_concurrent=5,
        ks7_cash_reserve_pct=0.05,
        sizing_mid_pct=0.05,
        sizing_high_pct=0.10,
    )
    assert cfg.displacement_enabled is False
    assert cfg.displacement_min_conviction == 8
    assert cfg.displacement_proceeds_haircut == pytest.approx(0.90)
    assert cfg.displacement_victim_max_gain_pct == pytest.approx(5.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"displacement_min_conviction": 5},
        {"displacement_min_conviction": 11},
        {"displacement_proceeds_haircut": 0.0},
        {"displacement_proceeds_haircut": 1.2},
    ],
)
def test_config_bounds_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _cfg(**overrides)


# ---------------------------------------------------------------------------
# (a) disabled → byte-identical / (b) conviction gate
# ---------------------------------------------------------------------------


def test_disabled_no_displacement_and_no_broker_calls(db: str) -> None:
    broker = _RecordingBroker()
    _seed_victim(db, symbol="WIDG", entry_at=AGED)
    result = _attempt(
        _engine(db, broker),
        positions=[_position("WIDG", unrealized_pct=-5.0)],
        cfg=_cfg(displacement_enabled=False),
    )
    assert result is None
    assert broker.submitted == []


def test_sizing_displacement_credit_defaults_to_zero_identical() -> None:
    """Regression guard for the sizer seam: omitting `displacement_credit`
    is byte-identical to passing 0.0 (the existing sizing suite pins the
    broader no-displacement behavior)."""
    sizer = SizingEngine()
    args = (_proposal(conviction=8), _account(), [], _quote(), _cfg())
    implicit = sizer.size(*args)
    explicit = sizer.size(*args, displacement_credit=0.0)
    assert isinstance(implicit, SizingRejected)
    assert implicit == explicit
    assert implicit.reason == "ks7_cash_reserve"


def test_conviction_below_threshold_no_displacement(db: str) -> None:
    broker = _RecordingBroker()
    _seed_victim(db, symbol="WIDG", entry_at=AGED)
    result = _attempt(
        _engine(db, broker),
        conviction=7,
        positions=[_position("WIDG", unrealized_pct=-5.0)],
        cfg=_cfg(),
    )
    assert result is None
    assert broker.submitted == []


# ---------------------------------------------------------------------------
# (c) no qualifying victim → normal rejection stands
# ---------------------------------------------------------------------------


def test_no_qualifying_victim_armed_young_green_or_today(db: str) -> None:
    broker = _RecordingBroker()
    _seed_victim(db, symbol="ARMD", entry_at=AGED, armed=True)
    _seed_victim(db, symbol="YNG", entry_at=YOUNG)
    _seed_victim(db, symbol="GRN", entry_at=AGED)
    _seed_victim(db, symbol="TDAY", entry_at=NOW - dt.timedelta(hours=1))
    result = _attempt(
        _engine(db, broker),
        conviction=8,
        positions=[
            _position("ARMD", unrealized_pct=-10.0),  # armed
            _position("YNG", unrealized_pct=-10.0),  # 2 trading days held
            _position("GRN", unrealized_pct=+8.0),  # above gain cutoff
            _position("TDAY", unrealized_pct=-10.0),  # entered today
        ],
        cfg=_cfg(),
    )
    assert result is None
    assert broker.submitted == []


def test_position_without_journal_lineage_not_evictable(db: str) -> None:
    broker = _RecordingBroker()
    result = _attempt(
        _engine(db, broker),
        positions=[_position("GHOST", unrealized_pct=-20.0)],
        cfg=_cfg(),
    )
    assert result is None
    assert broker.submitted == []


# ---------------------------------------------------------------------------
# (d) qualifying: weakest chosen, sell first, buy sized with haircut credit
# ---------------------------------------------------------------------------


def test_weakest_victim_chosen_sell_submitted_buy_sized_with_credit(
    db: str,
) -> None:
    broker = _RecordingBroker()
    _seed_victim(db, symbol="WEAK", entry_at=AGED, conviction=6)
    weakest_eid = _seed_victim(
        db, symbol="WKST", entry_at=AGED - dt.timedelta(days=1), conviction=7
    )
    weak = _position("WEAK", avg_entry=105.0, unrealized_pct=-4.0)
    weakest = _position("WKST", avg_entry=100.0, unrealized_pct=-8.0)

    result = _attempt(
        _engine(db, broker),
        conviction=8,
        positions=[weak, weakest],
        cfg=_cfg(),
    )

    assert isinstance(result, SizingAccepted)
    # Credit = 0.90 × victim market value (2000 cost − 8% = 1840 → 1656).
    # available = 350 + 1656 − 300 floor = 1706 ≥ tier ask 700 (7% of
    # 10k, cv8 mid tier) → qty = floor(700 / 60) = 11.
    assert result.qty == 11
    assert result.realized_dollar_size == pytest.approx(11 * 60.0)

    # The sell hit the broker (before the caller can submit any buy —
    # `attempt` only returns SizingAccepted after the sell was accepted).
    assert len(broker.submitted) == 1
    sell = broker.submitted[0]
    assert (sell.symbol, sell.side, sell.order_type, sell.qty) == (
        "WKST",
        "sell",
        "market",
        20,
    )
    assert sell.client_order_id == f"displace-close-exec-{weakest_eid}"
    # Live protective leg cancelled best-effort (same as thesis close).
    assert broker.canceled == ["stop-WKST"]

    # Audit trail: distinct journal row with the displacement: prefix
    # carrying victim, new symbol, and both convictions.
    with connect(db) as conn:
        row = conn.execute(
            "SELECT role, notes FROM orders WHERE notes LIKE ?",
            (f"{DISPLACEMENT_NOTES_PREFIX}%",),
        ).fetchone()
    assert row is not None
    assert row["role"] == "displacement_close"
    assert "WKST" in row["notes"]
    assert "ACME" in row["notes"]
    assert "conviction 7" in row["notes"]  # victim's
    assert "conviction 8" in row["notes"]  # new proposal's


def test_resize_still_unfundable_aborts_before_any_order(db: str) -> None:
    """If even the haircut credit cannot fund one share, no order is
    submitted at all — never evict a position we cannot replace."""
    broker = _RecordingBroker()
    _seed_victim(db, symbol="WIDG", entry_at=AGED)
    tiny = Position(
        symbol="WIDG",
        qty=1,
        avg_entry_price=20.0,
        market_value=18.0,
        unrealized_pnl=-2.0,
    )
    # credit = 0.9 × 18 = 16.2 → available 350 + 16.2 − 300 = 66.2, well
    # short of one share at 500 → the credited re-size still ks7-rejects.
    result = _engine(db, broker).attempt(
        proposal=_proposal(conviction=8),
        proposal_id=999,
        account=_account(),
        open_positions=[tiny],
        quote=_quote(last=500.0),
        cfg=_cfg(),
        sizer=SizingEngine(),
    )
    assert result is None
    assert broker.submitted == []


# ---------------------------------------------------------------------------
# (e) one displacement per trading day (hard cap)
# ---------------------------------------------------------------------------


def test_one_displacement_per_trading_day(db: str) -> None:
    broker = _RecordingBroker()
    _seed_victim(db, symbol="WIDG", entry_at=AGED)
    _seed_victim(db, symbol="OTHR", entry_at=AGED)
    positions = [
        _position("WIDG", unrealized_pct=-8.0),
        _position("OTHR", unrealized_pct=-6.0),
    ]
    first = _attempt(_engine(db, broker), positions=positions, cfg=_cfg())
    assert isinstance(first, SizingAccepted)
    assert displaced_today(db, now=NOW)

    second = _attempt(_engine(db, broker), positions=positions, cfg=_cfg())
    assert second is None
    assert len(broker.submitted) == 1  # only the first victim sale


def test_prior_day_displacement_does_not_block_today(db: str) -> None:
    broker = _RecordingBroker()
    eid = _seed_victim(db, symbol="WIDG", entry_at=AGED)
    # A displacement row from the PREVIOUS ET trading day.
    insert_order(
        db,
        OrderRow(
            execution_id=eid,
            role="displacement_close",
            symbol="OLDV",
            side="sell",
            order_type="market",
            qty=5,
            tif="day",
            broker_order_id="disp-old",
            submitted_at=NOW - dt.timedelta(days=1),
            final_status="accepted",
            notes=f"{DISPLACEMENT_NOTES_PREFIX} evicted OLDV (yesterday)",
        ),
    )
    assert not displaced_today(db, now=NOW)
    result = _attempt(
        _engine(db, broker),
        positions=[_position("WIDG", unrealized_pct=-8.0)],
        cfg=_cfg(),
    )
    assert isinstance(result, SizingAccepted)


# ---------------------------------------------------------------------------
# (f) broker sell failure aborts displacement entirely
# ---------------------------------------------------------------------------


def test_broker_sell_failure_aborts_buy_and_leaves_no_trail(db: str) -> None:
    broker = _RecordingBroker()
    _seed_victim(db, symbol="WIDG", entry_at=AGED)
    broker.fail_next_submit = True
    result = _attempt(
        _engine(db, broker),
        positions=[_position("WIDG", unrealized_pct=-8.0)],
        cfg=_cfg(),
    )
    assert result is None  # the buy must never be sized/submitted
    assert broker.submitted == []
    with connect(db) as conn:
        rows = conn.execute(
            "SELECT 1 FROM orders WHERE notes LIKE ?",
            (f"{DISPLACEMENT_NOTES_PREFIX}%",),
        ).fetchall()
    assert rows == []  # no displacement journal row → daily cap unconsumed
    assert not displaced_today(db, now=NOW)


# ---------------------------------------------------------------------------
# (h)/(i)/(j) graduated victim eligibility
# ---------------------------------------------------------------------------


def test_h_conviction_8_cannot_displace_plus7_unarmed_victim(db: str) -> None:
    broker = _RecordingBroker()
    _seed_victim(db, symbol="DRFT", entry_at=AGED)
    result = _attempt(
        _engine(db, broker),
        conviction=8,
        positions=[_position("DRFT", unrealized_pct=+7.0)],
        cfg=_cfg(),
    )
    assert result is None
    assert broker.submitted == []


def test_i_conviction_9_ignores_gain_cutoff_and_picks_weakest(db: str) -> None:
    broker = _RecordingBroker()
    _seed_victim(db, symbol="DRFT", entry_at=AGED)
    _seed_victim(db, symbol="DRFT2", entry_at=AGED)
    result = _attempt(
        _engine(db, broker),
        conviction=9,
        positions=[
            _position("DRFT2", unrealized_pct=+9.0),
            _position("DRFT", unrealized_pct=+7.0),  # weakest of the two
        ],
        cfg=_cfg(),
    )
    assert isinstance(result, SizingAccepted)
    assert len(broker.submitted) == 1
    assert broker.submitted[0].symbol == "DRFT"


def test_j_conviction_9_still_cannot_displace_armed_position(db: str) -> None:
    broker = _RecordingBroker()
    _seed_victim(db, symbol="ARMD", entry_at=AGED, armed=True)
    result = _attempt(
        _engine(db, broker),
        conviction=9,
        positions=[_position("ARMD", unrealized_pct=+2.0)],
        cfg=_cfg(),
    )
    assert result is None
    assert broker.submitted == []


# ---------------------------------------------------------------------------
# trading-day arithmetic
# ---------------------------------------------------------------------------


def test_trading_days_held_counts_market_open_days_only() -> None:
    # Fri 2026-06-05 → Mon 2026-06-08: one trading day elapsed.
    assert trading_days_held(dt.date(2026, 6, 5), dt.date(2026, 6, 8)) == 1
    # Same day → 0; future entry → 0.
    assert trading_days_held(dt.date(2026, 6, 9), dt.date(2026, 6, 9)) == 0
    assert trading_days_held(dt.date(2026, 6, 10), dt.date(2026, 6, 9)) == 0
    # Wed 2026-06-17 → Mon 2026-06-22 spans Juneteenth (Fri 06-19, NYSE
    # holiday): Thu 06-18 + Mon 06-22 = 2 trading days.
    assert trading_days_held(dt.date(2026, 6, 17), dt.date(2026, 6, 22)) == 2
