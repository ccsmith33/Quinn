"""`get_conviction_calibration` — realized outcomes by conviction band.

Covers: band math (N / win rate / mean / median per band), boundary
convictions landing in the right bands, open (non-flat) positions
excluded, partial closes excluded, no-conviction proposals excluded,
deterministic band ordering, and the zero-closed-trades shape.

Closed-trade definition under test (documented on the repo function):
an `accepted` execution whose FILLED orders (final_status in
('filled', 'partially_filled_closed') with non-NULL fill qty/price)
include at least one entry buy AND net flat: total filled sell qty ==
total filled buy qty. Realized % is fee-inclusive, on entry notional.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.models import (
    ExecutionRow,
    FilingRow,
    OrderRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    JournalRepo,
    get_conviction_calibration,
    get_prompt_by_version,
    insert_execution,
    insert_filing,
    insert_order,
    insert_prompt,
    insert_proposal,
)

_PROMPT_VERSION = "pv@aaaaaaaaaaaa"
_T0 = dt.datetime(2026, 4, 28, 14, 30, 0)


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return str(p)


_seq = 0


def _seed_execution(db_path: str, *, conviction: int | None) -> int:
    """One accepted execution whose proposal carries `conviction`."""
    global _seq
    _seq += 1
    if get_prompt_by_version(db_path, _PROMPT_VERSION) is None:
        insert_prompt(
            db_path,
            PromptRow(
                prompt_version=_PROMPT_VERSION,
                name="sonnet_filing_analysis_v2",
                file_path="src/prompts/sonnet_filing_analysis_v2.txt",
                content_hash="a" * 64,
            ),
        )
    fid = insert_filing(
        db_path,
        FilingRow(
            accession_number=f"acc-{_seq}",
            cik=1234567,
            form_type="8-K",
            filed_at=_T0,
            fetched_at=_T0,
            raw_text_path=f"/raw/{_seq}.txt",
            content_hash=f"h-{_seq}",
        ),
    )
    pid = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=fid,
            decision_id=f"dec-{_seq}",
            model_id="claude-sonnet-4-6",
            prompt_version=_PROMPT_VERSION,
            raw_response="{}",
            kind="trade_proposal",
            symbol=f"SYM{_seq}",
            direction="long",
            size_pct_requested=0.05,
            conviction=conviction,
            thesis="x",
            input_tokens=10,
            output_tokens=10,
            latency_ms=100,
            cost_usd=0.001,
        ),
    )
    return insert_execution(
        db_path,
        ExecutionRow(
            proposal_id=pid, decision="accepted", submitted_orders_json="[]"
        ),
    )


def _fill(
    db_path: str,
    execution_id: int,
    *,
    role: str,
    side: str,
    qty: int,
    price: float,
    status: str = "filled",
    fee: float = 0.0,
) -> None:
    global _seq
    _seq += 1
    insert_order(
        db_path,
        OrderRow(
            execution_id=execution_id,
            role=role,
            symbol="ACME",
            side=side,
            order_type="market",
            qty=qty,
            tif="day",
            broker_order_id=f"bo-{_seq}",
            submitted_at=_T0,
            final_status=status,
            realized_fill_price=price,
            realized_fill_qty=qty,
            realized_fill_at=_T0 + dt.timedelta(days=1),
            realized_fee=fee,
        ),
    )


def _closed_trade(
    db_path: str, *, conviction: int, entry: float, exit_: float, qty: int = 10
) -> int:
    """Seed one flat round trip: entry buy filled, stop sell filled."""
    eid = _seed_execution(db_path, conviction=conviction)
    _fill(db_path, eid, role="entry", side="buy", qty=qty, price=entry)
    _fill(db_path, eid, role="stop", side="sell", qty=qty, price=exit_)
    return eid


# ---------------------------------------------------------------------------
# shape + determinism
# ---------------------------------------------------------------------------


def test_zero_closed_trades_shape(db: str) -> None:
    calib = get_conviction_calibration(db)
    assert calib["total_n"] == 0
    assert [b["band"] for b in calib["bands"]] == ["2-3", "4-5", "6-7", "8-10"]
    for b in calib["bands"]:
        assert b["n"] == 0
        assert b["win_rate_pct"] is None
        assert b["mean_realized_pct"] is None
        assert b["median_realized_pct"] is None


def test_band_order_is_fixed_regardless_of_insert_order(db: str) -> None:
    _closed_trade(db, conviction=9, entry=100.0, exit_=110.0)
    _closed_trade(db, conviction=2, entry=100.0, exit_=90.0)
    calib = get_conviction_calibration(db)
    assert [b["band"] for b in calib["bands"]] == ["2-3", "4-5", "6-7", "8-10"]


def test_deterministic_across_calls(db: str) -> None:
    _closed_trade(db, conviction=5, entry=100.0, exit_=103.0)
    _closed_trade(db, conviction=8, entry=50.0, exit_=45.0)
    assert get_conviction_calibration(db) == get_conviction_calibration(db)


# ---------------------------------------------------------------------------
# band math
# ---------------------------------------------------------------------------


def test_band_stats_win_rate_mean_median(db: str) -> None:
    # cv6-7 band: three closed trades at +10%, +5%, -5%.
    _closed_trade(db, conviction=6, entry=100.0, exit_=110.0)
    _closed_trade(db, conviction=7, entry=100.0, exit_=105.0)
    _closed_trade(db, conviction=6, entry=100.0, exit_=95.0)
    calib = get_conviction_calibration(db)
    assert calib["total_n"] == 3
    band = next(b for b in calib["bands"] if b["band"] == "6-7")
    assert band["n"] == 3
    assert band["win_rate_pct"] == pytest.approx(66.7)
    assert band["mean_realized_pct"] == pytest.approx(3.3)
    assert band["median_realized_pct"] == pytest.approx(5.0)


def test_median_even_count_averages_middle_two(db: str) -> None:
    for pct_exit in (102.0, 104.0, 108.0, 120.0):
        _closed_trade(db, conviction=4, entry=100.0, exit_=pct_exit)
    band = next(
        b for b in get_conviction_calibration(db)["bands"] if b["band"] == "4-5"
    )
    assert band["n"] == 4
    assert band["median_realized_pct"] == pytest.approx(6.0)  # (4+8)/2


def test_boundary_convictions_land_in_right_bands(db: str) -> None:
    for cv in (2, 3, 4, 5, 6, 7, 8, 10):
        _closed_trade(db, conviction=cv, entry=100.0, exit_=101.0)
    calib = get_conviction_calibration(db)
    by_band = {b["band"]: b["n"] for b in calib["bands"]}
    assert by_band == {"2-3": 2, "4-5": 2, "6-7": 2, "8-10": 2}
    assert calib["total_n"] == 8


def test_zero_realized_is_not_a_win(db: str) -> None:
    _closed_trade(db, conviction=5, entry=100.0, exit_=100.0)  # exactly 0%
    _closed_trade(db, conviction=5, entry=100.0, exit_=110.0)
    band = next(
        b for b in get_conviction_calibration(db)["bands"] if b["band"] == "4-5"
    )
    assert band["n"] == 2
    assert band["win_rate_pct"] == pytest.approx(50.0)


def test_fees_reduce_realized_pct(db: str) -> None:
    eid = _seed_execution(db, conviction=8)
    _fill(db, eid, role="entry", side="buy", qty=10, price=100.0, fee=5.0)
    _fill(db, eid, role="stop", side="sell", qty=10, price=110.0, fee=5.0)
    band = next(
        b for b in get_conviction_calibration(db)["bands"] if b["band"] == "8-10"
    )
    # (1100 - 1000 - 10) / 1000 = +9.0%
    assert band["mean_realized_pct"] == pytest.approx(9.0)


def test_percents_rounded_one_decimal(db: str) -> None:
    _closed_trade(db, conviction=6, entry=90.0, exit_=91.0)  # +1.111...%
    band = next(
        b for b in get_conviction_calibration(db)["bands"] if b["band"] == "6-7"
    )
    assert band["mean_realized_pct"] == 1.1
    assert band["median_realized_pct"] == 1.1


# ---------------------------------------------------------------------------
# exclusions — open / partial / unbanded
# ---------------------------------------------------------------------------


def test_open_position_excluded(db: str) -> None:
    eid = _seed_execution(db, conviction=8)
    _fill(db, eid, role="entry", side="buy", qty=10, price=100.0)
    # no exit fill — position is open
    assert get_conviction_calibration(db)["total_n"] == 0


def test_partial_close_excluded(db: str) -> None:
    eid = _seed_execution(db, conviction=8)
    _fill(db, eid, role="entry", side="buy", qty=10, price=100.0)
    _fill(db, eid, role="stop", side="sell", qty=4, price=110.0)  # 6 still held
    assert get_conviction_calibration(db)["total_n"] == 0


def test_unfilled_orders_do_not_close_a_trade(db: str) -> None:
    eid = _seed_execution(db, conviction=8)
    _fill(db, eid, role="entry", side="buy", qty=10, price=100.0)
    # exit submitted but pending (final_status NULL, no fill columns)
    insert_order(
        db,
        OrderRow(
            execution_id=eid,
            role="stop",
            symbol="ACME",
            side="sell",
            order_type="stop",
            qty=10,
            stop_price=95.0,
            tif="gtc",
            broker_order_id="bo-pending",
            submitted_at=_T0,
        ),
    )
    assert get_conviction_calibration(db)["total_n"] == 0


def test_partially_filled_closed_exit_counts_when_flat(db: str) -> None:
    """A `partially_filled_closed` exit carries real fill qty; a flat
    entry+exit pair using it is a closed trade."""
    eid = _seed_execution(db, conviction=6)
    _fill(db, eid, role="entry", side="buy", qty=10, price=100.0)
    _fill(
        db, eid, role="stop", side="sell", qty=10, price=104.0,
        status="partially_filled_closed",
    )
    calib = get_conviction_calibration(db)
    assert calib["total_n"] == 1
    band = next(b for b in calib["bands"] if b["band"] == "6-7")
    assert band["mean_realized_pct"] == pytest.approx(4.0)


def test_exit_without_entry_fill_excluded(db: str) -> None:
    eid = _seed_execution(db, conviction=6)
    _fill(db, eid, role="stop", side="sell", qty=10, price=104.0)
    assert get_conviction_calibration(db)["total_n"] == 0


def test_null_conviction_excluded(db: str) -> None:
    eid = _seed_execution(db, conviction=None)
    _fill(db, eid, role="entry", side="buy", qty=10, price=100.0)
    _fill(db, eid, role="stop", side="sell", qty=10, price=110.0)
    assert get_conviction_calibration(db)["total_n"] == 0


def test_conviction_below_band_floor_excluded(db: str) -> None:
    _closed_trade(db, conviction=1, entry=100.0, exit_=110.0)
    assert get_conviction_calibration(db)["total_n"] == 0


def test_facade_binding(db: str) -> None:
    _closed_trade(db, conviction=5, entry=100.0, exit_=101.0)
    repo = JournalRepo(db)
    assert repo.get_conviction_calibration() == get_conviction_calibration(db)
