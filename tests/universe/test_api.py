"""S2.3 — in-process Universe lookup API tests (TDD).

Architecture references: §2.10 public interface, ADR-006, FR-10, FR-11.
Story: artifacts/implementation/stories/story-02-03-universe-lookup-api.md.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.models import UniverseMemberRow, UniverseSnapshotRow
from journal.repo import insert_universe_member, insert_universe_snapshot
from universe.api import NoUniverseSnapshot, Universe, UniverseMember


@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return str(db_path)


def _seed_snapshot(
    db_path: str,
    snapshot_date: dt.date,
    members: list[tuple[int, str, str, float, float]],
    *,
    is_degraded: int = 0,
) -> int:
    """Insert a snapshot + its members. Returns snapshot_id."""
    snap_id = insert_universe_snapshot(
        db_path,
        UniverseSnapshotRow(
            snapshot_date=snapshot_date,
            sec_tickers_hash=f"sec-{snapshot_date.isoformat()}",
            alpaca_assets_hash=f"alp-{snapshot_date.isoformat()}",
            yfinance_failures=0,
            member_count=len(members),
            is_degraded=is_degraded,
        ),
    )
    for cik, ticker, exchange, market_cap, prev_close in members:
        insert_universe_member(
            db_path,
            UniverseMemberRow(
                snapshot_id=snap_id,
                cik=cik,
                ticker=ticker,
                exchange=exchange,
                market_cap=market_cap,
                prev_close=prev_close,
            ),
        )
    return snap_id


# ---------------------------------------------------------------------------
# AC-1, AC-2: load + ticker/cik lookup
# ---------------------------------------------------------------------------


def test_load_and_lookup_by_ticker(db: str) -> None:
    snap_id = _seed_snapshot(
        db,
        dt.date(2026, 4, 28),
        [
            (1234567, "ACME", "NASDAQ", 500_000_000.0, 12.34),
            (2345678, "BETA", "NYSE", 800_000_000.0, 25.0),
        ],
    )
    u = Universe.load_latest(db)
    assert u.current_snapshot_id() == snap_id
    assert u.is_in_universe("ACME") is True
    assert u.is_in_universe("BETA") is True
    assert u.is_in_universe("NOPE") is False


def test_lookup_by_cik(db: str) -> None:
    _seed_snapshot(
        db,
        dt.date(2026, 4, 28),
        [
            (1234567, "ACME", "NASDAQ", 500_000_000.0, 12.34),
            (2345678, "BETA", "NYSE", 800_000_000.0, 25.0),
        ],
    )
    u = Universe.load_latest(db)
    assert u.is_in_universe_by_cik(1234567) is True
    assert u.is_in_universe_by_cik(2345678) is True
    assert u.is_in_universe_by_cik(9999999) is False


def test_get_member_returns_full_record(db: str) -> None:
    _seed_snapshot(
        db,
        dt.date(2026, 4, 28),
        [(1234567, "ACME", "NASDAQ", 500_000_000.0, 12.34)],
    )
    u = Universe.load_latest(db)
    m = u.get_member("ACME")
    assert isinstance(m, UniverseMember)
    assert m.ticker == "ACME"
    assert m.cik == 1234567
    assert m.exchange == "NASDAQ"
    assert m.market_cap == 500_000_000.0
    assert m.prev_close == 12.34
    assert u.get_member("MISSING") is None


# ---------------------------------------------------------------------------
# AC-4: missing snapshot → hard error
# ---------------------------------------------------------------------------


def test_no_snapshot_raises(db: str) -> None:
    with pytest.raises(NoUniverseSnapshot):
        Universe.load_latest(db)


# ---------------------------------------------------------------------------
# AC-3: reload_if_newer
# ---------------------------------------------------------------------------


def test_reload_when_new_snapshot_inserted(db: str) -> None:
    snap1 = _seed_snapshot(
        db,
        dt.date(2026, 4, 27),
        [(1, "OLD", "NASDAQ", 100_000_000.0, 10.0)],
    )
    u = Universe.load_latest(db)
    assert u.current_snapshot_id() == snap1
    assert u.is_in_universe("OLD") is True
    assert u.is_in_universe("NEW") is False

    snap2 = _seed_snapshot(
        db,
        dt.date(2026, 4, 28),
        [
            (1, "OLD", "NASDAQ", 100_000_000.0, 10.0),
            (2, "NEW", "NYSE", 200_000_000.0, 15.0),
        ],
    )
    assert u.reload_if_newer() is True
    assert u.current_snapshot_id() == snap2
    assert u.is_in_universe("NEW") is True
    assert u.is_in_universe_by_cik(2) is True


def test_reload_noop_when_no_change(db: str) -> None:
    _seed_snapshot(
        db,
        dt.date(2026, 4, 28),
        [(1, "ACME", "NASDAQ", 100_000_000.0, 10.0)],
    )
    u = Universe.load_latest(db)
    assert u.reload_if_newer() is False
    assert u.reload_if_newer() is False  # idempotent


def test_reload_drops_removed_members(db: str) -> None:
    """A name in yesterday's snapshot but not today's must drop out after reload."""
    _seed_snapshot(
        db,
        dt.date(2026, 4, 27),
        [
            (1, "STAY", "NASDAQ", 100_000_000.0, 10.0),
            (2, "DROP", "NYSE", 100_000_000.0, 10.0),
        ],
    )
    u = Universe.load_latest(db)
    assert u.is_in_universe("DROP") is True

    _seed_snapshot(
        db,
        dt.date(2026, 4, 28),
        [(1, "STAY", "NASDAQ", 100_000_000.0, 10.0)],
    )
    u.reload_if_newer()
    assert u.is_in_universe("STAY") is True
    assert u.is_in_universe("DROP") is False
    assert u.is_in_universe_by_cik(2) is False


# ---------------------------------------------------------------------------
# AC-5: ticker and CIK lookups consistent with latest snapshot membership
# ---------------------------------------------------------------------------


def test_lookups_are_consistent_with_latest_snapshot(db: str) -> None:
    """Insert two snapshots; only the latest's members are reachable via either key."""
    _seed_snapshot(
        db,
        dt.date(2026, 4, 27),
        [
            (1, "GONE1", "NASDAQ", 100_000_000.0, 10.0),
            (2, "GONE2", "NYSE", 100_000_000.0, 10.0),
        ],
    )
    _seed_snapshot(
        db,
        dt.date(2026, 4, 28),
        [
            (10, "KEEP1", "NASDAQ", 100_000_000.0, 10.0),
            (20, "KEEP2", "NYSE", 100_000_000.0, 10.0),
        ],
    )
    u = Universe.load_latest(db)
    # latest membership reachable via both keys
    assert u.is_in_universe("KEEP1") and u.is_in_universe_by_cik(10)
    assert u.is_in_universe("KEEP2") and u.is_in_universe_by_cik(20)
    # prior-snapshot members are NOT in the universe (FR-10/AC-5)
    assert not u.is_in_universe("GONE1")
    assert not u.is_in_universe_by_cik(1)
    assert not u.is_in_universe("GONE2")
    assert not u.is_in_universe_by_cik(2)


# ---------------------------------------------------------------------------
# AC-2: lookup data structure is a set (O(1) membership)
# ---------------------------------------------------------------------------


def test_lookup_uses_in_memory_sets(db: str) -> None:
    """Defense-in-depth: the lookup keys live in `set` instances, not list scans."""
    _seed_snapshot(
        db,
        dt.date(2026, 4, 28),
        [(1, "ACME", "NASDAQ", 100_000_000.0, 10.0)],
    )
    u = Universe.load_latest(db)
    # Access the underlying containers via documented attributes (AC-2 uses
    # in-memory sets — published as part of the implementation contract).
    assert isinstance(u._tickers, set)
    assert isinstance(u._ciks, set)
