"""S2.3 — In-process Universe lookup API.

Loaded once at agent boot from the latest `universe_snapshots` row. Exposes
O(1) membership checks for the ingestion gate (FR-11, ADR-002) and the
execution validator (FR-20). Day-boundary refresh happens via
`reload_if_newer()`, which the agent loop calls once per discovery cycle
(S3.2).

This module is read-only; only the journal package writes to the database
(architecture §2.9 invariant).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from journal.repo import connect, get_current_universe_snapshot


class NoUniverseSnapshot(Exception):
    """Raised when no `universe_snapshots` row exists at boot.

    The agent main treats this as a hard error per AC-4: trading without a
    universe is not permitted.
    """


class UniverseMember(BaseModel):
    """Public view of a snapshot member.

    Mirrors `journal.models.UniverseMemberRow` minus the foreign-key
    `snapshot_id`, which callers don't need.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cik: int
    ticker: str
    exchange: str
    market_cap: float
    prev_close: float


class Universe:
    """In-memory snapshot of the current trading universe."""

    def __init__(
        self,
        db_path: str,
        snapshot_id: int,
        members: list[UniverseMember],
    ) -> None:
        self._db_path = db_path
        self._snapshot_id = snapshot_id
        self._tickers: set[str] = {m.ticker for m in members}
        self._ciks: set[int] = {m.cik for m in members}
        self._members_by_ticker: dict[str, UniverseMember] = {
            m.ticker: m for m in members
        }

    @classmethod
    def load_latest(cls, db_path: str) -> Universe:
        snap = get_current_universe_snapshot(db_path)
        if snap is None or snap.snapshot_id is None:
            raise NoUniverseSnapshot(
                "no universe snapshot in database; run jobs/refresh_universe first"
            )
        members = _load_members(db_path, snap.snapshot_id)
        return cls(db_path, snap.snapshot_id, members)

    def current_snapshot_id(self) -> int:
        return self._snapshot_id

    def is_in_universe(self, ticker: str) -> bool:
        return ticker in self._tickers

    def member_count(self) -> int:
        """Number of members in the loaded snapshot.

        Zero is the in-process shape of "no usable snapshot": `load_latest`
        raises when the snapshot row is absent, but a snapshot whose
        `universe_members` rows failed to load leaves a Universe that
        answers "not a member" to every ticker. Callers that gate on
        membership use this to fail open rather than treat an empty
        snapshot as a universe-wide rejection (carry-forward D-038).
        """
        return len(self._tickers)

    def is_in_universe_by_cik(self, cik: int) -> bool:
        return cik in self._ciks

    def iter_ciks(self) -> list[int]:
        """Snapshot of universe CIKs for the reconciler (S3.4) to iterate.

        Returns a list (not a view) so callers can iterate safely across
        a `reload_if_newer()` boundary; cheap because the universe is
        bounded at ~3-4k entries (FR-7..FR-9).
        """
        return list(self._ciks)

    def get_member(self, ticker: str) -> UniverseMember | None:
        return self._members_by_ticker.get(ticker)

    def reload_if_newer(self) -> bool:
        """Re-read the latest snapshot; reload state iff its id is newer.

        Returns True if a reload happened. Idempotent: a second call with no
        new snapshot returns False.
        """
        snap = get_current_universe_snapshot(self._db_path)
        if snap is None or snap.snapshot_id is None:
            return False
        if snap.snapshot_id <= self._snapshot_id:
            return False
        members = _load_members(self._db_path, snap.snapshot_id)
        self._snapshot_id = snap.snapshot_id
        self._tickers = {m.ticker for m in members}
        self._ciks = {m.cik for m in members}
        self._members_by_ticker = {m.ticker: m for m in members}
        return True


def _load_members(db_path: str, snapshot_id: int) -> list[UniverseMember]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT cik, ticker, exchange, market_cap, prev_close "
            "FROM universe_members WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchall()
    return [
        UniverseMember(
            cik=int(r["cik"]),
            ticker=str(r["ticker"]),
            exchange=str(r["exchange"]),
            market_cap=float(r["market_cap"]),
            prev_close=float(r["prev_close"]),
        )
        for r in rows
    ]
