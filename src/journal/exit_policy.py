"""D-079 §3.5 — DAO for `exit_policy_state` (migration 006, ADR-011).

Operational state for the trailing ratchet: one row per execution under
trailing management, updated IN PLACE. This is deliberately not part of
`repo.py` (delta §1 ownership seam — repo.py is WS1's file this epic).
NFR-16 is not diluted: the audit trail of every stop movement is the
append-only `orders` replacement chain; this table only carries the
ratchet's working memory (high-water mark, engagement flag, pointer to
the current live stop's journal row).
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict

from journal.repo import connect


class ExitPolicyStateRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: int
    symbol: str
    trail_distance_pct: float
    trail_engaged: bool = False
    high_water_mark: float
    stop_order_journal_id: int | None = None
    updated_at: dt.datetime | None = None


def get_exit_policy_state(
    db_path: str, *, execution_id: int
) -> ExitPolicyStateRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM exit_policy_state WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["trail_engaged"] = bool(d["trail_engaged"])
    return ExitPolicyStateRow(**d)


def upsert_exit_policy_state(db_path: str, row: ExitPolicyStateRow) -> None:
    """Insert or update the single state row for an execution."""
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO exit_policy_state (
                execution_id, symbol, trail_distance_pct, trail_engaged,
                high_water_mark, stop_order_journal_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(execution_id) DO UPDATE SET
                trail_distance_pct = excluded.trail_distance_pct,
                trail_engaged = excluded.trail_engaged,
                high_water_mark = excluded.high_water_mark,
                stop_order_journal_id = excluded.stop_order_journal_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                row.execution_id,
                row.symbol,
                row.trail_distance_pct,
                int(row.trail_engaged),
                row.high_water_mark,
                row.stop_order_journal_id,
            ),
        )


def set_stop_order_journal_id(
    db_path: str, *, execution_id: int, order_journal_id: int
) -> None:
    """Rotate the tracked live-stop pointer after a replacement (§3.3
    step 4). No-op when no trailing state exists for the execution —
    callers invoke this unconditionally after any stop replacement."""
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE exit_policy_state
            SET stop_order_journal_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE execution_id = ?
            """,
            (order_journal_id, execution_id),
        )


__all__ = [
    "ExitPolicyStateRow",
    "get_exit_policy_state",
    "set_stop_order_journal_id",
    "upsert_exit_policy_state",
]
