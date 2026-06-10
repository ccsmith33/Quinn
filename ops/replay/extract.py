"""Read trade proposals out of the journal SQLite database (read-only).

Geometry fields (stop_loss_price / take_profit_price / time_horizon_days)
are not denormalized columns — they live inside `proposals.raw_response`.
The raw_response may be the proposal dict directly or wrap it under a
`trade` / `proposal` key; we try all three shapes, same resolution strategy
as `ops/scripts/place_stops.py` (the heritage script for this path).
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

from ops.replay.models import ReplayProposal, SkippedProposal

# Boundary between the bracket-era and PDT-era execution regimes; the report
# splits aggregates on proposal created_at relative to this instant.
REGIME_SPLIT_UTC = dt.datetime(2026, 5, 8, tzinfo=dt.UTC)


def _parse_created_at(raw: str) -> dt.datetime:
    """Journal timestamps are SQLite CURRENT_TIMESTAMP strings (UTC)."""
    s = raw.replace("T", " ").replace("Z", "")
    ts = dt.datetime.fromisoformat(s)
    return ts.replace(tzinfo=dt.UTC) if ts.tzinfo is None else ts.astimezone(dt.UTC)


def _geometry_from_raw_response(raw_response: str) -> dict[str, Any] | None:
    """Find the proposal-shaped dict carrying stop_loss_price, if any."""
    try:
        payload = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    for candidate in (payload, payload.get("trade"), payload.get("proposal")):
        if isinstance(candidate, dict) and candidate.get("stop_loss_price") is not None:
            return candidate
    return None


def load_proposals(
    journal_path: str | Path,
) -> tuple[list[ReplayProposal], list[SkippedProposal]]:
    """All long trade proposals with usable geometry, plus the skip list.

    Opens the journal strictly read-only (`mode=ro` URI) — this tool must
    never mutate the journal.
    """
    conn = sqlite3.connect(f"file:{journal_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT p.id, p.decision_id, p.symbol, p.direction, p.conviction, "
            "       p.created_at, p.raw_response, e.decision AS exec_decision "
            "FROM proposals p "
            "LEFT JOIN executions e ON e.proposal_id = p.id "
            "WHERE p.kind = 'trade_proposal' AND p.symbol IS NOT NULL "
            "ORDER BY p.created_at, p.id"
        ).fetchall()
    finally:
        conn.close()

    proposals: list[ReplayProposal] = []
    skipped: list[SkippedProposal] = []
    for row in rows:
        pid = int(row["id"])
        symbol = str(row["symbol"])
        if row["direction"] is not None and row["direction"] != "long":
            skipped.append(SkippedProposal(pid, symbol, f"direction={row['direction']}"))
            continue
        geo = _geometry_from_raw_response(row["raw_response"])
        if geo is None:
            skipped.append(SkippedProposal(pid, symbol, "no stop_loss_price in raw_response"))
            continue
        try:
            stop = float(geo["stop_loss_price"])
            tp_raw = geo.get("take_profit_price")
            tp = float(tp_raw) if tp_raw is not None else None
            horizon_raw = geo.get("time_horizon_days")
            horizon = int(horizon_raw) if horizon_raw is not None else None
            created = _parse_created_at(str(row["created_at"]))
        except (TypeError, ValueError) as e:
            skipped.append(SkippedProposal(pid, symbol, f"unparseable geometry: {e}"))
            continue
        if stop <= 0:
            skipped.append(SkippedProposal(pid, symbol, f"non-positive stop {stop}"))
            continue
        proposals.append(
            ReplayProposal(
                proposal_id=pid,
                decision_id=str(row["decision_id"]),
                symbol=symbol,
                created_at_utc=created,
                stop_loss_price=stop,
                take_profit_price=tp,
                time_horizon_days=horizon,
                conviction=int(row["conviction"]) if row["conviction"] is not None else None,
                exec_decision=(
                    str(row["exec_decision"]) if row["exec_decision"] is not None else None
                ),
            )
        )
    return proposals, skipped
