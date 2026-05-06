"""Journal repository — typed insert/get API for every table in §8.1.

Architectural invariants:
- §2.9: only this module opens SQLite connections; other modules call these
  functions.
- §8.2 / NFR-16: append-only tables expose only `insert_*` and `get_*`.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from observability.log_port import get_logger

from .models import (
    AccountSnapshotRow,
    BackupRow,
    ExecutionRow,
    FilingRow,
    KillSwitchStateRow,
    LlmCallRow,
    OrderRow,
    PositionRow,
    PrefilterDecisionRow,
    PromptRow,
    ProposalReviewRow,
    ProposalRow,
    SimilarityCacheRow,
    ThesisReviewRow,
    ThesisReviewScheduleRow,
    UniverseMemberRow,
    UniverseSnapshotRow,
)

log = get_logger(__name__)


# Python 3.12 deprecated the default datetime/date adapters; register explicit
# ISO-format converters so behaviour is stable and warning-free across 3.11+.
def _adapt_datetime(value: _dt.datetime) -> str:
    return value.isoformat(sep=" ")


def _adapt_date(value: _dt.date) -> str:
    return value.isoformat()


def _convert_timestamp(blob: bytes) -> _dt.datetime:
    text = blob.decode("utf-8")
    return _dt.datetime.fromisoformat(text.replace(" ", "T"))


def _convert_date(blob: bytes) -> _dt.date:
    return _dt.date.fromisoformat(blob.decode("utf-8"))


sqlite3.register_adapter(_dt.datetime, _adapt_datetime)
sqlite3.register_adapter(_dt.date, _adapt_date)
sqlite3.register_converter("TIMESTAMP", _convert_timestamp)
sqlite3.register_converter("DATE", _convert_date)


class DuplicateAccession(Exception):
    """Raised when an attempted insert into `filings` collides on accession_number."""

    def __init__(self, accession_number: str, existing_id: int) -> None:
        super().__init__(f"duplicate accession_number: {accession_number}")
        self.accession_number = accession_number
        self.existing_id = existing_id


@contextmanager
def connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection in WAL with FK on; row factory yields dicts."""
    conn = sqlite3.connect(
        db_path,
        isolation_level=None,
        timeout=30.0,
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _percentile(values: list[int], pct: int) -> int:
    """Nearest-rank percentile on a pre-sorted list. Returns 0 on empty
    input. Used by the S9.1 dashboard `/llm` page (D-063).
    """
    if not values:
        return 0
    if pct <= 0:
        return values[0]
    if pct >= 100:
        return values[-1]
    k = max(1, int(round((pct / 100.0) * len(values))))
    return values[min(k - 1, len(values) - 1)]


def _exec_write(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> sqlite3.Cursor:
    """Run an INSERT inside a BEGIN IMMEDIATE transaction (serialized writers)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute(sql, params)
        conn.execute("COMMIT")
        return cur
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# filings
# ---------------------------------------------------------------------------

_FILING_INSERT = """
INSERT INTO filings (
    accession_number, cik, form_type, filed_at, fetched_at,
    raw_text_path, content_hash, item_codes, issuer_ticker,
    ingest_state, ingest_error
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def insert_filing(db_path: str, row: FilingRow) -> int:
    with connect(db_path) as conn:
        try:
            cur = _exec_write(
                conn,
                _FILING_INSERT,
                (
                    row.accession_number,
                    row.cik,
                    row.form_type,
                    row.filed_at,
                    row.fetched_at,
                    row.raw_text_path,
                    row.content_hash,
                    row.item_codes,
                    row.issuer_ticker,
                    row.ingest_state,
                    row.ingest_error,
                ),
            )
        except sqlite3.IntegrityError as e:
            if "filings.accession_number" in str(e) or "UNIQUE" in str(e).upper():
                existing = conn.execute(
                    "SELECT id FROM filings WHERE accession_number = ?",
                    (row.accession_number,),
                ).fetchone()
                if existing is not None:
                    raise DuplicateAccession(row.accession_number, existing["id"]) from e
            raise
        new_id = int(cur.lastrowid or 0)
        log.debug("inserted filing", extra={"filing_id": new_id})
        return new_id


def get_filing_by_id(db_path: str, filing_id: int) -> FilingRow | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM filings WHERE id = ?", (filing_id,)).fetchone()
    d = _row_to_dict(row)
    return FilingRow(**d) if d is not None else None


def get_filing_by_accession(db_path: str, accession_number: str) -> FilingRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM filings WHERE accession_number = ?", (accession_number,)
        ).fetchone()
    d = _row_to_dict(row)
    return FilingRow(**d) if d is not None else None


def get_latest_filing_for_issuer_form(
    db_path: str, cik: int, form_type: str
) -> FilingRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM filings WHERE cik = ? AND form_type = ? "
            "ORDER BY filed_at DESC LIMIT 1",
            (cik, form_type),
        ).fetchone()
    d = _row_to_dict(row)
    return FilingRow(**d) if d is not None else None


# ---------------------------------------------------------------------------
# prefilter_decisions
# ---------------------------------------------------------------------------

def insert_prefilter_decision(db_path: str, row: PrefilterDecisionRow) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            "INSERT INTO prefilter_decisions "
            "(filing_id, decision, rule_fired, similarity_score, reason_detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                row.filing_id,
                row.decision,
                row.rule_fired,
                row.similarity_score,
                row.reason_detail,
            ),
        )
        return int(cur.lastrowid or 0)


def get_prefilter_decision_by_filing(
    db_path: str, filing_id: int
) -> PrefilterDecisionRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM prefilter_decisions WHERE filing_id = ?", (filing_id,)
        ).fetchone()
    d = _row_to_dict(row)
    return PrefilterDecisionRow(**d) if d is not None else None


# ---------------------------------------------------------------------------
# similarity_cache
# ---------------------------------------------------------------------------

def insert_similarity_cache(db_path: str, row: SimilarityCacheRow) -> None:
    with connect(db_path) as conn:
        _exec_write(
            conn,
            "INSERT INTO similarity_cache "
            "(cik, form_type, accession_number, minhash_blob, "
            "tfidf_vectorizer_path, tfidf_vector_path, fitted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row.cik,
                row.form_type,
                row.accession_number,
                row.minhash_blob,
                row.tfidf_vectorizer_path,
                row.tfidf_vector_path,
                row.fitted_at,
            ),
        )


def get_similarity_cache_for_issuer_form(
    db_path: str, cik: int, form_type: str
) -> list[SimilarityCacheRow]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM similarity_cache WHERE cik = ? AND form_type = ? "
            "ORDER BY fitted_at DESC",
            (cik, form_type),
        ).fetchall()
    return [SimilarityCacheRow(**dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def insert_prompt(db_path: str, row: PromptRow) -> None:
    with connect(db_path) as conn:
        _exec_write(
            conn,
            "INSERT INTO prompts (prompt_version, name, file_path, content_hash) "
            "VALUES (?, ?, ?, ?)",
            (row.prompt_version, row.name, row.file_path, row.content_hash),
        )


def get_prompt_by_version(db_path: str, prompt_version: str) -> PromptRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM prompts WHERE prompt_version = ?", (prompt_version,)
        ).fetchone()
    d = _row_to_dict(row)
    return PromptRow(**d) if d is not None else None


# ---------------------------------------------------------------------------
# proposals (append-only)
# ---------------------------------------------------------------------------

_PROPOSAL_INSERT = """
INSERT INTO proposals (
    filing_id, decision_id, model_id, prompt_version, raw_response, kind,
    symbol, direction, size_pct_requested, conviction, thesis,
    input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens,
    latency_ms, cost_usd, reasoning_quality, reasoning_notes
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def insert_proposal(db_path: str, row: ProposalRow) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            _PROPOSAL_INSERT,
            (
                row.filing_id,
                row.decision_id,
                row.model_id,
                row.prompt_version,
                row.raw_response,
                row.kind,
                row.symbol,
                row.direction,
                row.size_pct_requested,
                row.conviction,
                row.thesis,
                row.input_tokens,
                row.output_tokens,
                row.cache_read_tokens,
                row.cache_creation_tokens,
                row.latency_ms,
                row.cost_usd,
                row.reasoning_quality,
                row.reasoning_notes,
            ),
        )
        return int(cur.lastrowid or 0)


def get_proposal_by_id(db_path: str, proposal_id: int) -> ProposalRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
    d = _row_to_dict(row)
    return ProposalRow(**d) if d is not None else None


def get_proposal_by_decision_id(db_path: str, decision_id: str) -> ProposalRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM proposals WHERE decision_id = ?", (decision_id,)
        ).fetchone()
    d = _row_to_dict(row)
    return ProposalRow(**d) if d is not None else None


def find_retro_candidate(
    db_path: str,
    *,
    min_conviction: int,
    not_older_than: _dt.datetime,
) -> ProposalRow | None:
    """Feature C — find a high-conviction `trade_proposal` that was
    skipped at capacity and is still fresh enough to retro-fill.

    A row is a candidate when:
      - kind == 'trade_proposal'
      - conviction >= min_conviction
      - created_at >= not_older_than
      - has an `executions` row with `reject_reason='pending_capacity'`
        (the deterministic signal written by `AgentLoop._execute` when
        Feature B's gate skipped Opus and broker submission was
        blocked)
      - NO `proposal_reviews` row exists (defense-in-depth: Opus must
        not have already reviewed it via some other path)

    Tie-break: highest conviction first, then most recent. Returns at
    most one candidate per call (the caller serializes retro work).
    """
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT p.* FROM proposals p
            JOIN executions e ON e.proposal_id = p.id
            WHERE p.kind = 'trade_proposal'
              AND p.conviction >= ?
              AND p.created_at >= ?
              AND e.reject_reason = 'pending_capacity'
              AND NOT EXISTS (
                  SELECT 1 FROM proposal_reviews pr WHERE pr.proposal_id = p.id
              )
              -- Exclude proposals whose retro path already ran: any
              -- execution row that is NOT pending_capacity means the
              -- proposal has a terminal outcome (accepted, rejected,
              -- broker_unavailable, opus_reject, etc.) and must not be
              -- retro-filled again. Migration 003 allows multiple rows
              -- per proposal so this filter is necessary.
              AND NOT EXISTS (
                  SELECT 1 FROM executions e2
                  WHERE e2.proposal_id = p.id
                    AND (
                        e2.reject_reason IS NULL
                        OR e2.reject_reason != 'pending_capacity'
                    )
              )
            ORDER BY p.conviction DESC, p.created_at DESC
            LIMIT 1
            """,
            (min_conviction, not_older_than),
        ).fetchone()
    d = _row_to_dict(row)
    return ProposalRow(**d) if d is not None else None


def has_open_position(db_path: str, symbol: str) -> bool:
    """Feature C pyramiding guard — return True if `symbol` has a non-zero
    qty on its latest `positions` snapshot. Mirrors the open-position
    semantics used by `JournalRepo.get_open_positions` (latest snapshot
    per symbol with qty != 0).
    """
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT p.qty FROM positions p
            WHERE p.symbol = ?
            ORDER BY p.snapshot_at DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
    if row is None:
        return False
    return int(row["qty"]) != 0


# ---------------------------------------------------------------------------
# proposal_reviews (append-only)
# ---------------------------------------------------------------------------

def insert_proposal_review(db_path: str, row: ProposalReviewRow) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            "INSERT INTO proposal_reviews "
            "(proposal_id, model_id, prompt_version, decision, raw_response, "
            "rationale, modifications_json, input_tokens, output_tokens, "
            "cache_read_tokens, cache_creation_tokens, latency_ms, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.proposal_id,
                row.model_id,
                row.prompt_version,
                row.decision,
                row.raw_response,
                row.rationale,
                row.modifications_json,
                row.input_tokens,
                row.output_tokens,
                row.cache_read_tokens,
                row.cache_creation_tokens,
                row.latency_ms,
                row.cost_usd,
            ),
        )
        return int(cur.lastrowid or 0)


def get_proposal_review_by_proposal_id(
    db_path: str, proposal_id: int
) -> ProposalReviewRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM proposal_reviews WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
    d = _row_to_dict(row)
    return ProposalReviewRow(**d) if d is not None else None


# ---------------------------------------------------------------------------
# thesis_review_schedule + thesis_reviews (Feature A; both append-only)
# ---------------------------------------------------------------------------

def insert_thesis_review_schedule(
    db_path: str, row: ThesisReviewScheduleRow
) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            "INSERT INTO thesis_review_schedule "
            "(execution_id, due_at, scheduled_reason) VALUES (?, ?, ?)",
            (row.execution_id, row.due_at, row.scheduled_reason),
        )
        return int(cur.lastrowid or 0)


def get_latest_thesis_schedule_for_execution(
    db_path: str, execution_id: int
) -> ThesisReviewScheduleRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM thesis_review_schedule "
            "WHERE execution_id = ? ORDER BY id DESC LIMIT 1",
            (execution_id,),
        ).fetchone()
    d = _row_to_dict(row)
    return ThesisReviewScheduleRow(**d) if d is not None else None


def find_due_thesis_reviews(
    db_path: str, *, now: _dt.datetime
) -> list[ThesisReviewScheduleRow]:
    """Feature A — return schedule rows that:
      - are the LATEST per execution_id (most-recent scheduling wins);
      - have `due_at <= now`;
      - have NOT yet produced a `thesis_reviews` row (UNIQUE(schedule_id)
        ensures we never fire a schedule twice);
      - belong to an execution whose `decision = 'accepted'` (rejected
        executions never opened a position).

    Position-still-open is enforced by the caller via
    `has_open_position(symbol)` because the `proposals.symbol → execution`
    join requires extra context the caller already has.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.* FROM thesis_review_schedule s
            JOIN executions e ON e.id = s.execution_id
            WHERE s.due_at <= ?
              AND e.decision = 'accepted'
              AND s.id IN (
                  SELECT MAX(id) FROM thesis_review_schedule GROUP BY execution_id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM thesis_reviews tr WHERE tr.schedule_id = s.id
              )
            ORDER BY s.due_at ASC
            """,
            (now,),
        ).fetchall()
    return [ThesisReviewScheduleRow(**dict(r)) for r in rows]


def insert_thesis_review(db_path: str, row: ThesisReviewRow) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            "INSERT INTO thesis_reviews "
            "(execution_id, schedule_id, model_id, prompt_version, decision, "
            "raw_response, rationale, modifications_json, input_tokens, "
            "output_tokens, cache_read_tokens, cache_creation_tokens, "
            "latency_ms, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.execution_id,
                row.schedule_id,
                row.model_id,
                row.prompt_version,
                row.decision,
                row.raw_response,
                row.rationale,
                row.modifications_json,
                row.input_tokens,
                row.output_tokens,
                row.cache_read_tokens,
                row.cache_creation_tokens,
                row.latency_ms,
                row.cost_usd,
            ),
        )
        return int(cur.lastrowid or 0)


# ---------------------------------------------------------------------------
# executions (append-only)
# ---------------------------------------------------------------------------

def insert_execution(db_path: str, row: ExecutionRow) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            "INSERT INTO executions "
            "(proposal_id, decision, reject_reason, realized_size_pct, "
            "realized_dollar_size, submitted_orders_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                row.proposal_id,
                row.decision,
                row.reject_reason,
                row.realized_size_pct,
                row.realized_dollar_size,
                row.submitted_orders_json,
            ),
        )
        return int(cur.lastrowid or 0)


def get_execution_by_proposal_id(db_path: str, proposal_id: int) -> ExecutionRow | None:
    """Return the LATEST execution row for `proposal_id`.

    Migration 003 dropped UNIQUE(proposal_id), so a single proposal can
    now have multiple `executions` rows: a `pending_capacity` placeholder
    written by `AgentLoop._execute` when Feature B's gate skipped Opus,
    followed (later) by an `accepted`/`rejected` row written by Feature
    C's retro-fill. Callers want "the current outcome" — the latest row
    by id wins.
    """
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM executions WHERE proposal_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (proposal_id,),
        ).fetchone()
    d = _row_to_dict(row)
    return ExecutionRow(**d) if d is not None else None


# ---------------------------------------------------------------------------
# orders (append-only)
# ---------------------------------------------------------------------------

_ORDER_INSERT = """
INSERT INTO orders (
    execution_id, role, symbol, side, order_type, qty,
    limit_price, stop_price, tif, broker_order_id, submitted_at,
    pre_submission_bid, pre_submission_ask, pre_submission_last,
    pre_submission_quote_at, final_status, realized_fill_price,
    realized_fill_qty, realized_fill_at, realized_fee, notes
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def insert_order(db_path: str, row: OrderRow) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            _ORDER_INSERT,
            (
                row.execution_id,
                row.role,
                row.symbol,
                row.side,
                row.order_type,
                row.qty,
                row.limit_price,
                row.stop_price,
                row.tif,
                row.broker_order_id,
                row.submitted_at,
                row.pre_submission_bid,
                row.pre_submission_ask,
                row.pre_submission_last,
                row.pre_submission_quote_at,
                row.final_status,
                row.realized_fill_price,
                row.realized_fill_qty,
                row.realized_fill_at,
                row.realized_fee,
                row.notes,
            ),
        )
        return int(cur.lastrowid or 0)


def get_order_by_broker_id(db_path: str, broker_order_id: str) -> OrderRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE broker_order_id = ?", (broker_order_id,)
        ).fetchone()
    d = _row_to_dict(row)
    return OrderRow(**d) if d is not None else None


def get_orders_for_execution(db_path: str, execution_id: int) -> list[OrderRow]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM orders WHERE execution_id = ? ORDER BY id ASC",
            (execution_id,),
        ).fetchall()
    return [OrderRow(**dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# positions (append-only)
# ---------------------------------------------------------------------------

def insert_position(db_path: str, row: PositionRow) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            "INSERT INTO positions "
            "(snapshot_at, source, symbol, qty, avg_entry_price, "
            "market_value, unrealized_pnl, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.snapshot_at,
                row.source,
                row.symbol,
                row.qty,
                row.avg_entry_price,
                row.market_value,
                row.unrealized_pnl,
                row.notes,
            ),
        )
        return int(cur.lastrowid or 0)


def get_latest_position_for_symbol(db_path: str, symbol: str) -> PositionRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE symbol = ? "
            "ORDER BY snapshot_at DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    d = _row_to_dict(row)
    return PositionRow(**d) if d is not None else None


# ---------------------------------------------------------------------------
# account_snapshots (append-only)
# ---------------------------------------------------------------------------

def insert_account_snapshot(db_path: str, row: AccountSnapshotRow) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            "INSERT INTO account_snapshots "
            "(snapshot_at, equity, cash, buying_power, long_market_value, daypl) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                row.snapshot_at,
                row.equity,
                row.cash,
                row.buying_power,
                row.long_market_value,
                row.daypl,
            ),
        )
        return int(cur.lastrowid or 0)


def get_latest_account_snapshot(db_path: str) -> AccountSnapshotRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM account_snapshots ORDER BY snapshot_at DESC LIMIT 1"
        ).fetchone()
    d = _row_to_dict(row)
    return AccountSnapshotRow(**d) if d is not None else None


# ---------------------------------------------------------------------------
# kill_switch_state (append-only)
# ---------------------------------------------------------------------------

def insert_kill_switch_state(db_path: str, row: KillSwitchStateRow) -> None:
    with connect(db_path) as conn:
        _exec_write(
            conn,
            "INSERT INTO kill_switch_state(set_at, state, reason, set_by, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (row.set_at, row.state, row.reason, row.set_by, row.notes),
        )


def get_latest_kill_switch_state(db_path: str) -> KillSwitchStateRow:
    """Always returns a row — the seed row guarantees non-empty (S1.3 clarification)."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM kill_switch_state ORDER BY set_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise RuntimeError(
            "kill_switch_state has no rows; seed row missing — schema is corrupt"
        )
    return KillSwitchStateRow(**dict(row))


# ---------------------------------------------------------------------------
# universe_snapshots + universe_members (append-only)
# ---------------------------------------------------------------------------

def insert_universe_snapshot(db_path: str, row: UniverseSnapshotRow) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            "INSERT INTO universe_snapshots "
            "(snapshot_date, sec_tickers_hash, alpaca_assets_hash, "
            "yfinance_failures, member_count, is_degraded) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                row.snapshot_date,
                row.sec_tickers_hash,
                row.alpaca_assets_hash,
                row.yfinance_failures,
                row.member_count,
                row.is_degraded,
            ),
        )
        return int(cur.lastrowid or 0)


def insert_universe_member(db_path: str, row: UniverseMemberRow) -> None:
    with connect(db_path) as conn:
        _exec_write(
            conn,
            "INSERT INTO universe_members "
            "(snapshot_id, cik, ticker, exchange, market_cap, prev_close) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                row.snapshot_id,
                row.cik,
                row.ticker,
                row.exchange,
                row.market_cap,
                row.prev_close,
            ),
        )


def get_current_universe_snapshot(db_path: str) -> UniverseSnapshotRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM universe_snapshots "
            "ORDER BY snapshot_date DESC, snapshot_id DESC LIMIT 1"
        ).fetchone()
    d = _row_to_dict(row)
    return UniverseSnapshotRow(**d) if d is not None else None


def get_current_universe_member(db_path: str, ticker: str) -> UniverseMemberRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT m.* FROM universe_members m "
            "JOIN universe_snapshots s ON s.snapshot_id = m.snapshot_id "
            "WHERE m.ticker = ? "
            "ORDER BY s.snapshot_date DESC, s.snapshot_id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    d = _row_to_dict(row)
    return UniverseMemberRow(**d) if d is not None else None


# ---------------------------------------------------------------------------
# llm_calls (append-only)
# ---------------------------------------------------------------------------

def insert_llm_call(db_path: str, row: LlmCallRow) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            "INSERT INTO llm_calls "
            "(decision_id, purpose, model_id, prompt_version, "
            "input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, "
            "latency_ms, cost_usd, error_class) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.decision_id,
                row.purpose,
                row.model_id,
                row.prompt_version,
                row.input_tokens,
                row.output_tokens,
                row.cache_read_tokens,
                row.cache_creation_tokens,
                row.latency_ms,
                row.cost_usd,
                row.error_class,
            ),
        )
        return int(cur.lastrowid or 0)


def get_llm_calls_by_decision_id(db_path: str, decision_id: str) -> list[LlmCallRow]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM llm_calls WHERE decision_id = ? ORDER BY called_at ASC",
            (decision_id,),
        ).fetchall()
    return [LlmCallRow(**dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# backups (append-only)
# ---------------------------------------------------------------------------

def insert_backup(db_path: str, row: BackupRow) -> int:
    with connect(db_path) as conn:
        cur = _exec_write(
            conn,
            "INSERT INTO backups "
            "(started_at, completed_at, backup_path, db_size_bytes, sha256, "
            "verified_at, verify_result, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.started_at,
                row.completed_at,
                row.backup_path,
                row.db_size_bytes,
                row.sha256,
                row.verified_at,
                row.verify_result,
                row.notes,
            ),
        )
        return int(cur.lastrowid or 0)


def get_latest_backup(db_path: str) -> BackupRow | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM backups ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    d = _row_to_dict(row)
    return BackupRow(**d) if d is not None else None


# ---------------------------------------------------------------------------
# JournalRepo — thin object facade over the module-level functions.
#
# Stories that consume the journal (S6.1, S6.2, S6.4, S7.1, S7.4, S5.6) take a
# `JournalRepo` parameter so callers can be wired with a single dependency
# instead of plumbing `db_path` everywhere. New tables/methods are added here
# as their owning stories land — keep the surface minimal per scope.
# ---------------------------------------------------------------------------


class JournalRepo:
    """Object facade over the module-level repo functions in this file.

    Only methods needed by currently-landed stories are exposed; subsequent
    stories add new method bindings as they need them.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    # executions, orders (S6.4)
    def insert_execution(self, row: ExecutionRow) -> int:
        return insert_execution(self.db_path, row)

    def insert_order(self, row: OrderRow) -> int:
        return insert_order(self.db_path, row)

    # positions, account_snapshots (S6.5 reconciler)
    def insert_position(self, row: PositionRow) -> int:
        return insert_position(self.db_path, row)

    def insert_account_snapshot(self, row: AccountSnapshotRow) -> int:
        return insert_account_snapshot(self.db_path, row)

    # kill_switch_state (S7.1)
    def insert_kill_switch_state(self, row: KillSwitchStateRow) -> None:
        insert_kill_switch_state(self.db_path, row)

    def get_latest_kill_switch_state(self) -> KillSwitchStateRow | None:
        try:
            return get_latest_kill_switch_state(self.db_path)
        except RuntimeError:
            return None

    # /status surface (S7.2 — FR-34)
    def get_latest_account_snapshot(self) -> AccountSnapshotRow | None:
        return get_latest_account_snapshot(self.db_path)

    def get_open_positions(self) -> list[PositionRow]:
        """Open positions = latest non-zero-qty snapshot per symbol."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT p.* FROM positions p
                JOIN (
                    SELECT symbol, MAX(snapshot_at) AS latest
                    FROM positions GROUP BY symbol
                ) latest_by_symbol
                ON p.symbol = latest_by_symbol.symbol
                AND p.snapshot_at = latest_by_symbol.latest
                WHERE p.qty != 0
                ORDER BY p.symbol ASC
                """
            ).fetchall()
        return [PositionRow(**dict(r)) for r in rows]

    def get_last_filing_fetched_at(self) -> _dt.datetime | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(fetched_at) AS m FROM filings"
            ).fetchone()
        if row is None or row["m"] is None:
            return None
        v = row["m"]
        if isinstance(v, _dt.datetime):
            return v
        return _dt.datetime.fromisoformat(str(v).replace(" ", "T"))

    def get_last_proposal_created_at(self) -> _dt.datetime | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(created_at) AS m FROM proposals"
            ).fetchone()
        if row is None or row["m"] is None:
            return None
        v = row["m"]
        if isinstance(v, _dt.datetime):
            return v
        return _dt.datetime.fromisoformat(str(v).replace(" ", "T"))

    def get_mtd_inference_cost_usd(self, now: _dt.datetime) -> float:
        """Sum of llm_calls.cost_usd for the calendar month containing `now` (UTC)."""
        month_start = _dt.datetime(now.year, now.month, 1)
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM llm_calls "
                "WHERE called_at >= ?",
                (month_start,),
            ).fetchone()
        return float(row["s"]) if row is not None else 0.0

    # Auto-halt evaluator helpers (S7.4 — KS-1, KS-2, KS-3)
    def get_peak_equity_since(self, since: _dt.datetime) -> float | None:
        """Max(equity) across account_snapshots taken on/after `since`.

        Returns None if no snapshot in the window.
        """
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(equity) AS m FROM account_snapshots WHERE snapshot_at >= ?",
                (since,),
            ).fetchone()
        if row is None or row["m"] is None:
            return None
        return float(row["m"])

    # ------------------------------------------------------------------
    # S9.1 dashboard helpers (D-063). Documented per story dev-note: every
    # new helper must be listed here so future reviewers can find them.
    # ------------------------------------------------------------------

    def count_filings_since(self, since: _dt.datetime) -> int:
        """Number of `filings` rows with `fetched_at >= since`."""
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM filings WHERE fetched_at >= ?",
                (since,),
            ).fetchone()
        return int(row["c"]) if row is not None else 0

    def count_proposals_since(self, since: _dt.datetime) -> int:
        """Number of `proposals` rows with `created_at >= since`."""
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM proposals WHERE created_at >= ?",
                (since,),
            ).fetchone()
        return int(row["c"]) if row is not None else 0

    def count_executions_since(
        self, since: _dt.datetime, *, decision: str | None = None
    ) -> int:
        """Number of `executions` rows with `decided_at >= since`.

        Pass `decision="accepted"` to count submitted-trades-only (used by
        the dashboard's "trades submitted today" tile).
        """
        sql = "SELECT COUNT(*) AS c FROM executions WHERE decided_at >= ?"
        params: tuple[object, ...] = (since,)
        if decision is not None:
            sql += " AND decision = ?"
            params = (since, decision)
        with connect(self.db_path) as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["c"]) if row is not None else 0

    def get_account_snapshots_since(
        self, since: _dt.datetime
    ) -> list[AccountSnapshotRow]:
        """All `account_snapshots` rows with `snapshot_at >= since`,
        ordered ASC. Used by `/` for the 30d equity sparkline.
        """
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM account_snapshots WHERE snapshot_at >= ? "
                "ORDER BY snapshot_at ASC",
                (since,),
            ).fetchall()
        return [AccountSnapshotRow(**dict(r)) for r in rows]

    def list_recent_proposals(
        self,
        *,
        limit: int = 100,
        symbol: str | None = None,
        decision_status: str | None = None,
    ) -> list[tuple[ProposalRow, str | None]]:
        """Most-recent proposals (created_at DESC) with optional filters.

        Each result is a `(ProposalRow, filing_ticker)` tuple: the second
        element is the LEFT-JOINed `filings.issuer_ticker` so the dashboard
        can fall back to the filing-issuer ticker when `proposals.symbol`
        is NULL (typical for `kind='no_trade'`).

        `decision_status` filters on the JOINed `executions.decision` column
        (None when no execution row exists yet — those rows are dropped from
        the result when this filter is set).

        `symbol` matches against EITHER `proposals.symbol` OR
        `filings.issuer_ticker` (exact match), so an operator searching for
        "AAPL" finds rows where AAPL appears as either the analyzer-chosen
        trade symbol or the filing issuer.
        """
        sql = (
            "SELECT p.*, f.issuer_ticker AS filing_ticker FROM proposals p "
            "LEFT JOIN filings f ON f.id = p.filing_id"
        )
        params: list[object] = []
        wheres: list[str] = []
        if decision_status is not None:
            # Migration 003 allows multiple `executions` rows per
            # proposal (pending_capacity placeholder + later outcome).
            # Filter on the LATEST row per proposal so a single
            # `accepted` outcome doesn't return both rows.
            sql += (
                " JOIN executions e ON e.proposal_id = p.id "
                " AND e.id = ("
                "   SELECT MAX(id) FROM executions e2 "
                "   WHERE e2.proposal_id = p.id"
                " )"
            )
            wheres.append("e.decision = ?")
            params.append(decision_status)
        if symbol is not None:
            wheres.append("(p.symbol = ? OR f.issuer_ticker = ?)")
            params.append(symbol)
            params.append(symbol)
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY p.created_at DESC LIMIT ?"
        params.append(int(limit))
        with connect(self.db_path) as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        out: list[tuple[ProposalRow, str | None]] = []
        for r in rows:
            d = dict(r)
            ticker = d.pop("filing_ticker", None)
            out.append((ProposalRow(**d), ticker))
        return out

    def get_review_for_proposal(
        self, proposal_id: int
    ) -> ProposalReviewRow | None:
        return get_proposal_review_by_proposal_id(self.db_path, proposal_id)

    def get_execution_for_proposal(
        self, proposal_id: int
    ) -> ExecutionRow | None:
        return get_execution_by_proposal_id(self.db_path, proposal_id)

    def list_recent_executions_with_orders(
        self, limit: int = 100
    ) -> list[tuple[ExecutionRow, list[OrderRow], ProposalRow | None]]:
        """Most-recent executions DESC with their orders + the source
        proposal (for the symbol/decision_id link on `/trades`).
        """
        with connect(self.db_path) as conn:
            erows = conn.execute(
                "SELECT * FROM executions ORDER BY decided_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            out: list[tuple[ExecutionRow, list[OrderRow], ProposalRow | None]] = []
            for er in erows:
                erow = ExecutionRow(**dict(er))
                orows = conn.execute(
                    "SELECT * FROM orders WHERE execution_id = ? ORDER BY id ASC",
                    (erow.id,),
                ).fetchall()
                orders = [OrderRow(**dict(r)) for r in orows]
                prow = None
                if erow.proposal_id:
                    pr = conn.execute(
                        "SELECT * FROM proposals WHERE id = ?", (erow.proposal_id,)
                    ).fetchone()
                    if pr is not None:
                        prow = ProposalRow(**dict(pr))
                out.append((erow, orders, prow))
        return out

    def get_open_position_rows(self) -> list[PositionRow]:
        """Public alias for `get_open_positions` so the dashboard module
        does not depend on internal naming.
        """
        return self.get_open_positions()

    def llm_spend_breakdown_mtd(
        self, now: _dt.datetime
    ) -> list[dict[str, float | int | str]]:
        """Per-model spend for the calendar month containing `now`.

        Each row: model_id, calls, input_tokens, output_tokens,
        cache_read_tokens, cache_creation_tokens, cost_usd, p50_latency_ms,
        p95_latency_ms, cache_hit_pct.
        """
        month_start = _dt.datetime(now.year, now.month, 1)
        with connect(self.db_path) as conn:
            agg = conn.execute(
                """
                SELECT model_id,
                       COUNT(*) AS calls,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                       COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation_tokens,
                       COALESCE(SUM(cost_usd), 0.0) AS cost_usd
                FROM llm_calls
                WHERE called_at >= ?
                GROUP BY model_id
                ORDER BY cost_usd DESC
                """,
                (month_start,),
            ).fetchall()
            out: list[dict[str, float | int | str]] = []
            for r in agg:
                model = str(r["model_id"])
                latencies = [
                    int(x["latency_ms"])
                    for x in conn.execute(
                        "SELECT latency_ms FROM llm_calls "
                        "WHERE model_id = ? AND called_at >= ? "
                        "ORDER BY latency_ms ASC",
                        (model, month_start),
                    ).fetchall()
                ]
                p50 = _percentile(latencies, 50)
                p95 = _percentile(latencies, 95)
                cache_read = int(r["cache_read_tokens"])
                cache_creation = int(r["cache_creation_tokens"])
                input_tok = int(r["input_tokens"])
                denom = cache_read + cache_creation + input_tok
                cache_hit_pct = (cache_read / denom * 100.0) if denom > 0 else 0.0
                out.append(
                    {
                        "model_id": model,
                        "calls": int(r["calls"]),
                        "input_tokens": input_tok,
                        "output_tokens": int(r["output_tokens"]),
                        "cache_read_tokens": cache_read,
                        "cache_creation_tokens": cache_creation,
                        "cost_usd": float(r["cost_usd"]),
                        "p50_latency_ms": p50,
                        "p95_latency_ms": p95,
                        "cache_hit_pct": cache_hit_pct,
                    }
                )
        return out

    # ------------------------------------------------------------------
    # End S9.1 dashboard helpers
    # ------------------------------------------------------------------

    def get_closed_trade_pnls_chronological(self) -> list[tuple[int, float, _dt.datetime]]:
        """Realized P&L per closed `executions` row, ordered by close time ASC.

        A trade is "closed" when it has at least one fully-filled non-entry
        (stop or take_profit) order. Realized P&L = sum(sells) - sum(buys) over
        the execution's filled orders, less fees. Long-only (architecture §5).

        Returns list of (execution_id, realized_pnl_usd, closed_at).
        """
        with connect(self.db_path) as conn:
            # Closed executions: those with a non-entry filled order. Pick
            # latest such fill_at as the close time.
            rows = conn.execute(
                """
                SELECT e.id AS execution_id,
                       MAX(CASE WHEN o.role != 'entry' THEN o.realized_fill_at END)
                           AS closed_at
                FROM executions e
                JOIN orders o ON o.execution_id = e.id
                WHERE o.final_status = 'filled'
                  AND o.realized_fill_qty IS NOT NULL
                  AND o.realized_fill_price IS NOT NULL
                GROUP BY e.id
                HAVING closed_at IS NOT NULL
                ORDER BY closed_at ASC
                """
            ).fetchall()
            out: list[tuple[int, float, _dt.datetime]] = []
            for r in rows:
                eid = int(r["execution_id"])
                closed_at = r["closed_at"]
                if not isinstance(closed_at, _dt.datetime):
                    closed_at = _dt.datetime.fromisoformat(
                        str(closed_at).replace(" ", "T")
                    )
                fills = conn.execute(
                    """
                    SELECT side, realized_fill_qty AS q, realized_fill_price AS p,
                           COALESCE(realized_fee, 0.0) AS fee
                    FROM orders
                    WHERE execution_id = ?
                      AND final_status = 'filled'
                      AND realized_fill_qty IS NOT NULL
                      AND realized_fill_price IS NOT NULL
                    """,
                    (eid,),
                ).fetchall()
                pnl = 0.0
                for f in fills:
                    sign = -1.0 if f["side"] == "buy" else 1.0
                    pnl += sign * float(f["q"]) * float(f["p"]) - float(f["fee"])
                out.append((eid, pnl, closed_at))
        return out
