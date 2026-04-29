"""Pydantic v2 row models mirroring architecture §8.1 tables."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FilingRow(_Row):
    id: int | None = None
    accession_number: str
    cik: int
    form_type: str
    filed_at: dt.datetime
    fetched_at: dt.datetime
    raw_text_path: str
    content_hash: str
    item_codes: str | None = None
    issuer_ticker: str | None = None
    ingest_state: str = "ok"
    ingest_error: str | None = None


class PrefilterDecisionRow(_Row):
    id: int | None = None
    filing_id: int
    decision: str
    rule_fired: str
    similarity_score: float | None = None
    reason_detail: str | None = None
    decided_at: dt.datetime | None = None


class SimilarityCacheRow(_Row):
    cik: int
    form_type: str
    accession_number: str
    minhash_blob: bytes
    tfidf_vectorizer_path: str
    tfidf_vector_path: str
    fitted_at: dt.datetime


class PromptRow(_Row):
    prompt_version: str
    name: str
    file_path: str
    content_hash: str
    first_seen_at: dt.datetime | None = None


class ProposalRow(_Row):
    id: int | None = None
    filing_id: int
    decision_id: str
    model_id: str
    prompt_version: str
    raw_response: str
    kind: str
    symbol: str | None = None
    direction: str | None = None
    size_pct_requested: float | None = None
    conviction: int | None = None
    thesis: str | None = None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    latency_ms: int
    cost_usd: float
    created_at: dt.datetime | None = None
    reasoning_quality: str | None = None
    reasoning_notes: str | None = None


class ProposalReviewRow(_Row):
    id: int | None = None
    proposal_id: int
    model_id: str
    prompt_version: str
    decision: str
    raw_response: str
    rationale: str
    modifications_json: str | None = None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    latency_ms: int
    cost_usd: float
    created_at: dt.datetime | None = None


class ExecutionRow(_Row):
    id: int | None = None
    proposal_id: int
    decision: str
    reject_reason: str | None = None
    realized_size_pct: float | None = None
    realized_dollar_size: float | None = None
    submitted_orders_json: str | None = None
    decided_at: dt.datetime | None = None


class OrderRow(_Row):
    id: int | None = None
    execution_id: int
    role: str
    symbol: str
    side: str
    order_type: str
    qty: int
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str
    broker_order_id: str
    submitted_at: dt.datetime
    pre_submission_bid: float | None = None
    pre_submission_ask: float | None = None
    pre_submission_last: float | None = None
    pre_submission_quote_at: dt.datetime | None = None
    final_status: str | None = None
    realized_fill_price: float | None = None
    realized_fill_qty: int | None = None
    realized_fill_at: dt.datetime | None = None
    realized_fee: float | None = 0.0
    notes: str | None = None


class PositionRow(_Row):
    id: int | None = None
    snapshot_at: dt.datetime
    source: str
    symbol: str
    qty: int
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float
    notes: str | None = None


class AccountSnapshotRow(_Row):
    id: int | None = None
    snapshot_at: dt.datetime
    equity: float
    cash: float
    buying_power: float
    long_market_value: float
    daypl: float


class KillSwitchStateRow(_Row):
    set_at: dt.datetime
    state: str
    reason: str
    set_by: str
    notes: str | None = None


class UniverseSnapshotRow(_Row):
    snapshot_id: int | None = None
    snapshot_date: dt.date
    sec_tickers_hash: str
    alpaca_assets_hash: str
    yfinance_failures: int = 0
    member_count: int
    is_degraded: int = 0
    created_at: dt.datetime | None = None


class UniverseMemberRow(_Row):
    snapshot_id: int
    cik: int
    ticker: str
    exchange: str
    market_cap: float
    prev_close: float


class LlmCallRow(_Row):
    id: int | None = None
    decision_id: str
    purpose: str
    model_id: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    latency_ms: int
    cost_usd: float
    error_class: str | None = None
    called_at: dt.datetime | None = None


class BackupRow(_Row):
    id: int | None = None
    started_at: dt.datetime
    completed_at: dt.datetime | None = None
    backup_path: str
    db_size_bytes: int
    sha256: str
    verified_at: dt.datetime | None = None
    verify_result: str | None = None
    notes: str | None = None
