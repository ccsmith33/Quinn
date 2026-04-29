-- 001_init.sql — Quinn v1 initial schema (architecture §8.1).
-- D-016 (SQLite), D-023 (numbered SQL files, no Alembic).

-- Schema-version registry (architecture §8.3 / story S1.2 AC-1).
CREATE TABLE meta (
    schema_version INTEGER PRIMARY KEY,
    applied_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- filings (FR-5)
-- ----------------------------------------------------------------------------
CREATE TABLE filings (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    accession_number     TEXT NOT NULL UNIQUE,
    cik                  INTEGER NOT NULL,
    form_type            TEXT NOT NULL,
    filed_at             TIMESTAMP NOT NULL,
    fetched_at           TIMESTAMP NOT NULL,
    raw_text_path        TEXT NOT NULL,
    content_hash         TEXT NOT NULL,
    item_codes           TEXT,
    issuer_ticker        TEXT,
    ingest_state         TEXT NOT NULL DEFAULT 'ok',
    ingest_error         TEXT
);
CREATE INDEX idx_filings_cik_form ON filings(cik, form_type, filed_at DESC);
CREATE INDEX idx_filings_filed_at ON filings(filed_at DESC);

-- ----------------------------------------------------------------------------
-- prefilter_decisions (FR-11..FR-14)
-- ----------------------------------------------------------------------------
CREATE TABLE prefilter_decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id         INTEGER NOT NULL REFERENCES filings(id),
    decision          TEXT NOT NULL,
    rule_fired        TEXT NOT NULL,
    similarity_score  REAL,
    reason_detail     TEXT,
    decided_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(filing_id)
);

-- ----------------------------------------------------------------------------
-- similarity_cache (ADR-003)
-- ----------------------------------------------------------------------------
CREATE TABLE similarity_cache (
    cik                   INTEGER NOT NULL,
    form_type             TEXT NOT NULL,
    accession_number      TEXT NOT NULL,
    minhash_blob          BLOB NOT NULL,
    tfidf_vectorizer_path TEXT NOT NULL,
    tfidf_vector_path     TEXT NOT NULL,
    fitted_at             TIMESTAMP NOT NULL,
    PRIMARY KEY (cik, form_type, accession_number)
);
CREATE INDEX idx_simcache_cik_form ON similarity_cache(cik, form_type, fitted_at DESC);

-- ----------------------------------------------------------------------------
-- prompts (ADR-005, FR-29)
-- ----------------------------------------------------------------------------
CREATE TABLE prompts (
    prompt_version   TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    file_path        TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    first_seen_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- proposals (FR-19, FR-27)
-- ----------------------------------------------------------------------------
CREATE TABLE proposals (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id                INTEGER NOT NULL REFERENCES filings(id),
    decision_id              TEXT NOT NULL UNIQUE,
    model_id                 TEXT NOT NULL,
    prompt_version           TEXT NOT NULL REFERENCES prompts(prompt_version),
    raw_response             TEXT NOT NULL,
    kind                     TEXT NOT NULL,
    symbol                   TEXT,
    direction                TEXT,
    size_pct_requested       REAL,
    conviction               INTEGER,
    thesis                   TEXT,
    input_tokens             INTEGER NOT NULL,
    output_tokens            INTEGER NOT NULL,
    cache_read_tokens        INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens    INTEGER NOT NULL DEFAULT 0,
    latency_ms               INTEGER NOT NULL,
    cost_usd                 REAL NOT NULL,
    created_at               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reasoning_quality        TEXT,
    reasoning_notes          TEXT
);
CREATE INDEX idx_proposals_filing ON proposals(filing_id);
CREATE INDEX idx_proposals_symbol_created ON proposals(symbol, created_at DESC);

-- ----------------------------------------------------------------------------
-- proposal_reviews (FR-17, FR-28)
-- ----------------------------------------------------------------------------
CREATE TABLE proposal_reviews (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id              INTEGER NOT NULL REFERENCES proposals(id),
    model_id                 TEXT NOT NULL,
    prompt_version           TEXT NOT NULL REFERENCES prompts(prompt_version),
    decision                 TEXT NOT NULL,
    raw_response             TEXT NOT NULL,
    rationale                TEXT NOT NULL,
    modifications_json       TEXT,
    input_tokens             INTEGER NOT NULL,
    output_tokens            INTEGER NOT NULL,
    cache_read_tokens        INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens    INTEGER NOT NULL DEFAULT 0,
    latency_ms               INTEGER NOT NULL,
    cost_usd                 REAL NOT NULL,
    created_at               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(proposal_id)
);

-- ----------------------------------------------------------------------------
-- executions (FR-20, FR-22, FR-27)
-- ----------------------------------------------------------------------------
CREATE TABLE executions (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id              INTEGER NOT NULL REFERENCES proposals(id),
    decision                 TEXT NOT NULL,
    reject_reason            TEXT,
    realized_size_pct        REAL,
    realized_dollar_size     REAL,
    submitted_orders_json    TEXT,
    decided_at               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(proposal_id)
);

-- ----------------------------------------------------------------------------
-- orders (FR-21, FR-27)
-- ----------------------------------------------------------------------------
CREATE TABLE orders (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id             INTEGER NOT NULL REFERENCES executions(id),
    role                     TEXT NOT NULL,
    symbol                   TEXT NOT NULL,
    side                     TEXT NOT NULL,
    order_type               TEXT NOT NULL,
    qty                      INTEGER NOT NULL,
    limit_price              REAL,
    stop_price               REAL,
    tif                      TEXT NOT NULL,
    broker_order_id          TEXT NOT NULL UNIQUE,
    submitted_at             TIMESTAMP NOT NULL,
    pre_submission_bid       REAL,
    pre_submission_ask       REAL,
    pre_submission_last      REAL,
    pre_submission_quote_at  TIMESTAMP,
    final_status             TEXT,
    realized_fill_price      REAL,
    realized_fill_qty        INTEGER,
    realized_fill_at         TIMESTAMP,
    realized_fee             REAL DEFAULT 0,
    notes                    TEXT
);
CREATE INDEX idx_orders_symbol_submitted ON orders(symbol, submitted_at DESC);

-- ----------------------------------------------------------------------------
-- positions (FR-24)
-- ----------------------------------------------------------------------------
CREATE TABLE positions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at       TIMESTAMP NOT NULL,
    source            TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    qty               INTEGER NOT NULL,
    avg_entry_price   REAL NOT NULL,
    market_value      REAL NOT NULL,
    unrealized_pnl    REAL NOT NULL,
    notes             TEXT
);
CREATE INDEX idx_positions_symbol_snap ON positions(symbol, snapshot_at DESC);

-- ----------------------------------------------------------------------------
-- account_snapshots (FR-24, KS-1, KS-2)
-- ----------------------------------------------------------------------------
CREATE TABLE account_snapshots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at       TIMESTAMP NOT NULL,
    equity            REAL NOT NULL,
    cash              REAL NOT NULL,
    buying_power      REAL NOT NULL,
    long_market_value REAL NOT NULL,
    daypl             REAL NOT NULL
);
CREATE INDEX idx_account_snap_at ON account_snapshots(snapshot_at DESC);

-- ----------------------------------------------------------------------------
-- kill_switch_state (ADR-004, FR-32)
-- ----------------------------------------------------------------------------
CREATE TABLE kill_switch_state (
    set_at      TIMESTAMP PRIMARY KEY,
    state       TEXT NOT NULL,
    reason      TEXT NOT NULL,
    set_by      TEXT NOT NULL,
    notes       TEXT
);

-- Seed row so kill_switch_state is never empty (S1.3 tactical clarification).
INSERT INTO kill_switch_state(set_at, state, reason, set_by, notes)
VALUES ('1970-01-01 00:00:00', 'active', 'boot', 'system', NULL);

-- ----------------------------------------------------------------------------
-- universe_snapshots + universe_members (ADR-006, FR-7..FR-10)
-- ----------------------------------------------------------------------------
CREATE TABLE universe_snapshots (
    snapshot_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date      DATE NOT NULL UNIQUE,
    sec_tickers_hash   TEXT NOT NULL,
    alpaca_assets_hash TEXT NOT NULL,
    yfinance_failures  INTEGER NOT NULL DEFAULT 0,
    member_count       INTEGER NOT NULL,
    is_degraded        INTEGER NOT NULL DEFAULT 0,
    created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE universe_members (
    snapshot_id        INTEGER NOT NULL REFERENCES universe_snapshots(snapshot_id),
    cik                INTEGER NOT NULL,
    ticker             TEXT NOT NULL,
    exchange           TEXT NOT NULL,
    market_cap         REAL NOT NULL,
    prev_close         REAL NOT NULL,
    PRIMARY KEY (snapshot_id, ticker)
);
CREATE INDEX idx_um_cik ON universe_members(cik);
CREATE INDEX idx_um_ticker_snap ON universe_members(ticker, snapshot_id DESC);

-- ----------------------------------------------------------------------------
-- llm_calls (NFR-8)
-- ----------------------------------------------------------------------------
CREATE TABLE llm_calls (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id              TEXT NOT NULL,
    purpose                  TEXT NOT NULL,
    model_id                 TEXT NOT NULL,
    prompt_version           TEXT NOT NULL REFERENCES prompts(prompt_version),
    input_tokens             INTEGER NOT NULL,
    output_tokens            INTEGER NOT NULL,
    cache_read_tokens        INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens    INTEGER NOT NULL DEFAULT 0,
    latency_ms               INTEGER NOT NULL,
    cost_usd                 REAL NOT NULL,
    error_class              TEXT,
    called_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_llm_calls_decision ON llm_calls(decision_id);
CREATE INDEX idx_llm_calls_called_at ON llm_calls(called_at DESC);

-- ----------------------------------------------------------------------------
-- backups (FR-31)
-- ----------------------------------------------------------------------------
CREATE TABLE backups (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TIMESTAMP NOT NULL,
    completed_at    TIMESTAMP,
    backup_path     TEXT NOT NULL,
    db_size_bytes   INTEGER NOT NULL,
    sha256          TEXT NOT NULL,
    verified_at     TIMESTAMP,
    verify_result   TEXT,
    notes           TEXT
);
