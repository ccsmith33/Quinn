-- 010_trade_postmortems.sql — per-trade post-mortem record (LLM memory
-- feature). One row per closed execution the desk_journal path has
-- synthesized: the symbol_history provider serves these by symbol, and
-- the desk_journal path rolls them up into `desk_memory` synthesis rows.
--
-- `exit_quality_pct` is a provider-owned scalar (how much of the
-- available move the exit captured); the columns are deliberately loose
-- TEXT/REAL so the provider devs own the semantics. Append-only.
-- CREATE-only — no table rebuild, so the migration-003 FK gotcha does
-- not apply.

CREATE TABLE trade_postmortems (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id      INTEGER NOT NULL REFERENCES executions(id),
    symbol            TEXT NOT NULL,
    thesis_summary    TEXT,
    outcome_summary   TEXT,
    exit_quality_pct  REAL,
    lesson            TEXT,
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_trade_postmortems_symbol ON trade_postmortems(symbol);
