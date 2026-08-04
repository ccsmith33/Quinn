-- 007_chopping_block.sql — pre-ranked displacement sacrifice list
-- (chopping block). Written once per day after the pre-market thesis
-- sweep completes; consumed by execution/displacement.py so demand-driven
-- swaps into high-conviction signals never have to improvise victim
-- selection at fire time.
--
-- CREATE-only, no rebuilds — the migration-003 FK gotcha does not apply.
--
-- One row per sweep-reviewed, NON-armed open position per block_date.
-- `rank` is 1-based, 1 = first to sell (most expendable). The whole
-- day's block is rewritten atomically each sweep (delete-then-insert on
-- block_date), so a mid-morning re-sweep replaces rather than duplicates.
-- Armed (trail-engaged) positions are NEVER written here: they are
-- untouchable by displacement at every conviction.

CREATE TABLE chopping_block (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    block_date      TEXT NOT NULL,            -- ET calendar date, YYYY-MM-DD
    execution_id    INTEGER NOT NULL REFERENCES executions(id),
    symbol          TEXT NOT NULL,
    rank            INTEGER NOT NULL,         -- 1 = first to sell
    expendability   INTEGER NOT NULL,         -- 1 (last) .. 5 (first) to sacrifice
    reason          TEXT,                     -- reviewer's one-line expendability_reason
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_chopping_block_date ON chopping_block(block_date);
CREATE INDEX idx_chopping_block_date_rank ON chopping_block(block_date, rank);
