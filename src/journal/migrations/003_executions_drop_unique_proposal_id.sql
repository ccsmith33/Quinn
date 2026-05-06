-- 003_executions_drop_unique_proposal_id.sql — Feature B/C HIGH-1 fix.
--
-- Feature B writes an `executions` row with reject_reason='pending_capacity'
-- when the analyzer's KS-5 capacity gate skips Opus. Feature C's retro path
-- later runs Opus and (on accept) drives the validator/sizer/submitter chain,
-- which writes a SECOND `executions` row for the same proposal (an
-- 'accepted' outcome). The original UNIQUE(proposal_id) constraint blocks
-- this — so we drop it.
--
-- SQLite has no ALTER TABLE DROP CONSTRAINT; we rebuild the table.
-- Foreign-key targets to executions.id (orders.execution_id, the new
-- thesis_review_schedule.execution_id, thesis_reviews.execution_id) are
-- preserved because the rebuild uses INSERT INTO ... SELECT, keeping
-- existing ids stable.
--
-- Append-only invariant: still respected — we are creating a new table
-- with the same data, not UPDATEing or DELETEing existing rows. Once the
-- rebuild completes, the only writes that hit `executions` remain INSERTs.
-- The downstream defense-in-depth scan
-- (`tests/journal/test_repo.py::test_no_update_or_delete_on_append_only_tables`)
-- looks at `src/journal/repo.py`, not migration files, so this rebuild
-- does not violate the policy.
--
-- Consumers that previously assumed "at most one execution row per
-- proposal" are updated separately in `src/journal/repo.py` to select the
-- LATEST row by id.

CREATE TABLE executions_new (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id              INTEGER NOT NULL REFERENCES proposals(id),
    decision                 TEXT NOT NULL,
    reject_reason            TEXT,
    realized_size_pct        REAL,
    realized_dollar_size     REAL,
    submitted_orders_json    TEXT,
    decided_at               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO executions_new (
    id, proposal_id, decision, reject_reason, realized_size_pct,
    realized_dollar_size, submitted_orders_json, decided_at
)
SELECT
    id, proposal_id, decision, reject_reason, realized_size_pct,
    realized_dollar_size, submitted_orders_json, decided_at
FROM executions;

DROP TABLE executions;

ALTER TABLE executions_new RENAME TO executions;

CREATE INDEX idx_executions_proposal_id ON executions(proposal_id, id DESC);
