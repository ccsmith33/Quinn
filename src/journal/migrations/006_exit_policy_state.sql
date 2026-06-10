-- 006_exit_policy_state.sql — trailing-ratchet operational state
-- (D-079 §3.5, ADR-011). CREATE-only, no rebuilds — the migration-003
-- FK gotcha does not apply.
--
-- One row per execution under trailing management. This table is
-- OPERATIONAL state, updated in place; the audit trail of every stop
-- movement is the append-only `orders` replacement chain, so NFR-16 is
-- not diluted (delta §3.5 state note).
--
-- `stop_order_journal_id` points at the `orders` row of the CURRENT
-- live protective stop so the ratchet keeps tracking the right broker
-- order across replacements (ratchet steps and thesis adjust_stop both
-- rotate it).

CREATE TABLE exit_policy_state (
    execution_id            INTEGER PRIMARY KEY REFERENCES executions(id),
    symbol                  TEXT NOT NULL,
    trail_distance_pct      REAL NOT NULL,
    trail_engaged           INTEGER NOT NULL DEFAULT 0,
    high_water_mark         REAL NOT NULL,
    stop_order_journal_id   INTEGER REFERENCES orders(id),
    updated_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_exit_policy_state_symbol ON exit_policy_state(symbol);
