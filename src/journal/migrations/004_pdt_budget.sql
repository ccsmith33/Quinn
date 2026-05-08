-- 004_pdt_budget.sql — PDT day-trade budget feature (ADR-009).
-- PDT-SUNSET-2026-06-04: drop these tables when FINRA retires the PDT
-- rule. Migration is CREATE-only (no rebuild) per ADR-009 §"Migration
-- FK gotcha"; the runner's PRAGMA foreign_keys=OFF toggle is therefore
-- a no-op for this migration.

CREATE TABLE virtual_exits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id    INTEGER NOT NULL REFERENCES executions(id),
    proposal_id     INTEGER NOT NULL REFERENCES proposals(id),
    symbol          TEXT NOT NULL,
    qty             INTEGER NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('stop','tp')),
    entry_price     REAL NOT NULL,
    stop_price      REAL,
    tp_price        REAL,
    state           TEXT NOT NULL DEFAULT 'active'
                    CHECK (state IN ('active','submitted','obsolete')),
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    submitted_at    TIMESTAMP,
    submitted_broker_order_id TEXT,
    notes           TEXT
);
CREATE INDEX idx_virtual_exits_active_symbol
    ON virtual_exits(symbol, state) WHERE state='active';
CREATE INDEX idx_virtual_exits_execution ON virtual_exits(execution_id);

CREATE TABLE deferred_sells (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    virtual_exit_id INTEGER NOT NULL REFERENCES virtual_exits(id),
    execution_id    INTEGER NOT NULL REFERENCES executions(id),
    proposal_id     INTEGER NOT NULL REFERENCES proposals(id),
    symbol          TEXT NOT NULL,
    qty             INTEGER NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('stop','tp','thesis_close')),
    trigger_price   REAL NOT NULL,
    ev_at_defer     REAL NOT NULL,
    deferred_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deferred_reason TEXT NOT NULL CHECK (deferred_reason IN ('ev_lost','pdt_403')),
    replayed_at     TIMESTAMP,
    replay_broker_order_id TEXT,
    notes           TEXT
);
CREATE INDEX idx_deferred_sells_unreplayed
    ON deferred_sells(deferred_at) WHERE replayed_at IS NULL;
