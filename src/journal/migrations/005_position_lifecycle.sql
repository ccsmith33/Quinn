-- 005_position_lifecycle.sql — Option B WS1 position-truth contract
-- (D-078, architecture-option-b-2026-06-09.md §2.5, ADR-010).
-- Additive only: the orders fill columns already exist (001_init.sql);
-- this migration adds the indexes the FillIngestor and the lifecycle
-- classifier query on every reconcile tick. No table rebuilds — the
-- migration-003 FK gotcha does not apply.

CREATE INDEX IF NOT EXISTS idx_orders_pending_fill
    ON orders(symbol) WHERE final_status IS NULL;

CREATE INDEX IF NOT EXISTS idx_positions_symbol_latest
    ON positions(symbol, snapshot_at DESC);
