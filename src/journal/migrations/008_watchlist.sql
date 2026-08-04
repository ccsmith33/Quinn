-- 008_watchlist.sql — deferred-entry watchlist (capital-blocked
-- high-conviction proposals get a bounded second chance instead of
-- vanishing). CREATE-only, no rebuilds — the migration-003 FK gotcha
-- does not apply.
--
-- One row per enrollment. `status` lifecycle (terminal states set
-- `resolved_at`):
--   pending        → awaiting a retry window
--   entered        → retried through the NORMAL execution path and the
--                    broker accepted the submission
--   expired        → past expires_at (trading days, computed at enroll)
--   skipped_chase  → current last ran above
--                    reference_price * (1 + watchlist_max_chase_pct/100)
--   skipped_held   → symbol already held when the retry fired
--
-- `reference_price` is the broker NBBO `last` captured by the agent
-- loop at decision time — the SAME quote the SizingEngine sized
-- against (app/loop.py fetches broker.get_quote() immediately before
-- sizing; for a validator-stage `insufficient_capital` reject the loop
-- fetches the same quote source at enrollment time).
--
-- The partial unique index enforces "one ACTIVE row per symbol";
-- resolved rows are history and do not block a later re-enrollment of
-- the same symbol on a new filing.

CREATE TABLE watchlist (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id      INTEGER NOT NULL REFERENCES proposals(id),
    symbol           TEXT NOT NULL,
    conviction       INTEGER NOT NULL,
    reference_price  REAL NOT NULL,
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at       TIMESTAMP NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    resolved_at      TIMESTAMP,
    notes            TEXT
);
CREATE INDEX idx_watchlist_status_created ON watchlist(status, created_at ASC);
CREATE UNIQUE INDEX idx_watchlist_pending_symbol ON watchlist(symbol) WHERE status = 'pending';
