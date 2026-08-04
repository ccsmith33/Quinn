-- 009_desk_memory.sql — desk-level memory store (LLM memory feature).
--
-- One row per version of a desk-memory artifact. Two `kind`s:
--   doctrine   → distilled study base rates / trading doctrine
--   synthesis  → post-mortem synthesis produced by the desk_journal path
--
-- Versioned + soft-deactivated rather than rewritten in place, so the
-- provenance of every doctrine/synthesis the LLM was shown survives
-- (append-only history; `active = 1` marks the row a provider serves).
-- The provider devs own the read/write semantics; this migration only
-- pins the schema + numbering. CREATE-only — no table rebuild, so the
-- migration-003 FK gotcha does not apply.

CREATE TABLE desk_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL CHECK(kind IN ('doctrine', 'synthesis')),
    content     TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    active      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX idx_desk_memory_active_kind ON desk_memory(kind, active);
