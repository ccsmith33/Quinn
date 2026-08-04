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

-- Hardening (review advisory #9): at most ONE active row per kind. This
-- makes the "the active doctrine/synthesis" read unambiguous at the
-- schema level, not just by provider convention.
--
-- ORDERING CONSTRAINT for version publishers (desk-journal synthesis
-- writer + any future doctrine updater): within the single publish
-- transaction, DEACTIVATE the old row(s) BEFORE inserting the new
-- active row — insert-first trips this index while the prior version is
-- still active. (SQLite enforces the unique index per statement; there
-- is no deferred mode.)
CREATE UNIQUE INDEX idx_desk_memory_one_active
    ON desk_memory(kind) WHERE active = 1;
