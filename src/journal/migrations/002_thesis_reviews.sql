-- 002_thesis_reviews.sql — Feature A schedule + outcome tables.
--
-- Captures the time-horizon-driven thesis re-review pass that fires
-- when neither stop nor take-profit has triggered by the proposal's
-- declared horizon. Both tables are append-only — schedule rows are
-- written at entry and re-written on `hold` / `adjust_stop` outcomes;
-- the canonical "current schedule" per execution is the latest row.

CREATE TABLE thesis_review_schedule (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id        INTEGER NOT NULL REFERENCES executions(id),
    due_at              TIMESTAMP NOT NULL,
    scheduled_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    scheduled_reason    TEXT NOT NULL  -- 'entry' | 'hold' | 'adjust_stop'
);
CREATE INDEX idx_thesis_review_schedule_due
    ON thesis_review_schedule(due_at);
CREATE INDEX idx_thesis_review_schedule_execution
    ON thesis_review_schedule(execution_id, id DESC);

CREATE TABLE thesis_reviews (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id             INTEGER NOT NULL REFERENCES executions(id),
    schedule_id              INTEGER NOT NULL REFERENCES thesis_review_schedule(id),
    model_id                 TEXT NOT NULL,
    prompt_version           TEXT NOT NULL REFERENCES prompts(prompt_version),
    decision                 TEXT NOT NULL,  -- 'hold' | 'close' | 'adjust_stop'
    raw_response             TEXT NOT NULL,
    rationale                TEXT NOT NULL,
    modifications_json       TEXT,           -- e.g. {"new_stop_price": 5.40}
    input_tokens             INTEGER NOT NULL,
    output_tokens            INTEGER NOT NULL,
    cache_read_tokens        INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens    INTEGER NOT NULL DEFAULT 0,
    latency_ms               INTEGER NOT NULL,
    cost_usd                 REAL NOT NULL,
    created_at               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(schedule_id)  -- a schedule fires at most once
);
CREATE INDEX idx_thesis_reviews_execution
    ON thesis_reviews(execution_id, created_at DESC);
