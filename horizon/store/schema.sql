-- Durable store for the run. Preserves the EXPERIMENTS.md section 6 JSONL schema
-- field-for-field as columns; `make export-jsonl` regenerates that exact JSONL,
-- so the protocol's reproducibility claim holds unchanged.
--
-- Postgres rather than files because concurrent workers need atomic leasing and
-- the dashboard needs aggregate queries over ~115k rows while the run is live.

CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    cell              JSONB       NOT NULL,
    model_snapshot    TEXT,
    horizon           INTEGER     NOT NULL DEFAULT 100,
    v_star            DOUBLE PRECISION,
    status            TEXT        NOT NULL DEFAULT 'pending',
    spend_cap_usd     DOUBLE PRECISION,      -- NULL means uncapped (the default)
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at       TIMESTAMPTZ
);

-- The unit of parallelism. Days within an episode are sequential by necessity;
-- episodes are independent, so they are the axis workers fan out over.
CREATE TABLE IF NOT EXISTS work_units (
    run_id            TEXT        NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    episode           INTEGER     NOT NULL,
    status            TEXT        NOT NULL DEFAULT 'pending',
    lease_owner       TEXT,
    lease_expires_at  TIMESTAMPTZ,
    attempts          INTEGER     NOT NULL DEFAULT 0,
    last_error        TEXT,
    PRIMARY KEY (run_id, episode)
);

CREATE INDEX IF NOT EXISTS work_units_claimable
    ON work_units (run_id, status, lease_expires_at);

CREATE TABLE IF NOT EXISTS agent_days (
    run_id            TEXT        NOT NULL,
    episode           INTEGER     NOT NULL,
    day               INTEGER     NOT NULL,
    holding           SMALLINT    NOT NULL,
    age               INTEGER     NOT NULL,
    wealth            DOUBLE PRECISION NOT NULL,
    prompt_tokens     INTEGER     NOT NULL DEFAULT 0,
    completion_tokens INTEGER     NOT NULL DEFAULT 0,
    thought_tokens    INTEGER     NOT NULL DEFAULT 0,
    cache_hit         BOOLEAN,
    raw_completion    TEXT,
    thought_text      TEXT,
    parsed_belief     DOUBLE PRECISION,
    parsed_action     SMALLINT,
    repair_used       BOOLEAN     NOT NULL DEFAULT FALSE,
    error_class       TEXT        NOT NULL DEFAULT 'ok',
    -- Joined from the solver at write time so every analysis is computable from
    -- this table alone (EXPERIMENTS.md section 6).
    truth_m_t         DOUBLE PRECISION,
    truth_h_t         DOUBLE PRECISION,
    truth_pi_star     SMALLINT,
    truth_q_loss      DOUBLE PRECISION,
    latency_ms        DOUBLE PRECISION,
    ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, episode, day)
);

-- Added after the first cell was already running: the reasoning trace turned
-- out to be retrievable, and a live run should not have to be discarded to
-- start recording it.
ALTER TABLE agent_days ADD COLUMN IF NOT EXISTS thought_text TEXT;

CREATE INDEX IF NOT EXISTS agent_days_by_run ON agent_days (run_id, day);
CREATE INDEX IF NOT EXISTS agent_days_errors ON agent_days (run_id, error_class);

CREATE TABLE IF NOT EXISTS episodes (
    run_id            TEXT        NOT NULL,
    episode           INTEGER     NOT NULL,
    true_exit_day     INTEGER     NOT NULL,
    terminal_wealth   DOUBLE PRECISION NOT NULL,
    regret            DOUBLE PRECISION,
    holdings          INTEGER[]   NOT NULL,
    beliefs           DOUBLE PRECISION[],
    switch_days       INTEGER[]   NOT NULL DEFAULT '{}',
    format_error_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
    empty_trunc_rate  DOUBLE PRECISION NOT NULL DEFAULT 0,
    finished_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, episode)
);

-- Token accounting. Thought tokens get their own column because they dominate
-- output cost on this model and were not modelled in the original budget.
CREATE TABLE IF NOT EXISTS spend_ledger (
    id                BIGSERIAL PRIMARY KEY,
    run_id            TEXT        NOT NULL,
    ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
    prompt_tokens     BIGINT      NOT NULL DEFAULT 0,
    completion_tokens BIGINT      NOT NULL DEFAULT 0,
    thought_tokens    BIGINT      NOT NULL DEFAULT 0,
    usd               DOUBLE PRECISION NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS spend_ledger_by_run ON spend_ledger (run_id, ts);
