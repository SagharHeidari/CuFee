-- Customer Experience AI - database schema
-- Written in portable SQL: runs on PostgreSQL (production) and SQLite (local/tests).

DROP TABLE IF EXISTS feedback_analysis;
DROP TABLE IF EXISTS topics;
DROP TABLE IF EXISTS model_runs;
DROP TABLE IF EXISTS feedback;

-- One row per piece of customer feedback (PII already masked).
CREATE TABLE feedback (
    feedback_id       VARCHAR(32)  PRIMARY KEY,
    created_at        TIMESTAMP    NOT NULL,
    feedback_date     DATE         NOT NULL,
    feedback_week     DATE         NOT NULL,
    feedback_month    CHAR(7)      NOT NULL,      -- 'YYYY-MM'
    channel           VARCHAR(32)  NOT NULL,
    plan              VARCHAR(32)  NOT NULL,
    region            VARCHAR(32)  NOT NULL,
    customer_segment  VARCHAR(32)  NOT NULL,
    rating            INTEGER      CHECK (rating BETWEEN 1 AND 5),
    nps_group         VARCHAR(16),                -- promoter / passive / detractor
    text              TEXT         NOT NULL,
    clean_text        TEXT         NOT NULL,
    word_count        INTEGER      NOT NULL
);

CREATE INDEX idx_feedback_date    ON feedback (feedback_date);
CREATE INDEX idx_feedback_month   ON feedback (feedback_month);
CREATE INDEX idx_feedback_channel ON feedback (channel);

-- Complaint topics discovered by the topic model (fitted on complaints only).
CREATE TABLE topics (
    topic_id    INTEGER      PRIMARY KEY,
    topic_label VARCHAR(128) NOT NULL,
    top_terms   TEXT         NOT NULL,
    size        INTEGER      NOT NULL
);

-- NLP/AI output per feedback.
CREATE TABLE feedback_analysis (
    feedback_id        VARCHAR(32)  PRIMARY KEY REFERENCES feedback (feedback_id),
    sentiment_label    VARCHAR(16)  NOT NULL,     -- negative / neutral / positive
    sentiment_score    REAL         NOT NULL,     -- P(positive) - P(negative), in [-1, 1]
    sentiment_confidence REAL       NOT NULL,
    topic_id           INTEGER      REFERENCES topics (topic_id),  -- NULL for non-complaints
    complaint_category VARCHAR(32)  NOT NULL,
    is_complaint       INTEGER      NOT NULL      -- 0/1, portable boolean
);

CREATE INDEX idx_analysis_category ON feedback_analysis (complaint_category);
CREATE INDEX idx_analysis_sentiment ON feedback_analysis (sentiment_label);

-- Model evaluation history, one row per pipeline run.
CREATE TABLE model_runs (
    run_id       VARCHAR(32)  PRIMARY KEY,
    run_at       TIMESTAMP    NOT NULL,
    model_name   VARCHAR(64)  NOT NULL,
    n_train      INTEGER      NOT NULL,
    n_test       INTEGER      NOT NULL,
    accuracy     REAL         NOT NULL,
    macro_f1     REAL         NOT NULL,
    metrics_json TEXT         NOT NULL
);
