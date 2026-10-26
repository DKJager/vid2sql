-- build 7fd77e00de7f
-- vid2sql canonical relational schema: six tables derived from video, plus
-- two agent tables (agent_trace, review_queue) recording the agent loop's
-- plan/review decisions and the segments flagged for human review.
--
-- Portable DDL: runs unmodified on SQLite (dev writer) and DuckDB, and uses
-- only standard types so it is also valid on PostgreSQL. Timestamps are
-- ISO-8601 UTC text ("2026-10-03T00:00:00Z"); *_s columns are seconds from the
-- start of the video. Every analytic row carries a confidence in [0, 1]
-- (the agent tables hold decisions, not analytic measurements, and have none).

CREATE TABLE IF NOT EXISTS videos (
    video_id         TEXT PRIMARY KEY,
    uri              TEXT NOT NULL,
    filename         TEXT NOT NULL,
    source_url       TEXT NOT NULL,
    license          TEXT NOT NULL,
    processed_at_utc TEXT NOT NULL,
    recording_start_utc TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id   TEXT PRIMARY KEY,
    video_id     TEXT NOT NULL REFERENCES videos (video_id),
    start_s      DOUBLE PRECISION NOT NULL,
    end_s        DOUBLE PRECISION NOT NULL,
    reason       TEXT NOT NULL,
    motion_score DOUBLE PRECISION NOT NULL DEFAULT 0.0
                 CHECK (motion_score >= 0.0 AND motion_score <= 1.0),
    CHECK (end_s >= start_s)
);

CREATE TABLE IF NOT EXISTS events (
    event_id            TEXT PRIMARY KEY,
    video_id            TEXT NOT NULL REFERENCES videos (video_id),
    segment_id          TEXT REFERENCES segments (segment_id),
    event_type          TEXT NOT NULL,
    start_timestamp_utc TEXT NOT NULL,
    end_timestamp_utc   TEXT NOT NULL,
    confidence          DOUBLE PRECISION NOT NULL
                        CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE TABLE IF NOT EXISTS objects (
    object_id    TEXT PRIMARY KEY,
    video_id     TEXT NOT NULL REFERENCES videos (video_id),
    track_id     TEXT NOT NULL,
    class_label  TEXT NOT NULL,
    first_seen_s DOUBLE PRECISION NOT NULL,
    last_seen_s  DOUBLE PRECISION NOT NULL,
    max_count    INTEGER NOT NULL DEFAULT 1 CHECK (max_count >= 0),
    confidence   DOUBLE PRECISION NOT NULL
                 CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE TABLE IF NOT EXISTS counts (
    count_id     TEXT PRIMARY KEY,
    video_id     TEXT NOT NULL REFERENCES videos (video_id),
    minute_index INTEGER NOT NULL CHECK (minute_index >= 0),
    class_label  TEXT NOT NULL,
    "count"      INTEGER NOT NULL CHECK ("count" >= 0),
    confidence   DOUBLE PRECISION NOT NULL
                 CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE TABLE IF NOT EXISTS transcript (
    transcript_id TEXT PRIMARY KEY,
    video_id      TEXT NOT NULL REFERENCES videos (video_id),
    timestamp_s   DOUBLE PRECISION NOT NULL,
    text          TEXT NOT NULL,
    confidence    DOUBLE PRECISION NOT NULL
                  CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- Agent loop trace: one row per step (plan / review / retry) per segment.
-- observation is a JSON object (sorted keys) with the OpenCV measurements
-- that drove the decision.
CREATE TABLE IF NOT EXISTS agent_trace (
    trace_id       TEXT PRIMARY KEY,
    video_id       TEXT NOT NULL REFERENCES videos (video_id),
    segment_id     TEXT NOT NULL REFERENCES segments (segment_id),
    step           TEXT NOT NULL,
    observation    TEXT NOT NULL,
    decision       TEXT NOT NULL,
    reason         TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

-- Human-approval queue: segments the agent flagged for review.
CREATE TABLE IF NOT EXISTS review_queue (
    review_id      TEXT PRIMARY KEY,
    video_id       TEXT NOT NULL REFERENCES videos (video_id),
    segment_id     TEXT NOT NULL REFERENCES segments (segment_id),
    reason         TEXT NOT NULL,
    status         TEXT NOT NULL
                   CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at_utc TEXT NOT NULL
);
