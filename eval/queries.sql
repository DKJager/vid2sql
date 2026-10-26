-- vid2sql fixed evaluation queries over the event tables.
-- These run against the SQLite dev database written by v2s/sqlwriter.py.
-- Column names match v2s/schema.sql exactly. All queries are read-only
-- SELECT statements.
--
-- Scope note: this file targets the SQLite dev database ONLY. No Athena
-- compatibility is claimed for these statements as written. If Athena
-- execution is ever needed, a separate dialect-adjustment step must live
-- in the eval harness (not in this file) before the statements are sent
-- to Athena; treat these queries as SQLite-canonical until then.
--
-- Packaging note: per the deliverable file-type rules (.sql is not a
-- supported deliverable extension), the canonical packaged copy of this
-- query set is carried as a supported artifact (e.g. eval/queries.json:
-- query id + SQL string per entry) and the eval harness reads that
-- artifact. This file is the human-readable source of the same queries.

-- videos
-- Q1: every processed video, newest first.
SELECT video_id, uri, filename, source_url, license, processed_at_utc
FROM videos
ORDER BY processed_at_utc DESC;

-- segments
-- Q2: segments sampled per video, with motion scores.
SELECT s.segment_id, s.video_id, s.start_s, s.end_s, s.reason, s.motion_score
FROM segments AS s
JOIN videos AS v ON v.video_id = s.video_id
ORDER BY v.filename, s.start_s;

-- events
-- Q3: presence events ordered by confidence (highest first).
SELECT event_id, video_id, segment_id, event_type,
       start_timestamp_utc, end_timestamp_utc, confidence
FROM events
ORDER BY confidence DESC, start_timestamp_utc ASC;

-- objects
-- Q4: tracked objects with first/last seen offsets in seconds.
SELECT object_id, video_id, track_id, class_label,
       first_seen_s, last_seen_s, max_count, confidence
FROM objects
ORDER BY video_id, track_id;

-- counts
-- Q5: per-minute per-class counts for BI aggregation.
-- The counts column is named "count" in v2s/schema.sql; it is quoted
-- here so parsers never misbind it to the COUNT aggregate function.
SELECT video_id, minute_index, class_label, "count" AS object_count, confidence
FROM counts
ORDER BY video_id, minute_index, class_label;

-- transcript
-- Q6: transcript lines with their timestamps.
SELECT transcript_id, video_id, timestamp_s, text, confidence
FROM transcript
ORDER BY video_id, timestamp_s;

-- tag:sanity
-- Q7: sanity shape check - row counts for all six tables.
SELECT 'videos' AS table_name, COUNT(*) AS row_count FROM videos
UNION ALL
SELECT 'segments', COUNT(*) FROM segments
UNION ALL
SELECT 'events', COUNT(*) FROM events
UNION ALL
SELECT 'objects', COUNT(*) FROM objects
UNION ALL
SELECT 'counts', COUNT(*) FROM counts
UNION ALL
SELECT 'transcript', COUNT(*) FROM transcript;

-- tag:confidence
-- Q8: average stored confidence per event type (calibration surface).
SELECT event_type,
       COUNT(*) AS event_count,
       AVG(confidence) AS avg_confidence,
       MIN(confidence) AS min_confidence,
       MAX(confidence) AS max_confidence
FROM events
GROUP BY event_type
ORDER BY event_count DESC;

-- tag:integrity
-- Q9: foreign-key integrity diagnostic (orphan detection).
-- Returns any child rows whose parent reference does not exist.
-- Empty result set means every child row has a valid parent row.
SELECT 'orphan-segment' AS violation, s.segment_id
FROM segments AS s
LEFT JOIN videos AS v ON v.video_id = s.video_id
WHERE v.video_id IS NULL
UNION ALL
SELECT 'orphan-event-segment', e.event_id
FROM events AS e
LEFT JOIN segments AS seg ON seg.segment_id = e.segment_id
WHERE seg.segment_id IS NULL
UNION ALL
SELECT 'orphan-event-video', e.event_id
FROM events AS e
LEFT JOIN videos AS v ON v.video_id = e.video_id
WHERE v.video_id IS NULL
UNION ALL
SELECT 'orphan-object', o.object_id
FROM objects AS o
LEFT JOIN videos AS v ON v.video_id = o.video_id
WHERE v.video_id IS NULL
UNION ALL
SELECT 'orphan-counts', c.video_id || '-' || CAST(c.minute_index AS TEXT) || '-' || c.class_label
FROM counts AS c
LEFT JOIN videos AS v ON v.video_id = c.video_id
WHERE v.video_id IS NULL
UNION ALL
SELECT 'orphan-transcript', t.transcript_id
FROM transcript AS t
LEFT JOIN videos AS v ON v.video_id = t.video_id
WHERE v.video_id IS NULL;
