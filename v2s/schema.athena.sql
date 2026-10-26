-- build 7fd77e00de7f
-- vid2sql Athena DDL: external tables over the JSON-lines files written by
-- v2s.sqlwriter.JsonlWriter. Same eight tables and column names as schema.sql.
--
-- Layout: upload each file {table}.jsonl to its OWN prefix,
--   s3://<bucket>/<prefix>/<table>/<table>.jsonl
-- (Athena reads whole prefixes, so one prefix per table). Replace the
-- LOCATION placeholders below with your bucket/prefix before running.
-- Timestamps are ISO-8601 UTC strings, so they are declared STRING here.

CREATE DATABASE IF NOT EXISTS v2s;

CREATE EXTERNAL TABLE IF NOT EXISTS v2s.videos (
  `video_id` string,
  `uri` string,
  `filename` string,
  `source_url` string,
  `license` string,
  `processed_at_utc` string,
  `recording_start_utc` string
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES ('ignore.malformed.json' = 'false')
LOCATION 's3://YOUR-BUCKET/YOUR-PREFIX/videos/'
TBLPROPERTIES ('classification' = 'json');

CREATE EXTERNAL TABLE IF NOT EXISTS v2s.segments (
  `segment_id` string,
  `video_id` string,
  `start_s` double,
  `end_s` double,
  `reason` string,
  `motion_score` double
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES ('ignore.malformed.json' = 'false')
LOCATION 's3://YOUR-BUCKET/YOUR-PREFIX/segments/'
TBLPROPERTIES ('classification' = 'json');

CREATE EXTERNAL TABLE IF NOT EXISTS v2s.events (
  `event_id` string,
  `video_id` string,
  `segment_id` string,
  `event_type` string,
  `start_timestamp_utc` string,
  `end_timestamp_utc` string,
  `confidence` double
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES ('ignore.malformed.json' = 'false')
LOCATION 's3://YOUR-BUCKET/YOUR-PREFIX/events/'
TBLPROPERTIES ('classification' = 'json');

CREATE EXTERNAL TABLE IF NOT EXISTS v2s.objects (
  `object_id` string,
  `video_id` string,
  `track_id` string,
  `class_label` string,
  `first_seen_s` double,
  `last_seen_s` double,
  `max_count` int,
  `confidence` double
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES ('ignore.malformed.json' = 'false')
LOCATION 's3://YOUR-BUCKET/YOUR-PREFIX/objects/'
TBLPROPERTIES ('classification' = 'json');

CREATE EXTERNAL TABLE IF NOT EXISTS v2s.counts (
  `count_id` string,
  `video_id` string,
  `minute_index` int,
  `class_label` string,
  `count` int,
  `confidence` double
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES ('ignore.malformed.json' = 'false')
LOCATION 's3://YOUR-BUCKET/YOUR-PREFIX/counts/'
TBLPROPERTIES ('classification' = 'json');

CREATE EXTERNAL TABLE IF NOT EXISTS v2s.transcript (
  `transcript_id` string,
  `video_id` string,
  `timestamp_s` double,
  `text` string,
  `confidence` double
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES ('ignore.malformed.json' = 'false')
LOCATION 's3://YOUR-BUCKET/YOUR-PREFIX/transcript/'
TBLPROPERTIES ('classification' = 'json');

CREATE EXTERNAL TABLE IF NOT EXISTS v2s.agent_trace (
  `trace_id` string,
  `video_id` string,
  `segment_id` string,
  `step` string,
  `observation` string,
  `decision` string,
  `reason` string,
  `created_at_utc` string
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES ('ignore.malformed.json' = 'false')
LOCATION 's3://YOUR-BUCKET/YOUR-PREFIX/agent_trace/'
TBLPROPERTIES ('classification' = 'json');

CREATE EXTERNAL TABLE IF NOT EXISTS v2s.review_queue (
  `review_id` string,
  `video_id` string,
  `segment_id` string,
  `reason` string,
  `status` string,
  `created_at_utc` string
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES ('ignore.malformed.json' = 'false')
LOCATION 's3://YOUR-BUCKET/YOUR-PREFIX/review_queue/'
TBLPROPERTIES ('classification' = 'json');
