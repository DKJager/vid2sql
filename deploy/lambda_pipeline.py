# vid2sql: S3-triggered Lambda that runs the OpenCV 5 pipeline on one upload.
"""A video lands in ``s3://<UploadBucket>/uploads/``; this function downloads
it, runs the same agent loop used locally (``v2s.agent.process_clip``), merges
the rows into one SQLite database kept in the results bucket at
``db/v2s.db``, and uploads it back. The function runs with reserved
concurrency 1, so the read-modify-write on the database cannot race.
Optional S3 user metadata: license, source-url, recording-start-utc (ISO 8601 UTC); without recording-start-utc, event timestamps are processing-relative and a warning is logged.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote_plus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2s import agent, sqlwriter  # noqa: E402
from v2s.ingest import ClipWorkItem, normalise_recording_start  # noqa: E402

DB_OBJECT_KEY = "db/v2s.db"
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def _s3():
    import boto3

    return boto3.client("s3")


def _results_bucket() -> str:
    return os.environ["RESULTS_BUCKET"]


def _fetch_db(s3, local_db: str) -> None:
    try:
        s3.download_file(_results_bucket(), DB_OBJECT_KEY, local_db)
    except Exception as exc:  # first run: no database yet
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code not in ("404", "NoSuchKey", "NotFound"):
            raise


def process_object(bucket: str, key: str, s3=None) -> dict:
    """Process one uploaded video and publish the updated database."""
    if not key.lower().endswith(VIDEO_SUFFIXES):
        return {"key": key, "skipped": "not a video"}
    s3 = s3 or _s3()
    with tempfile.TemporaryDirectory() as tmp:
        local_video = str(Path(tmp) / Path(key).name)
        local_db = str(Path(tmp) / "v2s.db")
        s3.download_file(bucket, key, local_video)
        _fetch_db(s3, local_db)
        writer = sqlwriter.SqliteWriter(local_db)
        head = s3.head_object(Bucket=bucket, Key=key)
        meta = head.get("Metadata", {}) if isinstance(head, dict) else {}
        license_meta = meta.get("license")
        if not license_meta or not license_meta.strip():
            return {"key": key, "skipped": "no license metadata"}
        recording_start_utc = normalise_recording_start(meta.get("recording-start-utc"))
        clip = ClipWorkItem(
            uri=f"s3://{bucket}/{key}",
            filename=Path(key).name,
            source_url=meta.get("source-url", f"s3://{bucket}/{key}"),
            license=license_meta,
            recording_start_utc=recording_start_utc,
        )
        # The agent reads the local copy; the stored uri stays the S3 location.
        local_clip = ClipWorkItem(
            local_video,
            clip.filename,
            clip.source_url,
            clip.license,
            recording_start_utc=recording_start_utc,
        )
        result = agent.process_clip(local_clip, writer)
        _rewrite_uri(local_db, result["video_id"], clip.uri)
        s3.upload_file(local_db, _results_bucket(), DB_OBJECT_KEY)
    return {"key": key, "video_id": result["video_id"], "counts": result["counts"]}


def _rewrite_uri(db_path: str, video_id: str, uri: str) -> None:
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE videos SET uri = ? WHERE video_id = ?", (uri, video_id))


def handler(event, context=None):
    results = []
    for rec in event.get("Records", []):
        bucket = rec["s3"]["bucket"]["name"]
        key = unquote_plus(rec["s3"]["object"]["key"])
        results.append(process_object(bucket, key))
    return {"processed": results}
