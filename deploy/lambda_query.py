# vid2sql: HTTP API Lambda that serves the read-only query endpoint.
"""Adapts an API Gateway HTTP API (payload v2) request to the Flask app in
``web/app.py`` and answers from the SQLite database in the results bucket."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB_OBJECT_KEY = "db/v2s.db"
LOCAL_DB = "/tmp/v2s.db"
_state: dict = {"etag": None, "client": None}


def _s3():
    import boto3

    return boto3.client("s3")


def _sync_db(s3=None) -> bool:
    """Download the database when the bucket copy changed. False if none yet."""
    s3 = s3 or _s3()
    bucket = os.environ["RESULTS_BUCKET"]
    try:
        head = s3.head_object(Bucket=bucket, Key=DB_OBJECT_KEY)
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    if head.get("ETag") != _state["etag"] or not Path(LOCAL_DB).exists():
        s3.download_file(bucket, DB_OBJECT_KEY, LOCAL_DB)
        _state["etag"] = head.get("ETag")
    return True


def _client():
    if _state["client"] is None:
        os.environ.setdefault("V2S_DB_PATH", LOCAL_DB)
        from web import app as webapp

        _state["client"] = webapp.create_app().test_client()
    return _state["client"]


def _json(status: int, text: str) -> dict:
    return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": text}


def handler(event, context=None):
    path = event.get("rawPath", "/")
    if path not in ("/", "/healthz") and not _sync_db():
        return _json(503, '{"ok": false, "error": "no database yet: upload a video first"}')
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    qs = event.get("rawQueryString", "")
    resp = _client().open(
        path + (f"?{qs}" if qs else ""),
        method=event["requestContext"]["http"]["method"],
        data=body,
        content_type=(event.get("headers") or {}).get("content-type"),
    )
    return {
        "statusCode": resp.status_code,
        "headers": {"content-type": resp.headers.get("Content-Type", "text/plain")},
        "body": resp.get_data(as_text=True),
    }
