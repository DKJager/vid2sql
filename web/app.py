# build 7fd77e00de7f
"""Web endpoint: SQL or plain question -> rows with clickable video timestamps.

This app serves the query interface over the rows written by the OpenCV 5
pipeline (SQLite locally; the AWS Lambda downloads the SQLite database from
S3). It contains no vision or model code.

Configuration (read lazily, only at request time):
    V2S_DB_PATH -- SQLite database path for dev mode (default ``v2s.db``).
    V2S_ALLOW_UNSAFE_SQL -- set to ``1`` to allow raw ad-hoc SQL in dev;
        otherwise only read-only SELECT statements are accepted.
    V2S_CLIP_DIR -- optional extra directory whose local video files may be
        served by ``/media/<video_id>`` (``data/clips`` is always allowed).

No environment variable is read at import time, so tests can import this
module with no environment set.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from flask import Flask, Response, jsonify, render_template_string, request, send_file
from markupsafe import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2s import nlq  # noqa: E402

_PAGE_PATH = Path(__file__).with_name("index.html")
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Maximum number of rows returned by a single query.
_MAX_ROWS = 1000
#: Maximum wall-clock seconds a single query may run before it is aborted.
_MAX_QUERY_SECONDS = 5.0
#: Largest offset (seconds) accepted for a timestamp link.
_MAX_LINK_OFFSET_S = 86400.0


class QueryRejectedError(ValueError):
    """Raised when a submitted query is not a read-only SELECT."""


_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|replace)\b",
    re.IGNORECASE,
)

#: Very small, rule-based "plain question" translation for the demo. This is
#: a heuristic convenience only; no predictive or learned behavior is claimed.
_QUESTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(event|events)\b", re.IGNORECASE),
        "SELECT event_id, video_id, event_type, start_timestamp_utc,"
        " end_timestamp_utc, confidence FROM events ORDER BY start_timestamp_utc"
        " LIMIT 200",
    ),
    (
        re.compile(r"\b(count|counts|how many)\b", re.IGNORECASE),
        "SELECT video_id, minute_index, class_label, count, confidence"
        " FROM counts ORDER BY video_id, minute_index LIMIT 200",
    ),
    (
        re.compile(r"\b(object|objects|track|tracks)\b", re.IGNORECASE),
        "SELECT object_id, video_id, track_id, class_label, first_seen_s,"
        " last_seen_s, max_count, confidence FROM objects ORDER BY video_id,"
        " track_id LIMIT 200",
    ),
    (
        re.compile(r"\b(video|videos|clip|clips)\b", re.IGNORECASE),
        "SELECT video_id, uri, filename, source_url, license, processed_at_utc"
        " FROM videos ORDER BY processed_at_utc LIMIT 200",
    ),
)


def _db_path() -> str:
    return os.environ.get("V2S_DB_PATH", "v2s.db").strip() or "v2s.db"


def _allow_unsafe() -> bool:
    """True when the env var V2S_ALLOW_UNSAFE_SQL is set to '1'."""
    return os.environ.get("V2S_ALLOW_UNSAFE_SQL", "").strip() == "1"


def _strip_sql_comments(sql: str) -> str:
    """Remove ``--`` and ``/* */`` comments, leaving quoted strings intact."""
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c in ("'", '"'):
            j = i + 1
            while j < n:
                if sql[j] == c:
                    if j + 1 < n and sql[j + 1] == c:
                        j += 2
                        continue
                    break
                j += 1
            out.append(sql[i : j + 1])
            i = j + 1
        elif sql.startswith("--", i):
            while i < n and sql[i] != "\n":
                i += 1
        elif sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end < 0 else end + 2
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def _connect() -> sqlite3.Connection:
    """Open the database read-only unless unsafe SQL was explicitly enabled."""
    path = _db_path()
    if _allow_unsafe():
        return sqlite3.connect(path)
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)


def _is_safe_select(sql: str) -> bool:
    if _allow_unsafe():
        return True
    text = sql.strip().rstrip(";").strip()
    if not text:
        return False
    if not re.match(r"^(select|with)\b", text, re.IGNORECASE):
        return False
    return _FORBIDDEN_SQL.search(text) is None


def _question_to_sql(question: str) -> str:
    try:
        return nlq.translate(question)[0]
    except nlq.UnsupportedQuestion:
        pass
    for pattern, sql in _QUESTION_PATTERNS:
        if pattern.search(question):
            return sql
    raise QueryRejectedError(
        "could not map the question to a demo query; try keywords like"
        " 'events', 'counts', 'objects', or 'videos'"
    )


def _run_query(sql: str) -> list[dict[str, Any]]:
    """Run ``sql``; at most ``_MAX_ROWS`` rows, aborted after
    ``_MAX_QUERY_SECONDS`` seconds (reported as 'query too expensive')."""
    with contextlib.closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        deadline = time.monotonic() + _MAX_QUERY_SECONDS
        state = {"timed_out": False}

        def _progress() -> int:
            if time.monotonic() > deadline:
                state["timed_out"] = True
                return 1
            return 0

        conn.set_progress_handler(_progress, 1000)
        try:
            cursor = conn.execute(sql)
            return [dict(row) for row in cursor.fetchmany(_MAX_ROWS)]
        except sqlite3.OperationalError as exc:
            if state["timed_out"]:
                raise QueryRejectedError("query too expensive") from exc
            raise
        finally:
            conn.set_progress_handler(None, 0)


def _to_float_safe(value: Any) -> float | None:
    """Return float(value) if possible, otherwise None."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _seconds_from_timestamp_utc(text: Any) -> float | None:
    """Heuristically parse a UTC timestamp into seconds from the epoch, or
    a time-of-day string into a seconds offset. Returns None on failure."""
    val = str(text).strip()
    if not val:
        return None
    # ISO-8601 / "2025-01-01T12:34:56Z" and variants
    try:
        from datetime import datetime, timezone
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(val, fmt)
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
    except Exception:
        pass
    # HH:MM:SS or HH:MM or bare seconds string
    parts = val.split(":")
    if len(parts) == 3:
        try:
            h, m, s = map(float, parts)
            return h * 3600.0 + m * 60.0 + s
        except Exception:
            return None
    if len(parts) == 2:
        try:
            h, m = map(float, parts)
            return h * 3600.0 + m * 60.0
        except Exception:
            return None
    return _to_float_safe(val)


def _utc_epoch(value: Any) -> float | None:
    """Parse an absolute ISO-8601 date-time (naive values are UTC) into epoch
    seconds. Time-of-day strings and bare numbers are rejected (None)."""
    if value is None or isinstance(value, (int, float)):
        return None
    text = str(value).strip()
    if not text or "-" not in text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _lookup_recording_start(video_id: str) -> float | None:
    """Read-only lookup of videos.recording_start_utc as epoch seconds."""
    try:
        with contextlib.closing(_connect()) as conn:
            row = conn.execute(
                "SELECT recording_start_utc FROM videos WHERE video_id = ?",
                (video_id,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None or row[0] is None:
        return None
    return _utc_epoch(row[0])


def _lookup_video_id_by_filename(filename: str) -> str | None:
    """Read-only lookup of video_id by filename; only when exactly one matches."""
    try:
        with contextlib.closing(_connect()) as conn:
            rows = conn.execute(
                "SELECT video_id FROM videos WHERE filename = ? LIMIT 2",
                (filename,),
            ).fetchall()
    except sqlite3.Error:
        return None
    if len(rows) != 1 or rows[0][0] is None:
        return None
    return str(rows[0][0])


def _annotate_video_links(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach a playable timestamp link when a row has video + time fields."""
    annotated: list[dict[str, Any]] = []
    starts: dict[str, float | None] = {}
    ids_by_filename: dict[str, str | None] = {}
    for row in rows:
        out = dict(row)
        video_id = row.get("video_id")
        anchor_s: float | None = None

        # 1. numeric seconds columns
        for key in ("timestamp_s", "first_seen_s"):
            v = _to_float_safe(row.get(key))
            if v is not None:
                anchor_s = v
                break

        # 2. minute_index -> seconds
        if anchor_s is None:
            minute_val = _to_float_safe(row.get("minute_index"))
            if minute_val is not None:
                anchor_s = minute_val * 60.0

        second_val = _to_float_safe(row.get("second")) if anchor_s is None else None
        has_start_ts = anchor_s is None and row.get("start_timestamp_utc") is not None

        # Resolve video_id from a unique filename when needed.
        if (
            not video_id
            and (anchor_s is not None or second_val is not None or has_start_ts)
            and row.get("filename")
        ):
            fname = str(row["filename"])
            if fname not in ids_by_filename:
                ids_by_filename[fname] = _lookup_video_id_by_filename(fname)
            video_id = ids_by_filename[fname]

        # 3. numeric `second` column (seconds into the video)
        if anchor_s is None and second_val is not None:
            if math.isfinite(second_val) and 0.0 <= second_val <= _MAX_LINK_OFFSET_S:
                anchor_s = second_val

        # 4. start_timestamp_utc relative to videos.recording_start_utc
        if anchor_s is None and has_start_ts and video_id:
            key = str(video_id)
            if key not in starts:
                starts[key] = _lookup_recording_start(key)
            rec_start = starts[key]
            event_ts = _utc_epoch(row["start_timestamp_utc"])
            if rec_start is not None and event_ts is not None:
                offset = event_ts - rec_start
                if math.isfinite(offset) and 0.0 <= offset <= _MAX_LINK_OFFSET_S:
                    anchor_s = offset

        if video_id and anchor_s is not None:
            out["video_link"] = f"/clip/{video_id}?t={anchor_s:.1f}"
        annotated.append(out)
    return annotated


def _allowed_clip_dirs() -> list[Path]:
    dirs = [(_REPO_ROOT / "data" / "clips").resolve()]
    extra = os.environ.get("V2S_CLIP_DIR", "").strip()
    if extra:
        dirs.append(Path(extra).resolve())
    return dirs


def _local_clip_path(uri: Any) -> Path | None:
    """Return the resolved local file for ``uri`` if it lies inside an allowed
    clip directory and exists; otherwise None."""
    if not uri:
        return None
    text = str(uri).strip()
    parsed = urlparse(text)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            return None
        raw = Path(unquote(parsed.path))
    elif parsed.scheme and len(parsed.scheme) > 1:
        return None  # remote scheme such as s3://, http(s)://
    else:
        raw = Path(text)
    if not raw.is_absolute():
        raw = _REPO_ROOT / raw
    try:
        resolved = raw.resolve()
    except (OSError, RuntimeError):
        return None
    for base in _allowed_clip_dirs():
        if resolved.is_relative_to(base) and resolved.is_file():
            return resolved
    return None


def _wants_html() -> bool:
    accept = request.headers.get("Accept", "")
    if "text/html" not in accept:
        return False
    return request.args.get("format", "").lower() != "json"


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(
            _PAGE_PATH.read_text(encoding="utf-8")
        )

    @app.post("/query")
    def query() -> Any:
        payload = request.get_json(silent=True) or {}
        sql = _strip_sql_comments(payload.get("sql") or "").strip()
        question = (payload.get("question") or "").strip()
        try:
            if not sql and question:
                sql = _question_to_sql(question)
            if not _is_safe_select(sql):
                raise QueryRejectedError(
                    "only read-only SELECT queries are accepted by this demo"
                )
            rows = _annotate_video_links(_run_query(sql))
        except (QueryRejectedError, sqlite3.Error) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "sql": sql, "rows": rows})

    @app.get("/clip/<path:video_id>")
    def clip(video_id: str) -> Any:
        if not _VIDEO_ID_RE.match(video_id) or ".." in video_id:
            return jsonify({"ok": False, "error": "invalid video_id"}), 400
        raw_t = request.args.get("t")
        start_value = _to_float_safe(raw_t) if raw_t is not None else None
        if start_value is None or not math.isfinite(start_value) or start_value < 0:
            return jsonify(
                {"ok": False, "error": "t must be a finite number >= 0"}
            ), 400
        try:
            with contextlib.closing(_connect()) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT uri, filename, source_url FROM videos WHERE video_id = ?",
                    (video_id,),
                ).fetchone()
        except sqlite3.Error:
            return jsonify({"ok": False, "error": "database unavailable"}), 503
        if row is None:
            return jsonify({"ok": False, "error": "unknown video_id"}), 404
        if _wants_html():
            t_text = f"{start_value:.1f}"
            title = escape(row["filename"] if row["filename"] is not None else video_id)
            if _local_clip_path(row["uri"]) is not None:
                body = (
                    f'<video controls src="/media/{escape(video_id)}#t={escape(t_text)}">'
                    "</video>"
                )
            elif row["source_url"]:
                escaped_source_url = escape(row["source_url"])
                body = (
                    f'<p>Source: <a href="{escaped_source_url}#t={escape(t_text)}">{escaped_source_url}#t={escape(t_text)}</a></p>'
                )
            else:
                body = (
                    f"<p>Source: {escape(row['uri'])}, at {escape(t_text)} s."
                    " The video file is not hosted by this endpoint.</p>"
                )
            page = (
                "<!doctype html><html><head><meta charset=\"utf-8\">"
                f"<title>{title}</title></head><body>"
                f"<h1>{title}</h1><p>t = {escape(t_text)} s</p>{body}"
                "</body></html>"
            )
            return Response(page, mimetype="text/html")
        return jsonify(
            {
                "ok": True,
                "video_id": video_id,
                "uri": row["uri"],
                "filename": row["filename"],
                "start_s": start_value,
            }
        )

    @app.get("/media/<path:video_id>")
    def media(video_id: str) -> Any:
        if not _VIDEO_ID_RE.match(video_id) or ".." in video_id:
            return jsonify({"ok": False, "error": "invalid video_id"}), 400
        try:
            with contextlib.closing(_connect()) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT uri FROM videos WHERE video_id = ?",
                    (video_id,),
                ).fetchone()
        except sqlite3.Error:
            return jsonify({"ok": False, "error": "database unavailable"}), 503
        if row is None:
            return jsonify({"ok": False, "error": "unknown video_id"}), 404
        path = _local_clip_path(row["uri"])
        if path is None:
            return jsonify({"ok": False, "error": "video file not hosted"}), 404
        return send_file(path, conditional=True)

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify({"ok": True})

    return app


app = create_app()


if __name__ == "__main__":
    # Dev server only; production serving is handled by the deploy template.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8080")))
