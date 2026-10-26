#!/usr/bin/env python3
"""Export the agent trace of a Video-to-SQL database as a readable Markdown report.

Usage:
    export_trace.py DB [--video-id ID] [--out PATH]

The database is opened read-only (SQLite URI mode=ro). agent_trace(trace_id,
video_id, segment_id, step, observation, decision, reason, created_at_utc) is
joined with segments (start_s, end_s) and videos (filename). The report has
one section per video, one table per segment (start_s, end_s, step, decision,
reason, observation as key=value pairs; nested detector calls render as
first={...}, retry={...}), and a final section listing
review_queue rows (review_id, video, segment, reason, status, created_at_utc).

Exits 1 with a clear message when the agent tables or columns are missing. An
empty agent_trace (the agent recorded no decisions) and an empty review_queue
are valid outcomes: the report says so and the exit status is 0.
A --video-id that matches no trace rows yields a report saying so (exit 0).
Standard library only.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

AGENT_TABLES = ("agent_trace", "review_queue")
REQUIRED_TABLES = AGENT_TABLES + ("segments", "videos")

REQUIRED_COLUMNS = {
    "agent_trace": ("trace_id", "video_id", "segment_id", "step", "observation",
                    "decision", "reason", "created_at_utc"),
    "review_queue": ("review_id", "video_id", "segment_id", "reason", "status",
                     "created_at_utc"),
    "segments": ("segment_id", "video_id", "start_s", "end_s"),
    "videos": ("video_id", "filename"),
}

HEADER_LINES = (
    "plan = OpenCV 5 camera-motion estimate chose the detector settings; "
    "review = OpenCV 5 detections were judged;",
    "retry = second detector call; flag_review = human approval request",
)


def fail(message):
    print(f"export_trace: error: {message}", file=sys.stderr)
    sys.exit(1)


def open_read_only(db_path):
    path = Path(db_path)
    if not path.is_file():
        fail(f"database file not found: {db_path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchall()
    except sqlite3.Error as exc:
        fail(f"cannot open database read-only: {db_path} ({exc})")
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn, table):
    return [row["name"] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def check_schema(conn):
    present = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    missing = [t for t in REQUIRED_TABLES if t not in present]
    if missing:
        hint = ""
        if any(t in AGENT_TABLES for t in missing):
            hint = (" This database was probably written with the agent off; "
                    "re-run the pipeline with the agent enabled to record a trace.")
        fail("missing table(s): " + ", ".join(missing) + "." + hint)

    for table, needed in REQUIRED_COLUMNS.items():
        cols = table_columns(conn, table)
        absent = [c for c in needed if c not in cols]
        if absent:
            fail(
                f"table {table} is missing column(s): {', '.join(absent)} "
                f"(found: {', '.join(cols)})"
            )


def md_cell(value):
    if value is None:
        return ""
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def fmt_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def fmt_value(value):
    if isinstance(value, dict):
        return "{" + fmt_pairs(value) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt_value(v) for v in value) + "]"
    return fmt_scalar(value)


def fmt_pairs(data):
    return ", ".join(f"{key}={fmt_value(data[key])}" for key in sorted(data, key=str))


def parse_observation(raw):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    data = raw
    # Tolerate double-encoded JSON (a JSON string containing a JSON object).
    for _ in range(2):
        if not isinstance(data, str):
            break
        try:
            data = json.loads(data)
        except (ValueError, TypeError):
            break
    return data


def render_observation(raw):
    if raw is None or raw == "":
        return ""
    data = parse_observation(raw)
    if not isinstance(data, dict):
        return fmt_value(data) if not isinstance(data, str) else data

    # Retry observations nest the two detector calls, e.g. first={...}, retry={...}:
    # nested dicts render as key={key=value, ...}, so every cell is key=value pairs.
    return fmt_pairs(data)


def fmt_seconds(value):
    if value is None:
        return "?"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def table_row(cells):
    return "| " + " | ".join(md_cell(c) for c in cells) + " |"


def build_report(conn, video_id=None):
    check_schema(conn)

    where = ""
    params = []
    if video_id is not None:
        where = "WHERE t.video_id = ?"
        params.append(video_id)

    trace_sql = f"""
        SELECT t.video_id AS video_id, v.filename AS filename,
               t.segment_id AS segment_id, s.start_s AS start_s, s.end_s AS end_s,
               t.step AS step, t.decision AS decision, t.reason AS reason,
               t.observation AS observation
        FROM agent_trace AS t
        LEFT JOIN segments AS s ON s.segment_id = t.segment_id
        LEFT JOIN videos AS v ON v.video_id = t.video_id
        {where}
        ORDER BY t.video_id, s.start_s IS NULL, s.start_s, t.segment_id,
                 t.created_at_utc, t.trace_id
    """
    rows = conn.execute(trace_sql, params).fetchall()

    out = ["# Agent trace report", "", HEADER_LINES[0], HEADER_LINES[1], ""]

    if not rows:
        if video_id is not None:
            out.append(f"_No agent trace rows found for video id {md_cell(video_id)}._")
        else:
            out.append("_No agent trace rows (the agent recorded no decisions; a database written with the agent off has none)._")

    current_video = None
    current_segment = None
    first = True
    for row in rows:
        new_video = first or row["video_id"] != current_video
        new_segment = new_video or row["segment_id"] != current_segment
        if new_segment and not first:
            out.append("")
        if new_video:
            current_video = row["video_id"]
            name = row["filename"] if row["filename"] is not None else "(unknown file)"
            out.append(f"## Video {md_cell(current_video)}: {md_cell(name)}")
            out.append("")
        if new_segment:
            current_segment = row["segment_id"]
            out.append(
                f"### Segment {md_cell(current_segment)} "
                f"({fmt_seconds(row['start_s'])} s - {fmt_seconds(row['end_s'])} s)"
            )
            out.append("")
            out.append("| start_s | end_s | step | decision | reason | observation |")
            out.append("| --- | --- | --- | --- | --- | --- |")
        first = False
        out.append(table_row((
            fmt_seconds(row["start_s"]),
            fmt_seconds(row["end_s"]),
            row["step"],
            row["decision"],
            row["reason"],
            render_observation(row["observation"]),
        )))
    out.append("")

    rq_where = ""
    rq_params = []
    if video_id is not None:
        rq_where = "WHERE r.video_id = ?"
        rq_params.append(video_id)
    rq_sql = f"""
        SELECT r.review_id AS review_id, r.video_id AS video_id,
               r.segment_id AS segment_id,
               r.reason AS reason, r.status AS status,
               r.created_at_utc AS created_at_utc
        FROM review_queue AS r
        {rq_where}
        ORDER BY r.created_at_utc, r.review_id
    """
    rq_rows = conn.execute(rq_sql, rq_params).fetchall()

    out.append("## Review queue (human approval requests)")
    out.append("")
    if not rq_rows:
        out.append("_No review_queue rows (the agent flagged nothing for review)._")
    else:
        out.append("| review_id | video | segment | reason | status | created_at_utc |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for row in rq_rows:
            out.append(table_row((
                row["review_id"],
                row["video_id"],
                row["segment_id"],
                row["reason"],
                row["status"],
                row["created_at_utc"],
            )))
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export agent_trace and review_queue from a Video-to-SQL database as Markdown."
    )
    parser.add_argument("db", metavar="DB", help="path to the SQLite database")
    parser.add_argument("--video-id", dest="video_id", default=None,
                        help="only report this video id")
    parser.add_argument("--out", dest="out", default=None,
                        help="write the report to PATH instead of stdout")
    args = parser.parse_args(argv)

    conn = open_read_only(args.db)
    try:
        try:
            report = build_report(conn, args.video_id)
        except sqlite3.Error as exc:
            fail(f"query failed: {exc}")
    finally:
        conn.close()

    if not report.endswith("\n"):
        report += "\n"
    if args.out:
        try:
            Path(args.out).write_text(report, encoding="utf-8")
        except OSError as exc:
            fail(f"cannot write {args.out}: {exc}")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
