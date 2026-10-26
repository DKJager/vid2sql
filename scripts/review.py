#!/usr/bin/env python3
"""Human-approval step for the vid2sql agent loop (review_queue).

The agent loop (``v2s.agent.process_clip``) writes segments it could not
trust on its own to the ``review_queue`` table with status ``pending``. This
CLI lets a person list those rows and approve or reject them.

The AWS endpoint database is read-only by design, so approvals are recorded
on a LOCAL SQLite copy of the database. This script never creates a database:
it opens an existing file read-write and fails if the file does not exist.

Usage:
    python scripts/review.py DB list
    python scripts/review.py DB approve REVIEW_ID
    python scripts/review.py DB reject REVIEW_ID

Exit codes:
    0  success
    1  the database or review_queue table is missing, or REVIEW_ID does not
       exist or is not pending

Standard library only.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_STATUS_FOR_ACTION = {"approve": "approved", "reject": "rejected"}


def _connect(db_path: str) -> sqlite3.Connection:
    """Open an existing SQLite database read-write (never create one)."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {db_path}")
    uri = f"{path.resolve().as_uri()}?mode=rw"
    return sqlite3.connect(uri, uri=True)


def _has_review_queue(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'review_queue'"
    ).fetchone()
    return row is not None


def cmd_list(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT review_id, video_id, segment_id, reason, created_at_utc "
        "FROM review_queue WHERE status = 'pending' "
        "ORDER BY created_at_utc, review_id"
    ).fetchall()
    if not rows:
        print("no pending reviews")
        return 0
    print("review_id\tvideo_id\tsegment_id\tcreated_at_utc\treason")
    for review_id, video_id, segment_id, reason, created_at_utc in rows:
        print(f"{review_id}\t{video_id}\t{segment_id}\t{created_at_utc}\t{reason}")
    print(f"{len(rows)} pending review(s)")
    return 0


def cmd_set_status(conn: sqlite3.Connection, review_id: str, action: str) -> int:
    new_status = _STATUS_FOR_ACTION[action]
    row = conn.execute(
        "SELECT status FROM review_queue WHERE review_id = ?", (review_id,)
    ).fetchone()
    if row is None:
        print(f"error: review_id {review_id!r} does not exist", file=sys.stderr)
        return 1
    if row[0] != "pending":
        print(
            f"error: review_id {review_id!r} is not pending (status={row[0]})",
            file=sys.stderr,
        )
        return 1
    with conn:
        cur = conn.execute(
            "UPDATE review_queue SET status = ? "
            "WHERE review_id = ? AND status = 'pending'",
            (new_status, review_id),
        )
    if cur.rowcount != 1:
        print(
            f"error: review_id {review_id!r} is no longer pending",
            file=sys.stderr,
        )
        return 1
    print(f"{review_id}: {new_status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Human approval of segments flagged by the agent loop. Works on a "
            "LOCAL SQLite copy; the AWS endpoint database is read-only by design."
        )
    )
    parser.add_argument("db", help="path to an existing local SQLite database")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print pending review_queue rows")
    p_approve = sub.add_parser("approve", help="mark a pending review as approved")
    p_approve.add_argument("review_id")
    p_reject = sub.add_parser("reject", help="mark a pending review as rejected")
    p_reject.add_argument("review_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        conn = _connect(args.db)
    except (FileNotFoundError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        if not _has_review_queue(conn):
            print(
                f"error: table review_queue not found in {args.db}",
                file=sys.stderr,
            )
            return 1
        if args.command == "list":
            return cmd_list(conn)
        return cmd_set_status(conn, args.review_id, args.command)
    except sqlite3.Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
