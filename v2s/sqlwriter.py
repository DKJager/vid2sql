# build 7fd77e00de7f
"""Write reconciled rows into SQLite (dev) or S3/Athena-ready form (prod).

Thin wrapper: this module only persists the agent loop's output rows. Dev
mode targets a local SQLite database via the stdlib. Production mode writes
JSON-lines files that map onto the Athena DDL (``v2s/schema.athena.sql``);
upload and catalogue registration are explicit, caller-initiated actions and
are never performed automatically.

Every row carries a confidence value, per the buyer requirement for granular,
calibrated confidence on stored detections and events.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Union

TABLES = ("videos", "segments", "events", "objects", "counts", "transcript", "agent_trace", "review_queue")

#: Minimal column order per table for the SQLite dev writer. The canonical
#: definition lives in v2s/schema.sql; this mirrors it for local runs.
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "videos": ("video_id", "uri", "filename", "source_url", "license", "processed_at_utc", "recording_start_utc"),
    "segments": ("segment_id", "video_id", "start_s", "end_s", "reason", "motion_score"),
    "events": (
        "event_id",
        "video_id",
        "segment_id",
        "event_type",
        "start_timestamp_utc",
        "end_timestamp_utc",
        "confidence",
    ),
    "objects": (
        "object_id",
        "video_id",
        "track_id",
        "class_label",
        "first_seen_s",
        "last_seen_s",
        "max_count",
        "confidence",
    ),
    "counts": ("count_id", "video_id", "minute_index", "class_label", "count", "confidence"),
    "transcript": ("transcript_id", "video_id", "timestamp_s", "text", "confidence"),
    "agent_trace": (
        "trace_id",
        "video_id",
        "segment_id",
        "step",
        "observation",
        "decision",
        "reason",
        "created_at_utc",
    ),
    "review_queue": (
        "review_id",
        "video_id",
        "segment_id",
        "reason",
        "status",
        "created_at_utc",
    ),
}

#: Type coercion map for JSONL/Athena conformance.
#: Each entry maps column name -> callable(value) that returns the coerced value.
#: This mirrors the Athena DDL types in ``v2s/schema.athena.sql``.
JSONL_TYPE_COERCIONS: dict[str, dict[str, callable]] = {
    "videos": {},
    "segments": {
        "start_s": float,
        "end_s": float,
        "motion_score": float,
    },
    "events": {
        "confidence": float,
    },
    "objects": {
        "first_seen_s": float,
        "last_seen_s": float,
        "max_count": int,
        "confidence": float,
    },
    "counts": {
        "minute_index": int,
        "count": int,
        "confidence": float,
    },
    "transcript": {
        "timestamp_s": float,
        "confidence": float,
    },
    "agent_trace": {},
    "review_queue": {},
}

CONFIDENCE_COLUMNS: set[str] = {"confidence", "motion_score"}
"""Columns that must be validated for NaN and clamped to a valid range."""

_CONFIDENCE_MIN = 0.0
_CONFIDENCE_MAX = 1.0

#: The JsonlWriter lays out staging files as flat per-table files **without**
#: a ``dt=YYYY-MM-DD/`` partition prefix.  Example layout::
#:
#:     staging_dir/
#:         videos.jsonl
#:         segments.jsonl
#:         events.jsonl
#:         ...
#:
#: Any Athena projection template that requires a ``dt=`` prefix will
#: return zero rows unless the staging directory is restructured or the
#: template is adjusted to match this flat convention.
STAGING_LAYOUT_DOC = (
    "Flat per-table JSON-lines: {staging_dir}/{table}.jsonl.  "
    "No partition prefix (no dt=YYYY-MM-DD/ sub-directory) is created "
    "by this writer."
)


def _coerce_iso8601(value: Any) -> str:
    """Coerce a value to an ISO-8601 UTC string for Athena timestamp columns.

    If the value is already a string that looks like ISO-8601, return it as-is.
    If it's a datetime object, convert to UTC and format.
    Otherwise, stringify it as-is (caller's responsibility for correctness).
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _coerce_jsonl_row(table: str, row: dict[str, Any]) -> dict[str, Any]:
    """Apply type coercions for JSONL/Athena DDL conformance.

    Columns not listed in JSONL_TYPE_COERCIONS[table] pass through unchanged.
    Timestamp columns that are strings are normalized to ISO-8601 format.
    """
    coercions = JSONL_TYPE_COERCIONS.get(table, {})
    coerced: dict[str, Any] = {}
    for key, val in row.items():
        if key in coercions and val is not None:
            try:
                coerced[key] = coercions[key](val)
            except (TypeError, ValueError) as exc:
                import warnings

                warnings.warn(
                    f"Type coercion failed for {table}.{key}={val!r}: {exc}; "
                    f"using original value",
                    UserWarning,
                    stacklevel=2,
                )
                coerced[key] = val
        elif key.endswith("_utc"):  # Timestamp column
            coerced[key] = _coerce_iso8601(val) if val is not None else val
        else:
            coerced[key] = val
    return coerced


def _clamp_confidence(value: Any) -> float:
    """Validate and clamp a confidence/score value to [0.0, 1.0].

    NaN, infinity, or other non-finite floats are replaced with 0.0.
    Out-of-range finite floats are clamped to the valid range.
    A ``UserWarning`` is emitted for every non-conforming value.
    """
    try:
        fval = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        import warnings

        warnings.warn(
            f"Confidence value {value!r} is not convertible to float; clamping to 0.0",
            UserWarning,
            stacklevel=2,
        )
        return _CONFIDENCE_MIN

    if not math.isfinite(fval):
        import warnings

        warnings.warn(
            f"Confidence value {fval!r} is not finite; clamping to 0.0",
            UserWarning,
            stacklevel=2,
        )
        return _CONFIDENCE_MIN

    if fval < _CONFIDENCE_MIN:
        import warnings

        warnings.warn(
            f"Confidence value {fval!r} is below {_CONFIDENCE_MIN}; clamping to {_CONFIDENCE_MIN}",
            UserWarning,
            stacklevel=2,
        )
        return _CONFIDENCE_MIN

    if fval > _CONFIDENCE_MAX:
        import warnings

        warnings.warn(
            f"Confidence value {fval!r} is above {_CONFIDENCE_MAX}; clamping to {_CONFIDENCE_MAX}",
            UserWarning,
            stacklevel=2,
        )
        return _CONFIDENCE_MAX

    return fval


class SqlWriterError(RuntimeError):
    """Raised on invalid rows or writer misconfiguration."""


def _validate_table_name(table: str) -> None:
    """Raise ``SqlWriterError`` if *table* is not a known table name.

    This prevents SQL injection through dynamic table-name interpolation
    anywhere table names are embedded in SQL statements.
    """
    if not isinstance(table, str) or table not in TABLE_COLUMNS:
        raise SqlWriterError(f"unknown table: {table!r}")


def _validate_rows(table: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and clamp rows for a given table.

    Returns a list of validated row dicts with confidence values clamped.
    Each row dict is expected to contain exactly the columns defined in
    ``TABLE_COLUMNS`` for the table.

    Raises ``SqlWriterError`` with row-index and column-level context when
    validation fails, so callers can pinpoint the offending data.
    """
    _validate_table_name(table)
    expected = set(TABLE_COLUMNS[table])
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SqlWriterError(
                f"row {idx} for table {table!r} is not an object: {row!r}"
            )
        missing = expected - set(row)
        if missing:
            raise SqlWriterError(
                f"row {idx} for table {table!r} missing columns "
                f"{sorted(missing)}: {row!r}"
            )
        clamped_row: dict[str, Any] = {}
        for key in TABLE_COLUMNS[table]:
            val = row.get(key)
            if key in CONFIDENCE_COLUMNS:
                try:
                    val = _clamp_confidence(val)
                except Exception as exc:
                    raise SqlWriterError(
                        f"row {idx} for table {table!r}: failed to clamp "
                        f"column {key!r} value {val!r}: {exc}"
                    ) from exc
            clamped_row[key] = val
        out.append(clamped_row)
    return out


def create_schema(db_path: str | os.PathLike[str]) -> None:
    """Create the dev schema in a SQLite database if it does not exist.

    The canonical DDL in ``v2s/schema.sql`` is Postgres-compatible; this dev
    schema mirrors its tables with SQLite types so tests can run offline.

    Foreign keys are enabled and child tables reference ``videos.video_id``
    to prevent orphan rows from being inserted silently.
    """
    statements = [
        "CREATE TABLE IF NOT EXISTS videos ("
        "video_id TEXT PRIMARY KEY, uri TEXT NOT NULL, filename TEXT NOT NULL,"
        " source_url TEXT NOT NULL, license TEXT NOT NULL,"
        " processed_at_utc TEXT NOT NULL, recording_start_utc TEXT)",
        "CREATE TABLE IF NOT EXISTS segments ("
        "segment_id TEXT PRIMARY KEY, video_id TEXT NOT NULL,"
        " start_s REAL NOT NULL, end_s REAL NOT NULL,"
        " reason TEXT NOT NULL, motion_score REAL NOT NULL DEFAULT 0.0,"
        " FOREIGN KEY(video_id) REFERENCES videos(video_id))",
        "CREATE TABLE IF NOT EXISTS events ("
        "event_id TEXT PRIMARY KEY, video_id TEXT NOT NULL,"
        " segment_id TEXT, event_type TEXT NOT NULL,"
        " start_timestamp_utc TEXT NOT NULL, end_timestamp_utc TEXT NOT NULL,"
        " confidence REAL NOT NULL,"
        " FOREIGN KEY(video_id) REFERENCES videos(video_id))",
        "CREATE TABLE IF NOT EXISTS objects ("
        "object_id TEXT PRIMARY KEY, video_id TEXT NOT NULL,"
        " track_id TEXT NOT NULL, class_label TEXT NOT NULL,"
        " first_seen_s REAL NOT NULL, last_seen_s REAL NOT NULL,"
        " max_count INTEGER NOT NULL DEFAULT 1, confidence REAL NOT NULL,"
        " FOREIGN KEY(video_id) REFERENCES videos(video_id))",
        "CREATE TABLE IF NOT EXISTS counts ("
        "count_id TEXT PRIMARY KEY, video_id TEXT NOT NULL,"
        " minute_index INTEGER NOT NULL, class_label TEXT NOT NULL,"
        " count INTEGER NOT NULL, confidence REAL NOT NULL,"
        " FOREIGN KEY(video_id) REFERENCES videos(video_id))",
        "CREATE TABLE IF NOT EXISTS transcript ("
        "transcript_id TEXT PRIMARY KEY, video_id TEXT NOT NULL,"
        " timestamp_s REAL NOT NULL, text TEXT NOT NULL,"
        " confidence REAL NOT NULL,"
        " FOREIGN KEY(video_id) REFERENCES videos(video_id))",
        "CREATE TABLE IF NOT EXISTS agent_trace ("
        "trace_id TEXT PRIMARY KEY, video_id TEXT NOT NULL,"
        " segment_id TEXT NOT NULL, step TEXT NOT NULL,"
        " observation TEXT NOT NULL, decision TEXT NOT NULL,"
        " reason TEXT NOT NULL, created_at_utc TEXT NOT NULL,"
        " FOREIGN KEY(video_id) REFERENCES videos(video_id))",
        "CREATE TABLE IF NOT EXISTS review_queue ("
        "review_id TEXT PRIMARY KEY, video_id TEXT NOT NULL,"
        " segment_id TEXT NOT NULL, reason TEXT NOT NULL,"
        " status TEXT NOT NULL"
        " CHECK (status IN ('pending','approved','rejected')),"
        " created_at_utc TEXT NOT NULL,"
        " FOREIGN KEY(video_id) REFERENCES videos(video_id))",
    ]
    with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for statement in statements:
            conn.execute(statement)
        conn.commit()


class SqliteWriter:
    """Dev writer committing reconciled rows to a local SQLite database."""

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self.db_path = str(db_path)
        create_schema(self.db_path)

    def write_rows(self, table: str, rows: Iterable[dict[str, Any]]) -> int:
        """Insert rows into a table; returns the number of rows written.

        The *table* argument is validated against the known table list
        before any SQL is constructed, preventing SQL injection through
        dynamic table-name interpolation.
        """
        _validate_table_name(table)
        normalised = _validate_rows(table, rows)
        if not normalised:
            return 0
        columns = TABLE_COLUMNS[table]
        placeholders = ", ".join("?" for _ in columns)
        column_list = ", ".join(columns)
        sql = f"INSERT OR REPLACE INTO {table} ({column_list}) VALUES ({placeholders})"
        values = [tuple(row[col] for col in columns) for row in normalised]
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executemany(sql, values)
            conn.commit()
        return len(values)

    def write_all(self, batch: dict[str, Iterable[dict[str, Any]]]) -> dict[str, int]:
        """Write a full agent-loop batch atomically.

        All tables are validated first; then every table is written inside a
        single connection + transaction so a failure part-way does not leave a
        partially persisted batch.

        Every table name key in *batch* is validated against the known table
        list before any SQL is constructed, preventing SQL injection through
        dynamic table-name interpolation.
        """
        # Validate every table up front before touching the database.
        # A late validation failure (e.g. malformed rows in "counts") will
        # never leave earlier tables committed.
        validated: dict[str, list[dict[str, Any]]] = {}
        for table, rows in batch.items():
            _validate_table_name(table)
            validated[table] = _validate_rows(table, rows)

        counts: dict[str, int] = {}
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                for table, rows in validated.items():
                    if not rows:
                        counts[table] = 0
                        continue
                    columns = TABLE_COLUMNS[table]
                    placeholders = ", ".join("?" for _ in columns)
                    column_list = ", ".join(columns)
                    sql = (
                        f"INSERT OR REPLACE INTO {table}"
                        f" ({column_list}) VALUES ({placeholders})"
                    )
                    values = [tuple(row[col] for col in columns) for row in rows]
                    conn.executemany(sql, values)
                    counts[table] = len(values)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return counts

    #: Pattern for statements that may safely pass through ``query()``.
    _READONLY_STMT_PATTERN = re.compile(
        r"^\s*(SELECT|WITH|PRAGMA)\b", re.IGNORECASE
    )

    def query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Run a **read-only** SQL query and return rows as dicts.

        Raises ``SqlWriterError`` for text that is not a ``SELECT``,
        ``WITH``, or ``PRAGMA`` statement to prevent accidental data
        corruption through INSERT/UPDATE/DELETE/DROP.
        """
        if not self._READONLY_STMT_PATTERN.match(sql):
            raise SqlWriterError(
                f"query() only accepts read-only statements (SELECT, WITH, PRAGMA); "
                f"rejected: {sql!r}"
            )
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, parameters)
            return [dict(row) for row in cursor.fetchall()]


class JsonlWriter:
    """Write Athena-ready JSON-lines files to a local staging directory.

    Uploading to S3 is an explicit, owner-gated action outside this module;
    nothing here performs outbound transfer automatically.

    Each call to ``write_all()`` **overwrites** the target JSONL files to
    prevent silent row duplication on re-runs. Individual ``write_rows()``
    calls within a batch still append, but ``write_all()`` truncates each
    table's file at the start of the batch so the final state reflects
    exactly the batch contents.

    Staging layout (flat, no partition prefix)::

        {staging_dir}/
            videos.jsonl
            segments.jsonl
            events.jsonl
            ...
    """

    def __init__(self, staging_dir: str | os.PathLike[str]) -> None:
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def write_rows(self, table: str, rows: Iterable[dict[str, Any]]) -> int:
        """Append rows to a table's JSONL file.

        For single-table incremental writes, rows are appended.
        Call ``write_all()`` for full-batch overwrite semantics.

        The *table* argument is validated against the known table list
        before any file path is constructed, preventing path-traversal
        attacks through dynamic table-name interpolation.
        """
        _validate_table_name(table)
        normalised = _validate_rows(table, rows)
        if not normalised:
            return 0
        target = self.staging_dir / f"{table}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            for row in normalised:
                coerced = _coerce_jsonl_row(table, row)
                handle.write(json.dumps(coerced, ensure_ascii=True))
                handle.write("\n")
        return len(normalised)

    def write_all(self, batch: dict[str, Iterable[dict[str, Any]]]) -> dict[str, int]:
        """Write a full batch, **overwriting** each table's JSONL file.

        This prevents duplicate rows when the same batch is re-run.
        Each table's file is truncated before writing; the final state
        contains exactly the rows in this batch and nothing else.

        Every table name key in *batch* is validated against the known table
        list before any file path is constructed, preventing path-traversal
        attacks through dynamic table-name interpolation.
        """
        counts: dict[str, int] = {}
        for table, rows in batch.items():
            _validate_table_name(table)
            normalised = _validate_rows(table, rows)
            target = self.staging_dir / f"{table}.jsonl"
            if not normalised:
                # Truncate even for empty batches to remove stale data.
                target.write_text("", encoding="utf-8")
                counts[table] = 0
                continue
            with target.open("w", encoding="utf-8") as handle:
                for row in normalised:
                    coerced = _coerce_jsonl_row(table, row)
                    handle.write(json.dumps(coerced, ensure_ascii=True))
                    handle.write("\n")
            counts[table] = len(normalised)
        return counts
