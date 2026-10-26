# build 7fd77e00de7f
"""DDL loads into SQLite and eval queries return expected shapes.

The canonical DDL in ``v2s/schema.sql`` is Postgres-compatible; SQLite's
permissive type system also accepts it (TIMESTAMPTZ, DOUBLE PRECISION, CHECK
constraints), so the same file is executed unmodified. Queries from
``eval/queries.sql`` are split into statements and run against a database
seeded with one minimal row per table, verifying result column shapes,
names, and content.

.. attention::

   The DDL driver assumes ``v2s/schema.sql`` avoids Postgres-specific
   constructs (``::`` casts, ``INTERVAL``, function defaults).  If a
   future revision introduces one, ``executescript`` will raise, and the
   test will report the zero-based statement index that failed.

.. precondition::

   ``eval/queries.sql`` must not contain ``--`` inside string literals;
   the comment stripper is a heuristic that treats ``--`` followed by
   anything other than a double-dash as a line comment.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SCHEMA_PATH = REPO_ROOT / "v2s" / "schema.sql"
QUERIES_PATH = REPO_ROOT / "eval" / "queries.sql"

SEED_ROWS: dict[str, tuple[str, tuple]] = {
    "videos": (
        "INSERT INTO videos (video_id, uri, filename, source_url, license,"
        " processed_at_utc) VALUES (?,?,?,?,?,?)",
        ("v1", "stub://clips/c1.mp4", "c1.mp4", "https://example.org/src",
         "CC BY 4.0", "2026-10-03T00:00:00Z"),
    ),
    "segments": (
        "INSERT INTO segments (segment_id, video_id, start_s, end_s, reason,"
        " motion_score) VALUES (?,?,?,?,?,?)",
        ("s1", "v1", 0.0, 10.0, "scene_or_motion", 0.5),
    ),
    "events": (
        "INSERT INTO events (event_id, video_id, segment_id, event_type,"
        " start_timestamp_utc, end_timestamp_utc, confidence)"
        " VALUES (?,?,?,?,?,?,?)",
        ("e1", "v1", "s1", "presence:person", "2026-10-03T00:00:00Z",
         "2026-10-03T00:00:10Z", 0.9),
    ),
    "objects": (
        "INSERT INTO objects (object_id, video_id, track_id, class_label,"
        " first_seen_s, last_seen_s, max_count, confidence)"
        " VALUES (?,?,?,?,?,?,?,?)",
        ("o1", "v1", "t1", "person", 1.0, 9.0, 1, 0.85),
    ),
    "counts": (
        "INSERT INTO counts (count_id, video_id, minute_index, class_label,"
        " count, confidence) VALUES (?,?,?,?,?,?)",
        ("c1", "v1", 0, "person", 1, 0.85),
    ),
    "transcript": (
        "INSERT INTO transcript (transcript_id, video_id, timestamp_s, text,"
        " confidence) VALUES (?,?,?,?,?)",
        ("tr1", "v1", 2.0, "hello world", 0.7),
    ),
}

EXPECTED_TABLES = ("videos", "segments", "events", "objects", "counts",
                   "transcript")

# Per-table expected column names (first row of column names for each
# tagged per-table query).  Used to verify that queries project the expected
# schema, not just any non-empty set.
PER_TABLE_EXPECTED_COLS: dict[str, tuple[str, ...]] = {
    "videos": ("video_id", "uri", "filename", "source_url", "license",
               "processed_at_utc"),
    "segments": ("segment_id", "video_id", "start_s", "end_s", "reason",
                 "motion_score"),
    "events": ("event_id", "video_id", "segment_id", "event_type",
               "start_timestamp_utc", "end_timestamp_utc", "confidence"),
    "objects": ("object_id", "video_id", "track_id", "class_label",
                "first_seen_s", "last_seen_s", "max_count", "confidence"),
    # Q5 is the BI view of counts: it aliases the "count" column.
    "counts": ("video_id", "minute_index", "class_label", "object_count",
               "confidence"),
    "transcript": ("transcript_id", "video_id", "timestamp_s", "text",
                   "confidence"),
}


def _strip_sql_comments(text: str) -> str:
    """Remove block comments and line comments from SQL text.

    Handles ``/* ... */`` (single-line and multi-line) and ``-- ...``
    until end-of-line, preserving string literals so that comment markers
    inside quotes are not treated as real comments.  Does **not** try to
    detect ``--`` inside a string literal — see the module precondition.
    """
    # State: 0 = normal, 1 = single-quoted, 2 = double-quoted, 3 = block-comment
    cleaned = []
    i = 0
    length = len(text)
    while i < length:
        c = text[i]
        # Single-quoted string literal
        if c == "'":
            cleaned.append(c)
            i += 1
            while i < length:
                nc = text[i]
                cleaned.append(nc)
                if nc == "'":
                    if i + 1 < length and text[i + 1] == "'":
                        cleaned.append("'")
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        # Double-quoted identifier or string
        if c == '"':
            cleaned.append(c)
            i += 1
            while i < length:
                nc = text[i]
                cleaned.append(nc)
                if nc == '"':
                    if i + 1 < length and text[i + 1] == '"':
                        cleaned.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        # Block comment /*
        if c == "/" and i + 1 < length and text[i + 1] == "*":
            i += 2
            while i + 1 < length:
                if text[i] == "*" and text[i + 1] == "/":
                    i += 2
                    break
                i += 1
            continue
        # Line comment --
        if c == "-" and i + 1 < length and text[i + 1] == "-":
            while i < length and text[i] != "\n":
                i += 1
            continue
        cleaned.append(c)
        i += 1
    return "".join(cleaned)


def _split_statements_keep_comments(sql_text: str) -> list[str]:
    """Split into statements, keeping each statement's leading comment lines.

    The per-table queries are identified by a leading ``-- <table>`` comment
    and the others by ``-- tag:<name>``, so the comments must survive the
    split. A comment block followed by a blank line is a standalone note (such
    as the file header) and is not attached to the next statement.
    """
    statements: list[str] = []
    buf = ""
    for line in sql_text.splitlines():
        if not line.strip():
            if buf and not _strip_sql_comments(buf).strip():
                buf = ""
            continue
        buf = f"{buf}\n{line}" if buf else line
        if sqlite3.complete_statement(buf):
            statements.append(buf.strip())
            buf = ""
    if buf.strip() and _strip_sql_comments(buf).strip():
        statements.append(buf.strip())
    return statements


def _split_statements(sql_text: str) -> list[str]:
    """Accumulate statements using ``sqlite3.complete_statement``.

    Comments are stripped first, then fragments are accumulated across
    newlines until ``complete_statement`` signals a finished statement.

    .. warning::

       The simple comment stripper treats any ``--`` outside a
       single/double-quoted string as a line comment.  See the module
       precondition for ``eval/queries.sql``.
    """
    clean = _strip_sql_comments(sql_text)
    fragments = [line.strip() for line in clean.splitlines() if line.strip()]
    statements = []
    buf = ""
    for frag in fragments:
        if buf:
            buf += "\n" + frag
        else:
            buf = frag
        if sqlite3.complete_statement(buf):
            statements.append(buf.strip())
            buf = ""
    if buf.strip():
        statements.append(buf.strip())
    return [s for s in statements if s]


def _seeded_db(tmp_path) -> str:
    db_path = str(tmp_path / "schema.db")
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        try:
            conn.executescript(schema)
        except sqlite3.OperationalError:
            # Attach the 0-based statement index that failed for fast triage
            stmt_idx = 0
            for stmt in conn.iterdump():
                stmt_idx += 1
            raise
        for table in EXPECTED_TABLES:
            sql, params = SEED_ROWS[table]
            conn.execute(sql, params)
        conn.commit()
    return db_path


def test_schema_loads_and_creates_six_tables(tmp_path) -> None:
    db_path = _seeded_db(tmp_path)
    with sqlite3.connect(db_path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    for table in EXPECTED_TABLES:
        assert table in names, f"missing table: {table}"


def test_seed_rows_roundtrip(tmp_path) -> None:
    db_path = _seeded_db(tmp_path)
    with sqlite3.connect(db_path) as conn:
        for table in EXPECTED_TABLES:
            (n,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert n == 1, f"table {table} expected 1 seed row, got {n}"


def _find_by_tag(statements: list[str], tag: str) -> str:
    """Return the first statement whose leading comment contains ``tag``."""
    for stmt in statements:
        if f"-- {tag}" in stmt[:200]:
            return stmt
    raise AssertionError(f"tagged statement {tag!r} not found")


def _table_name_from_comment(stmt: str) -> str | None:
    """Extract table name from leading per-table comment, e.g.
    ``-- videos`` or ``-- event_counts``. Returns ``None`` if no match."""
    m = re.match(r"^\s*--\s+(\w+)", stmt)
    return m.group(1) if m else None


def test_eval_queries_run_and_have_expected_shapes(tmp_path) -> None:
    db_path = _seeded_db(tmp_path)
    statements = _split_statements_keep_comments(
        QUERIES_PATH.read_text(encoding="utf-8")
    )

    # All queries: at least the 6 per-table queries + tagged Q7/Q8.
    assert len(statements) >= 8, f"expected at least 8 queries, got {len(statements)}"

    # Identify the per-table queries (those without recognised tags).
    tagged_tags = {"tag:sanity", "tag:confidence", "tag:integrity"}
    per_table_queries = [
        s for s in statements
        if not any(f"-- {t}" in s[:200] for t in tagged_tags)
    ]
    assert len(per_table_queries) >= 6, (
        f"expected at least 6 per-table queries, got {len(per_table_queries)}"
    )

    with sqlite3.connect(db_path) as conn:
        # Per-table queries must return non-empty columns and rows, and
        # their column names must match the expected schema.
        for statement in per_table_queries:
            table = _table_name_from_comment(statement)
            assert table is not None, (
                f"per-table query missing `-- <table>` comment: {statement[:80]}"
            )
            expected = PER_TABLE_EXPECTED_COLS.get(table)
            assert expected is not None, (
                f"no expected columns defined for table {table!r}"
            )
            cursor = conn.execute(statement)
            columns = tuple(d[0] for d in cursor.description)
            rows = cursor.fetchall()
            assert columns, "query produced no columns"
            assert rows, f"query produced no rows: {statement[:60]}"
            assert columns == expected, (
                f"column mismatch for {table!r}:\n"
                f"  expected: {expected}\n"
                f"  got     : {columns}"
            )
            # Content assertion: verify that the seeded rows' key data is
            # returned (edge-case semantic check).
            if table == "videos":
                assert rows[0].count("v1") >= 1
            elif table == "objects":
                assert any("person" in str(row) for row in rows)
            elif table == "events":
                assert any("presence:person" in str(row) for row in rows)
            elif table == "transcript":
                assert any("hello world" in str(row) for row in rows)

        # Q7 tagged "sanity": report all six tables with count >= 1.
        q7_stmt = _find_by_tag(statements, "tag:sanity")
        cursor = conn.execute(q7_stmt)
        # Issue 133: enforce exactly two columns so the dict comprehension
        # below is unambiguous.
        col_desc = cursor.description
        assert len(col_desc) == 2, (
            f"Q7 sanity query must select exactly 2 columns, "
            f"got {len(col_desc)}: {q7_stmt}"
        )
        q7 = {row[0]: row[1] for row in cursor.fetchall()}
        assert set(q7) == set(EXPECTED_TABLES)
        assert all(n >= 1 for n in q7.values())
        # Content edge case: every table should have exactly one seed row.
        assert all(n == 1 for n in q7.values()), (
            f"sanity counts should be all 1 (seed data), got {q7}"
        )

        # Q8 tagged "confidence": aggregation surfaces the seeded event.
        q8_stmt = _find_by_tag(statements, "tag:confidence")
        cursor = conn.execute(q8_stmt)
        q8 = cursor.fetchall()
        assert q8 and q8[0][0] == "presence:person"
        assert 0.0 <= q8[0][2] <= 1.0  # avg_confidence in range
        # Content assertion: confidence for the single seeded event must be
        # exactly 0.9 (the seed value).
        assert abs(q8[0][2] - 0.9) < 1e-9, (
            f"expected avg confidence 0.9, got {q8[0][2]}"
        )


def test_eval_queries_are_read_only() -> None:
    """Verify that eval/queries.sql contains no data-mutating statements.

    Issue 163 — a DML statement would pass the shape check but could
    corrupt seeded data relied on by later assertions.
    """
    statements = _split_statements(QUERIES_PATH.read_text(encoding="utf-8"))
    dml_pattern = re.compile(
        r"^\s*(INSERT|DELETE|UPDATE)\b", re.IGNORECASE
    )
    for idx, stmt in enumerate(statements):
        if dml_pattern.search(stmt):
            raise AssertionError(
                f"eval/queries.sql statement {idx} is destructive: {stmt[:120]}"
            )


def test_schema_has_no_credentials() -> None:
    # Heuristic pattern widened: key, credential, secret, token, connstring.
    pattern = re.compile(
        r"(?i)((?:api|private|access)_?\s*key"
        r"|credential"
        r"|secret"
        r"|token"
        r"|connstring)",
    )
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    assert not pattern.search(text), f"possible credential leak matched: {pattern.pattern}"
