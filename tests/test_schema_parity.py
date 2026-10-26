# build 7fd77e00de7f
"""The three table definitions (SQLite dev writer, schema.sql, Athena DDL)
and the JSON-lines writer agree on columns, and the constraints bite."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from v2s import sqlwriter  # noqa: E402

SCHEMA = (REPO_ROOT / "v2s" / "schema.sql").read_text(encoding="utf-8")
ATHENA = (REPO_ROOT / "v2s" / "schema.athena.sql").read_text(encoding="utf-8")


def _columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(r[1] for r in conn.execute(f"PRAGMA table_info({table})"))


def test_schema_sql_matches_the_dev_writer_tables(tmp_path) -> None:
    ddl = sqlite3.connect(":memory:")
    ddl.executescript(SCHEMA)
    dev_path = tmp_path / "dev.db"
    sqlwriter.create_schema(dev_path)
    dev = sqlite3.connect(str(dev_path))
    for table in sqlwriter.TABLES:
        assert _columns(ddl, table) == sqlwriter.TABLE_COLUMNS[table]
        assert _columns(dev, table) == sqlwriter.TABLE_COLUMNS[table]


def test_athena_ddl_matches_the_writer_columns() -> None:
    blocks = re.findall(
        r"CREATE EXTERNAL TABLE IF NOT EXISTS v2s\.(\w+) \((.*?)\)\s*ROW FORMAT",
        ATHENA,
        flags=re.S,
    )
    assert [t for t, _ in blocks] == list(sqlwriter.TABLES)
    for table, body in blocks:
        cols = tuple(re.findall(r"`(\w+)`", body))
        assert cols == sqlwriter.TABLE_COLUMNS[table], table


def test_jsonl_output_has_exactly_the_athena_columns(tmp_path) -> None:
    writer = sqlwriter.JsonlWriter(tmp_path)
    writer.write_rows("counts", [{
        "count_id": "c1", "video_id": "v1", "minute_index": 0,
        "class_label": "person", "count": 2, "confidence": 0.5,
    }])
    line = json.loads((tmp_path / "counts.jsonl").read_text().splitlines()[0])
    assert tuple(line) == sqlwriter.TABLE_COLUMNS["counts"]


def test_check_constraints_reject_out_of_range_confidence() -> None:
    """Negative control on SQLite: a bad confidence must not be storable."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO videos (video_id, uri, filename, source_url, license, processed_at_utc) VALUES ('v','u','f','s','l','2026-01-01T00:00:00Z')")
    conn.execute(
        "INSERT INTO objects VALUES ('o1','v','t','person',0,1,1,0.9)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO objects VALUES ('o2','v','t','person',0,1,1,1.5)"
        )


def test_schema_is_portable_to_duckdb() -> None:
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    con.execute(SCHEMA)
    con.execute("INSERT INTO videos (video_id, uri, filename, source_url, license, processed_at_utc) VALUES ('v','u','f','s','l','2026-01-01T00:00:00Z')")
    con.execute("INSERT INTO counts VALUES ('c','v',0,'person',2,0.8)")
    assert con.execute('SELECT "count" FROM counts').fetchone() == (2,)
    with pytest.raises(Exception):
        con.execute("INSERT INTO counts VALUES ('c2','v',0,'person',-1,0.8)")
