"""export_trace.py: real sqlite databases, no mocks (HOLD #1499 flip items 1 and 2)."""
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_trace.py"
REAL_DB = ROOT / "evidence" / "real_clips.db"
SCHEMA = (ROOT / "v2s" / "schema.sql").read_text(encoding="utf-8")


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, timeout=60)


def make_db(path, trace_rows=(), review_rows=()):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO videos (video_id, uri, filename, source_url, license, processed_at_utc) VALUES ('v1', 'file://a.avi', 'a.avi', 'https://example.org/a', 'CC0', '2026-10-23T00:00:00Z')")
    conn.execute("INSERT INTO segments (segment_id, video_id, start_s, end_s, reason) VALUES ('s1', 'v1', 0.0, 2.0, 'scene_cut')")
    for n, r in enumerate(trace_rows):
        conn.execute("INSERT INTO agent_trace (trace_id, video_id, segment_id, step, observation, decision, reason, created_at_utc) VALUES (?,?,?,?,?,?,?,?)", (f"t{n}",) + tuple(r))
    for n, r in enumerate(review_rows):
        conn.execute("INSERT INTO review_queue (review_id, video_id, segment_id, reason, status, created_at_utc) VALUES (?,?,?,?,?,?)", (f"r{n}",) + tuple(r))
    conn.commit()
    conn.close()


def test_real_evidence_db_report_exit_0_and_documented_review_columns():
    assert REAL_DB.is_file(), "evidence/real_clips.db must be present for this test"
    r = run(REAL_DB)
    assert r.returncode == 0, r.stderr
    assert "| review_id | video | segment | reason | status | created_at_utc |" in r.stdout
    assert "| start_s | end_s | step | decision | reason | observation |" in r.stdout
    assert len(r.stdout.splitlines()) > 50


def test_nested_observation_has_no_colon_after_a_key():
    obs = '{"first": {"tracks": 3, "short": 2}, "retry": {"tracks": 4, "short": 1}}'
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        make_db(db, [("v1", "s1", "retry", obs, "keep_first", "retry did not help", "2026-10-23T00:00:00Z")])
        r = run(db)
    assert r.returncode == 0, r.stderr
    cell = [l for l in r.stdout.splitlines() if "keep_first" in l][0]
    assert "first={" in cell and "retry={" in cell
    assert not re.search(r"\b(first|retry):", cell)


def test_empty_agent_trace_is_exit_0_with_a_clear_report(tmp_path):
    db = tmp_path / "e.db"
    make_db(db)
    r = run(db)
    assert r.returncode == 0, r.stderr
    assert "No agent trace rows" in r.stdout and "No review_queue rows" in r.stdout


def test_missing_tables_still_exit_1(tmp_path):
    db = tmp_path / "bad.db"
    sqlite3.connect(db).executescript("CREATE TABLE videos (video_id TEXT, filename TEXT);")
    r = run(db)
    assert r.returncode == 1 and "missing table" in r.stderr


def test_review_rows_have_exactly_six_cells(tmp_path):
    db = tmp_path / "r.db"
    make_db(db, [("v1", "s1", "plan", '{"a": 1}', "default", "r", "2026-10-23T00:00:00Z")],
            [("v1", "s1", "low confidence", "pending", "2026-10-23T00:00:01Z")])
    r = run(db)
    row = [l for l in r.stdout.splitlines() if "low confidence" in l][0]
    assert row.count(" | ") == 5
