# vid2sql: tests for the real-clip evaluation checks.
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import run_eval  # noqa: E402
from v2s import sqlwriter  # noqa: E402


def _db(tmp_path, first, last, conf):
    path = tmp_path / "e.db"
    w = sqlwriter.SqliteWriter(str(path))
    w.write_rows("videos", [{"video_id": "v", "uri": "u", "filename": "c.webm", "source_url": "s",
                             "license": "CC0", "processed_at_utc": "2026-10-16T00:00:00Z",
                             "recording_start_utc": "2026-10-16T00:00:00Z"}])
    w.write_rows("objects", [{"object_id": "o", "video_id": "v", "track_id": "t",
                              "class_label": "moving_object", "first_seen_s": first,
                              "last_seen_s": last, "max_count": 1, "confidence": conf}])
    return path


def test_invariants_pass_on_a_valid_database(tmp_path):
    assert run_eval._invariants(_db(tmp_path, 1.0, 3.0, 0.6), {"c.webm": 10.0}) == []


def test_invariants_catch_an_interval_past_the_end_of_the_clip(tmp_path):
    problems = run_eval._invariants(_db(tmp_path, 1.0, 30.0, 0.6), {"c.webm": 10.0})
    assert problems and "outside" in problems[0]


def test_stats_count_simultaneous_tracks(tmp_path):
    stats = run_eval._stats(_db(tmp_path, 1.0, 3.0, 0.6), {"c.webm": 10.0})["c.webm"]
    assert stats["tracks"] == 1 and stats["peak_simultaneous_tracks"] == 1


def test_checksum_helper_detects_a_changed_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    a = run_eval._sha256(f)
    f.write_bytes(b"abd")
    assert run_eval._sha256(f) != a
