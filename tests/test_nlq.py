# vid2sql: tests for the rule-based question translator.
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2s import nlq, sqlwriter  # noqa: E402
from web import app as webapp  # noqa: E402


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "q.db")
    w = sqlwriter.SqliteWriter(path)
    for vid, name in (("vid_a", "a.webm"), ("vid_b", "b.webm")):
        w.write_rows("videos", [{"video_id": vid, "uri": "/x/" + name, "filename": name,
                                 "source_url": "https://e.org/" + name, "license": "CC BY 4.0",
                                 "processed_at_utc": "2026-10-16T00:00:00Z",
                                 "recording_start_utc": "2026-10-16T00:00:00Z"}])
    def obj(i, vid, a, b):
        return {"object_id": f"o{i}", "video_id": vid, "track_id": f"t{i}",
                "class_label": "moving_object", "first_seen_s": a, "last_seen_s": b,
                "max_count": 1, "confidence": 0.6}
    # vid_a: three tracks, all alive at t=2 (peak 3). vid_b: one track.
    w.write_rows("objects", [obj(1, "vid_a", 0.0, 4.0), obj(2, "vid_a", 1.0, 3.0),
                             obj(3, "vid_a", 2.0, 9.0), obj(4, "vid_b", 0.0, 1.0)])
    return path


def _ask(db, q):
    sql, name = nlq.translate(q)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        return name, [dict(r) for r in c.execute(sql)]


def test_how_many_objects_exact_counts(db):
    name, rows = _ask(db, "How many moving objects are in each video?")
    assert name == "objects_per_video"
    assert {r["filename"]: r["moving_objects"] for r in rows} == {"a.webm": 3, "b.webm": 1}


def test_busiest_second_is_the_true_peak(db):
    name, rows = _ask(db, "When is each video busiest?")
    assert name == "busiest_second"
    by = {r["filename"]: r for r in rows}
    assert by["a.webm"]["active_tracks"] == 3 and by["a.webm"]["second"] == 2
    assert by["b.webm"]["active_tracks"] == 1


def test_longest_track(db):
    name, rows = _ask(db, "Which object stays on screen the longest?")
    assert name == "longest_tracks"
    assert rows[0]["track_id"] == "t3" and rows[0]["seconds_visible"] == 7.0


def test_most_activity_video(db):
    name, rows = _ask(db, "Which video has the most activity?")
    assert name == "most_activity" and rows[0]["filename"] == "a.webm"


def test_unsupported_question_is_rejected_not_guessed():
    with pytest.raises(nlq.UnsupportedQuestion):
        nlq.translate("What colour is the bus?")
    with pytest.raises(nlq.UnsupportedQuestion):
        nlq.translate("   ")


def test_every_translation_passes_the_web_read_only_gate():
    for q in ("busiest moment", "longest object visible", "which clip has the most",
              "how many objects", "confidence?", "scenes", "licence"):
        sql, _ = nlq.translate(q)
        assert webapp._is_safe_select(sql), q


def test_web_endpoint_answers_through_nlq(db, monkeypatch):
    monkeypatch.setenv("V2S_DB_PATH", db)
    client = webapp.create_app().test_client()
    r = client.post("/query", json={"question": "How many moving objects are in each video?"})
    assert r.status_code == 200
    assert {x["filename"]: x["moving_objects"] for x in r.get_json()["rows"]} == {"a.webm": 3, "b.webm": 1}


def _one_video_db(tmp_path):
    # Fresh database whose schema is created by sqlwriter, holding exactly one videos row.
    path = str(tmp_path / "agent.db")
    w = sqlwriter.SqliteWriter(path)
    w.write_rows("videos", [{"video_id": "vid_a", "uri": "/x/a.webm", "filename": "a.webm",
                             "source_url": "https://e.org/a.webm", "license": "CC BY 4.0",
                             "processed_at_utc": "2026-10-16T00:00:00Z",
                             "recording_start_utc": "2026-10-16T00:00:00Z"}])
    return path


def _insert_row(path, table, values, uniq):
    # Insert one row into a table created by sqlwriter's schema. Required columns that
    # are not given explicitly get a type-appropriate placeholder (unique for text).
    with sqlite3.connect(path) as c:
        info = list(c.execute(f"PRAGMA table_info({table})"))
        assert info, f"table {table} missing from schema"
        row = dict(values)
        for _cid, col, typ, notnull, dflt, pk in info:
            if col in row:
                continue
            t = (typ or "").upper()
            if pk and "INT" in t:
                continue
            if not (notnull or pk) or (dflt is not None and not pk):
                continue
            if "INT" in t:
                row[col] = 0
            elif any(k in t for k in ("REAL", "FLOA", "DOUB", "NUM")):
                row[col] = 0.0
            else:
                row[col] = f"{col}_{uniq}"
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        c.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
        c.commit()


def _run_ro(path, sql):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(sql)]


def test_needs_review_returns_pending_row(tmp_path):
    path = _one_video_db(tmp_path)
    _insert_row(path, "review_queue", {"video_id": "vid_a", "status": "pending"}, "r1")
    sql, name = nlq.translate("What needs review?")
    assert name == "needs_review"
    rows = _run_ro(path, sql)
    assert len(rows) == 1
    assert set(rows[0].values()) & {"vid_a", "a.webm"}


def test_agent_decisions_grouped_counts(tmp_path):
    path = _one_video_db(tmp_path)
    for i in (1, 2):
        _insert_row(path, "agent_trace", {"video_id": "vid_a", "step": "detect",
                                          "decision": "flag_for_review"}, f"a{i}")
    sql, name = nlq.translate("What did the agent decide?")
    assert name == "agent_decisions"
    rows = _run_ro(path, sql)
    assert len(rows) == 1
    assert rows[0]["n"] == 2
    assert {"detect", "flag_for_review"} <= set(rows[0].values())


def test_segments_needing_human_review_is_not_scenes():
    sql, name = nlq.translate("Which segments need human review?")
    assert isinstance(sql, str) and sql.strip()
    assert name == "needs_review"
    assert name != "scenes"
