"""Judge-facing web endpoint: clip-relative links, clip/media routes, limits."""

from __future__ import annotations

import contextlib
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from v2s import nlq, sqlwriter  # noqa: E402

CLIP_BYTES = b"\x1a\x45\xdf\xa3" + b"fake-webm-payload-0123456789" * 8
REQUIRED_TABLES = ("videos", "segments", "events", "objects")
LONGEST_TRACKS_QUESTION = "What are the longest tracks?"


# --- local seeding helpers ---------------------------------------------------


def _tables(db_path: str) -> set[str]:
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        return {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }


def _ensure_schema(db_path: str) -> None:
    writer = sqlwriter.SqliteWriter(db_path)
    for attr in ("close",):
        fn = getattr(writer, attr, None)
        if callable(fn):
            with contextlib.suppress(Exception):
                fn()
    for attr in ("conn", "_conn", "connection"):
        c = getattr(writer, attr, None)
        if isinstance(c, sqlite3.Connection):
            with contextlib.suppress(Exception):
                c.commit()
                c.close()
    if not set(REQUIRED_TABLES) <= _tables(db_path) and hasattr(sqlwriter, "create_schema"):
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            try:
                sqlwriter.create_schema(conn)
                conn.commit()
            except TypeError:
                sqlwriter.create_schema(db_path)
    missing = set(REQUIRED_TABLES) - _tables(db_path)
    assert not missing, f"sqlwriter did not create tables: {missing}"


def _table_info(conn: sqlite3.Connection, table: str) -> list[tuple]:
    return list(conn.execute(f"PRAGMA table_info({table})"))


def _filler(name: str, coltype: str) -> Any:
    t = (coltype or "").upper()
    if "utc" in name:
        return "2026-03-01T10:00:00Z"
    if "confidence" in name:
        return 0.5
    if "INT" in t:
        return 0
    if any(k in t for k in ("REAL", "FLOA", "DOUB", "NUM")):
        return 0.0
    return "x"


def _complete_row(info: list[tuple], values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for _cid, name, coltype, notnull, dflt, pk in info:
        is_int_pk = pk and "INT" in (coltype or "").upper()
        if name in values:
            v = values[name]
            if is_int_pk and not isinstance(v, int):
                continue  # let SQLite assign the rowid
            out[name] = v
        elif notnull and dflt is None and not is_int_pk:
            out[name] = _filler(name, coltype)
    return out


def _insert(conn: sqlite3.Connection, table: str, values: dict[str, Any]) -> int:
    info = _table_info(conn, table)
    names = {r[1] for r in info}
    for key in ("video_id",):
        if key in values:
            assert key in names, f"{table} has no {key} column"
    row = _complete_row(info, values)
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    cur = conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
    return int(cur.lastrowid)


def _seed(tmp_path: Path, extra_videos: list[dict[str, Any]] | None = None) -> tuple[str, Path]:
    db_path = str(tmp_path / "links.db")
    clip = tmp_path / "clip_a.webm"
    clip.write_bytes(CLIP_BYTES)
    _ensure_schema(db_path)
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        _insert(conn, "videos", {
            "video_id": "vid_a", "uri": str(clip), "filename": "clip_a.webm",
            "source_url": "https://example.invalid/a", "license": "CC-BY-4.0",
            "processed_at_utc": "2026-03-02T00:00:00Z",
            "recording_start_utc": "2026-03-01T10:00:00Z",
        })
        seg_rowid = _insert(conn, "segments", {
            "segment_id": "seg_a", "video_id": "vid_a", "start_s": 0.0,
            "end_s": 10.0, "motion_score": 0.5,
        })
        seg_cols = {r[1]: r for r in _table_info(conn, "segments")}
        seg_id: Any = "seg_a"
        if "segment_id" in seg_cols and seg_cols["segment_id"][5] and "INT" in seg_cols["segment_id"][2].upper():
            seg_id = seg_rowid
        _insert(conn, "events", {
            "event_id": "evt_a", "video_id": "vid_a", "segment_id": seg_id,
            "event_type": "motion", "start_timestamp_utc": "2026-03-01T10:00:07Z",
            "end_timestamp_utc": "2026-03-01T10:00:09Z", "confidence": 0.9,
        })
        _insert(conn, "objects", {
            "object_id": "obj_a", "video_id": "vid_a", "track_id": 1,
            "class_label": "person", "first_seen_s": 3.5, "last_seen_s": 9.0,
            "max_count": 1, "confidence": 0.8,
        })
        _insert(conn, "videos", {
            "video_id": "vid_b", "uri": "/etc/hosts", "filename": "clip_b.webm",
            "source_url": "https://example.invalid/b", "license": "CC-BY-4.0",
            "processed_at_utc": "2026-03-02T00:00:01Z",
            "recording_start_utc": None,
        })
        _insert(conn, "events", {
            "event_id": "evt_b", "video_id": "vid_b", "segment_id": seg_id,
            "event_type": "motion", "start_timestamp_utc": "2026-03-01T10:00:07Z",
            "end_timestamp_utc": "2026-03-01T10:00:09Z", "confidence": 0.9,
        })
        for v in extra_videos or []:
            _insert(conn, "videos", v)
        conn.commit()

        # Verify the seeded state matches the brief before any test relies on it.
        expected = {
            "videos": 2 + len(extra_videos or []),
            "segments": 1,
            "events": 2,
            "objects": 1,
        }
        for table, n in expected.items():
            got = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert got == n, f"{table}: expected {n} rows, found {got}"
        rs = conn.execute("SELECT recording_start_utc FROM videos WHERE video_id='vid_b'").fetchone()
        assert rs is not None and rs[0] is None
        seg = conn.execute("SELECT video_id FROM segments").fetchone()
        assert seg[0] == "vid_a"
        obj = conn.execute("SELECT video_id, first_seen_s FROM objects").fetchone()
        assert obj[0] == "vid_a" and float(obj[1]) == 3.5
        evs = conn.execute(
            "SELECT video_id, start_timestamp_utc FROM events ORDER BY video_id"
        ).fetchall()
        assert [tuple(e) for e in evs] == [
            ("vid_a", "2026-03-01T10:00:07Z"),
            ("vid_b", "2026-03-01T10:00:07Z"),
        ]
    return db_path, clip


def _client(monkeypatch, db_path: str):
    monkeypatch.setenv("V2S_DB_PATH", db_path)
    monkeypatch.delenv("V2S_ALLOW_UNSAFE_SQL", raising=False)
    from web.app import create_app

    return create_app().test_client()


def _link_seconds(link: str) -> float:
    m = re.match(r"^/clip/[^?]+\?t=(\d+\.\d)$", link)
    assert m, f"malformed link {link!r}"
    return float(m.group(1))


# --- tests --------------------------------------------------------------------


def test_event_link_is_clip_relative_seconds(tmp_path, monkeypatch) -> None:
    db_path, _ = _seed(tmp_path)
    client = _client(monkeypatch, db_path)
    resp = client.post("/query", json={"sql": "SELECT event_id, video_id, start_timestamp_utc"
                                              " FROM events WHERE video_id='vid_a'"})
    assert resp.status_code == 200, resp.get_json()
    rows = resp.get_json()["rows"]
    assert len(rows) == 1
    assert rows[0]["video_link"] == "/clip/vid_a?t=7.0"
    for row in rows:
        assert _link_seconds(row["video_link"]) <= 86400.0


def test_event_without_recording_start_gets_no_link(tmp_path, monkeypatch) -> None:
    db_path, _ = _seed(tmp_path)
    client = _client(monkeypatch, db_path)
    resp = client.post("/query", json={"sql": "SELECT event_id, video_id, start_timestamp_utc"
                                              " FROM events WHERE video_id='vid_b'"})
    assert resp.status_code == 200, resp.get_json()
    rows = resp.get_json()["rows"]
    assert len(rows) == 1
    assert "video_link" not in rows[0], rows[0]


def test_object_link_uses_first_seen_s(tmp_path, monkeypatch) -> None:
    db_path, _ = _seed(tmp_path)
    client = _client(monkeypatch, db_path)
    resp = client.post("/query", json={"sql": "SELECT object_id, video_id, first_seen_s FROM objects"})
    assert resp.status_code == 200, resp.get_json()
    rows = resp.get_json()["rows"]
    assert len(rows) == 1
    assert rows[0]["video_link"] == "/clip/vid_a?t=3.5"


def test_question_rows_get_links_via_filename(tmp_path, monkeypatch) -> None:
    db_path, _ = _seed(tmp_path)
    client = _client(monkeypatch, db_path)

    # longest_tracks: rows carry filename + first_seen_s and no video_id.
    assert nlq.translate(LONGEST_TRACKS_QUESTION)[1] == "longest_tracks"
    resp = client.post("/query", json={"question": LONGEST_TRACKS_QUESTION})
    assert resp.status_code == 200, resp.get_json()
    rows = resp.get_json()["rows"]
    assert rows, "longest-tracks question returned no rows"
    row = rows[0]
    assert "video_id" not in row
    assert row["filename"] == "clip_a.webm"
    assert float(row["first_seen_s"]) == 3.5
    for r in rows:
        if "video_link" in r:
            assert r["video_link"].startswith("/clip/vid_a?t=")

    # busiest_second: rows carry filename + second and no video_id -> linked.
    question = "When is the busiest moment?"
    assert nlq.translate(question)[1] == "busiest_second"
    resp = client.post("/query", json={"question": question})
    assert resp.status_code == 200, resp.get_json()
    rows = resp.get_json()["rows"]
    assert rows, "busiest-second question returned no rows"
    row = rows[0]
    assert "video_id" not in row and row["filename"] == "clip_a.webm" and "second" in row
    link = row["video_link"]
    assert link.startswith("/clip/vid_a?t=")
    t = _link_seconds(link)
    assert t == float(row["second"])
    assert 3.5 <= t <= 9.0


def test_longest_tracks_rows_link_to_vid_a(tmp_path, monkeypatch) -> None:
    db_path, _ = _seed(tmp_path)
    client = _client(monkeypatch, db_path)
    resp = client.post("/query", json={"question": LONGEST_TRACKS_QUESTION})
    assert resp.status_code == 200
    rows = resp.get_json()["rows"]
    assert rows
    assert rows[0]["video_link"] == "/clip/vid_a?t=3.5"


def test_clip_default_is_json_and_html_for_browsers(tmp_path, monkeypatch) -> None:
    db_path, _ = _seed(tmp_path)
    client = _client(monkeypatch, db_path)

    resp = client.get("/clip/vid_a?t=2.0")
    assert resp.status_code == 200
    assert resp.mimetype == "application/json"
    data = resp.get_json()
    assert data is not None and data["ok"] is True
    assert data["filename"] == "clip_a.webm"

    resp = client.get("/clip/vid_a?t=2.0", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    body = resp.get_data(as_text=True)
    assert "clip_a.webm" in body and "2.0" in body

    resp = client.get("/clip/vid_a?t=2.0&format=json", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert resp.mimetype == "application/json"
    data = resp.get_json()
    assert data["ok"] is True and data["filename"] == "clip_a.webm"


def test_html_clip_page_escapes_values(tmp_path, monkeypatch) -> None:
    evil = "<script>alert(1)</script>.webm"
    db_path, _ = _seed(tmp_path, extra_videos=[{
        "video_id": "vid_c", "uri": str(tmp_path / "missing.webm"), "filename": evil,
        "source_url": "https://example.invalid/c", "license": "CC-BY-4.0",
        "processed_at_utc": "2026-03-02T00:00:02Z",
        "recording_start_utc": "2026-03-01T10:00:00Z",
    }])
    client = _client(monkeypatch, db_path)
    resp = client.get("/clip/vid_c?t=1.0", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    body = resp.get_data(as_text=True)
    assert "&lt;script&gt;" in body
    assert "<script>alert" not in body


def test_media_route_serves_only_allowed_files(tmp_path, monkeypatch) -> None:
    db_path, clip = _seed(tmp_path)
    monkeypatch.setenv("V2S_CLIP_DIR", str(tmp_path))
    client = _client(monkeypatch, db_path)

    resp = client.get("/media/vid_a")
    assert resp.status_code == 200
    assert resp.get_data() == clip.read_bytes() == CLIP_BYTES
    resp.close()

    resp = client.get("/media/vid_b")  # uri is /etc/hosts, outside V2S_CLIP_DIR
    assert resp.status_code == 404
    assert resp.get_json()["ok"] is False

    resp = client.get("/media/../x")
    assert resp.status_code in (400, 404)
    assert resp.status_code != 200


def test_sql_row_cap_and_runtime_limit(tmp_path, monkeypatch) -> None:
    db_path, _ = _seed(tmp_path)
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        info = _table_info(conn, "objects")
        rows = [
            _complete_row(info, {
                "object_id": f"bulk_{i}", "video_id": "vid_a", "track_id": 1000 + i,
                "class_label": "car", "first_seen_s": float(i % 50),
                "last_seen_s": float(i % 50) + 1.0, "max_count": 1, "confidence": 0.7,
            })
            for i in range(1500)
        ]
        cols = list(rows[0])
        conn.executemany(
            f"INSERT INTO objects ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
            [tuple(r[c] for c in cols) for r in rows],
        )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 1501

    client = _client(monkeypatch, db_path)
    resp = client.post("/query", json={"sql": "SELECT * FROM objects"})
    assert resp.status_code == 200, resp.get_json()
    returned = resp.get_json()["rows"]
    assert len(returned) <= 1000
    assert len(returned) == 1000

    expensive = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c"
        " WHERE x < 200000000) SELECT count(*) FROM c"
    )
    started = time.monotonic()
    resp = client.post("/query", json={"sql": expensive})
    elapsed = time.monotonic() - started
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["ok"] is False
    assert "expensive" in data["error"], data
    assert elapsed < 15.0, f"expensive query ran {elapsed:.1f}s"
