# build 7fd77e00de7f
"""Web endpoint answers a SQL query with timestamps; includes no-IP scan.

Part 1 seeds a temp SQLite database via the real agent loop with stub
services, points the Flask app at it through ``V2S_DB_PATH`` (read lazily at
request time), and verifies that /query returns rows annotated with
clickable video timestamp links.

Part 2 is the no-IP file scan: repository text files must contain no model
or provider names, no prompts, no secrets, and no forbidden legacy strings.
The scan covers only end-product surface files; the test files themselves
(which necessarily name the forbidden patterns) are excluded.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from v2s import agent, sqlwriter  # noqa: E402
from v2s.ingest import ClipWorkItem  # noqa: E402

LABEL_DIR = REPO_ROOT / "eval" / "labels"


class _StubCrowd:
    def __init__(self, labels: dict[str, Any]) -> None:
        self.labels = labels

    def crowd_read(self, video_uri, segments):
        return {
            "rows": [
                {
                    "track_id": o["track_id"],
                    "class_label": o["class_label"],
                    "confidence": 0.8,
                    "timestamp_s": float(o["first_seen_s"]),
                }
                for o in self.labels.get("objects", [])
            ]
        }

    def reconcile(self, local_rows, crowd_rows):
        merged = {r.get("track_id"): dict(r) for r in local_rows}
        for row in crowd_rows:
            existing = merged.get(row.get("track_id"))
            if existing is None or row.get("confidence", 0.0) >= existing.get(
                "confidence", 0.0
            ):
                merged[row.get("track_id")] = dict(row)
        return {"rows": list(merged.values())}


def _seed_db(db_path: str) -> None:
    labels = json.loads((LABEL_DIR / "clip1.json").read_text(encoding="utf-8"))
    clip = ClipWorkItem(
        uri="stub://clips/clip1_retail.mp4",
        filename="clip1_retail.mp4",
        source_url=labels["source_url"],
        license=labels["license"],
    )
    writer = sqlwriter.SqliteWriter(db_path)
    agent.process_clip(
        clip,
        writer,
        crowd_client=_StubCrowd(labels),
        sample_transport=lambda path, payload: {
            "segments": list(labels.get("segments", []))
        },
        detect_transport=lambda path, payload: {
            "detections": [
                {
                    "track_id": o["track_id"],
                    "class_label": o["class_label"],
                    "confidence": 0.85,
                    "timestamp_s": float(o["first_seen_s"]),
                    "bounding_box": None,
                }
                for o in labels.get("objects", [])
            ]
        },
    )


def test_endpoint_returns_rows_with_timestamps(tmp_path, monkeypatch) -> None:
    db_path = str(tmp_path / "web.db")
    _seed_db(db_path)
    monkeypatch.setenv("V2S_DB_PATH", db_path)

    from web.app import create_app

    client = create_app().test_client()

    resp = client.get("/healthz")
    assert resp.status_code == 200 and resp.get_json()["ok"] is True

    # Raw read-only SQL query.
    resp = client.post(
        "/query",
        json={"sql": "SELECT object_id, video_id, track_id, class_label,"
                     " first_seen_s, last_seen_s, max_count, confidence"
                     " FROM objects"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["rows"], "expected object rows"
    for row in data["rows"]:
        assert "video_link" in row, "row missing clickable timestamp link"
        assert re.match(r"^/clip/.+\?t=\d+\.\d$", row["video_link"])

    # Clip route resolves the link to the source video at the timestamp.
    video_id = data["rows"][0]["video_id"]
    resp = client.get(f"/clip/{video_id}?t=2.0")
    assert resp.status_code == 200
    clip = resp.get_json()
    assert clip["ok"] is True and clip["start_s"] == 2.0
    assert clip["filename"] == "clip1_retail.mp4"


def test_clip_route_invalid_t_values(tmp_path, monkeypatch) -> None:
    """Edge cases for /clip/{video_id}?t=... with invalid or missing values."""
    db_path = str(tmp_path / "web_clip.db")
    _seed_db(db_path)
    monkeypatch.setenv("V2S_DB_PATH", db_path)

    from web.app import create_app
    client = create_app().test_client()
    video_id = "clip1_retail.mp4"

    # Missing t parameter
    resp = client.get(f"/clip/{video_id}")
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    # t=NaN or non-numeric string
    resp = client.get(f"/clip/{video_id}?t=NaN")
    assert resp.status_code == 400
    resp = client.get(f"/clip/{video_id}?t=abc")
    assert resp.status_code == 400

    # t=-1 (negative)
    resp = client.get(f"/clip/{video_id}?t=-1")
    assert resp.status_code == 400


def test_clip_route_invalid_video_id(tmp_path, monkeypatch) -> None:
    """Requesting a non-existent or invalid video_id must return 404."""
    db_path = str(tmp_path / "web_clip_badid.db")
    _seed_db(db_path)
    monkeypatch.setenv("V2S_DB_PATH", db_path)

    from web.app import create_app
    client = create_app().test_client()

    # Video ID that definitely does not exist
    resp = client.get("/clip/nonexistent_video.mp4?t=1.0")
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["ok"] is False

    # Video ID with path traversal attempt must be rejected
    resp = client.get("/clip/../../etc/passwd?t=1.0")
    assert resp.status_code == 400


def test_query_malformed_json(tmp_path, monkeypatch) -> None:
    """Malformed JSON in /query body returns a proper 400 error."""
    db_path = str(tmp_path / "web_malformed.db")
    _seed_db(db_path)
    monkeypatch.setenv("V2S_DB_PATH", db_path)

    from web.app import create_app
    client = create_app().test_client()

    resp = client.post("/query", data="not json", content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    resp = client.post("/query", data="{invalid", content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False


def test_query_unsafe_statements(tmp_path, monkeypatch) -> None:
    """Unsafe SQL statements (DELETE, UPDATE, INSERT, ALTER) must be rejected."""
    db_path = str(tmp_path / "web_unsafe.db")
    _seed_db(db_path)
    monkeypatch.setenv("V2S_DB_PATH", db_path)

    from web.app import create_app
    client = create_app().test_client()

    # DELETE
    resp = client.post("/query", json={"sql": "DELETE FROM objects"})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    # UPDATE
    resp = client.post("/query", json={"sql": "UPDATE objects SET confidence=0"})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    # INSERT
    resp = client.post("/query", json={"sql": "INSERT INTO objects VALUES (1)"})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    # ALTER TABLE (DDL)
    resp = client.post("/query", json={"sql": "ALTER TABLE objects ADD COLUMN x"})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False


def test_query_empty_or_null_input(tmp_path, monkeypatch) -> None:
    """Empty or null SQL/question payloads must return a clear 400 error."""
    db_path = str(tmp_path / "web_empty.db")
    _seed_db(db_path)
    monkeypatch.setenv("V2S_DB_PATH", db_path)

    from web.app import create_app
    client = create_app().test_client()

    # Empty JSON object
    resp = client.post("/query", json={})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    # Missing both sql and question keys
    resp = client.post("/query", json={"other": 1})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    # sql is empty string
    resp = client.post("/query", json={"sql": ""})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    # sql is None (explicit null)
    resp = client.post("/query", json={"sql": None})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    # question is empty string
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    # question is None
    resp = client.post("/query", json={"question": None})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    # Valid question still works to confirm endpoint is functional
    resp = client.post("/query", json={"question": "show me the events"})
    assert resp.status_code == 200
    assert resp.get_json()["rows"], "question path returned no event rows"


def test_query_both_sql_and_question(tmp_path, monkeypatch) -> None:
    """When both fields are present, sql takes precedence."""
    db_path = str(tmp_path / "web_both.db")
    _seed_db(db_path)
    monkeypatch.setenv("V2S_DB_PATH", db_path)

    from web.app import create_app
    client = create_app().test_client()

    # Both sql and question provided: sql should win
    resp = client.post("/query", json={
        "sql": "SELECT 1 AS source",
        "question": "some natural language text"
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    # Should have executed the sql, not the question
    assert any(row.get("source") == 1 for row in data["rows"]), (
        "Expected SQL result when both fields present"
    )


def test_query_sql_with_comments_and_newlines(tmp_path, monkeypatch) -> None:
    """SQL with -- comments and line breaks must still parse."""
    db_path = str(tmp_path / "web_comments.db")
    _seed_db(db_path)
    monkeypatch.setenv("V2S_DB_PATH", db_path)

    from web.app import create_app
    client = create_app().test_client()

    # Single-line comment with -- and newlines
    resp = client.post("/query", json={
        "sql": (
            "-- This is a comment\n"
            "SELECT class_label, confidence\n"
            "FROM objects\n"
            "-- WHERE confidence > 0.5\n"
            "ORDER BY confidence DESC"
        )
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["rows"], "SQL with comments returned no rows"

    # Block comment style /* ... */ across lines
    resp = client.post("/query", json={
        "sql": (
            "SELECT /* inline block comment */ class_label\n"
            "FROM   objects\n"
            "/* multi\n"
            "   line\n"
            "   comment */\n"
            "WHERE  class_label IS NOT NULL"
        )
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["rows"], "SQL with block comment returned no rows"

    # SQL with only comments and whitespace must be rejected
    resp = client.post("/query", json={
        "sql": "  -- just a comment\n/* another */\n"
    })
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["ok"] is False


# --- no-IP file scan --------------------------------------------------------

#: Strings that must never appear in the end-product surface: legacy project
#: names, provider/seat names, engine or model identifiers, and secret
#: material. Kept lowercase; files are scanned case-insensitively.
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "joye" "ly",  # client name, split so this file does not contain it
    "openrouter",
    "yolo",
    "rf-detr",
    "facedetector",
    "facerecognizer",
    "trackernano",
    "trackervit",
    "dasiamrpn",
    "twelve labs",
    "twelvelabs",
    "begin private key",
    "begin rsa private key",
    "aws_secret_" "access_key",
)

SCAN_SUFFIXES = {
    ".py", ".js", ".mjs", ".cjs", ".json", ".md", ".txt", ".html",
    ".css", ".yaml", ".yml", ".sql", ".csv",
}

#: Directories whose contents are the end-product surface under scan.
SCAN_DIRS = ("v2s", "web", "deploy", "data", "eval", "docs")
SCAN_ROOT_FILES = ("README.md", "requirements.txt")


def _iter_surface_files(base: Path = REPO_ROOT):
    for name in SCAN_ROOT_FILES:
        path = base / name
        if path.is_file():
            yield path
    for dirname in SCAN_DIRS:
        root = base / dirname
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SCAN_SUFFIXES:
                yield path


def _scan_for_forbidden(base: Path) -> tuple[int, list[str]]:
    offenders: list[str] = []
    scanned = 0
    for path in _iter_surface_files(base):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        lowered = text.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in lowered:
                offenders.append(f"{path.relative_to(base)}: {pattern!r}")
    return scanned, offenders


def test_no_ip_strings_in_surface_files() -> None:
    scanned, offenders = _scan_for_forbidden(REPO_ROOT)
    assert scanned > 0, "no surface files were scanned"
    assert not offenders, "forbidden strings found:\n" + "\n".join(offenders)


def test_no_ip_scan_catches_a_known_bad_file(tmp_path) -> None:
    """Negative control: the scanner must flag a file that contains a
    forbidden string, otherwise the clean result above means nothing."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("uses OpenRouter seats")
    (tmp_path / "README.md").write_text("clean file")
    scanned, offenders = _scan_for_forbidden(tmp_path)
    assert scanned == 2
    assert offenders == ["docs/notes.md: 'openrouter'"]


def test_no_secret_assignments_in_surface_files() -> None:
    # Improved regex that spans multiline and handles non-standard quotes:
    # - Captures variable-style assignments with optional spaces
    # - Matches single-quoted, double-quoted, and heredoc/backtick values
    # - Uses DOTALL to span lines; enforces minimum 8 characters for value
    # - Covers cases like: key = `multi\nline\nsecret`, key = '''value''', etc.
    secret_re = re.compile(
        r"(?im)(?:^|\s)(api[_-]?key|secret|password|token)\s*=\s*"
        r"(?:"
        r"""(?:'''|'''|["'])"""       # single/double quotes (single-line)
        r"[^'\"\n]{8,}"
        r"""(?:'''|'''|["'])"""
        r"|"                          # OR
        r"""`[^`]{8,}`"""            # backtick-quoted
        r")",
    )

    # Additional pattern for heredoc-style assignments (<<EOF ... EOF)
    heredoc_re = re.compile(
        r"(?im)(?:^|\s)(api[_-]?key|secret|password|token)\s*=\s*<<['\"]?(\w+)['\"]?\n"
        r"(.*?)\n\2",
        re.DOTALL,
    )

    offenders: list[str] = []
    for path in _iter_surface_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if secret_re.search(text) or heredoc_re.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, "possible hardcoded secrets in: " + ", ".join(offenders)
