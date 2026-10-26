# build 7fd77e00de7f
"""3-compile agent loop produces rows in all tables using a local stub service.

Thin-wrapper posture: the external vision service and the private 3Dogs
service are replaced by deterministic local stubs, so no real network calls
occur (per the FORBIDDEN rule). The loop under test is the real
``v2s.agent.process_clip`` writing through the real
``v2s.sqlwriter.SqliteWriter`` into a temp SQLite database.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any
import tempfile
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from v2s import agent, sqlwriter  # noqa: E402
from v2s.ingest import ClipWorkItem  # noqa: E402

LABEL_DIR = REPO_ROOT / "eval" / "labels"


def _duration_s(labels: dict[str, Any]) -> float:
    if labels.get("duration_s") is not None:
        return float(labels["duration_s"])
    return float((labels.get("duration") or {}).get("fallback_value_s", 60.0))


def _segments_for(labels: dict[str, Any]) -> list[dict[str, Any]]:
    """Segments the sampler stub returns: the labelled ones, or - when a label
    file lists none - one segment spanning the whole clip."""
    segs = list(labels.get("segments") or [])
    if segs:
        return segs
    return [{"start_s": 0.0, "end_s": _duration_s(labels),
             "reason": "scene_or_motion", "motion_score": 0.5}]


class StubCrowdClient:
    """Local stub implementing the CrowdClient protocol from v2s.agent.

    Holds ONE clip's labels. It asserts that every segment the agent forwards
    carries a ``segment_id`` and matches a window the sampler stub produced,
    so a mismatched hand-off fails fast.
    """

    def __init__(self, labels: dict[str, Any]) -> None:
        self.labels = labels

    def crowd_read(
        self, video_uri: str, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        windows = {(float(s["start_s"]), float(s["end_s"]))
                   for s in _segments_for(self.labels)}
        assert segments, "agent forwarded no segments to the crowd read"
        for seg in segments:
            assert isinstance(seg.get("segment_id"), str) and seg["segment_id"]
            assert (float(seg["start_s"]), float(seg["end_s"])) in windows

        rows = [
            {
                "track_id": obj["track_id"],
                "class_label": obj["class_label"],
                "confidence": 0.8,
                "timestamp_s": float(obj["first_seen_s"]),
                "bounding_box": obj.get("bounding_box"),
            }
            for obj in self.labels.get("objects", [])
        ]
        return {"rows": rows}

    def reconcile(
        self,
        local_rows: list[dict[str, Any]],
        crowd_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        merged = {r.get("track_id"): dict(r) for r in local_rows}
        for row in crowd_rows:
            existing = merged.get(row.get("track_id"))
            if existing is None or row.get("confidence", 0.0) >= existing.get(
                "confidence", 0.0
            ):
                merged[row.get("track_id")] = dict(row)
        return {"rows": list(merged.values())}


def _stub_transports(labels: dict[str, Any]):
    def sample_transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert path == "/sample"
        # ------------------------------------------------------------------
        # Issue 101: validate payload structure expected by the real service
        assert "video_uri" in payload, (
            f"/sample payload missing required key 'video_uri': {sorted(payload.keys())}"
        )
        assert isinstance(payload["video_uri"], str), "video_uri must be a string"
        assert "video_duration_s" in payload, (
            f"/sample payload missing required key 'video_duration_s': {sorted(payload.keys())}"
        )
        assert isinstance(payload["video_duration_s"], (int, float)), (
            "video_duration_s must be numeric"
        )
        # ------------------------------------------------------------------
        return {"segments": _segments_for(labels)}

    def detect_transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert path == "/detect_track"
        assert "detector" in payload  # Audit Fix #4: detector always passed

        # ------------------------------------------------------------------
        # Issue 119: filter stub detections by the segment window so that
        # segment-scoped detection logic in the agent is properly exercised.
        seg_start = float(payload.get("segment_start_s", 0.0))
        seg_end = float(payload.get("segment_end_s", float("inf")))

        all_objects = labels.get("objects", [])
        windowed_objects = [
            obj
            for obj in all_objects
            if seg_start <= float(obj.get("first_seen_s", 0.0)) <= seg_end
        ]
        # ------------------------------------------------------------------

        detections = [
            {
                "track_id": obj["track_id"],
                "class_label": obj["class_label"],
                "confidence": 0.85,
                "timestamp_s": float(obj["first_seen_s"]),
                "bounding_box": obj.get("bounding_box"),
            }
            for obj in windowed_objects
        ]
        return {"detections": detections}

    return sample_transport, detect_transport


def _load_labels() -> dict[str, dict[str, Any]]:
    """Load ground-truth label files from eval/labels/.

    Raises
    ------
    FileNotFoundError
        If the label directory is missing (no JSON files found) –
        the mini-eval labels are required for the agent-loop test.
    AssertionError
        If a label file is unreadable, is not valid JSON, is missing
        required keys, or duplicates a ``clip`` key already seen.
    """
    json_paths = sorted(LABEL_DIR.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(
            f"No label files (*.json) found in {LABEL_DIR.resolve()}. "
            "Place exactly three mini-eval label files in eval/labels/ "
            "(see docs/REPORT.md and eval/labels/README.md)."
        )

    labels: dict[str, dict[str, Any]] = {}
    for path in json_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"Label file {path.name} is not valid JSON: {exc}"
            ) from exc

        clip_key = data.get("clip")
        if not clip_key:
            raise AssertionError(
                f"Label file {path.name} is missing the 'clip' key"
            )
        if clip_key in labels:
            raise AssertionError(
                f"Duplicate clip key '{clip_key}' in {path.name}; "
                "every label file must have a unique clip value"
            )
        labels[clip_key] = data

    return labels


def test_three_clip_loop_populates_tables(tmp_path) -> None:
    labels_by_clip = _load_labels()
    assert len(labels_by_clip) == 3, (
        f"mini-eval expects exactly 3 label files in {LABEL_DIR.resolve()}; "
        f"found {len(labels_by_clip)}"
    )

    db_path = str(tmp_path / "loop.db")
    writer = sqlwriter.SqliteWriter(db_path)

    try:
        for filename, labels in labels_by_clip.items():
            # ------------------------------------------------------------------
            # Issue 118: surface missing source metadata early
            source_url = labels.get("source_url")
            license_val = labels.get("license")
            assert source_url, (
                f"Label for clip '{filename}' is missing 'source_url'. "
                "Every label must provide a source_url (use 'NOT FOUND' only "
                "if the source is genuinely untraceable)."
            )
            assert license_val, (
                f"Label for clip '{filename}' is missing 'license'. "
                "Every label must provide a license (use 'NOT FOUND' only "
                "if the license is genuinely untraceable)."
            )
            # ------------------------------------------------------------------

            filename = Path(filename).name
            clip = ClipWorkItem(
                uri=f"stub://clips/{filename}",
                filename=filename,
                source_url=source_url,
                license=license_val,
            )
            sample_transport, detect_transport = _stub_transports(labels)
            result = agent.process_clip(
                clip,
                writer,
                crowd_client=StubCrowdClient(labels),
                sample_transport=sample_transport,
                detect_transport=detect_transport,
            )
            assert result["video_id"]
            counts = result["counts"]
            assert counts["videos"] == 1
            assert counts["segments"] == len(_segments_for(labels))
            assert counts["objects"] > 0
            assert counts["events"] > 0
            assert counts["counts"] > 0

            # ------------------------------------------------------------------
            # Issue 108: validate source_url and license are persisted correctly
            with sqlite3.connect(db_path) as conn:
                (stored_url, stored_license) = conn.execute(
                    "SELECT source_url, license FROM videos WHERE filename = ?",
                    (filename,),
                ).fetchone()
                assert stored_url == source_url, (
                    f"source_url mismatch for {filename}: "
                    f"stored={stored_url!r} expected={source_url!r}"
                )
                assert stored_license == license_val, (
                    f"license mismatch for {filename}: "
                    f"stored={stored_license!r} expected={license_val!r}"
                )
            # ------------------------------------------------------------------

        with sqlite3.connect(db_path) as conn:
            for table in sqlwriter.TABLES:
                # Verify the table exists in sqlite_master before counting.
                (exists,) = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                assert exists == 1, f"table {table} is missing from database"

                (n,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                if table in ("transcript", "agent_trace", "review_queue"):
                    # These three tables exist and are queryable, which is the
                    # contract this test enforces, but they are empty in this
                    # stub-transport loop: the demo loop emits no speech rows,
                    # and the agent layer (agent_trace, review_queue) runs only
                    # with the local detector (see tests/test_agent_policy.py).
                    assert n == 0, (
                        f"{table} should be empty after 3-clip loop "
                        f"(got {n} rows)"
                    )
                else:
                    assert n > 0, f"table {table} has no rows after 3-clip loop"

            # Confidence must be stored on every analytic row.
            for table in ("events", "objects", "counts"):
                (bad,) = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE confidence IS NULL"
                    " OR confidence < 0.0 OR confidence > 1.0"
                ).fetchone()
                assert bad == 0, f"table {table} has out-of-range confidence"

            # ------------------------------------------------------------------
            # Issue 113: Validate content correctness for events and counts
            # Get one of the clips to check specific values
            first_clip_labels = next(iter(labels_by_clip.values()))
            
            # Check events content
            event_rows = conn.execute(
                "SELECT start_timestamp_utc, end_timestamp_utc, event_type,"
                " confidence FROM events LIMIT 5"
            ).fetchall()
            assert len(event_rows) > 0, "No events found in database"
            for row in event_rows:
                t_start, t_end, event_type, confidence = row
                assert isinstance(t_start, str) and t_start.endswith("Z"), t_start
                assert isinstance(t_end, str) and t_end.endswith("Z"), t_end
                assert t_end >= t_start, f"t_end ({t_end}) should be >= t_start ({t_start})"
                assert isinstance(event_type, str), f"type should be string, got {type(event_type)}"
                assert 0.0 <= confidence <= 1.0, f"confidence should be between 0 and 1, got {confidence}"
            
            # Check counts content
            count_rows = conn.execute(
                "SELECT minute_index, class_label, count, confidence"
                " FROM counts LIMIT 5"
            ).fetchall()
            assert len(count_rows) > 0, "No counts found in database"
            for row in count_rows:
                minute, obj_class, count_val, confidence = row
                assert isinstance(minute, int), f"minute should be integer, got {type(minute)}"
                assert isinstance(obj_class, str), f"object_class should be string, got {type(obj_class)}"
                assert isinstance(count_val, int), f"count should be integer, got {type(count_val)}"
                assert count_val >= 0, f"count should be non-negative, got {count_val}"
                assert 0.0 <= confidence <= 1.0, f"confidence should be between 0 and 1, got {confidence}"
            # ------------------------------------------------------------------
    finally:
        # ------------------------------------------------------------------
        # Issue 115: close the SqliteWriter so its connection is explicitly
        # released before sqlite3.connect reads the same file.
        try:
            writer.close()
        except AttributeError:
            pass  # defensive – SqliteWriter may not have a .close() yet


def test_process_clip_defaults_do_not_raise(tmp_path) -> None:
    """process_clip must supply the detector argument that
    process_video_segment requires even when the caller does not pass one
    explicitly (Audit Fix #4). This test exercises the default-synthesis
    path by calling with a detect_transport whose ``detector`` assertion
    will fail if the default is not passed properly."""
    labels = next(iter(_load_labels().values()))
    clip = ClipWorkItem(
        uri="stub://clips/x.mp4",
        filename="x.mp4",
        source_url="NOT FOUND",
        license="NOT FOUND",
    )
    sample_transport, detect_transport = _stub_transports(labels)
    writer = sqlwriter.SqliteWriter(str(tmp_path / "defaults.db"))

    try:
        # Invoke process_clip with all optional arguments omitted except those
        # needed to route to stubs.  The real agent defaults inject a detector
        # string; the stub asserts that it is present so we confirm the path.
        result = agent.process_clip(
            clip,
            writer,
            crowd_client=StubCrowdClient(labels),
            sample_transport=sample_transport,
            detect_transport=detect_transport,
        )
        assert result["counts"]["videos"] == 1

        # ------------------------------------------------------------------
        # Issue 104: verify that the default detector path produces actual
        # rows (i.e. the detector string was passed and used, not silently
        # skipped).  An empty DB would indicate the stub path was never hit.
        with sqlite3.connect(str(tmp_path / "defaults.db")) as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM objects").fetchone()
            assert n > 0, (
                f"Expected at least one object row when the default detector is active; "
                f"got {n}.  This means the detect_transport stub was never called with a "
                f"valid detector, or the agent silently omitted detection."
            )
        # ------------------------------------------------------------------
    finally:
        try:
            writer.close()
        except AttributeError:
            pass


# ------------------------------------------------------------------
# Issue 112: Test for malformed JSON in label files
def test_malformed_json_handling(tmp_path) -> None:
    """Test that malformed JSON files are properly handled with clear errors."""
    # Create a temporary directory for our test labels
    test_label_dir = tmp_path / "test_labels"
    test_label_dir.mkdir()
    
    # Create a valid label file for comparison
    valid_label = {
        "clip": "valid.mp4",
        "source_url": "http://example.com/video.mp4",
        "license": "CC0",
        "segments": [],
        "objects": []
    }
    valid_file = test_label_dir / "valid.json"
    valid_file.write_text(json.dumps(valid_label), encoding="utf-8")
    
    # Create a malformed JSON file
    malformed_file = test_label_dir / "malformed.json"
    malformed_file.write_text('{"clip": "malformed.mp4", "source_url": "http://example.com/video.mp4", "license": "CC0", "segments": [], "objects": []', encoding="utf-8")
    
    # Temporarily change the LABEL_DIR to our test directory
    original_label_dir = LABEL_DIR
    globals()["LABEL_DIR"] = test_label_dir
    
    try:
        # This should raise an AssertionError due to malformed JSON
        try:
            _load_labels()
            assert False, "Expected AssertionError for malformed JSON"
        except AssertionError as e:
            assert "is not valid JSON" in str(e), f"Unexpected error message: {e}"
    finally:
        # Restore the original LABEL_DIR
        globals()["LABEL_DIR"] = original_label_dir
# ------------------------------------------------------------------
