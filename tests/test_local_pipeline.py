# build 7fd77e00de7f
"""End-to-end tests of the local OpenCV 5 pipeline on generated video.

No stubs stand in for the vision step here: a synthetic video is written to
disk and sampled/detected/tracked by real OpenCV 5 code, then the agent loop
writes the results into SQLite.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

cv2 = pytest.importorskip("cv2")

from v2s import agent, detect_track, sample, sqlwriter  # noqa: E402
from v2s.ingest import ClipWorkItem  # noqa: E402

FPS = 30
SIZE = (320, 240)


def _write(path: Path, frames: list[np.ndarray]) -> str:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
    return str(path)


def _square_frames(n: int, background: int, x0: int = 20, step: int = 4):
    frames = []
    for i in range(n):
        frame = np.full((SIZE[1], SIZE[0], 3), background, dtype=np.uint8)
        x = x0 + i * step
        cv2.rectangle(frame, (x, 100), (x + 30, 130), (255, 255, 255), -1)
        frames.append(frame)
    return frames


@pytest.fixture
def cut_video(tmp_path) -> str:
    """6 s: a square moving across a black scene, hard cut at 3.0 s to a grey
    scene where a square keeps moving."""
    frames = _square_frames(90, 0) + _square_frames(90, 140)
    return _write(tmp_path / "cut.mp4", frames)


@pytest.fixture
def static_video(tmp_path) -> str:
    frames = [np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)] * 90
    return _write(tmp_path / "static.mp4", frames)


def test_opencv_is_version_5() -> None:
    assert cv2.__version__.startswith("5"), cv2.__version__


def test_local_sampling_finds_the_scene_cut(cut_video) -> None:
    segments = sample.sample_video(cut_video)
    assert len(segments) == 2
    assert abs(segments[0]["end_s"] - 3.0) < 0.1
    assert abs(segments[1]["start_s"] - 3.0) < 0.1
    for seg in segments:
        assert 0.0 <= seg["motion_score"] <= 1.0


def test_local_sampling_static_video_has_no_segments(static_video) -> None:
    """Negative control: nothing moves and nothing cuts, so nothing is sampled."""
    assert sample.sample_video(static_video) == []


def test_local_sampling_rejects_missing_and_unreadable_input(tmp_path) -> None:
    with pytest.raises(sample.SampleServiceError):
        sample.sample_video(str(tmp_path / "does_not_exist.mp4"))
    with pytest.raises(sample.SampleServiceError):
        sample.sample_video("s3://bucket/clip.mp4")


def test_local_detection_tracks_the_moving_square(cut_video) -> None:
    rows = detect_track.process_video_segment(
        cut_video, {"start_s": 0.0, "end_s": 2.5}
    )
    assert rows, "expected detections for a moving square"
    assert all(0.0 <= r["timestamp_s"] <= 2.6 for r in rows)
    assert all(0.0 <= r["confidence"] <= 1.0 for r in rows)
    assert all(r["class_label"] == "moving_object" for r in rows)
    box = rows[len(rows) // 2]["bounding_box"]
    assert box["w"] > 10 and box["h"] > 10
    assert len({r["track_id"] for r in rows}) <= 3  # one object, tracked


def test_local_detection_static_video_has_no_detections(static_video) -> None:
    rows = detect_track.process_video_segment(
        static_video, {"start_s": 0.0, "end_s": 2.5}
    )
    assert rows == []


def test_local_detection_respects_the_segment_window(cut_video) -> None:
    rows = detect_track.process_video_segment(
        cut_video, {"start_s": 4.0, "end_s": 5.5}
    )
    assert rows
    assert all(3.9 <= r["timestamp_s"] <= 5.6 for r in rows)


def test_agent_loop_end_to_end_with_real_opencv(cut_video, tmp_path) -> None:
    db = str(tmp_path / "e2e.db")
    writer = sqlwriter.SqliteWriter(db)
    clip = ClipWorkItem(
        uri=cut_video,
        filename="cut.mp4",
        source_url="https://example.org/generated",
        license="CC0",
    )
    result = agent.process_clip(clip, writer)
    counts = result["counts"]
    assert counts["videos"] == 1
    assert counts["segments"] == 2
    assert counts["objects"] >= 1
    assert counts["events"] >= 1
    assert counts["counts"] >= 1
    with sqlite3.connect(db) as conn:
        (bad,) = conn.execute(
            "SELECT COUNT(*) FROM segments WHERE reason LIKE 'error:%'"
        ).fetchone()
        assert bad == 0, "no segment may be recorded as failed"
        classes = {
            r[0] for r in conn.execute("SELECT DISTINCT class_label FROM objects")
        }
        assert classes == {"moving_object"}


def test_remote_service_is_used_only_when_configured(cut_video, monkeypatch) -> None:
    calls: list[str] = []

    def fake_http(path: str, payload: dict) -> dict:
        calls.append(path)
        return {"segments": [{"start_s": 0.0, "end_s": 1.0, "motion_score": 0.1}]}

    monkeypatch.setattr(sample, "_http_transport", fake_http)

    # No endpoint configured: local OpenCV runs, remote is never touched.
    assert len(sample.sample_video(cut_video)) == 2
    assert calls == []

    # Endpoint configured: the remote path is used.
    monkeypatch.setenv("V2S_VISION_ENDPOINT", "https://vision.example.org")
    assert len(sample.sample_video(cut_video)) == 1
    assert calls == ["/sample"]

    # Forcing local mode wins over the endpoint.
    monkeypatch.setenv("V2S_VISION_MODE", "local")
    assert len(sample.sample_video(cut_video)) == 2
    assert calls == ["/sample"]


def test_network_guard_is_active(cut_video, monkeypatch) -> None:
    """Negative control: the real HTTP transport cannot reach the network."""
    monkeypatch.setenv("V2S_VISION_ENDPOINT", "http://127.0.0.1:9")
    with pytest.raises(Exception):
        sample.sample_video(cut_video)
