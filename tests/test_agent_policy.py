# tests/test_agent_policy.py
"""Agentic-vision tests: an OpenCV 5 observation (camera motion, track
statistics) changes what the agent does next, and every step lands in SQL.

All videos are tiny synthetic MJPG clips generated in ``tmp_path`` with a
fixed ``numpy.random.RandomState``.  Detection, tracking and camera-motion
measurement are the real local OpenCV code.  Sampling is given one segment
covering the whole clip through ``sample_transport``, so the agent layer
(which runs only for the local detector) is always exercised.  No network.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

cv2 = pytest.importorskip("cv2")

from v2s import agent, detect_track, policy, sqlwriter  # noqa: E402

FPS = 10
SECONDS = 4
SMALL = (160, 120)
CROWD_SIZE = (320, 240)

# Crowd clips are drawn on a plain background for these first frames, then
# the squares appear and start moving.
CROWD_WARMUP_FRAMES = 5
CROWD_ROWS = 9

# Crowd clips tried in order: (frame size, number of squares, square side px).
# The first rung matches the brief (45 squares of about 6 px at 160x120).
# The later rungs are larger, up to 320x240, in case the detector's minimum
# blob area hides tiny squares.  In every rung the squares sit in sparse lanes
# (alternating left/right), so a pixel is crossed at most once during the
# first two seconds and the background model never learns the crowd.
CROWD_LADDER = [
    (SMALL, 45, 6),
    (CROWD_SIZE, 45, 10),
    (CROWD_SIZE, 45, 14),
    (CROWD_SIZE, 45, 18),
    (CROWD_SIZE, 45, 22),
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class _Clip:
    """Minimal clip object with the attributes process_clip requires."""

    def __init__(self, path: Path) -> None:
        self.uri = str(path)
        self.filename = Path(path).name
        self.source_url = "https://example.org/generated"
        self.license = "CC0"
        self.recording_start_utc = "2026-01-01T00:00:00Z"


def _background(rs: np.random.RandomState, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    bg = rs.randint(0, 120, (h, w)).astype(np.uint8)
    return cv2.GaussianBlur(bg, (3, 3), 0)


def _pair_objects(size: tuple[int, int]):
    """Two large rectangles crossing the frame once, in opposite directions."""
    w, _ = size
    return [
        # x0, y0, vx, vy, width, height
        (5, 20, 3, 0, 30, 26),
        (w - 35, 75, -3, 0, 30, 26),
    ]


def _crowd_objects(size: tuple[int, int], n_objects: int, obj: int):
    """Small squares in sparse horizontal lanes, alternating directions.

    Squares inside one lane share a speed, so they never merge, and the gap
    between them exceeds the distance travelled in the first 20 frames, so no
    pixel sees two squares in that window.  Squares never wrap around.
    """
    w, h = size
    cols = int(np.ceil(n_objects / CROWD_ROWS))
    spacing = w // cols
    row_gap = h // CROWD_ROWS
    speed = max(1, int(round(w / 160)))
    objs = []
    for r in range(CROWD_ROWS):
        y = r * row_gap + max((row_gap - obj) // 2, 0)
        direction = 1 if r % 2 == 0 else -1
        for k in range(cols):
            if len(objs) >= n_objects:
                break
            if direction > 0:
                x = 2 + k * spacing
            else:
                x = w - 2 - obj - k * spacing
            objs.append((x, y, direction * speed, 0, obj, obj))
    return objs


def _make_clip(
    path: Path,
    jitter: int = 0,
    n_objects: int = 2,
    size: tuple[int, int] = SMALL,
    seconds: int = SECONDS,
    obj_size: int = 8,
    seed: int = 7,
) -> Path:
    rs = np.random.RandomState(seed)
    w, h = size
    bg = _background(rs, size)
    if n_objects <= 2:
        objects = _pair_objects(size)
        start_frame = 0
    else:
        objects = _crowd_objects(size, n_objects, obj_size)
        start_frame = CROWD_WARMUP_FRAMES
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, size)
    assert writer.isOpened(), "cv2.VideoWriter could not open an MJPG .avi"
    for i in range(seconds * FPS):
        frame = bg.copy()
        if i >= start_frame:
            t = i - start_frame
            for x0, y0, vx, vy, ow, oh in objects:
                x = int(round(x0 + vx * t))
                y = int(round(y0 + vy * t))
                cv2.rectangle(frame, (x, y), (x + ow - 1, y + oh - 1), 255, -1)
        if jitter:
            dx, dy = rs.randint(-jitter, jitter + 1, 2)
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


def _end_s(seconds: int = SECONDS) -> float:
    return round((seconds * FPS - 1) / FPS, 3)


def _whole_clip_transport(seconds: int = SECONDS):
    end_s = _end_s(seconds)

    def transport(path: str, payload: dict) -> dict:
        return {
            "segments": [
                {"start_s": 0.0, "end_s": end_s, "reason": "scene_or_motion", "motion_score": 0.5}
            ]
        }

    return transport


def _run(path: Path, tmp_path: Path, db_name: str = "out.db", seconds: int = SECONDS, **kw):
    db = tmp_path / db_name
    writer = sqlwriter.SqliteWriter(str(db))
    result = agent.process_clip(
        _Clip(path), writer, sample_transport=_whole_clip_transport(seconds), **kw
    )
    return result, db


def _rows(db: Path, sql: str) -> list[tuple]:
    with sqlite3.connect(str(db)) as conn:
        try:
            return conn.execute(sql).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return []
            raise


def _plan_rows(db: Path) -> list[tuple[str, dict]]:
    return [
        (decision, json.loads(obs))
        for decision, obs in _rows(
            db, "SELECT decision, observation FROM agent_trace WHERE step = 'plan'"
        )
    ]


def _track_stats(path: Path, seconds: int = SECONDS) -> tuple[int, int]:
    """(peak simultaneous tracks, total tracks) from the real local detector."""
    end_s = _end_s(seconds)
    rows = detect_track.process_video_segment(str(path), {"start_s": 0.0, "end_s": end_s})
    rows = rows if isinstance(rows, list) else []
    obs = policy.observe(rows, 0.0, end_s)
    return obs.peak_simultaneous, obs.n_tracks


@pytest.fixture(autouse=True)
def _agent_env_clean(monkeypatch):
    monkeypatch.delenv("V2S_AGENT", raising=False)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_steady_camera_plans_default_and_needs_no_review(tmp_path) -> None:
    clip = _make_clip(tmp_path / "steady.avi", jitter=0, n_objects=2)
    result, db = _run(clip, tmp_path)

    plans = _plan_rows(db)
    assert plans, "agent wrote no 'plan' row"
    for decision, obs in plans:
        assert decision == "default", (decision, obs)
        assert obs["camera_motion_px"] < policy.CAMERA_MOTION_STABILIZE_PX, obs
    assert _rows(db, "SELECT * FROM review_queue") == []
    assert result["counts"]["review_queue"] == 0
    assert result["counts"]["agent_trace"] >= 2  # plan + review


def test_jittery_camera_plans_stabilize(tmp_path) -> None:
    clip = _make_clip(tmp_path / "jitter.avi", jitter=10, n_objects=2)
    _, db = _run(clip, tmp_path)

    plans = _plan_rows(db)
    assert plans, "agent wrote no 'plan' row"
    stabilized = [obs for decision, obs in plans if decision == "stabilize"]
    assert stabilized, plans
    assert any(
        obs["camera_motion_px"] >= policy.CAMERA_MOTION_STABILIZE_PX for obs in stabilized
    ), stabilized


def test_agent_off_writes_no_trace_rows(tmp_path, monkeypatch) -> None:
    clip = _make_clip(tmp_path / "jitter.avi", jitter=10, n_objects=2)

    result, db = _run(clip, tmp_path, db_name="off_param.db", agent=False)
    assert _rows(db, "SELECT * FROM agent_trace") == []
    assert _rows(db, "SELECT * FROM review_queue") == []
    assert result["counts"]["agent_trace"] == 0
    assert result["counts"]["review_queue"] == 0
    assert result["counts"]["segments"] >= 1  # the pipeline itself still ran

    monkeypatch.setenv("V2S_AGENT", "off")
    result2, db2 = _run(clip, tmp_path, db_name="off_env.db")
    assert _rows(db2, "SELECT * FROM agent_trace") == []
    assert _rows(db2, "SELECT * FROM review_queue") == []
    assert result2["counts"]["agent_trace"] == 0
    assert result2["counts"]["review_queue"] == 0


def test_crowd_clip_is_flagged_for_human_review(tmp_path) -> None:
    # Pick the first crowd clip on which the real detector reaches the
    # policy's dense-scene threshold; fail loudly (with the measured peaks)
    # if none of them does.
    tried = []
    chosen = None
    for idx, (size, n_objects, obj_size) in enumerate(CROWD_LADDER):
        path = _make_clip(
            tmp_path / f"crowd_{idx}.avi",
            jitter=0,
            n_objects=n_objects,
            size=size,
            obj_size=obj_size,
            seed=11,
        )
        peak, n_tracks = _track_stats(path)
        tried.append(
            {"size": size, "squares": n_objects, "side": obj_size,
             "peak": peak, "tracks": n_tracks}
        )
        if peak >= policy.PEAK_REVIEW_TRACKS:
            chosen = path
            break
    assert chosen is not None, (
        "no crowd clip reached PEAK_REVIEW_TRACKS simultaneous tracks", tried
    )

    result, db = _run(chosen, tmp_path, db_name="crowd.db")
    trace = _rows(db, "SELECT step, decision, observation FROM agent_trace")

    reviews = _rows(db, "SELECT status, reason FROM review_queue")
    assert any(status == "pending" for status, _ in reviews), (reviews, trace, tried)
    flags = [row for row in trace if row[1] == "flag_review"]
    assert len(flags) >= 1, trace
    assert result["counts"]["review_queue"] >= 1


def test_policy_decision_changes_the_detector_call(tmp_path, monkeypatch) -> None:
    clip = _make_clip(tmp_path / "jitter.avi", jitter=10, n_objects=2)
    real = detect_track.process_video_segment
    calls: list[dict] = []

    def recording(*args, **kwargs):
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(detect_track, "process_video_segment", recording)

    _, db_on = _run(clip, tmp_path, db_name="on.db", agent=True)
    on_calls = list(calls)
    assert on_calls, "detector was never called with the agent on"
    assert any(c.get("stabilize") is True for c in on_calls), on_calls
    # The camera-motion observation made the plan choose stabilisation ...
    plans = _plan_rows(db_on)
    assert [d for d, _ in plans] == ["stabilize"], plans
    assert plans[0][1]["camera_motion_px"] >= policy.CAMERA_MOTION_STABILIZE_PX, plans
    # ... and that plan, not a retry, is what the first (planned) detector call
    # received: one segment, so the first call is the planned one.
    assert on_calls[0].get("stabilize") is True, on_calls
    assert on_calls[0].get("var_threshold") == policy.VAR_THRESHOLD_STABILIZED, on_calls

    calls.clear()
    _run(clip, tmp_path, db_name="off.db", agent=False)
    off_calls = list(calls)
    assert off_calls, "detector was never called with the agent off"
    assert all("stabilize" not in c for c in off_calls), off_calls
