#!/usr/bin/env python3
"""Synthetic ground-truth evaluation of the Video-to-SQL agent layer.

Usage:
    python eval/synth_eval.py [--out evidence/synth_eval.json] [--seed 7]

Generates small synthetic videos with KNOWN ground truth in a temporary
directory:

  * steady    - steady camera, 3 moving objects
  * jittery   - same scene with random per-frame camera jitter, 3 objects
  * crowd     - steady camera, 45 small (10 px) objects on a 9x5 grid, each
                moving slowly in its own direction inside its own grid cell
  * slow_stop - steady camera, 1 slow object that stops moving half-way

Each video is processed twice with ``v2s.agent.process_clip`` (agent=False and
agent=True) into temporary SQLite databases. The JSON report lists, per
scenario: truth_objects, tracks_agent_off, tracks_agent_on, camera_motion_px
(from the plan trace row), final_decisions (counts by decision), review_flags,
abs_error_off and abs_error_on.

These are synthetic videos with exact ground truth, NOT real footage. Failures
are reported, not hidden. No thresholds are asserted: the exit code is 0
whenever the script ran.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

NOTE = (
    "Synthetic videos with exact ground truth, NOT real footage; "
    "failures are reported, not hidden. No thresholds are asserted."
)

WIDTH = 320
HEIGHT = 240
FPS = 10
DURATION_S = 6
MARGIN = 8  # extra canvas border used to simulate camera jitter
RECORDING_START_UTC = "2026-01-01T00:00:00Z"


# --------------------------------------------------------------------------
# Synthetic video generation
# --------------------------------------------------------------------------

def _background(rng: np.random.RandomState, w: int, h: int) -> np.ndarray:
    noise = rng.randint(0, 256, size=(h, w, 3)).astype(np.uint8)
    bg = cv2.GaussianBlur(noise, (0, 0), 3)
    # stretch contrast so phase correlation has texture to lock on to
    bg = cv2.normalize(bg, None, 40, 200, cv2.NORM_MINMAX)
    return bg


def _random_objects(rng: np.random.RandomState, n: int, size: int, speed: float):
    """Objects moving freely over the whole visible area (bouncing at edges)."""
    lo_x, hi_x = MARGIN, WIDTH + MARGIN - size
    lo_y, hi_y = MARGIN, HEIGHT + MARGIN - size
    objs = []
    for _ in range(n):
        x = float(rng.uniform(lo_x + size, hi_x - size))
        y = float(rng.uniform(lo_y + size, hi_y - size))
        ang = float(rng.uniform(0, 2 * np.pi))
        color = tuple(int(c) for c in rng.randint(0, 256, size=3))
        objs.append({"x": x, "y": y, "vx": speed * np.cos(ang), "vy": speed * np.sin(ang),
                     "size": size, "color": color,
                     "bounds": (lo_x, hi_x, lo_y, hi_y)})
    return objs


def _grid_objects(rng: np.random.RandomState, cols: int, rows: int, size: int, speed: float):
    """One object per grid cell, each confined to its own cell so they never merge."""
    cell_w = WIDTH / cols
    cell_h = HEIGHT / rows
    pad = 3
    objs = []
    for r in range(rows):
        for c in range(cols):
            lo_x = MARGIN + c * cell_w + pad
            hi_x = MARGIN + (c + 1) * cell_w - pad - size
            lo_y = MARGIN + r * cell_h + pad
            hi_y = MARGIN + (r + 1) * cell_h - pad - size
            x = float(rng.uniform(lo_x, hi_x))
            y = float(rng.uniform(lo_y, hi_y))
            ang = float(rng.uniform(0, 2 * np.pi))
            # bright saturated colours so each square contrasts with the background
            color = tuple(int(v) for v in rng.choice([0, 255], size=3))
            if color in ((0, 0, 0), (255, 255, 255)):
                color = (255, 255, 0)
            objs.append({"x": x, "y": y, "vx": speed * np.cos(ang), "vy": speed * np.sin(ang),
                         "size": size, "color": color,
                         "bounds": (lo_x, hi_x, lo_y, hi_y)})
    return objs


def _open_writer(path_stem: Path) -> tuple[cv2.VideoWriter, Path]:
    for ext, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        path = path_stem.with_suffix(ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), FPS, (WIDTH, HEIGHT))
        if writer.isOpened():
            return writer, path
        writer.release()
    raise RuntimeError("cv2.VideoWriter could not open any codec (mp4v, MJPG)")


def generate_video(path_stem: Path, *, seed: int, n_objects: int, size: int,
                   speed: float, jitter_px: int = 0, stop_after_s: float | None = None,
                   grid: tuple[int, int] | None = None) -> Path:
    rng = np.random.RandomState(seed)
    cw, ch = WIDTH + 2 * MARGIN, HEIGHT + 2 * MARGIN
    bg = _background(rng, cw, ch)
    if grid is not None:
        objs = _grid_objects(rng, grid[0], grid[1], size, speed)
    else:
        objs = _random_objects(rng, n_objects, size, speed)
    writer, path = _open_writer(path_stem)
    try:
        for i in range(FPS * DURATION_S):
            t = i / FPS
            canvas = bg.copy()
            moving = stop_after_s is None or t < stop_after_s
            for o in objs:
                if moving:
                    o["x"] += o["vx"]
                    o["y"] += o["vy"]
                    lo_x, hi_x, lo_y, hi_y = o["bounds"]
                    if o["x"] < lo_x or o["x"] > hi_x:
                        o["vx"] = -o["vx"]
                        o["x"] = min(max(o["x"], lo_x), hi_x)
                    if o["y"] < lo_y or o["y"] > hi_y:
                        o["vy"] = -o["vy"]
                        o["y"] = min(max(o["y"], lo_y), hi_y)
                x0, y0 = int(round(o["x"])), int(round(o["y"]))
                cv2.rectangle(canvas, (x0, y0), (x0 + o["size"] - 1, y0 + o["size"] - 1),
                              o["color"], -1)
            if jitter_px:
                dx = int(rng.randint(-jitter_px, jitter_px + 1))
                dy = int(rng.randint(-jitter_px, jitter_px + 1))
            else:
                dx = dy = 0
            frame = canvas[MARGIN + dy:MARGIN + dy + HEIGHT, MARGIN + dx:MARGIN + dx + WIDTH]
            writer.write(np.ascontiguousarray(frame))
    finally:
        writer.release()
    return path


SCENARIOS = [
    # name, truth_objects, generator kwargs
    ("steady", 3, dict(n_objects=3, size=24, speed=3.0)),
    ("jittery", 3, dict(n_objects=3, size=24, speed=3.0, jitter_px=3)),
    ("crowd", 45, dict(n_objects=45, size=10, speed=0.7, grid=(9, 5))),
    ("slow_stop", 1, dict(n_objects=1, size=24, speed=1.0, stop_after_s=DURATION_S / 2)),
]


# --------------------------------------------------------------------------
# Pipeline glue
# --------------------------------------------------------------------------

def _build_clip(video_path: Path):
    from v2s.ingest import ClipWorkItem
    clip = ClipWorkItem(
        uri=str(video_path),
        filename=video_path.name,
        source_url="synthetic://v2s/eval/synth_eval.py",
        license="CC0 (synthetic, generated by eval/synth_eval.py)",
    )
    # process_clip reads recording_start_utc via getattr; ClipWorkItem has no field for it.
    clip.recording_start_utc = RECORDING_START_UTC
    return clip


def _make_writer(db_path: Path):
    from v2s import sqlwriter
    writer = sqlwriter.SqliteWriter(str(db_path))
    create = getattr(writer, "create_schema", None)
    if callable(create):
        try:
            create()
        except Exception:  # schema may already be created by the constructor
            pass
    return writer


def _run_process_clip(video_path: Path, db_path: Path, agent_on: bool) -> dict:
    from v2s import agent as agent_mod
    clip = _build_clip(video_path)
    writer = _make_writer(db_path)
    try:
        result = agent_mod.process_clip(clip, writer, agent=agent_on)
    finally:
        close = getattr(writer, "close", None)
        if callable(close):
            close()
    return dict(result) if isinstance(result, dict) else {}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _count_tracks(db_path: Path) -> int | None:
    conn = sqlite3.connect(str(db_path))
    try:
        if "objects" not in _tables(conn):
            return None
        return int(conn.execute("SELECT COUNT(DISTINCT track_id) FROM objects").fetchone()[0])
    finally:
        conn.close()


def _agent_summary(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        tables = _tables(conn)
        motion: list[float] = []
        final: dict[str, int] = {}
        peaks: list[int] = []
        flags = 0
        if "agent_trace" in tables:
            for (obs,) in conn.execute(
                    "SELECT observation FROM agent_trace WHERE step='plan' ORDER BY rowid"):
                try:
                    motion.append(float(json.loads(obs).get("camera_motion_px", 0.0)))
                except (ValueError, TypeError, AttributeError):
                    pass
            last_by_segment: dict[str, tuple[str, str]] = {}
            for seg, dec, obs in conn.execute(
                    "SELECT segment_id, decision, observation FROM agent_trace "
                    "WHERE step='review' ORDER BY rowid"):
                last_by_segment[seg] = (dec, obs)
            for dec, obs in last_by_segment.values():
                final[dec] = final.get(dec, 0) + 1
                try:
                    peaks.append(int(json.loads(obs).get("peak_simultaneous", 0)))
                except (ValueError, TypeError, AttributeError):
                    pass
        if "review_queue" in tables:
            flags = int(conn.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0])
    finally:
        conn.close()
    camera_motion = round(float(np.median(motion)), 4) if motion else None
    return {
        "camera_motion_px": camera_motion,
        "camera_motion_px_per_segment": [round(m, 4) for m in motion],
        "final_decisions": final,
        "peak_simultaneous_max": max(peaks) if peaks else None,
        "review_flags": flags,
    }


def run_scenario(tmp: Path, name: str, truth: int, gen_kwargs: dict, seed: int) -> dict:
    result: dict[str, Any] = {"scenario": name, "truth_objects": truth, "errors": []}
    try:
        video = generate_video(tmp / name, seed=seed, **gen_kwargs)
    except Exception as exc:
        result["errors"].append(f"video generation failed: {exc!r}")
        result.update(tracks_agent_off=None, tracks_agent_on=None,
                      abs_error_off=None, abs_error_on=None,
                      camera_motion_px=None, final_decisions={}, review_flags=0)
        return result

    for label, agent_on in (("off", False), ("on", True)):
        db = tmp / f"{name}_agent_{label}.sqlite"
        t0 = time.monotonic()
        try:
            out = _run_process_clip(video, db, agent_on)
            tracks = _count_tracks(db)
            result[f"tracks_agent_{label}"] = tracks
            result[f"abs_error_{label}"] = abs(tracks - truth) if tracks is not None else None
            result[f"table_counts_agent_{label}"] = out.get("counts", {})
            if agent_on:
                result.update(_agent_summary(db))
        except Exception as exc:
            result[f"tracks_agent_{label}"] = None
            result[f"abs_error_{label}"] = None
            result["errors"].append(
                f"process_clip agent={agent_on} failed: {exc!r}\n{traceback.format_exc(limit=3)}")
        result[f"runtime_s_agent_{label}"] = round(time.monotonic() - t0, 2)
    result.setdefault("camera_motion_px", None)
    result.setdefault("final_decisions", {})
    result.setdefault("review_flags", 0)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="evidence/synth_eval.json", help="output JSON path")
    parser.add_argument("--seed", type=int, default=7, help="random seed for video generation")
    args = parser.parse_args(argv)

    env_agent = os.environ.get("V2S_AGENT")
    report: dict[str, Any] = {
        "note": NOTE,
        "seed": args.seed,
        "opencv_version": cv2.__version__,
        "video": {"width": WIDTH, "height": HEIGHT, "fps": FPS, "duration_s": DURATION_S},
        "env_V2S_AGENT": env_agent,
        "scenarios": [],
    }
    if env_agent is not None and env_agent.strip().lower() == "off":
        report["warning"] = "V2S_AGENT=off is set: the agent=True runs are also OFF."

    t_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="v2s_synth_") as tmpdir:
        tmp = Path(tmpdir)
        for i, (name, truth, gen_kwargs) in enumerate(SCENARIOS):
            res = run_scenario(tmp, name, truth, gen_kwargs, args.seed + i)
            report["scenarios"].append(res)
            print(f"{name}: truth={truth} off={res.get('tracks_agent_off')} "
                  f"on={res.get('tracks_agent_on')} motion={res.get('camera_motion_px')} "
                  f"decisions={res.get('final_decisions')} flags={res.get('review_flags')}"
                  + (f" errors={len(res['errors'])}" if res["errors"] else ""))
    report["runtime_s"] = round(time.monotonic() - t_start, 2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
