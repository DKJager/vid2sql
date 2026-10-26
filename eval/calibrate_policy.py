#!/usr/bin/env python3
"""Calibrate the agent's camera-motion stabilisation threshold on SYNTHETIC videos.

Usage:
    python eval/calibrate_policy.py [--out evidence/calibration.json] [--seeds 3] [--quick]

What it does
------------
Scenarios:
  * Background in {smooth_noise, facade} x camera jitter amplitude
    j in {0, 1, 2, 3, 4, 6, 10} px x seeds 1..N.
  * Each video is 320x240, 10 fps, 6 s, MJPG .avi in a temporary directory.
  * Three 24x24 rectangles cross the frame (two bright, one dark), so the
    exact ground truth is 3 objects.
  * Jitter is a uniform integer shift in [-j, j] on both axes per frame
    (cv2.warpAffine with BORDER_REPLICATE); j = 0 is a steady camera.
  * The generator is the one from diag_calib_example.py.  The facade
    background is deterministic, so for facade the seeds only change the
    jitter sequence.

Each scenario is processed twice with v2s.agent.process_clip, each into its
own temporary SQLite database.  The policy is NOT modified.
  * agent=False -> objects_off, the plain detector.
  * agent=True  -> objects_on, the real agent with the current policy.
    camera_motion_px is the mean of the 'plan' trace rows over segments.
    plan_decision is the real 'plan' decision.

Threshold study (no further runs): for each candidate t the policy
stabilises iff camera_motion_px >= t.  The policy error is abs_error_on if
stabilised, else abs_error_off.  The best t has the lowest mean error; on a
tie the LARGEST t wins, because stabilisation costs time.

Caveats, reported rather than hidden:
  * If the agent-on run did not stabilise (measured motion below the current
    threshold), abs_error_on for that scenario is an UNSTABILISED agent
    result.  For each t the study therefore reports
    n_counterfactual_stabilised: the number of scenarios the study counts as
    stabilised although the actual run was not.
  * The agent-off run does not include the agent's review/retry step.

These are synthetic videos with exact ground truth, NOT real footage.  The
script reports what the tracker does.  Scenarios are not tuned in favour of
the agent.  Exit code is 0 whenever the script ran.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from v2s import agent as agent_mod  # noqa: E402
from v2s import policy, sqlwriter  # noqa: E402
from v2s.ingest import ClipWorkItem  # noqa: E402

NOTE = (
    "Synthetic videos with exact ground truth (3 objects per clip), NOT real "
    "footage. Both runs use the unmodified pipeline and policy; camera motion "
    "and the plan decision come from the agent's own 'plan' trace rows. "
    "Results are what the tracker does; scenarios are not tuned in favour of "
    "the agent. Where the agent-on run did not stabilise, abs_error_on is an "
    "unstabilised result; the threshold study counts such cases per threshold "
    "(n_counterfactual_stabilised). The agent-off run does not model the "
    "agent's review/retry step."
)

WIDTH = 320
HEIGHT = 240
FPS = 10
DURATION_S = 6
TRUTH_OBJECTS = 3
OBJ_SIZE = 24

BACKGROUNDS_FULL = ["smooth_noise", "facade"]
JITTERS_FULL = [0, 1, 2, 3, 4, 6, 10]
JITTERS_QUICK = [0, 4, 10]
QUICK_BACKGROUND = "facade"
THRESHOLDS = [0.5, 0.8, 1.5, 2.5, 4.0, 6.0, 8.0, 12.0]
CURRENT_THRESHOLD = float(policy.CAMERA_MOTION_STABILIZE_PX)
MAX_WARNINGS_KEPT = 10


# --------------------------------------------------------------------------
# Synthetic video generation (generator of diag_calib_example.py)
# --------------------------------------------------------------------------

def facade_background(w: int, h: int, rng: np.random.RandomState) -> np.ndarray:
    """Regular window grid; deterministic (rng unused)."""
    f = np.full((h, w), 120, np.uint8)
    for y in range(8, h, 28):
        for x in range(8, w, 26):
            cv2.rectangle(f, (x, y), (x + 14, y + 18), 40, -1)
            cv2.rectangle(f, (x + 14, y), (x + 20, y + 18), 200, -1)
    return f


def smooth_noise_background(w: int, h: int, rng: np.random.RandomState) -> np.ndarray:
    """Blurred random noise, contrast stretched to 40..200 (seeded)."""
    noise = rng.randint(0, 256, size=(h, w)).astype(np.uint8)
    bg = cv2.GaussianBlur(noise, (0, 0), 3)
    bg = cv2.normalize(bg, None, 40, 200, cv2.NORM_MINMAX)
    return bg.astype(np.uint8)


BACKGROUND_FUNCS = {
    "smooth_noise": smooth_noise_background,
    "facade": facade_background,
}


def make_video(path: Path, background: str, jitter: int, seed: int) -> Path:
    bg_rng = np.random.RandomState(1000 * seed + 17)
    jit_rng = np.random.RandomState(seed)
    bg = BACKGROUND_FUNCS[background](WIDTH, HEIGHT, bg_rng)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    if not vw.isOpened():
        vw.release()
        raise RuntimeError("cv2.VideoWriter could not open MJPG .avi for writing")
    try:
        for i in range(FPS * DURATION_S):
            f = bg.copy()
            for k in range(TRUTH_OBJECTS):
                x = 20 + k * 90 + int(1.5 * i)
                y = 40 + k * 50
                # 24x24 filled square (cv2 corners are inclusive); k=0,2 bright, k=1 dark
                cv2.rectangle(f, (x, y), (x + OBJ_SIZE - 1, y + OBJ_SIZE - 1),
                              255 if k % 2 == 0 else 0, -1)
            if jitter:
                dx, dy = jit_rng.randint(-jitter, jitter + 1, 2)
                m = np.float32([[1, 0, dx], [0, 1, dy]])
                f = cv2.warpAffine(f, m, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REPLICATE)
            vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    finally:
        vw.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"video file {path} was not written")
    return path


# --------------------------------------------------------------------------
# Pipeline glue
# --------------------------------------------------------------------------

class _WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            msg = str(record.msg)
        self.messages.append(f"{record.name}: {msg}")


def run_pipeline(video: Path, db: Path, agent_on: bool) -> list[str]:
    """Run the unmodified process_clip once; return collected warning messages."""
    collector = _WarningCollector()
    root = logging.getLogger()
    root.addHandler(collector)
    writer = sqlwriter.SqliteWriter(str(db))
    try:
        clip = ClipWorkItem(uri=str(video), filename=video.name,
                            source_url="synthetic", license="CC0-1.0")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            agent_mod.process_clip(clip, writer, agent=agent_on)
        msgs = collector.messages + [f"warnings: {w.message}" for w in caught]
    finally:
        root.removeHandler(collector)
        close = getattr(writer, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
    return msgs


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def count_objects(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        if "objects" not in _tables(conn):
            return 0  # no object rows were written
        return int(conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0])
    finally:
        conn.close()


def agent_summary(db: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db))
    motions: list[float] = []
    decisions: list[str] = []
    steps: dict[str, int] = {}
    flags = 0
    try:
        tables = _tables(conn)
        if "agent_trace" in tables:
            for obs, dec in conn.execute(
                    "SELECT observation, decision FROM agent_trace "
                    "WHERE step='plan' ORDER BY rowid"):
                try:
                    motions.append(float(json.loads(obs).get("camera_motion_px")))
                except (ValueError, TypeError, AttributeError):
                    pass
                decisions.append(str(dec))
            for step, dec, n in conn.execute(
                    "SELECT step, decision, COUNT(*) FROM agent_trace GROUP BY 1, 2"):
                steps[f"{step}:{dec}"] = int(n)
        if "review_queue" in tables:
            flags = int(conn.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0])
    finally:
        conn.close()
    if decisions:
        plan_decision = decisions[0] if len(set(decisions)) == 1 else "mixed"
    else:
        plan_decision = None
    return {
        "camera_motion_px": round(float(np.mean(motions)), 4) if motions else None,
        "camera_motion_px_per_segment": [round(m, 4) for m in motions],
        "plan_decision": plan_decision,
        "plan_decisions_per_segment": decisions,
        # True only if every segment of the agent-on run was planned with stabilisation
        "on_run_stabilised": bool(decisions) and all(d == "stabilize" for d in decisions),
        "agent_steps": steps,
        "review_flags": flags,
    }


def run_scenario(tmp: Path, background: str, jitter: int, seed: int) -> dict[str, Any]:
    name = f"{background}_j{jitter}_s{seed}"
    res: dict[str, Any] = {
        "scenario": name, "background": background, "jitter_px": jitter, "seed": seed,
        "truth_objects": TRUTH_OBJECTS, "objects_off": None, "objects_on": None,
        "abs_error_off": None, "abs_error_on": None, "camera_motion_px": None,
        "camera_motion_px_per_segment": [], "plan_decision": None,
        "plan_decisions_per_segment": [], "on_run_stabilised": False,
        "agent_steps": {}, "review_flags": 0, "pipeline_warnings": [], "errors": [],
    }
    sdir = tmp / name
    sdir.mkdir(parents=True, exist_ok=True)
    try:
        try:
            video = make_video(sdir / f"{name}.avi", background, jitter, seed)
        except Exception as exc:  # noqa: BLE001
            res["errors"].append(f"video generation failed: {exc!r}")
            return res
        for label, agent_on in (("off", False), ("on", True)):
            db = sdir / f"agent_{label}.sqlite"
            t0 = time.monotonic()
            try:
                msgs = run_pipeline(video, db, agent_on)
                res["pipeline_warnings"].extend(f"[{label}] {m}" for m in msgs)
                n = count_objects(db)
                res[f"objects_{label}"] = n
                res[f"abs_error_{label}"] = abs(n - TRUTH_OBJECTS)
                if agent_on:
                    res.update(agent_summary(db))
            except Exception as exc:  # noqa: BLE001
                res["errors"].append(
                    f"process_clip agent={agent_on} failed: {exc!r}\n"
                    f"{traceback.format_exc(limit=3)}")
            res[f"runtime_s_{label}"] = round(time.monotonic() - t0, 2)
        res["pipeline_warning_count"] = len(res["pipeline_warnings"])
        res["pipeline_warnings"] = res["pipeline_warnings"][:MAX_WARNINGS_KEPT]
    finally:
        shutil.rmtree(sdir, ignore_errors=True)
    return res


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def _mean(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _valid(s: dict[str, Any]) -> bool:
    return (s.get("abs_error_off") is not None and s.get("abs_error_on") is not None
            and s.get("camera_motion_px") is not None)


def policy_error(s: dict[str, Any], t: float) -> int:
    return s["abs_error_on"] if s["camera_motion_px"] >= t else s["abs_error_off"]


def by_jitter(scenarios: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for s in scenarios:
        groups.setdefault((s["background"], s["jitter_px"]), []).append(s)
    out: dict[str, dict[str, Any]] = {}
    for (bg, j), items in sorted(groups.items()):
        out[f"{bg}/{j}"] = {
            "mean_error_off": _mean([s["abs_error_off"] for s in items]),
            "mean_error_on": _mean([s["abs_error_on"] for s in items]),
            "mean_camera_motion_px": _mean([s["camera_motion_px"] for s in items]),
            "mean_objects_off": _mean([s["objects_off"] for s in items]),
            "mean_objects_on": _mean([s["objects_on"] for s in items]),
            "on_run_stabilised_fraction": _mean(
                [1.0 if s.get("on_run_stabilised") else 0.0 for s in items]),
            "n": len(items),
            "n_valid": sum(1 for s in items if _valid(s)),
        }
    return out


def threshold_study(valid: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float | None]:
    study: list[dict[str, Any]] = []
    if not valid:
        return study, None
    for t in THRESHOLDS:
        errs = [policy_error(s, t) for s in valid]
        stab = [s for s in valid if s["camera_motion_px"] >= t]
        study.append({
            "threshold": t,
            "mean_policy_error": round(sum(errs) / len(errs), 4),
            "stabilised_fraction": round(len(stab) / len(valid), 4),
            # scenarios counted as stabilised whose actual agent-on run did not stabilise
            "n_counterfactual_stabilised": sum(
                1 for s in stab if not s.get("on_run_stabilised")),
        })
    # lowest mean error; tie -> largest threshold
    best = min(study, key=lambda r: (round(r["mean_policy_error"], 9), -r["threshold"]))
    return study, best["threshold"]


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_tables(report: dict[str, Any]) -> None:
    print("## Error by background / jitter (synthetic, truth = 3 objects)\n")
    print("| background/jitter px | mean camera motion px | agent-on stabilised fraction | "
          "mean objects off | mean objects on | mean error off | mean error on | n valid |")
    print("|---|---|---|---|---|---|---|---|")
    for key, row in report["by_jitter"].items():
        print(f"| {key} | {_fmt(row['mean_camera_motion_px'])} | "
              f"{_fmt(row['on_run_stabilised_fraction'])} | {_fmt(row['mean_objects_off'])} | "
              f"{_fmt(row['mean_objects_on'])} | {_fmt(row['mean_error_off'])} | "
              f"{_fmt(row['mean_error_on'])} | {row['n_valid']}/{row['n']} |")
    print("\n## Threshold study (stabilise iff camera_motion_px >= t)\n")
    print("| threshold t | mean policy error | stabilised fraction | "
          "counterfactual (run not stabilised) |")
    print("|---|---|---|---|")
    for row in report["threshold_study"]:
        mark = " (current)" if abs(row["threshold"] - CURRENT_THRESHOLD) < 1e-9 else ""
        best = " **best**" if row["threshold"] == report["best_threshold"] else ""
        print(f"| {row['threshold']}{mark}{best} | {_fmt(row['mean_policy_error'])} | "
              f"{_fmt(row['stabilised_fraction'])} | {row['n_counterfactual_stabilised']} |")
    print(f"\nnever stabilise: mean error {_fmt(report['mean_error_never'])}; "
          f"always stabilise: mean error {_fmt(report['mean_error_always'])}; "
          f"current threshold {CURRENT_THRESHOLD}: mean error "
          f"{_fmt(report['current_threshold_error'])}; valid scenarios "
          f"{report['n_valid_scenarios']}/{len(report['scenarios'])}\n")
    if report["best_threshold"] is not None:
        print(f"RECOMMENDED CAMERA_MOTION_STABILIZE_PX = {report['best_threshold']}")
    else:
        print("RECOMMENDED CAMERA_MOTION_STABILIZE_PX = undetermined "
              "(no valid scenarios; see errors in the JSON)")

    # MAJOR FIX SECTION STARTS HERE
    if "best_threshold" in report and report["best_threshold"] is not None:
        selected_row = next((r for r in report["threshold_study"] 
                            if r["threshold"] == report["best_threshold"]), None)
        if selected_row:
            report["selected_threshold"] = selected_row["threshold"]
            report["selected_rule"] = "argmin mean_policy_error; ties go to the larger threshold"
            report["selected_mean_policy_error"] = selected_row["mean_policy_error"]

            candidates_str = ", ".join([
                f"{r['threshold']:.3f}:{r['mean_policy_error']:.3f}" 
                for r in report["threshold_study"]
            ])

            print(
                f"SELECTED threshold={selected_row['threshold']:.3f} "
                f"rule=argmin(mean_policy_error) "
                f"mean_policy_error={selected_row['mean_policy_error']:.3f} "
                f"candidates={candidates_str}"
            )
    # MAJOR FIX SECTION ENDS HERE


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="evidence/calibration.json", help="output JSON path")
    parser.add_argument("--quick", action="store_true",
                        help="smoke run: jitter {0,4,10}, one seed, one background")
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds (1..N)")
    args = parser.parse_args(argv)

    if args.quick:
        backgrounds, jitters, seeds = [QUICK_BACKGROUND], list(JITTERS_QUICK), [1]
    else:
        backgrounds, jitters = list(BACKGROUNDS_FULL), list(JITTERS_FULL)
        seeds = list(range(1, max(1, args.seeds) + 1))

    env_agent = os.environ.get("V2S_AGENT")
    settings: dict[str, Any] = {
        "width": WIDTH, "height": HEIGHT, "fps": FPS, "duration_s": DURATION_S,
        "container": "MJPG .avi", "truth_objects": TRUTH_OBJECTS, "object_size_px": OBJ_SIZE,
        "objects": "3 squares 24x24 (2 bright 255, 1 dark 0), all moving horizontally "
                   "to the right with x = x0 + int(1.5 * frame) px (15 px/s at 10 fps)",
        "backgrounds": backgrounds, "jitter_px": jitters, "seeds": seeds,
        "jitter_model": "uniform integer shift in [-j, j] per frame on both axes, "
                        "cv2.warpAffine BORDER_REPLICATE",
        "thresholds": THRESHOLDS, "tie_rule": "lowest mean error; tie -> largest threshold",
        "agent_on_run": "unmodified pipeline and policy (current threshold "
                        f"{CURRENT_THRESHOLD})",
        "quick": bool(args.quick), "env_V2S_AGENT": env_agent,
    }
    report: dict[str, Any] = {"note": NOTE, "opencv_version": cv2.__version__,
                              "settings": settings, "scenarios": []}
    if env_agent is not None and env_agent.strip().lower() == "off":
        report["warning"] = ("V2S_AGENT=off is set: the agent=True runs are also OFF, "
                             "no camera motion is measured and the study is empty.")
        print("WARNING: " + report["warning"], file=sys.stderr)

    t_start = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="v2s_calib_") as tmpdir:
            tmp = Path(tmpdir)
            total = len(backgrounds) * len(jitters) * len(seeds)
            idx = 0
            for bg in backgrounds:
                for j in jitters:
                    for seed in seeds:
                        idx += 1
                        s = run_scenario(tmp, bg, j, seed)
                        report["scenarios"].append(s)
                        print(f"[{idx}/{total}] {s['scenario']}: off={s['objects_off']} "
                              f"on={s['objects_on']} motion={s['camera_motion_px']} "
                              f"plan={s['plan_decision']}"
                              + (f" errors={len(s['errors'])}" if s["errors"] else ""),
                              file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001 - report, do not hide
        report["fatal_error"] = f"{exc!r}\n{traceback.format_exc(limit=5)}"
        print(f"FATAL: {exc!r}", file=sys.stderr)

    scenarios = report["scenarios"]
    valid = [s for s in scenarios if _valid(s)]
    study, best = threshold_study(valid)
    report["by_jitter"] = by_jitter(scenarios)
    report["threshold_study"] = study
    report["best_threshold"] = best

    # Populate selected_* fields for JSON output before writing file
    if report["best_threshold"] is not None:
        selected_row = next((r for r in report["threshold_study"]
                             if r["threshold"] == report["best_threshold"]), None)
        if selected_row:
            report["selected_threshold"] = selected_row["threshold"]
            report["selected_rule"] = "argmin mean_policy_error; ties go to the larger threshold"
            report["selected_mean_policy_error"] = selected_row["mean_policy_error"]

    report["n_valid_scenarios"] = len(valid)
    report["mean_error_never"] = _mean([s["abs_error_off"] for s in valid])
    report["mean_error_always"] = _mean([s["abs_error_on"] for s in valid])
    report["n_always_counterfactual"] = sum(1 for s in valid if not s.get("on_run_stabilised"))
    report["current_threshold"] = CURRENT_THRESHOLD
    report["current_threshold_error"] = (
        round(sum(policy_error(s, CURRENT_THRESHOLD) for s in valid) / len(valid), 4)
        if valid else None)
    report["runtime_s"] = round(time.monotonic() - t_start, 2)

    try:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out} ({report['runtime_s']} s)", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"could not write {args.out}: {exc!r}", file=sys.stderr)

    print_tables(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
