# vid2sql: evaluation of the real OpenCV 5 pipeline on the licensed clips.
"""Run the pipeline over the clips in data/CLIPS.csv and check it.

There is no human-annotated ground truth for these clips, so this script does
NOT report precision or recall. It checks what can be checked honestly:

* every clip in the registry is processed and has a licence and checksum that
  matches the file on disk;
* determinism: two independent runs produce identical tables;
* invariants: times lie inside the clip, confidences in [0, 1], boxes valid;
* coverage and load statistics (tracks, peak simultaneous tracks, fraction of
  the clip with at least one track) so a reader can judge plausibility.

Usage: python eval/run_eval.py [--clips-dir DIR] [--out evidence/eval_real.json]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v2s import agent, core_cv, ingest, sqlwriter  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _registry_rows() -> list[dict[str, str]]:
    with (ROOT / "data" / "CLIPS.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _run_once(clips_dir: Path, db: Path) -> None:
    writer = sqlwriter.SqliteWriter(str(db))
    for item in ingest.ingest_local(clips_dir):
        agent.process_clip(item, writer)


def _snapshot(db: Path) -> dict[str, list[tuple]]:
    """Table contents without the randomly generated ids and wall-clock times."""
    queries = {
        "objects": "SELECT v.filename, o.track_id, o.first_seen_s, o.last_seen_s, o.max_count,"
                   " o.confidence FROM objects o JOIN videos v USING (video_id) ORDER BY 1, 2",
        "segments": "SELECT v.filename, s.start_s, s.end_s, s.motion_score FROM segments s"
                    " JOIN videos v USING (video_id) ORDER BY 1, 2",
    }
    with sqlite3.connect(db) as conn:
        return {k: conn.execute(q).fetchall() for k, q in queries.items()}


def _invariants(db: Path, durations: dict[str, float]) -> list[str]:
    problems: list[str] = []
    with sqlite3.connect(db) as conn:
        for name, first, last, conf in conn.execute(
            "SELECT v.filename, o.first_seen_s, o.last_seen_s, o.confidence FROM objects o"
            " JOIN videos v USING (video_id)"
        ):
            dur = durations[name]
            if not (0.0 <= first <= last <= dur + 0.1):
                problems.append(f"{name}: object interval [{first},{last}] outside 0..{dur:.1f}")
            if not (0.0 <= conf <= 1.0):
                problems.append(f"{name}: confidence {conf} outside [0,1]")
        for name, a, b in conn.execute(
            "SELECT v.filename, s.start_s, s.end_s FROM segments s JOIN videos v USING (video_id)"
        ):
            if not (0.0 <= a < b <= durations[name] + 0.1):
                problems.append(f"{name}: segment [{a},{b}] invalid")
    return problems


def _stats(db: Path, durations: dict[str, float]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with sqlite3.connect(db) as conn:
        for name, dur in durations.items():
            ivals = conn.execute(
                "SELECT o.first_seen_s, o.last_seen_s FROM objects o JOIN videos v USING (video_id)"
                " WHERE v.filename = ?", (name,)).fetchall()
            seconds = int(dur) + 1
            occ = [sum(1 for a, b in ivals if a <= t <= b) for t in range(seconds)]
            out[name] = {
                "duration_s": round(dur, 2),
                "tracks": len(ivals),
                "peak_simultaneous_tracks": max(occ) if occ else 0,
                "fraction_of_seconds_with_a_track": round(sum(1 for n in occ if n) / max(1, seconds), 3),
                "median_track_seconds": round(sorted(b - a for a, b in ivals)[len(ivals) // 2], 2) if ivals else 0.0,
                "scene_segments": conn.execute(
                    "SELECT COUNT(*) FROM segments s JOIN videos v USING (video_id) WHERE v.filename = ?",
                    (name,)).fetchone()[0],
            }
    return out


def run_eval(clips_dir: Path) -> dict:
    registry = _registry_rows()
    report: dict = {"kind": "sanity and determinism checks; no ground-truth precision/recall",
                    "opencv": core_cv.cv2.__version__, "clips": {}}
    durations: dict[str, float] = {}
    for row in registry:
        path = clips_dir / row["filename"]
        ok = path.is_file() and _sha256(path) == row["sha256"]
        durations[row["filename"]] = core_cv.probe_duration(str(path)) if path.is_file() else 0.0
        report["clips"][row["filename"]] = {"license": row["license"], "checksum_matches": ok}
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.db", Path(tmp) / "b.db"
        _run_once(clips_dir, a)
        _run_once(clips_dir, b)
        report["deterministic"] = _snapshot(a) == _snapshot(b)
        report["invariant_violations"] = _invariants(a, durations)
        for name, st in _stats(a, durations).items():
            report["clips"][name].update(st)
    report["passed"] = (
        report["deterministic"] and not report["invariant_violations"]
        and all(c["checksum_matches"] and c.get("tracks", 0) > 0 for c in report["clips"].values())
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-dir", type=Path, default=ROOT / "data" / "clips")
    ap.add_argument("--out", type=Path, default=ROOT / "evidence" / "eval_real.json")
    args = ap.parse_args(argv)
    report = run_eval(args.clips_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
