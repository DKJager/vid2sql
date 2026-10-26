"""End to end on a REAL licensed clip with the real OpenCV pipeline: no stubs, no mocked detector.
Skipped when data/clips is absent (the clips are not part of the public repo)."""
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "data" / "clips" / "clip1_slovenska.webm"

pytestmark = pytest.mark.skipif(not CLIP.is_file(), reason="licensed clips not present")


def test_real_clip_runs_through_real_opencv_into_sqlite(tmp_path):
    import cv2
    assert cv2.__version__.startswith("5."), cv2.__version__
    clips = tmp_path / "clips"
    clips.mkdir()
    shutil.copy(CLIP, clips / CLIP.name)
    db = tmp_path / "out" / "r.db"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_real_clips.py"), str(clips), "--db", str(db)],
                       capture_output=True, text=True, timeout=900, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-800:]
    conn = sqlite3.connect(db)
    n_tracks = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
    n_trace = conn.execute("SELECT COUNT(*) FROM agent_trace").fetchone()[0]
    conn.close()
    assert n_tracks and n_tracks > 0
    assert n_trace > 0
    ref = json.loads((ROOT / "evidence" / "real_ab.json").read_text())["agent_on"][CLIP.name]["tracks"]
    assert n_tracks == ref, (n_tracks, ref)
