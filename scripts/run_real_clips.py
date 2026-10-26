# vid2sql: run the local OpenCV 5 pipeline over data/clips into SQLite.
"""Usage: python scripts/run_real_clips.py [clips_dir] [--db PATH] [--update-evidence] [--no-agent]

By default the database is written to out/real_clips.db and an existing
database at that path is never replaced. Only --update-evidence writes to
evidence/real_clips.db (and evidence/pipeline_run.json), and it is the only
case where an existing database is replaced. The run record (JSON) is written
beside the chosen database file: evidence/pipeline_run.json with
--update-evidence, otherwise the database path with a .json suffix
(for example out/real_clips.json).
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2s import agent, ingest, sqlwriter  # noqa: E402

def _parse_args(args):
    p = argparse.ArgumentParser(
        prog="run_real_clips.py",
        description="Run the local OpenCV 5 pipeline over a folder of clips into SQLite.")
    p.add_argument("clips_dir", nargs="?", default=str(ROOT / "data" / "clips"),
                   help="folder of licensed clips (default: data/clips)")
    p.add_argument("--db", default=str(ROOT / "out" / "real_clips.db"),
                   help="SQLite database to write (default: out/real_clips.db); "
                        "an existing database is never replaced; the run record "
                        "is written beside it with a .json suffix")
    p.add_argument("--update-evidence", action="store_true",
                   help="write evidence/real_clips.db and evidence/pipeline_run.json "
                        "instead, replacing the existing evidence database")
    p.add_argument("--no-agent", action="store_true",
                   help="run the pipeline with the agent loop disabled")
    return p.parse_args(args)

def main(argv):
    ns = _parse_args(argv[1:])
    clips_dir = Path(ns.clips_dir)
    if ns.update_evidence:
        db = ROOT / "evidence" / "real_clips.db"
        run_json = ROOT / "evidence" / "pipeline_run.json"
    else:
        db = Path(ns.db)
        run_json = db.with_suffix(".json")
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        if ns.update_evidence:
            db.unlink()
        else:
            raise SystemExit(f"refusing to replace existing database {db}; "
                             "choose another --db path or use --update-evidence")
    items = ingest.ingest_local(clips_dir)
    if not items:
        raise SystemExit(f"no licensed clips found in {clips_dir}")
    writer = sqlwriter.SqliteWriter(str(db))
    out = {}
    for item in items:
        t0 = time.time()
        if ns.no_agent:
            res = agent.process_clip(item, writer, agent=False)
        else:
            res = agent.process_clip(item, writer)
        out[item.filename] = {"video_id": res["video_id"], "counts": res["counts"],
                              "wall_s": round(time.time() - t0, 1)}
        print(item.filename, out[item.filename], flush=True)
    run_json.write_text(json.dumps(out, indent=2))
    for filename, rec in out.items():
        counts = rec["counts"] or {}
        print(f"summary {filename}: agent_trace={counts.get('agent_trace')} "
              f"review_queue={counts.get('review_queue')}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
