# vid2sql: ask a question of a pipeline database from the command line.
"""Ask plain questions of a vid2sql pipeline database (prints JSON).

Usage:
  python scripts/ask.py [--db PATH] ["<question>" ...]

Options:
  --db PATH   SQLite database to query, opened read-only
              (default: evidence/real_clips.db relative to the repository root,
              as written by scripts/run_real_clips.py).
  question    One or more questions. If none is given, the single question
              "How many moving objects are in each video?" is asked.

Questions are translated by the rule-based v2s.nlq module; unsupported
questions are reported with an "error" entry rather than guessed.
"""
import argparse
import contextlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2s import nlq  # noqa: E402

DEFAULT_DB = ROOT / "evidence" / "real_clips.db"
DEFAULT_QUESTION = "How many moving objects are in each video?"


def _parse(argv):
    parser = argparse.ArgumentParser(
        prog=Path(argv[0]).name if argv else "ask.py",
        description="Ask plain questions of a vid2sql pipeline database (prints JSON).",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=str(DEFAULT_DB),
        help="SQLite database to query read-only "
        "(default: evidence/real_clips.db relative to the repository root)",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help=f'question(s) to ask (default: "{DEFAULT_QUESTION}")',
    )
    args = parser.parse_args(argv[1:])
    questions = list(args.question) or [DEFAULT_QUESTION]
    return args.db, questions


def main(argv):
    db, questions = _parse(argv)
    out = []
    for q in questions:
        try:
            sql, pattern = nlq.translate(q)
        except nlq.UnsupportedQuestion as exc:
            out.append({"question": q, "error": str(exc)})
            continue
        with contextlib.closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(sql).fetchall()]
        out.append({"question": q, "pattern": pattern, "sql": sql, "rows": rows})
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
