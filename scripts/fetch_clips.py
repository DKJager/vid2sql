# vid2sql: download the evaluation clips listed in data/CLIPS.csv.
"""Usage: python scripts/fetch_clips.py [--dest DIR]

Downloads each clip from its recorded URL (retrying when the server asks us
to slow down) and refuses any file whose SHA-256 differs from the registry.

Options:
  --dest DIR   directory to store the clips in (default: data/clips under the
               repository root).
  -h, --help   show this help and exit.
"""
import argparse
import csv
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fetch_clips.py",
        description=(
            "Download the evaluation clips listed in data/CLIPS.csv and refuse "
            "any file whose SHA-256 differs from the registry."
        ),
    )
    parser.add_argument(
        "--dest",
        metavar="DIR",
        default=None,
        help="directory to store the clips in (default: data/clips under the repository root)",
    )
    return parser


def main(argv):
    args = build_parser().parse_args(argv[1:])
    dest = Path(args.dest) if args.dest is not None else ROOT / "data" / "clips"
    dest.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data" / "CLIPS.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        target = dest / row["filename"]
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]:
            print("ok (already present)", target.name)
            continue
        req = urllib.request.Request(row["download_url"], headers={"User-Agent": "vid2sql-fetch/1.0 (research)"})
        for attempt in range(8):
            try:
                data = urllib.request.urlopen(req, timeout=300).read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429:
                    raise
                time.sleep(20 * (attempt + 1))
        else:
            raise SystemExit(f"gave up on {row['filename']}")
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise SystemExit(f"checksum mismatch for {row['filename']}")
        target.write_bytes(data)
        print("downloaded", target.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
