#!/usr/bin/env bash
# Build the submission archive from the repository root, excluding the
# virtualenv, caches and git metadata. Fails if any .venv file leaks in.
# Uses python3 (zipfile) so it works without the 'zip' command.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="vid2sql-submission.zip"
rm -f "$OUT"

PKG_OUT="$OUT" python3 - <<'PY'
import os, sys, zipfile
out = os.environ["PKG_OUT"]
skip_dirs = {".venv", ".git", "__pycache__", ".pytest_cache", ".aws-sam",
             "data/clips", "out"}
if os.environ.get("PKG_TEST_LEAK"):
    skip_dirs.discard(".venv")  # self-test: prove the leak check fires
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in skip_dirs
                   and os.path.relpath(os.path.join(root, d), ".").replace(os.sep, "/") not in skip_dirs]
        for f in files:
            # One combined exclusion test: the .pyc and output-archive
            # exclusions are always evaluated together with the audit and
            # backup exclusions, so none of them can be bypassed.
            if (f.endswith(".pyc") or f == out
                    or f.endswith(".orig") or f == "AUDIT_LOG.jsonl"
                    or (f.startswith("AUDIT_REVIEW_") and f.endswith(".md"))):
                continue
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, "."))
names = zipfile.ZipFile(out).namelist()
if any(n.startswith(".venv/") for n in names):
    print("ERROR: .venv files found in", out, file=sys.stderr)
    sys.exit(1)
required = ["README.md", "docs/REPORT.md", "v2s/schema.sql", "requirements.txt",
            "data/CLIPS.csv", "evidence/real_clips.db",
            "LICENSE", "docs/architecture.svg", "docs/agent_workflow.svg",
            "evidence/synth_eval.json", "evidence/agent_trace_real.md"]
missing = [r for r in required if r not in names]
if missing:
    print("ERROR: missing from", out, ":", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
print("OK:", out, len(names), "files")
PY
du -h "$OUT"
