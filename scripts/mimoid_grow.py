# Overnight library growth after `mimoid_data.py select --extend`: build the
# new grids, encode them with the trained
# AE, re-embed captions, score quality (+ duplicates),
# export for the browser, then render review sheets. Each step's output goes to
# the log; the chain stops at the first failure and is safe to rerun (build
# resumes, encode/quality only touch rows they haven't seen).
# Run: .venv/Scripts/python scripts/mimoid_grow.py [--limit N] [--log data/mimoid/grow.log]
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRA = "tableware,food,fauna,household,structures,machines,plants,people,tools,insects,instruments"  # the batch being grown (review sheets only)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="build only the next N objects (smoke test)")
    ap.add_argument("--log", default=str(ROOT / "data" / "mimoid" / "grow.log"))
    args = ap.parse_args()
    build = ["mimoid_data.py", "build"] + (["--limit", str(args.limit)] if args.limit else [])
    steps = [
        build,
        ["mimoid_orient.py", "--minutes", "40", "--dry-run"],  # up-orientation net + review sheet only; apply after review
        ["mimoid_encode.py"],
        ["mimoid_captions.py"],
        ["mimoid_quality.py", "--sheet"],
        ["export_dream.py"],
        ["mimoid_data.py", "preview", "--families", EXTRA, "--per-family", "36"],
        ["mimoid_gaps.py"],
        ["mimoid_dream.py", "--tag", "_grown"],
    ]
    with open(args.log, "a", encoding="utf-8") as log:
        for cmd in steps:
            t0 = time.time()
            head = f"\n===== {' '.join(cmd)}  ({time.strftime('%H:%M:%S')})"
            print(head, flush=True)
            log.write(head + "\n"); log.flush()
            p = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / cmd[0]), *cmd[1:]], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            tail = f"----- exit {p.returncode} after {(time.time() - t0) / 60:.1f} min"
            print(tail, flush=True)
            log.write(tail + "\n"); log.flush()
            if p.returncode:
                sys.exit(f"stopped at {cmd[0]}; see {args.log}")
    print(f"done; see {args.log}")


if __name__ == "__main__":
    main()
