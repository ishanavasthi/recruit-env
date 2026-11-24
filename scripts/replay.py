"""Replay transition logs and check they reproduce exactly.

    python scripts/replay.py outputs/run/*.jsonl
    python scripts/replay.py docs/results/runs/            # every log under a directory

Each log is replayed against a fresh ``reset`` with its logged parameters:
every observation hash, reward and the final grade must match. A log written
by a different generator version is refused (``--no-version-check`` skips
that). Exit code 0 means every log reproduced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recruitenv.trajectory import ReplayError, replay  # noqa: E402


def _expand(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out.extend(sorted(p.rglob("*.jsonl")) if p.is_dir() else [p])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="log files or directories")
    ap.add_argument("--no-version-check", action="store_true")
    args = ap.parse_args(argv)
    logs = _expand(args.paths)
    failed = 0
    for log in logs:
        try:
            grade = replay(log, check_versions=not args.no_version_check)
            score = grade.get("score")
            print(f"ok   {log}" + (f"  score={score}" if score is not None else "  (unfinished)"))
        except ReplayError as exc:
            failed += 1
            print(f"FAIL {log}: {exc}")
    print(f"{len(logs) - failed}/{len(logs)} logs reproduced")
    return 1 if failed or not logs else 0


if __name__ == "__main__":
    sys.exit(main())
