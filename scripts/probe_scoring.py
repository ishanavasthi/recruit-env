"""Scoreboard of the reference policies (read-only).

Plays every reference policy in ``recruitenv/policies.py`` on every task and prints
the mean score and the share of seeds that pass the task's success
threshold. The 1.0.0 findings in docs/research/2026-11-21-findings-and-roadmap.md
came from an earlier version of this script (tag ``v1.0.0``). Acceptance
criteria for the 3.0.0 generator are checked by scripts/difficulty_sweep.py
and tests/test_acceptance.py.

    python scripts/probe_scoring.py [--seeds 200]
"""

from __future__ import annotations

import argparse
import statistics as st
import sys

sys.path.insert(0, ".")

from recruitenv import policies  # noqa: E402
from recruitenv.environment import RecruitmentEnvironment  # noqa: E402
from recruitenv.tasks import TASK_REGISTRY  # noqa: E402

ROWS = [
    "oracle", "rubric_heuristic", "fixed_order", "random",
    "single_github", "single_leetcode", "single_kaggle", "single_experience", "single_skills",
    "lsq_github", "lsq_skills",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()
    seeds = range(args.seeds)
    print(f"{'policy':20s} " + " ".join(f"{t + ' mean':>11s} {t + ' pass':>10s}" for t in TASK_REGISTRY))
    for name in ROWS:
        cells = []
        for task, cfg in TASK_REGISTRY.items():
            scores = []
            for seed in seeds:
                env = RecruitmentEnvironment()
                env.reset(task, seed)
                policies.POLICIES[name](env)
                scores.append(env.state().episode_score)
            passed = sum(s >= cfg.success_threshold for s in scores) / len(scores)
            cells.append(f"{st.mean(scores):11.3f} {100 * passed:9.0f}%")
        print(f"{name:20s} " + " ".join(cells), flush=True)


if __name__ == "__main__":
    main()
