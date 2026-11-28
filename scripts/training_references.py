"""Reference scores for reading a training curve, on the training seeds.

The GRPO example trains on seeds from 10,000. A rising curve only means
something next to what trivial policies score on the same seeds, so this
prints (and with ``--json`` writes) the mean score and pass rate of:

- idle: no decisions at all (the reward floor);
- blind constant labels: shortlist, hold or reject everyone without reading;
- blind quota fill: label in id order to match the quotas, without reading;
- random labels;
- the rubric heuristic and the oracle (which read the evidence).

    python scripts/training_references.py --task easy --seeds 10000:10200
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Callable

from recruitenv.environment import RecruitmentEnvironment
from recruitenv.policies import POLICIES
from recruitenv.tools import RecruitToolEnv

PASS = 0.5


def _blind(task: str, seed: int, labels: Callable[[list[str], dict, random.Random], list[str]]) -> float:
    env = RecruitToolEnv()
    env.reset(task=task, seed=seed)
    obs = env.env.reset(task, seed)
    ids = [c["id"] if isinstance(c, dict) else c.id for c in obs.candidates_summary]
    for cid, label in zip(ids, labels(ids, obs.quotas, random.Random(seed))):
        try:
            getattr(env, label)(cid)
        except ValueError:
            pass
    return env.get_reward()


def _quota_fill(ids: list[str], quotas: dict, _rng: random.Random) -> list[str]:
    out = [k for k in ("shortlist", "hold", "reject") for _ in range(int(quotas.get(k, 0)))]
    return (out + ["reject"] * len(ids))[: len(ids)]


def _policy(task: str, seed: int, name: str) -> float:
    env = RecruitmentEnvironment()
    env.reset(task, seed)
    POLICIES[name](env)
    return env.state().episode_score or 0.0


def references(task: str, seeds: range) -> dict[str, dict[str, float]]:
    runs: dict[str, Callable[[int], float]] = {
        "idle": lambda s: _blind(task, s, lambda ids, q, r: []),
        "shortlist everyone": lambda s: _blind(task, s, lambda ids, q, r: ["shortlist"] * len(ids)),
        "hold everyone": lambda s: _blind(task, s, lambda ids, q, r: ["hold"] * len(ids)),
        "reject everyone": lambda s: _blind(task, s, lambda ids, q, r: ["reject"] * len(ids)),
        "blind quota fill": lambda s: _blind(task, s, _quota_fill),
        "random labels": lambda s: _blind(
            task, s, lambda ids, q, r: [r.choice(["shortlist", "hold", "reject"]) for _ in ids]),
        "rubric heuristic": lambda s: _policy(task, s, "rubric_heuristic"),
        "oracle": lambda s: _policy(task, s, "oracle"),
    }
    out = {}
    for name, fn in runs.items():
        scores = [fn(s) for s in seeds]
        out[name] = {"mean": round(statistics.mean(scores), 4),
                     "pass_rate": round(sum(x >= PASS for x in scores) / len(scores), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    ap.add_argument("--seeds", default="10000:10200", help="start:stop")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    start, stop = (int(x) for x in args.seeds.split(":"))
    refs = references(args.task, range(start, stop))
    for name, r in refs.items():
        print(f"{name:20s} mean {r['mean']:.3f}  pass rate {r['pass_rate']:.2f}")
    if args.json:
        args.json.write_text(json.dumps({"task": args.task, "seeds": [start, stop - 1], "references": refs},
                                        indent=2) + "\n")


if __name__ == "__main__":
    main()
