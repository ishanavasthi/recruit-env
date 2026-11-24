"""Difficulty sweep: tune the generator against the acceptance criteria.

Each configuration patches the generator and rubric constants in-process,
re-measures the rubric priors, and reports the criteria from the design
(docs/superpowers/specs/2026-11-24-real-task-and-fairness-audit-design.md,
section 7 and 10):

  C1  Bayes ceiling (checked by tests/test_scoring_contract.py, not here)
  C2  best single source passes medium/hard in at most 20% of seeds
      (chosen on training seeds 1000+, evaluated on seeds 0+; pooled and per role)
  C3  medium: full read beats the best single source by at least 0.2
  C4  hard: prioritising beats fixed-order by >= 0.05 (paired 95% CI excludes 0),
      prioritising mean in [0.5, 0.85], random passes in under 5% of seeds
  C17 easy: rubric heuristic passes at least 90% of seeds within 40 steps

Usage (repo root):

    python scripts/difficulty_sweep.py --seeds 100            # the grid
    python scripts/difficulty_sweep.py --seeds 200 --current  # current constants only
    python scripts/difficulty_sweep.py --priors               # print measured PRIORS
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import statistics as st
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")

import env.profile_factory as pf  # noqa: E402
import tasks  # noqa: E402
from env import policies, rubric  # noqa: E402
from env.environment import RecruitmentEnvironment  # noqa: E402

TRAIN0 = 1000
SINGLE = [f"single_{s}" for s in rubric.SOURCES] + [f"lsq_{s}" for s in rubric.SOURCES]


def measure_priors(n: int = 400) -> dict[str, dict[str, float]]:
    """Role means of every rubric input over generated candidates (seeds 5000+)."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    factory = pf.ProfileFactory()
    for seed in range(5000, 5000 + n):
        ep = factory.generate_episode(seed, {"shortlist": 5, "hold": 8, "reject": 7})
        for c in ep.candidates:
            rec = rubric.reveal_record(c, rubric.SOURCES)
            acc[ep.jd.role_type]["github"].append(rubric.github_subscore(rec["platforms"]["github"]))
            acc[ep.jd.role_type]["leetcode"].append(rubric.leetcode_subscore(rec["platforms"]["leetcode"]))
            acc[ep.jd.role_type]["kaggle"].append(rubric.kaggle_subscore(rec["platforms"]["kaggle"]))
            acc[ep.jd.role_type]["github_activity"].append(rubric.github_activity(rec["platforms"]["github"]))
            acc[ep.jd.role_type]["leetcode_practice"].append(rubric.leetcode_practice(rec["platforms"]["leetcode"]))
            acc[ep.jd.role_type]["claimed_years"].append(rec["resume_sections"]["experience"]["years_experience"])
            acc[ep.jd.role_type]["skill_match"].append(
                rubric.skill_match(rec["resume_sections"]["skills"]["skills"], ep.jd))
    return {role: {k: round(st.mean(v), 3) for k, v in d.items()} for role, d in sorted(acc.items())}


def play(task: str, seed: int, name: str) -> float:
    env = RecruitmentEnvironment()
    env.reset(task, seed)
    policies.POLICIES[name](env)
    return env.state().episode_score


def scores(task: str, seeds: range, name: str) -> list[float]:
    return [play(task, s, name) for s in seeds]


def role(seed: int) -> str:
    return pf.ROLES[seed % len(pf.ROLES)]


def best_single(task: str, n: int) -> tuple[dict[str, str], list[float]]:
    """Per role, the single-source policy with the best mean on training seeds;
    returns the choice and its scores on evaluation seeds 0..n-1."""
    train = range(TRAIN0, TRAIN0 + n)
    choice = {}
    for r in pf.ROLES:
        seeds = [s for s in train if role(s) == r]
        means = {p: st.mean(play(task, s, p) for s in seeds) for p in SINGLE}
        choice[r] = max(means, key=means.get)
    return choice, [play(task, s, choice[role(s)]) for s in range(n)]


def paired_ci(a: list[float], b: list[float], reps: int = 2000) -> tuple[float, float, float]:
    d = np.array(a) - np.array(b)
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, size=len(d)).mean() for _ in range(reps)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def apply(cfg: dict) -> None:
    pf.NOISE = cfg.get("noise", pf.NOISE)
    pf.RHO_TECH = cfg.get("rho", pf.RHO_TECH)
    pf.SKILL_COVERAGE = cfg.get("coverage", pf.SKILL_COVERAGE)
    rubric.SKILL_FLOOR = cfg.get("skill_floor", rubric.SKILL_FLOOR)
    if "hard_budget" in cfg:
        tasks.TASK_REGISTRY["hard"] = dataclasses.replace(tasks.TASK_REGISTRY["hard"], max_steps=cfg["hard_budget"])
    if "easy_budget" in cfg:
        tasks.TASK_REGISTRY["easy"] = dataclasses.replace(tasks.TASK_REGISTRY["easy"], max_steps=cfg["easy_budget"])
    pf._signal_pool.cache_clear()
    policies.fit_single_source.cache_clear()
    rubric.PRIORS.update(measure_priors(200))
    policies.SOURCE_PRIORITY.update(measure_source_priority())


def measure_source_priority(per_role: int = 40) -> dict[str, tuple[str, ...]]:
    """Per role, sources ordered by the hard-task score of judging on that source alone."""
    out = {}
    for r in pf.ROLES:
        seeds = [s for s in range(2000, 2000 + 4 * per_role) if role(s) == r]
        means = {src: st.mean(play("hard", s, f"single_{src}") for s in seeds) for src in rubric.SOURCES}
        out[r] = tuple(sorted(means, key=lambda k: -means[k]))
    return out


def evaluate(cfg: dict, n: int) -> dict:
    apply(cfg)
    return measure(n, cfg)


def measure(n: int, cfg: dict | None = None) -> dict:
    """Criteria for the current constants over evaluation seeds 0..n-1."""
    out: dict = {"cfg": cfg or {}, "source_priority": dict(policies.SOURCE_PRIORITY)}
    thr = {t: tasks.TASK_REGISTRY[t].success_threshold for t in ("easy", "medium", "hard")}

    easy_h = scores("easy", range(n), "rubric_heuristic")
    out["C17_easy_heuristic_pass"] = sum(x >= thr["easy"] for x in easy_h) / n

    for task in ("medium", "hard"):
        choice, s = best_single(task, n)
        out[f"C2_{task}_best_single_pass"] = sum(x >= thr[task] for x in s) / n
        out[f"C2_{task}_best_single_mean"] = st.mean(s)
        per_role = defaultdict(list)
        for seed, x in zip(range(n), s):
            per_role[role(seed)].append(x >= thr[task])
        out[f"C2_{task}_per_role_pass"] = {r: round(sum(v) / len(v), 3) for r, v in per_role.items()}
        out[f"C2_{task}_choice"] = choice

    full = scores("medium", range(n), "fixed_order")
    out["C3_medium_full_minus_best_single"] = st.mean(full) - out["C2_medium_best_single_mean"]

    pri = scores("hard", range(n), "rubric_heuristic")
    fix = scores("hard", range(n), "fixed_order")
    rnd = scores("hard", range(n), "random")
    d, lo, hi = paired_ci(pri, fix)
    out["C4_hard_prioritising_mean"] = st.mean(pri)
    out["C4_hard_fixed_mean"] = st.mean(fix)
    out["C4_hard_diff_ci"] = (round(d, 3), round(lo, 3), round(hi, 3))
    out["C4_hard_random_pass"] = sum(x >= thr["hard"] for x in rnd) / n
    out["C4_hard_heuristic_pass"] = sum(x >= thr["hard"] for x in pri) / n

    att = [pf.ProfileFactory().generate_episode(s, tasks.TASK_REGISTRY[t].quotas,
                                                tasks.TASK_REGISTRY[t].required_conflicts).attempts
           for t in ("easy", "medium", "hard") for s in range(n)]
    out["attempts_mean_max"] = (round(st.mean(att), 2), max(att))

    out["ok"] = {
        "C2": out["C2_medium_best_single_pass"] <= 0.20 and out["C2_hard_best_single_pass"] <= 0.20,
        "C3": out["C3_medium_full_minus_best_single"] >= 0.2,
        "C4": (d >= 0.05 and lo > 0 and 0.5 <= out["C4_hard_prioritising_mean"] <= 0.85
               and out["C4_hard_random_pass"] < 0.05),
        "C17": out["C17_easy_heuristic_pass"] >= 0.90,  # within the easy budget
    }
    return out


def show(r: dict) -> None:
    print(r["cfg"], "->", r["ok"])
    for k, v in r.items():
        if k not in ("cfg", "ok"):
            print(f"    {k}: {v if not isinstance(v, float) else round(v, 3)}")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--current", action="store_true")
    ap.add_argument("--priors", action="store_true")
    args = ap.parse_args()

    if args.priors:
        for r, d in measure_priors(1000).items():
            print(f'    "{r}": {d},')
        return
    if args.current:
        show(measure(args.seeds))
        return
    grid = itertools.product([0.65, 0.75], [(3.0, 3.0), (5.0, 5.0)], [0.0])
    for floor, cov, rho in grid:
        show(evaluate({"skill_floor": floor, "coverage": cov, "rho": rho}, args.seeds))


if __name__ == "__main__":
    main()
