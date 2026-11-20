"""Scoring and difficulty probes for RecruitEnv (read-only, 200 seeds per task).

Evidence for docs/research/2026-11-21-findings-and-roadmap.md. Run from the
repo root:

    PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.11 \
        --with fastapi --with pydantic --with numpy --with python-dotenv \
        python scripts/probe_scoring.py

Columns: `grader` is the task grader's score, `env_score` is the environment's
own terminal `episode_score`, `pass%` is the share of seeds where the grader
score meets the task's success threshold.
"""
import statistics as st
import sys

sys.path.insert(0, ".")

from env.environment import RecruitmentEnvironment
from env.fairness import FairnessChecker
from env.models import CheckPlatformAction, MakeDecisionAction, ScoreDimensionAction
from tasks import GRADER_REGISTRY, TASK_REGISTRY

SEEDS = range(200)
TASKS = ["easy", "medium", "hard"]


def run(task, seed, policy):
    env = RecruitmentEnvironment()
    obs = env.reset(task, seed)
    final = policy(env, obs)
    state = env.state()
    return GRADER_REGISTRY[task].grade(obs, state), state.episode_score, state, final


def finish(env):
    """Burn remaining budget with a free no-op so the episode terminates."""
    s = env.state()
    last = None
    cid = s.candidates[0].id
    while not env.state().is_done:
        last = env.step(ScoreDimensionAction(candidate_id=cid, dimension="technical", score=0.5))
    return last


# --- policies ---------------------------------------------------------------

def oracle(env, obs):
    """Decide everything correctly using ground truth, no information gathering."""
    last = None
    for c in env.state().candidates:
        if env.state().is_done:
            break
        last = env.step(MakeDecisionAction(candidate_id=c.id, decision=c.ground_truth_label))
    return last


def do_nothing(env, obs):
    return finish(env)


def all_hold(env, obs):
    last = None
    for c in obs.candidates_summary:
        if env.state().is_done:
            break
        last = env.step(MakeDecisionAction(candidate_id=c["id"], decision="hold"))
    return last


def all_shortlist(env, obs):
    last = None
    for c in obs.candidates_summary:
        if env.state().is_done:
            break
        last = env.step(MakeDecisionAction(candidate_id=c["id"], decision="shortlist"))
    return last


def one_signal(env, obs):
    """One LeetCode check per candidate, threshold on `contest_rating` only.

    Shortlists are decided as soon as found; other labels only if budget allows.
    """
    last = None
    pending = []
    for c in obs.candidates_summary:
        if env.state().is_done:
            return last
        _, _, done, info = env.step(CheckPlatformAction(candidate_id=c["id"], platform="leetcode"))
        if done:
            return last
        rating = info["data"]["contest_rating"]
        label = "shortlist" if rating >= 1601 else ("reject" if rating < 1200 else "hold")
        if label == "shortlist":
            last = env.step(MakeDecisionAction(candidate_id=c["id"], decision=label))
            if last[2]:
                return last
        else:
            pending.append((c["id"], label))
    for cid, label in pending:
        if env.state().is_done:
            break
        last = env.step(MakeDecisionAction(candidate_id=cid, decision=label))
    if not env.state().is_done:
        last = finish(env)
    return last


def one_correct_reject(env, obs):
    """Decide exactly one candidate (a true reject) and let the budget run out."""
    c = next(c for c in env.state().candidates if c.ground_truth_label == "reject")
    env.step(MakeDecisionAction(candidate_id=c.id, decision="reject"))
    return finish(env)


POLICIES = {
    "oracle (all correct, 0 checks)": oracle,
    "do nothing": do_nothing,
    "all hold": all_hold,
    "all shortlist": all_shortlist,
    "1 leetcode check + threshold": one_signal,
    "decide ONE true reject only": one_correct_reject,
}

print(f"{'policy':34s} {'task':7s} {'grader':>8s} {'env_score':>10s} {'pass%':>6s}")
for name, pol in POLICIES.items():
    for task in TASKS:
        g, e = [], []
        for seed in SEEDS:
            gs, es, _, _ = run(task, seed, pol)
            g.append(gs)
            e.append(es if es is not None else float("nan"))
        thr = TASK_REGISTRY[task].success_threshold
        print(f"{name:34s} {task:7s} {st.mean(g):8.3f} {st.mean(e):10.3f} {100*sum(x>=thr for x in g)/len(g):5.0f}%")

# --- fairness penalty on a perfectly correct (oracle) agent -------------------
print("\nfairness penalty applied to a 100%-correct agent:")
for task in TASKS:
    pens = []
    for seed in SEEDS:
        env = RecruitmentEnvironment()
        env.reset(task, seed)
        s = env.state()
        dec = {c.id: c.ground_truth_label for c in s.candidates}
        pens.append(FairnessChecker.compute_penalty(dec, s.candidates))
    print(f"  {task:7s} mean={st.mean(pens):.3f}  penalised in {100*sum(p>0 for p in pens)/len(pens):.0f}% of seeds, max(0.2) in {100*sum(p>=0.2 for p in pens)/len(pens):.0f}%")

# --- is any single revealed field a perfect label separator? ------------------
print("\nsingle-field separability of labels (200 seeds, hard task):")
rows = []
for seed in SEEDS:
    env = RecruitmentEnvironment()
    env.reset("hard", seed)
    for c in env.state().candidates:
        rows.append((c.ground_truth_label, c))
fields = {
    "leetcode.contest_rating": lambda c: c.leetcode.contest_rating,
    "leetcode.hard": lambda c: c.leetcode.hard,
    "leetcode.global_rank_percentile": lambda c: c.leetcode.global_rank_percentile,
    "github.stars_received": lambda c: c.github.stars_received,
    "github.repos": lambda c: c.github.repos,
    "github.contributions_last_year": lambda c: c.github.contributions_last_year,
    "kaggle.best_finish_percentile": lambda c: c.kaggle.best_finish_percentile,
    "resume.experience.years": lambda c: c.resume_sections["experience"].years_experience,
}
for fname, f in fields.items():
    rng = {}
    for lab, c in rows:
        v = f(c)
        lo, hi = rng.get(lab, (v, v))
        rng[lab] = (min(lo, v), max(hi, v))
    r, h, s_ = rng["reject"], rng["hold"], rng["shortlist"]
    sep = r[1] < h[0] and h[1] < s_[0]
    print(f"  {fname:34s} reject={r} hold={h} shortlist={s_}  perfectly separable={sep}")

# --- conflicted candidates in medium -----------------------------------------
print("\nmedium task: do candidates 0 and 1 actually have conflicting signals?")
tier = lambda v, a, b: "senior" if v >= b else ("junior" if v < a else "mid")
conf = 0
for seed in SEEDS:
    env = RecruitmentEnvironment()
    env.reset("medium", seed)
    for c in env.state().candidates[:2]:
        t = {tier(c.leetcode.contest_rating, 1200, 1601), tier(c.github.repos, 10, 30)}
        conf += len(t) > 1
print(f"  conflicted (github tier != leetcode tier): {conf} of {2*len(SEEDS)}")

# --- step reward farming ------------------------------------------------------
env = RecruitmentEnvironment()
obs = env.reset("easy", 42)
tot = 0.0
cid = obs.candidates_summary[0]["id"]
for _ in range(40):
    _, r, done, _ = env.step(ScoreDimensionAction(candidate_id=cid, dimension="technical", score=0.0))
    tot += r.step_reward
print(f"\nsum of step rewards from 40x score_dimension on one candidate, zero decisions: {tot:.2f}")
print("observation.revealed_data after a check carries stats?:", end=" ")
env = RecruitmentEnvironment(); obs = env.reset("easy", 42)
o, _, _, info = env.step(CheckPlatformAction(candidate_id=cid, platform="github"))
print(o.revealed_data, "| info keys:", list(info["data"].keys())[:3])
