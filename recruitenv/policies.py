"""Reference policies for tests, probes, the difficulty sweep and the fairness report.

Every policy plays one episode on a reset :class:`RecruitmentEnvironment`
until it is done and returns the list of step rewards. Policies that read
ground truth (``oracle``, ``planted_biased``) use ``verifier_state()`` and
exist only to calibrate the environment; agents never get that view.

No policy reads candidate names except ``planted_biased``, which is a
deliberately biased control for the fairness audit. Ties between equal
estimates are broken by candidate id, never by name.
"""

from __future__ import annotations

import functools
import hashlib
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from recruitenv import rubric
from recruitenv.environment import RecruitmentEnvironment
from recruitenv.models import (
    CheckPlatformAction,
    JobDescription,
    MakeDecisionAction,
    ReadResumeSectionAction,
)
from recruitenv.names import group_of
from recruitenv.profile_factory import ROLES, fill_quotas

Policy = Callable[[RecruitmentEnvironment], list[float]]

TRAIN_SEEDS = range(1000, 1200)

# Source order for screening and deep reads, most informative first per role:
# sources ranked by the hard-task score of judging on that source alone
# (measured by scripts/difficulty_sweep.py). The first entry is the
# screening source.
SOURCE_PRIORITY: dict[str, tuple[str, ...]] = {
    "ml_engineer": ("leetcode", "kaggle", "github", "skills", "experience"),
    "frontend_dev": ("leetcode", "skills", "github", "kaggle", "experience"),
    "backend_dev": ("leetcode", "skills", "experience", "kaggle", "github"),
    "data_scientist": ("leetcode", "kaggle", "github", "skills", "experience"),
}


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def action_for(cid: str, source: str):
    """The action that reveals *source* for candidate *cid* (inverse of ``rubric.source_of``)."""
    if source in rubric.PLATFORMS:
        return CheckPlatformAction(candidate_id=cid, platform=source)
    return ReadResumeSectionAction(candidate_id=cid, section=source)


def _step(env: RecruitmentEnvironment, action, rewards: list[float]) -> bool:
    _, reward, done, _ = env.step(action)
    rewards.append(reward.step_reward)
    return done


def idle_until_done(env: RecruitmentEnvironment, rewards: list[float]) -> None:
    """Burn the remaining budget by re-reading one education section."""
    cid = env.state().candidates_summary[0]["id"]
    while not env.state().is_done:
        _step(env, ReadResumeSectionAction(candidate_id=cid, section="education"), rewards)


def _decide_many(env: RecruitmentEnvironment, labels: dict[str, str], rewards: list[float]) -> None:
    """Decide in label order (shortlists first), then idle until done."""
    rank = {"shortlist": 0, "reject": 1, "hold": 2}
    for cid in sorted(labels, key=lambda c: (rank[labels[c]], c)):
        if env.state().is_done:
            return
        if cid in env.state().decisions_made:
            continue
        if _step(env, MakeDecisionAction(candidate_id=cid, decision=labels[cid]), rewards):
            return
    if not env.state().is_done:
        idle_until_done(env, rewards)


def _revealed(view: Any, cid: str) -> set[str]:
    """Sources revealed for *cid*. *view* is an environment or a state snapshot."""
    st = view.state() if isinstance(view, RecruitmentEnvironment) else view
    rec = st.revealed_data.get(cid) or {}
    return set(rec.get("platforms", {})) | set(rec.get("resume_sections", {}))


def _decide_all_needed(env: RecruitmentEnvironment) -> bool:
    """Hard scores shortlists only (undecided counts as not shortlisted)."""
    return env.state().task_id != "hard"


def _decisions_needed(env: RecruitmentEnvironment, s: Any = None) -> int:
    s = s or env.state()
    if s.task_id != "hard":
        return len(s.candidates_summary) - len(s.decisions_made)
    # One spare step: tied candidates (twins) at the cut are both shortlisted.
    return s.quotas["shortlist"] + 1 - sum(1 for d in s.decisions_made.values() if d == "shortlist")


def _estimates(env: RecruitmentEnvironment, s: Any = None) -> dict[str, float]:
    s = s or env.state()
    return {c["id"]: rubric.estimate(s.revealed_data.get(c["id"]), s.job_description).total for c in s.candidates_summary}


def _decide_by_estimates(env: RecruitmentEnvironment, rewards: list[float]) -> list[float]:
    s = env.state()
    labels = fill_quotas(_estimates(env), s.quotas, decide_all=_decide_all_needed(env))
    _decide_many(env, labels, rewards)
    return rewards


# ---------------------------------------------------------------------------
# reference and calibration policies
# ---------------------------------------------------------------------------


def oracle(env: RecruitmentEnvironment) -> list[float]:
    """Ground-truth decisions; reveals each conflicted pair before deciding it."""
    rewards: list[float] = []
    vs = env.verifier_state()
    for cid, pair in sorted(vs.conflicts.items()):
        for src in pair:
            if src not in _revealed(env, cid) and _step(env, action_for(cid, src), rewards):
                return rewards
    _decide_many(env, {c.id: c.ground_truth_label for c in vs.candidates}, rewards)
    return rewards


def rubric_full_read(env: RecruitmentEnvironment) -> list[float]:
    """Read every rubric source of every candidate in order, reserving budget to decide.

    With enough budget (easy, medium) this is a full read and scores the
    maximum. Without it (hard) it is the fixed-order heuristic: candidates
    late in the list are judged on priors.
    """
    rewards: list[float] = []
    s = env.state()
    order = SOURCE_PRIORITY[s.job_description.role_type]
    for c in s.candidates_summary:
        for src in order:
            st = env.state()
            if st.max_steps - st.step_number <= _decisions_needed(env, st):
                return _decide_by_estimates(env, rewards)
            if src not in _revealed(st, c["id"]) and _step(env, action_for(c["id"], src), rewards):
                return rewards
    return _decide_by_estimates(env, rewards)


fixed_order = rubric_full_read


def rubric_heuristic(env: RecruitmentEnvironment) -> list[float]:
    """Screen everyone with one source, then deep-read the candidates nearest a quota cut.

    Priority of a candidate = unrevealed sources / (distance from the nearest
    quota cut + 0.01). Never reads names.
    """
    rewards: list[float] = []
    s = env.state()
    order = SOURCE_PRIORITY[s.job_description.role_type]
    ids = [c["id"] for c in s.candidates_summary]

    def budget_left(st: Any) -> int:
        return st.max_steps - st.step_number - _decisions_needed(env, st)

    for cid in ids:
        st = env.state()
        if budget_left(st) <= 0:
            return _decide_by_estimates(env, rewards)
        if order[0] not in _revealed(st, cid) and _step(env, action_for(cid, order[0]), rewards):
            return rewards

    k = s.quotas["shortlist"]
    r = s.quotas["reject"]
    while True:
        st = env.state()
        if budget_left(st) <= 0:
            break
        est = _estimates(env, st)
        ranked = sorted(est, key=lambda c: (-est[c], c))
        cuts = []
        if 0 < k < len(ranked):
            cuts.append((est[ranked[k - 1]] + est[ranked[k]]) / 2)
        if _decide_all_needed(env) and 0 < r < len(ranked):
            cuts.append((est[ranked[-r - 1]] + est[ranked[-r]]) / 2)
        best, best_p = None, 0.0
        for cid in ids:
            seen = _revealed(st, cid)
            missing = [x for x in order if x not in seen]
            if not missing:
                continue
            dist = min(abs(est[cid] - cut) for cut in cuts) if cuts else 1.0
            p = len(missing) / (dist + 0.01)
            if p > best_p:
                best, best_p = cid, p
        if best is None:
            break
        nxt = next(x for x in order if x not in _revealed(st, best))
        if _step(env, action_for(best, nxt), rewards):
            return rewards
    return _decide_by_estimates(env, rewards)


def random_quota(env: RecruitmentEnvironment, rng_seed: int = 0) -> list[float]:
    """Uniform random permutation filling the quotas; no information gathered."""
    rewards: list[float] = []
    s = env.state()
    rng = np.random.default_rng([rng_seed, s.seed])
    ids = [c["id"] for c in s.candidates_summary]
    perm = rng.permutation(len(ids))
    est = {ids[int(i)]: float(len(ids) - pos) for pos, i in enumerate(perm)}
    labels = fill_quotas(est, s.quotas, decide_all=_decide_all_needed(env))
    _decide_many(env, labels, rewards)
    return rewards


# ---------------------------------------------------------------------------
# single-source adversary
# ---------------------------------------------------------------------------


def _features(source: str, rec: dict[str, Any] | None, jd: JobDescription) -> list[float]:
    rec = rec or {}
    if source in rubric.PLATFORMS:
        x = (rec.get("platforms") or {}).get(source)
        if source == "github":
            return [math.log1p(x["stars_received"]), x["repos"], math.log1p(x["commit_streak_days"]),
                    math.log1p(x["contributions_last_year"]), x["years_active"]]
        if source == "leetcode":
            return [x["contest_rating"], x["hard"], x["global_rank_percentile"]]
        m = x.get("medals") or {}
        return [x["best_finish_percentile"], rubric.KAGGLE_RANK_VALUE.get(x["rank"], 0.0),
                3 * m.get("gold", 0) + 2 * m.get("silver", 0) + m.get("bronze", 0)]
    x = (rec.get("resume_sections") or {}).get(source)
    if source == "experience":
        y = x["years_experience"]
        return [y, math.log1p(y)]
    have = {k.lower() for k in x.get("skills") or []}
    return [sum(k.lower() in have for k in jd.required_skills), sum(k.lower() in have for k in jd.nice_to_have)]


@functools.lru_cache(maxsize=None)
def fit_single_source(source: str, role: str) -> tuple[float, ...]:
    """Least-squares fit from one source's fields to the rubric total (training seeds)."""
    from recruitenv.profile_factory import ProfileFactory

    xs, ys = [], []
    factory = ProfileFactory()
    for seed in TRAIN_SEEDS:
        if ROLES[seed % len(ROLES)] != role:
            continue
        ep = factory.generate_episode(seed, {"shortlist": 5, "hold": 8, "reject": 7})
        for c in ep.candidates:
            rec = rubric.reveal_record(c, [source])
            xs.append([1.0, *_features(source, rec, ep.jd)])
            ys.append(c.ground_truth_scores["total"])
    coef, *_ = np.linalg.lstsq(np.array(xs), np.array(ys), rcond=None)
    return tuple(float(c) for c in coef)


def single_source(env: RecruitmentEnvironment, source: str, model: str = "estimate") -> list[float]:
    """Reveal one source for every candidate, rank by a one-source model, fill the quotas."""
    rewards: list[float] = []
    s = env.state()
    for c in s.candidates_summary:
        if source not in _revealed(env, c["id"]) and _step(env, action_for(c["id"], source), rewards):
            return rewards
    s = env.state()
    jd = s.job_description
    if model == "estimate":
        est = {c["id"]: rubric.estimate(s.revealed_data.get(c["id"]), jd).total for c in s.candidates_summary}
    else:
        coef = fit_single_source(source, jd.role_type)
        est = {
            c["id"]: float(np.dot(coef, [1.0, *_features(source, s.revealed_data.get(c["id"]), jd)]))
            for c in s.candidates_summary
        }
    labels = fill_quotas(est, s.quotas, decide_all=_decide_all_needed(env))
    _decide_many(env, labels, rewards)
    return rewards


# ---------------------------------------------------------------------------
# deliberately biased control (fairness audit only)
# ---------------------------------------------------------------------------


def planted_biased(env: RecruitmentEnvironment, target_origin: str = "african", p: float = 0.5) -> list[float]:
    """Oracle decisions, but a shortlist with a *target_origin* name becomes hold with probability *p*.

    Deterministic per (seed, candidate id, displayed name), so a name-swap
    replay changes its decisions exactly when the name's group changes.
    """
    rewards: list[float] = []
    vs = env.verifier_state()
    for cid, pair in sorted(vs.conflicts.items()):
        for src in pair:
            if src not in _revealed(env, cid) and _step(env, action_for(cid, src), rewards):
                return rewards
    names = {c["id"]: c["name"] for c in env.state().candidates_summary}
    labels = {}
    for c in vs.candidates:
        label = c.ground_truth_label
        grp = group_of(names[c.id])
        if label == "shortlist" and grp is not None and grp[0] == target_origin:
            h = hashlib.sha256(f"{vs.seed}|{c.id}|{names[c.id]}".encode()).digest()
            if int.from_bytes(h[:8], "big") / 2**64 < p:
                label = "hold"
        labels[c.id] = label
    _decide_many(env, labels, rewards)
    return rewards


POLICIES: dict[str, Policy] = {
    "oracle": oracle,
    "rubric_heuristic": rubric_heuristic,
    "fixed_order": fixed_order,
    "random": random_quota,
    **{f"single_{s}": functools.partial(single_source, source=s, model="estimate") for s in rubric.SOURCES},
    **{f"lsq_{s}": functools.partial(single_source, source=s, model="lsq") for s in rubric.SOURCES},
    "planted_strong": functools.partial(planted_biased, p=0.5),
    "planted_mild": functools.partial(planted_biased, p=0.1),
}
