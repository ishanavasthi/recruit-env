"""Scoring contract: a reference solution and mutant policies.

The environment has one score per episode. The terminal step's reward is the
task grader's score, the sum of all step rewards equals it, and ``/grader``
reports the same number. A reference solution (the oracle) must score near the
maximum, and every lazy or degenerate "mutant" policy must fail the task by a
wide margin on every seed.

The oracle reads ground truth through ``verifier_state()``, which is the
verifier's view and is never exposed to an agent.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from env.environment import RecruitmentEnvironment
from env.models import CheckPlatformAction, MakeDecisionAction, ReadResumeSectionAction
from tasks import GRADER_REGISTRY, TASK_REGISTRY

SEEDS = range(200)
TASKS = ["easy", "medium", "hard"]
ORACLE_MIN = 0.99
MIN_GAP = 0.4

Policy = Callable[[RecruitmentEnvironment], list[float]]


def _step(env: RecruitmentEnvironment, action, rewards: list[float]) -> bool:
    _, reward, done, _ = env.step(action)
    rewards.append(reward.step_reward)
    return done


def _idle_until_done(env: RecruitmentEnvironment, rewards: list[float]) -> None:
    """Burn the remaining budget by re-reading one resume section."""
    cid = env.state().candidates_summary[0]["id"]
    while not env.state().is_done:
        _step(env, ReadResumeSectionAction(candidate_id=cid, section="education"), rewards)


def _decide_all(env: RecruitmentEnvironment, decide: Callable[[str], str]) -> list[float]:
    rewards: list[float] = []
    for c in env.state().candidates_summary:
        if _step(env, MakeDecisionAction(candidate_id=c["id"], decision=decide(c["id"])), rewards):
            break
    if not env.state().is_done:
        _idle_until_done(env, rewards)
    return rewards


# --- reference solution -------------------------------------------------------


def oracle(env: RecruitmentEnvironment) -> list[float]:
    truth = {c.id: c.ground_truth_label for c in env.verifier_state().candidates}
    return _decide_all(env, lambda cid: truth[cid])


# --- mutants ------------------------------------------------------------------


def do_nothing(env: RecruitmentEnvironment) -> list[float]:
    rewards: list[float] = []
    _idle_until_done(env, rewards)
    return rewards


def all_hold(env: RecruitmentEnvironment) -> list[float]:
    return _decide_all(env, lambda cid: "hold")


def all_shortlist(env: RecruitmentEnvironment) -> list[float]:
    return _decide_all(env, lambda cid: "shortlist")


def all_reject(env: RecruitmentEnvironment) -> list[float]:
    return _decide_all(env, lambda cid: "reject")


def decide_one_then_idle(env: RecruitmentEnvironment) -> list[float]:
    """Decide one true reject correctly, then let the budget run out."""
    reject = next(c for c in env.verifier_state().candidates if c.ground_truth_label == "reject")
    rewards: list[float] = []
    _step(env, MakeDecisionAction(candidate_id=reject.id, decision="reject"), rewards)
    _idle_until_done(env, rewards)
    return rewards


def gather_then_idle(env: RecruitmentEnvironment) -> list[float]:
    """Reveal as much as the budget allows and never decide."""
    rewards: list[float] = []
    for c in env.state().candidates_summary:
        for platform in ("github", "leetcode", "kaggle"):
            if _step(env, CheckPlatformAction(candidate_id=c["id"], platform=platform), rewards):
                return rewards
        for section in ("education", "experience", "skills"):
            if _step(env, ReadResumeSectionAction(candidate_id=c["id"], section=section), rewards):
                return rewards
    _idle_until_done(env, rewards)
    return rewards


MUTANTS: dict[str, Policy] = {
    "do_nothing": do_nothing,
    "all_hold": all_hold,
    "all_shortlist": all_shortlist,
    "all_reject": all_reject,
    "decide_one_then_idle": decide_one_then_idle,
    "gather_then_idle": gather_then_idle,
}


def _play(task: str, seed: int, policy: Policy) -> tuple[float, list[float], RecruitmentEnvironment]:
    env = RecruitmentEnvironment()
    env.reset(task, seed)
    rewards = policy(env)
    final = env.state()
    assert final.is_done
    assert final.episode_score is not None
    return final.episode_score, rewards, env


@pytest.fixture(scope="module")
def oracle_scores() -> dict[tuple[str, int], float]:
    return {(t, s): _play(t, s, oracle)[0] for t in TASKS for s in SEEDS}


@pytest.mark.parametrize("task", TASKS)
def test_oracle_scores_near_maximum(task: str, oracle_scores) -> None:
    low = min(oracle_scores[(task, s)] for s in SEEDS)
    assert low >= ORACLE_MIN, f"{task}: oracle scored {low:.3f} on some seed"


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_mutant_fails_by_wide_margin(task: str, mutant: str, oracle_scores) -> None:
    threshold = TASK_REGISTRY[task].success_threshold
    for seed in SEEDS:
        score, _, _ = _play(task, seed, MUTANTS[mutant])
        assert score < threshold, f"{task}/{mutant} seed={seed}: {score:.3f} passes {threshold}"
        gap = oracle_scores[(task, seed)] - score
        assert gap >= MIN_GAP, f"{task}/{mutant} seed={seed}: only {gap:.3f} below the oracle"


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("policy", ["oracle", *sorted(MUTANTS)])
def test_one_score(task: str, policy: str) -> None:
    """Step rewards sum to the episode score, which is the task grader's score."""
    fn = oracle if policy == "oracle" else MUTANTS[policy]
    for seed in range(20):
        initial_obs = RecruitmentEnvironment().reset(task, seed)
        score, rewards, env = _play(task, seed, fn)
        assert sum(rewards) == pytest.approx(score, abs=1e-9)
        assert all(r == 0.0 for r in rewards[:-1]), "only the terminal step carries reward"
        regraded = GRADER_REGISTRY[task].grade(initial_obs, env.verifier_state())
        assert score == pytest.approx(regraded, abs=1e-6)
        assert env.grade_report()["score"] == score
