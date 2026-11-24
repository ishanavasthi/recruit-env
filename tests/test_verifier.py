"""The verifier's named checks, the twin gate and the score formula (#7)."""

from __future__ import annotations

import pytest

from recruitenv import policies
from recruitenv.environment import RecruitmentEnvironment
from recruitenv.models import MakeDecisionAction
from recruitenv.tasks import GRADER_REGISTRY, TASK_REGISTRY

TASKS = ["easy", "medium", "hard"]
FORMULA_POLICIES = ["oracle", "rubric_heuristic", "fixed_order", "random", "single_skills", "planted_strong"]


def _play(task: str, seed: int, name: str) -> RecruitmentEnvironment:
    env = RecruitmentEnvironment()
    env.reset(task, seed)
    policies.POLICIES[name](env)
    assert env.state().is_done
    return env


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("name", FORMULA_POLICIES)
def test_score_formula_invariant(task: str, name: str) -> None:
    for seed in range(30):
        report = _play(task, seed, name).grade_report()
        before = sum(c["earned"] for c in report["checks"])
        assert report["score_before_gates"] == pytest.approx(before, abs=1e-6)
        caps = [g["cap"] for g in report["gates"] if not g["passed"]]
        expected = max(0.001, min(0.999, min([before, *caps])))
        assert report["score"] == pytest.approx(expected, abs=1e-6)
        for c in report["checks"]:
            assert 0.0 <= c["earned"] <= c["weight"] + 1e-12
            assert c["status"] in {"passed", "partial", "failed", "blocked"}


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("name", ["oracle", "rubric_heuristic", "fixed_order", "single_github"])
def test_clean_policies_never_fail_the_gate(task: str, name: str) -> None:
    for seed in range(100):
        report = _play(task, seed, name).grade_report()
        assert all(g["passed"] for g in report["gates"]), f"{task}/{name} seed={seed}"


def _twin_env(task: str = "easy", seed: int = 3) -> tuple[RecruitmentEnvironment, str, str]:
    env = RecruitmentEnvironment()
    env.reset(task, seed)
    a, b = env.verifier_state().twins[0]["pair"]
    return env, a, b


def _finish_with_truth(env: RecruitmentEnvironment) -> None:
    truth = {c.id: c.ground_truth_label for c in env.verifier_state().candidates}
    for cid, label in truth.items():
        if env.state().is_done:
            return
        if cid not in env.state().decisions_made:
            env.step(MakeDecisionAction(candidate_id=cid, decision=label))


def test_twin_gate_like_for_like_flip_caps() -> None:
    env, a, b = _twin_env()
    env.step(MakeDecisionAction(candidate_id=a, decision="shortlist"))
    env.step(MakeDecisionAction(candidate_id=b, decision="reject"))
    _finish_with_truth(env)
    report = env.grade_report()
    gate = report["gates"][0]
    assert gate["name"] == "twin_consistency" and gate["passed"] is False
    assert report["score"] <= TASK_REGISTRY["easy"].success_threshold - 0.05 + 1e-9


def test_twin_gate_ignores_information_difference() -> None:
    env, a, b = _twin_env()
    env.step(policies.action_for(a, "github"))
    env.step(MakeDecisionAction(candidate_id=a, decision="shortlist"))
    env.step(MakeDecisionAction(candidate_id=b, decision="reject"))
    _finish_with_truth(env)
    assert env.grade_report()["gates"][0]["passed"] is True


def test_twin_gate_skips_undecided_twin_on_easy() -> None:
    env, a, b = _twin_env("easy", 3)
    env.step(MakeDecisionAction(candidate_id=a, decision="shortlist"))
    policies.idle_until_done(env, [])
    assert env.grade_report()["gates"][0]["passed"] is True


def test_twin_gate_counts_undecided_as_hold_on_hard() -> None:
    """On hard an undecided twin is not shortlisted, so shortlisting only one is a flip."""
    env, a, b = _twin_env("hard", 3)
    env.step(MakeDecisionAction(candidate_id=a, decision="shortlist"))
    policies.idle_until_done(env, [])
    report = env.grade_report()
    assert report["gates"][0]["passed"] is False
    assert report["score"] <= TASK_REGISTRY["hard"].success_threshold - 0.05 + 1e-9


def test_medium_investigated_blocked_when_wrong() -> None:
    env = RecruitmentEnvironment()
    env.reset("medium", 0)
    vs = env.verifier_state()
    cid, pair = sorted(vs.conflicts.items())[0]
    for src in pair:
        env.step(policies.action_for(cid, src))
    wrong = "reject" if vs.candidates[[c.id for c in vs.candidates].index(cid)].ground_truth_label != "reject" else "shortlist"
    env.step(MakeDecisionAction(candidate_id=cid, decision=wrong))
    _finish_with_truth(env)
    checks = {c["name"]: c for c in env.grade_report()["checks"]}
    assert checks[f"conflicts_investigated:{cid}"]["status"] == "blocked"
    assert checks[f"conflicts_investigated:{cid}"]["earned"] == 0.0


def test_medium_uninvestigated_conflict_loses_its_share() -> None:
    env = RecruitmentEnvironment()
    env.reset("medium", 0)
    _finish_with_truth(env)  # correct decisions, nothing revealed
    report = env.grade_report()
    inv = [c for c in report["checks"] if c["name"].startswith("conflicts_investigated")]
    assert len(inv) == 2 and all(c["status"] == "failed" for c in inv)
    assert report["score"] == pytest.approx(0.8, abs=1e-6)


def test_hard_no_efficiency_bonus() -> None:
    scores = []
    for idle in (0, 20):
        env = RecruitmentEnvironment()
        env.reset("hard", 5)
        for _ in range(idle):
            env.step(policies.action_for("candidate_001", "github"))
        truth = [c.id for c in env.verifier_state().candidates if c.ground_truth_label == "shortlist"]
        for cid in truth:
            env.step(MakeDecisionAction(candidate_id=cid, decision="shortlist"))
        policies.idle_until_done(env, [])
        scores.append(env.grade_report()["score"])
    assert scores[0] == scores[1] == 0.999


def test_grader_registry_threshold_matches_task() -> None:
    for task, grader in GRADER_REGISTRY.items():
        assert grader.threshold == TASK_REGISTRY[task].success_threshold


def test_twin_gate_on_hard_treats_reject_and_undecided_alike() -> None:
    env, a, b = _twin_env("hard", 3)
    env.step(MakeDecisionAction(candidate_id=a, decision="reject"))
    policies.idle_until_done(env, [])
    assert env.grade_report()["gates"][0]["passed"] is True
