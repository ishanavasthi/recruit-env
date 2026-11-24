"""Transition logs (#8): schema, exact replay, and refusal on mismatch."""

from __future__ import annotations

import json

import pytest

from recruitenv import policies
from recruitenv.environment import RecruitmentEnvironment
from recruitenv.models import CheckPlatformAction, MakeDecisionAction
from recruitenv.trajectory import LoggedEnvironment, ReplayError, read_log, replay
from scripts.replay import main as replay_main


def _run(tmp_path, policy, task="medium", seed=3, **reset_kw):
    path = tmp_path / f"{policy}-{task}-{seed}.jsonl"
    env = LoggedEnvironment(RecruitmentEnvironment(), path, agent=policy)
    env.reset(task, seed, **reset_kw)
    policies.POLICIES[policy](env)
    env.close()
    return path


def test_log_schema(tmp_path):
    path = _run(tmp_path, "rubric_heuristic", "hard", 5)
    recs = read_log(path)
    head, steps, final = recs[0], recs[1:-1], recs[-1]
    assert head["kind"] == "header"
    for key in ("episode_id", "task", "seed", "name_seed", "name_blind",
                "generator_version", "package_version", "agent", "started_at"):
        assert key in head
    assert head["task"] == "hard" and head["seed"] == 5
    assert all(r["kind"] == "step" for r in steps)
    assert [r["step"] for r in steps if r["error"] is None] == list(range(1, len(steps) + 1))
    assert steps[-1]["done"] is True
    assert final["kind"] == "final"
    assert final["grade"]["score"] == steps[-1]["reward"]
    assert {"checks", "gates", "score_before_gates"} <= set(final["grade"])
    # a reveal records exactly what it revealed; a decision reveals nothing
    reveal = next(r for r in steps if r["action"]["type"] == "check_platform")
    assert isinstance(reveal["revealed"], dict) and reveal["revealed"]
    decision = next(r for r in steps if r["action"]["type"] == "make_decision")
    assert decision["revealed"] is None


@pytest.mark.parametrize("policy", ["oracle", "rubric_heuristic", "fixed_order", "random"])
@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_replay_reproduces(tmp_path, policy, task):
    path = _run(tmp_path, policy, task, seed=11)
    grade = replay(path)
    assert grade["score"] == read_log(path)[-1]["grade"]["score"]


def test_replay_with_name_controls(tmp_path):
    path = _run(tmp_path, "rubric_heuristic", "medium", 4, name_seed=2, name_blind=True)
    head = read_log(path)[0]
    assert head["name_seed"] == 2 and head["name_blind"] is True
    replay(path)


def test_rejected_action_is_logged_and_replayed(tmp_path):
    path = tmp_path / "rej.jsonl"
    env = LoggedEnvironment(RecruitmentEnvironment(), path, agent="test")
    obs = env.reset("easy", 1)
    cid = obs.candidates_summary[0]["id"]
    env.step(MakeDecisionAction(type="make_decision", candidate_id=cid, decision="hold"))
    with pytest.raises(ValueError):
        env.step(MakeDecisionAction(type="make_decision", candidate_id=cid, decision="reject"))
    env.step(CheckPlatformAction(type="check_platform", candidate_id=cid, platform="github"))
    env.close()
    steps = read_log(path)[1:]
    assert steps[1]["error"] and steps[1]["obs_sha256"] is None
    assert steps[1]["step"] == 1 and steps[2]["step"] == 2   # the rejection cost no step
    assert replay(path) == {}                                # unfinished episode reproduces


def _rewrite(path, fn):
    recs = read_log(path)
    fn(recs)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_replay_detects_tampering(tmp_path):
    path = _run(tmp_path, "rubric_heuristic", "easy", 2)

    def flip_hash(recs):
        recs[3]["obs_sha256"] = "0" * 64
    _rewrite(path, flip_hash)
    with pytest.raises(ReplayError, match="observation differs"):
        replay(path)


def test_replay_detects_changed_grade(tmp_path):
    path = _run(tmp_path, "rubric_heuristic", "easy", 2)

    def bump(recs):
        recs[-1]["grade"]["score"] = 0.5
    _rewrite(path, bump)
    with pytest.raises(ReplayError, match="final grade differs"):
        replay(path)


@pytest.mark.parametrize("field", ["generator_version", "package_version"])
def test_replay_refuses_version_mismatch(tmp_path, field):
    path = _run(tmp_path, "oracle", "easy", 2)

    def stale(recs):
        recs[0][field] = "0"
    _rewrite(path, stale)
    with pytest.raises(ReplayError, match="version"):
        replay(path)
    replay(path, check_versions=False)


def test_replay_cli(tmp_path, capsys):
    _run(tmp_path, "oracle", "easy", 1)
    _run(tmp_path, "fixed_order", "hard", 1)
    assert replay_main([str(tmp_path)]) == 0
    assert "2/2 logs reproduced" in capsys.readouterr().out
