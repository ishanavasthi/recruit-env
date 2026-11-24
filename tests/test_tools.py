"""recruitenv.tools.RecruitToolEnv: the TRL environment_factory contract."""

from __future__ import annotations

import inspect

import pytest

from recruitenv.environment import RecruitmentEnvironment
from recruitenv.tools import RecruitToolEnv

TOOLS = {"check_github", "check_leetcode", "check_kaggle", "read_resume", "shortlist", "hold", "reject"}


def test_contract_shape():
    env = RecruitToolEnv()                         # no-argument constructor
    public = {n for n, m in inspect.getmembers(env, inspect.ismethod) if not n.startswith("_")}
    assert public - {"reset", "get_reward"} == TOOLS
    for name in TOOLS:
        fn = getattr(env, name)
        sig = inspect.signature(fn)
        assert all(p.annotation is not inspect.Parameter.empty for p in sig.parameters.values()), name
        assert "Args:" in (fn.__doc__ or "") and "Returns:" in fn.__doc__, name


def test_tool_episode_matches_core_score():
    ref = RecruitmentEnvironment()
    ref.reset("easy", 8)
    truth = {c.id: c.ground_truth_label for c in ref.verifier_state().candidates}
    env = RecruitToolEnv()
    intro = env.reset(task="easy", seed=8, prompt="ignored dataset column")
    assert "Quotas" in intro and "candidate_001" in intro
    first = next(iter(truth))
    assert "Steps remaining" in env.check_kaggle(first)
    assert "years_experience" in env.read_resume(first, "experience")
    out = ""
    for cid, label in truth.items():
        out = getattr(env, label)(cid)
    assert out.startswith("Episode over")
    assert env.get_reward() >= 0.99


def test_illegal_actions_raise_and_reward_waits_for_the_end():
    env = RecruitToolEnv()
    env.reset(task="easy", seed=1)
    assert env.get_reward() == 0.0
    env.hold("candidate_001")
    with pytest.raises(ValueError, match="already has a decision"):
        env.reject("candidate_001")
    with pytest.raises(ValueError):
        env.check_github("nobody")


def test_reset_starts_fresh():
    env = RecruitToolEnv()
    env.reset(task="easy", seed=1)
    for i in range(1, 11):
        env.hold(f"candidate_{i:03d}")
    assert env.done and env.get_reward() > 0
    env.reset(task="easy", seed=2)
    assert not env.done and env.get_reward() == 0.0
