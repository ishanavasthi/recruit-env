"""Tests for the core environment (reset, step, determinism)."""

import pytest

from env.environment import RecruitmentEnvironment
from env.models import (
    CheckPlatformAction,
    MakeDecisionAction,
    ReadResumeSectionAction,
)


@pytest.fixture()
def env() -> RecruitmentEnvironment:
    return RecruitmentEnvironment()


# -----------------------------------------------------------------------
# reset
# -----------------------------------------------------------------------


class TestReset:
    def test_reset_deterministic(self, env: RecruitmentEnvironment) -> None:
        """reset('easy', 42) called twice yields identical observations."""
        obs1 = env.reset("easy", seed=42)
        obs2 = env.reset("easy", seed=42)
        assert obs1.model_dump() == obs2.model_dump()

    def test_reset_different_seeds(self, env: RecruitmentEnvironment) -> None:
        """Different seeds produce different observations."""
        obs42 = env.reset("easy", seed=42)
        obs99 = env.reset("easy", seed=99)
        # Candidate names should differ because different seed shuffles the pool
        names42 = {c["name"] for c in obs42.candidates_summary}
        names99 = {c["name"] for c in obs99.candidates_summary}
        assert names42 != names99

    def test_reset_returns_correct_shape(self, env: RecruitmentEnvironment) -> None:
        """Initial observation has expected field values."""
        obs = env.reset("easy", seed=42)
        assert obs.task_id == "easy"
        assert obs.step_number == 0
        assert obs.steps_remaining == 40
        assert len(obs.candidates_summary) == 10
        assert obs.revealed_data == {}
        assert obs.decisions_made == {}
        assert obs.done is False

    def test_reset_invalid_task(self, env: RecruitmentEnvironment) -> None:
        """Unknown task_id raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset("nonexistent")


# -----------------------------------------------------------------------
# step — actions
# -----------------------------------------------------------------------


class TestStep:
    def test_step_read_resume(self, env: RecruitmentEnvironment) -> None:
        """ReadResumeSectionAction reveals data; the reward is sparse (0.0)."""
        obs = env.reset("easy", seed=42)
        cid = obs.candidates_summary[0]["id"]

        action = ReadResumeSectionAction(candidate_id=cid, section="education")
        obs, reward, done, info = env.step(action)

        assert reward.step_reward == 0.0
        assert cid in obs.revealed_data
        assert "education" in obs.revealed_data[cid]["resume_sections"]
        assert info.get("section_found") is True
        assert done is False

    def test_step_check_platform_github(self, env: RecruitmentEnvironment) -> None:
        """CheckPlatformAction for GitHub returns stats with expected keys."""
        obs = env.reset("easy", seed=42)
        cid = obs.candidates_summary[0]["id"]

        action = CheckPlatformAction(candidate_id=cid, platform="github")
        obs, reward, done, info = env.step(action)

        assert reward.step_reward == 0.0
        assert "github" in obs.revealed_data[cid]["platforms"]
        gh = info["data"]
        for key in ("repos", "top_languages", "commit_streak_days",
                     "stars_received", "contributions_last_year"):
            assert key in gh

    def test_step_repeat_action_reveals_nothing(
        self, env: RecruitmentEnvironment
    ) -> None:
        """Repeating the same read reveals nothing new."""
        obs = env.reset("easy", seed=42)
        cid = obs.candidates_summary[0]["id"]
        action = ReadResumeSectionAction(candidate_id=cid, section="education")

        env.step(action)
        _, reward2, _, info = env.step(action)

        assert reward2.step_reward == 0.0
        assert info["already_revealed"] is True

    def test_step_make_decision(self, env: RecruitmentEnvironment) -> None:
        """MakeDecisionAction records the decision with 0.0 step reward."""
        obs = env.reset("easy", seed=42)
        cid = obs.candidates_summary[0]["id"]

        action = MakeDecisionAction(candidate_id=cid, decision="hold")
        obs, reward, done, info = env.step(action)

        assert reward.step_reward == 0.0
        assert obs.decisions_made[cid] == "hold"
        assert info.get("decision_recorded") is True


# -----------------------------------------------------------------------
# terminal conditions
# -----------------------------------------------------------------------


class TestTermination:
    def test_episode_terminates_on_all_decided(
        self, env: RecruitmentEnvironment
    ) -> None:
        """Episode ends when every candidate has a decision."""
        obs = env.reset("easy", seed=42)
        state = env.verifier_state()
        gt = {c.id: c.ground_truth_label for c in state.candidates}

        for cid in list(gt.keys()):
            obs, _, done, info = env.step(
                MakeDecisionAction(candidate_id=cid, decision=gt[cid])
            )

        assert done is True
        assert info["termination_reason"] == "all_decided"

    def test_episode_terminates_on_budget(
        self, env: RecruitmentEnvironment
    ) -> None:
        """Episode ends when step budget is exhausted."""
        obs = env.reset("easy", seed=42)
        cids = [c["id"] for c in obs.candidates_summary]

        done = False
        for i in range(40):
            cid = cids[i % len(cids)]
            obs, _, done, info = env.step(
                ReadResumeSectionAction(candidate_id=cid, section="education")
            )
            if done:
                break

        assert done is True
        assert info["termination_reason"] == "budget_exhausted"


# -----------------------------------------------------------------------
# error handling
# -----------------------------------------------------------------------


class TestErrors:
    def test_cannot_step_without_reset(self) -> None:
        """Stepping on a fresh environment raises RuntimeError."""
        env = RecruitmentEnvironment()
        with pytest.raises(RuntimeError, match="No active episode"):
            env.step(
                ReadResumeSectionAction(
                    candidate_id="candidate_001", section="education"
                )
            )

    def test_duplicate_decision_raises(
        self, env: RecruitmentEnvironment
    ) -> None:
        """Deciding the same candidate twice raises ValueError."""
        obs = env.reset("easy", seed=42)
        cid = obs.candidates_summary[0]["id"]

        env.step(MakeDecisionAction(candidate_id=cid, decision="hold"))
        with pytest.raises(ValueError, match="already has a decision"):
            env.step(MakeDecisionAction(candidate_id=cid, decision="reject"))

    def test_invalid_candidate_raises(
        self, env: RecruitmentEnvironment
    ) -> None:
        """Unknown candidate_id raises ValueError."""
        env.reset("easy", seed=42)
        with pytest.raises(ValueError, match="Unknown candidate_id"):
            env.step(
                ReadResumeSectionAction(
                    candidate_id="does_not_exist", section="skills"
                )
            )

    def test_step_after_done_raises(
        self, env: RecruitmentEnvironment
    ) -> None:
        """Stepping after episode is done raises RuntimeError."""
        obs = env.reset("easy", seed=42)
        state = env.verifier_state()
        for c in state.candidates:
            env.step(MakeDecisionAction(candidate_id=c.id, decision="hold"))

        with pytest.raises(RuntimeError, match="already done"):
            env.step(
                ReadResumeSectionAction(
                    candidate_id=state.candidates[0].id, section="education"
                )
            )


# -----------------------------------------------------------------------
# state()
# -----------------------------------------------------------------------


class TestState:
    def test_state_returns_deep_copy(self, env: RecruitmentEnvironment) -> None:
        """Mutating a returned state does not affect the environment."""
        env.reset("easy", seed=42)
        for get in (env.state, env.verifier_state):
            s1 = get()
            s1.decisions_made["injected"] = "hack"
            assert "injected" not in get().decisions_made

    def test_state_without_reset_raises(self) -> None:
        """state() on a fresh environment raises ValueError."""
        env = RecruitmentEnvironment()
        with pytest.raises(ValueError, match="No active episode"):
            env.state()


# -----------------------------------------------------------------------
# the observation is the agent's whole view
# -----------------------------------------------------------------------


class TestObservationIsSufficient:
    def test_revealed_values_are_in_observation(self, env: RecruitmentEnvironment) -> None:
        obs = env.reset("easy", seed=42)
        cid = obs.candidates_summary[0]["id"]

        obs, _, _, info = env.step(CheckPlatformAction(candidate_id=cid, platform="leetcode"))
        assert obs.revealed_data[cid]["platforms"]["leetcode"] == info["data"]

        obs, _, _, info = env.step(ReadResumeSectionAction(candidate_id=cid, section="experience"))
        assert obs.revealed_data[cid]["resume_sections"]["experience"] == info["data"]

    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_observation_only_policy_plays_every_task(self, task: str) -> None:
        """A policy that reads nothing but Observation can finish any task."""
        env = RecruitmentEnvironment()
        obs = env.reset(task, seed=11)
        done = False
        for c in obs.candidates_summary:
            cid = c["id"]
            obs, _, done, _ = env.step(CheckPlatformAction(candidate_id=cid, platform="github"))
            if done:
                break
            stars = obs.revealed_data[cid]["platforms"]["github"]["stars_received"]
            decision = "shortlist" if stars > 200 else "reject" if stars < 20 else "hold"
            obs, reward, done, _ = env.step(MakeDecisionAction(candidate_id=cid, decision=decision))
            if done:
                break
        while not done:
            cid = obs.candidates_summary[0]["id"]
            obs, reward, done, _ = env.step(ReadResumeSectionAction(candidate_id=cid, section="skills"))
        assert 0.0 <= reward.cumulative_reward <= 1.0
        assert env.state().episode_score == reward.cumulative_reward

    def test_state_hides_ground_truth(self, env: RecruitmentEnvironment) -> None:
        env.reset("hard", seed=1)
        dumped = env.state().model_dump()
        assert "candidates" not in dumped
        assert "ground_truth" not in str(dumped)
        assert env.verifier_state().candidates[0].ground_truth_label in {"shortlist", "hold", "reject"}


class TestNameBlind:
    def test_blind_episode_keeps_twins_and_signals(self) -> None:
        named, blind = RecruitmentEnvironment(), RecruitmentEnvironment()
        named.reset("medium", 9)
        obs = blind.reset("medium", 9, name_blind=True)
        assert [c["name"] for c in obs.candidates_summary] == [f"Candidate {i:02d}" for i in range(1, 9)]
        assert blind.state().name_blind is True
        vb, vn = blind.verifier_state(), named.verifier_state()
        assert vb.twins == vn.twins and vb.conflicts == vn.conflicts
        assert [c.model_dump() for c in vb.candidates] == [c.model_dump() for c in vn.candidates]

    def test_decision_context_records_sources_seen(self) -> None:
        env = RecruitmentEnvironment()
        obs = env.reset("easy", 1)
        cid = obs.candidates_summary[0]["id"]
        env.step(CheckPlatformAction(candidate_id=cid, platform="kaggle"))
        env.step(ReadResumeSectionAction(candidate_id=cid, section="education"))
        env.step(ReadResumeSectionAction(candidate_id=cid, section="skills"))
        env.step(MakeDecisionAction(candidate_id=cid, decision="hold"))
        assert env.verifier_state().decision_context[cid] == ["kaggle", "skills"]
