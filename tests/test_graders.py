"""Tests for task graders."""

import numpy as np
import pytest

from env.environment import RecruitmentEnvironment
from env.models import MakeDecisionAction, ReadResumeSectionAction
from tasks import GRADER_REGISTRY
from tasks.easy_task import EasyGrader
from tasks.hard_task import HardGrader


@pytest.fixture()
def env() -> RecruitmentEnvironment:
    return RecruitmentEnvironment()


def _decide_all(env: RecruitmentEnvironment, decision_map: dict[str, str]) -> None:
    """Issue MakeDecisionAction for every candidate in *decision_map*."""
    for cid, decision in decision_map.items():
        env.step(MakeDecisionAction(candidate_id=cid, decision=decision))


# -----------------------------------------------------------------------
# EasyGrader
# -----------------------------------------------------------------------


class TestEasyGrader:
    def test_easy_grader_perfect_score(self, env: RecruitmentEnvironment) -> None:
        """All correct decisions → score >= 0.95 (includes bonus)."""
        obs = env.reset("easy", seed=42)
        state = env.state()
        gt = {c.id: c.ground_truth_label for c in state.candidates}
        _decide_all(env, gt)

        final = env.state()
        grader = EasyGrader()
        score = grader.grade(obs, final)
        assert score >= 0.95

    def test_easy_grader_all_wrong(self, env: RecruitmentEnvironment) -> None:
        """All wrong decisions → score <= 0.2."""
        obs = env.reset("easy", seed=42)
        state = env.state()
        gt = {c.id: c.ground_truth_label for c in state.candidates}
        wrong = {}
        for cid, label in gt.items():
            if label == "shortlist":
                wrong[cid] = "reject"
            elif label == "reject":
                wrong[cid] = "shortlist"
            else:
                wrong[cid] = "reject"
        _decide_all(env, wrong)

        final = env.state()
        grader = EasyGrader()
        score = grader.grade(obs, final)
        assert score <= 0.2

    def test_easy_grader_partial(self, env: RecruitmentEnvironment) -> None:
        """~Half correct → score in 0.3-0.7 range."""
        obs = env.reset("easy", seed=42)
        state = env.state()
        gt = {c.id: c.ground_truth_label for c in state.candidates}

        # First half correct, second half wrong
        cids = list(gt.keys())
        mixed: dict[str, str] = {}
        for i, cid in enumerate(cids):
            if i < len(cids) // 2:
                mixed[cid] = gt[cid]
            else:
                mixed[cid] = "hold"  # may or may not be correct
        _decide_all(env, mixed)

        final = env.state()
        grader = EasyGrader()
        score = grader.grade(obs, final)
        assert 0.3 <= score <= 0.7


class TestGraderDeterminism:
    def test_grader_deterministic(self, env: RecruitmentEnvironment) -> None:
        """Same inputs produce identical output across 3 runs."""
        grader = EasyGrader()
        scores = []
        for _ in range(3):
            obs = env.reset("easy", seed=42)
            state = env.state()
            gt = {c.id: c.ground_truth_label for c in state.candidates}
            _decide_all(env, gt)
            final = env.state()
            scores.append(grader.grade(obs, final))

        assert scores[0] == scores[1] == scores[2]


class TestGraderRange:
    def test_grader_range(self, env: RecruitmentEnvironment) -> None:
        """100 random decision sets all produce scores in [0.0, 1.0]."""
        grader = EasyGrader()
        rng = np.random.default_rng(12345)
        choices = ["shortlist", "hold", "reject"]

        for i in range(100):
            obs = env.reset("easy", seed=i)
            state = env.state()
            random_decisions = {
                c.id: choices[int(rng.integers(0, 3))]
                for c in state.candidates
            }
            _decide_all(env, random_decisions)
            final = env.state()
            score = grader.grade(obs, final)
            assert 0.0 <= score <= 1.0, f"seed={i}: score {score} out of range"


# -----------------------------------------------------------------------
# HardGrader — F1-based
# -----------------------------------------------------------------------


class TestHardGrader:
    def test_hard_grader_uses_f1(self, env: RecruitmentEnvironment) -> None:
        """Shortlist-all has recall=1.0 but low precision → moderate score.
        Shortlist-none has F1=0 → near-zero score.
        This validates the grader uses precision/recall (F1)."""
        grader = HardGrader()

        # Shortlist everyone
        obs_all = env.reset("hard", seed=42)
        state_all = env.state()
        for c in state_all.candidates:
            env.step(MakeDecisionAction(candidate_id=c.id, decision="shortlist"))
        final_all = env.state()
        score_all = grader.grade(obs_all, final_all)

        # Shortlist no one (all hold)
        obs_none = env.reset("hard", seed=42)
        state_none = env.state()
        for c in state_none.candidates:
            env.step(MakeDecisionAction(candidate_id=c.id, decision="hold"))
        final_none = env.state()
        score_none = grader.grade(obs_none, final_none)

        # Perfect shortlist only
        obs_perf = env.reset("hard", seed=42)
        state_perf = env.state()
        gt = {c.id: c.ground_truth_label for c in state_perf.candidates}
        for cid, label in gt.items():
            decision = "shortlist" if label == "shortlist" else "reject"
            env.step(MakeDecisionAction(candidate_id=cid, decision=decision))
        final_perf = env.state()
        score_perf = grader.grade(obs_perf, final_perf)

        # F1 assertions:
        # shortlist-all: precision = 5/20 = 0.25, recall = 1.0 → F1 ≈ 0.40
        assert 0.3 < score_all < 0.6, f"shortlist-all: {score_all}"
        # shortlist-none: F1 = 0.0
        assert score_none < 0.15, f"shortlist-none: {score_none}"
        # perfect: F1 = 1.0 (best)
        assert score_perf > score_all
        assert score_perf > 0.95, f"perfect: {score_perf}"

    def test_hard_grader_undecided_auto_hold(
        self, env: RecruitmentEnvironment
    ) -> None:
        """Undecided candidates are graded as 'hold'."""
        grader = HardGrader()

        # Exhaust budget without deciding anyone
        obs = env.reset("hard", seed=42)
        cids = [c["id"] for c in obs.candidates_summary]
        for i in range(30):
            cid = cids[i % len(cids)]
            obs, _, done, _ = env.step(
                ReadResumeSectionAction(candidate_id=cid, section="education")
            )
            if done:
                break

        final = env.state()
        score = grader.grade(obs, final)
        # No shortlists → F1 = 0.0
        assert score < 0.1
