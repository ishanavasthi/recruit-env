"""Tests for reward functions and fairness graders."""

import pytest

from env.rewards import accuracy_score, compute_reward, efficiency_score
from env.fairness import fairness_penalty


class TestAccuracyScore:
    """accuracy_score must return float in [0, 1]."""

    def test_perfect_accuracy(self) -> None:
        """All decisions match ground truth → 1.0."""
        pytest.skip("Not implemented yet")

    def test_zero_accuracy(self) -> None:
        """All decisions wrong → 0.0."""
        pytest.skip("Not implemented yet")


class TestEfficiencyScore:
    """efficiency_score must return float in [0, 1]."""

    def test_minimum_steps(self) -> None:
        """Using fewest possible steps → 1.0."""
        pytest.skip("Not implemented yet")


class TestFairnessPenalty:
    """fairness_penalty must return float in [0, 1]."""

    def test_fair_decisions(self) -> None:
        """Balanced shortlisting across tiers → 0.0 penalty."""
        pytest.skip("Not implemented yet")

    def test_biased_decisions(self) -> None:
        """Heavily biased shortlisting → high penalty."""
        pytest.skip("Not implemented yet")


class TestComputeReward:
    """compute_reward aggregates sub-scores correctly."""

    def test_reward_in_range(self) -> None:
        """Final reward is always in [0.0, 1.0]."""
        pytest.skip("Not implemented yet")
