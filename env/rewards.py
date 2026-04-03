"""Reward function logic for the RecruitEnv environment.

All graders return float in [0.0, 1.0] and are fully deterministic.
"""

from __future__ import annotations

from env.models import EpisodeState, Reward


def compute_reward(state: EpisodeState) -> Reward:
    """Compute the overall reward for the current episode state.

    Compares agent decisions against ground-truth labels embedded at
    profile generation time.  Returns a Reward with value in [0.0, 1.0]
    and an optional per-component breakdown.
    """
    raise NotImplementedError


def accuracy_score(state: EpisodeState) -> float:
    """Fraction of decisions matching ground truth.  Returns float in [0, 1]."""
    raise NotImplementedError


def efficiency_score(state: EpisodeState) -> float:
    """Reward for using fewer steps.  Returns float in [0, 1]."""
    raise NotImplementedError
