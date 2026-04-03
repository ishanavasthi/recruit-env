"""Reward function logic for the RecruitEnv environment.

All graders return float in [0.0, 1.0] and are fully deterministic.
"""

from __future__ import annotations

from env.models import Reward, State


def compute_reward(state: State) -> Reward:
    """Compute the overall reward for the current episode state.

    Compares agent decisions against ground-truth labels embedded at
    profile generation time.  Returns a Reward with value in [0.0, 1.0]
    and an optional per-component breakdown.
    """
    raise NotImplementedError


def accuracy_score(state: State) -> float:
    """Fraction of decisions matching ground truth.  Returns float in [0, 1]."""
    raise NotImplementedError


def efficiency_score(state: State) -> float:
    """Reward for using fewer steps.  Returns float in [0, 1]."""
    raise NotImplementedError
