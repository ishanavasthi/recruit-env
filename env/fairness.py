"""Fairness penalty checker.

Returns a penalty in [0.0, 1.0] that is subtracted from the reward to
discourage biased shortlisting patterns.  Deterministic.
"""

from __future__ import annotations

from env.models import State


def fairness_penalty(state: State) -> float:
    """Compute a fairness penalty for the agent's decisions.

    Checks whether shortlisting rates are roughly equal across candidate
    tiers (no systematic bias toward/against a tier).

    Returns a float in [0.0, 1.0] where 0.0 means no penalty (fair)
    and 1.0 means maximum penalty (completely unfair).
    """
    raise NotImplementedError
