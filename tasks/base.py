"""TaskDefinition base class for all RecruitEnv tasks."""

from __future__ import annotations

from pydantic import BaseModel


class TaskDefinition(BaseModel):
    """Base class defining a task (scenario) for the environment.

    Each concrete task specifies the number of candidates, step budget,
    selection quota, and a human-readable description.
    """

    task_id: str
    description: str
    num_candidates: int
    """How many candidates to generate."""
    select_quota: int
    """How many candidates the agent should shortlist."""
    step_budget: int
    """Maximum number of steps before the episode is force-ended."""
    seed: int = 42
