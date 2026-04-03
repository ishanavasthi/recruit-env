"""TaskConfig and BaseGrader — shared foundations for all tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from env.models import EpisodeState, Observation


@dataclass(frozen=True)
class TaskConfig:
    """Immutable configuration for a single task scenario."""

    id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int
    candidate_count: int
    label_distribution: dict[str, int] = field(default_factory=dict)
    success_threshold: float = 0.70
    role_type: str = "backend_dev"


class BaseGrader(ABC):
    """Abstract grader — every task must provide one.

    ``grade()`` receives the initial observation and the final episode state
    and returns a score in [0.0, 1.0].
    """

    @abstractmethod
    def grade(self, initial_obs: Observation, final_state: EpisodeState) -> float:
        """Return a score in [0.0, 1.0] for the completed episode.

        Must be **deterministic**: same inputs always produce same output.
        """
        ...
