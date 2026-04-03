"""Core RecruitEnv environment: reset(), step(), state()."""

from __future__ import annotations

from env.models import Action, EpisodeState, Observation, Reward


class RecruitmentEnvironment:
    """OpenEnv-compliant RL environment for candidate pipeline triage.

    Deterministic: reset(seed=42) always produces identical state.
    """

    def __init__(self) -> None:
        self._state: EpisodeState | None = None

    def reset(self, task_id: str, seed: int = 42) -> Observation:
        """Reset the environment for a new episode.

        Generates a job description and candidate pool deterministically
        from the given seed, then returns the initial observation.
        """
        raise NotImplementedError

    def step(self, action: Action) -> tuple[Observation, Reward]:
        """Apply an agent action and return (observation, reward).

        Supported action types:
        - read_resume_section: reveal a section of a candidate's resume
        - check_platform: reveal platform-specific data (leetcode, github, kaggle)
        - score_dimension: record an agent-assigned score for a dimension
        - make_decision: terminal shortlist/reject decision for a candidate
        """
        raise NotImplementedError

    def state(self) -> EpisodeState:
        """Return the full internal state for serialisation / grading."""
        raise NotImplementedError
