"""Pydantic v2 models for the RecruitEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Possible action types an agent can take."""

    READ_RESUME_SECTION = "read_resume_section"
    CHECK_PLATFORM = "check_platform"
    SCORE_DIMENSION = "score_dimension"
    MAKE_DECISION = "make_decision"


class Decision(str, Enum):
    """Terminal decision for a candidate."""

    SHORTLIST = "shortlist"
    REJECT = "reject"


class Platform(str, Enum):
    """External platforms that can be checked."""

    LEETCODE = "leetcode"
    GITHUB = "github"
    KAGGLE = "kaggle"


class Tier(str, Enum):
    """Candidate seniority tier used for ground-truth generation."""

    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"


class CandidateProfile(BaseModel):
    """Full synthetic candidate profile with ground-truth label."""

    candidate_id: str
    name: str
    tier: Tier
    resume_sections: dict[str, str] = Field(default_factory=dict)
    platform_data: dict[Platform, dict[str, Any]] = Field(default_factory=dict)
    ground_truth_label: Decision
    """The correct decision, embedded at generation time."""


class JobDescription(BaseModel):
    """Synthetic job description generated per episode."""

    title: str
    required_skills: list[str] = Field(default_factory=list)
    min_tier: Tier = Tier.MID
    description: str = ""


class Action(BaseModel):
    """An action submitted by the agent at each step."""

    action_type: ActionType
    candidate_id: str
    section: str | None = None
    """Resume section name — required for READ_RESUME_SECTION."""
    platform: Platform | None = None
    """Platform name — required for CHECK_PLATFORM."""
    dimension: str | None = None
    """Scoring dimension — required for SCORE_DIMENSION."""
    score: float | None = None
    """Score value — required for SCORE_DIMENSION."""
    decision: Decision | None = None
    """Terminal decision — required for MAKE_DECISION."""


class Observation(BaseModel):
    """Observation returned to the agent after each step."""

    candidates_remaining: int
    """Number of candidates still awaiting a decision."""
    steps_remaining: int
    """Budget of steps left in the episode."""
    data: dict[str, Any] = Field(default_factory=dict)
    """Payload specific to the action that was just taken."""
    done: bool = False
    """Whether the episode has ended."""


class Reward(BaseModel):
    """Reward signal returned alongside an observation."""

    value: float = Field(ge=0.0, le=1.0)
    """Scalar reward in [0.0, 1.0]."""
    breakdown: dict[str, float] = Field(default_factory=dict)
    """Optional per-component breakdown of the reward."""


class State(BaseModel):
    """Full internal state of the environment (for serialisation)."""

    task_id: str
    seed: int
    job_description: JobDescription
    candidates: list[CandidateProfile] = Field(default_factory=list)
    decisions: dict[str, Decision] = Field(default_factory=dict)
    """candidate_id → decision made so far."""
    scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    """candidate_id → {dimension: score}."""
    steps_taken: int = 0
    step_budget: int = 0
    done: bool = False
