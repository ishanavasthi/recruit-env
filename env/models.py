"""Pydantic v2 models for the RecruitEnv environment."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Platform stats models
# ---------------------------------------------------------------------------


class GitHubStats(BaseModel):
    """Synthetic GitHub profile statistics for a candidate."""

    repos: int = Field(description="Total number of public repositories")
    top_languages: list[str] = Field(
        default_factory=list,
        description="Most-used programming languages, ordered by frequency",
    )
    commit_streak_days: int = Field(
        default=0, description="Longest consecutive-day commit streak"
    )
    stars_received: int = Field(
        default=0, description="Total stars across all repositories"
    )
    notable_projects: list[str] = Field(
        default_factory=list,
        description="Names of standout repositories (high stars / forks)",
    )
    contributions_last_year: int = Field(
        default=0, description="Total contributions in the last 365 days"
    )


class LeetCodeStats(BaseModel):
    """Synthetic LeetCode profile statistics for a candidate."""

    problems_solved: int = Field(description="Total problems solved")
    easy: int = Field(default=0, description="Easy problems solved")
    medium: int = Field(default=0, description="Medium problems solved")
    hard: int = Field(default=0, description="Hard problems solved")
    contest_rating: int = Field(
        default=0, description="Competitive contest Elo-like rating"
    )
    global_rank_percentile: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentile rank among all users (0-100, higher is better)",
    )


class KaggleStats(BaseModel):
    """Synthetic Kaggle profile statistics for a candidate."""

    rank: str = Field(
        description="Kaggle tier string, e.g. 'Unranked', 'Contributor', 'Expert', 'Master', 'Grandmaster'"
    )
    competitions_entered: int = Field(
        default=0, description="Number of competitions entered"
    )
    best_finish_percentile: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Best competition finish as percentile (0-100, higher is better)",
    )
    medals: dict[str, int] = Field(
        default_factory=dict,
        description="Medal counts keyed by type, e.g. {'gold': 1, 'silver': 2, 'bronze': 3}",
    )


class ResumeSection(BaseModel):
    """A single section of a candidate's resume."""

    section_name: str = Field(description="Section heading, e.g. 'education', 'experience', 'skills'")
    content: str = Field(description="Free-text content of the section")
    years_experience: float = Field(
        default=0.0,
        ge=0.0,
        description="Relevant years of experience conveyed by this section",
    )


# ---------------------------------------------------------------------------
# Candidate profile
# ---------------------------------------------------------------------------


class CandidateProfile(BaseModel):
    """Full synthetic candidate profile with ground-truth label."""

    id: str = Field(description="Unique candidate identifier")
    name: str = Field(description="Candidate display name")
    resume_sections: dict[str, ResumeSection] = Field(
        default_factory=dict,
        description="Mapping of section name → ResumeSection",
    )
    github: GitHubStats = Field(description="Synthetic GitHub statistics")
    leetcode: LeetCodeStats = Field(description="Synthetic LeetCode statistics")
    kaggle: KaggleStats = Field(description="Synthetic Kaggle statistics")
    ground_truth_label: Literal["shortlist", "hold", "reject"] = Field(
        description="Correct decision, embedded at generation time"
    )
    ground_truth_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Ground-truth dimension scores, e.g. {'technical': 0.8, 'experience': 0.6}",
    )


# ---------------------------------------------------------------------------
# Job description
# ---------------------------------------------------------------------------


class JobDescription(BaseModel):
    """Synthetic job description generated per episode."""

    role: str = Field(description="Job title / role name")
    required_skills: list[str] = Field(
        default_factory=list, description="Skills the candidate must have"
    )
    nice_to_have: list[str] = Field(
        default_factory=list, description="Bonus skills that are not required"
    )
    experience_years: int = Field(
        default=0, ge=0, description="Minimum years of experience required"
    )
    seniority: Literal["junior", "mid", "senior"] = Field(
        description="Target seniority level for the role"
    )
    weight_technical: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Grading weight for technical ability",
    )
    weight_experience: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Grading weight for experience",
    )
    weight_growth: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Grading weight for growth potential",
    )

    @model_validator(mode="after")
    def _weights_must_sum_to_one(self) -> JobDescription:
        total = self.weight_technical + self.weight_experience + self.weight_growth
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"weight_technical + weight_experience + weight_growth must sum to 1.0, got {total:.6f}"
            )
        return self


# ---------------------------------------------------------------------------
# Actions — discriminated union
# ---------------------------------------------------------------------------


class ReadResumeSectionAction(BaseModel):
    """Request a specific section of a candidate's resume."""

    type: Literal["read_resume_section"] = Field(
        default="read_resume_section", description="Action discriminator"
    )
    candidate_id: str = Field(description="Target candidate identifier")
    section: str = Field(
        description="Resume section to read, e.g. 'education', 'experience', 'skills'"
    )


class CheckPlatformAction(BaseModel):
    """Request platform statistics for a candidate."""

    type: Literal["check_platform"] = Field(
        default="check_platform", description="Action discriminator"
    )
    candidate_id: str = Field(description="Target candidate identifier")
    platform: Literal["github", "leetcode", "kaggle"] = Field(
        description="Platform to query"
    )


class ScoreDimensionAction(BaseModel):
    """Record the agent's assessment score for a candidate dimension."""

    type: Literal["score_dimension"] = Field(
        default="score_dimension", description="Action discriminator"
    )
    candidate_id: str = Field(description="Target candidate identifier")
    dimension: Literal["technical", "experience", "growth"] = Field(
        description="Dimension being scored"
    )
    score: float = Field(
        ge=0.0, le=1.0, description="Agent-assigned score in [0.0, 1.0]"
    )


class MakeDecisionAction(BaseModel):
    """Terminal shortlist / hold / reject decision for a candidate."""

    type: Literal["make_decision"] = Field(
        default="make_decision", description="Action discriminator"
    )
    candidate_id: str = Field(description="Target candidate identifier")
    decision: Literal["shortlist", "hold", "reject"] = Field(
        description="Final decision for this candidate"
    )


Action = Annotated[
    Union[
        ReadResumeSectionAction,
        CheckPlatformAction,
        ScoreDimensionAction,
        MakeDecisionAction,
    ],
    Field(discriminator="type"),
]
"""Discriminated union of all possible agent actions."""


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """Observation returned to the agent after each step."""

    task_id: str = Field(description="Current task identifier")
    step_number: int = Field(description="Current step index (0-based)")
    steps_remaining: int = Field(description="Budget of steps left in the episode")
    job_description: JobDescription = Field(
        description="The job description for this episode"
    )
    candidates_summary: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {'id': ..., 'name': ...} — no stats, just identifiers",
    )
    revealed_data: dict[str, Any] = Field(
        default_factory=dict,
        description="candidate_id → data revealed so far (resume sections, platform stats)",
    )
    decisions_made: dict[str, str] = Field(
        default_factory=dict,
        description="candidate_id → decision string for candidates already decided",
    )
    scores_recorded: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="candidate_id → {dimension: score} for scores recorded so far",
    )
    done: bool = Field(
        default=False, description="Whether the episode has ended"
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


class Reward(BaseModel):
    """Reward signal returned alongside an observation."""

    step_reward: float = Field(
        default=0.0, description="Reward attributed to this single step"
    )
    cumulative_reward: float = Field(
        default=0.0, description="Total reward accumulated so far in the episode"
    )
    fairness_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Penalty in [0, 1] for biased shortlisting patterns",
    )
    accuracy_bonus: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Bonus in [0, 1] for decisions matching ground truth",
    )
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Optional per-component breakdown of the reward",
    )


# ---------------------------------------------------------------------------
# Episode state (full internal state for serialisation / grading)
# ---------------------------------------------------------------------------


class EpisodeState(BaseModel):
    """Full internal state of the environment."""

    task_id: str = Field(description="Current task identifier")
    seed: int = Field(description="RNG seed used for this episode")
    step_number: int = Field(default=0, description="Current step index (0-based)")
    max_steps: int = Field(description="Step budget for this episode")
    job_description: JobDescription = Field(
        description="The job description for this episode"
    )
    candidates: list[CandidateProfile] = Field(
        default_factory=list,
        description="All candidate profiles (including ground-truth labels)",
    )
    candidates_summary: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {'id': ..., 'name': ...} — agent-visible summary",
    )
    revealed_data: dict[str, Any] = Field(
        default_factory=dict,
        description="candidate_id → data revealed so far",
    )
    decisions_made: dict[str, str] = Field(
        default_factory=dict,
        description="candidate_id → decision string",
    )
    scores_recorded: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="candidate_id → {dimension: score}",
    )
    is_done: bool = Field(
        default=False,
        description="Whether the episode has ended",
    )
    episode_score: float | None = Field(
        default=None,
        description="Final episode score in [0, 1], set when episode ends",
    )
