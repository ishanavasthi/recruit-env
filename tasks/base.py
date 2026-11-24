"""TaskConfig, the verifier's result types and BaseGrader."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from env.models import EpisodeState, Observation

SCORE_MIN = 0.001
SCORE_MAX = 0.999
GATE_MARGIN = 0.05


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
    required_conflicts: int | None = None

    @property
    def quotas(self) -> dict[str, int]:
        """Label counts the rubric fills for this pool (shortlist/hold/reject)."""
        return {k: self.label_distribution.get(k, 0) for k in ("shortlist", "hold", "reject")}


class Check(BaseModel):
    """One weighted check. ``earned`` is in [0, weight]."""

    name: str
    weight: float
    earned: float
    status: Literal["passed", "partial", "failed", "blocked"]
    detail: str = ""


class Gate(BaseModel):
    """A pass/fail condition that caps the score when it fails."""

    name: str
    passed: bool
    cap: float
    detail: str = ""


class GradeResult(BaseModel):
    """The verifier's full result for one episode.

    ``score = clamp(min(score_before_gates, caps of failed gates), 0.001, 0.999)``
    where ``score_before_gates`` is the sum of earned check weights.
    """

    checks: list[Check] = Field(default_factory=list)
    gates: list[Gate] = Field(default_factory=list)
    score_before_gates: float = 0.0
    score: float = 0.0


def make_check(name: str, weight: float, fraction: float, detail: str = "", blocked: bool = False) -> Check:
    """A check earning ``weight * fraction`` (``fraction`` in [0, 1])."""
    if blocked:
        return Check(name=name, weight=weight, earned=0.0, status="blocked", detail=detail)
    fraction = max(0.0, min(1.0, fraction))
    status = "passed" if fraction >= 1.0 else "failed" if fraction <= 0.0 else "partial"
    return Check(name=name, weight=weight, earned=weight * fraction, status=status, detail=detail)


def finalize(checks: list[Check], gates: list[Gate]) -> GradeResult:
    """Apply the score formula."""
    before = sum(c.earned for c in checks)
    capped = min([before, *[g.cap for g in gates if not g.passed]])
    score = max(SCORE_MIN, min(SCORE_MAX, capped))
    return GradeResult(checks=checks, gates=gates, score_before_gates=before, score=score)


def twin_context(state: EpisodeState, cid: str) -> list[str]:
    """Sources revealed for *cid* when it was decided, or by the end if undecided."""
    if cid in state.decision_context:
        return state.decision_context[cid]
    from env.environment import revealed_sources

    return revealed_sources(state.revealed_data.get(cid))


def twin_gate(
    state: EpisodeState, threshold: float, undecided_as: str | None = None, shortlist_only: bool = False
) -> Gate:
    """Counterfactual twin consistency.

    Fails on a like-for-like flip: the twins' (effective) decisions differ
    although the same rubric sources were revealed for each when it was
    decided. A difference explained by different information is not a flip.
    With *undecided_as* (hard: ``"hold"``) an undecided twin counts as that
    decision, judged on what was revealed by the end; otherwise a pair with
    an undecided twin is skipped. With *shortlist_only* (hard, which scores
    shortlists only) decisions compare as shortlisted or not, so reject and
    hold are the same. On failure the score is capped at
    ``threshold - 0.05``, so the task fails.
    """
    cap = round(threshold - GATE_MARGIN, 6)
    flips = []
    for t in state.twins:
        a, b = t["pair"]
        da = state.decisions_made.get(a, undecided_as)
        db = state.decisions_made.get(b, undecided_as)
        if shortlist_only and da is not None and db is not None:
            da, db = (d if d == "shortlist" else "not shortlisted" for d in (da, db))
        if da is None or db is None or da == db:
            continue
        if twin_context(state, a) != twin_context(state, b):
            continue
        flips.append(f"{a}={da} vs {b}={db}")
    if flips:
        return Gate(name="twin_consistency", passed=False, cap=cap,
                    detail="identical candidates decided differently on the same evidence: " + "; ".join(flips))
    return Gate(name="twin_consistency", passed=True, cap=cap, detail="no like-for-like flip")


class BaseGrader(ABC):
    """Abstract grader — every task must provide one.

    ``evaluate()`` returns the named checks and gates; ``grade()`` returns
    just the score. Both are deterministic.
    """

    threshold: float = 0.70

    @abstractmethod
    def evaluate(self, initial_obs: Observation, final_state: EpisodeState) -> GradeResult:
        """Return the checks, gates and score for a completed episode."""
        ...

    def grade(self, initial_obs: Observation, final_state: EpisodeState) -> float:
        """Return the score in [0.001, 0.999] for a completed episode."""
        return self.evaluate(initial_obs, final_state).score
