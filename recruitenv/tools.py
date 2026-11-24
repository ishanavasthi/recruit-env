"""Tool-per-action environment for tool-calling agents and TRL's GRPOTrainer.

``RecruitToolEnv`` follows the ``environment_factory`` contract of TRL's
``GRPOTrainer`` (TRL >= 1.9): a no-argument constructor, ``reset(**row)``
returning the episode's opening text, one typed and documented method per
action (each becomes a tool), and ``get_reward()``. It runs the core in
process, so training needs no server. It has no TRL dependency.

    env = RecruitToolEnv()
    print(env.reset(task="hard", seed=3))
    print(env.check_github("candidate_004"))
    env.shortlist("candidate_004")
    ...
    env.get_reward()   # the grader's score once the episode is over, else 0.0

An illegal action raises ``ValueError`` with the reason; TRL shows the
message to the model as the tool's response. A rejected action costs no step.
"""

from __future__ import annotations

import json
from typing import Any

from recruitenv.environment import RecruitmentEnvironment
from recruitenv.models import CheckPlatformAction, MakeDecisionAction, ReadResumeSectionAction

INTRO = (
    "You are triaging candidates for the job below. Use the tools to reveal "
    "evidence (each reveal costs one step), then label every candidate with "
    "shortlist, hold or reject so the labels match the quotas. The published "
    "rubric ranks candidates by technical skill (GitHub, LeetCode, Kaggle), "
    "experience and growth, scaled by how many required skills they list. "
    "Education is not used. Names carry no information."
)


def _compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


class RecruitToolEnv:
    """One episode per ``reset``; tools are the public methods below."""

    def __init__(self) -> None:
        self.env = RecruitmentEnvironment()
        self.reward = 0.0
        self.done = False

    # -- episode ---------------------------------------------------------

    def reset(self, task: str = "easy", seed: int = 0, name_seed: int = 0,
              name_blind: bool = False, **_: Any) -> str:
        """Start an episode; returns the job, quotas, budget and candidates."""
        obs = self.env.reset(task, int(seed), name_seed=int(name_seed), name_blind=bool(name_blind))
        self.reward, self.done = 0.0, False
        jd = obs.job_description
        return (
            f"{INTRO}\n\nJob: {jd.role} ({jd.seniority}, {jd.experience_years}+ years). "
            f"Required skills: {', '.join(jd.required_skills)}. Nice to have: {', '.join(jd.nice_to_have)}.\n"
            f"Weights: technical {jd.weight_technical}, experience {jd.weight_experience}, growth {jd.weight_growth}.\n"
            f"Quotas: {_compact(obs.quotas)}. Step budget: {obs.steps_remaining}.\n"
            f"Candidates: {_compact(obs.candidates_summary)}"
        )

    def _act(self, action: Any) -> str:
        if self.done:
            raise ValueError("The episode is over.")
        try:
            obs, reward, done, info = self.env.step(action)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(str(exc)) from exc
        self.done = done
        if done:
            self.reward = reward.step_reward
            return f"Episode over. Score: {self.reward:.3f}."
        if "data" in info:
            shown = info["data"] if info["data"] else "already revealed or not available"
            body = _compact(shown)
        else:
            body = "decision recorded"
        return f"{body}\nSteps remaining: {obs.steps_remaining}. Decided: {len(obs.decisions_made)}."

    def _reveal(self, candidate_id: str, platform: str) -> str:
        return self._act(CheckPlatformAction(type="check_platform", candidate_id=candidate_id, platform=platform))

    def _decide(self, candidate_id: str, decision: str) -> str:
        return self._act(MakeDecisionAction(type="make_decision", candidate_id=candidate_id, decision=decision))

    # -- tools -----------------------------------------------------------

    def check_github(self, candidate_id: str) -> str:
        """
        Reveal a candidate's GitHub statistics (costs one step).

        Args:
            candidate_id: The candidate's id, for example "candidate_004".

        Returns:
            The revealed statistics and the steps remaining.
        """
        return self._reveal(candidate_id, "github")

    def check_leetcode(self, candidate_id: str) -> str:
        """
        Reveal a candidate's LeetCode statistics (costs one step).

        Args:
            candidate_id: The candidate's id, for example "candidate_004".

        Returns:
            The revealed statistics and the steps remaining.
        """
        return self._reveal(candidate_id, "leetcode")

    def check_kaggle(self, candidate_id: str) -> str:
        """
        Reveal a candidate's Kaggle statistics (costs one step).

        Args:
            candidate_id: The candidate's id, for example "candidate_004".

        Returns:
            The revealed statistics and the steps remaining.
        """
        return self._reveal(candidate_id, "kaggle")

    def read_resume(self, candidate_id: str, section: str) -> str:
        """
        Read one section of a candidate's resume (costs one step).

        Args:
            candidate_id: The candidate's id, for example "candidate_004".
            section: One of "experience", "skills" or "education".

        Returns:
            The section's content and the steps remaining.
        """
        return self._act(ReadResumeSectionAction(type="read_resume_section", candidate_id=candidate_id,
                                                 section=section))

    def shortlist(self, candidate_id: str) -> str:
        """
        Label a candidate shortlist. Decisions are final and cost one step.

        Args:
            candidate_id: The candidate's id.

        Returns:
            Confirmation, or the final score when the episode ends.
        """
        return self._decide(candidate_id, "shortlist")

    def hold(self, candidate_id: str) -> str:
        """
        Label a candidate hold. Decisions are final and cost one step.

        Args:
            candidate_id: The candidate's id.

        Returns:
            Confirmation, or the final score when the episode ends.
        """
        return self._decide(candidate_id, "hold")

    def reject(self, candidate_id: str) -> str:
        """
        Label a candidate reject. Decisions are final and cost one step.

        Args:
            candidate_id: The candidate's id.

        Returns:
            Confirmation, or the final score when the episode ends.
        """
        return self._decide(candidate_id, "reject")

    # -- reward ----------------------------------------------------------

    def get_reward(self) -> float:
        """The grader's score if the episode ended, else 0.0."""
        return self.reward
