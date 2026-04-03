"""Reward function logic for the RecruitEnv environment.

All graders return float in [0.0, 1.0] and are fully deterministic.
"""

from __future__ import annotations

from env.models import (
    Action,
    CandidateProfile,
    CheckPlatformAction,
    EpisodeState,
    JobDescription,
    MakeDecisionAction,
    ReadResumeSectionAction,
    Reward,
    ScoreDimensionAction,
)


class RewardCalculator:
    """Stateless calculator for step rewards, terminal rewards, and efficiency."""

    # -- step reward --------------------------------------------------------

    @staticmethod
    def calculate_step_reward(action: Action, state: EpisodeState) -> float:
        """Return the immediate reward for *action* given current *state*.

        - ReadResumeSectionAction:  +0.02 if section not previously read, else 0.0
        - CheckPlatformAction:     +0.03 if platform not previously checked, else 0.0
        - ScoreDimensionAction:    +0.02 always
        - MakeDecisionAction:       0.0  (scored at episode end)
        """
        if isinstance(action, ReadResumeSectionAction):
            cid = action.candidate_id
            revealed = state.revealed_data.get(cid, {})
            read_sections: list[str] = revealed.get("resume_sections", [])
            if action.section in read_sections:
                return 0.0
            return 0.02

        if isinstance(action, CheckPlatformAction):
            cid = action.candidate_id
            revealed = state.revealed_data.get(cid, {})
            checked: list[str] = revealed.get("platforms", [])
            if action.platform in checked:
                return 0.0
            return 0.03

        if isinstance(action, ScoreDimensionAction):
            return 0.02

        if isinstance(action, MakeDecisionAction):
            return 0.0

        return 0.0

    # -- terminal reward ----------------------------------------------------

    @staticmethod
    def calculate_terminal_reward(
        decisions: dict[str, str],
        candidates: list[CandidateProfile],
        jd: JobDescription,
    ) -> dict[str, float]:
        """Score all decisions against ground-truth labels at episode end.

        Returns a dict with per-candidate scores plus ``"total"`` normalised
        to [0, 1].

        Scoring rules per candidate:
        - Correct shortlist:  +1.0 * jd.weight_technical
        - Correct reject:     +0.5
        - Correct hold:       +0.3
        - Wrong shortlist when ground truth is reject: -0.5
        - Wrong reject when ground truth is shortlist: -0.3
        - Any other hold error:                        -0.1
        """
        gt_map: dict[str, str] = {c.id: c.ground_truth_label for c in candidates}

        breakdown: dict[str, float] = {}
        raw_total = 0.0

        for cid, decision in decisions.items():
            truth = gt_map.get(cid)
            if truth is None:
                continue

            if decision == truth:
                if decision == "shortlist":
                    score = 1.0 * jd.weight_technical
                elif decision == "reject":
                    score = 0.5
                else:  # hold
                    score = 0.3
            else:
                if decision == "shortlist" and truth == "reject":
                    score = -0.5
                elif decision == "reject" and truth == "shortlist":
                    score = -0.3
                else:
                    score = -0.1

            breakdown[cid] = score
            raw_total += score

        # Normalise to [0, 1].
        # Theoretical max: every candidate scored at max positive value.
        # We use the candidate count to define the range.
        n = len(decisions) or 1
        # Max possible per candidate is ~1.0, min is -0.5.
        # Map from [-0.5*n, 1.0*n] → [0, 1].
        max_raw = 1.0 * n
        min_raw = -0.5 * n
        span = max_raw - min_raw
        normalised = (raw_total - min_raw) / span if span > 0 else 0.0
        normalised = max(0.0, min(1.0, normalised))

        breakdown["total"] = round(normalised, 6)
        return breakdown

    # -- efficiency bonus ---------------------------------------------------

    @staticmethod
    def calculate_efficiency_bonus(
        steps_used: int,
        max_steps: int,
        correct_decisions: int,
        total_candidates: int,
    ) -> float:
        """Return an efficiency bonus if the agent was both fast and accurate.

        +0.1 bonus when ALL decisions are correct AND fewer than 60% of the
        step budget was consumed.  Otherwise 0.0.
        """
        if total_candidates == 0:
            return 0.0
        all_correct = correct_decisions == total_candidates
        under_budget = steps_used < 0.6 * max_steps
        if all_correct and under_budget:
            return 0.1
        return 0.0


# ---------------------------------------------------------------------------
# Module-level convenience functions (kept for backward compatibility with
# env/__init__.py imports)
# ---------------------------------------------------------------------------

_calc = RewardCalculator()


def compute_reward(state: EpisodeState) -> Reward:
    """Compute the full Reward object for the current episode state."""
    terminal = _calc.calculate_terminal_reward(
        state.decisions_made, state.candidates, state.job_description,
    )

    gt_map = {c.id: c.ground_truth_label for c in state.candidates}
    correct = sum(
        1 for cid, d in state.decisions_made.items()
        if gt_map.get(cid) == d
    )

    efficiency = _calc.calculate_efficiency_bonus(
        state.step_number, state.max_steps, correct, len(state.candidates),
    )

    from env.fairness import FairnessChecker

    penalty = FairnessChecker.compute_penalty(
        state.decisions_made, state.candidates,
    )

    accuracy = terminal.get("total", 0.0)
    raw_score = accuracy + efficiency - penalty
    clamped = max(0.0, min(1.0, raw_score))

    breakdown = {k: v for k, v in terminal.items() if k != "total"}
    breakdown["accuracy_total"] = accuracy
    breakdown["efficiency_bonus"] = efficiency
    breakdown["fairness_penalty"] = penalty

    return Reward(
        step_reward=0.0,
        cumulative_reward=round(clamped, 6),
        fairness_penalty=round(max(0.0, min(1.0, penalty)), 6),
        accuracy_bonus=round(max(0.0, min(1.0, accuracy)), 6),
        breakdown=breakdown,
    )
