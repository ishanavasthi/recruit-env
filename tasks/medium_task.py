"""Medium task — 'Conflicted Signals'.

5 candidates, 2 of whom have deliberately conflicting platform signals
(e.g. great GitHub + bad LeetCode).  Tests whether the agent gathers
enough information before deciding.
"""

from __future__ import annotations

from env.models import EpisodeState, Observation
from tasks.base import BaseGrader, TaskConfig

MEDIUM_CONFIG = TaskConfig(
    id="medium",
    name="Conflicted Signals",
    description=(
        "Evaluate 5 candidates with conflicting signals across platforms. "
        "Shortlist the best 2."
    ),
    difficulty="medium",
    max_steps=25,
    candidate_count=5,
    label_distribution={"shortlist": 2, "hold": 2, "reject": 1},
    success_threshold=0.70,
    role_type="ml_engineer",
)

# The first two candidates in the pool (indices 0 and 1) are treated as the
# "conflicted" candidates whose platform signals are deliberately mixed.
_CONFLICTED_INDICES = {0, 1}
_ALL_PLATFORMS = {"github", "leetcode", "kaggle"}


class MediumGrader(BaseGrader):
    """Grader for the medium task.

    Scoring
    -------
    - Base = correct_decisions / total_candidates
    - BONUS per conflicted candidate: +0.15 if agent checked ALL 3 platforms
      before deciding (thoroughness)
    - PENALTY per conflicted candidate: -0.15 if agent decided with < 2
      platforms checked
    - Final score clamped to [0.0, 1.0]
    """

    def grade(self, initial_obs: Observation, final_state: EpisodeState) -> float:
        candidates = final_state.candidates
        decisions = final_state.decisions_made
        revealed = final_state.revealed_data
        total = len(candidates)
        if total == 0:
            return 0.0

        gt = {c.id: c.ground_truth_label for c in candidates}

        # Base accuracy
        correct = sum(
            1 for c in candidates
            if decisions.get(c.id, "hold") == gt[c.id]
        )
        base_score = correct / total

        # Identify conflicted candidate ids by their pool index
        conflicted_ids = {
            candidates[i].id
            for i in _CONFLICTED_INDICES
            if i < len(candidates)
        }

        # Thoroughness adjustment for conflicted candidates
        adjustment = 0.0
        for cid in conflicted_ids:
            cid_revealed = revealed.get(cid, {})
            platforms_checked = set(cid_revealed.get("platforms", []))
            has_decision = cid in decisions

            if has_decision and platforms_checked >= _ALL_PLATFORMS:
                adjustment += 0.15  # bonus: thorough
            elif has_decision and len(platforms_checked) < 2:
                adjustment -= 0.15  # penalty: hasty

        raw = base_score + adjustment
        return max(0.0, min(1.0, raw))
