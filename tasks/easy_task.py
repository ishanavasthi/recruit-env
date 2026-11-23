"""Easy task — 'Screening Sprint'.

Screen 10 candidates, pick the top 3.  Straightforward signals,
generous step budget.
"""

from __future__ import annotations

from env.models import EpisodeState, Observation
from tasks.base import BaseGrader, TaskConfig

EASY_CONFIG = TaskConfig(
    id="easy",
    name="Screening Sprint",
    description="Screen 10 candidates and shortlist the top 3.",
    difficulty="easy",
    max_steps=40,
    candidate_count=10,
    label_distribution={"shortlist": 3, "hold": 4, "reject": 3},
    success_threshold=0.75,
    role_type="backend_dev",
)

_PERFECT_SHORTLIST_BONUS = 0.2


class EasyGrader(BaseGrader):
    """Grader for the easy task.

    Scoring
    -------
    - Base = weighted share of correct decisions; ground-truth shortlists
      weigh 2x. An undecided candidate counts as wrong.
    - Perfect shortlist bonus: +0.2 if the agent shortlisted exactly the
      3 ground-truth shortlists (no false positives, no misses)
    - Score = (base + bonus) / 1.2, so a perfect episode scores 1.0 and any
      mistake scores below it. Clamped to [0.001, 0.999].
    """

    def grade(self, initial_obs: Observation, final_state: EpisodeState) -> float:
        candidates = final_state.candidates
        decisions = final_state.decisions_made
        total = len(candidates)
        if total == 0:
            return 0.0

        gt = {c.id: c.ground_truth_label for c in candidates}

        # Weighted correct count: shortlist errors cost 2x
        weighted_correct = 0.0
        weighted_total = 0.0
        for c in candidates:
            cid = c.id
            decision = decisions.get(cid)  # undecided → wrong
            truth = gt[cid]

            if truth == "shortlist":
                weight = 2.0
            else:
                weight = 1.0

            weighted_total += weight
            if decision == truth:
                weighted_correct += weight

        base_score = weighted_correct / weighted_total if weighted_total > 0 else 0.0

        # Perfect shortlist bonus
        gt_shortlist_ids = {cid for cid, label in gt.items() if label == "shortlist"}
        agent_shortlist_ids = {cid for cid, d in decisions.items() if d == "shortlist"}
        bonus = (
            _PERFECT_SHORTLIST_BONUS if agent_shortlist_ids == gt_shortlist_ids else 0.0
        )

        raw = (base_score + bonus) / (1.0 + _PERFECT_SHORTLIST_BONUS)
        return max(0.001, min(0.999, raw))
