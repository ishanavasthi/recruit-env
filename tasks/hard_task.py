"""Hard task — 'Batch Triage'.

20 candidates under a very tight step budget (~1.5 steps per candidate).
Tests the agent's ability to prioritise information gathering and make
rapid, accurate shortlist decisions.
"""

from __future__ import annotations

from env.models import EpisodeState, Observation
from tasks.base import BaseGrader, TaskConfig

HARD_CONFIG = TaskConfig(
    id="hard",
    name="Batch Triage",
    description=(
        "Batch triage 20 candidates with a tight step budget. "
        "Shortlist the best 5."
    ),
    difficulty="hard",
    max_steps=30,
    candidate_count=20,
    label_distribution={"shortlist": 5, "hold": 8, "reject": 7},
    success_threshold=0.60,
    role_type="data_scientist",
)


class HardGrader(BaseGrader):
    """Grader for the hard task.

    Scoring
    -------
    - Primary metric: F1 score of shortlist decisions
        - Precision = true shortlists / all agent shortlists
        - Recall    = true shortlists / all ground-truth shortlists
    - Undecided candidates at episode end are auto-graded as "hold"
    - Efficiency bonus: (steps_saved / max_steps) * 0.1
    - Final score clamped to [0.0, 1.0]
    """

    def grade(self, initial_obs: Observation, final_state: EpisodeState) -> float:
        candidates = final_state.candidates
        decisions = final_state.decisions_made
        max_steps = final_state.max_steps
        steps_used = final_state.step_number

        if len(candidates) == 0:
            return 0.0

        gt = {c.id: c.ground_truth_label for c in candidates}

        # Effective decisions: undecided → "hold"
        effective: dict[str, str] = {}
        for c in candidates:
            effective[c.id] = decisions.get(c.id, "hold")

        # Shortlist precision / recall / F1
        gt_shortlist = {cid for cid, label in gt.items() if label == "shortlist"}
        agent_shortlist = {cid for cid, d in effective.items() if d == "shortlist"}

        true_positives = len(gt_shortlist & agent_shortlist)

        precision = (
            true_positives / len(agent_shortlist) if agent_shortlist else 0.0
        )
        recall = (
            true_positives / len(gt_shortlist) if gt_shortlist else 0.0
        )

        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # Efficiency bonus
        steps_saved = max(0, max_steps - steps_used)
        efficiency_bonus = (steps_saved / max_steps) * 0.1 if max_steps > 0 else 0.0

        raw = f1 + efficiency_bonus
        return max(0.001, min(0.999, raw))
