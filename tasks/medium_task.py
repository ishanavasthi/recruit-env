"""Medium task — 'Conflicted Signals'.

5 candidates; shortlist the best 2. A small pool with a generous budget, so
the agent can afford to check every signal for every candidate.

The generator does not yet produce candidates whose platform signals
conflict, so this grader scores accuracy only. Issue #9 adds a real conflict
mode and will score whether conflicted candidates were investigated.
"""

from __future__ import annotations

from env.models import EpisodeState, Observation
from tasks.base import BaseGrader, TaskConfig

MEDIUM_CONFIG = TaskConfig(
    id="medium",
    name="Conflicted Signals",
    description=(
        "Evaluate 5 candidates across platforms. "
        "Shortlist the best 2."
    ),
    difficulty="medium",
    max_steps=25,
    candidate_count=5,
    label_distribution={"shortlist": 2, "hold": 2, "reject": 1},
    success_threshold=0.70,
    role_type="ml_engineer",
)


class MediumGrader(BaseGrader):
    """Grader for the medium task.

    Scoring
    -------
    - Score = correct_decisions / total_candidates; an undecided candidate
      counts as wrong.
    - Clamped to [0.001, 0.999].
    """

    def grade(self, initial_obs: Observation, final_state: EpisodeState) -> float:
        candidates = final_state.candidates
        decisions = final_state.decisions_made
        total = len(candidates)
        if total == 0:
            return 0.0

        correct = sum(
            1 for c in candidates
            if decisions.get(c.id) == c.ground_truth_label
        )
        return max(0.001, min(0.999, correct / total))
