"""Hard task — 'Batch Triage'.

20 candidates under a budget far below a full read (5 sources × 20 = 100
reveals). Undecided candidates count as not shortlisted, so only the five
shortlist decisions are needed; the rest of the budget is for choosing which
candidates to investigate.
"""

from __future__ import annotations

from env.models import EpisodeState, Observation
from tasks.base import BaseGrader, GradeResult, TaskConfig, finalize, make_check, twin_gate

HARD_CONFIG = TaskConfig(
    id="hard",
    name="Batch Triage",
    description=(
        "Batch triage 20 candidates with a tight step budget. "
        "Shortlist the best 5."
    ),
    difficulty="hard",
    max_steps=50,
    candidate_count=20,
    label_distribution={"shortlist": 5, "hold": 8, "reject": 7},
    success_threshold=0.70,
)


class HardGrader(BaseGrader):
    """Grader for the hard task.

    Checks
    ------
    - ``shortlist_f1`` (1.0): F1 of the agent's shortlist against the true
      shortlist. Undecided candidates count as not shortlisted. Unused steps
      earn nothing: paying for them would reward gathering less.

    Gate: ``twin_consistency`` (cap 0.65); an undecided twin counts as hold.
    """

    threshold = HARD_CONFIG.success_threshold

    def evaluate(self, initial_obs: Observation, final_state: EpisodeState) -> GradeResult:
        truth = {c.id for c in final_state.candidates if c.ground_truth_label == "shortlist"}
        agent = {cid for cid, d in final_state.decisions_made.items() if d == "shortlist"}
        tp = len(truth & agent)
        precision = tp / len(agent) if agent else 0.0
        recall = tp / len(truth) if truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        checks = [
            make_check("shortlist_f1", 1.0, f1,
                       f"{tp} of {len(truth)} true shortlists found, {len(agent) - tp} false"),
        ]
        return finalize(checks, [twin_gate(final_state, self.threshold, undecided_as="hold")])
