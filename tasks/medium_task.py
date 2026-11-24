"""Medium task — 'Conflicted Signals'.

8 candidates; shortlist the best 3. Exactly two candidates are conflicted:
two of their sources disagree sharply and judging them by either one alone
gives the wrong label. The budget (50 steps) covers reading all five rubric
sources for everyone plus every decision.
"""

from __future__ import annotations

from env.models import EpisodeState, Observation
from tasks.base import BaseGrader, GradeResult, TaskConfig, finalize, make_check, twin_gate

MEDIUM_CONFIG = TaskConfig(
    id="medium",
    name="Conflicted Signals",
    description=(
        "Evaluate 8 candidates, two of whom have sources that disagree. "
        "Shortlist the best 3."
    ),
    difficulty="medium",
    max_steps=50,
    candidate_count=8,
    label_distribution={"shortlist": 3, "hold": 3, "reject": 2},
    success_threshold=0.70,
    required_conflicts=2,
)


class MediumGrader(BaseGrader):
    """Grader for the medium task.

    Checks
    ------
    - ``decision_accuracy`` (0.8): weighted share of correct decisions;
      conflicted candidates weigh 2x, and an undecided candidate counts as
      wrong.
    - ``conflicts_investigated`` (0.2, split evenly per conflicted candidate):
      both conflicting sources were revealed before the candidate was
      decided. A candidate's share is ``blocked`` unless its decision is
      correct.

    Gate: ``twin_consistency`` (cap 0.65).
    """

    threshold = MEDIUM_CONFIG.success_threshold

    def evaluate(self, initial_obs: Observation, final_state: EpisodeState) -> GradeResult:
        decisions = final_state.decisions_made
        conflicts = final_state.conflicts
        weighted = total = 0.0
        for c in final_state.candidates:
            w = 2.0 if c.id in conflicts else 1.0
            total += w
            weighted += w * (decisions.get(c.id) == c.ground_truth_label)
        checks = [
            make_check("decision_accuracy", 0.8, weighted / total if total else 0.0,
                       f"{weighted:g} of {total:g} weighted decisions correct"),
        ]
        truth = {c.id: c.ground_truth_label for c in final_state.candidates}
        share = 0.2 / len(conflicts) if conflicts else 0.0
        for cid, pair in sorted(conflicts.items()):
            name = f"conflicts_investigated:{cid}"
            if decisions.get(cid) != truth[cid]:
                checks.append(make_check(name, share, 0.0, "decision not correct", blocked=True))
                continue
            seen = set(final_state.decision_context.get(cid, []))
            done = all(src in seen for src in pair)
            checks.append(make_check(name, share, float(done),
                                     f"needed {pair[0]} and {pair[1]} before deciding; had {sorted(seen)}"))
        return finalize(checks, [twin_gate(final_state, self.threshold)])
