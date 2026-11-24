"""Easy task — 'Screening Sprint'.

Screen 10 candidates and shortlist the top 3. The budget (62 steps) covers
reading all five rubric sources for everyone, every decision and 2 spare
steps: easy tests applying the rubric to full evidence.
"""

from __future__ import annotations

from recruitenv.models import EpisodeState, Observation
from recruitenv.tasks.base import BaseGrader, GradeResult, TaskConfig, finalize, make_check, twin_gate

EASY_CONFIG = TaskConfig(
    id="easy",
    name="Screening Sprint",
    description="Screen 10 candidates and shortlist the top 3.",
    difficulty="easy",
    max_steps=62,
    candidate_count=10,
    label_distribution={"shortlist": 3, "hold": 4, "reject": 3},
    success_threshold=0.75,
)


class EasyGrader(BaseGrader):
    """Grader for the easy task.

    Checks
    ------
    - ``decision_accuracy`` (0.8): weighted share of correct decisions; true
      shortlists weigh 2x, and an undecided candidate counts as wrong.
    - ``exact_shortlist`` (0.2): the agent's shortlist equals the true one.

    Gate: ``twin_consistency`` (cap 0.70).
    """

    threshold = EASY_CONFIG.success_threshold

    def evaluate(self, initial_obs: Observation, final_state: EpisodeState) -> GradeResult:
        decisions = final_state.decisions_made
        weighted = total = 0.0
        for c in final_state.candidates:
            w = 2.0 if c.ground_truth_label == "shortlist" else 1.0
            total += w
            weighted += w * (decisions.get(c.id) == c.ground_truth_label)
        truth_sl = {c.id for c in final_state.candidates if c.ground_truth_label == "shortlist"}
        agent_sl = {cid for cid, d in decisions.items() if d == "shortlist"}
        checks = [
            make_check("decision_accuracy", 0.8, weighted / total if total else 0.0,
                       f"{weighted:g} of {total:g} weighted decisions correct"),
            make_check("exact_shortlist", 0.2, float(agent_sl == truth_sl),
                       f"shortlisted {len(agent_sl & truth_sl)} of {len(truth_sl)} true shortlists, "
                       f"{len(agent_sl - truth_sl)} false"),
        ]
        return finalize(checks, [twin_gate(final_state, self.threshold)])
