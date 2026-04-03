"""Hard task: batch triage 20 candidates under a tight step budget."""

from tasks.base import TaskDefinition

HARD_TASK = TaskDefinition(
    task_id="hard",
    description=(
        "Batch triage 20 candidates with a tight step budget. "
        "Shortlist the best 5."
    ),
    num_candidates=20,
    select_quota=5,
    step_budget=60,
)
