"""Medium task: conflicting signals evaluation."""

from tasks.base import TaskDefinition

MEDIUM_TASK = TaskDefinition(
    task_id="medium",
    description=(
        "Evaluate candidates with conflicting signals across platforms. "
        "Shortlist the best 5 out of 15."
    ),
    num_candidates=15,
    select_quota=5,
    step_budget=80,
)
