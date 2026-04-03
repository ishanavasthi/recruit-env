"""Easy task: screen 10 candidates, pick top 3."""

from tasks.base import TaskDefinition

EASY_TASK = TaskDefinition(
    task_id="easy",
    description="Screen 10 candidates and shortlist the top 3.",
    num_candidates=10,
    select_quota=3,
    step_budget=50,
)
