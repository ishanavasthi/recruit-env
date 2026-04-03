"""Task definitions for RecruitEnv."""

from tasks.base import TaskDefinition
from tasks.easy_task import EASY_TASK
from tasks.medium_task import MEDIUM_TASK
from tasks.hard_task import HARD_TASK

TASK_REGISTRY: dict[str, TaskDefinition] = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}

__all__ = [
    "TaskDefinition",
    "EASY_TASK",
    "MEDIUM_TASK",
    "HARD_TASK",
    "TASK_REGISTRY",
]
