"""Task definitions for RecruitEnv."""

from tasks.base import BaseGrader, TaskConfig
from tasks.easy_task import EASY_CONFIG, EasyGrader
from tasks.medium_task import MEDIUM_CONFIG, MediumGrader
from tasks.hard_task import HARD_CONFIG, HardGrader

TASK_REGISTRY: dict[str, TaskConfig] = {
    "easy": EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard": HARD_CONFIG,
}

GRADER_REGISTRY: dict[str, BaseGrader] = {
    "easy": EasyGrader(),
    "medium": MediumGrader(),
    "hard": HardGrader(),
}

__all__ = [
    "BaseGrader",
    "TaskConfig",
    "EASY_CONFIG",
    "EasyGrader",
    "MEDIUM_CONFIG",
    "MediumGrader",
    "HARD_CONFIG",
    "HardGrader",
    "TASK_REGISTRY",
    "GRADER_REGISTRY",
]
