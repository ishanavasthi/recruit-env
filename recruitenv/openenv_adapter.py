"""OpenEnv adapter: serves the core environment through ``openenv.core``.

Needs the ``openenv`` extra (``pip install "recruitenv[openenv]"``). The core
(``recruitenv.environment``) never imports this module.

- ``RecruitAction``: one flat action class (OpenEnv validates actions with a
  single class). ``type`` selects the action; the adapter converts it to the
  core's discriminated union.
- ``RecruitObservation``: the agent-visible observation, plus ``reward`` and
  ``done``. On the terminal step ``metadata["grade"]`` holds the grade report
  (checks, gates and ground truth), which the server cannot serve separately
  because OpenEnv's HTTP routes are stateless.
- ``RecruitOpenEnv``: one core environment per session. ``reset(task=...,
  seed=..., name_seed=..., name_blind=...)``. Tasks are listed through the
  OpenEnv Task API (split ``test``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional

from openenv.core import Action, Environment, Observation, State
from openenv.core.env_server.types import ResetRequest
from openenv.core.env_server.types import EnvironmentMetadata
from pydantic import Field, TypeAdapter

from recruitenv import __version__
from recruitenv.environment import RecruitmentEnvironment
from recruitenv.models import Action as CoreAction
from recruitenv.models import JobDescription
from recruitenv.tasks import TASK_REGISTRY

_CORE_ACTION = TypeAdapter(CoreAction)

DESCRIPTION = (
    "Candidate pipeline triage. An agent reveals evidence (GitHub, LeetCode, "
    "Kaggle, resume sections) under a step budget and labels each candidate "
    "shortlist, hold or reject against quotas. Scored by a published rubric, "
    "with a counterfactual fairness audit."
)
SPLIT = "test"


class RecruitAction(Action):
    """One step. Set the fields the chosen ``type`` needs.

    - ``read_resume_section``: ``candidate_id``, ``section`` (``experience``,
      ``skills``, ``education``)
    - ``check_platform``: ``candidate_id``, ``platform`` (``github``,
      ``leetcode``, ``kaggle``)
    - ``make_decision``: ``candidate_id``, ``decision`` (``shortlist``,
      ``hold``, ``reject``)
    """

    type: Literal["read_resume_section", "check_platform", "make_decision"]
    candidate_id: str
    section: Optional[str] = None
    platform: Optional[Literal["github", "leetcode", "kaggle"]] = None
    decision: Optional[Literal["shortlist", "hold", "reject"]] = None

    def to_core(self) -> CoreAction:
        data = self.model_dump(exclude={"metadata"}, exclude_none=True)
        return _CORE_ACTION.validate_python(data)


class RecruitObservation(Observation):
    """What the agent sees after ``reset`` or a step."""

    task_id: str = ""
    step_number: int = 0
    steps_remaining: int = 0
    job_description: Optional[JobDescription] = None
    candidates_summary: list[dict[str, str]] = Field(default_factory=list)
    revealed_data: dict[str, Any] = Field(default_factory=dict)
    decisions_made: dict[str, str] = Field(default_factory=dict)
    quotas: dict[str, int] = Field(default_factory=dict)
    last_action_result: dict[str, Any] = Field(
        default_factory=dict,
        description="The step's info: what it revealed, or why it was rejected",
    )
    error: Optional[str] = Field(
        default=None,
        description="Set when the action was rejected. A rejected action costs no step.",
    )


class RecruitState(State):
    """Agent-visible episode state (no ground truth)."""

    task_id: Optional[str] = None
    seed: Optional[int] = None
    name_seed: int = 0
    name_blind: bool = False
    max_steps: int = 0
    is_done: bool = False
    episode_score: Optional[float] = None
    generator_version: Optional[str] = None


class RecruitResetRequest(ResetRequest):
    """Reset parameters (published in ``/schema``)."""

    task: Literal["easy", "medium", "hard"] = Field(default="easy", description="Task id")
    name_seed: int = Field(default=0, description="Redraws only the names: a name-swap replay")
    name_blind: bool = Field(default=False, description="Show 'Candidate 07' instead of names")


def task_info(task_id: str) -> dict[str, Any]:
    cfg = dataclasses.asdict(TASK_REGISTRY[task_id])
    cfg["quotas"] = cfg.pop("label_distribution")
    return cfg


class RecruitOpenEnv(Environment[RecruitAction, RecruitObservation, RecruitState]):
    """OpenEnv environment: one core ``RecruitmentEnvironment`` per session."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self.core = RecruitmentEnvironment()
        self._episode_id: Optional[str] = None
        self._name_seed = 0

    # -- episode -------------------------------------------------------

    def _observation(self, obs: Any, *, reward: float | None, done: bool,
                     info: dict[str, Any] | None = None, error: str | None = None,
                     metadata: dict[str, Any] | None = None) -> RecruitObservation:
        keep = {"task_id", "step_number", "steps_remaining", "job_description",
                "candidates_summary", "revealed_data", "decisions_made", "quotas"}
        data = {k: v for k, v in obs.model_dump(mode="json").items() if k in keep}
        if "steps_remaining" not in data:  # a PublicState (rejected action)
            data["steps_remaining"] = obs.max_steps - obs.step_number
        return RecruitObservation(
            **data, reward=reward, done=done,
            last_action_result=info or {}, error=error, metadata=metadata or {},
        )

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task: str = "easy", name_seed: int = 0, name_blind: bool = False,
              **kwargs: Any) -> RecruitObservation:
        if task not in TASK_REGISTRY:
            raise ValueError(f"Unknown task {task!r}. Available: {sorted(TASK_REGISTRY)}")
        obs = self.core.reset(task, 42 if seed is None else seed,
                              name_seed=name_seed, name_blind=name_blind)
        self._episode_id = episode_id
        self._name_seed = name_seed
        return self._observation(obs, reward=0.0, done=False)

    def step(self, action: RecruitAction, timeout_s: Optional[float] = None,
             **kwargs: Any) -> RecruitObservation:
        try:
            core_action = action.to_core()
            obs, reward, done, info = self.core.step(core_action)
        except (ValueError, RuntimeError) as exc:
            # Illegal action: nothing changes and no step is spent.
            s = self.core.state()
            return self._observation(s, reward=0.0, done=s.is_done, error=str(exc))
        metadata = {"grade": self.core.grade_report()} if done else {}
        return self._observation(obs, reward=reward.step_reward, done=done,
                                 info=info, metadata=metadata)

    @property
    def state(self) -> RecruitState:
        try:
            s = self.core.state()
        except ValueError:
            return RecruitState(episode_id=self._episode_id, step_count=0)
        return RecruitState(
            episode_id=self._episode_id, step_count=s.step_number, task_id=s.task_id,
            seed=s.seed, name_seed=self._name_seed, name_blind=s.name_blind,
            max_steps=s.max_steps, is_done=s.is_done, episode_score=s.episode_score,
            generator_version=s.generator_version,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="RecruitEnv", description=DESCRIPTION, version=__version__,
            documentation_url="https://github.com/ishanavasthi/recruit-env",
        )

    # -- Task API ------------------------------------------------------

    def list_splits(self) -> list[str]:
        return [SPLIT]

    def _check_split(self, split: str) -> None:
        if split != SPLIT:
            raise ValueError(f"Unknown split {split!r}; the only split is {SPLIT!r}")

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        self._check_split(split)
        return [task_info(t) for t in TASK_REGISTRY]

    def num_tasks(self, split: str) -> int:
        self._check_split(split)
        return len(TASK_REGISTRY)

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        tasks = self.list_tasks(split)
        if not 0 <= index < len(tasks):
            raise IndexError(f"task index {index} out of range 0..{len(tasks) - 1}")
        return tasks[index]

    def get_task_range(self, split: str, start: Optional[int] = None,
                       stop: Optional[int] = None) -> list[dict[str, Any]]:
        return self.list_tasks(split)[start:stop]
