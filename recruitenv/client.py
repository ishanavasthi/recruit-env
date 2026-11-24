"""Typed OpenEnv client for a RecruitEnv server (needs the ``openenv`` extra).

    from recruitenv.client import RecruitEnvClient, RecruitAction

    with RecruitEnvClient(base_url="http://localhost:7860").sync() as env:
        r = env.reset(task="hard", seed=3, name_blind=True)
        cid = r.observation.candidates_summary[0]["id"]
        r = env.step(RecruitAction(type="check_platform", candidate_id=cid, platform="github"))

Episodes run over the ``/ws`` websocket, one server-side environment per
connection. On the terminal step ``r.observation.metadata["grade"]`` holds the
grade report.
"""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from recruitenv.openenv_adapter import RecruitAction, RecruitObservation, RecruitState

DEFAULT_URL = "https://heyavasthi-recruitenv.hf.space"

__all__ = ["RecruitEnvClient", "RecruitAction", "RecruitObservation", "RecruitState", "DEFAULT_URL"]


class RecruitEnvClient(EnvClient[RecruitAction, RecruitObservation, RecruitState]):
    def __init__(self, base_url: str = DEFAULT_URL, **kwargs: Any) -> None:
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: RecruitAction) -> dict[str, Any]:
        return action.model_dump(exclude={"metadata"}, exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[RecruitObservation]:
        data = dict(payload.get("observation") or {})
        data.setdefault("metadata", payload.get("metadata") or {})
        obs = RecruitObservation(**data, reward=payload.get("reward"), done=payload.get("done", False))
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: dict[str, Any]) -> RecruitState:
        return RecruitState(**payload)
