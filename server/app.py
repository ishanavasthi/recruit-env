"""RecruitEnv server: the core served through OpenEnv.

    uvicorn server.app:app --host 0.0.0.0 --port 7860
    recruitenv-server                       # same, after pip install ".[openenv]"

Episodes run over ``/ws`` (one environment per connection, up to
``MAX_CONCURRENT_ENVS``, default 16). OpenEnv's HTTP ``/reset``, ``/step`` and
``/state`` are stateless: each request uses a fresh environment. Task
discovery is under ``/recruitenv/...`` (OpenEnv Task API). Extra routes:
``GET /`` (about), ``GET /tasks`` (tasks and action schema) and
``GET /baseline`` (rule-based reference scores, seed 42).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from openenv.core import create_app

from recruitenv import __version__, policies
from recruitenv.environment import RecruitmentEnvironment
from recruitenv.openenv_adapter import (
    DESCRIPTION,
    RecruitAction,
    RecruitObservation,
    RecruitOpenEnv,
    RecruitResetRequest,
    RecruitState,
    task_info,
)
from recruitenv.tasks import TASK_REGISTRY

MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "16"))

app = create_app(
    RecruitOpenEnv,
    RecruitAction,
    RecruitObservation,
    env_name="recruitenv",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
    state_cls=RecruitState,
    reset_cls=RecruitResetRequest,
)

_baseline_cache: dict[str, Any] | None = None


@app.get("/", tags=["RecruitEnv"])
def about() -> dict[str, Any]:
    """What this server is and where to start."""
    return {
        "name": "RecruitEnv",
        "version": __version__,
        "description": DESCRIPTION,
        "episodes": "websocket /ws (reset, step, state); see /schema",
        "tasks": "/tasks or the Task API under /recruitenv/",
        "source": "https://github.com/ishanavasthi/recruit-env",
    }


@app.get("/tasks", tags=["RecruitEnv"])
def tasks() -> dict[str, Any]:
    """Every task's configuration and the action JSON schema."""
    return {
        "tasks": [task_info(t) for t in TASK_REGISTRY],
        "action_schema": RecruitAction.model_json_schema(),
    }


@app.get("/baseline", tags=["RecruitEnv"])
def baseline() -> dict[str, Any]:
    """Rule-based rubric heuristic on every task at seed 42 (no model calls)."""
    global _baseline_cache
    if _baseline_cache is None:
        scores = {}
        for task_id in TASK_REGISTRY:
            env = RecruitmentEnvironment()
            env.reset(task_id, 42)
            policies.rubric_heuristic(env)
            scores[f"{task_id}_task"] = round(env.state().episode_score or 0.0, 4)
        _baseline_cache = {"scores": scores, "policy": "rubric_heuristic", "seed": 42,
                           "timestamp": datetime.now(timezone.utc).isoformat()}
    return _baseline_cache


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
