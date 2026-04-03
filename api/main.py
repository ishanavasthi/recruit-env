"""FastAPI application — all RecruitEnv endpoints.

Run with: uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from env.environment import RecruitmentEnvironment
from env.models import Action, EpisodeState, Observation, Reward

app = FastAPI(
    title="RecruitEnv",
    description="OpenEnv-compliant RL environment for candidate pipeline triage.",
    version="0.1.0",
)

_env = RecruitmentEnvironment()


@app.get("/")
def root() -> dict:
    """Health-check / landing page."""
    return {"status": "ok", "environment": "RecruitEnv"}


@app.post("/reset")
def reset(task_id: str = "easy", seed: int = 42) -> Observation:
    """Reset the environment and return the initial observation."""
    raise NotImplementedError


@app.post("/step")
def step(action: Action) -> dict:
    """Execute one agent action. Returns {observation, reward}."""
    raise NotImplementedError


@app.get("/state")
def get_state() -> EpisodeState:
    """Return full internal environment state."""
    raise NotImplementedError


@app.get("/tasks")
def list_tasks() -> list[str]:
    """List available task IDs."""
    raise NotImplementedError
