"""FastAPI application — all RecruitEnv endpoints.

Run with: uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import asyncio
import dataclasses
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, TypeAdapter
from starlette.middleware.base import BaseHTTPMiddleware

from env.environment import RecruitmentEnvironment
from env.models import (
    Action,
    CheckPlatformAction,
    MakeDecisionAction,
    Observation,
    PublicState,
    Reward,
)
from tasks import TASK_REGISTRY

logger = logging.getLogger("recruitenv.api")

VERSION = "2.0.0"

# TypeAdapter for the Action discriminated union's JSON Schema
_action_adapter = TypeAdapter(Action)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_env = RecruitmentEnvironment()
_lock = asyncio.Lock()
# Cache for the simple baseline agent results
_baseline_cache: dict[str, Any] | None = None

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


class GradeResponse(BaseModel):
    score: float
    breakdown: dict[str, Any]
    task_id: str


class BaselineResponse(BaseModel):
    scores: dict[str, float]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RecruitEnv API starting up")
    yield
    logger.info("RecruitEnv API shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RecruitEnv",
    description="RL environment for candidate pipeline triage.",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict:
    """Root landing page."""
    return {"status": "ok", "environment": "RecruitEnv", "version": VERSION}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check."""
    return HealthResponse(status="ok", version=VERSION)


@app.post("/reset", response_model=Observation)
async def reset(body: ResetRequest | None = None) -> Observation:
    """Reset the environment and return the initial observation."""
    if body is None:
        body = ResetRequest()
    async with _lock:
        try:
            return _env.reset(body.task_id, body.seed)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResponse)
async def step(action: Action) -> StepResponse:
    """Execute one agent action."""
    async with _lock:
        try:
            obs, reward, done, info = _env.step(action)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )


@app.get("/state", response_model=PublicState)
async def get_state() -> PublicState:
    """Return the agent-visible episode state (no ground truth)."""
    async with _lock:
        try:
            return _env.state()
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))


@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    """List available tasks and the Action JSON schema."""
    tasks = []
    for tid, cfg in TASK_REGISTRY.items():
        tasks.append(dataclasses.asdict(cfg))
    action_schema = _action_adapter.json_schema()
    return {"tasks": tasks, "action_schema": action_schema}


@app.post("/grader", response_model=GradeResponse)
async def grade() -> GradeResponse:
    """Grade the current completed episode.

    The score equals the terminal step's ``cumulative_reward``. The breakdown
    includes ground-truth labels, so it is only available after the episode.
    """
    async with _lock:
        try:
            report = _env.grade_report()
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        score = report.pop("score")
        task_id = report.pop("task_id")
        breakdown: dict[str, Any] = {"grader_score": score, **report}
        return GradeResponse(score=score, breakdown=breakdown, task_id=task_id)


@app.get("/baseline", response_model=BaselineResponse)
async def baseline() -> BaselineResponse:
    """Run a simple rule-based baseline agent on all tasks.

    This does NOT call an LLM — it uses a deterministic heuristic:
    check leetcode + github for each candidate, then decide based on
    problems_solved and repo count thresholds.
    """
    global _baseline_cache
    if _baseline_cache is not None:
        return BaselineResponse(**_baseline_cache)

    scores: dict[str, float] = {}

    for task_id in TASK_REGISTRY:
        score = await _run_rule_baseline(task_id)
        scores[f"{task_id}_task"] = round(score, 4)

    result = {
        "scores": scores,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _baseline_cache = result
    return BaselineResponse(**result)


# ---------------------------------------------------------------------------
# Rule-based baseline agent
# ---------------------------------------------------------------------------


async def _run_rule_baseline(task_id: str, seed: int = 42) -> float:
    """Play one episode with a simple heuristic agent, return its episode score."""
    env = RecruitmentEnvironment()
    obs = env.reset(task_id, seed)
    cids = [c["id"] for c in obs.candidates_summary]
    done = False

    for cid in cids:
        if done:
            break

        obs, _, done, _ = env.step(
            CheckPlatformAction(candidate_id=cid, platform="leetcode")
        )
        if done:
            break
        obs, _, done, _ = env.step(
            CheckPlatformAction(candidate_id=cid, platform="github")
        )
        if done:
            break

        platforms = obs.revealed_data[cid]["platforms"]
        lc_data = platforms.get("leetcode", {})
        gh_data = platforms.get("github", {})

        # Simple heuristic thresholds
        problems = lc_data.get("problems_solved", 0)
        repos = gh_data.get("repos", 0)
        hard = lc_data.get("hard", 0)

        if problems >= 200 and repos >= 25 and hard >= 15:
            decision = "shortlist"
        elif problems < 80 and repos < 8:
            decision = "reject"
        else:
            decision = "hold"

        obs, _, done, _ = env.step(
            MakeDecisionAction(candidate_id=cid, decision=decision)
        )

    # Burn any remaining budget so the episode ends and is scored.
    while not done:
        obs, _, done, _ = env.step(
            CheckPlatformAction(candidate_id=cids[0], platform="leetcode")
        )

    score = env.state().episode_score
    return score if score is not None else 0.0
