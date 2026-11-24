"""
RecruitEnv LLM inference script
===============================

Plays one episode against a running RecruitEnv server with an
OpenAI-compatible chat model, then reports the score from ``POST /grader``.

Environment variables:
    API_BASE_URL   OpenAI-compatible endpoint (default: Hugging Face router)
    MODEL_NAME     Model identifier (default: gpt-4o-mini)
    HF_TOKEN       API key (or API_KEY)
    ENV_URL        RecruitEnv server URL (default: http://localhost:7860)
    TASK           Task id: easy | medium | hard (default: easy)
    SEED           Episode seed (default: 42)
    MAX_TOKENS     Completion token cap per call (default: 512)

Output lines: one [START], one [STEP] per environment step, one [END]
with the grader's score.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from typing import Any, Optional

import httpx

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_URL = os.getenv("ENV_URL") or "http://localhost:7860"
TASK_NAME = os.getenv("TASK") or "easy"
SEED = int(os.getenv("SEED") or "42")

BENCHMARK = "recruitenv"
TEMPERATURE = 0.1
MAX_TOKENS = int(os.getenv("MAX_TOKENS") or "512")

# An action chooser maps (observation, history) to one action dict.
ChooseAction = Callable[[dict, list[str]], dict]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# HTTP environment client
# ---------------------------------------------------------------------------


class EnvHTTPClient:
    """Minimal async HTTP client for the RecruitEnv REST API."""

    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=60.0, transport=transport,
        )

    async def reset(self, task_id: str, seed: int) -> dict:
        r = await self._client.post("/reset", json={"task_id": task_id, "seed": seed})
        r.raise_for_status()
        return r.json()

    async def step(self, action: dict) -> tuple[dict | None, str | None]:
        """Return (result, None) on success or (None, error) for a rejected action."""
        r = await self._client.post("/step", json=action)
        if r.status_code in (400, 422):
            return None, _error_detail(r)
        r.raise_for_status()
        return r.json(), None

    async def grade(self) -> dict:
        r = await self._client.post("/grader")
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self._client.aclose()


def _error_detail(r: httpx.Response) -> str:
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    return re.sub(r"\s+", " ", json.dumps(detail) if not isinstance(detail, str) else detail)[:200]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RUBRIC_GUIDE = """How candidates are judged: the published hiring rubric (env/rubric.py).
Sources: github, leetcode, kaggle, the resume's experience section and its
skills section. The education section is never used. h(x, s) = x / (x + s).

Platform sub-scores (each 0..1):
- leetcode = 0.5*sigmoid((contest_rating - 1500)/200) + 0.25*h(hard, 15) + 0.25*global_rank_percentile/100
- github   = 0.6*h(ln(1+stars_received), ln(151)) + 0.4*h(repos, 15)
- kaggle   = 0.5*best_finish_percentile/100 + 0.3*rank_value + 0.2*h(3*gold + 2*silver + bronze, 6)
  (rank_value: Novice 0, Contributor 0.25, Expert 0.5, Master 0.75, Grandmaster 1)
Components:
- technical = (sum of w * sub**-2 over the three platforms) ** -0.5, a
  weighted power mean that sits below the plain mean when they disagree.
  Weights (leetcode/github/kaggle):
  ml_engineer .30/.30/.40, data_scientist .30/.25/.45,
  backend_dev .40/.35/.25, frontend_dev .35/.40/.25.
- experience = h(years, job.experience_years), where years = the resume's
  years_experience, capped at github years_active + 1.5.
- growth = 0.5*(0.5*h(commit_streak_days, 45) + 0.5*h(contributions_last_year, 300))
         + 0.5*h(leetcode problems_solved, 150)
- skill_match = 0.85*(share of the job's required_skills listed in the
  resume's skills) + 0.15*(share of nice_to_have listed)
- total = (weight_technical*technical + weight_experience*experience +
  weight_growth*growth) * (0.75 + 0.25*skill_match), with the job's weights.
An unrevealed source counts as an average candidate's.

The pool is ranked by total: the top `quotas.shortlist` candidates are
shortlist, the bottom `quotas.reject` are reject, the rest are hold.
Identical candidates must get identical decisions. Names carry no
information about ability."""

SYSTEM_PROMPT = """You are a hiring agent screening candidates for one job.
Each step you take ONE action. Revealing a source costs a step, and the
episode ends when every candidate has a decision or the step budget runs out.

Actions (respond with ONE JSON object only, no explanation, no markdown):
1. {"type": "check_platform", "candidate_id": "...", "platform": "github|leetcode|kaggle"}
2. {"type": "read_resume_section", "candidate_id": "...", "section": "education|experience|skills"}
3. {"type": "make_decision", "candidate_id": "...", "decision": "shortlist|hold|reject"}

""" + RUBRIC_GUIDE + """

Decide each candidate by the rubric. Decisions are final. Candidates you
never decide earn no credit, except on the hard task where an undecided
candidate simply counts as not shortlisted."""


def build_user_prompt(step: int, observation: dict, history: list[str]) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    jd = observation.get("job_description", {})
    undecided = [
        c["id"] for c in observation.get("candidates_summary", [])
        if c["id"] not in observation.get("decisions_made", {})
    ]
    return (
        f"Step {step}. Steps remaining: {observation.get('steps_remaining', 0)}. "
        f"Undecided candidates: {len(undecided)}.\n"
        f"Job: {json.dumps(jd, separators=(',', ':'))}\n"
        f"Quotas: {json.dumps(observation.get('quotas', {}))}\n"
        f"Candidates: {json.dumps(observation.get('candidates_summary', []), separators=(',', ':'))}\n"
        f"Revealed data: {json.dumps(observation.get('revealed_data', {}), separators=(',', ':'))}\n"
        f"Decisions made: {json.dumps(observation.get('decisions_made', {}))}\n"
        f"Previous steps:\n{history_block}\n\n"
        f"What is your next action? Respond with JSON only."
    )


# ---------------------------------------------------------------------------
# LLM action extraction
# ---------------------------------------------------------------------------


def parse_action(text: str) -> dict | None:
    """Extract one action object from model output, or None."""
    text = re.sub(r"```json|```", "", text or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def fallback_action(observation: dict) -> dict:
    """Hold the first undecided candidate."""
    decided = set(observation.get("decisions_made", {}))
    for c in observation.get("candidates_summary", []):
        if c["id"] not in decided:
            return {"type": "make_decision", "candidate_id": c["id"], "decision": "hold"}
    first = observation["candidates_summary"][0]["id"]
    return {"type": "check_platform", "candidate_id": first, "platform": "github"}


def make_llm_chooser(client: Any, model: str) -> ChooseAction:
    """Build an action chooser backed by an OpenAI-compatible chat client."""
    def choose(observation: dict, history: list[str]) -> dict:
        step = observation.get("step_number", 0) + 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(step, observation, history)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = completion.choices[0].message.content or ""
            action = parse_action(text)
            if action is not None:
                return action
            print(f"[DEBUG] Unparseable model output: {text[:200]!r}", flush=True)
        except Exception as exc:  # network or provider errors: fall back, keep playing
            print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return fallback_action(observation)

    return choose


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------


async def run_episode(
    env: EnvHTTPClient, choose: ChooseAction, task: str, seed: int,
) -> dict[str, Any]:
    """Play one episode and return the grader's result plus step rewards."""
    observation = await env.reset(task_id=task, seed=seed)
    history: list[str] = []
    rewards: list[float] = []
    # Rejected actions do not consume budget; cap them so a confused model
    # cannot loop forever.
    rejections_left = 2 * observation["steps_remaining"]

    while not observation.get("done", False):
        action = choose(observation, history)
        result, error = await env.step(action)
        if error is not None:
            rejections_left -= 1
            history.append(f"{json.dumps(action)} -> rejected: {error}")
            if rejections_left <= 0:
                action = fallback_action(observation)
                result, error = await env.step(action)
                if error is not None:
                    raise RuntimeError(f"fallback action rejected: {error}")
            else:
                print(f"[DEBUG] action rejected: {error}", flush=True)
                continue

        observation = result["observation"]
        reward = float(result["reward"]["step_reward"])
        done = bool(result["done"])
        rewards.append(reward)
        log_step(step=observation["step_number"], action=json.dumps(action), reward=reward, done=done, error=None)
        history.append(f"Step {observation['step_number']}: {json.dumps(action)}")

    grade = await env.grade()
    return {
        "score": grade["score"],
        "success": bool(grade["breakdown"].get("success", False)),
        "steps": observation["step_number"],
        "rewards": rewards,
        "grade": grade,
    }


async def main() -> None:
    from openai import OpenAI

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvHTTPClient(base_url=ENV_URL)
    result: dict[str, Any] = {"score": 0.0, "success": False, "steps": 0, "rewards": []}

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    try:
        result = await run_episode(env, make_llm_chooser(client, MODEL_NAME), TASK_NAME, SEED)
    except Exception as exc:
        print(f"[DEBUG] Unhandled exception: {exc}", flush=True)
    finally:
        await env.close()
        log_end(success=result["success"], steps=result["steps"], score=result["score"], rewards=result["rewards"])


if __name__ == "__main__":
    asyncio.run(main())
