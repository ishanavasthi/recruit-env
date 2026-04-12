"""
RecruitEnv Inference Script
===================================
MANDATORY env vars:
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   Docker image name (optional).
"""

import asyncio
import json
import os
import re
from typing import List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (match sample_inference.py pattern exactly)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # NO default

# Environment server URL (our HF Space — plain REST, not WebSocket)
ENV_URL = "https://heyavasthi-recruitenv.hf.space"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_NAME = "easy"
BENCHMARK = "recruitenv"
MAX_STEPS = 40
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.75
TEMPERATURE = 0.1
MAX_TOKENS = 200

# ---------------------------------------------------------------------------
# Logging (inline, matching required stdout format exactly)
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


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# HTTP Environment Client (plain REST — no WebSocket)
# ---------------------------------------------------------------------------


class EnvHTTPClient:
    """Minimal async HTTP client for our REST API endpoints."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=60.0)

    async def reset(self, task_id: str = "easy", seed: int = 42) -> dict:
        r = await self._client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        r.raise_for_status()
        return r.json()

    async def step(self, action: dict) -> dict:
        r = await self._client.post(
            f"{self.base_url}/step",
            json=action,
        )
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a hiring agent evaluating candidates for a job.
You must review candidate profiles and make shortlisting decisions.

Available actions (respond with ONE JSON object only, no explanation):
1. {"type": "check_platform", "candidate_id": "...", "platform": "github|leetcode|kaggle"}
2. {"type": "read_resume_section", "candidate_id": "...", "section": "skills|experience|education"}
3. {"type": "make_decision", "candidate_id": "...", "decision": "shortlist|hold|reject"}

Rules:
- Check github + leetcode for each candidate (max 2 checks per candidate)
- Immediately make_decision after checking signals
- shortlist: stars>200 AND solved>250
- reject: stars<20 AND solved<100
- hold: everything else
- Respond with ONLY valid JSON. No markdown. No explanation."""


def build_user_prompt(step: int, observation: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return (
        f"Step {step}:\n"
        f"Steps remaining: {observation.get('steps_remaining', 0)}\n"
        f"Candidates: {observation.get('candidates_summary', [])}\n"
        f"Revealed data: {observation.get('revealed_data', {})}\n"
        f"Decisions made: {observation.get('decisions_made', {})}\n"
        f"Previous steps:\n{history_block}\n\n"
        f"What is your next action? Respond with JSON only."
    )


# ---------------------------------------------------------------------------
# LLM action extraction
# ---------------------------------------------------------------------------


def get_model_action(
    client: OpenAI, step: int, observation: dict, history: List[str]
) -> dict:
    """Call the LLM and extract a valid action JSON."""
    user_prompt = build_user_prompt(step, observation, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences
        text = re.sub(r"```json|```", "", text).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        # Extract first {...} block
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
    # Fallback: hold for first undecided candidate
    decided = set(observation.get("decisions_made", {}).keys())
    for c in observation.get("candidates_summary", []):
        if c["id"] not in decided:
            return {"type": "make_decision", "candidate_id": c["id"], "decision": "hold"}
    return {"type": "make_decision", "candidate_id": "candidate_001", "decision": "hold"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    # LLM client — uses API_BASE_URL (the LLM router endpoint)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Environment client — plain HTTP to our HF Space REST API
    env = EnvHTTPClient(base_url=ENV_URL)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = await env.reset(task_id=TASK_NAME, seed=42)
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if observation.get("done", False):
                break

            action = get_model_action(client, step, observation, history)
            result = await env.step(action)

            observation = result.get("observation", {})
            reward = result.get("reward", {})
            if isinstance(reward, dict):
                reward = reward.get("step_reward", 0.0)
            done = result.get("done", False)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            error = None

            log_step(step=step, action=str(action), reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Unhandled exception: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
