import asyncio
import os
from typing import List
from openai import OpenAI
from openenv_core.client import HTTPEnvClient
from openenv_core.logging import log_start, log_step, log_end

# --- Required env vars (exact names, exact defaults pattern) ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://heyavasthi-recruitenv.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")  # NO default
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # NO default

# --- Constants ---
TASK_NAME = "easy"
BENCHMARK = "recruitenv"
MAX_STEPS = 40
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.75
TEMPERATURE = 0.1
MAX_TOKENS = 200

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

def build_user_prompt(step, observation, history):
    return f"""Step {step}:
Steps remaining: {observation.get('steps_remaining', 0)}
Candidates: {observation.get('candidates_summary', [])}
Revealed data: {observation.get('revealed_data', {})}
Decisions made: {observation.get('decisions_made', {})}

What is your next action? Respond with JSON only."""

def get_model_action(client, step, observation, history):
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
        import re, json
        text = re.sub(r"```json|```", "", text).strip()
        try:
            return json.loads(text)
        except:
            match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
    # fallback
    return {"type": "make_decision", "candidate_id": "candidate_001", "decision": "hold"}

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Use from_docker_image if LOCAL_IMAGE_NAME set, else connect to live Space
    if LOCAL_IMAGE_NAME:
        from openenv_core.client import HTTPEnvClient
        env = await HTTPEnvClient.from_docker_image(LOCAL_IMAGE_NAME, base_url=API_BASE_URL)
    else:
        from openenv_core.client import HTTPEnvClient
        env = HTTPEnvClient(base_url=API_BASE_URL)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=TASK_NAME, seed=42)
        observation = result if isinstance(result, dict) else result.dict()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if observation.get("done", False):
                break

            action = get_model_action(client, step, observation, history)
            result = await env.step(action)

            if isinstance(result, dict):
                observation = result.get("observation", {})
                reward = result.get("reward", {})
                if isinstance(reward, dict):
                    reward = reward.get("step_reward", 0.0)
                done = result.get("done", False)
            else:
                observation = result.observation
                reward = result.reward or 0.0
                done = result.done

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

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
