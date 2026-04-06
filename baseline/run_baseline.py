"""LLM-powered baseline agent for RecruitEnv (via OpenRouter).

Uses the OpenAI Python SDK pointed at OpenRouter's API endpoint.
Requires OPENROUTER_API_KEY in the environment.  The RecruitEnv
server must be running before you start this script.

Usage:
    python baseline/run_baseline.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("RECRUITENV_URL", "http://localhost:7860")
MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
SEED = 42

TASKS = [
    ("easy", "easy_task"),
    ("medium", "medium_task"),
    ("hard", "hard_task"),
]

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_candidates_summary(candidates: list[dict[str, str]]) -> str:
    """Format candidate list as a clean numbered list."""
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(f"  {i}. {c['id']} — {c['name']}")
    return "\n".join(lines)


def format_revealed_data(revealed: dict[str, Any]) -> str:
    """Format incrementally revealed candidate info as readable text."""
    if not revealed:
        return "  (nothing revealed yet)"

    sections = []
    for cid, data in revealed.items():
        parts = [f"  [{cid}]"]

        resume_sections = data.get("resume_sections", [])
        if resume_sections:
            parts.append(f"    Resume sections read: {', '.join(resume_sections)}")

        platforms = data.get("platforms", [])
        if platforms:
            parts.append(f"    Platforms checked: {', '.join(platforms)}")

        # Include any nested data (platform stats, resume content)
        for key, val in data.items():
            if key in ("resume_sections", "platforms"):
                continue
            if isinstance(val, dict):
                parts.append(f"    {key}: {json.dumps(val, default=str)}")
            elif val:
                parts.append(f"    {key}: {val}")

        sections.append("\n".join(parts))

    return "\n".join(sections)


def build_system_prompt(obs: dict) -> str:
    """Build the system prompt from the initial observation."""
    jd = obs["job_description"]
    steps_remaining = obs["steps_remaining"]

    return (
        f"You are a hiring agent evaluating candidates for: {jd['role']}\n"
        f"\n"
        f"Required skills: {', '.join(jd['required_skills'])}\n"
        f"Seniority: {jd['seniority']}\n"
        f"Step budget: {steps_remaining}\n"
        f"\n"
        f"Available actions (respond with exactly ONE as valid JSON):\n"
        f"\n"
        f'1. {{"type": "read_resume_section", "candidate_id": "...", "section": "education|experience|skills"}}\n'
        f'2. {{"type": "check_platform", "candidate_id": "...", "platform": "github|leetcode|kaggle"}}\n'
        f'3. {{"type": "score_dimension", "candidate_id": "...", "dimension": "technical|experience|growth", "score": 0.0-1.0}}\n'
        f'4. {{"type": "make_decision", "candidate_id": "...", "decision": "shortlist|hold|reject"}}\n'
        f"\n"
        f"Strategy: gather signals efficiently, then decide. "
        f"You have limited steps so prioritize wisely.\n"
        f"Respond with ONLY the JSON action, no explanation."
    )


def build_observation_message(obs: dict) -> str:
    """Format the current observation as a human-readable user message."""
    step = obs["step_number"]
    remaining = obs["steps_remaining"]
    candidates = format_candidates_summary(obs["candidates_summary"])
    revealed = format_revealed_data(obs.get("revealed_data", {}))
    decisions = obs.get("decisions_made", {})
    scores = obs.get("scores_recorded", {})

    decisions_str = (
        json.dumps(decisions, indent=2) if decisions else "(none yet)"
    )
    scores_str = (
        json.dumps(scores, indent=2) if scores else "(none yet)"
    )

    return (
        f"Step {step}, {remaining} steps remaining\n"
        f"\n"
        f"Candidates:\n{candidates}\n"
        f"\n"
        f"What you know so far:\n{revealed}\n"
        f"\n"
        f"Decisions made: {decisions_str}\n"
        f"Scores recorded: {scores_str}\n"
        f"\n"
        f"What is your next action? Respond with JSON only."
    )


def parse_action(text: str) -> dict | None:
    """Parse an action JSON from LLM output, stripping markdown fences."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{[^{}]+\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def fallback_action(obs: dict) -> dict:
    """Return a safe fallback: 'hold' for the first undecided candidate."""
    decided = set(obs.get("decisions_made", {}).keys())
    for c in obs["candidates_summary"]:
        if c["id"] not in decided:
            return {
                "type": "make_decision",
                "candidate_id": c["id"],
                "decision": "hold",
            }
    # All decided — shouldn't reach here
    return {
        "type": "make_decision",
        "candidate_id": obs["candidates_summary"][0]["id"],
        "decision": "hold",
    }


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------


def check_api_key() -> OpenAI:
    """Verify OPENROUTER_API_KEY is set and return an OpenAI client."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.")
        print("Get a free key at https://openrouter.ai/keys")
        sys.exit(1)

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def check_server() -> None:
    """Verify the RecruitEnv server is reachable."""
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        r.raise_for_status()
    except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException):
        print(f"ERROR: Cannot reach RecruitEnv server at {BASE_URL}")
        print("Start the server first:")
        print("  uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Agent loop for one task
# ---------------------------------------------------------------------------


def run_task(
    client: OpenAI,
    http: httpx.Client,
    task_id: str,
) -> dict[str, Any]:
    """Run one full episode and return the result dict."""
    print(f"\n{'='*50}")
    print(f"  Task: {task_id} (seed={SEED})")
    print(f"{'='*50}")

    # Step 1 — Reset
    r = http.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": SEED})
    r.raise_for_status()
    obs = r.json()
    max_steps = obs["steps_remaining"]

    print(f"  Candidates: {len(obs['candidates_summary'])}")
    print(f"  Step budget: {max_steps}")

    # Step 2 — Build system prompt
    system_prompt = build_system_prompt(obs)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Step 3 — Agent loop
    done = False
    steps_used = 0

    while not done and steps_used < max_steps:
        # a. Format observation
        user_msg = build_observation_message(obs)
        messages.append({"role": "user", "content": user_msg})

        # b. Call OpenRouter
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=200,
                temperature=0.1,
            )
            action_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  [step {steps_used}] LLM error: {e}")
            action_text = ""

        # c. Parse action
        action = parse_action(action_text)
        if action is None:
            print(f"  [step {steps_used}] Parse failed, using fallback. Raw: {action_text[:80]}")
            action = fallback_action(obs)

        # Append assistant message
        messages.append({"role": "assistant", "content": action_text or json.dumps(action)})

        # d. POST /step
        try:
            r = http.post(f"{BASE_URL}/step", json=action)
            r.raise_for_status()
            step_result = r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            print(f"  [step {steps_used}] Step error: {detail}")
            # Use fallback and retry
            action = fallback_action(obs)
            messages[-1] = {"role": "assistant", "content": json.dumps(action)}
            r = http.post(f"{BASE_URL}/step", json=action)
            r.raise_for_status()
            step_result = r.json()

        obs = step_result["observation"]
        done = step_result["done"]
        steps_used = obs["step_number"]

        # Progress indicator
        action_type = action.get("type", "?")
        cid = action.get("candidate_id", "?")
        extra = ""
        if action_type == "make_decision":
            extra = f" -> {action.get('decision', '?')}"
        elif action_type == "check_platform":
            extra = f" ({action.get('platform', '?')})"
        elif action_type == "read_resume_section":
            extra = f" ({action.get('section', '?')})"
        print(f"  [step {steps_used:2d}/{max_steps}] {action_type} {cid}{extra}")

    # Step 4 — Grade
    r = http.post(f"{BASE_URL}/grader")
    if r.status_code == 200:
        grade_result = r.json()
        score = grade_result["score"]
        breakdown = grade_result["breakdown"]
    else:
        print(f"  Grader error: {r.text}")
        score = 0.0
        breakdown = {}

    # Step 5 — Print result
    threshold = breakdown.get("threshold", 0.0)
    success = breakdown.get("success", False)
    status = "PASS" if success else "FAIL"
    print(f"\n  Score: {score:.4f} (threshold: {threshold:.2f}) [{status}]")
    print(f"  Steps used: {steps_used}/{max_steps}")
    correct = breakdown.get("correct", "?")
    total = breakdown.get("total", "?")
    print(f"  Correct decisions: {correct}/{total}")

    return {
        "score": score,
        "steps_used": steps_used,
        "max_steps": max_steps,
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_results_table(
    results: dict[str, dict[str, Any]],
    model: str,
) -> None:
    """Print the final results table."""
    avg = sum(r["score"] for r in results.values()) / len(results) if results else 0.0

    w = 54
    print()
    print(f"\u2554{'═' * w}\u2557")
    print(f"\u2551{'RecruitEnv Baseline Results':^{w}}\u2551")
    print(f"\u2551{f'Model: {model}':^{w}}\u2551")
    print(f"\u2560{'═' * w}\u2563")
    print(f"\u2551 {'Task':<16}{'Score':>8}{'Steps Used':>14}{'Status':>12}   \u2551")
    print(f"\u2551{'\u2500' * w}\u2551")

    for label, data in results.items():
        score = data["score"]
        steps = data["steps_used"]
        mx = data["max_steps"]
        threshold = data["breakdown"].get("threshold", 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"\u2551 {label:<16}{score:>8.4f}{steps:>10d}/{mx:<4d}{status:>10}   \u2551")

    print(f"\u2551{'\u2500' * w}\u2551")
    print(f"\u2551 {'Average':<16}{avg:>8.4f}{'':>14}{'':>12}   \u2551")
    print(f"\u255a{'═' * w}\u255d")


def save_scores(results: dict[str, dict[str, Any]], model: str) -> None:
    """Save results to baseline/baseline_scores.json."""
    output = {
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "scores": {
            label: {
                "score": data["score"],
                "steps_used": data["steps_used"],
                "breakdown": data["breakdown"],
            }
            for label, data in results.items()
        },
        "average_score": (
            sum(r["score"] for r in results.values()) / len(results)
            if results
            else 0.0
        ),
    }

    path = os.path.join(os.path.dirname(__file__), "baseline_scores.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nScores saved to {path}")


def main() -> None:
    """Run the LLM baseline agent on all tasks."""
    print(f"RecruitEnv Baseline Agent")
    print(f"Model: {MODEL}")
    print(f"Server: {BASE_URL}")

    # Startup checks
    llm_client = check_api_key()
    check_server()
    print("Server reachable. Starting evaluation...\n")

    results: dict[str, dict[str, Any]] = {}

    with httpx.Client(timeout=30.0) as http:
        for task_id, label in TASKS:
            result = run_task(llm_client, http, task_id)
            results[label] = result

    print_results_table(results, MODEL)
    save_scores(results, MODEL)


if __name__ == "__main__":
    main()
