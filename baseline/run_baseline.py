"""LLM-powered baseline agent for RecruitEnv (via OpenRouter).

Uses the OpenAI Python SDK pointed at OpenRouter's API endpoint.
Requires OPENROUTER_API_KEY in the environment.  The RecruitEnv
server must be running before you start this script.

Usage:
    python baseline/run_baseline.py
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()


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
        f"STRICT RULES — follow exactly:\n"
        f"\n"
        f"SIGNALS GUIDE:\n"
        f"- GitHub stars > 200 AND LeetCode solved > 250: shortlist\n"
        f"- GitHub stars < 20 AND LeetCode solved < 100: reject\n"
        f"- Everything in between: hold\n"
        f"\n"
        f"EFFICIENCY RULES:\n"
        f"- Check MAXIMUM 2 signals per candidate (github + leetcode only)\n"
        f"- Immediately after 2 checks: make_decision for that candidate\n"
        f"- Never check resume unless you have 0 other signals\n"
        f"- Never score_dimension — skip it entirely, go straight to make_decision\n"
        f"\n"
        f"DECISION RULES:\n"
        f"- You MUST make a decision after every 1-2 checks\n"
        f"- Do NOT shortlist everyone — only top candidates deserve shortlist\n"
        f"- Reject weak candidates aggressively\n"
        f"- Hold borderline candidates\n"
        f"\n"
        f"ORDER: check_platform github -> check_platform leetcode -> make_decision\n"
        f"Repeat for each candidate. Never deviate from this order.\n"
        f"\n"
        f"Respond with ONLY a single JSON object. No explanation. No markdown."
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


def extract_action(text: str, fallback_candidate: str) -> dict:
    """Bulletproof JSON extraction from LLM output.

    Tries: direct parse, markdown-fence stripping, regex extraction.
    Always returns a valid action dict — never None.
    """
    fallback = {
        "type": "make_decision",
        "candidate_id": fallback_candidate,
        "decision": "hold",
    }

    if not text:
        return fallback

    # Strip markdown fences
    text = re.sub(r"```json|```", "", text).strip()

    # Try direct parse
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

    # True fallback
    return fallback


def _first_undecided(obs: dict) -> str:
    """Return the candidate_id of the first undecided candidate."""
    decided = set(obs.get("decisions_made", {}).keys())
    for c in obs["candidates_summary"]:
        if c["id"] not in decided:
            return c["id"]
    return obs["candidates_summary"][0]["id"]


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
    loop_iter = 0

    while not done and steps_used < max_steps:
        # Reset conversation history every 3 iterations to prevent
        # context overflow (especially important for hard_task).
        if loop_iter % 3 == 0:
            messages = [{"role": "system", "content": system_prompt}]
        loop_iter += 1

        # a. Format observation
        user_msg = build_observation_message(obs)
        messages.append({"role": "user", "content": user_msg})

        # b. Call OpenRouter
        fallback_cid = _first_undecided(obs)
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=150,
                temperature=0.0,
            )
            action_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  [step {steps_used}] LLM error: {e}")
            action_text = ""

        # c. Parse action (bulletproof — always returns a valid dict)
        action = extract_action(action_text, fallback_cid)

        # Log raw LLM output
        print(f"  [step {steps_used}] Raw LLM output: {action_text[:300]}")

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
            action = extract_action("", fallback_cid)
            messages[-1] = {"role": "assistant", "content": json.dumps(action)}
            r = http.post(f"{BASE_URL}/step", json=action)
            r.raise_for_status()
            step_result = r.json()

        obs = step_result["observation"]
        reward = step_result["reward"]
        done = step_result["done"]
        steps_used = obs["step_number"]

        # Verbose per-step logging
        action_type = action.get("type", "?")
        cid = action.get("candidate_id", "?")
        detail = (
            action.get("section")
            or action.get("platform")
            or action.get("dimension")
            or action.get("decision")
            or ""
        )
        print(f"  [step {steps_used}] Action: {action_type} | Candidate: {cid} | Detail: {detail}")
        print(f"  [step {steps_used}] Reward: {reward.get('step_reward', 0.0)} | Cumulative: {reward.get('cumulative_reward', 0.0)} | Done: {done}")

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
    correct_count = breakdown.get("correct", "?")
    total_count = breakdown.get("total", "?")
    print(f"  Correct decisions: {correct_count}/{total_count}")

    # Step 6 — Per-candidate summary
    _print_candidate_summary(http, obs, breakdown)

    return {
        "score": score,
        "steps_used": steps_used,
        "max_steps": max_steps,
        "breakdown": breakdown,
    }


def _print_candidate_summary(
    http: httpx.Client,
    final_obs: dict,
    breakdown: dict[str, Any],
) -> None:
    """Print full per-candidate breakdown table."""
    decisions = breakdown.get("decisions", {})
    ground_truth = breakdown.get("ground_truth", {})
    revealed = final_obs.get("revealed_data", {})
    scores_recorded = final_obs.get("scores_recorded", {})

    if not decisions and not ground_truth:
        return

    print(f"\n  {'='*115}")
    print(f"  CANDIDATE BREAKDOWN")
    print(f"  {'='*115}")
    print(
        f"  {'Candidate':<18}"
        f"{'Signals Checked':<40}"
        f"{'Dims Scored':<30}"
        f"{'Decision':<12}"
        f"{'Truth':<10}"
        f"{'Match':>5}"
    )
    print(f"  {'-'*115}")

    all_cids = sorted(set(list(ground_truth.keys()) + list(decisions.keys())))
    match_count = 0
    for cid in all_cids:
        # Signals gathered
        cid_data = revealed.get(cid, {})
        platforms = cid_data.get("platforms", [])
        sections = cid_data.get("resume_sections", [])
        signals = []
        if platforms:
            signals.extend(platforms)
        if sections:
            signals.extend(f"resume:{s}" for s in sections)
        signals_str = ", ".join(signals) if signals else "(none)"

        # Dimensions scored
        cid_scores = scores_recorded.get(cid, {})
        if cid_scores:
            dims_str = ", ".join(
                f"{d}={v:.2f}" for d, v in sorted(cid_scores.items())
            )
        else:
            dims_str = "(none)"

        decision = decisions.get(cid, "(undecided)")
        truth = ground_truth.get(cid, "?")
        is_match = decision == truth
        match_str = "Y" if is_match else "N"
        if is_match:
            match_count += 1

        print(
            f"  {cid:<18}"
            f"{signals_str:<40}"
            f"{dims_str:<30}"
            f"{decision:<12}"
            f"{truth:<10}"
            f"{match_str:>5}"
        )

    print(f"  {'-'*115}")
    print(f"  Total: {match_count}/{len(all_cids)} correct")
    print(f"  {'='*115}")


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
