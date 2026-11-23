"""LLM-powered baseline agent for RecruitEnv.

Uses the OpenAI Python SDK with the official OpenAI API.
Requires OPENAI_API_KEY in the environment.  The RecruitEnv
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
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
    """Format revealed candidate data (values included) as readable text."""
    if not revealed:
        return "  (nothing revealed yet)"

    sections = []
    for cid, data in revealed.items():
        parts = [f"  [{cid}]"]
        for name, section in data.get("resume_sections", {}).items():
            parts.append(f"    resume.{name}: {json.dumps(section, default=str)}")
        for platform, stats in data.get("platforms", {}).items():
            parts.append(f"    {platform}: {json.dumps(stats, default=str)}")
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
        f'3. {{"type": "make_decision", "candidate_id": "...", "decision": "shortlist|hold|reject"}}\n'
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

    decisions_str = (
        json.dumps(decisions, indent=2) if decisions else "(none yet)"
    )

    return (
        f"Step {step}, {remaining} steps remaining\n"
        f"\n"
        f"Candidates:\n{candidates}\n"
        f"\n"
        f"What you know so far:\n{revealed}\n"
        f"\n"
        f"Decisions made: {decisions_str}\n"
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
    """Verify OPENAI_API_KEY is set and return an OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    return OpenAI(api_key=api_key)


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


def _do_step(
    http: httpx.Client, action: dict, obs: dict
) -> tuple[dict, dict, bool]:
    """POST /step, log, return (obs, reward, done). Fallback on error."""
    steps_before = obs["step_number"]
    try:
        r = http.post(f"{BASE_URL}/step", json=action)
        r.raise_for_status()
        result = r.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        print(f"  [step {steps_before}] Step error: {detail}")
        # Fallback: hold on first undecided
        fallback = {
            "type": "make_decision",
            "candidate_id": _first_undecided(obs),
            "decision": "hold",
        }
        r = http.post(f"{BASE_URL}/step", json=fallback)
        r.raise_for_status()
        result = r.json()
        action = fallback

    new_obs = result["observation"]
    reward = result["reward"]
    done = result["done"]
    step_num = new_obs["step_number"]

    # Verbose per-step logging
    action_type = action.get("type", "?")
    cid = action.get("candidate_id", "?")
    act_detail = (
        action.get("section")
        or action.get("platform")
        or action.get("decision")
        or ""
    )
    print(f"  [step {step_num}] Action: {action_type} | Candidate: {cid} | Detail: {act_detail}")
    print(
        f"  [step {step_num}] Reward: {reward.get('step_reward', 0.0)} "
        f"| Cumulative: {reward.get('cumulative_reward', 0.0)} | Done: {done}"
    )
    return new_obs, reward, done


def _ask_llm_decision(
    client: OpenAI,
    cid: str,
    gh_data: dict,
    lc_data: dict,
    jd: dict,
) -> str:
    """Ask the LLM only for the decision given platform stats. Returns shortlist/hold/reject."""
    prompt = (
        f"You are evaluating candidate {cid} for: {jd.get('role', 'Software Engineer')}\n"
        f"\n"
        f"GitHub stats: {json.dumps(gh_data)}\n"
        f"LeetCode stats: {json.dumps(lc_data)}\n"
        f"\n"
        f"Rules:\n"
        f"- GitHub stars > 200 AND LeetCode solved > 250: shortlist\n"
        f"- GitHub stars < 20 AND LeetCode solved < 100: reject\n"
        f"- Everything in between: hold\n"
        f"\n"
        f"Respond with EXACTLY one word: shortlist, hold, or reject"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,  # room for reasoning models; output is one word
            temperature=0.0,
        )
        raw = (completion.choices[0].message.content or "").strip().lower()
        print(f"  [step ?] Raw LLM output: {raw[:300]}")
        # Extract a valid decision from the response
        for choice in ("shortlist", "reject", "hold"):
            if choice in raw:
                return choice
    except Exception as e:
        print(f"  LLM decision error: {e}")
    return "hold"


def _rule_based_decision(gh_data: dict, lc_data: dict) -> str:
    """Pure rule-based fallback using profile stat thresholds."""
    if not gh_data and not lc_data:
        return "hold"  # nothing revealed: no evidence either way

    stars = gh_data.get("stars_received", 0)
    repos = gh_data.get("repos", 0)
    contribs = gh_data.get("contributions_last_year", 0)
    solved = lc_data.get("problems_solved", 0)
    hard = lc_data.get("hard", 0)

    has_lc = bool(lc_data)

    if has_lc:
        # Full signal: both github + leetcode
        if stars >= 200 and solved >= 250:
            return "shortlist"
        if repos >= 30 and solved >= 200 and hard >= 15:
            return "shortlist"
        if stars < 20 and solved < 100:
            return "reject"
        if repos < 8 and solved < 80:
            return "reject"
    else:
        # Github-only signal: use repos/stars/streak/contributions
        # Senior tier: 30-60 repos, 200-1000 stars, 500-1500 contribs
        if repos >= 30 and stars >= 200:
            return "shortlist"
        if repos >= 30 and contribs >= 500:
            return "shortlist"
        # Junior tier: <10 repos, <20 stars, <100 contribs
        if repos < 10 and stars < 20:
            return "reject"
        if repos < 10 and contribs < 100:
            return "reject"

    return "hold"


def run_task(
    client: OpenAI,
    http: httpx.Client,
    task_id: str,
) -> dict[str, Any]:
    """Run one full episode using a structured candidate loop.

    Instead of letting the LLM choose which action to take each step,
    Python controls the sequence: for each candidate, check github,
    optionally check leetcode (if budget allows), then decide.
    The LLM is only asked for the decision; a rule-based fallback
    handles tight budgets.
    """
    print(f"\n{'='*50}")
    print(f"  Task: {task_id} (seed={SEED})")
    print(f"{'='*50}")

    # Step 1 — Reset
    r = http.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": SEED})
    r.raise_for_status()
    obs = r.json()
    max_steps = obs["steps_remaining"]
    jd = obs["job_description"]
    candidates = obs["candidates_summary"]
    n_candidates = len(candidates)

    print(f"  Candidates: {n_candidates}")
    print(f"  Step budget: {max_steps}")

    # Budget strategy per candidate:
    #   full  (>=2.8 steps/candidate): github + leetcode, then decide
    #   quick (>=1.8):                 github, then decide
    #   batch (<1.8):                  github while the budget allows, then decide
    # Every statistic comes from the observation; gathering always costs a step.
    steps_per_candidate = max_steps / n_candidates if n_candidates > 0 else 3
    if steps_per_candidate >= 2.8:
        mode = "full"
    elif steps_per_candidate >= 1.8:
        mode = "quick"
    else:
        mode = "batch"

    print(f"  Steps/candidate: {steps_per_candidate:.1f} -> {mode} mode")

    done = False
    for i, candidate in enumerate(candidates):
        if done:
            break

        cid = candidate["id"]
        cname = candidate.get("name", "")
        print(f"\n  --- Candidate {i+1}/{n_candidates}: {cid} ({cname}) ---")
        undecided = n_candidates - i  # including this candidate

        # a. Check GitHub if the budget still leaves a step for every decision
        if obs["steps_remaining"] > undecided:
            action = {"type": "check_platform", "candidate_id": cid, "platform": "github"}
            obs, _, done = _do_step(http, action, obs)
            if done:
                break

        # b. Check LeetCode in full mode if budget allows
        if mode == "full" and obs["steps_remaining"] > (undecided - 1) * 2 + 2:
            action = {"type": "check_platform", "candidate_id": cid, "platform": "leetcode"}
            obs, _, done = _do_step(http, action, obs)
            if done:
                break

        # c. Read what was revealed from the observation
        platforms = obs.get("revealed_data", {}).get(cid, {}).get("platforms", {})
        gh = platforms.get("github", {})
        lc = platforms.get("leetcode", {})

        # d. Decide — LLM if comfortable, else rules
        budget_tight = mode == "batch" or obs["steps_remaining"] <= (undecided - 1) + 2
        if budget_tight or not gh:
            decision = _rule_based_decision(gh, lc)
            print(f"  [rule-based] Decision: {decision}")
        else:
            decision = _ask_llm_decision(client, cid, gh, lc, jd)
            print(f"  [LLM] Decision: {decision}")

        action = {"type": "make_decision", "candidate_id": cid, "decision": decision}
        obs, _, done = _do_step(http, action, obs)

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
    steps_used = obs.get("step_number", 0)
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

    if not decisions and not ground_truth:
        return

    print(f"\n  {'='*85}")
    print(f"  CANDIDATE BREAKDOWN")
    print(f"  {'='*85}")
    print(
        f"  {'Candidate':<18}"
        f"{'Signals Checked':<40}"
        f"{'Decision':<12}"
        f"{'Truth':<10}"
        f"{'Match':>5}"
    )
    print(f"  {'-'*85}")

    all_cids = sorted(set(list(ground_truth.keys()) + list(decisions.keys())))
    match_count = 0
    for cid in all_cids:
        # Signals gathered
        cid_data = revealed.get(cid, {})
        signals = list(cid_data.get("platforms", {}))
        signals.extend(f"resume:{s}" for s in cid_data.get("resume_sections", {}))
        signals_str = ", ".join(signals) if signals else "(none)"

        decision = decisions.get(cid, "(undecided)")
        truth = ground_truth.get(cid, "?")
        is_match = decision == truth
        match_str = "Y" if is_match else "N"
        if is_match:
            match_count += 1

        print(
            f"  {cid:<18}"
            f"{signals_str:<40}"
            f"{decision:<12}"
            f"{truth:<10}"
            f"{match_str:>5}"
        )

    print(f"  {'-'*85}")
    print(f"  Total: {match_count}/{len(all_cids)} correct")
    print(f"  {'='*85}")


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
    thin = "\u2500" * w
    print()
    print(f"\u2554{'═' * w}\u2557")
    print(f"\u2551{'RecruitEnv Baseline Results':^{w}}\u2551")
    print(f"\u2551{f'Model: {model}':^{w}}\u2551")
    print(f"\u2560{'═' * w}\u2563")
    print(f"\u2551 {'Task':<16}{'Score':>8}{'Steps Used':>14}{'Status':>12}   \u2551")
    print(f"\u2551{thin}\u2551")

    for label, data in results.items():
        score = data["score"]
        steps = data["steps_used"]
        mx = data["max_steps"]
        threshold = data["breakdown"].get("threshold", 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"\u2551 {label:<16}{score:>8.4f}{steps:>10d}/{mx:<4d}{status:>10}   \u2551")

    print(f"\u2551{thin}\u2551")
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
