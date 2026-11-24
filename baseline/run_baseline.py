"""LLM baseline agent for RecruitEnv.

Python gathers evidence; the model decides from it. For each task the script
reveals sources breadth-first in the role's priority order (every
candidate's first source, then every candidate's second, ...) while keeping
enough budget for the decisions. It then sends the job description, the
quotas, the published rubric and everything revealed to the model in ONE
request and asks for a decision per candidate. If the model's answer is
unusable, the rubric itself decides (``env.rubric.estimate`` + quotas).

Requires OPENAI_API_KEY (and OPENAI_BASE_URL for an OpenAI-compatible
endpoint). The RecruitEnv server must be running.

Usage:
    python baseline/run_baseline.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import rubric  # noqa: E402
from env.models import JobDescription  # noqa: E402
from env.policies import SOURCE_PRIORITY  # noqa: E402
from env.profile_factory import fill_quotas  # noqa: E402
from inference import RUBRIC_GUIDE  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("RECRUITENV_URL", "http://localhost:7860")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
SEED = int(os.getenv("SEED", "42"))

TASKS = [
    ("easy", "easy_task"),
    ("medium", "medium_task"),
    ("hard", "hard_task"),
]

_ACTION = {
    "github": ("check_platform", "platform"),
    "leetcode": ("check_platform", "platform"),
    "kaggle": ("check_platform", "platform"),
    "experience": ("read_resume_section", "section"),
    "skills": ("read_resume_section", "section"),
}


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------


def check_api_key() -> OpenAI:
    """Verify OPENAI_API_KEY is set and return a client (OPENAI_BASE_URL is honoured)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    return OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)


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


def _post_step(http: httpx.Client, action: dict) -> dict:
    r = http.post(f"{BASE_URL}/step", json=action)
    r.raise_for_status()
    return r.json()


def _decisions_needed(obs: dict) -> int:
    if obs["task_id"] == "hard":  # undecided counts as not shortlisted; +1 for a tie at the cut
        return obs["quotas"]["shortlist"] + 1
    return len(obs["candidates_summary"]) - len(obs["decisions_made"])


def gather(http: httpx.Client, obs: dict) -> dict:
    """Reveal sources breadth-first in the role's priority order, keeping budget to decide."""
    order = SOURCE_PRIORITY[obs["job_description"]["role_type"]]
    for source in order:
        for c in obs["candidates_summary"]:
            if obs["steps_remaining"] <= _decisions_needed(obs):
                return obs
            kind, key = _ACTION[source]
            obs = _post_step(http, {"type": kind, "candidate_id": c["id"], key: source})["observation"]
    return obs


def rubric_decisions(obs: dict) -> dict[str, str]:
    """The published rubric applied to what was revealed (unrevealed inputs use priors)."""
    jd = JobDescription(**obs["job_description"])
    est = {c["id"]: rubric.estimate(obs["revealed_data"].get(c["id"]), jd).total
           for c in obs["candidates_summary"]}
    return fill_quotas(est, obs["quotas"])


def llm_decisions(client: OpenAI, obs: dict) -> dict[str, str] | None:
    """One request: the model decides every candidate from the revealed evidence."""
    ids = [c["id"] for c in obs["candidates_summary"]]
    prompt = (
        f"You are deciding candidates for one job.\n\n{RUBRIC_GUIDE}\n\n"
        f"Job description: {json.dumps(obs['job_description'])}\n"
        f"Quotas: {json.dumps(obs['quotas'])}\n"
        f"Candidates: {json.dumps(obs['candidates_summary'])}\n"
        f"Revealed evidence (unrevealed sources are unknown): {json.dumps(obs['revealed_data'])}\n\n"
        f"Return ONE JSON object mapping every candidate id to shortlist, hold or reject, "
        f"respecting the quotas. No explanation."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        text = re.sub(r"```json|```", "", completion.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group() if match else text)
    except Exception as exc:  # network, provider or parse error
        print(f"  LLM error: {exc}")
        return None
    if not isinstance(parsed, dict):
        return None
    out = {cid: str(parsed.get(cid, "")).lower() for cid in ids}
    if any(d not in {"shortlist", "hold", "reject"} for d in out.values()):
        print(f"  LLM answer incomplete: {text[:200]!r}")
        return None
    return out


def _fit_shortlists(obs: dict, decisions: dict[str, str]) -> dict[str, str]:
    """On hard, keep only as many shortlists as the remaining steps allow.

    Shortlists are kept best-first by rubric estimate; a group of equal
    estimates (identical candidates) is kept or dropped as a whole, so two
    identical candidates never end up decided differently.
    """
    budget = obs["steps_remaining"]
    jd = JobDescription(**obs["job_description"])
    est = {cid: rubric.estimate(obs["revealed_data"].get(cid), jd).total
           for cid, d in decisions.items() if d == "shortlist"}
    kept: list[str] = []
    for value in sorted(set(est.values()), reverse=True):
        group = sorted(c for c in est if est[c] == value)
        if len(kept) + len(group) > budget:
            break
        kept.extend(group)
    return {cid: ("shortlist" if cid in kept else "hold") if d == "shortlist" else d
            for cid, d in decisions.items()}


def run_task(client: OpenAI, http: httpx.Client, task_id: str) -> dict[str, Any]:
    """Run one episode: gather, ask the model once, decide, grade."""
    print(f"\n{'='*50}\n  Task: {task_id} (seed={SEED})\n{'='*50}")
    r = http.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": SEED})
    r.raise_for_status()
    obs = r.json()
    max_steps = obs["steps_remaining"]
    print(f"  Candidates: {len(obs['candidates_summary'])}  Step budget: {max_steps}")

    obs = gather(http, obs)
    print(f"  Revealed {obs['step_number']} sources")
    decisions = llm_decisions(client, obs)
    source = "model"
    if decisions is None:
        decisions, source = rubric_decisions(obs), "rubric fallback"
    print(f"  Decisions from: {source}")

    if task_id == "hard":
        decisions = _fit_shortlists(obs, decisions)

    rank = {"shortlist": 0, "reject": 1, "hold": 2}
    done = False
    for cid in sorted(decisions, key=lambda c: (rank[decisions[c]], c)):
        if done:
            break
        if task_id == "hard" and decisions[cid] != "shortlist":
            continue  # undecided counts as not shortlisted
        result = _post_step(http, {"type": "make_decision", "candidate_id": cid, "decision": decisions[cid]})
        obs, done = result["observation"], result["done"]
    while not done:  # hard: burn the remaining budget so the episode is scored
        first = obs["candidates_summary"][0]["id"]
        result = _post_step(http, {"type": "read_resume_section", "candidate_id": first, "section": "education"})
        obs, done = result["observation"], result["done"]

    grade = http.post(f"{BASE_URL}/grader").json()
    score, breakdown = grade["score"], grade["breakdown"]
    status = "PASS" if breakdown.get("success") else "FAIL"
    print(f"\n  Score: {score:.4f} (threshold: {breakdown.get('threshold', 0):.2f}) [{status}]")
    for check in breakdown.get("checks", []):
        print(f"    {check['name']:<36} {check['earned']:.3f}/{check['weight']:.2f} {check['status']}")
    for gate in breakdown.get("gates", []):
        print(f"    gate {gate['name']:<31} {'passed' if gate['passed'] else 'FAILED'}")
    _print_candidate_summary(http, obs, breakdown)
    return {
        "score": score,
        "steps_used": obs.get("step_number", 0),
        "max_steps": max_steps,
        "decided_by": source,
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
                "decided_by": data.get("decided_by"),
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
