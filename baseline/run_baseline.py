"""Decide-only model baseline: Python gathers evidence, the model decides.

For each task the script reveals sources breadth-first in the role's
priority order (every candidate's first source, then every candidate's
second, ...) while keeping enough budget for the decisions. It then sends the
job description, the quotas, the published rubric and everything revealed to
the model in ONE request and asks for a decision per candidate.

This isolates one skill: applying the rubric to evidence. The step-by-step
agent in ``inference.py`` also has to choose what to read.

There is no fallback: if the model's answer is unusable after retries, the
episode is reported as incomplete. Runs in process; every episode is logged.

    python baseline/run_baseline.py --model claude-sonnet-5-5 --seed 42

Environment: API_BASE_URL and API_KEY (an OpenAI-compatible endpoint).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import (  # noqa: E402
    MAX_TOKENS,
    MAX_TOKENS_CEILING,
    MODEL_ATTEMPTS,
    MODEL_NAME,
    RUBRIC_GUIDE,
    IncompleteEpisode,
    Usage,
    make_client,
)
from recruitenv import rubric  # noqa: E402
from recruitenv.environment import RecruitmentEnvironment  # noqa: E402
from recruitenv.models import JobDescription  # noqa: E402
from recruitenv.policies import SOURCE_PRIORITY  # noqa: E402
from recruitenv.profile_factory import fill_quotas  # noqa: E402
from recruitenv.trajectory import LoggedEnvironment  # noqa: E402

TASKS = ("easy", "medium", "hard")

_ACTION = {
    "github": ("check_platform", "platform"),
    "leetcode": ("check_platform", "platform"),
    "kaggle": ("check_platform", "platform"),
    "experience": ("read_resume_section", "section"),
    "skills": ("read_resume_section", "section"),
}


def _step(env: Any, action: dict) -> tuple[dict, bool]:
    from pydantic import TypeAdapter

    from recruitenv.models import Action

    obs, _, done, _ = env.step(TypeAdapter(Action).validate_python(action))
    return obs.model_dump(mode="json"), done


def _decisions_needed(obs: dict) -> int:
    if obs["task_id"] == "hard":  # undecided counts as not shortlisted; +1 for a tie at the cut
        return obs["quotas"]["shortlist"] + 1
    return len(obs["candidates_summary"]) - len(obs["decisions_made"])


def gather(env: Any, obs: dict) -> dict:
    """Reveal sources breadth-first in the role's priority order, keeping budget to decide."""
    order = SOURCE_PRIORITY[obs["job_description"]["role_type"]]
    for source in order:
        for c in obs["candidates_summary"]:
            if obs["steps_remaining"] <= _decisions_needed(obs):
                return obs
            kind, key = _ACTION[source]
            obs, _ = _step(env, {"type": kind, "candidate_id": c["id"], key: source})
    return obs


def rubric_decisions(obs: dict) -> dict[str, str]:
    """The published rubric applied to what was revealed (unrevealed inputs use priors)."""
    jd = JobDescription(**obs["job_description"])
    est = {c["id"]: rubric.estimate(obs["revealed_data"].get(c["id"]), jd).total
           for c in obs["candidates_summary"]}
    return fill_quotas(est, obs["quotas"])


def decision_prompt(obs: dict) -> str:
    return (
        f"You are deciding candidates for one job.\n\n{RUBRIC_GUIDE}\n\n"
        f"Job description: {json.dumps(obs['job_description'])}\n"
        f"Quotas: {json.dumps(obs['quotas'])}\n"
        f"Candidates: {json.dumps(obs['candidates_summary'])}\n"
        f"Revealed evidence (unrevealed sources are unknown): {json.dumps(obs['revealed_data'])}\n\n"
        f"Return ONE JSON object mapping every candidate id to shortlist, hold or reject, "
        f"respecting the quotas. No explanation."
    )


def parse_decisions(text: str, ids: list[str]) -> dict[str, str] | None:
    text = re.sub(r"```json|```", "", text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        parsed = json.loads(match.group() if match else text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    out = {cid: str(parsed.get(cid, "")).lower() for cid in ids}
    if any(d not in {"shortlist", "hold", "reject"} for d in out.values()):
        return None
    return out


def llm_decisions(client: Any, model: str, obs: dict, usage: Usage, *,
                  max_tokens: int = MAX_TOKENS, attempts: int = MODEL_ATTEMPTS,
                  backoff_s: float = 2.0) -> dict[str, str]:
    """One request: the model decides every candidate. Raises IncompleteEpisode."""
    ids = [c["id"] for c in obs["candidates_summary"]]
    cap, problem = max_tokens, "no attempt"
    for attempt in range(attempts):
        try:
            completion = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": decision_prompt(obs)}],
                max_tokens=cap, temperature=0.0,
            )
        except Exception as exc:
            usage.api_errors += 1
            problem = f"model request failed: {exc}"[:300]
            time.sleep(backoff_s * (2 ** attempt))
            continue
        usage.calls += 1
        u = getattr(completion, "usage", None)
        if u is not None:
            usage.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
            usage.completion_tokens += getattr(u, "completion_tokens", 0) or 0
        choice = completion.choices[0]
        decisions = parse_decisions(choice.message.content or "", ids)
        if decisions is not None:
            return decisions
        if getattr(choice, "finish_reason", None) == "length":
            usage.length_cutoffs += 1
            problem = f"reply cut off at max_tokens={cap}"
            cap = min(cap * 2, MAX_TOKENS_CEILING)
        else:
            usage.parse_failures += 1
            problem = f"unusable answer: {(choice.message.content or '')[:200]!r}"
    raise IncompleteEpisode(problem)


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


def run_task(client: Any, model: str, task_id: str, seed: int, *,
             log_path: Path | None = None, **llm_kw: Any) -> dict[str, Any]:
    """Gather, ask the model once, decide, grade. Returns ``incomplete=True`` on failure."""
    core = RecruitmentEnvironment()
    env: Any = LoggedEnvironment(core, log_path, agent="decide_only", model=model) if log_path else core
    usage = Usage()
    t0 = time.time()
    obs = env.reset(task_id, seed).model_dump(mode="json")
    obs = gather(env, obs)
    try:
        decisions = llm_decisions(client, model, obs, usage, **llm_kw)
    except IncompleteEpisode as exc:
        if isinstance(env, LoggedEnvironment):
            env.close()
        return {"task": task_id, "seed": seed, "incomplete": True, "reason": str(exc),
                "usage": {**usage.as_dict(), "seconds": round(time.time() - t0, 1)}}
    if task_id == "hard":
        decisions = _fit_shortlists(obs, decisions)
    rank = {"shortlist": 0, "reject": 1, "hold": 2}
    done = False
    for cid in sorted(decisions, key=lambda c: (rank[decisions[c]], c)):
        if done:
            break
        if task_id == "hard" and decisions[cid] != "shortlist":
            continue  # undecided counts as not shortlisted
        obs, done = _step(env, {"type": "make_decision", "candidate_id": cid, "decision": decisions[cid]})
    while not done:  # hard: spend the remaining budget so the episode is scored
        first = obs["candidates_summary"][0]["id"]
        obs, done = _step(env, {"type": "read_resume_section", "candidate_id": first, "section": "education"})
    grade = core.grade_report()
    usage.seconds = round(time.time() - t0, 1)
    return {"task": task_id, "seed": seed, "incomplete": False, "score": grade["score"],
            "success": grade["success"], "grade": grade, "usage": usage.as_dict()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-dir", default="outputs/decide_only")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    client = make_client()
    failed = False
    for task in TASKS:
        log = None if args.no_log else Path(args.log_dir) / f"{args.model}-{task}-s{args.seed}.jsonl"
        r = run_task(client, args.model, task, args.seed, log_path=log)
        if r["incomplete"]:
            failed = True
            print(f"{task:<7} incomplete: {r['reason']}")
        else:
            print(f"{task:<7} score={r['score']:.3f} {'PASS' if r['success'] else 'FAIL'}"
                  f"  calls={r['usage']['calls']} tokens={r['usage']['completion_tokens']}")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
