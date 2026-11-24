"""
RecruitEnv model agent
======================

Plays episodes step by step with an OpenAI-compatible chat model. Runs the
environment in process by default; ``--url`` plays against a running server
over its websocket instead. Every local episode is logged as JSONL (see
``recruitenv.trajectory``) and can be replayed with ``scripts/replay.py``.

    python inference.py --task hard --seed 3 --model claude-sonnet-5-5
    python inference.py --task easy --url http://localhost:7860

Environment variables:
    API_BASE_URL   OpenAI-compatible endpoint (default: Hugging Face router)
    API_KEY        API key (HF_TOKEN is also read)
    MODEL_NAME     default model (default: gpt-4o-mini)

No fallback actions: a model error, an unparseable reply or a reply cut off
by the token cap is retried (a cut-off reply doubles the cap, up to
``--max-tokens-ceiling``). When retries run out the episode is reported as
incomplete, never finished by a substitute policy.

Output lines: one [START], one [STEP] per environment step, one [END] with
the grader's score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import TypeAdapter

from recruitenv.environment import RecruitmentEnvironment
from recruitenv.models import Action
from recruitenv.trajectory import LoggedEnvironment

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"

BENCHMARK = "recruitenv"
TEMPERATURE = 0.1
MAX_TOKENS = 16000          # reasoning models spend thousands of tokens before a decision
MAX_TOKENS_CEILING = 32000
MODEL_ATTEMPTS = 4          # per step: replies that are unparseable or cut off
API_ATTEMPTS = 8            # per step: failed requests, with backoff up to a minute (about 4 minutes in all)
API_ERROR_PREFIX = "model request failed"

_ACTION = TypeAdapter(Action)


class IncompleteEpisode(Exception):
    """The agent could not produce a valid action; the episode has no score."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
          flush=True)


def log_end(result: dict[str, Any]) -> None:
    if result.get("incomplete"):
        print(f"[END] incomplete=true reason={result.get('reason')!r} steps={result.get('steps', 0)}", flush=True)
        return
    print(f"[END] success={str(result['success']).lower()} steps={result['steps']} score={result['score']:.3f}",
          flush=True)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RUBRIC_GUIDE = """How candidates are judged: the published hiring rubric (recruitenv/rubric.py).
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
    undecided = [
        c["id"] for c in observation.get("candidates_summary", [])
        if c["id"] not in observation.get("decisions_made", {})
    ]
    return (
        f"Step {step}. Steps remaining: {observation.get('steps_remaining', 0)}. "
        f"Undecided candidates: {len(undecided)}.\n"
        f"Job: {json.dumps(observation.get('job_description', {}), separators=(',', ':'))}\n"
        f"Quotas: {json.dumps(observation.get('quotas', {}))}\n"
        f"Candidates: {json.dumps(observation.get('candidates_summary', []), separators=(',', ':'))}\n"
        f"Revealed data: {json.dumps(observation.get('revealed_data', {}), separators=(',', ':'))}\n"
        f"Decisions made: {json.dumps(observation.get('decisions_made', {}))}\n"
        f"Previous steps:\n{history_block}\n\n"
        f"What is your next action? Respond with JSON only."
    )


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


# ---------------------------------------------------------------------------
# Model chooser
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_errors: int = 0
    parse_failures: int = 0
    length_cutoffs: int = 0
    rejections: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# A chooser maps (observation, history, usage) to one action dict, or raises
# IncompleteEpisode.
ChooseAction = Callable[[dict, list[str], Usage], dict]


@dataclass
class ModelChooser:
    """Asks an OpenAI-compatible chat model for one action per step."""

    client: Any
    model: str
    max_tokens: int = MAX_TOKENS
    max_tokens_ceiling: int = MAX_TOKENS_CEILING
    attempts: int = MODEL_ATTEMPTS
    api_attempts: int = API_ATTEMPTS
    temperature: float = TEMPERATURE
    backoff_s: float = 2.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __call__(self, observation: dict, history: list[str], usage: Usage) -> dict:
        cap = self.max_tokens
        notes: list[str] = []
        step = observation.get("step_number", 0) + 1
        last_problem = "no attempt"
        replies = failures = 0
        while replies < self.attempts:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(step, observation, history + notes)},
                    ],
                    temperature=self.temperature,
                    max_tokens=cap,
                    **self.extra,
                )
            except Exception as exc:  # network or provider error: back off and retry
                usage.api_errors += 1
                failures += 1
                if failures >= self.api_attempts:
                    raise IncompleteEpisode(f"{API_ERROR_PREFIX}: {exc}"[:300]) from exc
                time.sleep(min(60.0, self.backoff_s * (2 ** (failures - 1))))
                continue
            replies += 1
            usage.calls += 1
            u = getattr(completion, "usage", None)
            if u is not None:
                usage.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
                usage.completion_tokens += getattr(u, "completion_tokens", 0) or 0
            choice = completion.choices[0]
            text = choice.message.content or ""
            action = parse_action(text)
            if action is not None:
                return action
            if getattr(choice, "finish_reason", None) == "length":
                usage.length_cutoffs += 1
                last_problem = f"reply cut off at max_tokens={cap}"
                cap = min(cap * 2, self.max_tokens_ceiling)
            else:
                usage.parse_failures += 1
                last_problem = f"unparseable reply: {text[:200]!r}"
                notes = ["Your last reply was not a single JSON action. Reply with exactly one JSON object."]
        raise IncompleteEpisode(last_problem)


# ---------------------------------------------------------------------------
# Environments: in process (default) or a server over its websocket
# ---------------------------------------------------------------------------


class LocalDriver:
    """The core environment in process, optionally logged."""

    def __init__(self, log_path: Path | None = None, *, agent: str = "inference", model: str | None = None):
        core = RecruitmentEnvironment()
        self.env = LoggedEnvironment(core, log_path, agent=agent, model=model) if log_path else core

    def reset(self, task: str, seed: int, name_seed: int = 0, name_blind: bool = False) -> dict:
        return self.env.reset(task, seed, name_seed=name_seed, name_blind=name_blind).model_dump(mode="json")

    def step(self, action: dict) -> tuple[dict | None, float, bool, str | None]:
        """Return (observation, reward, done, None), or (None, 0, False, error) if rejected."""
        try:
            parsed = _ACTION.validate_python(action)
        except Exception as exc:
            return None, 0.0, False, f"invalid action: {exc}"[:300]
        try:
            obs, reward, done, _ = self.env.step(parsed)
        except (ValueError, RuntimeError) as exc:
            return None, 0.0, False, str(exc)[:300]
        return obs.model_dump(mode="json"), reward.step_reward, done, None

    def grade(self) -> dict:
        return self.env.grade_report()

    def close(self) -> None:
        if isinstance(self.env, LoggedEnvironment):
            self.env.close()


class RemoteDriver:
    """A RecruitEnv server over its OpenEnv websocket (needs the openenv extra)."""

    def __init__(self, url: str):
        from recruitenv.client import RecruitAction, RecruitEnvClient

        self._action_cls = RecruitAction
        self._client = RecruitEnvClient(base_url=url).sync()
        self._client.__enter__()
        self._grade: dict | None = None

    def reset(self, task: str, seed: int, name_seed: int = 0, name_blind: bool = False) -> dict:
        r = self._client.reset(task=task, seed=seed, name_seed=name_seed, name_blind=name_blind)
        return r.observation.model_dump(mode="json")

    def step(self, action: dict) -> tuple[dict | None, float, bool, str | None]:
        try:
            act = self._action_cls(**action)
        except Exception as exc:
            return None, 0.0, False, f"invalid action: {exc}"[:300]
        r = self._client.step(act)
        obs = r.observation.model_dump(mode="json")
        if obs.get("error"):
            return None, 0.0, False, obs["error"]
        if r.done:
            self._grade = obs.get("metadata", {}).get("grade")
        return obs, float(r.reward or 0.0), bool(r.done), None

    def grade(self) -> dict:
        if self._grade is None:
            raise RuntimeError("episode not finished")
        return self._grade

    def close(self) -> None:
        self._client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------


def run_episode(driver: Any, choose: ChooseAction, task: str, seed: int, *,
                name_seed: int = 0, name_blind: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Play one episode. Returns the grader's result, or ``incomplete=True``.

    A rejected action costs no step; the rejection is shown to the agent in
    its history. More rejections than twice the budget ends the episode as
    incomplete.
    """
    usage = Usage()
    t0 = time.time()
    observation = driver.reset(task, seed, name_seed=name_seed, name_blind=name_blind)
    history: list[str] = []
    rewards: list[float] = []
    rejections_left = 2 * observation["steps_remaining"]
    done = False
    result: dict[str, Any] = {"task": task, "seed": seed, "name_seed": name_seed, "name_blind": name_blind}
    try:
        while not done:
            action = choose(observation, history, usage)
            obs, reward, done, error = driver.step(action)
            if error is not None:
                usage.rejections += 1
                rejections_left -= 1
                history.append(f"{json.dumps(action)} -> rejected: {error}")
                if verbose:
                    log_step(observation["step_number"], json.dumps(action), 0.0, False, error)
                if rejections_left <= 0:
                    raise IncompleteEpisode("too many rejected actions")
                continue
            observation = obs
            rewards.append(reward)
            history.append(f"Step {observation['step_number']}: {json.dumps(action)}")
            if verbose:
                log_step(observation["step_number"], json.dumps(action), reward, done, None)
    except IncompleteEpisode as exc:
        usage.seconds = round(time.time() - t0, 1)
        return {**result, "incomplete": True, "reason": str(exc), "steps": observation.get("step_number", 0),
                "usage": usage.as_dict()}
    grade = driver.grade()
    usage.seconds = round(time.time() - t0, 1)
    return {
        **result,
        "incomplete": False,
        "score": grade["score"],
        "success": bool(grade["success"]),
        "steps": grade["steps_used"],
        "rewards": rewards,
        "grade": grade,
        "usage": usage.as_dict(),
    }


def make_client(base_url: str | None = None, api_key: str | None = None):
    from openai import OpenAI

    return OpenAI(base_url=base_url or API_BASE_URL, api_key=api_key or API_KEY)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Play RecruitEnv episodes with a chat model.")
    ap.add_argument("--task", default=os.getenv("TASK", "easy"), choices=["easy", "medium", "hard"])
    ap.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    ap.add_argument("--name-seed", type=int, default=0)
    ap.add_argument("--name-blind", action="store_true")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--url", default=os.getenv("ENV_URL"), help="play against a server instead of in process")
    ap.add_argument("--log-dir", default="outputs/inference", help="where episode logs go (local runs)")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)

    chooser = ModelChooser(make_client(), args.model, max_tokens=args.max_tokens)
    if args.url:
        driver: Any = RemoteDriver(args.url)
    else:
        log = None if args.no_log else Path(args.log_dir) / (
            f"{args.model}-{args.task}-s{args.seed}-n{args.name_seed}{'-blind' if args.name_blind else ''}.jsonl")
        driver = LocalDriver(log, agent="inference", model=args.model)
    log_start(task=args.task, env=BENCHMARK, model=args.model)
    try:
        result = run_episode(driver, chooser, args.task, args.seed,
                             name_seed=args.name_seed, name_blind=args.name_blind)
    finally:
        driver.close()
    log_end(result)
    return 0 if not result["incomplete"] else 2


if __name__ == "__main__":
    sys.exit(main())
