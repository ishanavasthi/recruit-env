"""Run an agent over seeds × tasks, logging every episode. Resumable.

    # reference policies (no model calls)
    python scripts/run_benchmark.py --agent rubric_heuristic --seeds 0-199
    # a model, step by step (inference.py's agent)
    python scripts/run_benchmark.py --agent model:claude-sonnet-5-5 --seeds 0-9 --concurrency 4
    # a model that only decides, after Python gathers evidence (baseline/run_baseline.py)
    python scripts/run_benchmark.py --agent decide:claude-sonnet-5-5 --seeds 0-9

Each episode writes ``<out>/<agent>/<task>-s<seed>.jsonl`` (the transition
log) and ``.json`` (its result). Finished episodes are skipped on a rerun;
incomplete ones are rerun only with ``--retry-incomplete``. A run stops
scheduling new episodes after ``--max-rate-limited`` episodes end incomplete
because of rate limiting (HTTP 429), so it does not push through a shared
quota. ``scripts/baselines_report.py`` aggregates the results.

Model runs read API_BASE_URL and API_KEY (an OpenAI-compatible endpoint).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recruitenv import __version__, policies  # noqa: E402
from recruitenv.environment import RecruitmentEnvironment  # noqa: E402
from recruitenv.trajectory import LoggedEnvironment  # noqa: E402

TASKS = ("easy", "medium", "hard")


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def is_rate_limited(reason: str) -> bool:
    """Only API failures count; model reply text never does."""
    import inference

    if not reason.startswith(inference.API_ERROR_PREFIX):
        return False
    low = reason.lower()
    return "429" in low or "rate limit" in low or "ratelimit" in low or "too many requests" in low


def safe_name(agent: str) -> str:
    return agent.replace(":", "__").replace("/", "_")


def run_policy(name: str, task: str, seed: int, log: Path) -> dict[str, Any]:
    env = LoggedEnvironment(RecruitmentEnvironment(), log, agent=name)
    env.reset(task, seed)
    policies.POLICIES[name](env)
    env.close()
    grade = env.grade_report()
    return {"task": task, "seed": seed, "incomplete": False, "score": grade["score"],
            "success": grade["success"], "steps": grade["steps_used"]}


def run_model(model: str, task: str, seed: int, log: Path, client: Any, max_tokens: int) -> dict[str, Any]:
    import inference

    chooser = inference.ModelChooser(client, model, max_tokens=max_tokens)
    driver = inference.LocalDriver(log, agent="inference", model=model)
    try:
        r = inference.run_episode(driver, chooser, task, seed, verbose=False)
    finally:
        driver.close()
    r.pop("rewards", None)
    r.pop("grade", None)
    return r


def run_decide(model: str, task: str, seed: int, log: Path, client: Any, max_tokens: int) -> dict[str, Any]:
    from baseline import run_baseline

    r = run_baseline.run_task(client, model, task, seed, log_path=log, max_tokens=max_tokens)
    r.pop("grade", None)
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run an agent over seeds and tasks.")
    ap.add_argument("--agent", required=True,
                    help="a policy name, model:<model> (step by step) or decide:<model> (decide-only)")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-49 or 0,3,7")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--out", default="outputs/bench")
    ap.add_argument("--retry-incomplete", action="store_true")
    ap.add_argument("--max-rate-limited", type=int, default=3)
    args = ap.parse_args(argv)

    tasks = args.tasks.split(",")
    seeds = parse_seeds(args.seeds)
    out = Path(args.out) / safe_name(args.agent)
    out.mkdir(parents=True, exist_ok=True)

    kind, _, model = args.agent.partition(":")
    client = None
    if kind in ("model", "decide"):
        import inference

        client = inference.make_client()
    elif args.agent not in policies.POLICIES:
        ap.error(f"unknown agent {args.agent!r}; policies: {sorted(policies.POLICIES)}")

    meta = {
        "agent": args.agent, "kind": kind if model else "policy", "model": model or None,
        "package_version": __version__,
        "settings": ({"temperature": 0.1 if kind == "model" else 0.0, "max_tokens": args.max_tokens,
                      "max_tokens_ceiling": 32000, "reasoning_effort": "provider default"}
                     if model else {}),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    todo = []
    for task in tasks:
        for seed in seeds:
            res = out / f"{task}-s{seed}.json"
            if res.exists():
                prev = json.loads(res.read_text())
                if not prev.get("incomplete") or not args.retry_incomplete:
                    continue
            todo.append((task, seed))
    print(f"{args.agent}: {len(todo)} episodes to run ({len(tasks) * len(seeds) - len(todo)} already done)",
          flush=True)

    stop = threading.Event()
    rate_limited = 0
    lock = threading.Lock()

    def one(task: str, seed: int) -> dict[str, Any] | None:
        if stop.is_set():
            return None
        log = out / f"{task}-s{seed}.jsonl"
        if kind == "model" and model:
            r = run_model(model, task, seed, log, client, args.max_tokens)
        elif kind == "decide" and model:
            r = run_decide(model, task, seed, log, client, args.max_tokens)
        else:
            r = run_policy(args.agent, task, seed, log)
        (out / f"{task}-s{seed}.json").write_text(json.dumps(r, indent=1) + "\n")
        return r

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = {ex.submit(one, t, s): (t, s) for t, s in todo}
        for fut in as_completed(futures):
            r = fut.result()
            if r is None:
                continue
            done += 1
            t, s = futures[fut]
            if r.get("incomplete"):
                msg = f"INCOMPLETE ({r.get('reason')})"
                if is_rate_limited(str(r.get("reason", ""))):
                    with lock:
                        rate_limited += 1
                        if rate_limited >= args.max_rate_limited and not stop.is_set():
                            stop.set()
                            print("rate limited repeatedly: stopping new episodes", flush=True)
            else:
                msg = f"score={r['score']:.3f}"
            print(f"[{done}/{len(todo)}] {t} seed={s} {msg}", flush=True)
    return 3 if stop.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
