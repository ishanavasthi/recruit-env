# Learnings

Newest first.

## 2026-11-25: 4.0.0 — standalone core, OpenEnv adapter, logs, honest model runs (PR C)

- **What:**
  - `recruitenv` is now one package, with a core that imports nothing from the web.
  - An OpenEnv 0.14 server gives one environment per websocket session and exposes the Task API.
  - Episodes are written as JSONL logs, and a replay checks every observation hash and the grade.
  - Agents no longer have fallbacks.
  - A resumable benchmark runner and a TRL tool environment were added.
- **Why:** the roadmap's Phases 4 and 5. Training needs parallel sessions, and published model numbers have to be attributable to the model.
- **Gotchas:**
  - OpenEnv's HTTP `/reset` and `/step` are stateless, so episodes need `/ws`.
  - `action_cls` must be one class, not a union.
  - `openenv validate --level static` reads WARN with exit 0 even when it passes.
  - The worst one: at `max_tokens=512`, both Claude models spent the cap reasoning and returned empty text. `inference.py` silently turned each of those replies into "hold".
    - The scripts never recorded when a fallback fired, so the model scores published with 3.0.0 cannot be attributed to the models, and they are withdrawn.
    - Evidence: the 512-token pilot was cut off within a few steps of every episode. At 16,000 tokens, 19 of 30 Sonnet decide-only replies needed more than the old 4,000 cap.
- **Takeaway:** a fallback in an evaluation harness is a measurement bug. Count failures; never substitute for them. Pilot a few full episodes before committing to a large model run.

## 2026-11-25: first model baselines (10 seeds)

- **What:** 120 model episodes, step by step and decide-only, for Claude Haiku 5.5 and Sonnet 5.5. None were incomplete, there were no cut-offs or API errors, and every log replays. They used about 17M tokens and about 1 h 50 min at concurrency 3 per model.
- **Finding:** applying the rubric is easy for both models. Choosing what to read step by step costs them 0.10–0.22 on easy and medium against the rule-based heuristic.
- **Gotcha:** `git filter-branch` removed regenerable outputs from history, but its final checkout also deleted them from disk.
- **Takeaway:** never `git add -A` while a run is writing files into the repo. Ignore output directories first.
- **Still open:** #12 asks for 50+ seeds. The runner is resumable, so more seeds can be added in PR D.

## 2026-11-25: Space redeployed to 3.0.0

- **What:** the Hugging Face Space ran the v1 code from April. It now runs 3.0.0, uploaded from a clean `git archive` of `main` with `hf upload --delete "*"`. That also removed stale files, including a committed `node_modules/`.
- **Why:** nothing used the v1 API (the owner confirmed on 2026-11-24), and the public demo should match the repo.
- **Gotcha:** the Hugging Face account is `heyavasthi`, which differs from the GitHub account.
- **Takeaway:** deploying from `git archive` means only tracked files ship.

## 2026-11-24: 3.0.0, real task and fairness audit (PR #14)

- **What:** generator v3, published rubric, conflicts, twins, name-swap replays, fairness metrics with a generator-redraw null, verifier checks and gates.
- **Why:** in 2.0.0, one field solved every task (roadmap F1), and the fairness penalty punished correct agents (F5).
- **Gotchas:**
  - Tuning was iterative: each fix to a single-source exploit exposed the next strongest source.
  - Two review rounds found real bugs: non-monotone scoring, tasks sharing candidates, and a fairness null that flagged honest agents.
  - Deep-copying `env.state()` on every step made the policies 70× slower.
- **Takeaway:** state numeric targets before tuning, and keep a test per target (`tests/test_acceptance.py`). Count each tuning round and each evidence regeneration as its own stage when estimating time.

## 2026-11-24: 2.0.0, one honest score (PR #13)

- **What:** the grader became the terminal reward. Ground-truth leaks were removed, contract tests added, and the package fixed.
- **Why:** two disagreeing scores existed, and lazy policies passed.
- **Takeaway:** build the measuring instrument (oracle and lazy-policy tests) before changing the task.
