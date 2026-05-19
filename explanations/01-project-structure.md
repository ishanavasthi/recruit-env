# 01 — Project Structure

A file-by-file map of the repo. Every important file is listed; tooling-only files (`.venv/`, `node_modules/`, `uv.lock`) are skipped.

## Tree (annotated)

```
recruitenv/
│
├── CLAUDE.md                ← AI-assistant rules: hard constraints (no live HTTP, determinism,
│                              port 7860, Pydantic v2). Lives in the repo so any LLM that
│                              touches this codebase reads these rules first.
├── PROBLEM_STATEMENT.md     ← Original hackathon problem statement (what the env had to do
│                              to qualify). Useful as a "spec" reference.
├── README.md                ← Public docs. Has the HF Spaces front-matter at the top
│                              (title, emoji, sdk: docker, app_port: 7860). Long, polished,
│                              includes API examples + tables.
├── openenv.yaml             ← OpenEnv manifest. Names the env, version, entry_point,
│                              tasks, and the HF Space URL.
│
├── Dockerfile               ← python:3.11-slim, creates non-root `appuser`, installs curl
│                              for healthchecks, runs uvicorn on port 7860.
├── docker-compose.yml       ← Single-service compose for local dev (forwards OPENAI_API_KEY).
├── requirements.txt         ← Runtime pip deps: fastapi, uvicorn, pydantic, numpy, openai,
│                              pytest, httpx, python-dotenv, openenv-core.
├── pyproject.toml           ← Project metadata (setuptools backend). Defines `server` script
│                              entry point that openenv-core uses.
├── package.json / node_modules/  ← Pulled in only to satisfy a Node-based validator step;
│                              not used by the runtime.
│
├── client.py                ← Thin wrapper around openenv-core's GenericEnvClient. Lets
│                              outside callers do `RecruitEnvClient().reset()` against the
│                              deployed HF Space without writing httpx by hand.
├── inference.py             ← Standalone HF-router LLM evaluator. Reads env vars HF_TOKEN /
│                              MODEL_NAME / API_BASE_URL, runs the easy task, logs in the
│                              standard [START]/[STEP]/[END] format expected by the
│                              automated judging system.
├── sample_inference.py      ← Original reference inference (used to derive inference.py).
├── validate_submission.sh   ← Pre-submission validator: spins up Docker, hits /health,
│                              /reset, /step, /grader, /baseline; verifies determinism.
├── __init__.py              ← Empty package marker for the repo root.
├── models.py                ← Compatibility re-export at repo root (so `from models import …`
│                              still works for any external importers).
│
├── env/                     ── The RL environment proper (no FastAPI, no HTTP). Importable
│   │                          and runnable from pure Python.
│   ├── __init__.py          ← Re-exports the public surface (RecruitmentEnvironment, all
│   │                          models, ProfileFactory, RewardCalculator, FairnessChecker).
│   ├── models.py            ← All Pydantic v2 typed models — see explanations/02.
│   ├── environment.py       ← The RecruitmentEnvironment class — see explanations/03.
│   ├── profile_factory.py   ← Seeded synthetic profile generator — see explanations/04.
│   ├── rewards.py           ← RewardCalculator — see explanations/05.
│   └── fairness.py          ← FairnessChecker (demographic parity) — see explanations/06.
│
├── tasks/                   ── Task definitions + graders. Three difficulty tiers.
│   ├── __init__.py          ← Builds TASK_REGISTRY and GRADER_REGISTRY dicts.
│   ├── base.py              ← @dataclass TaskConfig (id, max_steps, label_distribution,
│   │                          success_threshold, role_type) + BaseGrader ABC.
│   ├── easy_task.py         ← EASY_CONFIG + EasyGrader  (10 cands, 40 steps, weighted acc).
│   ├── medium_task.py       ← MEDIUM_CONFIG + MediumGrader (5 cands, 25 steps, thoroughness).
│   └── hard_task.py         ── HARD_CONFIG + HardGrader (20 cands, 30 steps, F1).
│   See explanations/07 for the full rubric of each.
│
├── api/                     ── FastAPI HTTP layer.
│   ├── __init__.py
│   └── main.py              ← All endpoints: GET /, /health, /tasks, /state, /baseline;
│                              POST /reset, /step, /grader. See explanations/08.
│
├── server/                  ── OpenEnv-required standard entry point.
│   ├── __init__.py
│   ├── app.py               ← `def main(): uvicorn.run(app, …, port=7860)` — the
│   │                          `server` console-script declared in pyproject.toml.
│   └── models.py            ← Re-exports env.models for openenv-core compatibility.
│
├── baseline/                ── Reference agent.
│   ├── __init__.py
│   ├── run_baseline.py      ← Structured (Python-driven, not LLM-driven) candidate loop.
│   │                          Uses OpenAI for the *decision* only; rule-based fallback
│   │                          for tight budgets. Writes baseline_scores.json.
│   └── baseline_scores.json ← Latest reproducible scores: easy 1.0, medium 1.0, hard 0.92.
│
├── tests/                   ── pytest suite (~35 tests).
│   ├── __init__.py
│   ├── test_environment.py  ← reset determinism, step actions, terminal conditions, errors.
│   ├── test_graders.py      ← grader accuracy, determinism, F1 validation, range checks.
│   └── test_api.py          ── endpoint shapes, full episode through HTTP, error codes.
│
├── outputs/                 ── ad-hoc CLI / log captures (git-ignored).
└── .env / .env.example      ── OPENAI_API_KEY etc.; .env.example is the committed template.
```

## Mental Model — Layered View

Think of the project in **three concentric rings**:

1. **Core (`env/` + `tasks/`)** — Pure Python, no FastAPI, no HTTP. Could be `pip install`-ed and used as a library by an RL training loop directly. This is what implements the RL semantics.
2. **Transport (`api/` + `server/` + `client.py`)** — HTTP wrapping. Translates JSON ↔ Pydantic, holds the single shared env instance, adds locking and CORS.
3. **Ecosystem (`Dockerfile`, `openenv.yaml`, `baseline/`, `inference.py`, `tests/`)** — Everything else needed to deploy, validate, and demonstrate the env.

The point of this layering: you can run *all 35 tests* without ever starting the HTTP server (tests import the core directly via `from env.environment import RecruitmentEnvironment`). And conversely, you can hit the HTTP API without caring how the env is implemented internally. Clean separation.

## Where to Read Next

| If you want to understand…              | Open…                                |
| --------------------------------------- | ------------------------------------ |
| The *typed data shapes*                 | `02-models.md`                       |
| How `reset` / `step` actually work      | `03-environment-core.md`             |
| How profiles get synthesised            | `04-profile-factory.md`              |
| How the reward signal is computed       | `05-rewards.md`                      |
| Demographic-parity fairness penalty     | `06-fairness.md`                     |
| The three difficulty tiers + graders    | `07-tasks-and-graders.md`            |
| HTTP endpoints + serialisation          | `08-api-server.md`                   |
| The baseline agent loop                 | `09-baseline-agent.md`               |
| Test coverage                           | `10-tests.md`                        |
| Docker + HF Spaces deployment           | `11-deployment.md`                   |
| Interview questions you should expect   | `12-interview-qa.md`                 |
