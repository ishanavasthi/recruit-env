# Standalone core, transition logs and baselines: plan

Spec: `docs/superpowers/specs/2026-11-25-standalone-core-logs-baselines-design.md`. Branch: `ishan/pr-c-adapter-logs-baselines`. One PR.

Each task ends with the full suite passing, and the pinned generator hashes and acceptance numbers unchanged.

| # | Task | Files | Test first | Status |
|---|---|---|---|---|
| 1 | Rename to `recruitenv/` (done, `28d151b`) | `git mv` plus imports | the existing 204 tests | done `28d151b` |
| 2 | Dependencies: runtime is `pydantic` and `numpy`; extras are `server`, `agents`, `openenv` and `dev`; drop `openenv-core`; remove root `models.py`, `__init__.py` and `server/models.py` | `pyproject.toml`, `requirements*.txt`, `Dockerfile` | `tests/test_package.py`: a fresh venv installs and imports with no FastAPI, and the core has no web or OpenEnv imports | done `a0068ab` |
| 3 | Transition logs | `recruitenv/trajectory.py`, `scripts/replay.py` | `tests/test_trajectory.py`: writer schema, hashes, refusal on a version mismatch, round trip for each reference policy | done `a0068ab` |
| 4 | OpenEnv adapter and server | `recruitenv/openenv_adapter.py`, `recruitenv/client.py`, `server/app.py`, delete `api/`, `openenv.yaml`, `Dockerfile` | `tests/test_server.py`: action round trip, grade in terminal metadata, Task API, websocket episode, `/health`, `/schema`, `/baseline` | done `77d7640`; server moved into `recruitenv/server.py` in `0aa4bc3` |
| 5 | Concurrency safety | factory and cache, if needed | `tests/test_concurrency.py`: same seed, different `name_seed`, interleaved; a threaded version; a websocket version | done `77d7640` |
| 6 | Leak test on the new surface | `tests/test_scoring_contract.py` | websocket pre-terminal observations, `/state`, Task API, `/schema`, `/metadata` | done: the leak test is in `tests/test_server.py` |
| 7 | Agents: `inference.py` in process by default, `--url` for the websocket; no silent fallback; retries; `max_tokens` large enough for reasoning models; logs on by default | `inference.py`, `baseline/run_baseline.py` | `tests/test_inference.py`: an error marks the episode incomplete instead of substituting an action | done `77d7640`, `0aa4bc3` |
| 8 | Benchmark runner | `scripts/run_benchmark.py`, `scripts/baselines_report.py` | `tests/test_benchmark.py`: resumes and skips finished episodes, counts incomplete ones, bootstrap CI | done `2983884` |
| 9 | Training tools | `recruitenv/tools.py`, `examples/train_grpo.py` | `tests/test_tools.py`: a tool episode matches the core episode, plus `get_reward` | done `2983884`; CPU smoke test passed |
| 10 | Validate: `openenv validate --level static` and `--url`; wheel in a fresh venv; Docker image | — | commands recorded in the PR | done: static WARN/exit 0, `--url` PASS, wheel, Docker |
| 11 | Docs: README, CHANGELOG 4.0.0, `.plans/`, roadmap status | — | — | done `8dc6a34` |
| 12 | Review round (delegate), then fixes | — | — | done: round 1 fixes in `0aa4bc3` |
| 13 | Model and policy runs, then `docs/results/baselines.md` | `docs/results/` | each committed log replays | in progress |
| 14 | PR, CI, merge, redeploy the Space, verify it live | — | — | pending |

The pilot (2026-11-25) found that a 512-token cap lets the models spend the whole budget on reasoning and return no text (`finish_reason="length"`). That is what task 7's `max_tokens` fix addresses.
