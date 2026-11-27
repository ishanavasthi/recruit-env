# Standalone core, transition logs and baselines: design

Date: 2026-11-25. Release: 4.0.0. Roadmap: Phases 4 and 5 (`docs/research/2026-11-21-findings-and-roadmap.md`). Issues: #11, #8 and #12 (#12 only in part).

Decisions marked **(delegate)** were made by the owner's delegate, a Fable reviewer. The owner was away and had asked that design questions go to Fable. Decisions marked **(builder)** are calls the builder made and that reviewers can question.

## Intent

RecruitEnv should stand on its own as a library: `pip install`, then `reset` and `step` in process, with no server. The HTTP and OpenEnv surfaces become thin adapters over that core. Every episode can be logged and replayed exactly. And there is published evidence of how well real agents do, across many seeds.

## Goals

1. **Library.**
   - One import package, `recruitenv`, matching the distribution name, with a plain API: `RecruitmentEnvironment().reset(task, seed, name_seed=, name_blind=)`, `.step(action)`, `.state()` and `.grade_report()`.
   - The core has no web or OpenEnv imports.
2. **OpenEnv adapter (#11).**
   - An `Environment` subclass served by `create_app`, with one core instance per session and `SUPPORTS_CONCURRENT_SESSIONS = True`.
   - Tasks are exposed through the Task API.
   - `openenv validate --level static` and `--url` both pass.
   - A typed `EnvClient` replaces `client.py`.
3. **Transition logs (#8).**
   - JSONL with a header, one record per step, and a final record holding the grade report.
   - A replay script reproduces every logged observation and the score exactly.
4. **Baseline table (#12).**
   - Reference policies and two real models over many seeds, with intervals and stated seed counts.
   - The model logs are committed.
5. **Training example (#12).**
   - A tool-per-action environment class for TRL's `environment_factory`.
   - A GRPO example script that runs on Colab or HF Jobs.

## Non-goals

- The reward curve from a real training run. It needs a GPU, and the owner decides whether it runs on Colab or HF Jobs, because that costs money. #12 stays open for it.
- PyPI release, results page, report and demo (Phase 6, a later PR).
- Any change to the task, generator, rubric or graders. The generator hashes and acceptance numbers must not move.

## Anti-goals

- **A model score that is secretly another policy's score.** A fallback action on an API or parse error does exactly this, so it is not allowed in the evaluation runner.
- **Concurrency that shares mutable state between sessions.**
- **A core that imports OpenEnv.** OpenEnv's `env_server` pulls in gradio.

## Facts this design rests on

The research note is `docs/research/2026-11-25-openenv-and-trl-refresh.md`. A throwaway spike against `openenv` 0.14.0 on 2026-11-25 checked the three points the delegate asked to verify:

| Question | Result |
|---|---|
| Does `Observation.metadata` survive the websocket? | Yes. `serialize_observation` sends it as a top-level `metadata` key. |
| Can `action_cls` be our discriminated union? | No. The server calls `action_cls.model_validate`, which needs a class. A flat `Action` subclass with a `type` literal and optional fields works, and the adapter converts it to the core union. MCP interception applies only when `action_cls` is the base `Action`, so the `type` field is safe. |
| Do `task`, `name_seed` and `name_blind` reach `reset` over the websocket? | Yes. Three concurrent sessions on medium seed 7 with name_seed 0, 1 and 2 got different names. Name-blind showed "Candidate 01". |

Also observed:
- `/health` returns `"healthy"`.
- `/metadata` defaults to the class name, so it should be set explicitly.
- `create_app` mounts `/{env_name}/tasks`, `/task`, `/num_tasks`, `/task_range` and `/splits`.
- HTTP `/reset` and `/step` are stateless: each request builds a fresh environment. A multi-step episode therefore needs the websocket or the in-process core.

## Approaches considered

1. **Keep the custom FastAPI app and add OpenEnv routes beside it.** Rejected. It gives two servers' worth of behaviour, the global single episode stays, and `--url` validation needs `/mcp` and `/schema` anyway.
2. **Replace the server with `create_app` and keep a few custom routes (chosen, delegate).** One server, concurrent sessions, and validation passes. The grade report rides in the terminal observation's `metadata`, because stateless HTTP cannot serve a separate `/grader` call.
3. **Run with no server at all and publish only the library.** Rejected. The Space and any OpenEnv user need a server.

## Design

### 1. Package (delegate)

- `env/` and `tasks/` move to `recruitenv/` (`recruitenv/tasks/` for the tasks), with no compatibility shims, since nothing outside the repo imports them. The rename is its own commit: `git mv` plus import changes only. The full suite, the pinned generator hashes and the acceptance numbers must be unchanged before anything else changes.
- Root `models.py`, `__init__.py`, `client.py` and `server/models.py` existed only for the old OpenEnv layout and are removed. The server is `recruitenv/server.py`, inside the one import package (delegate, review round 1). `server/app.py` remains as an unpackaged shim for `uvicorn server.app:app`.
- `openenv>=0.14,<0.15` moves to an optional extra, `[openenv]`, with `uvicorn`, and the `openenv-core` dependency is dropped. Runtime dependencies shrink to `pydantic` and `numpy`. The `[agents]` extra adds `openai` and `python-dotenv`. FastAPI comes with `openenv`, so there is no separate server extra (builder).

### 2. OpenEnv adapter and server (delegate)

- `recruitenv/openenv_adapter.py` defines:
  - `RecruitAction`: a flat action with `type`, `candidate_id`, and optional `section`, `platform` and `decision`, validated into the core union.
  - `RecruitObservation`: the core observation's fields plus `reward` and `done`. The grade report goes in `metadata["grade"]` on the terminal step only.
  - `RecruitState`.
  - `RecruitEnvironment(Environment)`: `reset(seed, episode_id, task="easy", name_seed=0, name_blind=False)` and the Task API (`list_tasks` and `get_task` over easy, medium and hard).
- `recruitenv/server.py` calls `create_app(..., env_name="recruitenv", max_concurrent_envs=N)`. It keeps `/tasks` and `/baseline` as extra routes and sets the metadata (name, description, README, version).
- The old `api/` package is deleted. The Space's Dockerfile runs the new server on port 7860.
- `recruitenv/client.py` defines `RecruitEnvClient(EnvClient[...])` and is importable only with the extra.
- `openenv.yaml` gets a `validation:` block. Today a static pass reads WARN with exit 0, because the build checks are skipped, and the PR says so plainly.

### 3. Concurrency safety (delegate)

- The shared `_signal_pool` cache and the factory must not hand two sessions the same mutable candidate objects.
- Test: two in-process sessions on the same seed with different `name_seed`, steps interleaved. Each sees only its own names and decisions, and each final grade equals a solo run of the same actions.
- A threaded version of the same test runs too.
- A websocket version runs against a live `create_app` server.

### 4. Ground-truth leak test (delegate)

- The test that no agent-reachable surface shows ground truth before the episode ends moves to `tests/test_server.py` and covers:
  - websocket observations before the terminal step;
  - `/state`;
  - the Task API routes;
  - `/schema` and `/metadata`.

### 5. Transition logs (#8)

`recruitenv/trajectory.py` writes JSONL with three kinds of record:

- **Header:** `episode_id`, `task`, `seed`, `name_seed`, `name_blind`, `generator_version`, package version, agent name and model, and start time.
- **Step:** `step`, `action`, `revealed` (only what this step revealed), `obs_sha256` (a hash of the canonical observation JSON), `reward`, `steps_remaining`, `done`, and `error` for a rejected action.
- **Final:** the full grade report, including checks and gates.

`scripts/replay.py <log>` works in four stages:
1. Reads the header and refuses if `generator_version` differs. A different package version is replayed, and named in the error only if the replay diverges. The delegate first asked to refuse both, then reversed it in review round 1: refusing on the package version would invalidate every committed log at the next release.
2. Resets with the logged parameters.
3. Replays every action, checking each hash.
4. Checks that the score and checks match exactly.

Exit code 0 means the log reproduced.

Logging is on by default in `inference.py`, `baseline/run_baseline.py` and `scripts/run_benchmark.py`, written to `outputs/<run>/` and controlled by `--log-dir` and `--no-log`.

### 6. Agents (builder)

- `inference.py` runs in process by default. `--url` switches to the websocket client.
- Silent fallbacks are removed.
- Per step, up to 4 replies are allowed. A reply cut off by the token cap doubles the cap (16,000 by default, to at most 32,000), and an unparseable reply is retried with a correction note.
- Failed requests have their own budget: 8 attempts, backing off up to a minute (delegate, review round 1), so a brief proxy outage does not discard a long episode.
- When either budget runs out, the episode is marked incomplete, not finished with a substitute action.
- A rejected action still costs no budget and is fed back to the model, capped as before.

### 7. Baseline table (#12)

`scripts/run_benchmark.py` runs any policy or model over seeds × tasks:
- in process and resumable, with one log per episode and finished episodes skipped;
- with bounded concurrency;
- recording incomplete episodes and the count of model errors.

`docs/results/baselines.{json,md}` reports, per task, mean score, pass rate and a 95% bootstrap interval for:
- **Reference policies:** random, fixed order, rubric heuristic and oracle, on 200 seeds and on the same seeds as the models (delegate).
- **Models:** Claude Haiku 5.5 and Claude Sonnet 5.5, step by step. The seed count is set after the pilot (see Budget). The table states the actual count.

Model logs are committed under `docs/results/runs/`. Policy logs are regenerable and are not committed (delegate).

### 8. Training example (#12, partial)

- `recruitenv/tools.py` defines `RecruitToolEnv`: a no-argument constructor, `reset(**kw) -> str`, and one typed, docstringed method per action (`read_resume`, `check_github`, `check_leetcode`, `check_kaggle`, `shortlist`, `hold`, `reject`), plus `get_reward()`. It matches TRL's `environment_factory` contract (TRL ≥ 1.9).
- `get_reward()` plays an unfinished episode out to the budget and grades it (delegate, review round 1). On hard the episode ends only at the budget, so an agent that shortlists and stops would otherwise always get 0.
- It is unit-tested without TRL installed.
- `examples/train_grpo.py` is a PEP 723 uv script (`trl>=1.9,<2`, `transformers>=5.2`) for Qwen2.5-0.5B-Instruct. It runs with `uv run` on Colab or with `hf jobs uv run`.
- In this PR it was run only as a CPU smoke test with TRL's tiny Qwen3 MoE model, which exercises the whole pipeline but does no learning.

## Budget (delegate)

- The model runs come last, after the code is reviewed.
- A pilot of 2 seeds × 3 tasks × 2 models measures calls, tokens and time. The seed count and concurrency are then set against a hard budget, starting below 8.
- If the budget cuts a run short, the table ships with the actual seed counts.

### Pilot (2026-11-25)

Two Claude models, one seed per task, step by step, at a 16,000-token cap. Every episode completed, with no cut-off replies, API errors or parse failures.

| Model | easy | medium | hard | Tokens per episode (prompt + completion) | Minutes per episode |
|---|---|---|---|---|---|
| Claude Haiku 5.5 | 0.615 | 0.48 | 0.40 | 250k–440k | 5–13 |
| Claude Sonnet 5.5 | 0.999 | 0.999 | 0.60 | 190k–300k | 4–8 |

Decision steps used up to about 7,500 completion tokens. An earlier pilot at the old 512-token cap cut off every episode.

The delegate set the first tranche at 10 seeds per task per model and concurrency 3 per model. Extending it to 20 seeds depends on there being no rate limiting.

## Done when

- `pip install .` in a fresh virtualenv runs an episode in process with no server.
- The full suite passes, and the generator hashes and acceptance numbers are unchanged.
- `openenv validate --level static` exits 0 and `openenv validate --url http://127.0.0.1:<port>` passes.
- Concurrent same-seed sessions do not interfere, in process and over the websocket.
- Every baseline run leaves logs, and `scripts/replay.py` reproduces each committed log exactly.
- `docs/results/baselines.md` shows reference policies and two models, with seed counts and intervals.
- The Space runs 4.0.0, and its `/health`, `/schema` and a websocket episode work.

## PR wording (delegate)

- "Closes #11" and "Closes #8".
- "Part of #12", because the reward curve remains.
