# 00 — Overview & Architecture

## What the Project Is (One Paragraph)

RecruitEnv is a **Reinforcement Learning environment** (not an agent, not a model — an *environment* that agents train against). It is built to comply with the **OpenEnv specification**, a standard API for RL environments published by Meta + Hugging Face. The "task" it simulates is **candidate pipeline triage**: an AI agent is given a job description and a pool of synthetic developer profiles, and it must spend a limited "step budget" gathering signals (GitHub stats, LeetCode stats, resume sections) and making shortlist / hold / reject decisions on each candidate. It exposes a REST API (FastAPI on port 7860), runs in a Docker container, deploys to Hugging Face Spaces, and ships with three difficulty-graded tasks, deterministic graders, a fairness penalty, and a baseline agent.

## Why It Matters (the "selling pitch" for the interview)

Recruiters at scale face a **combinatorial information-gathering problem**. Each candidate has signals scattered across multiple platforms; checking every signal for every candidate is too expensive. Good recruiters learn heuristics — they check the most informative signal first, decide fast on clear cases, and spend time only on ambiguous ones.

RecruitEnv turns this into a **sequential decision problem** so RL agents can be trained to learn this same prioritization behaviour. That makes it useful for:

- benchmarking LLM agents on information-foraging tasks
- training RL agents on a non-toy, non-game domain
- evaluating fairness behaviour (the env explicitly penalises demographically skewed decisions)

## What "OpenEnv-compliant" Actually Means

OpenEnv is a spec that says any RL environment must expose **three core verbs** through a typed API:

| Verb        | Purpose                                                    |
| ----------- | ---------------------------------------------------------- |
| `reset()`   | Start a new episode; return the initial `Observation`.     |
| `step(a)`   | Apply action `a`; return `(observation, reward, done, info)`. |
| `state()`   | Return the full internal state (used for debug / grading). |

It also requires:
- **Typed Pydantic models** for `Observation`, `Action`, `Reward`.
- An `openenv.yaml` file describing the env metadata.
- A **Dockerfile** so the env runs in any container runtime.
- At least **3 tasks** with **agent graders** producing scores in `[0.0, 1.0]`.
- A reproducible **baseline inference script** (typically using OpenAI's API).

This project meets every one of those.

## End-to-End Flow (the picture in your head)

```
┌────────────────────────────────────────────────────────────────────────┐
│  AGENT  (LLM, RL policy, or rule-based — lives OUTSIDE the env)        │
└────────────────────┬─────────────────────────────────┬─────────────────┘
                     │ HTTP POST /reset                │ HTTP POST /step
                     ▼                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  FastAPI server  (api/main.py, port 7860)                              │
│   - validates Action JSON via Pydantic discriminated union             │
│   - holds a single shared RecruitmentEnvironment instance              │
│   - serialises Observation / Reward / EpisodeState to JSON             │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │ calls
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  RecruitmentEnvironment   (env/environment.py)                         │
│   - thread-safe (asyncio.Lock at API + threading.Lock at env)          │
│   - holds the current EpisodeState                                     │
│   - delegates work to: ProfileFactory, RewardCalculator,               │
│                        FairnessChecker, tasks.GRADER_REGISTRY          │
└────┬──────────────────┬───────────────────┬─────────────────────┬──────┘
     │                  │                   │                     │
     ▼                  ▼                   ▼                     ▼
ProfileFactory   RewardCalculator   FairnessChecker        TASK_REGISTRY +
(generates       (step rewards +    (demographic           GRADER_REGISTRY
synthetic        terminal scoring   parity penalty         (3 TaskConfigs,
profiles +       + efficiency       on shortlist           3 BaseGrader
job description  bonus)             rate)                  subclasses)
deterministically
from a seed)
```

When the user / agent calls `POST /reset`:
1. The API hands the call to `RecruitmentEnvironment.reset(task_id, seed)`.
2. The env looks up the `TaskConfig` (e.g. easy / medium / hard) and asks `ProfileFactory` to generate the job description and the candidate pool using the seed.
3. A fresh `EpisodeState` is constructed and stored in the env.
4. An `Observation` (a redacted, agent-visible view of the state) is returned and serialised back to JSON.

When the agent calls `POST /step` with an action:
1. The action JSON is parsed into one of four `Action` subtypes via Pydantic's **discriminated union** on the `type` field.
2. The env computes a tiny step reward (e.g. +0.03 for revealing a new platform) *before* mutating state so it can detect novelty.
3. The action is executed (reveals data, records a score, or registers a terminal decision).
4. The step counter increments; the env checks termination (all decided **or** budget exhausted).
5. If still running → return a `Reward` with just `step_reward`.
6. If done → compute the **terminal reward**: accuracy across all decisions, +0.1 efficiency bonus if fully correct under 60% of the budget, minus a fairness penalty (0.0 / 0.1 / 0.2). Clamp to `[0, 1]`. This becomes the final `episode_score`.

When `POST /grader` is called *after* an episode is done:
1. The API looks up the appropriate `Grader` (one per task) from `GRADER_REGISTRY`.
2. The grader scores the episode using a **task-specific** rubric (weighted accuracy for easy, thoroughness for medium, F1 for hard) and returns a score in `[0.0, 1.0]`.

## Key Design Decisions to Mention in the Interview

1. **Determinism is a hard contract.** `reset(seed=42)` *must* always produce identical state. Achieved by routing **every** random draw through a single `numpy.random.default_rng(seed)` generator that is reseeded at the start of `generate_pool` and `generate_job_description`. Tested explicitly in `tests/test_environment.py::TestReset::test_reset_deterministic`.

2. **No live HTTP calls anywhere in `env/` or `api/`.** All candidate data is synthetic, generated locally. The `baseline/run_baseline.py` script is the *only* place that calls an external LLM, and that runs *outside* the environment loop.

3. **Discriminated union for actions.** All four action types share a `type: Literal[...]` field. Pydantic uses this as a discriminator so JSON like `{"type": "check_platform", ...}` is automatically routed to the correct subtype with full validation. No `if/elif` parsing, no untyped dicts.

4. **Information asymmetry is the core game.** The initial `Observation` contains only candidate IDs and names — *zero* stats. The agent must "pay" in steps to reveal anything. This is what makes the env a real sequential-decision problem rather than a one-shot classification problem.

5. **Multi-component reward.** Step reward (small, dense, encourages exploration) + terminal accuracy (large, sparse, encourages correct decisions) + efficiency bonus + fairness penalty. This is reward shaping done deliberately to avoid both "agent does nothing" and "agent gathers infinitely without deciding".

6. **Thread safety at two levels.** The env uses `threading.Lock` internally; the FastAPI handlers wrap each call in an `asyncio.Lock`. Critical because the env is a *shared singleton* — only one episode runs at a time.

7. **Ground-truth labels are embedded at generation time, not computed at grade time.** When the factory builds a `shortlist` candidate it knows ahead of time the candidate is a shortlist; the stats are then drawn from a distribution that *matches* that label. This guarantees correctness of grading without any post-hoc inference.

8. **The grader is a separate concept from the reward.** The reward is what the *agent sees during training*. The grader is the *official scoreboard* for an episode (used by `POST /grader`). They overlap but are not identical — graders are deliberately task-specific (F1 for hard, weighted accuracy for easy) while the in-episode reward is uniform.

## File Map of the Architecture

| Layer                  | File                              | Purpose                                                    |
| ---------------------- | --------------------------------- | ---------------------------------------------------------- |
| Spec / metadata        | `openenv.yaml`                    | OpenEnv compliance manifest                                |
| HTTP transport         | `api/main.py`                     | FastAPI app, all REST endpoints, request locking + CORS    |
| OpenEnv compat shims   | `server/app.py`, `server/models.py`, `client.py` | Re-exports + uvicorn launcher                              |
| RL env core            | `env/environment.py`              | `reset` / `step` / `state` implementation                  |
| Typed data             | `env/models.py`                   | Pydantic v2 models for everything                          |
| Data generation        | `env/profile_factory.py`          | Seeded synthetic profile + JD generator                    |
| Reward function        | `env/rewards.py`                  | Step + terminal + efficiency reward calculator             |
| Fairness               | `env/fairness.py`                 | Demographic parity penalty                                 |
| Tasks                  | `tasks/{base,easy,medium,hard}_task.py` | `TaskConfig` + grader per difficulty                       |
| Baseline               | `baseline/run_baseline.py`        | Structured LLM-or-rules loop, writes `baseline_scores.json`|
| Tests                  | `tests/test_*.py`                 | 35-test pytest suite                                       |
| Containerisation       | `Dockerfile`, `docker-compose.yml` | Python 3.11-slim non-root build, healthcheck               |

Read the rest of `/explanations` in numerical order — each file zooms into one of these layers.
