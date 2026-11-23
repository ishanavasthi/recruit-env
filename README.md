---
title: RecruitEnv
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RecruitEnv

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![HF Spaces](https://img.shields.io/badge/HF%20Spaces-ready-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)
[![Tests](https://github.com/ishanavasthi/recruit-env/actions/workflows/tests.yml/badge.svg)](https://github.com/ishanavasthi/recruit-env/actions/workflows/tests.yml)

A reinforcement learning environment that simulates **candidate pipeline triage**, the process of reviewing developer profiles across multiple platforms and making shortlisting decisions against a job description.

---

## Overview

Recruiters at scale face a combinatorial problem: each candidate has signals scattered across GitHub, LeetCode, Kaggle, and their resume. Checking every signal for every candidate is expensive. The best recruiters develop efficient heuristics, check the most informative signals first, decide fast on clear cases, spend time on ambiguous ones.

RecruitEnv turns this into a sequential decision problem. An RL agent receives a job description and a pool of candidates. Each step, it can gather one signal (check a platform, read a resume section) or make a decision (shortlist, hold, reject). The agent has a limited step budget, so it must learn to **prioritize information gathering** and **make accurate decisions under time pressure**.

The environment is fully synthetic and deterministic. No live API calls. `reset(seed=42)` always produces identical state. All candidate profiles are generated from controlled statistical distributions matching real-world developer tiers (junior/mid/senior), with ground-truth labels embedded at generation time.

### Why train here

- **Information asymmetry**: the agent starts knowing nothing about candidates and must pay (in steps) to reveal signals
- **Exploration-exploitation tradeoff**: gathering more signals improves decisions but consumes budget
- **One honest score**: the episode's reward is the task grader's score, tested against a reference solution and a set of degenerate policies (see [Scoring](#scoring))
- **Graded difficulty**: three tasks from generous budgets (4 steps/candidate) to extreme time pressure (1.5 steps/candidate)

---

## Environment Description

Each episode follows this pipeline:

```
reset(task_id, seed)
  --> generates JobDescription + CandidatePool
  --> returns initial Observation (candidate names only, no stats)

loop:
  agent picks action:
    read_resume_section(candidate_id, section)    -- reveals resume text
    check_platform(candidate_id, platform)        -- reveals platform stats
    make_decision(candidate_id, decision)          -- terminal per candidate

  environment returns:
    observation  -- everything revealed so far (values included), decisions so far
    reward       -- 0.0, except the last step, which carries the episode score
    done         -- True when all decided OR budget exhausted
    info         -- action-specific metadata

episode ends --> the task grader scores decisions against ground truth
```

Candidate profiles are generated with realistic distributions. A senior developer has 30-60 GitHub repos, 250+ LeetCode problems solved, and Kaggle Expert+ rank. A junior developer has <10 repos, <100 problems, and no Kaggle medals. Ground-truth labels (shortlist/hold/reject) are always consistent with the profile tier, plus controlled noise for ambiguous cases.

---

## Observation Space

The agent receives an `Observation` after every step:

| Field                | Type              | Description                                                            |
| -------------------- | ----------------- | ---------------------------------------------------------------------- |
| `task_id`            | `str`             | Current task identifier                                                |
| `step_number`        | `int`             | Current step (0-indexed)                                               |
| `steps_remaining`    | `int`             | Budget left in this episode                                            |
| `job_description`    | `JobDescription`  | Role, required skills, seniority, dimension weights                    |
| `candidates_summary` | `list[dict]`      | `[{"id": "candidate_001", "name": "Wei Zhang"}, ...]` &mdash; no stats |
| `revealed_data`      | `dict`            | `candidate_id -> {"resume_sections": {name: section}, "platforms": {name: stats}}` |
| `decisions_made`     | `dict[str, str]`  | `candidate_id -> "shortlist" \| "hold" \| "reject"`                    |
| `done`               | `bool`            | Whether the episode has ended                                          |

The observation is the agent's whole view: every value it has revealed is in `revealed_data`. Ground-truth labels never reach an agent during an episode.

The `job_description` includes dimension weights that sum to 1.0:

| Role Type          | `weight_technical` | `weight_experience` | `weight_growth` |
| ------------------ | :----------------: | :-----------------: | :-------------: |
| ML Engineer        |        0.50        |        0.25         |      0.25       |
| Backend Developer  |        0.45        |        0.30         |      0.25       |
| Data Scientist     |        0.40        |        0.30         |      0.30       |
| Frontend Developer |        0.35        |        0.35         |      0.30       |

---

## Action Space

All actions are JSON objects with a `type` discriminator field:

Every action costs one step of the budget.

| Action                | Parameters                                                                | Effect                                            |
| --------------------- | ------------------------------------------------------------------------- | ------------------------------------------------- |
| `read_resume_section` | `candidate_id`, `section` (`"education"` \| `"experience"` \| `"skills"`) | Reveals resume section text + years of experience |
| `check_platform`      | `candidate_id`, `platform` (`"github"` \| `"leetcode"` \| `"kaggle"`)     | Reveals full platform statistics                  |
| `make_decision`       | `candidate_id`, `decision` (`"shortlist"` \| `"hold"` \| `"reject"`)      | Final for this candidate. Cannot be reversed.     |

Example action:

```json
{
  "type": "check_platform",
  "candidate_id": "candidate_003",
  "platform": "github"
}
```

Platform stats revealed per `check_platform`:

| Platform     | Fields                                                                                                          |
| ------------ | --------------------------------------------------------------------------------------------------------------- |
| **GitHub**   | `repos`, `top_languages`, `commit_streak_days`, `stars_received`, `notable_projects`, `contributions_last_year` |
| **LeetCode** | `problems_solved`, `easy`, `medium`, `hard`, `contest_rating`, `global_rank_percentile`                         |
| **Kaggle**   | `rank`, `competitions_entered`, `best_finish_percentile`, `medals`                                              |

---

## Tasks

|   Task   | Name               | Candidates | Budget | Steps/Candidate | Label Split  | Threshold | Grading                                                         |
| :------: | ------------------ | :--------: | :----: | :-------------: | :----------: | :-------: | --------------------------------------------------------------- |
|  `easy`  | Screening Sprint   |     10     |   40   |       4.0       | 3S / 4H / 3R |   0.75    | Weighted accuracy (shortlists 2x) + perfect-shortlist bonus     |
| `medium` | Conflicted Signals |     5      |   25   |       5.0       | 2S / 2H / 1R |   0.70    | Accuracy                                                        |
|  `hard`  | Batch Triage       |     20     |   30   |       1.5       | 5S / 8H / 7R |   0.60    | Shortlist F1 score + efficiency bonus                           |

### Task-specific grading

In every task an undecided candidate counts as wrong.

**Easy** : Weighted accuracy where ground-truth shortlists weigh 2x. If the agent shortlists exactly the 3 ground-truth shortlists: +0.2 bonus. Score = (accuracy + bonus) / 1.2, so only a perfect episode reaches the maximum.

**Medium** : Share of correct decisions. The task is meant to test conflicting signals across platforms, but the generator does not produce them yet, so the grader does not reward extra checks. [#9](https://github.com/ishanavasthi/recruit-env/issues/9) adds a real conflict mode.

**Hard** : Primary metric is **shortlist F1 score** (precision &times; recall of shortlist decisions). Efficiency bonus: `(steps_saved / max_steps) * 0.1`. With only 1.5 steps per candidate, the agent must triage aggressively.

Scores are clamped to [0.001, 0.999].

---

## Scoring

An episode has one score: the task grader's score.

- **Sparse reward.** Every step before the last returns `step_reward = 0.0`. The terminal step returns the grader's score as `step_reward`, and `cumulative_reward` (the running sum) equals it. `POST /grader` reports the same number.
- **Scoring contract.** `tests/test_scoring_contract.py` plays a reference solution and six degenerate policies over 200 seeds per task. The reference solution must score at least 0.99 on every seed. Doing nothing, holding everyone, shortlisting everyone, rejecting everyone, deciding one candidate and idling, and revealing everything without deciding must each fall below the task threshold, and at least 0.4 below the reference, on every seed.

Mean score over 200 seeds per task (`scripts/probe_scoring.py`):

| Policy                                   |  easy |  medium |  hard |
| ---------------------------------------- | ----: | ------: | ----: |
| Reference solution (all correct)         | 0.999 |   0.999 | 0.999 |
| Do nothing                               | 0.001 |   0.001 | 0.001 |
| All hold                                 | 0.256 |   0.400 | 0.033 |
| All shortlist                            | 0.385 |   0.400 | 0.433 |
| Decide one true reject, then idle        | 0.064 |   0.200 | 0.001 |

**Known limitation.** A single platform field still separates the labels, so one check per candidate is enough to pass every task. [#9](https://github.com/ishanavasthi/recruit-env/issues/9) makes information gathering matter.

---

## Fairness

Version 2.0.0 takes fairness out of the score. The v1 penalty compared shortlist rates across six name-origin groups on pools of 5&ndash;20 candidates, where a group often has one member, so it penalised a 100%-correct agent in 94&ndash;100% of seeds. Names are assigned independently of quality, so the penalty measured small-sample noise, not bias.

`env/fairness.py` (`FairnessChecker`) is still in the codebase. [#10](https://github.com/ishanavasthi/recruit-env/issues/10) replaces it with counterfactual twins: candidate pairs with identical signals and names from different groups, where a different decision within a pair is a measurable flip.

---

## Quick Start

### Docker (recommended)

```bash
docker build -t recruitenv .
docker run -p 7860:7860 recruitenv
```

The server starts at `http://localhost:7860`. Health check:

```bash
curl http://localhost:7860/health
# {"status": "ok", "version": "2.0.0"}
```

### Local

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Running the Baseline

The baseline agent uses a structured candidate loop with LLM-assisted decisions (OpenAI API) and rule-based fallbacks for tight budgets.

```bash
# Start the server first
uvicorn api.main:app --host 0.0.0.0 --port 7860 &

# Run the baseline
export OPENAI_API_KEY=your_key_here
python baseline/run_baseline.py
```

The baseline uses three strategies depending on step budget:

- **Full mode** (easy, &ge;2.8 steps/candidate): check GitHub + LeetCode, then LLM decides
- **Quick mode** (medium, &ge;1.8 steps/candidate): check GitHub only, then LLM decides
- **Batch mode** (hard, <1.8 steps/candidate): check GitHub while the budget still leaves a step for every remaining decision, then rule-based decisions

Every statistic comes from the observation, and each check costs a step.

### LLM inference script

`inference.py` plays one episode with any OpenAI-compatible model and prints the score from `POST /grader`:

```bash
ENV_URL=http://localhost:7860 TASK=easy SEED=42 \
API_BASE_URL=https://router.huggingface.co/v1 MODEL_NAME=<model> HF_TOKEN=<key> \
python inference.py
```

Results are saved to `baseline/baseline_scores.json`.

---

## Baseline Scores

| Task        |    Score    | Threshold | Steps Used | Status  |
| ----------- | :---------: | :-------: | :--------: | :-----: |
| easy        |   &mdash;   |   0.75    | &mdash;/40 | &mdash; |
| medium      |   &mdash;   |   0.70    | &mdash;/25 | &mdash; |
| hard        |   &mdash;   |   0.60    | &mdash;/30 | &mdash; |
| **Average** | **&mdash;** |           |            |         |

_Run `python baseline/run_baseline.py` to populate._

---

## API Reference

### `GET /health`

```bash
curl http://localhost:7860/health
```

```json
{ "status": "ok", "version": "2.0.0" }
```

### `POST /reset`

Start a new episode.

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'
```

```json
{
  "task_id": "easy",
  "step_number": 0,
  "steps_remaining": 40,
  "job_description": {"role": "Backend Developer", "required_skills": ["REST APIs", "SQL", "Docker", "CI/CD", "System Design"], ...},
  "candidates_summary": [{"id": "candidate_001", "name": "Dmitri Volkov"}, ...],
  "revealed_data": {},
  "decisions_made": {},
  "done": false
}
```

### `POST /step`

Execute one agent action.

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "check_platform", "candidate_id": "candidate_001", "platform": "github"}'
```

```json
{
  "observation": {"task_id": "easy", "step_number": 1, "steps_remaining": 39,
                  "revealed_data": {"candidate_001": {"resume_sections": {}, "platforms": {"github": {"repos": 3, ...}}}}, ...},
  "reward": {"step_reward": 0.0, "cumulative_reward": 0.0, "breakdown": {}},
  "done": false,
  "info": {"data": {"repos": 3, "stars_received": 7, "commit_streak_days": 4, ...}}
}
```

### `GET /state`

Return the agent-visible episode state: task, budget, job description, candidate ids and names, revealed data, decisions, and the episode score once done. It contains no candidate profiles and no ground truth.

```bash
curl http://localhost:7860/state
```

### `GET /tasks`

List available tasks and the Action JSON Schema.

```bash
curl http://localhost:7860/tasks
```

```json
{
  "tasks": [
    {"id": "easy", "name": "Screening Sprint", "max_steps": 40, "candidate_count": 10, ...},
    {"id": "medium", "name": "Conflicted Signals", "max_steps": 25, "candidate_count": 5, ...},
    {"id": "hard", "name": "Batch Triage", "max_steps": 30, "candidate_count": 20, ...}
  ],
  "action_schema": {"oneOf": [...], "discriminator": {"propertyName": "type"}}
}
```

### `POST /grader`

Grade a completed episode. Returns 400 if the episode is still running. The score equals the terminal step's `cumulative_reward`; the breakdown includes the ground truth, which is why it is only available after the episode.

```bash
curl -X POST http://localhost:7860/grader
```

```json
{
  "score": 0.999,
  "task_id": "easy",
  "breakdown": {
    "grader_score": 0.999,
    "success": true,
    "threshold": 0.75,
    "correct": 10,
    "total": 10,
    "steps_used": 10,
    "max_steps": 40,
    "decisions": {"candidate_001": "reject", "candidate_002": "shortlist", ...},
    "ground_truth": {"candidate_001": "reject", "candidate_002": "shortlist", ...}
  }
}
```

### `GET /baseline`

Run the built-in rule-based baseline agent on all tasks (no API key needed).

```bash
curl http://localhost:7860/baseline
```

```json
{
  "scores": { "easy_task": 0.9359, "medium_task": 0.999, "hard_task": 0.3333 },
  "timestamp": "2026-04-08T00:00:00+00:00"
}
```

---

## Project Structure

```
recruitenv/
├── __init__.py                # Package root
├── client.py                  # OpenEnv client stub (rewritten in Phase 4 of the roadmap)
├── inference.py               # LLM agent: plays one episode, reports the /grader score
├── pyproject.toml             # Project metadata, dependencies, entry points
├── openenv.yaml               # OpenEnv manifest
├── Dockerfile                 # Python 3.11-slim, non-root user, healthcheck
├── docker-compose.yml         # Single service, port 7860
├── requirements.txt           # Runtime dependencies
├── requirements-dev.txt       # + pytest
├── CHANGELOG.md
├── LICENSE
├── README.md
│
├── server/                    # OpenEnv standard entry point
│   ├── __init__.py
│   ├── app.py                 # uvicorn launcher (openenv runs `uv run server`)
│   └── models.py              # Re-exports env.models for openenv compatibility
│
├── env/                       # Core environment (zero external dependencies)
│   ├── __init__.py            # Re-exports all public symbols
│   ├── models.py              # Pydantic v2 models: Action (discriminated union), Observation, Reward, EpisodeState
│   ├── environment.py         # RecruitmentEnvironment: reset(), step(), state(), verifier_state(), grade_report()
│   ├── profile_factory.py     # Synthetic profile generator: 50 diverse names, 3 tiers, 4 role types
│   └── fairness.py            # FairnessChecker (v1 parity penalty; not part of the score since 2.0.0)
│
├── tasks/                     # Task definitions and graders
│   ├── __init__.py            # TASK_REGISTRY + GRADER_REGISTRY
│   ├── base.py                # TaskConfig dataclass + BaseGrader ABC
│   ├── easy_task.py           # Screening Sprint: 10 candidates, weighted accuracy grader
│   ├── medium_task.py         # Conflicted Signals: 5 candidates, accuracy grader
│   └── hard_task.py           # Batch Triage: 20 candidates, F1 grader
│
├── api/                       # FastAPI application
│   ├── __init__.py
│   └── main.py                # All endpoints, CORS, request logging, async locking
│
├── baseline/                  # Baseline agent
│   └── run_baseline.py        # Structured loop: LLM decisions + rule-based fallback
│
├── scripts/
│   └── probe_scoring.py       # Scores fixed policies over 200 seeds per task
│
├── docs/research/             # Findings and roadmap
│
└── tests/                     # pytest suite
    ├── test_environment.py    # Reset determinism, step actions, terminal conditions, observation sufficiency
    ├── test_graders.py        # Grader accuracy, determinism, F1 validation, range checks
    ├── test_api.py            # Endpoint responses, full episode flow, no ground truth mid-episode
    ├── test_inference.py      # inference.py episode loop against the API, scripted model
    └── test_scoring_contract.py  # Reference solution vs. degenerate policies, 200 seeds per task
```
