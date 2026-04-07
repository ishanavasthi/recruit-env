# RecruitEnv

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)
![HF Spaces](https://img.shields.io/badge/HF%20Spaces-ready-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)
![Tests](https://img.shields.io/badge/tests-35%20passed-brightgreen)

An OpenEnv-compliant reinforcement learning environment that simulates **candidate pipeline triage**, the process of reviewing developer profiles across multiple platforms and making shortlisting decisions against a job description.

---

## Overview

Recruiters at scale face a combinatorial problem: each candidate has signals scattered across GitHub, LeetCode, Kaggle, and their resume. Checking every signal for every candidate is expensive. The best recruiters develop efficient heuristics, check the most informative signals first, decide fast on clear cases, spend time on ambiguous ones.

RecruitEnv turns this into a sequential decision problem. An RL agent receives a job description and a pool of candidates. Each step, it can gather one signal (check a platform, read a resume section) or make a decision (shortlist, hold, reject). The agent has a limited step budget, so it must learn to **prioritize information gathering** and **make accurate decisions under time pressure**.

The environment is fully synthetic and deterministic. No live API calls. `reset(seed=42)` always produces identical state. All candidate profiles are generated from controlled statistical distributions matching real-world developer tiers (junior/mid/senior), with ground-truth labels embedded at generation time.

### Why train here

- **Information asymmetry**: the agent starts knowing nothing about candidates and must pay (in steps) to reveal signals
- **Exploration-exploitation tradeoff**: gathering more signals improves decisions but consumes budget
- **Multi-objective optimization**: accuracy, efficiency, and demographic fairness are all scored
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
    score_dimension(candidate_id, dimension, score) -- records assessment
    make_decision(candidate_id, decision)          -- terminal per candidate

  environment returns:
    observation  -- updated revealed data, decisions so far
    reward       -- step reward for information gathering
    done         -- True when all decided OR budget exhausted
    info         -- action-specific metadata

episode ends --> grader scores decisions against ground truth
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
| `revealed_data`      | `dict`            | `candidate_id -> {"resume_sections": [...], "platforms": [...]}`       |
| `decisions_made`     | `dict[str, str]`  | `candidate_id -> "shortlist" \| "hold" \| "reject"`                    |
| `scores_recorded`    | `dict[str, dict]` | `candidate_id -> {"technical": 0.8, ...}`                              |
| `done`               | `bool`            | Whether the episode has ended                                          |

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

| Action                | Parameters                                                                                     |        Step Reward         | Effect                                            |
| --------------------- | ---------------------------------------------------------------------------------------------- | :------------------------: | ------------------------------------------------- |
| `read_resume_section` | `candidate_id`, `section` (`"education"` \| `"experience"` \| `"skills"`)                      | +0.02 (new) / 0.0 (repeat) | Reveals resume section text + years of experience |
| `check_platform`      | `candidate_id`, `platform` (`"github"` \| `"leetcode"` \| `"kaggle"`)                          | +0.03 (new) / 0.0 (repeat) | Reveals full platform statistics                  |
| `score_dimension`     | `candidate_id`, `dimension` (`"technical"` \| `"experience"` \| `"growth"`), `score` (0.0-1.0) |           +0.02            | Records agent's assessment score                  |
| `make_decision`       | `candidate_id`, `decision` (`"shortlist"` \| `"hold"` \| `"reject"`)                           |            0.0             | Terminal for this candidate. Cannot be reversed.  |

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
|  `easy`  | Screening Sprint   |     10     |   40   |       4.0       | 3S / 4H / 3R |   0.75    | Weighted accuracy (shortlist errors 2x) + perfect bonus         |
| `medium` | Conflicted Signals |     5      |   25   |       5.0       | 2S / 2H / 1R |   0.70    | Accuracy + thoroughness bonus/penalty for conflicted candidates |
|  `hard`  | Batch Triage       |     20     |   30   |       1.5       | 5S / 8H / 7R |   0.60    | Shortlist F1 score + efficiency bonus                           |

### Task-specific grading

**Easy** : Weighted accuracy where shortlist misclassifications cost 2x. If the agent identifies exactly the 3 ground-truth shortlists with zero false positives: +0.2 bonus. Score clamped to [0, 1].

**Medium** : Two candidates have deliberately conflicting signals (e.g., strong GitHub but weak LeetCode). Per-conflicted-candidate: +0.15 if agent checked all 3 platforms before deciding (thoroughness), -0.15 if agent decided with <2 platforms checked (hastiness).

**Hard** : Primary metric is **shortlist F1 score** (precision &times; recall of shortlist decisions). Undecided candidates auto-graded as "hold". Efficiency bonus: `(steps_saved / max_steps) * 0.1`. With only 1.5 steps per candidate, the agent must triage aggressively.

---

## Reward Function

The reward has four components:

### Step Reward

Small positive reward for gathering new information. Incentivizes exploration over idle actions.

```
new resume section:  +0.02
new platform check:  +0.03
score dimension:     +0.02
repeat/decision:      0.00
```

### Terminal Accuracy Reward

At episode end, each decision is scored against ground truth:

| Decision             | Truth     |                           Score |
| -------------------- | --------- | ------------------------------: |
| Correct shortlist    | shortlist | +1.0 &times; `weight_technical` |
| Correct reject       | reject    |                            +0.5 |
| Correct hold         | hold      |                            +0.3 |
| Shortlisted a reject | reject    |                            -0.5 |
| Rejected a shortlist | shortlist |                            -0.3 |
| Any other error      | &mdash;   |                            -0.1 |

Raw score normalized to [0, 1] across the range [-0.5n, 1.0n].

### Efficiency Bonus

+0.1 if **all** decisions are correct **and** the agent used less than 60% of the step budget.

### Fairness Penalty

See below.

**Final score** = `clamp(accuracy + efficiency - fairness_penalty, 0, 1)`

---

## Fairness Mechanism

The environment checks whether the agent's shortlist rate is balanced across demographic groups. Candidate names are mapped to 6 origin groups (South Asian, East Asian, Latin American, European, Middle Eastern/North African, African) using a deterministic first-name lookup.

| Max shortlist-rate difference across groups | Penalty |
| :-----------------------------------------: | :-----: |
|                    < 0.2                    |   0.0   |
|               0.2 &ndash; 0.4               |   0.1   |
|                  &ge; 0.4                   |   0.2   |

This rewards agents that make decisions based on candidate qualifications rather than demographic patterns in the data. The penalty is subtracted from the final score.

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
# {"status": "ok", "version": "1.0.0"}
```

### Local

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload
```

### Tests

```bash
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
- **Batch mode** (hard, <1.8 steps/candidate): read all stats from `/state`, rule-based decisions for all 20 candidates in 20 steps

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
{ "status": "ok", "version": "1.0.0" }
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
  "scores_recorded": {},
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
  "observation": {"task_id": "easy", "step_number": 1, "steps_remaining": 39, ...},
  "reward": {"step_reward": 0.03, "cumulative_reward": 0.0, "fairness_penalty": 0.0, "accuracy_bonus": 0.0},
  "done": false,
  "info": {"data": {"repos": 3, "stars_received": 7, "commit_streak_days": 4, ...}}
}
```

### `GET /state`

Return full internal state (including ground-truth labels which are useful for debugging, not for agents).

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

Grade a completed episode. Returns 400 if episode is still running.

```bash
curl -X POST http://localhost:7860/grader
```

```json
{
  "score": 1.0,
  "task_id": "easy",
  "breakdown": {
    "grader_score": 1.0,
    "success": true,
    "threshold": 0.75,
    "correct": 10,
    "total": 10,
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
  "scores": { "easy_task": 1.0, "medium_task": 1.0, "hard_task": 0.3333 },
  "timestamp": "2026-04-08T00:00:00+00:00"
}
```

---

## Project Structure

```
recruitenv/
├── __init__.py                # Package root
├── client.py                  # OpenEnv client: RecruitEnvClient with reset(), step(), state()
├── pyproject.toml             # Project metadata, dependencies, entry points
├── openenv.yaml               # OpenEnv specification
├── Dockerfile                 # Python 3.11-slim, non-root user, healthcheck
├── docker-compose.yml         # Single service, port 7860
├── requirements.txt           # fastapi, uvicorn, pydantic, numpy, openai, pytest, httpx
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
│   ├── environment.py         # RecruitmentEnvironment: reset(), step(), state() — thread-safe
│   ├── profile_factory.py     # Synthetic profile generator: 50 diverse names, 3 tiers, 4 role types
│   ├── rewards.py             # RewardCalculator: step rewards, terminal scoring, efficiency bonus
│   └── fairness.py            # FairnessChecker: demographic parity of shortlist rates
│
├── tasks/                     # Task definitions and graders
│   ├── __init__.py            # TASK_REGISTRY + GRADER_REGISTRY
│   ├── base.py                # TaskConfig dataclass + BaseGrader ABC
│   ├── easy_task.py           # Screening Sprint: 10 candidates, weighted accuracy grader
│   ├── medium_task.py         # Conflicted Signals: 5 candidates, thoroughness grader
│   └── hard_task.py           # Batch Triage: 20 candidates, F1 grader
│
├── api/                       # FastAPI application
│   ├── __init__.py
│   └── main.py                # All endpoints, CORS, request logging, async locking
│
├── baseline/                  # Baseline agent
│   └── run_baseline.py        # Structured loop: LLM decisions + rule-based fallback
│
└── tests/                     # pytest suite (35 tests)
    ├── test_environment.py    # Reset determinism, step actions, terminal conditions, error handling
    ├── test_graders.py        # Grader accuracy, determinism, F1 validation, range checks
    └── test_api.py            # Endpoint responses, full episode flow, error codes
```
