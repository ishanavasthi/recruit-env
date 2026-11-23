# Changelog

## 2.0.0

Breaking changes to the scoring and the API. The previous behaviour is tagged `v1.0.0`.
The reasons are in [docs/research/2026-11-21-findings-and-roadmap.md](docs/research/2026-11-21-findings-and-roadmap.md) (Phases 0 and 1).

### One honest score (#6)

- The reward is sparse. Every step before the last returns 0.0, and the terminal step returns the task grader's score. `cumulative_reward`, the episode score and `POST /grader` are the same number.
- Removed the per-step information rewards (+0.02 / +0.03), the separate terminal accuracy formula, the efficiency bonus in the environment and the fairness penalty. `env/rewards.py` is gone.
- Removed the `score_dimension` action, which paid +0.02 per call and could be farmed. `Observation.scores_recorded` is gone with it.
- An undecided candidate now counts as wrong in the easy and medium graders.
- The easy grader divides by 1.2, so only a perfect episode reaches the maximum.
- The medium grader no longer pays for checking candidates 0 and 1 on every platform. The generator never made those candidates conflicted; #9 adds a real conflict mode.
- `tests/test_scoring_contract.py`: a reference solution must score at least 0.99, and six degenerate policies must each fail the task by at least 0.4, on every one of 200 seeds per task.
- `Reward` now has only `step_reward`, `cumulative_reward` and `breakdown`.

### Hidden verifier (#5)

- `Observation.revealed_data` carries the revealed values: `{"resume_sections": {name: section}, "platforms": {name: stats}}`.
- `GET /state` and `RecruitmentEnvironment.state()` return `PublicState`, with no candidate profiles and no ground truth. `verifier_state()` returns the full state, for graders and tests only.
- `RecruitmentEnvironment.grade_report()` returns the score and the per-candidate breakdown once the episode is done.
- `baseline/run_baseline.py` no longer reads `/state`; it reads statistics from the observation and pays a step for each check.

### Hygiene (part of #11)

- Root `models.py` is UTF-8, so `import models` works.
- Build backend is `setuptools.build_meta`; `uv build` succeeds.
- Added `LICENSE` (MIT), `requirements-dev.txt` and a GitHub Actions test workflow. `pytest` left the runtime requirements.
- Removed `package.json`, `package-lock.json` and `validate_submission.sh`.
- Corrected the task text and version in `openenv.yaml`.
- CORS no longer allows credentials with a wildcard origin.

### Agent scripts

- `inference.py` puts revealed values and the job description in the prompt, reports the `POST /grader` score, and takes `ENV_URL`, `TASK`, `SEED` and `MAX_TOKENS` from the environment.
- `baseline/run_baseline.py` parses on Python 3.11; it used a backslash inside an f-string expression.
