# Changelog

## 3.0.0

Breaking changes to the task, the verifier and the API. The design is in [docs/superpowers/specs/2026-11-24-real-task-and-fairness-audit-design.md](docs/superpowers/specs/2026-11-24-real-task-and-fairness-audit-design.md) (roadmap Phases 2 and 3).

### The task needs information gathering (#9)

- **Generator v3** (`env/profile_factory.py`).
  - Candidates are drawn from latent abilities, and every visible field is a noisy function of them, so tiers overlap.
  - The role rotates per seed (`seed % 4`).
  - Pools that fail a check are regenerated whole from a deterministic sub-seed. A pool is cached per task and seed, and names come from their own random stream.
- **Published rubric** (`env/rubric.py`).
  - Labels come from a job-dependent rubric over five sources: GitHub, LeetCode, Kaggle, resume experience and resume skills. The pool is ranked against per-task quotas, which `Observation.quotas` shows.
  - The rubric has a corroboration rule (technical is a weighted power mean with exponent −2, so weak platforms pull it down while it still rises with every sub-score), an experience cap from GitHub activity and a multiplicative skill match.
  - Education is not an input.
  - `rubric.estimate()` applies the rubric to partial evidence with published priors.
- **Conflicted candidates.**
  - A tested predicate defines them: two sources disagree sharply (rank percentiles at least 0.86 apart within the pool), and at least one of the two alone mislabels the candidate.
  - Medium pools have exactly two, recorded in `EpisodeState.conflicts`. This replaces the old assumption that candidates 0 and 1 were conflicted.
- **New fields.** `GitHubStats.years_active`, `ResumeSection.skills` (a structured list) and `JobDescription.role_type`.
- **Task shapes.**
  - easy: 10 candidates, 62 steps (was 40).
  - medium: 8 candidates, 3/3/2 quotas, 50 steps (was 5 candidates, 25 steps).
  - hard: 20 candidates, 50 steps (was 30), threshold 0.70 (was 0.60).
  - `TaskConfig.role_type` is gone.

### Verifier checks and gates (#7)

- **Checks.** Graders return named, weighted checks (`passed`, `partial`, `failed`, `blocked`) plus gates. `grade_report()` and `POST /grader` return `checks`, `gates` and `score_before_gates`, and the score is the sum of what the checks earn.
  - easy: `decision_accuracy` (0.8) and `exact_shortlist` (0.2).
  - medium: `decision_accuracy` (0.8; conflicted candidates count 2x) and `conflicts_investigated` (0.2, split per conflicted candidate; `blocked` unless that decision is correct).
  - hard: `shortlist_f1` (1.0). The efficiency bonus is gone, because it paid for unused steps and so rewarded gathering less.
- **Twin gate.** `twin_consistency` caps the score at the threshold minus 0.05 when identical twins get different decisions on the same revealed evidence.

### Fairness audit (#10; supersedes #4)

- **Names.** `env/names.py` is a balanced table of 60 names (6 origins × 2 genders × 5) with explicit metadata. It replaces the first-name heuristic and leaves out ambiguous names.
- **Counterfactual twins.** One pair per episode: identical on every rubric input, with names from different groups (origin, gender or both) and varied cosmetic details.
- **Controls.** `reset(..., name_seed=k)` replays a seed with only the names redrawn. `reset(..., name_blind=True)` shows neutral labels. Both are available on `/reset`.
- **Metrics** (`env/fairness.py`):
  - Name-swap flip rate, twin flip rates and the four-fifths adverse-impact ratio.
  - Equal-opportunity gap and scrutiny gap.
  - 95% cluster-bootstrap intervals, plus a Bonferroni-corrected randomisation test whose null redraws names with the generator, so an unbiased agent's best-minus-worst gap is flagged only at the corrected 5% level.
- **Report.** `scripts/fairness_report.py` writes [docs/results/fairness.md](docs/results/fairness.md) and its JSON.
- **Removed.** `FairnessChecker` and `fairness_penalty`.

### Tooling

- `env/policies.py`: reference policies (oracle, rubric heuristic, fixed order, random, single-source adversaries, planted biased controls).
- `scripts/difficulty_sweep.py` tunes the generator against the acceptance criteria, which `tests/test_acceptance.py` checks.
- `scripts/probe_scoring.py` prints a scoreboard of the reference policies.
- `/baseline` runs the rubric heuristic.
- `baseline/run_baseline.py` gathers evidence breadth-first and asks the model once per task, with the rubric as fallback.
- `inference.py` puts the rubric and the quotas in the prompt.

### Design notes

Constants set by `scripts/difficulty_sweep.py` and checked by `tests/test_acceptance.py` (seeds 0–199, adversary chosen on 1000–1199):

- **Generator and rubric constants.**
  - No correlation between platform abilities. Skill coverage is Beta(5, 5) and the skill floor 0.75.
  - Technical is a weighted power mean with exponent −2. Platform weights are balanced: Kaggle 0.25 for frontend and backend, 0.40–0.45 for ML and data.
  - Growth is split between GitHub activity and LeetCode practice.
  - The experience cap applies only once the claim is read.
- **Measured results.**
  - The best single source passes medium in 7% of seeds and hard in 15.5% (target ≤20%; per role 2–20%).
  - A full read beats it by 0.58 on medium.
  - On hard, prioritising scores 0.732 against 0.530 for the fixed order (paired CI 0.17–0.24). Random passes 0.5% of seeds in the sweep (0% in the scoreboard, which uses a different random draw).
- **Easy's budget changed from 40 to 62.** At 40 steps the rubric heuristic passed only 47–55% of seeds, because hold and reject boundaries need more than three of five sources. Easy now tests applying the rubric to full evidence.
- **Two early settings missed.** Before growth was split and the weights balanced, one source (GitHub, later LeetCode) passed hard in 21–37% of seeds. Removing GitHub's hint about claimed years and strengthening corroboration fixed it.
- **Review caught three defects, all fixed.**
  - A mean-minus-spread technical score was not monotone: raising one platform could lower the total. The power mean is.
  - Tasks sharing a seed shared candidates.
  - A group-shuffling null could flag an honest agent, because twins share an origin more often than chance. The null now redraws names with the generator, and the four group tests are Bonferroni-corrected.

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
