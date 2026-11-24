# Changelog

## 4.0.0

Breaking changes to the package layout, the server and the agent scripts. Design: [docs/superpowers/specs/2026-11-25-standalone-core-logs-baselines-design.md](docs/superpowers/specs/2026-11-25-standalone-core-logs-baselines-design.md) (roadmap Phases 4 and 5). The task, generator, rubric and graders are unchanged: the pinned generator hashes and acceptance numbers are identical.

### A standalone library (#11)

- `env/` and `tasks/` are now one package, `recruitenv` (`recruitenv.tasks` for the tasks). The root `models.py`, `__init__.py`, `client.py` and `server/models.py` are gone.
- The core depends only on `pydantic` and `numpy` and runs in process: `RecruitmentEnvironment().reset(...)`, `.step(...)`, `.grade_report()`. A test checks that it imports no web, OpenEnv or model-client module.
- Extras: `openenv` (server and typed client), `agents` (model agents), `dev`.

### OpenEnv adapter and server (#11)

- `server/app.py` serves the core through `openenv.core.create_app` (openenv 0.14): one environment per `/ws` session, up to `MAX_CONCURRENT_ENVS` (default 16), with the Task API under `/recruitenv/`, `/health`, `/schema`, `/metadata` and `/mcp`. `GET /tasks` and `GET /baseline` remain.
- `recruitenv/openenv_adapter.py` defines a flat `RecruitAction`, `RecruitObservation` (the grade report is in `metadata["grade"]` on the terminal step), `RecruitState` and the reset parameters `task`, `seed`, `name_seed`, `name_blind`.
- `recruitenv/client.py` is a typed `EnvClient`.
- **Removed:** the custom FastAPI app (`api/`) and its stateful `POST /reset`, `POST /step` and `POST /grader`. OpenEnv's HTTP routes are stateless; episodes run over `/ws` or in process. `v1.0.0` and `v3.0.0` remain available by tag.
- `openenv.yaml` has a `validation:` block. `openenv validate --url` passes every check; `--level static` passes the manifest check (the other static checks are not implemented in openenv 0.14, so the verdict is WARN with exit 0). CI runs the static check.
- Concurrency tests: two sessions on the same seed with different name draws, interleaved in process, threaded, and over two websockets, each match a solo run.

### Transition logs and replay (#8)

- `recruitenv/trajectory.py`: `LoggedEnvironment` writes one JSONL file per episode (header, one record per step with the action, what it revealed and an observation hash, and a final grade record).
- `scripts/replay.py` replays logs and checks every observation and the grade reproduce exactly; it refuses logs from another generator or package version.

### Agents without fallbacks

- `inference.py` runs in process by default and against a server with `--url`. It logs every episode.
- `baseline/run_baseline.py` (decide-only: Python gathers, the model decides once) also runs in process and logs.
- **No substitute actions.** Both used to replace an unusable model reply with a fallback ("hold" the next candidate, or the rubric's own decisions), so a reported model score could be partly another policy's. Now a model error, an unparseable reply or a reply cut off by the token cap is retried (a cut-off doubles the cap, up to 32,000), and when retries run out the episode is reported as incomplete.
- **Token cap.** The default cap rose from 512 to 16,000. With 512, every pilot episode of Claude Haiku 5.5 and Sonnet 5.5 hit empty replies cut off while the model was still reasoning. At 16,000, every pilot episode completed with no cut-off, and decision steps used up to about 7,500 completion tokens.

### Benchmarks and training (#12, in part)

- `scripts/run_benchmark.py` runs a policy or a model over seeds and tasks, resumably, and stops scheduling on repeated rate limiting. `scripts/baselines_report.py` writes [docs/results/baselines.md](docs/results/baselines.md) with bootstrap intervals, seed counts and incomplete episodes.
- `recruitenv/tools.py`: a tool-per-action environment matching TRL's `GRPOTrainer(environment_factory=...)`. `examples/train_grpo.py` is a uv script for a GPU (Colab or HF Jobs). The reward curve from a real run is still to come.

### Correction to 3.0.0's published model scores

The 3.0.0 pull request reported model scores that are now withdrawn:
- Claude Sonnet 5.5, decide-only: 0.999, 0.999 and 0.40.
- Claude Haiku 5.5, step by step: 0.34 on medium and 0.33 on hard.

Both scripts then substituted a fallback for an unusable reply, and their output did not record whether the fallback fired.
- **Haiku.** At the step-by-step script's 512-token cap, a pilot of the same models on 2026-11-25 returned empty, cut-off replies within the first few steps of every episode. Each of those would have become a fallback "hold".
- **Sonnet.** The decide-only script used a 4,000-token cap. In this release's decide-only runs at 16,000, 19 of Sonnet's 30 replies (and all 30 of Haiku's) needed more than 4,000 completion tokens. So the old scores were probably largely the rubric fallback's.

[docs/results/baselines.md](docs/results/baselines.md) replaces these numbers.

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
