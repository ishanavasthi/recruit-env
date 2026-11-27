# Implementation

## Milestones

| Phase | What | Status | Where |
|---|---|---|---|
| 0 | Hygiene: packaging, encoding, licence, `v1.0.0` tag | Done (2.0.0) | PR #13 |
| 1 | One honest score: grader = terminal reward, no ground-truth leaks, contract tests | Done (2.0.0) | PR #13 |
| 2 | Real task: latent generator v3, published rubric, conflicts, role rotation | Done (3.0.0) | PR #14 |
| 3 | Fairness audit: names table, twins, name-swap replays, metrics with CIs | Done (3.0.0) | PR #14 |
| — | Verifier checks and gates (#7) | Done (3.0.0) | PR #14 |
| — | Hugging Face Space redeployed to 3.0.0 | Done 2026-11-25 | Space commit `3e29bbe` |
| — | Hugging Face Space redeployed to 4.0.0 | Done 2026-11-25 | Space commit `f15a1ca` |
| 4 | Standalone core, OpenEnv adapter, concurrency (#11) | Done (4.0.0) | PR #15 |
| — | JSONL transition logs and replay (#8) | Done (4.0.0) | PR #15 |
| 5 | Baseline table and training example (#12) | Mostly done. The baseline table covers 10 seeds (goal 50+, still open). First reward curve done on Kaggle (2026-11-27): reward rises to near the blind-labelling line, not past it | PR #15; reward-curve PR |
| 6 | Public face: README, results page, report, demo, PyPI | Not started | later |
| 7 | Extensions (#2 ranking, #3 reasons, per-action costs, …) | Not started | later |

## What exists (4.0.0)

- **Core (`recruitenv/`, depends on pydantic and numpy only):**
  - environment, generator, rubric, names, fairness and reference policies;
  - `trajectory.py` (JSONL logs and replay) and `tools.py` (the TRL tool environment);
  - `tasks/`.
- **OpenEnv (the `openenv` extra):**
  - `openenv_adapter.py` and `client.py`;
  - `recruitenv/server.py`, which the Space runs: one environment per `/ws` session, the Task API, `/tasks` and `/baseline`.
- **Agents:**
  - `inference.py`, step by step;
  - `baseline/run_baseline.py`, decide-only.

  Both run in process, log every episode, and have no fallback.
- **Evidence:**
  - `scripts/run_benchmark.py` and `baselines_report.py` write `docs/results/baselines.md`;
  - `scripts/fairness_report.py`, `probe_scoring.py` and `difficulty_sweep.py`;
  - `scripts/replay.py`.
- **Training:** `examples/train_grpo.py` (GRPO with TRL), run on Kaggle through `scripts/kaggle_train.py`. `scripts/training_references.py` and `training_report.py` produce `docs/results/training.md`.

## Integrations

- **Hugging Face Space:** `heyavasthi/recruitenv`, a Docker Space on port 7860, running 4.0.0 (redeployed 2026-11-25, Space commit `f15a1ca`).
- **Model access for evaluation:** an OpenAI-compatible proxy on the development server. Claude Sonnet 5.5 and Haiku 5.5 have been run.
- **Kaggle:** free T4 notebooks for training, driven by the Kaggle CLI from the development server. Notebooks are private, and no credentials go into them.
- **OpenEnv 0.14:** `openenv validate --url` passes every check, and static validation passes the manifest check.
