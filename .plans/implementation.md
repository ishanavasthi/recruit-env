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
| 4 | Standalone core, OpenEnv adapter, concurrency (#11) | In progress | PR C |
| — | JSONL transition logs and replay (#8) | In progress | PR C |
| 5 | Baseline table and training example (#12) | In progress | PR C; the GPU reward curve awaits the owner's choice of Colab or HF Jobs |
| 6 | Public face: README, results page, report, demo, PyPI | Not started | PR D |
| 7 | Extensions (#2 ranking, #3 reasons, per-action costs, …) | Not started | later |

## What exists (3.0.0)

- **Core:** `env/` holds the environment, generator, rubric, names, fairness and reference policies. `tasks/` holds three task configs and graders.
- **HTTP API:** `api/main.py` (FastAPI) exposes `/reset`, `/step`, `/state`, `/grader` and `/baseline`. It runs one global episode and serves the Space.
- **Agents:** `inference.py` (step by step, through an OpenAI-compatible endpoint) and `baseline/run_baseline.py` (Python gathers the evidence, then one model call per task).
- **Evidence scripts:** `scripts/probe_scoring.py`, `scripts/difficulty_sweep.py` and `scripts/fairness_report.py`, which writes `docs/results/fairness.{json,md}`.
- **Tests:** 204, including the scoring contract and acceptance criteria.

## Integrations

- **Hugging Face Space:** `heyavasthi/recruitenv`, a Docker Space on port 7860, running 3.0.0.
- **Model access for evaluation:** an OpenAI-compatible proxy on the development server. Claude Sonnet 5.5 and Haiku 5.5 have been run.
- **OpenEnv:** being reworked in PR C. The old `openenv-core` client is out of date.
