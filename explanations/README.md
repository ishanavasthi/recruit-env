# RecruitEnv — Study Guides

Comprehensive study material covering every major file and concept in this project. Read in order; each file builds on the previous one.

## Reading Order

| #  | File                                              | Time | What you'll learn                                              |
| -- | ------------------------------------------------- | ---- | -------------------------------------------------------------- |
| 00 | [Overview & Architecture](00-overview-and-architecture.md) | 8 min | The 60-second pitch + the system-wide picture                  |
| 01 | [Project Structure](01-project-structure.md)      | 5 min | File-by-file map of the repo                                   |
| 02 | [Pydantic Models](02-models.md)                   | 10 min | Every typed shape — actions, observations, profiles            |
| 03 | [Environment Core](03-environment-core.md)        | 12 min | How `reset` / `step` / `state` actually work                   |
| 04 | [Profile Factory](04-profile-factory.md)          | 10 min | How candidates and JDs are synthesised deterministically       |
| 05 | [Reward System](05-rewards.md)                    | 10 min | The four-component reward and why it's shaped that way         |
| 06 | [Fairness Checker](06-fairness.md)                | 5 min  | Demographic parity penalty                                     |
| 07 | [Tasks & Graders](07-tasks-and-graders.md)        | 8 min  | The three tasks and their distinct grading rubrics             |
| 08 | [FastAPI Server](08-api-server.md)                | 10 min | All HTTP endpoints + serialisation                             |
| 09 | [Baseline Agent](09-baseline-agent.md)            | 8 min  | How the LLM baseline structured loop works                     |
| 10 | [Tests](10-tests.md)                              | 8 min  | The 35-test pytest suite                                       |
| 11 | [Deployment](11-deployment.md)                    | 6 min  | Docker, HF Spaces, OpenEnv manifest                            |
| 12 | [Interview Q&A](12-interview-qa.md)               | 10 min | The questions you'll be asked + ready-made answers             |

**Total: ~110 minutes** for a thorough read. The night before the interview, prioritise: 00 → 12 → 05 → 03 → 02.

## Quick-Recall Cheat Sheet

If you only have 5 minutes:

- **What is it:** OpenEnv-compliant RL environment for candidate pipeline triage. FastAPI app on port 7860, Docker, deployed to HF Spaces.
- **Three tasks:** easy (10 cands / 40 steps), medium (5 / 25), hard (20 / 30). Graders: weighted accuracy, thoroughness, F1.
- **Four actions:** `read_resume_section`, `check_platform`, `score_dimension`, `make_decision`. Discriminated union via Pydantic.
- **Four reward components:** step novelty (+0.02 / +0.03), terminal accuracy (role-weighted, asymmetric error costs), efficiency bonus (+0.1 all-or-nothing), fairness penalty (0 / 0.1 / 0.2).
- **Determinism:** all RNG flows through one `numpy.random.default_rng(seed)`, reseeded per generator call. `tests/test_environment.py:test_reset_deterministic` enforces it.
- **Baseline scores:** easy 1.0, medium 1.0, hard 0.92, avg 0.974 (recorded with Gemini 2.0 Flash Lite).
- **Layering:** `env/` is pure RL core (no HTTP), `api/` is the FastAPI transport, `tasks/` defines scenarios + graders, `baseline/` is the LLM agent script.

## How These Docs Were Written

These docs were written by reading every Python file in the project end-to-end and explaining what each module does, why the design decisions were made, and how the pieces fit together. They reference specific files and line numbers so you can always cross-check.

**The point of studying these is not to memorise the answers verbatim — it's to build a mental model strong enough that you can answer follow-up questions you don't see coming.** If an interviewer asks "what would happen if the JD weights didn't sum to 1.0?", you should be able to derive the answer (Pydantic's `@model_validator(mode='after')` would raise at construction time → covered in `02-models.md`) rather than recalling the exact sentence.

Good luck.
