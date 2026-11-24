# RecruitEnv: findings and roadmap

Date: 2026-11-21. Audited commit: `c195dbb` (`main`).

**Verdict.** RecruitEnv works as a first version and is incomplete as an RL environment. The server runs and all 35 tests pass, but one revealed field solves every task, the episode score rewards deciding less, and the fairness penalty punishes correct agents.

**Direction (owner decision, 2026-11-21).** RecruitEnv becomes a standalone flagship project: a benchmark and training environment for hiring agents whose distinctive feature is an audit-grade fairness measurement. It stands on its own name and results. OpenEnv is one integration it supports, not its identity.

The roadmap below fixes the environment first, then makes the core a standalone package with an OpenEnv adapter, then publishes evidence and a public face for the project.

This document is the spec for the next phases of work. Each phase gets its own implementation plan when it starts (see [Workflow and skills](#workflow-and-skills)).

## Status

Updated 2026-11-25. The living build log is in [`.plans/`](../../.plans/implementation.md).

| Phase | Status | Shipped in |
|---|---|---|
| 0. Hygiene | Done | 2.0.0, PR #13 |
| 1. One honest score | Done | 2.0.0, PR #13 |
| 2. Make the task real | Done | 3.0.0, PR #14 ([spec](../superpowers/specs/2026-11-24-real-task-and-fairness-audit-design.md)) |
| 3. Fairness as an audit | Done | 3.0.0, PR #14 |
| 4. Standalone core, OpenEnv adapter | Done | 4.0.0, PR C ([spec](../superpowers/specs/2026-11-25-standalone-core-logs-baselines-design.md)) |
| 5. Show learning | Baseline table and training example done; the reward curve needs a GPU run (the owner chooses Colab or HF Jobs) | 4.0.0, PR C |
| 6. Public face | Not started | PR D |
| 7. Extensions | Not started | later |

The Hugging Face Space runs 3.0.0 (redeployed 2026-11-25).

## How the evidence was produced

- **Tests:** `pytest tests/ -q` in a throwaway Python 3.11 environment: 35 passed.
- **Probes:** [`scripts/probe_scoring.py`](../../scripts/probe_scoring.py) plays six fixed policies over 200 seeds per task and reports both scores. Every number in the findings comes from this script. Its header has the run command.
- **External research:** primary sources only (the OpenEnv repo at HEAD `1f9a763`, PyPI, TRL docs, arXiv). Claims I re-ran myself are marked *verified*; claims taken from a research subagent's reading are marked *reported*.
- **Not checked:** whether the Hugging Face Space is still live, and the LLM baseline's real scores (no API key was used).

## Findings

Two scores exist and they disagree, so both are shown throughout:

- **Grader score:** `GRADER_REGISTRY[task].grade(...)`, returned by `POST /grader`. This is what the README calls the task score.
- **Episode score:** `EpisodeState.episode_score`, returned as `reward.cumulative_reward` on the last step. This is the only score that includes fairness.

### Probe results (mean over 200 seeds)

| Policy | Task | Grader score | Episode score | Seeds passing threshold |
|---|---|---:|---:|---:|
| Oracle: every decision correct, zero checks | easy | 0.999 | 0.506 | 100% |
| | medium | 0.700 | 0.516 | 100% |
| | hard | 0.999 | 0.403 | 100% |
| One LeetCode check per candidate, threshold on `contest_rating` | easy | 0.997 | 0.504 | 100% |
| | medium | 0.700 | 0.516 | 100% |
| | hard | 0.998 | 0.403 | 100% |
| Decide one true reject, then idle | easy | 0.385 | **0.667** | 0% |
| | medium | 0.537 | **0.667** | 0% |
| | hard | 0.001 | **0.667** | 0% |
| Do nothing | easy | 0.308 | 0.333 | 0% |
| | medium | 0.400 | 0.333 | 0% |
| | hard | 0.001 | 0.333 | 0% |
| All hold | easy | 0.308 | 0.373 | 0% |
| | medium | 0.100 | 0.373 | 0% |
| | hard | 0.033 | 0.373 | 0% |
| All shortlist | easy | 0.462 | 0.297 | 0% |
| | medium | 0.100 | 0.373 | 0% |
| | hard | 0.433 | 0.257 | 0% |

### F1. The tasks are solved by one field

- **Evidence:** `github.stars_received` separates the labels with no overlap: reject 0–19, hold 20–200, shortlist 201–1000. `github.contributions_last_year` does the same. The other six fields probed overlap only at a tier boundary.
- **Cause:** each label is generated from its own tier with disjoint ranges (`env/profile_factory.py:381-509`). No noise or cross-tier mixing exists.
- **Consequence:** a one-check threshold policy passes all three tasks in 100% of seeds. The step budget on hard (30 steps, 20 candidates) is not a constraint, because 20 checks plus 5 shortlist decisions fit in 25 steps and undecided candidates count as hold.

### F2. The "conflicted signals" in medium do not exist

- **Evidence:** 2 of 400 "conflicted" candidates had a GitHub tier different from their LeetCode tier, both boundary ties.
- **Cause:** the grader assumes pool indices 0 and 1 are conflicted (`tasks/medium_task.py:30`), but the generator has no conflict mode.
- **Consequence:** the thoroughness bonus pays for three checks that carry no information, and a perfectly correct agent that skips them scores exactly the 0.70 threshold. Deciding "hold" for everyone (0.100) scores worse than doing nothing (0.400).

### F3. The job description does not affect the answer

- **Cause:** labels come from the tier alone. `required_skills`, `seniority` and `experience_years` are never read by the generator, and resume skills are drawn from a generic pool unrelated to the role (`env/profile_factory.py:530-537`).
- **Consequence:** an agent can ignore the job description entirely. The dimension weights matter only in one place, where they make a correct shortlist (0.35–0.50) worth no more than a correct reject (0.5) (`env/rewards.py:92`).

### F4. The episode score rewards deciding less

- **Evidence:** deciding one true reject and idling scores 0.667; a perfect agent scores 0.40–0.52.
- **Cause:** the terminal reward normalises over decided candidates only (`n = len(decisions) or 1`, `env/rewards.py:111`), against a per-candidate maximum of 1.0 that no decision can reach.
- **Consequence:** the reward signal an RL trainer would optimise points away from the task.

### F5. The fairness penalty punishes correct agents

- **Evidence:** a 100%-correct agent is penalised in 98–100% of seeds, at the maximum 0.2 in 94–98%.
- **Cause:** shortlist-rate gaps are computed across six name groups on 5–20 candidates (`env/fairness.py:132-147`). A group with one member has a rate of 0 or 1, so the gap is almost always at least 0.4.
- **Consequence:** names are assigned independently of quality, so there is no bias for the agent to avoid. The penalty is small-sample noise, and the graders ignore it entirely.

### F6. Smaller scoring defects

- **`score_dimension` is unused and farmable.** It pays +0.02 on every call, repeatably (`env/rewards.py:51-52`); 40 calls earn 0.80 in step reward with zero decisions. Recorded scores are never compared to `ground_truth_scores`, which no grader reads.
- **Step rewards never reach a final score.** `cumulative_reward` stays 0.0 until the terminal step and then holds the episode score only.
- **`/state` leaks the answers.** It returns ground-truth labels and all stats at no step cost (`api/main.py:198`), and the LLM baseline's batch mode reads it (`baseline/run_baseline.py:422`, `:469`).

### F7. The agent scripts are broken

- **`inference.py` decides blind.** Revealed stats exist only in `info["data"]`; the observation lists which platforms were checked, not their values (`env/environment.py:302-318`). The script's prompt uses the observation and an action/reward history, so the model never sees a statistic (`inference.py:129`, `:226`).
- **`inference.py` reports the wrong score.** It sums step rewards and compares the sum to the 0.75 threshold (`inference.py:231`) and never calls `/grader`.
- **The README baseline table is empty.**

### F8. Packaging and spec

| Defect | Evidence |
|---|---|
| Root `models.py` cannot be imported | UTF-16 file; `import models` raises `SyntaxError: source code string cannot contain null bytes` (*verified*) |
| Build backend does not exist | `pyproject.toml:3` names `setuptools.backends.legacy:build`; the module is absent in setuptools 84 (*verified*) |
| `client.py` cannot connect | `GenericEnvClient` connects to `/ws`; the server has no `/ws`, `/schema`, `/metadata` or `/mcp` (*reported*; server routes *verified*) |
| One global episode | `_env`, `_initial_obs` and `_baseline_cache` are module globals (`api/main.py:57-62`); two clients overwrite each other |
| Current validator fails | `openenv==0.14.0 validate --level static` → `FAIL static.manifest: openenv.yaml has no validation: block`, exit 1 (*verified*) |
| Stale manifest text | `openenv.yaml:16` says medium has "15 candidates, pick 5"; it has 5, pick 2. Manifest version 0.1.0, package 1.0.0 |
| Loose ends | No `LICENSE` despite the MIT badge, no CI, leftover `package.json` (`@openrouter/sdk`), `pytest` in runtime requirements, CORS `*` with credentials (`api/main.py:142-143`) |

## External context

### OpenEnv has moved since the repo pinned it

- **Package rename.** The repo pins `openenv-core` 0.2.3 (`uv.lock:1301`). `openenv-core` stopped at 0.3.0 (2026-05-11); the project now ships as `openenv`, latest 0.14.0 (2026-11-19) (*verified*, [PyPI](https://pypi.org/project/openenv/)). `openenv_core` is a deprecation shim (*reported*).
- **Canonical build** (*reported*, OpenEnv `docs/source/guides/first-environment.md`): subclass `Environment` with `reset(seed=None, episode_id=None, **kwargs)`, `step(action, ...)` returning an `Observation` that carries `done` and `reward`, and a `state` property. Serve it with `create_app(EnvClass, Action, Observation, max_concurrent_envs=N)`, which creates one instance per WebSocket session; set `SUPPORTS_CONCURRENT_SESSIONS = True`.
- **Manifest.** `spec_version`, `name`, `version`, `type: space`, `runtime: fastapi`, `app`, `port`, and a `validation:` block (reward range, resources, capabilities, tags). The repo's `entry_point`, `space_url` and `tasks` keys are not read by the tooling (*reported*); tasks belong in the Task API routes.
- **Validation is now three levels** (*reported*, RFC 008): static (manifest, pinning, build), runtime (episode isolation, determinism, seed control, oracle containment), and semantic (`oracle_max`, `floor_gap`, `canary_floor`, `no_solution_leakage`). F4, F6 and the `/state` leak are exactly the classes the runtime and semantic levels target.
- **What a port unlocks** (*reported*): `openenv push`, `pip install` from the Space, `AutoEnv.from_env`, the `/web` playground, `openenv collect`, and every trainer below.

### Training and evaluation paths

- **Main path** (*reported*, [TRL OpenEnv docs](https://huggingface.co/docs/trl/openenv)): `GRPOTrainer(environment_factory=Wrapper)`. The wrapper has a no-argument `__init__`, a `reset(**kwargs)` returning a string, and one typed, docstringed method per tool. TRL advises against a generic `step(action)`.
- **Server requirement:** WebSocket sessions with `max_concurrent_envs` at least the generation batch size.
- **Reward advice from TRL:** rewards must differ within a group, partial credit helps, and an SFT warm-up is needed if the base model rarely succeeds.
- **Evaluation without training:** Inspect AI via `InspectAIHarness`, and `openenv collect` for rollouts.

### Related work on LLM hiring fairness

Takeaways are *reported* from the arXiv abstracts; the design ideas are this audit's.

| Reference | Takeaway | Design idea for this repo |
|---|---|---|
| [Wilson & Caliskan 2024](https://arxiv.org/abs/2407.20371) | Resume retrieval favoured White-associated names in 85.1% of tests; intersectional harm to Black men | Score intersectional groups, not one coarse origin axis |
| [Gaebler et al. 2024](https://arxiv.org/abs/2404.03086) | Correspondence experiments (swap the name, hold the application fixed) as the audit method | Counterfactual twin candidates; score decision flips |
| [An et al. 2024](https://arxiv.org/abs/2403.15281) | ~361k resumes; 1–2 point gaps in hiring probability | Effects are small: report over many seeds with intervals, not a per-episode penalty |
| [JobFair, Wang et al. 2024](https://arxiv.org/abs/2406.15484) | Separates level bias from spread bias; rank-based impact ratio | Report the impact ratio beside the parity gap |
| [Karvonen & Marks 2025](https://arxiv.org/abs/2506.10922) | Anti-bias prompts fail under realistic context and selectivity pressure | A hard tier with distractor context ("top 10% only", culture fit) |
| [Armstrong et al. 2024](https://arxiv.org/abs/2405.04412) | Name audit with anonymous controls across occupations | Name-blind control episodes |

Metrics to draw on: equal opportunity ([Hardt et al. 2016](https://arxiv.org/abs/1610.02413)), counterfactual fairness ([Kusner et al. 2017](https://arxiv.org/abs/1703.06856)), and the four-fifths adverse-impact ratio ([29 CFR 1607.4(D)](https://www.law.cornell.edu/cfr/text/29/1607.4)). The research found no existing RL environment dedicated to hiring agents, which makes the fairness audit this project's most distinctive angle.

### What a good environment looks like

OpenEnv RFC 008 names six properties (*verified* in the RFC text, `rfcs/008-environment-auto-validation.md:35-41`): scales on infra, learnable within O(100s) of steps, secure, not prone to reward hacking, observable and self-describing, reproducible. Its reward guidance (*reported*, `guides/rewards.md`): start sparse, keep shaping small, prefer a gate to a subtracted penalty, and measure a random policy's success rate.

## Roadmap

Order matters. Phase 1 builds the measuring instrument (oracle and canary tests) that proves Phase 2 worked, so scoring comes before task realism.

Suggested batching into four PRs: **A** = Phases 0–1, **B** = Phases 2–3 (both change the generator), **C** = Phases 4–5, **D** = Phase 6. Phase 7 items are separate later work.

### Phase 0: hygiene

**Goal:** the package installs, imports and describes itself correctly, and reads as a standalone project.

- Tag the audited commit `v1.0.0` before any breaking change, so the current API and scores stay recoverable. Phases 1–2 ship as 2.0.0.
- Remove leftover scaffolding and wording from the project's origin: delete `validate_submission.sh`, and rewrite the affected lines in `.env.example`, `inference.py` and `explanations/` (00, 01, 04, 06, 08, 11, 12).
- Re-encode root `models.py` as UTF-8.
- Set `build-backend = "setuptools.build_meta"`.
- Correct the task descriptions and version in `openenv.yaml`.
- Add `LICENSE` (MIT), delete `package.json` and `package-lock.json`, move `pytest` to dev dependencies.
- Fix `inference.py`: feed `info["data"]` to the model and report the `/grader` score.

**Done when:** `uv build` succeeds, `python -c "import models"` succeeds, and `inference.py` run against a local server prints the same score as `POST /grader`.

### Phase 1: one honest score

**Goal:** a single episode score that an oracle maximises and every lazy policy floors.

- Normalise the terminal reward over all candidates, with undecided candidates scored as wrong.
- Keep one grader per task and make its score the episode's terminal reward, so `/grader` and `cumulative_reward` agree.
- Put revealed stats into `Observation.revealed_data`, so the observation is sufficient on its own.
- Remove `ground_truth_label` and `ground_truth_scores` from anything an agent can fetch during an episode. Keep a post-episode breakdown in `/grader`.
- Either score `score_dimension` against `ground_truth_scores` or delete the action.
- Promote the probe policies to tests in `tests/test_scoring_contract.py`.

**Done when**, over 200 seeds per task (the 0.99 and 0.4 margins are proposals):

- the oracle scores at least 0.99,
- do-nothing, all-hold, all-shortlist and decide-one-then-idle each score below the task's success threshold and below the oracle by at least 0.4,
- no agent-reachable endpoint returns a ground-truth field before the episode ends.

**Open decision:** whether fairness stays in the episode score during this phase or is removed until Phase 3 rebuilds it. Recommended: remove it, because F5 makes the oracle test impossible to pass otherwise.

### Phase 2: make the task real

**Goal:** information gathering changes the answer.

- Generate each candidate from latent abilities (technical, experience, growth) and emit platform signals as noisy functions of them, so tiers overlap.
- Derive the label from the job description: JD-weighted latent score plus required-skill match and seniority fit, then top-k for shortlist.
- Add an explicit conflict mode to the generator and record which candidates are conflicted in `EpisodeState`, replacing `_CONFLICTED_INDICES`.
- Rotate the role per seed so the job description varies within a task.
- Extend `scripts/probe_scoring.py` with a best-single-field policy and a check-everything policy.

**Done when**, over 200 seeds per task:

- no single-field threshold policy passes medium or hard in more than 20% of seeds,
- a policy that reads every signal scores at least 0.2 above the best single-field policy on medium,
- on hard, the check-everything policy cannot finish within budget, and a prioritising heuristic beats a fixed-order one.

**Open decisions:** the noise level and the 20% / 0.2 targets are proposals; settle them in brainstorming with a difficulty sweep.

### Phase 3: fairness as an audit

**Goal:** fairness measures something an agent can get right or wrong.

- **Counterfactual twins:** inject pairs with identical signals and names from different groups. A different decision within a pair is a flip.
- **Per-episode component:** twin consistency as a gate on the episode score. This is deterministic and defined on any pool size.
- **Aggregate report:** a `scripts/fairness_report.py` that runs many seeds and reports flip rate, four-fifths adverse-impact ratio and equal-opportunity gap per group, with confidence intervals.
- **Controls:** name-blind episodes (ids only) as the baseline.
- Replace `FairnessChecker.compute_penalty` and the README's fairness section.

**Done when:** the oracle and any name-blind policy score a 0% flip rate and no penalty, and a planted name-biased policy is flagged by all three metrics.

### Phase 4: standalone core, OpenEnv as an adapter

**Goal:** the environment is a pip-installable library that runs in-process with no server, and the same core serves concurrent sessions through OpenEnv.

- Make `env/` and `tasks/` the installable core with a plain Python API (`reset`, `step`, `state`) and no web or OpenEnv imports. The package name `env` is too generic to publish; rename it under one top-level package.
- Add an OpenEnv adapter: an `Environment` subclass served by `create_app`, one instance per session.
- Replace `openenv-core` with `openenv` as an optional extra, rewrite `openenv.yaml` with a `validation:` block, and expose tasks through the Task API.
- Replace `client.py` with a typed client.
- Drop the v1 REST routes unless a consumer has turned up (open question 1); `v1.0.0` remains available by tag.

**Done when:** a fresh virtualenv can `pip install` the package and run an episode in-process without starting a server, `openenv validate --level static` passes, `openenv validate --url <local server>` passes, and two concurrent clients run different seeds without interference.

**Before starting:** re-read the OpenEnv docs. The package released 0.14.0 two days before this audit and the *reported* details above may have changed.

### Phase 5: show learning

**Goal:** published evidence that the environment separates agents and that a model improves on it.

- Fill the README baseline table: random, rule-based, and at least two LLMs through the fixed `inference.py`, each over 50+ seeds.
- Add a tool-per-action wrapper (`reveal_github`, `reveal_leetcode`, `reveal_kaggle`, `read_resume`, `shortlist`, `hold`, `reject`) and a TRL GRPO example.
- Publish a before/after reward curve.

**Done when:** the README shows the baseline table with seed counts and a reward curve from a real run.

**Constraint:** the development server has integrated graphics only, so training runs elsewhere (Colab or HF Jobs). Evaluation against API models runs here.

### Phase 6: the public face

**Goal:** a visitor understands in one screen what the project measures and why the result is interesting.

- Rewrite the README to lead with the headline result (the baseline table and one fairness finding), then a three-line quickstart.
- Publish a results page: models ranked by task score beside flip rate and adverse-impact ratio, regenerated from committed run outputs.
- Write a short technical report: the task design, the scoring contract, the fairness method and its grounding in the audit literature, and the results.
- Ship an interactive demo where a person plays an episode and sees their own score and fairness audit.
- Release to PyPI with a changelog, and add CI running the tests and the scoring-contract checks.

**Done when:** the README, results page and report all cite numbers reproducible by one documented command, and the PyPI package installs and runs the quickstart.

**Open decisions:** the project name (keep RecruitEnv or rename), where the results page is hosted, and how many models the first results table covers.

### Phase 7: extensions

Pick after Phase 6, in any order:

- Per-action costs (a Kaggle check costs more than a resume section) in place of a flat step budget.
- A ranking task over a larger pool.
- A multi-stage pipeline: screen, then a costly interview action that reveals a low-noise signal.
- A hard tier with realistic distractor context, after Karvonen & Marks.
- Roles beyond software (the signals are currently GitHub, LeetCode and Kaggle only).

## Workflow and skills

Each phase follows the same loop. The skill names are the ones installed in this environment.

| Step | Skill | Use in this repo |
|---|---|---|
| Settle open decisions | `superpowers:brainstorming` | Required for Phases 2 and 3, whose designs are proposals here |
| Write the phase plan | `superpowers:writing-plans` | One plan per PR batch, with this document as the spec |
| Isolate | `superpowers:using-git-worktrees` | `~/projects/wt/<name>/recruit-env`, branch `ishan/<name>`, off `origin/main` |
| Implement | `superpowers:test-driven-development` | Scoring-contract tests go red first; they are the acceptance criteria above |
| Execute the plan | `superpowers:subagent-driven-development` or `superpowers:executing-plans` | Read `build-time-estimation` and state the estimate before any multi-agent run |
| Refresh the spec | `mattpocock-skills:research` | Before Phase 4, against the OpenEnv repo and docs |
| Deploy the Space | `hf-cli` | Phase 4 push and Phase 5 training jobs |
| Prove it | `superpowers:verification-before-completion` | Re-run `scripts/probe_scoring.py` and paste the table into the PR |
| Review and land | `code-review:code-review`, `superpowers:finishing-a-development-branch` | One PR per batch |

## Decisions and open questions

Decided by the owner on 2026-11-21:

- **Audience.** A standalone flagship portfolio project, so every phase through 6 is in scope.
- **Wording.** New code, docs, commits and PRs describe the project on its own terms only.

Still open:

1. **Backward compatibility.** *Settled 2026-11-24: the owner confirmed nothing consumes the v1 API.* Originally: whether anything consumes the deployed Space's current API is unknown. The roadmap assumes nothing does and protects against being wrong with the `v1.0.0` tag and a major version bump. Checking the Space's request logs before Phase 1 merges would settle it.
2. **Training budget.** Phase 5 needs GPU time off the development server. Is Colab or HF Jobs acceptable?
