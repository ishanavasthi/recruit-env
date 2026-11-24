# Problem

## What RecruitEnv is

A benchmark and training environment for agents that triage job candidates. An agent sees a job description and a pool of candidates. It spends a step budget revealing evidence (GitHub, LeetCode and Kaggle stats, resume sections), then labels each candidate shortlist, hold or reject against fixed quotas.

What sets it apart is an audit-grade fairness measurement. Because every candidate is synthetic, the environment can replay an episode with only the names changed and measure whether an agent's decisions move.

## Goals

- **Honest score.** The oracle maximises the score, and every lazy or one-field policy falls below the pass threshold.
- **A real task.** Gathering information changes the answer, and on the hard tier the agent has to choose what to read.
- **Fairness as measurement, not penalty.** Name-swap replays, counterfactual twins, impact ratios and equal-opportunity gaps, all with confidence intervals and a calibrated null.
- **Standalone.** A pip-installable library that runs in process. OpenEnv and HTTP are adapters around it.
- **Evidence in public.** Baseline tables across seeds and models, a training curve, a results page and a short report, each reproducible from one command.

## Scope

Roadmap Phases 0 to 6 in `docs/research/2026-11-21-findings-and-roadmap.md` are in scope. Phase 7 extensions come later.

## Constraints

- The project is presented on its own terms. Its origin as an event entry is never mentioned.
- The development server has no GPU. Model evaluation goes through an API proxy, and training runs elsewhere (Colab or HF Jobs; the owner decides, because it costs money).
- CI runs cost money, so work is batched into a few large PRs.
- Results must be reproducible from committed seeds and commands.
