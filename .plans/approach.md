# Approach

The full reasoning is in the roadmap (`docs/research/2026-11-21-findings-and-roadmap.md`) and in the per-PR design specs under `docs/superpowers/specs/`. This page keeps the decisions that shape everything else.

## Key decisions

- **Measure the instrument first, then change the task.** Phase 1 built the oracle and lazy-policy contract tests (`tests/test_scoring_contract.py`). Phase 2 then changed the generator, and those tests proved each change. A difficulty change counts only when the contract and acceptance tests show it.
- **Labels come from a published rubric** (`env/rubric.py`), not hidden weights. An agent that reads everything and applies the rubric scores about 1.0, so failures measure how evidence was gathered and weighed, not luck. The technical score is a weighted power mean (exponent −2): weak platforms pull it down, while it still rises whenever any sub-score rises. A mean-minus-spread form was rejected because it was not monotone.
- **Latent abilities with noise.** Visible fields are noisy functions of latent abilities, so no single source decides the label. The noise level and quotas were tuned by sweep against stated targets (best single source ≤ 20% pass).
- **Fairness lives outside the score.** The only fairness item in the score is one deterministic gate: identical twins decided differently on the same evidence cap the score. Everything else is an aggregate audit (`env/fairness.py`). Its randomisation null redraws names with the generator itself, because a group-shuffling null flagged honest agents.
- **Verifier checks, not one number.** Graders return named, weighted checks and gates, so a score can be explained.
- **Independent random streams.** Pool, twin, cosmetic and name streams are tagged and keyed by pool size, so a name swap changes only names and tasks sharing a seed do not share candidates.
- **Owner delegate when the owner is away.** In unattended builds a Fable reviewer answers design questions and reviews the branch. Each spec records its decisions as the delegate's.

## Alternatives considered

- Keeping a fairness penalty in the episode score. Rejected, because it punished correct agents (roadmap finding F5).
- An efficiency bonus on hard. Rejected, because it paid for unused steps and so rewarded gathering less.
- A shuffle-based fairness null. Rejected, because twins share an origin more often than chance, which inflated false flags.
