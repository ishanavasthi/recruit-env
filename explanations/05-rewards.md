# 05 — Reward System (`env/rewards.py`)

The reward function is what teaches the agent *what to do*. It has **four components**, each shaped to solve a specific learning problem.

```
final reward  =  clamp(  accuracy_total  +  efficiency_bonus  -  fairness_penalty,   0,  1 )
                          ▲                  ▲                       ▲
                  episode-end scoring    +0.1 if perfect &        0.0 / 0.1 / 0.2
                  of all decisions        used < 60% budget       (see explanations/06)
```

Plus, every step also returns a **dense step reward** (small, immediate, novelty-based) that pushes the agent to gather information rather than do nothing.

This file implements the first three; the fairness penalty lives in `env/fairness.py`.

## `RewardCalculator` — All Static, No State

```python
class RewardCalculator:
    @staticmethod
    def calculate_step_reward(action: Action, state: EpisodeState) -> float: ...

    @staticmethod
    def calculate_terminal_reward(decisions, candidates, jd) -> dict[str, float]: ...

    @staticmethod
    def calculate_efficiency_bonus(steps_used, max_steps, correct, total) -> float: ...
```

All three are static — they take inputs and return outputs, no instance state. This is deliberate: rewards must be **pure functions** so they're testable and deterministic.

## Component 1 — Step Reward (Dense Signal)

```python
@staticmethod
def calculate_step_reward(action: Action, state: EpisodeState) -> float:
    if isinstance(action, ReadResumeSectionAction):
        if action.section in state.revealed_data.get(cid, {}).get("resume_sections", []):
            return 0.0
        return 0.02

    if isinstance(action, CheckPlatformAction):
        if action.platform in state.revealed_data.get(cid, {}).get("platforms", []):
            return 0.0
        return 0.03

    if isinstance(action, ScoreDimensionAction):
        return 0.02

    if isinstance(action, MakeDecisionAction):
        return 0.0

    return 0.0
```

The numeric values:

| Action                | New                                | Repeat |
| --------------------- | ---------------------------------- | ------ |
| `read_resume_section` | **+0.02**                          | 0.0    |
| `check_platform`      | **+0.03** (slightly more valuable) | 0.0    |
| `score_dimension`     | **+0.02** always                   | n/a    |
| `make_decision`       | 0.0 (scored at terminal)           | n/a    |

**Reward shaping logic:**

- Platform checks are weighted slightly higher than resume reads (0.03 vs 0.02) because they carry more dense quantitative information (repos, problems_solved, stars, etc.).
- *Repeat* reads earn 0.0 — the agent has no incentive to spam the same action. This prevents the easy hack of "just keep checking GitHub on candidate_001 for free reward".
- `score_dimension` always earns 0.02 regardless of previous scores (you can re-score and overwrite). This is a cheap way to encourage the agent to externalise its reasoning.
- `make_decision` earns 0.0 — decisions are reflected in the *terminal* reward, not in step rewards.

### Why "before mutation"?

`environment.py` calls `calculate_step_reward(action, s)` **before** it executes the action. Because the function checks "has this section/platform been revealed before?" against the current state, computing it *after* mutation would always return 0.0. (The env file documents this with the comment `# 1. Compute step reward BEFORE mutating state (novelty check)`.)

## Component 2 — Terminal Reward (Sparse, Accuracy-Based)

This runs once at episode end. It scores every decision against ground truth, with **role-aware weighting**.

```python
@staticmethod
def calculate_terminal_reward(decisions, candidates, jd) -> dict[str, float]:
    gt_map = {c.id: c.ground_truth_label for c in candidates}
    breakdown: dict[str, float] = {}
    raw_total = 0.0

    for cid, decision in decisions.items():
        truth = gt_map.get(cid)
        if truth is None: continue

        if decision == truth:
            if   decision == "shortlist": score = 1.0 * jd.weight_technical
            elif decision == "reject":    score = 0.5
            else:                          score = 0.3                       # hold
        else:
            if   decision == "shortlist" and truth == "reject":    score = -0.5
            elif decision == "reject"    and truth == "shortlist": score = -0.3
            else:                                                   score = -0.1

        breakdown[cid] = score
        raw_total += score
    ...
```

### The scoring table

|                              | Score                              | Why                                                    |
| ---------------------------- | ---------------------------------- | ------------------------------------------------------ |
| Correct **shortlist**        | **+1.0 × `jd.weight_technical`**   | Highest reward — finding a hire is the whole point     |
| Correct **reject**           | **+0.5**                           | Filtering noise is valuable but cheaper than hiring    |
| Correct **hold**             | **+0.3**                           | Hedging is fine but not heroic                         |
| Shortlisted a reject         | **−0.5**                           | Worst error — wastes recruiter time on a bad hire     |
| Rejected a shortlist         | **−0.3**                           | Bad — but losing one good candidate hurts less than wasting a slot |
| Other hold error             | **−0.1**                           | Mild penalty                                           |

**Two design choices stand out:**

1. **Correct-shortlist reward scales with `weight_technical`.** A backend dev's weight_technical is 0.45, ML engineer's is 0.50. So nailing a shortlist for an ML role is worth slightly more raw reward than nailing one for a frontend role. Encourages the agent to read the JD.
2. **Asymmetric error penalties.** Shortlist-a-reject (−0.5) is *worse* than reject-a-shortlist (−0.3). The reasoning is real-world: false positives waste interviewer time and damage the brand; false negatives "just" lose one candidate from a pool of many.

### Normalisation to [0, 1]

```python
n = len(decisions) or 1
max_raw = 1.0 * n
min_raw = -0.5 * n
span = max_raw - min_raw
normalised = (raw_total - min_raw) / span if span > 0 else 0.0
normalised = max(0.0, min(1.0, normalised))
breakdown["total"] = round(normalised, 6)
```

The raw score lives in `[-0.5n, 1.0n]` (worst case: every decision is a shortlist-a-reject; best case: every decision is a correct shortlist with weight_technical=1.0). Linear-rescale to `[0, 1]`, clamp, round to 6 d.p., write into `breakdown["total"]`.

The `breakdown` dict carries per-candidate scores too, so downstream consumers can see exactly what each decision contributed.

## Component 3 — Efficiency Bonus

```python
@staticmethod
def calculate_efficiency_bonus(steps_used, max_steps, correct, total) -> float:
    if total == 0:
        return 0.0
    all_correct = correct == total
    under_budget = steps_used < 0.6 * max_steps
    if all_correct and under_budget:
        return 0.1
    return 0.0
```

**The all-or-nothing +0.1.**

You only get the bonus if:
- **Every** decision is correct (no partial credit), AND
- You used **less than 60% of the budget**.

For the easy task (40 steps), that's "under 24 steps, 100% accurate". This encourages the agent to *commit early* once it has enough information rather than burn the full budget on unnecessary checks.

It's deliberately a small bonus (0.1) so the agent doesn't sacrifice accuracy to chase it.

## Component 4 — Fairness Penalty

Lives in `env/fairness.py` — see `explanations/06`. Recap:

| Max shortlist-rate diff across origin groups | Penalty |
| -------------------------------------------- | ------- |
| < 0.2                                        | 0.0     |
| [0.2, 0.4)                                   | 0.1     |
| ≥ 0.4                                        | 0.2     |

Subtracted from the final score.

## How `environment.py` Combines Everything (At Terminal Step)

```python
terminal   = self._reward_calc.calculate_terminal_reward(decisions, candidates, jd)
efficiency = self._reward_calc.calculate_efficiency_bonus(step_number, max_steps, correct, total)
penalty    = self._fairness.compute_penalty(decisions, candidates)
accuracy   = terminal.get("total", 0.0)
raw        = accuracy + efficiency - penalty
clamped    = max(0.0, min(1.0, raw))
s.episode_score = round(clamped, 6)
```

The `Reward` returned at terminal carries:

- `step_reward` — the small reward earned by *this last step*
- `cumulative_reward` = the **final clamped episode score** (this is what the agent learns from)
- `accuracy_bonus` = the rounded accuracy component
- `fairness_penalty` = the rounded penalty
- `breakdown` = per-candidate + per-component dict

## Module-Level `compute_reward(state)` — Backward Compat

```python
def compute_reward(state: EpisodeState) -> Reward:
    """Compute the full Reward object for the current episode state."""
```

This is a convenience entry point that does the same combination logic outside the env. Used by `env/__init__.py` to expose a top-level reward function. The real implementation in `environment.py:_step_locked` re-derives the same thing inline; both paths should produce identical results.

## Why Four Components, Not One?

Single-signal reward functions have classic failure modes:

| If the reward were…                                            | The agent learns to…                                          |
| -------------------------------------------------------------- | ------------------------------------------------------------- |
| only terminal accuracy                                         | wait until the last step (sparse credit assignment problem)   |
| only dense step reward                                         | never decide — just keep checking platforms (reward hacking)  |
| accuracy + efficiency only                                     | be biased — fast & accurate but only on a single demographic  |
| accuracy + fairness only                                       | be slow — gather every signal before deciding                 |

The four-component blend is what makes the env teach a *full skill*: explore efficiently, decide on time, decide accurately, decide without bias.

## Reward Trajectory in One Sentence

> Throughout an episode the agent earns small positive step rewards for *gathering new information* and zero for repeats or decisions; at episode end it earns a large accuracy-weighted reward, possibly a small efficiency bonus if it was both perfect and fast, minus a fairness penalty if its shortlist rate is skewed across origin groups — all clamped into `[0, 1]`.

## Soundbites

- *"Step rewards are dense and novelty-based — +0.02 for new resume sections, +0.03 for new platform checks, 0.0 for repeats. That stops reward hacking."*
- *"Correct-shortlist reward is multiplied by `jd.weight_technical` — so a perfect shortlist on an ML role (0.50) is worth more than on a frontend role (0.35). Forces the agent to read the JD."*
- *"Shortlist-a-reject costs 0.5; reject-a-shortlist costs 0.3. Asymmetric penalty because false positives are more expensive in real recruiting."*
- *"Efficiency bonus is all-or-nothing: +0.1 only if every decision is correct AND you used under 60% of the budget. Small enough that the agent won't sacrifice accuracy chasing it."*
- *"Final reward is `clamp(accuracy + efficiency − fairness_penalty, 0, 1)`. Four components, each shaped to solve a specific learning failure mode."*
