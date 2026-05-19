# 07 — Tasks & Graders (`tasks/`)

A task is **a complete scenario configuration** (how many candidates, what label split, what step budget) plus **a grader** (how to score the agent's decisions on that scenario). Three tasks, three graders, all in `tasks/`.

## File Layout

```
tasks/
├── __init__.py         ← builds TASK_REGISTRY + GRADER_REGISTRY dicts
├── base.py             ← TaskConfig dataclass + BaseGrader ABC
├── easy_task.py        ← EASY_CONFIG + EasyGrader
├── medium_task.py      ← MEDIUM_CONFIG + MediumGrader
└── hard_task.py        ── HARD_CONFIG + HardGrader
```

## `base.py` — Shared Foundations

### `TaskConfig` (frozen dataclass)

```python
@dataclass(frozen=True)
class TaskConfig:
    id:                  str                           # "easy" | "medium" | "hard"
    name:                str                           # human-readable
    description:         str
    difficulty:          Literal["easy", "medium", "hard"]
    max_steps:           int                           # step budget
    candidate_count:     int                           # pool size
    label_distribution:  dict[str, int]                # e.g. {"shortlist":3,"hold":4,"reject":3}
    success_threshold:   float = 0.70                  # pass/fail line for the breakdown
    role_type:           str   = "backend_dev"         # picks JD template
```

`frozen=True` makes the dataclass immutable — you can't mutate a task config at runtime, so the env can trust it.

### `BaseGrader` (ABC)

```python
class BaseGrader(ABC):
    @abstractmethod
    def grade(self, initial_obs: Observation, final_state: EpisodeState) -> float:
        """Return a score in [0.0, 1.0] for the completed episode.
        Must be **deterministic**: same inputs always produce same output."""
```

Every grader subclass takes the same two inputs (initial observation + final state) and returns a float in `[0, 1]`. This is the contract `POST /grader` enforces.

## `__init__.py` — The Registries

```python
TASK_REGISTRY:   dict[str, TaskConfig]  = {"easy": EASY_CONFIG,   "medium": MEDIUM_CONFIG,   "hard": HARD_CONFIG}
GRADER_REGISTRY: dict[str, BaseGrader] = {"easy": EasyGrader(),  "medium": MediumGrader(),  "hard": HardGrader()}
```

Two dicts keyed by task id. The environment reads `TASK_REGISTRY` at reset time; the API reads `GRADER_REGISTRY` at grade time. Adding a fourth task = define a `TaskConfig`, define a `BaseGrader` subclass, register both.

---

## Task 1 — Easy: "Screening Sprint"

### Config

```python
EASY_CONFIG = TaskConfig(
    id="easy",
    name="Screening Sprint",
    description="Screen 10 candidates and shortlist the top 3.",
    difficulty="easy",
    max_steps=40,
    candidate_count=10,
    label_distribution={"shortlist": 3, "hold": 4, "reject": 3},
    success_threshold=0.75,
    role_type="backend_dev",
)
```

- **10 candidates / 40 steps = 4 steps per candidate.** Generous.
- Label split is balanced (3 / 4 / 3).
- Pass = ≥ 0.75.

### `EasyGrader` — Weighted Accuracy + Perfect Bonus

```python
class EasyGrader(BaseGrader):
    def grade(self, initial_obs, final_state) -> float:
        weighted_correct = 0.0
        weighted_total = 0.0
        for c in candidates:
            decision = decisions.get(c.id, "hold")    # undecided → hold
            truth = gt[c.id]
            weight = 2.0 if truth == "shortlist" else 1.0
            weighted_total += weight
            if decision == truth:
                weighted_correct += weight

        base_score = weighted_correct / weighted_total

        gt_shortlist_ids    = {cid for cid, label in gt.items() if label == "shortlist"}
        agent_shortlist_ids = {cid for cid, d in decisions.items() if d == "shortlist"}
        bonus = 0.2 if agent_shortlist_ids == gt_shortlist_ids else 0.0

        raw = base_score + bonus
        return max(0.001, min(0.999, raw))
```

**The rubric:**

1. Iterate every candidate. Map undecided candidates to `"hold"` (so leaving someone unrated isn't a free pass).
2. Compute weighted accuracy: shortlist truths count *twice* in the denominator + (if correct) twice in the numerator. So getting a shortlist right is worth 2× a hold/reject.
3. If the agent's shortlist set is **exactly** the ground-truth shortlist set (no false positives, no misses): +0.2 bonus.
4. Clamp to `(0.001, 0.999)` — leaves epsilon room so an exact 0.0 or 1.0 can't occur (avoids edge-case issues in downstream scoreboards that treat 0.0 as "no score recorded").

**Why weight shortlist 2×?** Because in real recruiting, the cost of a missed shortlist (losing a great hire) and the cost of a false-positive shortlist (wasted interview slot) are both higher than the cost of a misclassified hold. The 2× weight makes the score reflect that.

## Task 2 — Medium: "Conflicted Signals"

### Config

```python
MEDIUM_CONFIG = TaskConfig(
    id="medium",
    name="Conflicted Signals",
    description="Evaluate 5 candidates with conflicting signals across platforms. Shortlist the best 2.",
    difficulty="medium",
    max_steps=25,
    candidate_count=5,
    label_distribution={"shortlist": 2, "hold": 2, "reject": 1},
    success_threshold=0.70,
    role_type="ml_engineer",
)
```

- **5 candidates / 25 steps = 5 steps per candidate.** Even more generous *per-candidate* than easy.
- Smaller pool, different role (ML engineer this time → 0.50 technical weight).
- Pass = ≥ 0.70.

### The "Conflicted" Candidates

```python
_CONFLICTED_INDICES = {0, 1}
_ALL_PLATFORMS = {"github", "leetcode", "kaggle"}
```

The first two candidates in the pool are designated *conflicted*. The README explains: they have deliberately mixed signals (e.g. great GitHub but bad LeetCode). Note that *this is just a convention* — the profile factory doesn't currently inject genuinely conflicting signals; the grader simply expects that conflict-resolution is the agent's job on those two.

### `MediumGrader` — Accuracy + Thoroughness Adjustment

```python
class MediumGrader(BaseGrader):
    def grade(self, initial_obs, final_state) -> float:
        correct = sum(1 for c in candidates if decisions.get(c.id, "hold") == gt[c.id])
        base_score = correct / total

        conflicted_ids = {candidates[i].id for i in _CONFLICTED_INDICES if i < len(candidates)}

        adjustment = 0.0
        for cid in conflicted_ids:
            platforms_checked = set(revealed.get(cid, {}).get("platforms", []))
            has_decision = cid in decisions

            if has_decision and platforms_checked >= _ALL_PLATFORMS:
                adjustment += 0.15      # thoroughness bonus
            elif has_decision and len(platforms_checked) < 2:
                adjustment -= 0.15      # hastiness penalty

        raw = base_score + adjustment
        return max(0.001, min(0.999, raw))
```

**The rubric:**

1. Base accuracy = correct / total (no weighting here, unlike easy).
2. For each of the two conflicted candidates:
    - If the agent **checked all 3 platforms** and made a decision: +0.15 (thoroughness bonus).
    - If the agent decided with **fewer than 2 platforms** checked: −0.15 (hastiness penalty).
    - Otherwise (2 platforms + decision, or no decision): no adjustment.
3. Maximum bonus is 2 × 0.15 = 0.30. Maximum penalty is 2 × 0.15 = −0.30.

**Why this design?** Medium tests *information-gathering discipline*. On clean signal candidates the agent should decide fast; on the two conflicted candidates it should slow down and look at everything. The grader rewards exactly that behaviour.

## Task 3 — Hard: "Batch Triage"

### Config

```python
HARD_CONFIG = TaskConfig(
    id="hard",
    name="Batch Triage",
    description="Batch triage 20 candidates with a tight step budget. Shortlist the best 5.",
    difficulty="hard",
    max_steps=30,
    candidate_count=20,
    label_distribution={"shortlist": 5, "hold": 8, "reject": 7},
    success_threshold=0.60,
    role_type="data_scientist",
)
```

- **20 candidates / 30 steps = 1.5 steps per candidate.** Extremely tight. The agent can't possibly check every signal — it has to decide *somehow* with very little information.
- Pass = ≥ 0.60 (lower threshold reflects the difficulty).

### `HardGrader` — Shortlist F1 + Efficiency Bonus

```python
class HardGrader(BaseGrader):
    def grade(self, initial_obs, final_state) -> float:
        # Undecided candidates → "hold"
        effective = {c.id: decisions.get(c.id, "hold") for c in candidates}

        gt_shortlist    = {cid for cid, label in gt.items() if label == "shortlist"}
        agent_shortlist = {cid for cid, d in effective.items() if d == "shortlist"}

        tp = len(gt_shortlist & agent_shortlist)
        precision = tp / len(agent_shortlist) if agent_shortlist else 0.0
        recall    = tp / len(gt_shortlist)    if gt_shortlist    else 0.0

        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        steps_saved      = max(0, max_steps - steps_used)
        efficiency_bonus = (steps_saved / max_steps) * 0.1 if max_steps > 0 else 0.0

        raw = f1 + efficiency_bonus
        return max(0.001, min(0.999, raw))
```

**The rubric:**

1. **F1 score** of the shortlist set:
    - precision = how many of the agent's shortlists were correct shortlists?
    - recall = how many of the ground-truth shortlists did the agent catch?
    - F1 = harmonic mean of the two
2. **Efficiency bonus** = `steps_saved / max_steps × 0.1`. Continuous (unlike the env-level efficiency bonus which is binary). So saving 12 of 30 steps gives +0.04.

**Why F1?** Because at 20 candidates with only 5 shortlists, plain accuracy is misleading — an agent that rejects everyone gets 75% accuracy (15/20 correct) but is useless. F1 explicitly rewards *finding* the shortlists.

### Worked Examples (from `tests/test_graders.py`)

- **Shortlist-everyone** → precision = 5/20 = 0.25, recall = 1.0, F1 ≈ 0.40. Plus efficiency bonus → ~0.4.
- **Hold-everyone** → no shortlists → F1 = 0.0. Score < 0.15.
- **Perfect shortlists** → precision = 1.0, recall = 1.0, F1 = 1.0. Score > 0.95.

These exact assertions are encoded in `TestHardGrader::test_hard_grader_uses_f1`.

## A Common Grader Idiom — `clamp(0.001, 0.999)`

All three graders end with `max(0.001, min(0.999, raw))`. Why the epsilon margin? Two reasons:

1. **Sentinel collision.** Many systems treat 0.0 and 1.0 as "no value" or "max-perfect-skip-eval". The epsilon keeps actual grades distinguishable from sentinels.
2. **Numerical robustness.** Avoids `(1.0 == 1.0)` flakiness in tests that use exact equality.

It's a small, deliberate stylistic choice.

## Where the Graders Get Called

`api/main.py:grade()`:

```python
state = _env.state()
if not state.is_done:
    raise HTTPException(400, "Episode is not complete...")

task_id = state.task_id
grader = GRADER_REGISTRY.get(task_id)
initial_obs = _initial_obs.get(task_id)   # cached at reset time
score = grader.grade(initial_obs, state)
```

The flow:

1. Reset caches the `Observation` returned: `_initial_obs[task_id] = obs`.
2. Episode plays out via repeated `/step` calls.
3. `POST /grader` looks up both the grader and the cached initial obs, hands them to `grader.grade(...)`, and returns the result + a breakdown including the success threshold.

Why cache the initial obs? Because the grader contract is `grade(initial_obs, final_state)` — but the env doesn't preserve the initial observation across steps. So the API layer keeps a copy.

## Grader vs Reward — Two Distinct Concepts

| Aspect                     | In-episode Reward (`env/rewards.py`)             | Grader (`tasks/*_task.py`)                       |
| -------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| Who consumes it?           | The agent (during training / inference)         | The judging system / leaderboard                 |
| When is it computed?       | Every step, plus a terminal aggregate           | Once, at the end, on demand via `POST /grader`   |
| What signal does it carry? | Step rewards + accuracy + efficiency − fairness | Task-specific rubric (weighted acc / thoroughness / F1) |
| Is it task-specific?       | No — same logic across all tasks                | Yes — different per task                         |

They are deliberately decoupled so the in-episode reward can be a smooth training signal while the grader can be the *official* score the leaderboard publishes.

## Soundbites

- *"Three tasks: 4 / 5 / 1.5 steps per candidate. Difficulty progresses by tightening the budget, not by making individual decisions harder."*
- *"Easy uses weighted accuracy (shortlists 2×) plus a perfect-shortlist bonus. Medium adds thoroughness — +0.15 for 3 platforms checked, −0.15 for <2 — on two designated 'conflicted' candidates. Hard switches to F1 because plain accuracy rewards rejecting everyone."*
- *"Graders and in-episode rewards are deliberately separate concepts. The reward is a *training* signal; the grader is the *scoreboard*."*
- *"Adding a fourth task is two lines in `tasks/__init__.py`: register the `TaskConfig` and the grader."*
