# 03 — Environment Core (`env/environment.py`)

The 340-line file that *is* the RL environment. It implements the three OpenEnv verbs: `reset`, `step`, `state`. Everything else in `env/` is something this file delegates to.

## Class at a Glance

```python
class RecruitmentEnvironment:
    def __init__(self) -> None:
        self._state:          EpisodeState | None
        self._factory:        ProfileFactory
        self._reward_calc:    RewardCalculator
        self._fairness:       FairnessChecker
        self._task_registry:  dict[str, TaskConfig]   # loaded lazily
        self._lock:           threading.Lock
```

Notice: a *single* environment instance holds *one* `_state` at a time. The whole project assumes one episode runs at a time. Concurrency is handled by the lock, not by sharding state.

## Lazy Task Loading — Breaking a Circular Import

```python
def _load_task_registry() -> dict:
    """Lazy import to break the env ↔ tasks circular dependency."""
    from tasks import TASK_REGISTRY
    return dict(TASK_REGISTRY)
```

`tasks/__init__.py` imports `BaseGrader` from `tasks.base`, which imports `EpisodeState` and `Observation` from `env.models`. If `env/environment.py` did `from tasks import TASK_REGISTRY` at the top, Python would hit a partial-import error on first load. The fix is to do the import inside a function so it runs *after* both modules are fully defined.

This is a common Python pattern; mention it as evidence of careful dependency hygiene.

---

## `reset(task_id, seed) -> Observation`

```python
def reset(self, task_id: str, seed: int = 42) -> Observation:
    with self._lock:
        if task_id not in self._task_registry:
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {...}")

        task = self._task_registry[task_id]

        jd = self._factory.generate_job_description(seed=seed, role_type=task.role_type)
        pool = self._factory.generate_pool(
            seed=seed, count=task.candidate_count,
            label_distribution=dict(task.label_distribution),
        )
        summary = [{"id": c.id, "name": c.name} for c in pool]

        self._state = EpisodeState(
            task_id=task_id, seed=seed, step_number=0,
            max_steps=task.max_steps,
            job_description=jd, candidates=pool, candidates_summary=summary,
            revealed_data={}, decisions_made={}, scores_recorded={},
            is_done=False, episode_score=None,
        )
        return self._build_observation()
```

**Step-by-step:**

1. **Lock acquired.** Prevents two threads from resetting mid-step.
2. **Task lookup.** Validates `task_id` (raises `ValueError` if unknown — the API converts this to a 400 response).
3. **Job description generated** from `seed` and `task.role_type` (e.g. `"backend_dev"`).
4. **Candidate pool generated** matching the task's `label_distribution` (e.g. easy = `{"shortlist": 3, "hold": 4, "reject": 3}` for 10 candidates).
5. **`candidates_summary`** built from the pool — IDs and names only.
6. **`EpisodeState` constructed** with empty `revealed_data` / `decisions_made` / `scores_recorded` and `is_done=False`.
7. **Observation built** via `_build_observation()` (which deep-copies everything so callers can't mutate the env).

### Determinism check

`reset("easy", seed=42)` called twice must produce *the same* observation. That's guaranteed because:

- `_factory.generate_job_description(seed=42, ...)` reseeds its internal RNG.
- `_factory.generate_pool(seed=42, ...)` *also* reseeds its internal RNG before drawing.

Both calls re-key on `seed`, so the sequence of random draws is byte-identical across calls. Verified by `tests/test_environment.py::TestReset::test_reset_deterministic` which asserts `obs1.model_dump() == obs2.model_dump()`.

---

## `step(action) -> tuple[Observation, Reward, bool, dict]`

The public entry point is a thin lock wrapper around `_step_locked`. All the meat is in the latter.

```python
def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
    with self._lock:
        return self._step_locked(action)
```

### `_step_locked` — the seven phases

```python
def _step_locked(self, action: Action) -> ...:
    s = self._state
    if s is None:               raise RuntimeError("No active episode. Call reset() first.")
    if s.is_done:               raise RuntimeError("Episode is already done. Call reset().")
```

**Phase 0: precondition checks.** Reject step-without-reset and step-after-done. Both raise `RuntimeError` which the API converts to a 400.

```python
    cid = action.candidate_id
    valid_ids = {c.id for c in s.candidates}
    if cid not in valid_ids:
        raise ValueError(f"Unknown candidate_id '{cid}'. Valid: {sorted(valid_ids)}")

    if isinstance(action, MakeDecisionAction) and cid in s.decisions_made:
        raise ValueError(f"Candidate '{cid}' already has a decision: '{s.decisions_made[cid]}'")
```

**Phase 1: domain validation.** Two checks:

- The candidate id must be valid for *this* episode.
- A decision can't be overwritten — once made, it's terminal for that candidate.

Note that Pydantic already validated the *shape* of the action; these checks validate *episode-relative semantics* that Pydantic cannot know about.

```python
    step_reward = self._reward_calc.calculate_step_reward(action, s)
```

**Phase 2: compute step reward BEFORE mutating state.** This is critical. The step reward depends on **novelty** — was this section read before? Was this platform checked before? If you mutate first, then `calculate_step_reward` would always think the action is a repeat. So we read the reward against the *pre-mutation* state.

```python
    info: dict[str, Any] = {}
    if isinstance(action, ReadResumeSectionAction):
        self._exec_read_resume(action, s, info)
    elif isinstance(action, CheckPlatformAction):
        self._exec_check_platform(action, s, info)
    elif isinstance(action, ScoreDimensionAction):
        self._exec_score_dimension(action, s, info)
    elif isinstance(action, MakeDecisionAction):
        self._exec_make_decision(action, s, info)
```

**Phase 3: dispatch.** Use `isinstance` to route to one of four static executor methods. They each mutate `s` in place and write to `info`. We'll cover each below.

```python
    s.step_number += 1
    all_decided = len(s.decisions_made) == len(s.candidates)
    budget_exhausted = s.step_number >= s.max_steps
    s.is_done = all_decided or budget_exhausted
    info["termination_reason"] = (
        "all_decided" if all_decided
        else "budget_exhausted" if budget_exhausted
        else None
    )
```

**Phase 4: increment step + check termination.** The episode ends if either:

- Every candidate has a decision, OR
- The step budget is hit.

`termination_reason` goes into `info` so the agent / debugger can tell *why* the episode ended.

```python
    if s.is_done:
        terminal = self._reward_calc.calculate_terminal_reward(...)
        gt_map = {c.id: c.ground_truth_label for c in s.candidates}
        correct = sum(1 for cid_, d in s.decisions_made.items() if gt_map.get(cid_) == d)
        efficiency = self._reward_calc.calculate_efficiency_bonus(...)
        penalty = self._fairness.compute_penalty(s.decisions_made, s.candidates)
        accuracy = terminal.get("total", 0.0)
        raw = accuracy + efficiency - penalty
        clamped = max(0.0, min(1.0, raw))
        s.episode_score = round(clamped, 6)

        breakdown = {k: v for k, v in terminal.items() if k != "total"}
        breakdown["accuracy_total"] = accuracy
        breakdown["efficiency_bonus"] = efficiency
        breakdown["fairness_penalty"] = penalty

        reward = Reward(
            step_reward=step_reward,
            cumulative_reward=s.episode_score,
            fairness_penalty=round(max(0.0, min(1.0, penalty)), 6),
            accuracy_bonus=round(max(0.0, min(1.0, accuracy)), 6),
            breakdown=breakdown,
        )
    else:
        reward = Reward(step_reward=step_reward)
```

**Phase 5: build the reward.**

- **If still running:** `Reward(step_reward=step_reward)` — only the small immediate signal.
- **If terminal:** compute accuracy (`calculate_terminal_reward` → normalized to [0,1]), efficiency bonus (+0.1 if all correct AND under 60% budget), fairness penalty (0.0 / 0.1 / 0.2 based on group disparity), combine `accuracy + efficiency - penalty`, clamp to [0,1], persist to `s.episode_score` rounded to 6 d.p.

Everything goes into `breakdown` so the agent / API consumer can audit each component.

```python
    obs = self._build_observation()
    return obs, reward, s.is_done, info
```

**Phase 6: return.** Build a fresh observation (deep-copied), return the four-tuple. This matches OpenAI Gym / OpenEnv conventions.

---

## Action Executors (the `_exec_*` static methods)

All four are `@staticmethod` and follow the same pattern: read the action, look up the candidate, mutate `s.revealed_data` / `s.scores_recorded` / `s.decisions_made`, write metadata to `info`.

### `_exec_read_resume`

```python
if cid not in s.revealed_data:
    s.revealed_data[cid] = {"resume_sections": [], "platforms": []}

section_name = action.section
revealed = s.revealed_data[cid]

if section_name in revealed["resume_sections"]:
    info["already_revealed"] = True
    info["data"] = {}
    return

section = candidate.resume_sections.get(section_name)
if section is None:
    info["section_found"] = False
    info["data"] = {}
    info["available_sections"] = list(candidate.resume_sections.keys())
    return

revealed["resume_sections"].append(section_name)
info["section_found"] = True
info["data"] = section.model_dump()
```

Behaviour:

- First time reading any section for a candidate → initialise `{"resume_sections": [], "platforms": []}`.
- Repeat read → return `already_revealed=True`, empty data, but the step still counts (reward calculator returned 0.0 already).
- Unknown section name → return `section_found=False` plus the available section names so the agent can self-correct.
- Successful read → push to `resume_sections` list, dump the `ResumeSection` model into `info["data"]`.

### `_exec_check_platform`

Identical pattern but for platforms. The platform → model mapping is hardcoded:

```python
if platform == "github":   info["data"] = candidate.github.model_dump()
elif platform == "leetcode": info["data"] = candidate.leetcode.model_dump()
elif platform == "kaggle":   info["data"] = candidate.kaggle.model_dump()
```

The Pydantic `Literal["github","leetcode","kaggle"]` validator already guarantees one of these three matches.

### `_exec_score_dimension`

```python
if cid not in s.scores_recorded:
    s.scores_recorded[cid] = {}
s.scores_recorded[cid][action.dimension] = action.score
info["recorded"] = True
```

Just records the agent's self-reported assessment. No validation against ground truth (the agent's score is a *belief*, not a truth claim). The grader does not currently use these scores — they're a future hook for richer evaluation.

### `_exec_make_decision`

```python
s.decisions_made[action.candidate_id] = action.decision
info["decision_recorded"] = True
```

Already protected from overwrite by phase 1 (the dup check earlier in `_step_locked`).

---

## `state() -> EpisodeState` — Deep Copy or Bust

```python
def state(self) -> EpisodeState:
    with self._lock:
        if self._state is None:
            raise ValueError("No active episode. Call reset() first.")
        return self._state.model_copy(deep=True)
```

**`deep=True` is non-negotiable.** Pydantic's default `model_copy()` is shallow — it would share nested lists/dicts. If the caller did `state.decisions_made["x"] = "hack"`, that would leak into the env's real state.

Verified by `tests/test_environment.py::TestState::test_state_returns_deep_copy`:

```python
env.reset("easy", seed=42)
s1 = env.state()
s1.decisions_made["injected"] = "hack"
s2 = env.state()
assert "injected" not in s2.decisions_made     # ← passes
```

---

## `_build_observation` — Same Defence, Shallower Object

```python
def _build_observation(self) -> Observation:
    s = self._state
    return Observation(
        task_id=s.task_id,
        step_number=s.step_number,
        steps_remaining=s.max_steps - s.step_number,
        job_description=s.job_description.model_copy(deep=True),
        candidates_summary=list(s.candidates_summary),
        revealed_data={k: _deep_copy_dict(v) for k, v in s.revealed_data.items()},
        decisions_made=dict(s.decisions_made),
        scores_recorded={k: dict(v) for k, v in s.scores_recorded.items()},
        done=s.is_done,
    )
```

Two things to notice:

1. **No `candidates` field.** Observation only carries `candidates_summary`. The full profiles never leak.
2. **All collections are freshly built.** `list(...)`, `dict(...)`, plus the recursive `_deep_copy_dict` helper for `revealed_data`. Same anti-mutation principle as `state()`.

---

## Concurrency Model

Two locks in series:

- **`api/main.py`** wraps every endpoint in an `async with _lock:` (an `asyncio.Lock`). Prevents two HTTP requests from interleaving inside the async handler.
- **`env/environment.py`** wraps every public method in `with self._lock:` (a `threading.Lock`). Defensive — if someone *bypasses* the API and calls the env directly from multiple threads (e.g. in tests), still safe.

The async lock is the one that matters in production. The threading lock is for non-HTTP callers (the rule-based baseline at `api/main.py:_run_rule_baseline` constructs a *separate* env instance per call, so it doesn't even share the singleton with the HTTP layer).

---

## Things to Mention in the Interview

- *"Step reward is computed before any state mutation — otherwise the novelty check would always see the action as a repeat."*
- *"Termination is `all_decided or budget_exhausted`. The reason is surfaced in `info` so callers can distinguish 'finished cleanly' from 'ran out of time'."*
- *"`state()` and `_build_observation` both return deep copies. That's the contract that lets callers freely mutate returned objects."*
- *"The env ↔ tasks circular import is resolved by a lazy `from tasks import TASK_REGISTRY` inside `_load_task_registry`."*
- *"Pydantic validates action *shape* before the env ever sees it; the env validates *episode-relative semantics* (unknown candidate, duplicate decision)."*
