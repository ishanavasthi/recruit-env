# 02 — Pydantic Models (`env/models.py`)

This is the **single source of truth for every typed shape** in the project. Everything that crosses an API boundary, every internal record, every grader input — they all live here. The file is ~340 lines and uses **Pydantic v2** exclusively.

## Why Pydantic v2?

Pydantic v2 gives you:

1. **Runtime validation** — Build a model with bad data, raise immediately with a clear error path.
2. **JSON Schema generation** — `Model.model_json_schema()` and `TypeAdapter(Action).json_schema()`. The FastAPI `/tasks` endpoint exposes the Action schema directly to clients via this.
3. **Discriminated unions** — Critical for the Action type (see below).
4. **`model_copy(deep=True)` + `model_dump()`** — Used in `environment.py` to return deep copies of state so callers can't mutate internal data.

`CLAUDE.md` enforces "Pydantic v2 only (not v1)" as a hard rule because v1's `BaseModel.copy(deep=True)` and `dict()` have different semantics — mixing them would break determinism guarantees.

## The Five Families of Models

```
1. Platform stats         GitHubStats, LeetCodeStats, KaggleStats
2. Resume + Profile       ResumeSection, CandidateProfile
3. Job spec               JobDescription
4. Actions                ReadResumeSectionAction, CheckPlatformAction,
                          ScoreDimensionAction, MakeDecisionAction
                          (+ the discriminated union `Action`)
5. Episode I/O            Observation, Reward, EpisodeState
```

---

## 1. Platform Stats

Three independent stat blocks, one per platform. Each is a flat `BaseModel` with numeric and list fields plus descriptions.

### `GitHubStats`

```python
repos:                    int
top_languages:            list[str]
commit_streak_days:       int
stars_received:           int
notable_projects:         list[str]
contributions_last_year:  int
```

### `LeetCodeStats`

```python
problems_solved:          int
easy / medium / hard:     int
contest_rating:           int          # Elo-ish
global_rank_percentile:   float (0-100 with ge/le validators)
```

### `KaggleStats`

```python
rank:                     str          # "Novice" → "Grandmaster"
competitions_entered:     int
best_finish_percentile:   float (0-100)
medals:                   dict[str, int]   # e.g. {"gold": 1, "silver": 2}
```

**Why three separate models instead of one big dict?** Because:

- Each platform has different semantics (Kaggle has *medals*, LeetCode has *contest rating*).
- Pydantic validation catches typos at construction time.
- When the env reveals platform data via `info["data"]`, it can call `candidate.github.model_dump()` and get exactly the platform-specific keys.

---

## 2. Resume + Profile

### `ResumeSection`

```python
section_name:      str       # "education" | "experience" | "skills"
content:           str       # free-text body
years_experience:  float     # ≥ 0, surfaced from this section
```

### `CandidateProfile`

The **complete record** for one synthetic candidate, including ground-truth labels:

```python
id:                    str                # "candidate_001"
name:                  str                # display name
resume_sections:       dict[str, ResumeSection]
github:                GitHubStats
leetcode:              LeetCodeStats
kaggle:                KaggleStats
ground_truth_label:    Literal["shortlist", "hold", "reject"]
ground_truth_scores:   dict[str, float]   # {"technical": 0.8, ...}
```

**The ground truth lives inside the profile.** This is deliberate. It means:

- Graders just iterate candidates and read `.ground_truth_label`. No second source of truth, no mismatch risk.
- The profile factory generates a profile knowing in advance what label it should have, then samples the stats from a distribution that matches.
- `Observation` (the agent-visible view) carefully *excludes* `candidates`, so the agent never sees these fields. The full state including ground truth is only exposed via `GET /state` — which the README calls out as "useful for debugging, not for agents".

---

## 3. `JobDescription`

```python
role:                str                       # "Backend Developer"
required_skills:     list[str]
nice_to_have:        list[str]
experience_years:    int                       # ≥ 0
seniority:           Literal["junior", "mid", "senior"]
weight_technical:    float (0-1)
weight_experience:   float (0-1)
weight_growth:       float (0-1)
```

The three `weight_*` fields are the **dimension weights** that bias scoring toward what the role values most (e.g. ML Engineer gets 0.50 technical, Frontend Dev gets 0.35 technical / 0.35 experience).

### `@model_validator(mode="after")` — the weight check

```python
@model_validator(mode="after")
def _weights_must_sum_to_one(self) -> JobDescription:
    total = self.weight_technical + self.weight_experience + self.weight_growth
    if abs(total - 1.0) > 1e-6:
        raise ValueError(...)
    return self
```

This is a Pydantic v2 **after-validator**: it runs once *all* individual field validations have passed, with `self` being the fully-constructed instance. If the weights ever drift (e.g. due to a typo in `_ROLE_META`), construction fails immediately. This guards against silent scoring bugs where the weights look like 0.99 and the reward function quietly normalizes them away.

The tolerance `1e-6` is for floating-point safety (0.45 + 0.30 + 0.25 is exact in decimal but not always in binary).

---

## 4. Actions — Discriminated Union

This is the **single most important piece of Pydantic in the project**. Read it carefully.

There are four action types, each a separate `BaseModel`:

```python
class ReadResumeSectionAction(BaseModel):
    type: Literal["read_resume_section"] = "read_resume_section"
    candidate_id: str
    section: str          # "education" | "experience" | "skills"

class CheckPlatformAction(BaseModel):
    type: Literal["check_platform"] = "check_platform"
    candidate_id: str
    platform: Literal["github", "leetcode", "kaggle"]

class ScoreDimensionAction(BaseModel):
    type: Literal["score_dimension"] = "score_dimension"
    candidate_id: str
    dimension: Literal["technical", "experience", "growth"]
    score: float (ge=0.0, le=1.0)

class MakeDecisionAction(BaseModel):
    type: Literal["make_decision"] = "make_decision"
    candidate_id: str
    decision: Literal["shortlist", "hold", "reject"]
```

They are then combined into one union:

```python
Action = Annotated[
    Union[
        ReadResumeSectionAction,
        CheckPlatformAction,
        ScoreDimensionAction,
        MakeDecisionAction,
    ],
    Field(discriminator="type"),
]
```

### What is a "discriminator"?

When you POST `{"type": "check_platform", "candidate_id": "candidate_001", "platform": "github"}`, Pydantic could *try* each subtype in the union and pick whichever validates. That's slow and ambiguous.

A **discriminator** says: "Don't guess. Look at the `type` field. Use that to dispatch directly to the matching subtype." Pydantic builds a lookup table internally; parsing is O(1).

Benefits:

- **Validation errors point at the right type.** If you POST `{"type": "check_platform", "platform": "twitter"}`, the error is `"platform: Input should be 'github', 'leetcode' or 'kaggle'"` — not a generic union failure.
- **JSON Schema** generated by `TypeAdapter(Action).json_schema()` includes `discriminator: {propertyName: "type"}` and `oneOf: [...]`. The `/tasks` endpoint returns this schema; clients (including LLMs) can read it to know the exact shape of every action.

The `Annotated[Union[...], Field(discriminator="type")]` pattern is the canonical Pydantic v2 way of declaring this.

---

## 5. Episode I/O — Observation, Reward, EpisodeState

### `Observation` — what the agent sees

```python
task_id:               str
step_number:           int
steps_remaining:       int
job_description:       JobDescription
candidates_summary:    list[dict[str, str]]      # [{"id": ..., "name": ...}]
revealed_data:         dict[str, Any]            # candidate_id → revealed bits
decisions_made:        dict[str, str]            # candidate_id → "shortlist"|...
scores_recorded:       dict[str, dict[str, float]]
done:                  bool
```

Three things to notice:

1. **No `candidates` field.** Only `candidates_summary` (just IDs + names). The full `CandidateProfile` objects with ground-truth labels are *deliberately excluded*. This is the information-asymmetry enforcement at the type level.
2. **`revealed_data` is `dict[str, Any]`.** It's a hand-built shape that grows over time — see `environment.py:_exec_read_resume` and `_exec_check_platform`. The shape per candidate is `{"resume_sections": [...], "platforms": [...]}`. The actual revealed data is returned in `step()`'s `info` dict, not stuffed back into the observation.
3. **`done`** is the same boolean as the second tuple element returned by `step()`. Redundant by design — the agent can just look at the observation.

### `Reward`

```python
step_reward:        float                   # this step only
cumulative_reward:  float                   # episode-so-far (set on terminal)
fairness_penalty:   float (0-1)
accuracy_bonus:     float (0-1)
breakdown:          dict[str, float]        # optional per-candidate + bonuses
```

`step_reward` is set every step (small, positive for novelty, 0 otherwise). The rest are zero until the **terminal step**, at which point the environment fills them in and sets `cumulative_reward` to the *final clamped episode score*. The `breakdown` dict carries every component (per-candidate accuracy, efficiency_bonus, fairness_penalty, accuracy_total).

### `EpisodeState` — the full internal state

```python
task_id, seed, step_number, max_steps:    book-keeping
job_description:                          JobDescription
candidates:                               list[CandidateProfile]   # ← includes ground truth!
candidates_summary:                       list[dict[str, str]]
revealed_data:                            dict[str, Any]
decisions_made:                           dict[str, str]
scores_recorded:                          dict[str, dict[str, float]]
is_done:                                  bool
episode_score:                            float | None             # set at terminal
```

This is what `environment.state()` returns (deep-copied). Exposed via `GET /state` for debugging. Graders consume this directly — they need the ground-truth labels and the full step count.

---

## Why Each Layer of Validation Pays Off

Here are concrete failure modes the type system catches *before* any application logic runs:

| If the agent POSTs…                                                  | Pydantic catches it because…                                  |
| -------------------------------------------------------------------- | ------------------------------------------------------------- |
| `{"type": "check_platform", "platform": "facebook"}`                 | `platform` is `Literal["github","leetcode","kaggle"]`         |
| `{"type": "score_dimension", "score": 1.5}`                          | `score: float = Field(ge=0.0, le=1.0)`                        |
| `{"type": "make_decision", "decision": "maybe"}`                     | `decision` is `Literal["shortlist","hold","reject"]`          |
| `{"type": "score_dimension"}` (missing `dimension`, `score`, `cand`) | Required fields are required                                  |
| `{"type": "wat"}`                                                    | Discriminator finds no match → clean error                    |
| A `JobDescription` with weights summing to 0.85                      | `@model_validator(mode="after")` raises                       |

This is roughly **30% less defensive code** in `environment.py` — it can trust that any `Action` it receives is well-formed.

## Interview-Level Soundbites

- *"All public surfaces are typed Pydantic v2 models. Actions are a discriminated union keyed on `type`, so JSON like `{type: 'check_platform', …}` dispatches in O(1) to the right subtype with full validation."*
- *"Ground-truth labels live inside `CandidateProfile`. The `Observation` deliberately omits the `candidates` list — only `candidates_summary` with IDs and names is exposed. Information asymmetry is enforced at the type level."*
- *"The `JobDescription` model uses an `@model_validator(mode='after')` to enforce that the three dimension weights sum to 1.0, which means a misconfigured task fails at startup, not silently during grading."*
