# 04 — Profile Factory (`env/profile_factory.py`)

This file generates **synthetic candidate profiles and job descriptions** entirely from a seed. It contains zero live network calls — `CLAUDE.md` explicitly bans any HTTP requests in `env/`. Everything is hardcoded pools + seeded `numpy.random` draws.

The challenge: build profiles that *look real enough* for an agent to learn meaningful heuristics, while guaranteeing **byte-exact determinism** from a seed.

## The Hardcoded Pools (Top of File)

The file opens with several constant lists:

| Constant            | Contents                                                        | Used for                                |
| ------------------- | --------------------------------------------------------------- | --------------------------------------- |
| `_NAMES`            | 50 diverse full names (Aisha Patel, Carlos Rivera, Wei Zhang …) | Drawn without replacement per pool      |
| `_LANGUAGES_BY_ROLE`| Per-role language pools (ML: Python/C++/CUDA; FE: TS/CSS …)     | Top-language sampling on GitHubStats    |
| `_REQUIRED_SKILLS`  | Per-role required skill list                                    | `JobDescription.required_skills`        |
| `_NICE_TO_HAVE`     | Per-role bonus-skill list                                       | `JobDescription.nice_to_have` (sampled) |
| `_NOTABLE_PROJECTS` | 15 fake project names                                           | GitHubStats.notable_projects            |
| `_UNIVERSITIES`     | 10 schools (MIT, Stanford, IIT Bombay …)                        | Resume education line                   |
| `_DEGREES`          | 8 degree strings                                                | Resume education line                   |
| `_ROLE_META`        | Per-role-type seniority + experience-years + weights            | `JobDescription` construction           |

**Why diverse names?** Two reasons:
1. Realism (the simulated pipeline is global).
2. The fairness checker (`env/fairness.py`) maps first names to broad origin groups; a diverse pool is what makes demographic-parity testing meaningful.

`_ROLE_META` is worth flagging: the three weights for each role *must* sum to 1.0, and `JobDescription`'s `@model_validator` will reject anything that doesn't. So you can read off the four supported roles + their weights from the dict:

```python
"ml_engineer":    {0.50 tech, 0.25 exp, 0.25 growth, senior, 4 yrs}
"backend_dev":    {0.45 tech, 0.30 exp, 0.25 growth, mid,    3 yrs}
"data_scientist": {0.40 tech, 0.30 exp, 0.30 growth, mid,    2 yrs}
"frontend_dev":   {0.35 tech, 0.35 exp, 0.30 growth, mid,    2 yrs}
```

## The Class

```python
class ProfileFactory:
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng: Generator = np.random.default_rng(seed)
```

Stores its own RNG. Crucially, `generate_pool` and `generate_job_description` both **reseed** at the start of every call:

```python
self._rng = np.random.default_rng(seed)
```

This is the magic line. It means an outside caller doesn't have to worry about the factory's internal state being polluted by a previous call. `generate_job_description(seed=42, ...)` and `generate_pool(seed=42, ...)` are *independent* — they each produce the same output as long as the caller passes the same seed, regardless of what was called before.

## `generate_pool(seed, count, label_distribution) -> list[CandidateProfile]`

The pool generator. Four steps:

```python
dist_total = sum(label_distribution.values())
if dist_total != count:
    raise ValueError(f"label_distribution sums to {dist_total}, expected {count}")
```

**Step 1: validate inputs.** The label distribution (e.g. `{"shortlist": 3, "hold": 4, "reject": 3}`) must sum to `count`. Catches misconfigured tasks early.

```python
self._rng = np.random.default_rng(seed)
name_indices = self._rng.choice(len(_NAMES), size=count, replace=False)
names = [_NAMES[i] for i in name_indices]
```

**Step 2: pick names without replacement.** No duplicate names in one pool. Deterministic because `_rng.choice` with a fixed seed is deterministic.

```python
labels: list[Literal[...]] = []
for label, n in label_distribution.items():
    labels.extend([label] * n)
self._rng.shuffle(labels)
```

**Step 3: build the label list, then shuffle.** Without shuffling, the first three candidates would always be "shortlist", etc. The shuffle is also seeded — same seed → same shuffle.

```python
profiles: list[CandidateProfile] = []
for idx, (name, label) in enumerate(zip(names, labels)):
    cid = f"candidate_{idx + 1:03d}"
    profile = self._build_profile(cid, name, label)
    profiles.append(profile)
return profiles
```

**Step 4: build each profile** with id `candidate_001`, `candidate_002`, … and dispatch to one of three label-specific builders.

## `generate_job_description(seed, role_type)`

```python
self._rng = np.random.default_rng(seed)
meta = _ROLE_META[role_type]
required = list(_REQUIRED_SKILLS[role_type])
nice = list(_NICE_TO_HAVE[role_type])
self._rng.shuffle(nice)
nice = nice[: self._rng.integers(2, len(nice) + 1)]
return JobDescription(role=meta["role"], required_skills=required, nice_to_have=list(nice), ...)
```

**The interesting part:** `nice_to_have` is *sampled*. The full nice-to-have list is shuffled, then the first 2 to `len(nice)+1` entries are kept. So different seeds surface different nice-to-haves on the same role type — the agent can't just memorise "for backend_dev the bonus skills are X".

`required_skills` is *not* sampled — every backend_dev job posting requires the same five skills. That's a deliberate design choice (the *required* skills are part of the role's identity; the nice-to-haves are the variable surface).

## The Three Builders — `_build_shortlist`, `_build_hold`, `_build_reject`

All three have the same skeleton:

```python
def _build_shortlist(self, cid: str, name: str) -> CandidateProfile:
    rng = self._rng
    leetcode = self._senior_leetcode(rng)
    github   = self._senior_github(rng)
    kaggle   = self._senior_kaggle(rng)
    years    = float(rng.uniform(5.0, 10.0))
    resume   = self._build_resume(rng, years, skill_coverage=1.0)
    gt_scores = self._compute_ground_truth_scores(leetcode, github, kaggle, years, resume)
    return CandidateProfile(id=cid, name=name, resume_sections=resume,
                            github=github, leetcode=leetcode, kaggle=kaggle,
                            ground_truth_label="shortlist", ground_truth_scores=gt_scores)
```

The differences across tiers are in:

|              | Shortlist (senior)        | Hold (mid)               | Reject (junior)       |
| ------------ | ------------------------- | ------------------------ | --------------------- |
| LeetCode     | `_senior_leetcode`        | `_mid_leetcode`          | `_junior_leetcode`    |
| GitHub       | `_senior_github`          | `_mid_github`            | `_junior_github`      |
| Kaggle       | `_senior_kaggle`          | `_mid_kaggle`            | `_junior_kaggle`      |
| Years exp.   | `uniform(5.0, 10.0)`      | `uniform(2.0, 5.0)`      | `uniform(0.0, 2.0)`   |
| Skill cover. | `1.0` (all 15 generic)    | `0.7` (~10 of 15)        | `0.3` (~4 of 15)      |

## Tier-specific Stat Distributions

The full distribution choices, drawn from the file:

### LeetCode

| Field                  | Senior              | Mid                   | Junior                |
| ---------------------- | ------------------- | --------------------- | --------------------- |
| `hard`                 | `[20, 60)`          | `[5, 21)`             | `[0, 5)`              |
| `medium`               | `[80, 180)`         | `[40, 120)`           | `[5, 40)`             |
| `easy`                 | `[60, 160)`         | `[30, 110)`           | `[10, 60)`            |
| `problems_solved`      | sum (~160–400)      | sum (~75–250)         | sum (~15–105)         |
| `contest_rating`       | `[1600, 2001)`      | `[1200, 1601)`        | `[800, 1200)`         |
| `global_rank_pctile`   | `uniform(85, 99.5)` | `uniform(50, 85)`     | `uniform(5, 50)`      |

### GitHub

| Field                | Senior                          | Mid                                 | Junior                       |
| -------------------- | ------------------------------- | ----------------------------------- | ---------------------------- |
| `repos`              | `[30, 61)`                      | `[10, 31)`                          | `[1, 10)`                    |
| `top_languages`      | 4 from senior pool              | 3 from mid pool                     | 2 from junior pool           |
| `commit_streak_days` | `[60, 181)`                     | `[10, 61)`                          | `[0, 10)`                    |
| `stars_received`     | `[200, 1001)`                   | `[20, 201)`                         | `[0, 20)`                    |
| `notable_projects`   | 2–4 sampled                     | 1–2 sampled                         | none                         |
| `contribs_last_year` | `[500, 1500)`                   | `[100, 500)`                        | `[10, 100)`                  |

### Kaggle

| Field                    | Senior                                   | Mid                                  | Junior                              |
| ------------------------ | ---------------------------------------- | ------------------------------------ | ----------------------------------- |
| `rank`                   | Expert / Master / Grandmaster            | Contributor / Expert                 | Novice / Contributor                |
| `competitions_entered`   | `[10, 40)`                               | `[3, 12)`                            | `[0, 3)`                            |
| `best_finish_percentile` | `uniform(85, 99)`                        | `uniform(40, 80)`                    | `uniform(0, 40)`                    |
| `medals`                 | gold/silver/bronze counts                | silver/bronze only                   | none                                |

These ranges match the `CLAUDE.md` "realistic synthetic distributions":

> - Junior dev: leetcode <100 solved, github <10 repos, kaggle unranked
> - Mid dev: leetcode 100-250, github 10-30 repos, kaggle Contributor
> - Senior dev: leetcode 250+, github 30+ repos, kaggle Expert+

## Resume Builder

```python
def _build_resume(self, rng, years, skill_coverage):
    uni    = _UNIVERSITIES[int(rng.integers(0, len(_UNIVERSITIES)))]
    degree = _DEGREES[int(rng.integers(0, len(_DEGREES)))]
    grad_year = int(2024 - years)

    all_skills = [...]    # generic 15-skill pool
    n_skills = max(1, int(len(all_skills) * skill_coverage))
    skill_idx = rng.choice(len(all_skills), size=n_skills, replace=False)
    chosen_skills = [all_skills[i] for i in sorted(skill_idx)]

    role = role_choices[int(rng.integers(0, len(role_choices)))]
    company = company_choices[int(rng.integers(0, len(company_choices)))]

    return {
        "education":  ResumeSection(content=f"{degree} from {uni}, graduated {grad_year}.", years_experience=0.0),
        "experience": ResumeSection(content=f"{role} at {company} for {years:.1f} years. …", years_experience=years),
        "skills":     ResumeSection(content=", ".join(chosen_skills),                       years_experience=0.0),
    }
```

Notice:

- `grad_year = 2024 - years` — keeps the resume consistent with the candidate's claimed experience.
- `skill_coverage` controls *how many* generic skills appear. A junior (0.3) lists ~4 skills; a senior (1.0) lists all 15.
- The "experience" section is the *only* one that surfaces a non-zero `years_experience` — the env's `read_resume_section` info dict exposes this.

## Ground-Truth Score Computation

```python
@staticmethod
def _compute_ground_truth_scores(leetcode, github, kaggle, years, resume) -> dict[str, float]:
    lc_norm    = min(leetcode.problems_solved / 400.0, 1.0)
    hard_ratio = leetcode.hard / leetcode.problems_solved if leetcode.problems_solved > 0 else 0.0
    gh_norm    = min(github.repos / 60.0, 1.0)
    stars_norm = min(github.stars_received / 1000.0, 1.0)
    technical  = 0.30*lc_norm + 0.20*hard_ratio + 0.30*gh_norm + 0.20*stars_norm

    experience = min(years / 10.0, 1.0)

    streak_norm   = min(github.commit_streak_days / 180.0, 1.0)
    activity_norm = min(github.contributions_last_year / 1500.0, 1.0)
    contest_norm  = min(leetcode.contest_rating / 2000.0, 1.0)
    growth        = 0.40*streak_norm + 0.35*activity_norm + 0.25*contest_norm

    return {"technical": round(technical, 4), "experience": round(experience, 4), "growth": round(growth, 4)}
```

This is a **bookkeeping** function — it derives the three dimension scores that the grader *could* compare against (although currently the graders don't use them; they use the discrete ground-truth label instead). Useful if you wanted to grade an agent's `score_dimension` actions against ground truth.

The formula in plain English:

- **technical** = a 30/20/30/20 mix of LeetCode volume, hard-problem ratio, repo count, stars. Capped at 1.0.
- **experience** = years / 10, capped at 1.0.
- **growth** = 40/35/25 mix of commit-streak length, last-year contributions, and contest rating. Capped at 1.0.

These are *plausible-looking* scores, not derived from any real recruiter study. They're designed so a senior tier maxes them out, a junior tier bottoms them out, and a mid tier sits in the middle — giving graders that *do* use them a meaningful signal.

## Determinism Verification

Try this experiment in your head (or in a Python REPL):

```python
f = ProfileFactory()
pool1 = f.generate_pool(seed=42, count=10, label_distribution={"shortlist":3,"hold":4,"reject":3})
pool2 = f.generate_pool(seed=42, count=10, label_distribution={"shortlist":3,"hold":4,"reject":3})
assert [c.model_dump() for c in pool1] == [c.model_dump() for c in pool2]   # passes
```

Because the RNG is reseeded inside `generate_pool`, even if you call `generate_job_description` between the two pool calls, the assertion still holds.

## Soundbites

- *"Profiles are entirely synthetic — no live HTTP. All randomness flows through one `numpy.random.default_rng(seed)` that gets reseeded at the start of every generator call. That's why `reset(seed=42)` is byte-identical across calls."*
- *"Ground-truth labels are picked first, then the stats are drawn from a tier-matched distribution. So a 'shortlist' candidate always *looks* like a senior dev — `LeetCode` 250+, `GitHub` 30+ repos, `Kaggle` Expert+. No post-hoc inference."*
- *"`nice_to_have` skills are sampled per seed but `required_skills` are not — required skills are part of the role's identity; the variation comes from the bonus skills."*
- *"`_compute_ground_truth_scores` derives per-dimension scores from the raw stats so any grader that wants to evaluate `score_dimension` actions against truth has a target to compare against."*
