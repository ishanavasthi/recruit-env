# 12 — Interview Q&A Cheat Sheet

The questions an interviewer is most likely to ask about this project, plus the answers you should be ready to give. Use this as **the night-before-interview crib sheet**. Every answer has the file/line you can point at if pressed.

---

## "Give me the 60-second pitch."

> "RecruitEnv is an OpenEnv-compliant reinforcement learning environment that turns candidate pipeline triage into a sequential decision problem. An agent receives a job description and a pool of synthetic candidate profiles; each step it can either gather a signal — read a resume section, check GitHub / LeetCode / Kaggle — or commit a decision: shortlist, hold, or reject. There's a finite step budget, so it has to learn to prioritise. I built three difficulty tiers from 4 steps per candidate down to 1.5, a four-component reward function with a demographic-parity penalty, a FastAPI server on port 7860, a Docker image deployed to Hugging Face Spaces, and a baseline LLM agent that hits 0.97 average score across all three tasks. The whole thing is deterministic — `reset(seed=42)` always returns identical state — and there are 35 pytest tests guarding that contract."

---

## "Why this domain? Why not games?"

> "OpenEnv specifically wants real-world tasks rather than games — that's in the spec. Recruiting is a real combinatorial information-gathering problem: signals scattered across multiple platforms, a step budget, fairness concerns. It maps cleanly onto sequential decision-making. And there's existing demand — recruiters do this work daily; an agent that can do it well has obvious utility. The alternative was something like email triage or content moderation — recruiting won because the signals are objective enough that grading is unambiguous."

---

## "Walk me through what happens when an agent calls `step()`."

Reference: `env/environment.py:_step_locked`.

> "Six phases.
> 1. Precondition checks: refuse if no episode or already done. Both raise `RuntimeError`.
> 2. Domain validation: confirm `candidate_id` is in the pool, refuse duplicate decisions. Both raise `ValueError`.
> 3. Compute the step reward — *before* mutating state, because the reward depends on whether the action is novel.
> 4. Dispatch to one of four executor methods using `isinstance`. They mutate the state and populate the `info` dict.
> 5. Increment the step counter, check termination — `all_decided` or `budget_exhausted`, record which in `info`.
> 6. If terminal, compute accuracy + efficiency − fairness, clamp to `[0,1]`, persist as `episode_score`. Build the final `Reward` object. Otherwise just return a step reward.
>
> Finally build a fresh `Observation` (deep-copied) and return the four-tuple."

---

## "How do you guarantee determinism?"

Reference: `env/profile_factory.py:generate_pool` line 231, `tests/test_environment.py:test_reset_deterministic`.

> "Three things have to hold. First, every random draw flows through a single `numpy.random.default_rng(seed)`. Second, `generate_pool` and `generate_job_description` both *reseed* their internal RNG at the start of every call — so calls are independent of each other and only depend on the input seed. Third, `tests/test_environment.py:test_reset_deterministic` calls `reset('easy', 42)` twice and asserts `obs1.model_dump() == obs2.model_dump()`. It catches any drift immediately. CLAUDE.md lists determinism as a hard rule and that test is the enforcement mechanism."

---

## "What does the reward function look like and why?"

Reference: `env/rewards.py`, plus `env/fairness.py`.

> "Four components:
> - **Step reward**: +0.02 for a new resume section, +0.03 for a new platform check, +0.02 for any score_dimension, 0 for repeats or decisions. Dense, novelty-based. Stops reward hacking.
> - **Terminal accuracy**: correct shortlist = +1.0 × `weight_technical`, correct reject = +0.5, correct hold = +0.3. Asymmetric penalties for errors: shortlist-a-reject = −0.5, reject-a-shortlist = −0.3, other = −0.1. Linearly normalised into `[0,1]`.
> - **Efficiency bonus**: all-or-nothing +0.1 if every decision is correct AND used less than 60% of the budget.
> - **Fairness penalty**: 0.0 / 0.1 / 0.2 based on max shortlist-rate disparity across origin groups.
>
> Final = `clamp(accuracy + efficiency − fairness_penalty, 0, 1)`.
>
> Each piece solves a known reward-shaping failure: dense step reward avoids the credit-assignment problem; weighted accuracy makes the JD matter; asymmetric penalties match real recruiting costs; efficiency bonus stops the agent from gathering forever; fairness penalty stops it from learning a demographic shortcut."

---

## "How does the fairness penalty work and isn't this oversimplified?"

Reference: `env/fairness.py`.

> "Every candidate name is mapped to one of six broad origin groups — South Asian, East Asian, Latin American, European, Middle Eastern / North African, African — via a hardcoded first-name lookup. At episode end we compute the per-group shortlist rate (shortlists / decided), take the max minus min across groups, and threshold it: under 0.2 is free, 0.2 to 0.4 is −0.1, ≥0.4 is −0.2.
>
> Is it oversimplified? Yes, and the file explicitly comments on that. The lookup table is a coarse heuristic, not a Census Bureau classifier. The goal is to catch gross imbalances — 'agent shortlists every European name but no African names' — not to be a precise ethnic classifier. A more rigorous system would use multi-name resolution, intersectional groups, and statistical tests; this is a hackathon-grade signal that lets graders penalise obvious bias."

---

## "Why discriminated unions for actions?"

Reference: `env/models.py` lines 170-229.

> "Four action types, all with a `type: Literal[...]` field. Pydantic's `Annotated[Union[...], Field(discriminator='type')]` builds an O(1) dispatch on that field. Three concrete benefits:
> 1. **Cleaner errors.** If someone posts `{"type":"check_platform","platform":"twitter"}`, the error is `platform: Input should be 'github', 'leetcode' or 'kaggle'`, not a generic union failure.
> 2. **JSON Schema auto-generated** with `oneOf` + `discriminator: {propertyName: 'type'}`. The `/tasks` endpoint serves that schema to clients — including LLMs — so they can discover valid actions.
> 3. **FastAPI integration.** Declaring `action: Action` in the `/step` handler triggers automatic parsing + validation. The handler receives an already-typed Pydantic model."

---

## "What's the threading model?"

Reference: `env/environment.py:__init__`, `api/main.py:_lock`.

> "Two locks in series. The environment owns a `threading.Lock` that guards every public method (`reset`, `step`, `state`). The FastAPI layer owns an `asyncio.Lock` that wraps every endpoint touching the env. Belt and suspenders — async tasks won't interleave inside one handler, and direct non-HTTP callers (like the test suite, or the in-server baseline) are still safe.
>
> The env is a singleton. uvicorn runs with `--workers 1` so episode state isn't sharded across processes. One client at a time."

---

## "What if I asked you to add a fourth task — what would you change?"

Reference: `tasks/`.

> "Three steps.
> 1. Add `EXTRA_CONFIG = TaskConfig(id='extra', name=..., max_steps=..., candidate_count=..., label_distribution={...}, success_threshold=..., role_type=...)` to a new file `tasks/extra_task.py`.
> 2. Implement `class ExtraGrader(BaseGrader)` with a `grade(initial_obs, final_state) -> float` method that returns something in `[0, 1]`. Add it in the same file.
> 3. Register both in `tasks/__init__.py`: append to `TASK_REGISTRY` and `GRADER_REGISTRY`.
>
> Optionally update `openenv.yaml` and `README.md`. The env and API need no changes — they read from the registries dynamically."

---

## "Why is the LeetCode platform reward (+0.03) higher than the resume read (+0.02)?"

> "Platform checks return dense quantitative signal — repos, problems_solved, contest_rating, stars. Resume reads return free-text plus a single years_experience number. The reward differential gently nudges the agent toward the higher-information action when it has a choice. It's small enough that the agent will still read resumes when that's the only option, but biases exploration in the right direction."

---

## "What's the difference between the reward and the grader?"

Reference: `env/rewards.py` vs `tasks/easy_task.py`.

> "Different purposes. The **reward** is what the agent sees *during training and inference* — it has to be dense enough to learn from, uniform across tasks so the agent's policy generalises, and is set incrementally every step. The **grader** is the *official scoreboard* — task-specific, runs once at episode end via `POST /grader`, and uses the rubric that matches that task's difficulty profile. Easy uses weighted accuracy + a perfect-shortlist bonus; medium uses accuracy + a thoroughness adjustment for two designated 'conflicted' candidates; hard switches to F1 because plain accuracy rewards rejecting everyone. They overlap conceptually but are deliberately decoupled in code."

---

## "How does the baseline agent actually decide?"

Reference: `baseline/run_baseline.py`.

> "It uses a structured candidate loop with budget-aware mode selection. Steps-per-candidate ≥ 2.8 → 'full' mode: check GitHub, check LeetCode, decide. ≥ 1.8 → 'quick' mode: GitHub only then decide. < 1.8 → 'batch' mode: harvest stats from `GET /state` (which doesn't cost steps), then spend the whole budget on decisions.
>
> For the decision itself, when the budget is comfortable it asks the LLM with `temperature=0` and `max_tokens=10` — a one-word prompt with the threshold rules embedded. When the budget is tight, it uses the same threshold rules in pure Python — no LLM call. The substring-parse pattern lets it accept 'shortlist' or 'the answer is shortlist'.
>
> Recorded scores: easy 1.0, medium 1.0, hard 0.92, average 0.974 with Gemini 2.0 Flash Lite through an OpenAI-compatible endpoint."

---

## "What happens if an agent posts an invalid action?"

> "Three layers catch it.
> 1. **Pydantic** rejects shape problems immediately. Posting `{type:'check_platform', platform:'twitter'}` gets a 422 with `platform: Input should be 'github', 'leetcode' or 'kaggle'`. The handler never runs.
> 2. **Env-level semantic checks** for things Pydantic can't know: unknown `candidate_id`, duplicate decision, stepping before reset, stepping after done. These raise `ValueError` / `RuntimeError` which the API converts to 400s with descriptive messages.
> 3. **Step reward of 0** when an action repeats. The action still completes (you can re-read the same section), but the agent earns nothing — there's no reward exploit for spamming."

---

## "Where would you take this next?"

> "Three directions:
> 1. **Richer profile factory** — currently the 'conflicted' candidates in the medium task are just convention; the grader treats positions 0 and 1 as conflicted but the factory doesn't actually mix their signals. I'd inject true cross-platform tension (great GitHub, weak LeetCode, mid Kaggle) and let the agent's thoroughness pay off.
> 2. **Train an actual policy** instead of just a baseline. The env is set up for it — gym-style API, dense reward, finite action space. A PPO agent with a transformer encoder should be able to beat the LLM baseline on hard once it learns to budget allocate.
> 3. **Better fairness signal** — instead of a hard threshold, use a continuous statistical-parity score (e.g. demographic-parity ratio). And maybe a separate 'calibration' component that penalises over-confident scores on under-checked candidates."

---

## Quick-Recall One-Liners

Memorise these. They double as section headers if you're whiteboarding.

| Concept                          | One-liner                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| OpenEnv                          | Spec for RL envs: typed reset/step/state + openenv.yaml + Docker + ≥3 graded tasks.  |
| Determinism                      | All RNG through one seeded `numpy.random.default_rng`; reseeded per generator call.  |
| Discriminated union              | `Annotated[Union[...], Field(discriminator='type')]` — O(1) dispatch + clean errors. |
| Step reward                      | +0.02 resume / +0.03 platform / +0.02 score / 0 repeat / 0 decision.                 |
| Terminal accuracy                | Correct shortlist = `1.0 × weight_technical`; asymmetric error penalties.            |
| Efficiency bonus                 | All-or-nothing +0.1 for 100% correct under 60% budget.                               |
| Fairness penalty                 | Max shortlist-rate gap across 6 origin groups: <0.2 / [0.2,0.4) / ≥0.4 → 0 / 0.1 / 0.2. |
| Tasks                            | easy=10/40, medium=5/25, hard=20/30 — budget tightens with difficulty.               |
| Graders                          | weighted accuracy / thoroughness / F1.                                               |
| Termination                      | `all_decided or budget_exhausted`, reason in `info`.                                 |
| Locking                          | asyncio.Lock at API + threading.Lock in env. Single uvicorn worker.                  |
| Tests                            | 35 pytest tests across env / graders / API. Determinism + range + fuzz all covered. |
| Deployment                       | Python 3.11-slim, non-root, port 7860, healthcheck, HF Spaces `sdk: docker`.         |

---

## Final Tip

Don't claim more credit than you have. **Be honest that AI assisted the build** — and frame it as: *"I designed the architecture, the reward function, the task progression, and the test surface. AI helped me write the code at speed; the design decisions are mine."* That's a much stronger interview position than pretending you typed every line yourself.

Use the docs in this directory as evidence of *understanding* — if you can pull up `05-rewards.md` and walk an interviewer through the four reward components from memory, that proves you know the project regardless of who wrote the code.
