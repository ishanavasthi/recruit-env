# 10 — Tests (`tests/`)

A pytest suite of ~35 tests across three files. They cover the env, the graders, and the API. Running `pytest tests/ -v` exercises every layer of the system.

```
tests/
├── test_environment.py    ── ~12 tests on the env directly (no HTTP)
├── test_graders.py        ── ~5 tests covering EasyGrader + HardGrader
└── test_api.py            ── ~8 tests against the FastAPI TestClient
```

The README's badge `Tests: 35 passed` refers to this suite.

## Why These Tests Matter

`CLAUDE.md` lists four hard rules:

1. NO live HTTP calls anywhere in env/ or api/.
2. `reset(seed=42)` MUST always return identical state.
3. All graders MUST return float in `[0.0, 1.0]`, MUST be deterministic.
4. Pydantic v2 only.

Rules 2 and 3 are *behavioural* — they can't be enforced by the type system. They're enforced by **these tests**. If you broke determinism (e.g. by adding a non-seeded random call inside the factory), `test_reset_deterministic` would catch you immediately.

---

## `test_environment.py` — Core Env

### `TestReset` (4 tests)

```python
def test_reset_deterministic(self, env):
    obs1 = env.reset("easy", seed=42)
    obs2 = env.reset("easy", seed=42)
    assert obs1.model_dump() == obs2.model_dump()
```

**The most important test in the project.** Calls `reset("easy", 42)` twice, dumps both observations to dicts, asserts equality. If anything in the env or factory ever introduces non-determinism, this fails.

```python
def test_reset_different_seeds(self, env):
    obs42 = env.reset("easy", seed=42)
    obs99 = env.reset("easy", seed=99)
    names42 = {c["name"] for c in obs42.candidates_summary}
    names99 = {c["name"] for c in obs99.candidates_summary}
    assert names42 != names99
```

Different seed → different name shuffle. This catches the bug where someone accidentally hardcodes the seed instead of routing it through `default_rng`.

```python
def test_reset_returns_correct_shape(self, env):
    obs = env.reset("easy", seed=42)
    assert obs.task_id == "easy"
    assert obs.step_number == 0
    assert obs.steps_remaining == 40
    assert len(obs.candidates_summary) == 10
    assert obs.revealed_data == {} and obs.decisions_made == {} and obs.scores_recorded == {}
    assert obs.done is False
```

Asserts initial observation matches the easy task config (40 steps, 10 candidates) and starts clean.

```python
def test_reset_invalid_task(self, env):
    with pytest.raises(ValueError, match="Unknown task_id"):
        env.reset("nonexistent")
```

Bad task_id → `ValueError` with the right message.

### `TestStep` (5 tests)

```python
def test_step_read_resume(self, env):
    obs = env.reset("easy", seed=42)
    cid = obs.candidates_summary[0]["id"]
    action = ReadResumeSectionAction(candidate_id=cid, section="education")
    obs, reward, done, info = env.step(action)
    assert reward.step_reward == 0.02
    assert cid in obs.revealed_data
    assert "education" in obs.revealed_data[cid]["resume_sections"]
    assert info.get("section_found") is True
    assert done is False
```

Asserts the +0.02 step reward, the bookkeeping update in `revealed_data`, and the `section_found=True` info.

```python
def test_step_check_platform_github(self, env):
    ...
    obs, reward, done, info = env.step(CheckPlatformAction(candidate_id=cid, platform="github"))
    assert reward.step_reward == 0.03
    assert "github" in obs.revealed_data[cid]["platforms"]
    gh = info["data"]
    for key in ("repos","top_languages","commit_streak_days","stars_received","contributions_last_year"):
        assert key in gh
```

+0.03 for platform check, and the `info["data"]` blob contains all expected GitHubStats fields.

```python
def test_step_repeat_action_no_extra_reward(self, env):
    ...
    env.step(action)                  # first time: 0.02
    _, reward2, _, _ = env.step(action)   # repeat: 0.0
    assert reward2.step_reward == 0.0
```

The novelty check — repeat reads earn nothing.

```python
def test_step_score_dimension(self, env):
    obs, reward, _, _ = env.step(ScoreDimensionAction(candidate_id=cid, dimension="technical", score=0.75))
    assert reward.step_reward == 0.02
    assert obs.scores_recorded[cid]["technical"] == 0.75
```

```python
def test_step_make_decision(self, env):
    obs, reward, done, info = env.step(MakeDecisionAction(candidate_id=cid, decision="hold"))
    assert reward.step_reward == 0.0
    assert obs.decisions_made[cid] == "hold"
    assert info.get("decision_recorded") is True
```

Decisions earn 0.0 step reward; the value is recorded in `decisions_made`.

### `TestTermination` (2 tests)

```python
def test_episode_terminates_on_all_decided(self, env):
    obs = env.reset("easy", seed=42)
    state = env.state()
    gt = {c.id: c.ground_truth_label for c in state.candidates}
    for cid in list(gt.keys()):
        obs, _, done, info = env.step(MakeDecisionAction(candidate_id=cid, decision=gt[cid]))
    assert done is True
    assert info["termination_reason"] == "all_decided"
```

10 ground-truth-correct decisions → episode ends with `"all_decided"`.

```python
def test_episode_terminates_on_budget(self, env):
    obs = env.reset("easy", seed=42)
    cids = [c["id"] for c in obs.candidates_summary]
    done = False
    for i in range(40):
        cid = cids[i % len(cids)]
        obs, _, done, info = env.step(ReadResumeSectionAction(candidate_id=cid, section="education"))
        if done: break
    assert done is True
    assert info["termination_reason"] == "budget_exhausted"
```

Spam the same read 40 times → budget runs out, episode ends with `"budget_exhausted"`. Tests both termination reasons.

### `TestErrors` (4 tests)

```python
def test_cannot_step_without_reset(self):
    env = RecruitmentEnvironment()
    with pytest.raises(RuntimeError, match="No active episode"):
        env.step(ReadResumeSectionAction(candidate_id="candidate_001", section="education"))
```

```python
def test_duplicate_decision_raises(self, env):
    obs = env.reset("easy", seed=42)
    cid = obs.candidates_summary[0]["id"]
    env.step(MakeDecisionAction(candidate_id=cid, decision="hold"))
    with pytest.raises(ValueError, match="already has a decision"):
        env.step(MakeDecisionAction(candidate_id=cid, decision="reject"))
```

Decisions are terminal — second attempt raises.

```python
def test_invalid_candidate_raises(self, env):
    env.reset("easy", seed=42)
    with pytest.raises(ValueError, match="Unknown candidate_id"):
        env.step(ReadResumeSectionAction(candidate_id="does_not_exist", section="skills"))
```

```python
def test_step_after_done_raises(self, env):
    # ... decide everyone ...
    with pytest.raises(RuntimeError, match="already done"):
        env.step(ReadResumeSectionAction(candidate_id=state.candidates[0].id, section="education"))
```

After episode ends, further steps raise.

### `TestState` (2 tests)

```python
def test_state_returns_deep_copy(self, env):
    env.reset("easy", seed=42)
    s1 = env.state()
    s1.decisions_made["injected"] = "hack"
    s2 = env.state()
    assert "injected" not in s2.decisions_made
```

Confirms `state()` returns a deep copy — caller mutations don't leak.

```python
def test_state_without_reset_raises(self):
    env = RecruitmentEnvironment()
    with pytest.raises(ValueError, match="No active episode"):
        env.state()
```

---

## `test_graders.py` — Grader Correctness

### `TestEasyGrader` (3 tests)

```python
def test_easy_grader_perfect_score(self, env):
    # decide everyone with ground truth → score >= 0.95
    ...
    assert score >= 0.95

def test_easy_grader_all_wrong(self, env):
    # flip every label → score <= 0.2
    ...
    assert score <= 0.2

def test_easy_grader_partial(self, env):
    # first half correct, second half "hold" → score in [0.3, 0.7]
    ...
    assert 0.3 <= score <= 0.7
```

Three boundary cases: perfect, all-wrong, half-and-half. The score brackets verify the grader produces *graded* output, not just binary pass/fail.

### `TestGraderDeterminism` (1 test)

```python
def test_grader_deterministic(self, env):
    grader = EasyGrader()
    scores = []
    for _ in range(3):
        obs = env.reset("easy", seed=42)
        state = env.state()
        gt = {c.id: c.ground_truth_label for c in state.candidates}
        _decide_all(env, gt)
        final = env.state()
        scores.append(grader.grade(obs, final))
    assert scores[0] == scores[1] == scores[2]
```

**Hardest rule to verify mechanically.** Three identical episodes, same grader, same inputs → same outputs. Catches any non-determinism in the grader (e.g. if someone accidentally used `set` iteration order in a Python without dict-ordering guarantees, or a hash-randomized lookup).

### `TestGraderRange` (1 test)

```python
def test_grader_range(self, env):
    grader = EasyGrader()
    rng = np.random.default_rng(12345)
    choices = ["shortlist", "hold", "reject"]
    for i in range(100):
        obs = env.reset("easy", seed=i)
        state = env.state()
        random_decisions = {c.id: choices[int(rng.integers(0, 3))] for c in state.candidates}
        _decide_all(env, random_decisions)
        final = env.state()
        score = grader.grade(obs, final)
        assert 0.0 <= score <= 1.0, f"seed={i}: score {score} out of range"
```

**Fuzz test.** 100 episodes with random decisions on random seeds. Every score must land in `[0, 1]`. Catches off-by-one bonus stacking, clamping bugs, and accidental negative penalties.

### `TestHardGrader` (2 tests)

```python
def test_hard_grader_uses_f1(self, env):
    # shortlist-all:    score in (0.3, 0.6)   (precision=0.25, recall=1.0, F1≈0.4)
    # shortlist-none:   score < 0.15           (F1=0.0)
    # perfect:          score > 0.95          (F1=1.0)
    ...
    assert score_perf > score_all
```

**Validates the F1 grader by construction.** All three reference points are hand-computed from the precision/recall of the corresponding decision sets. If someone replaced F1 with plain accuracy, shortlist-all would suddenly score very high (because rejecting all 15 non-shortlists is "correct"), and this test would fail.

```python
def test_hard_grader_undecided_auto_hold(self, env):
    # spend all 30 steps on read_resume → never decide → all undecided
    ...
    assert score < 0.1
```

Verifies undecided candidates are auto-graded as `"hold"`, not skipped. If the grader skipped them, shortlist set would be empty (which it is) but the total denominator would also shrink (which it shouldn't).

---

## `test_api.py` — HTTP Layer (via TestClient)

`fastapi.testclient.TestClient` lets you hit the FastAPI app without starting uvicorn. All tests use `client = TestClient(app)`.

### `TestHealth` (2 tests)

```python
def test_root(self):                client.get("/")        # 200, status==ok
def test_health_returns_200(self):  client.get("/health")  # 200, version 1.0.0
```

### `TestTasks` (2 tests)

```python
def test_tasks_endpoint_returns_3_tasks(self):
    r = client.get("/tasks")
    assert len(r.json()["tasks"]) == 3
    assert {t["id"] for t in r.json()["tasks"]} == {"easy", "medium", "hard"}

def test_tasks_includes_action_schema(self):
    schema = client.get("/tasks").json()["action_schema"]
    assert "oneOf" in schema or "$defs" in schema
```

Three task ids; action schema is a `oneOf`-style discriminated union JSON Schema.

### `TestReset` (2 tests)

```python
def test_reset_valid_task(self):
    r = client.post("/reset", json={"task_id":"easy","seed":42})
    assert r.status_code == 200
    obs = r.json()
    assert obs["task_id"] == "easy" and obs["step_number"] == 0 and obs["steps_remaining"] == 40
    assert len(obs["candidates_summary"]) == 10 and obs["done"] is False

def test_reset_invalid_task_400(self):
    r = client.post("/reset", json={"task_id":"nonexistent","seed":42})
    assert r.status_code == 400
    assert "Unknown task_id" in r.json()["detail"]
```

Same shape checks as the env test, but through the HTTP layer.

### `TestStep` (2 tests)

```python
def test_step_without_reset_400(self):
    # set up + finish an episode, then attempt another step → 400
    ...

def test_step_returns_correct_shape(self):
    client.post("/reset", json={"task_id":"easy","seed":42})
    state = client.get("/state").json()
    cid = state["candidates"][0]["id"]
    r = client.post("/step", json={"type":"check_platform","candidate_id":cid,"platform":"leetcode"})
    body = r.json()
    assert "observation" in body and "reward" in body and "done" in body and "info" in body
    assert body["reward"]["step_reward"] == 0.03
```

Tests the StepResponse shape end-to-end.

### `TestFullEpisode` (2 tests)

```python
def test_full_easy_episode(self):
    # Reset
    client.post("/reset", json={"task_id":"easy","seed":42})
    # Get ground truth from /state
    gt = {c["id"]: c["ground_truth_label"] for c in client.get("/state").json()["candidates"]}
    # Decide all with correct labels
    for cid, label in gt.items():
        r = client.post("/step", json={"type":"make_decision","candidate_id":cid,"decision":label})
    assert r.json()["done"] is True
    # Grade
    grade = client.post("/grader").json()
    assert 0.0 <= grade["score"] <= 1.0 and grade["task_id"] == "easy" and "breakdown" in grade
```

End-to-end: reset → 10 decisions → done → grade. Confirms the entire HTTP plumbing works.

```python
def test_grader_rejects_incomplete_episode(self):
    client.post("/reset", json={"task_id":"easy","seed":42})
    r = client.post("/grader")
    assert r.status_code == 400 and "not complete" in r.json()["detail"]
```

Can't grade an in-progress episode.

### `TestState` (1 test)

Verifies `GET /state` returns the expected `EpisodeState` shape after reset on a different task (medium → 5 candidates).

---

## Running the Tests

```bash
pytest tests/ -v
```

With `-v` you see each test name + pass/fail. The fixtures (`env` in particular) construct a fresh `RecruitmentEnvironment` per test, so tests don't bleed state across one another.

**TestClient subtlety:** the API tests share *one* global `_env` (the singleton in `api/main.py`). That's why `test_step_without_reset_400` carefully *reset → finish an episode* before testing the after-done error, instead of just hitting a never-reset env. Comment in the file calls this out.

## Coverage Map

| Area                | Tests                                                                                |
| ------------------- | ------------------------------------------------------------------------------------ |
| Determinism         | `test_reset_deterministic`, `test_grader_deterministic`                              |
| Step rewards        | All `TestStep` tests                                                                 |
| Termination         | `test_episode_terminates_on_all_decided`, `_on_budget`                               |
| Errors (env)        | All `TestErrors` tests                                                               |
| Errors (API)        | `test_reset_invalid_task_400`, `test_step_without_reset_400`, `test_grader_rejects_incomplete_episode` |
| Graders — accuracy  | `TestEasyGrader`, `TestHardGrader::test_hard_grader_uses_f1`                         |
| Graders — range     | `test_grader_range` (100 random episodes)                                            |
| State deep-copy     | `test_state_returns_deep_copy`                                                       |
| Schema discovery    | `test_tasks_includes_action_schema`                                                  |
| Full episode flow   | `test_full_easy_episode`                                                             |

## Soundbites

- *"35 tests across three files. The core invariants — determinism, score range, deep-copy isolation — each have a dedicated test, plus a 100-run fuzz test that confirms the grader is bounded for arbitrary decision sets."*
- *"`test_reset_deterministic` is the highest-value test in the suite. It guards CLAUDE.md's hardest contract — `reset(seed=42)` must return byte-identical state — which can't be enforced at the type level."*
- *"`test_hard_grader_uses_f1` validates the grader by construction: shortlist-all gives ~0.4 (F1 of precision 0.25 and recall 1.0), shortlist-none gives ~0 (F1 of 0), perfect gives ~1. If someone swapped F1 for accuracy, the shortlist-all case would suddenly score high and the test would fail."*
- *"API tests use FastAPI's `TestClient` so no uvicorn process is needed — and a full episode is tested end-to-end via `test_full_easy_episode`."*
