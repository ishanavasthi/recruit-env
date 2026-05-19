# 08 — FastAPI Server (`api/main.py`)

The HTTP transport. ~350 lines. Wraps `RecruitmentEnvironment` with REST endpoints, validates incoming JSON, serialises outgoing Pydantic models, adds CORS + request logging, and includes a built-in rule-based baseline accessible via `GET /baseline`.

## High-Level Shape

```
api/main.py
├── imports + env setup
├── shared state           (single RecruitmentEnvironment, asyncio lock, caches)
├── Pydantic req/resp models
├── RequestLoggingMiddleware
├── lifespan context manager
├── FastAPI app + CORS + middleware mount
└── endpoints:
        GET  /
        GET  /health
        POST /reset
        POST /step
        GET  /state
        GET  /tasks
        POST /grader
        GET  /baseline
        _run_rule_baseline   (helper, not an endpoint)
```

## Shared State (Module Globals)

```python
_env = RecruitmentEnvironment()
_lock = asyncio.Lock()
_initial_obs: dict[str, Observation] = {}      # task_id → observation captured at reset
_baseline_cache: dict[str, Any] | None = None  # memoises /baseline result
```

**One env instance.** Singleton. All requests share it.

**One asyncio lock.** Wraps every endpoint that touches `_env`. Prevents two concurrent requests from interleaving inside the env (which is also protected by its own `threading.Lock` — belt and suspenders).

**`_initial_obs` cache.** Graders need the *initial* observation, not the current one. Since `reset` overwrites internal state, the API caches the initial obs at reset time so `POST /grader` can pass it back to the grader later.

**`_baseline_cache`.** First call to `GET /baseline` computes all three task scores; subsequent calls just return the cached dict. The baseline is deterministic (seed=42) so this caching is safe.

## Action Type Adapter

```python
Action = Annotated[
    Union[ReadResumeSectionAction, CheckPlatformAction,
          ScoreDimensionAction, MakeDecisionAction],
    Field(discriminator="type"),
]
_action_adapter = TypeAdapter(Action)
```

Same discriminated union as `env/models.py`, but rebuilt here so the API can:
1. Use it as a FastAPI request body type (the framework picks the right subtype automatically).
2. Generate the JSON Schema via `_action_adapter.json_schema()` and serve it from `/tasks`.

`TypeAdapter` is the Pydantic v2 way of treating a non-model type (a `Union` here) as if it were a model — gives you `.validate_python()`, `.dump_json()`, `.json_schema()`, etc.

## Request / Response Models

```python
class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed:    int = 42

class StepResponse(BaseModel):
    observation: Observation
    reward:      Reward
    done:        bool
    info:        dict[str, Any]

class GradeResponse(BaseModel):
    score:     float
    breakdown: dict[str, Any]
    task_id:   str

class BaselineResponse(BaseModel):
    scores:    dict[str, float]
    timestamp: str

class HealthResponse(BaseModel):
    status:  str
    version: str
```

All request/response payloads are explicitly typed → FastAPI auto-generates OpenAPI docs at `/docs`.

## Request Logging Middleware

```python
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s -> %d (%.1fms)",
                    request.method, request.url.path,
                    response.status_code, duration_ms)
        return response
```

Logs every request as `METHOD PATH -> STATUS (latency)` — useful for debugging HF Spaces deployments where you can't easily attach a profiler.

## Lifespan Context

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RecruitEnv API starting up")
    yield
    logger.info("RecruitEnv API shutting down")
```

FastAPI's preferred way to run startup / shutdown code in newer versions (vs deprecated `@app.on_event`). Currently just logs — but it's the right hook if you ever need to warm a cache or close a DB pool.

## App Construction + CORS

```python
app = FastAPI(title="RecruitEnv", description="...", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware)
```

**CORS is fully open.** That's deliberate for HF Spaces — the env is meant to be called from arbitrary clients (a notebook in Colab, a Hugging Face evaluation script, your local dev box). For a production system you'd lock this down.

---

## Endpoints

### `GET /` — root

```python
return {"status": "ok", "environment": "RecruitEnv", "version": "1.0.0"}
```

Just a landing page so hitting the bare URL returns something useful.

### `GET /health`

```python
return HealthResponse(status="ok", version="1.0.0")
```

Used by:
- The Dockerfile `HEALTHCHECK` (`curl -f http://localhost:7860/health`).
- HF Spaces' deployment-validation ping.
- `baseline/run_baseline.py::check_server()`.

### `POST /reset`

```python
@app.post("/reset", response_model=Observation)
async def reset(body: ResetRequest | None = None) -> Observation:
    if body is None:
        body = ResetRequest()
    async with _lock:
        try:
            obs = _env.reset(body.task_id, body.seed)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _initial_obs[body.task_id] = obs
        return obs
```

**Flow:**

1. Default body if not provided (`task_id="easy"`, `seed=42`).
2. Acquire async lock.
3. Call `_env.reset(...)`. If task_id is unknown → `ValueError` → return 400.
4. Cache the returned obs by task_id (needed by `POST /grader` later).
5. Return the obs (FastAPI serialises Pydantic → JSON).

### `POST /step`

```python
@app.post("/step", response_model=StepResponse)
async def step(action: Action) -> StepResponse:
    async with _lock:
        try:
            obs, reward, done, info = _env.step(action)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
```

**FastAPI does the heavy lifting here.** The `action: Action` parameter triggers:
- Read the JSON body.
- Look at the `type` discriminator field.
- Validate the rest of the body against that subtype.
- Pass an already-typed Pydantic model into your handler.

If validation fails, FastAPI returns 422 *automatically* with a structured error pointing at the bad field — no try/except needed.

`RuntimeError` (no active episode, episode done) and `ValueError` (unknown candidate, duplicate decision) are converted to 400s.

### `GET /state`

```python
@app.get("/state", response_model=EpisodeState)
async def get_state() -> EpisodeState:
    async with _lock:
        try:
            return _env.state()
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
```

Returns the *full* internal state — **including ground-truth labels**. The README is explicit:

> "Return full internal state (including ground-truth labels which are useful for debugging, not for agents)."

For a well-behaved agent, this is a debug-only endpoint. For the rule-based baseline, this is how it sneakily reads stats without spending step budget (batch mode in `run_baseline.py`).

### `GET /tasks`

```python
@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    tasks = [dataclasses.asdict(cfg) for cfg in TASK_REGISTRY.values()]
    action_schema = _action_adapter.json_schema()
    return {"tasks": tasks, "action_schema": action_schema}
```

Two pieces:

1. **`tasks`** — list of all `TaskConfig`s as dicts (id, name, max_steps, candidate_count, etc.).
2. **`action_schema`** — the full JSON Schema of the discriminated `Action` union. Includes `oneOf` + `discriminator: {propertyName: "type"}` so an LLM client can read this and know exactly what valid actions look like.

This endpoint is how external evaluators discover what the env supports without hardcoding any of it.

### `POST /grader`

```python
@app.post("/grader", response_model=GradeResponse)
async def grade() -> GradeResponse:
    async with _lock:
        try:
            state = _env.state()
        except ValueError as exc:
            raise HTTPException(400, str(exc))

        if not state.is_done:
            raise HTTPException(400, "Episode is not complete. Finish the episode before grading.")

        task_id     = state.task_id
        grader      = GRADER_REGISTRY.get(task_id)
        initial_obs = _initial_obs.get(task_id)
        # ... 400 checks on each ...

        score = grader.grade(initial_obs, state)
        gt_map = {c.id: c.ground_truth_label for c in state.candidates}
        breakdown = {
            "grader_score":  score,
            "success":       score >= TASK_REGISTRY[task_id].success_threshold,
            "threshold":     TASK_REGISTRY[task_id].success_threshold,
            "decisions":     dict(state.decisions_made),
            "ground_truth":  gt_map,
            "correct":       sum(1 for cid, d in state.decisions_made.items() if gt_map.get(cid) == d),
            "total":         len(state.candidates),
        }
        return GradeResponse(score=score, breakdown=breakdown, task_id=task_id)
```

**Flow:**

1. Get current state.
2. Refuse to grade an unfinished episode → 400.
3. Look up grader + cached initial obs by task_id.
4. Call `grader.grade(initial_obs, state)` → float.
5. Build a verbose breakdown: score, pass/fail vs threshold, all decisions, all ground truths, correct count.
6. Return `GradeResponse`.

The breakdown is what makes this endpoint useful — judges can audit exactly why the score is what it is.

### `GET /baseline`

```python
@app.get("/baseline", response_model=BaselineResponse)
async def baseline() -> BaselineResponse:
    global _baseline_cache
    if _baseline_cache is not None:
        return BaselineResponse(**_baseline_cache)

    scores: dict[str, float] = {}
    for task_id in TASK_REGISTRY:
        score = await _run_rule_baseline(task_id)
        scores[f"{task_id}_task"] = round(score, 4)

    result = {"scores": scores, "timestamp": datetime.now(timezone.utc).isoformat()}
    _baseline_cache = result
    return BaselineResponse(**result)
```

**Important distinction:** this is the *built-in, no-API-key-required* rule-based baseline, not the LLM baseline in `baseline/run_baseline.py`. Judges can hit `GET /baseline` from any HF Space and get a reproducible score without needing an OpenAI key.

It memoises the first call's result in `_baseline_cache` — subsequent calls are O(1).

## `_run_rule_baseline` — the Built-In Heuristic Agent

```python
async def _run_rule_baseline(task_id: str, seed: int = 42) -> float:
    env = RecruitmentEnvironment()       # ← LOCAL env, not the singleton
    obs = env.reset(task_id, seed)
    initial_obs = obs
    cids = [c["id"] for c in obs.candidates_summary]

    for cid in cids:
        if obs.done: break

        _, _, done, info_lc = env.step(CheckPlatformAction(candidate_id=cid, platform="leetcode"))
        lc_data = info_lc.get("data", {})
        if done: break

        obs, _, done, info_gh = env.step(CheckPlatformAction(candidate_id=cid, platform="github"))
        gh_data = info_gh.get("data", {})
        if done: break

        problems = lc_data.get("problems_solved", 0)
        repos    = gh_data.get("repos", 0)
        hard     = lc_data.get("hard", 0)

        if   problems >= 200 and repos >= 25 and hard >= 15: decision = "shortlist"
        elif problems < 80  and repos < 8:                   decision = "reject"
        else:                                                decision = "hold"

        obs, _, done, _ = env.step(MakeDecisionAction(candidate_id=cid, decision=decision))
        if done: break

    final_state = env.state()
    grader = GRADER_REGISTRY.get(task_id)
    return grader.grade(initial_obs, final_state) if grader else 0.0
```

**Two key design choices:**

1. **Local env instance**, not the singleton. Avoids stomping on whatever episode the HTTP client is currently running.
2. **Pure heuristic, deterministic.** `problems_solved`, `repos`, and `hard` thresholds are simple if/elif/else. This is the *floor* the LLM baseline has to beat.

## Error Handling Map

| Source                            | Maps to HTTP code  |
| --------------------------------- | ------------------ |
| Pydantic validation failure       | 422 (auto)         |
| Unknown task_id                   | 400                |
| `step()` without `reset()`        | 400                |
| `step()` after `done`             | 400                |
| Unknown candidate_id              | 400                |
| Duplicate decision                | 400                |
| `state()` without active episode  | 404                |
| `grade()` on unfinished episode   | 400                |

## Soundbites

- *"Single shared environment instance behind an asyncio.Lock at the HTTP layer plus a threading.Lock inside the env. Belt and suspenders."*
- *"FastAPI handles action validation automatically because `action: Action` uses Pydantic's discriminated union — bad payloads return 422 with structured field errors before any handler code runs."*
- *"`/tasks` returns both the task list and the JSON Schema of the action union — clients (including LLMs) can read the schema to discover the action shape without hardcoding anything."*
- *"`/baseline` runs the built-in rule-based agent on a *separate* env instance so it doesn't disturb the user's active episode, then memoises the result."*
- *"The initial observation is cached at reset time because the grader needs it but the env doesn't preserve it across steps."*
