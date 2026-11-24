# OpenEnv and TRL refresh (2026-11-25)

This refreshes the 2026-11-21 audit (`2026-11-21-findings-and-roadmap.md`) before Phase 4. Phase 4 covers four pieces of work: the installable core, the OpenEnv adapter, the `openenv.yaml` rewrite and the typed client. It also covers the TRL GRPO example.

**Method.** I read primary sources only.
- PyPI JSON API, fetched 2026-11-25.
- The installed `openenv==0.14.0` wheel, in a throwaway venv (`uv venv /tmp/oe-probe`, Python 3.11).
- The `openenv/OpenEnv` GitHub repo, at tag `v0.14.0` (`58c2783bcc`) and at `main` (`3659dc73eb`, 2026-11-24).
- The `huggingface/trl` GitHub repo at tag `v1.18.2`.

**Labels.**
- *Run*: I executed it here.
- *Source*: I read it in code or docs at the stated ref.
- *Unverified*: neither.

**Resolving paths to URLs.**
- Paths like `openenv/core/env_client.py` are relative to the installed `site-packages`. They resolve at `https://github.com/openenv/OpenEnv/blob/v0.14.0/src/` + path. Example: https://github.com/openenv/OpenEnv/blob/v0.14.0/src/openenv/core/env_client.py.
- OpenEnv repo paths like `envs/...`, `docs/...` and `tests/...` resolve at `https://github.com/openenv/OpenEnv/blob/v0.14.0/` + path.
- TRL paths like `trl/...`, `docs/...` and `tests/...` resolve at `https://github.com/huggingface/trl/blob/v1.18.2/` + path.
- Every source in this note was read on 2026-11-25.

**Repo location.** `github.com/meta-pytorch/OpenEnv` now resolves to `github.com/openenv/OpenEnv`; the GitHub API returns `full_name: openenv/OpenEnv`. Code docstrings and the TRL docs link to `openenv/OpenEnv`. *Source, GitHub API, 2026-11-25.*

---

## 1. Package, version and import paths

### Versions

| Package | Latest | Released | requires_python | Source |
|---|---|---|---|---|
| `openenv` | **0.14.0** | 2026-11-19 | `>=3.10` | https://pypi.org/pypi/openenv/json (2026-11-25) |
| `openenv-core` | 0.3.0, the last release | 2026-05-11 | `>=3.10` | https://pypi.org/pypi/openenv-core/json (2026-11-25) |
| `trl` | **1.18.2** | 2026-11-19 | `>=3.11` | https://pypi.org/pypi/trl/json (2026-11-25) |

- `openenv` releases weekly: 0.7.0 came out on 2026-10-01, then 0.8, 0.9, 0.10, 0.11, 0.12 and 0.13, then 0.14.0 on 2026-11-19. GitHub release `v0.14.0` was published on 2026-11-19. Nothing newer was on PyPI on 2026-11-25. *Source.*
- `main` is 240 commits ahead of `v0.14.0`, including changes to `env_client.py`, `http_server.py` and `mcp_environment.py` (a FastMCP 4 migration). That file list comes from the GitHub compare API, which can truncate, so do not treat it as complete. Expect a 0.15.0 soon. Pin `openenv>=0.14,<0.15` and re-check before you bump. *Source, `GET repos/openenv/OpenEnv/compare/v0.14.0...main`, 2026-11-25.*
- **`openenv_core` is a shim.** The `openenv` 0.14.0 wheel ships two top-level packages, `openenv/` and `openenv_core/`.
  - `openenv_core/__init__.py` re-exports `openenv.core` and emits `DeprecationWarning: openenv_core is deprecated; import from openenv.core instead.`
  - `from openenv_core import GenericEnvClient` still works and resolves to `openenv.core.generic_client.GenericEnvClient`. *Run.*
- **Install conflict.** The `openenv-core` 0.3.0 wheel also ships `openenv/` and `openenv_core/`, so installing both distributions makes them overwrite each other's files. Replace the dependency rather than adding a second one. *Source: wheel file list, fetched 2026-11-25.*

### Import paths (0.14.0)

`openenv.core` re-exports everything in `openenv.core.env_server.__all__` and also provides the clients. *Source: `openenv/core/__init__.py`, `openenv/core/env_server/__init__.py`.*

```python
from openenv.core import (
    Environment,                 # openenv.core.env_server.interfaces
    Action, Observation, State,  # openenv.core.env_server.types
    create_app, create_fastapi_app, HTTPEnvServer,  # openenv.core.env_server.http_server
    ConcurrencyConfig,           # openenv.core.env_server.types
    EnvClient, SyncEnvClient, GenericEnvClient,     # lazy attrs
    TaskProvider,                # Protocol, structural only
)
from openenv.core.client_types import StepResult       # dataclass
from openenv.core.env_server.types import ResetRequest, EnvironmentMetadata
from openenv import AutoEnv, AutoAction                # top-level lazy attrs
```

All of these imports, plus `SessionLostError`, ran without error in the 0.14.0 venv. *Run.*

### `Environment` (`openenv/core/env_server/interfaces.py`)

```python
ActT = TypeVar("ActT", bound=Action)
ObsT = TypeVar("ObsT", bound=Observation)
StateT = TypeVar("StateT", bound=State)

class Environment(ABC, Generic[ActT, ObsT, StateT]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    REQUIRES_SINGLE_THREAD_EXECUTOR: bool = False
    rubric: Optional["Rubric"]

    def __init__(self, transform: Optional[Transform[ObsT]] = None,
                 rubric: Optional["Rubric"] = None): ...

    @abstractmethod
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              **kwargs: Any) -> ObsT: ...
    async def reset_async(self, seed=None, episode_id=None, **kwargs) -> ObsT: ...  # default calls reset

    @abstractmethod
    def step(self, action: ActT, timeout_s: Optional[float] = None, **kwargs: Any) -> ObsT: ...
    async def step_async(self, action, timeout_s=None, **kwargs) -> ObsT: ...       # default calls step

    @property
    @abstractmethod
    def state(self) -> StateT: ...

    def get_metadata(self) -> EnvironmentMetadata: ...   # default: class name, "<Name> environment", "1.0.0"
    def render_web(self, observation: dict) -> Optional[str]: ...
    def web_actions(self, observation: dict) -> list[tuple[str, dict]]: ...
    def close(self) -> None: ...
    # helpers: _apply_transform, _apply_rubric(_async), _reset_rubric(_async)
```

### Base types (`openenv/core/env_server/types.py`, pydantic v2)

| Type | Config | Fields |
|---|---|---|
| `Action` | `extra="forbid"`, `validate_assignment=True` | `metadata: dict = {}` |
| `Observation` | `extra="forbid"` | `done: bool = False`; `reward: bool \| int \| float \| None = None`; `metadata: dict = {}` |
| `State` | `extra="allow"` | `episode_id: Optional[str] = None`; `step_count: int = 0` (ge=0) |
| `ResetRequest` | `extra="allow"` | `seed: Optional[int]` (ge=0); `episode_id: Optional[str]` |
| `ConcurrencyConfig` | | `max_concurrent_envs: int = 1` (ge=1); `session_timeout: Optional[float] = None` |
| `EnvironmentMetadata` | | `name`, `description`, `readme_content?`, `version?`, `author?`, `documentation_url?` |

**Consequence for the adapter.** `Action` forbids extra fields, so the adapter's action model has to declare every field the agent sends.

### `create_app` and `create_fastapi_app` (`openenv/core/env_server/http_server.py`)

```python
def create_app(
    env: Callable[[], Environment],      # a class or factory, NOT an instance
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[ConcurrencyConfig] = None,
    gradio_builder: Optional[Callable[..., Any]] = None,
    custom_tab_name: str = "Custom",
    custom_tab_primary: bool = False,
    show_default_tab: bool = True,
    title_override: Optional[str] = None,
    state_cls: Type[State] = State,
    *,
    mode: Optional[ServerMode | str] = None,          # None -> $OPENENV_MODE, default "simulation"
    reset_observation_cls: Optional[Type[Observation]] = None,
    reset_cls: Type[ResetRequest] = ResetRequest,      # published in /schema only, not validated
) -> FastAPI

def create_fastapi_app(
    env, action_cls, observation_cls,
    max_concurrent_envs=None, concurrency_config=None, env_name=None,
    state_cls=State, *, mode=None, reset_observation_cls=None, reset_cls=ResetRequest,
) -> FastAPI
```

- `create_app` returns `create_fastapi_app(...)` unless `ENABLE_WEB_INTERFACE` is `true`, `1` or `yes`. In that case it mounts the Gradio UI at `/web`.
- `create_fastapi_app` sets the OpenAPI `info.version` to `"1.0.0"`, which the `--url` validator reads.

**Concurrency.**
- `max_concurrent_envs` and `concurrency_config` are mutually exclusive; passing both raises `ValueError`. The default is 1.
- If `max_concurrent_envs > 1` and the env class lacks `SUPPORTS_CONCURRENT_SESSIONS = True`, construction raises `ConcurrencyConfigurationError`. A plain factory function gets instantiated once for this check.
- The server keeps one environment instance per WebSocket session, plus a 32-worker thread pool for sync `reset` and `step`.
- OpenEnv's own envs read the limit from an env var: `max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))` (`envs/echo_env/server/app.py` at v0.14.0). TRL's docs tell users to set `MAX_CONCURRENT_ENVS` at or above the generation batch size.

*Source, plus Run: a toy env with `max_concurrent_envs=8` served six concurrent async clients and returned their totals `[0, 1, 2, 3, 4, 5]` with no interference.*

### `EnvClient` (`openenv/core/env_client.py`)

```python
class EnvClient(ABC, Generic[ActT, ObsT, StateT]):
    def __init__(self, base_url: Optional[str] = None, connect_timeout_s: float = 10.0,
                 message_timeout_s: float = 60.0, max_message_size_mb: float = 100.0,
                 websocket_ping_interval_s: Optional[float] = 20.0,
                 websocket_ping_timeout_s: Optional[float] = 20.0,
                 provider=None, mode: Optional[str] = None,
                 headers: Optional[dict[str, str]] = None): ...

    @abstractmethod
    def _step_payload(self, action: ActT) -> dict: ...
    @abstractmethod
    def _parse_result(self, payload: dict) -> StepResult[ObsT]: ...
    @abstractmethod
    def _parse_state(self, payload: dict) -> StateT: ...

    # Public methods are annotated `-> Any`: an awaitable in async code, a concrete value in sync code.
    # The typed returns are on the private coroutines they dispatch to:
    def reset(self, **kwargs) -> Any            # async def _reset_async(**kwargs) -> StepResult[ObsT]
    def step(self, action: ActT, **kwargs) -> Any  # async def _step_async(action, **kwargs) -> StepResult[ObsT]
    def state(self) -> Any                      # async def _state_async() -> StateT
    def sync(self) -> SyncEnvClient                 # `with Client(base_url=...).sync() as c:`
    async def __aenter__ / __aexit__
    @classmethod from_docker_image(...), from_env(...)

@dataclass
class StepResult(Generic[ObsT]):
    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
    metadata: Optional[dict] = None
```

`base_url` can be `http(s)://` or `ws(s)://`. The client derives `<ws_url>/ws`. Since 0.12.0, a dropped socket in mid-episode raises `SessionLostError` instead of silently reconnecting. *Source: `env_client.py`, and the TRL OpenEnv doc's "Long runs against a Space" section.*

**Typed client shape.** This follows the `openenv init` template at `openenv/cli/templates/openenv_env/client.py`.

```python
class RecruitClient(EnvClient[RecruitAction, RecruitObservation, RecruitState]):
    def _step_payload(self, action): return action.model_dump(exclude={"metadata"})
    def _parse_result(self, p):
        obs = RecruitObservation(**p.get("observation", {}), done=p.get("done", False), reward=p.get("reward"))
        return StepResult(observation=obs, reward=p.get("reward"), done=p.get("done", False), metadata=p.get("metadata"))
    def _parse_state(self, p): return RecruitState(**p)
```

The server's `serialize_observation` lifts only `reward` and `done` to the top level of the payload. `metadata` stays inside `observation`, and the top-level `metadata` was `null`. For example, `/reset` returned `{"observation": {"metadata": {}, "total": 0}, "reward": 0.0, "done": false, "metadata": null}`. Parse `metadata` from the observation dict. I ran this pattern against a toy server, both sync and async. *Run.*

---

## 2. `openenv.yaml` and the validator

### Schema the validator reads

`openenv/validation/parsers/openenv_yaml.py` and `openenv/validation/manifest.py`. *Source, 0.14.0.*

The validator reads only these keys:
- top-level `name` (required) and `version`;
- the `validation:` mapping, which is required.

Every other top-level key is ignored by `validate`: `spec_version`, `type`, `runtime`, `app`, `port`, `entry_point`, `space_url`, `tasks`. Other CLI commands do read some of them:
- `openenv serve` reads `app` (`module:attr`), `runtime` (default `fastapi`) and `port` (default 8000).
- `openenv push` reads `name` and `variables:`.

*Source: `cli/commands/serve.py`, `cli/commands/push.py`.*

`validation:` maps onto `NormalizedManifest`, schema 1, where every model uses `extra="forbid"`:

```yaml
validation:
  reward:                     # required
    range: [0.0, 1.0]         # default (0,1); must be strictly increasing
    oracle_tolerance: 0.0     # >=0; policy v1 max 0.1
    floor_margin: 0.1         # REQUIRED; policy v1 min 0.1
    variance_tolerance: ...   # required iff capabilities.llm_judged
  resources:                  # required
    cpu: 1.0                  # >0
    memory_mb: 1024           # >0
    disk_mb: 512              # >0
    episode_timeout_s: 60.0   # >0; policy v1 max 3600
    gpus: 0                   # optional
  capabilities:               # required
    verifier: {kind: reward_channel}   # REQUIRED; or {kind: script, entry: path}
    oracle: {form: injected_state|script, location: path}   # optional; absence fails semantic.oracle_max
    set_state: false          # must be true if oracle.form == injected_state
    llm_judged: false         # true requires a top-level validation.judge pin
    rubric_tree: false
    task_api: false
    canaries: path
    declared_tools: [...]
    declared_task_count: {split: n}
  types:                      # required
    tags: [triage]            # at least one non-empty tag
  network: {mode: public|no-network|allowlist, allowed_hosts: [...]}   # optional
  task_distribution: {generator_seed, dataset_version, generator_model} # optional
  judge: {model, version, params}   # only with llm_judged
  execution: {...}            # optional; its presence switches to manifest schema 2 (runtime validation)
```

### Minimal real examples from the OpenEnv repo

Both files are identical at `v0.14.0` and at `main` (SHA-1 compared, 2026-11-25).

`envs/echo_env/openenv.yaml`, a served environment like ours:

```yaml
spec_version: 1
name: echo_env
version: 0.1.0
type: space
runtime: fastapi
app: server.app:app
port: 8000
validation:
  reward:
    range: [0.0, 1.0]
    oracle_tolerance: 0.0
    floor_margin: 0.5
  resources:
    cpu: 1.0
    memory_mb: 1024
    disk_mb: 512
    episode_timeout_s: 60.0
  capabilities:
    verifier:
      kind: reward_channel
    declared_tools: [echo_message, echo_with_length]
  types:
    tags: [demo]
```

`tests/fixtures/validation/served_min_pass/openenv.yaml` is the validator's own passing fixture. It differs from the echo file in four places: it uses `type: env` and `runtime: python`, and it adds `oracle: {form: injected_state, location: oracle/solved_state.json}` and `set_state: true`.

The `openenv init` template, in the 0.14.0 wheel, matches echo_env apart from `floor_margin: 0.1` and having no `declared_tools`. On `main`, the template adds a `validation.execution` block with `kind: openenv_ws`, `probe_path: validation/runtime.json`, `dockerfile: server/Dockerfile` and `context: .`. That block is for runtime validation and is not released yet.

### What `--level static` checks

The static run has three steps. *Source: `validation/runner.py`, `graders/static/manifest.py`, `policies/severity-v1.json`.*
1. Detect the package type by its well-known file: exactly one of `openenv.yaml` or Harbor `task.toml`. Having both is refused as ambiguous, with exit 2.
2. Parse the manifest. A missing or invalid `validation:` block gives `FAIL static.manifest`, with exit 1.
3. Run `static.manifest`, which checks the declared tolerances against policy v1:
   - `oracle_tolerance <= 0.1`
   - `floor_margin >= 0.1`
   - `variance_tolerance <= 0.2`
   - `episode_timeout_s <= 3600`

Policy v1 lists nine other static checks: `reproducible_build`, `image_hygiene`, `layout`, `sbom`, `oci_labels`, `resource_declaration`, `timeout_ceiling`, `dependency_pinning` and `task_distribution_pinning`. In 0.14.0, each reports `SKIP: grader not implemented in this build`.

The verdict is then `WARN`, which exits 0. **Passing static means `static.manifest: PASS` with `Verdict: WARN`, exit 0.**

Package detection and the source digest have three rules:
- `detect_signature` looks for exactly one of `openenv.yaml`, `task.toml` (Harbor) or `task.md`. A `task.md` counts only if it has YAML frontmatter.
- `source_digest` hashes the whole tree except `.git`, `__pycache__`, `.venv`, `.worktrees` and `outputs`.
- A symlink anywhere else in the tree aborts validation with "validation source may not contain symbolic links".

The repo has no `task.toml`, no `task.md` and no symlinks outside `.git`. *Source: `validation/types.py:SignatureKind`, `validation/runner.py:source_digest`.*

Results from this repo's manifest. *Run, on 2026-11-25, with `openenv` 0.14.0.*
- **Current `openenv.yaml`, alone in a directory:** `FAIL static.manifest: openenv.yaml has no validation: block`, exit 1. This confirms the 2026-11-21 audit.
- **A full copy of the worktree** (everything except `.git`), with the block below appended to `openenv.yaml`: `PASS static.manifest`, the nine SKIPs, `Verdict: WARN`, exit 0. The result is identical **with and without `--skip-build`**.

```yaml
validation:
  reward: {range: [0.0, 1.0], oracle_tolerance: 0.0, floor_margin: 0.1}
  resources: {cpu: 1.0, memory_mb: 1024, disk_mb: 512, episode_timeout_s: 300.0}
  capabilities:
    verifier: {kind: reward_channel}
    task_api: true
    declared_task_count: {test: 3}
  types: {tags: [triage]}
```

The `--level runtime` and `--level semantic` levels need policy v2 and a Docker build of the package, and served packages SKIP the semantic oracle checks ("oracle execution is not implemented for served packages"). The task does not require either level.

### What `--url <server>` checks

`openenv/cli/_validation.py:validate_running_environment`. *Source; Run against a toy `create_app` server, all six PASS, exit 0.*

| Criterion | Request | Pass condition |
|---|---|---|
| `openapi_version_available` | `GET /openapi.json` | 200, with `info.version` a string |
| `health_endpoint` | `GET /health` | 200 and `{"status": "healthy"}` |
| `metadata_endpoint` | `GET /metadata` | 200, with string `name` and `description` |
| `schema_endpoint` | `GET /schema` | 200, with dict `action`, `observation` and `state` |
| `mcp_endpoint` | `POST /mcp` with body `{}` | 200, JSON containing `"jsonrpc": "2.0"` (a JSON-RPC error body passes) |
| `mode_endpoint_consistency` | OpenAPI `paths` | if `/reset` exists, `/step` and `/state` must also exist (simulation mode); otherwise none of the three may exist |

`--url` ignores `--level`. It does not open `/ws` and does not run an episode.

The current `api/main.py` would fail four of the six criteria. I inferred this from the code and did not run it.
- It has no `/metadata`, `/schema` or `/mcp`.
- Its `/health` returns `{"status": "ok", ...}` (`api/main.py:159`), but the validator wants `"healthy"`.
- `/openapi.json` and the mode check should pass.

Serving the adapter through `create_app` passes all six with no extra code.

---

## 3. Task API (multiple tasks per environment)

Yes, there is one. It is optional and works by **discovery only**: a task is selected through `reset()` kwargs. *Source: `interfaces.py:TaskProvider`, `docs/source/guides/task-api.md` (identical at v0.14.0 and main), https://huggingface.co/docs/openenv/guides/task-api.*

Define these methods on the `Environment` subclass. There is no inheritance; the protocol is structural, and each method may be sync or async:

```python
def list_splits(self) -> list[Any]                  # "train"/"validation"/"test" strings, dicts or models
def list_tasks(self, split: str) -> list[Any]
def num_tasks(self, split: str) -> int
def get_task(self, split: str, index: int) -> Any   # raise IndexError when out of range -> HTTP 400
def get_task_range(self, split: str, start: Optional[int] = None, stop: Optional[int] = None) -> list[Any]
```

**HTTP routes.** These are HTTP only; `/ws` has no message types for tasks.

| Method | Route | Body |
|---|---|---|
| GET | `/list_environments` | none |
| GET | `/{env_name}/splits` | none |
| POST | `/{env_name}/tasks` | `{"split": ...}` |
| POST | `/{env_name}/num_tasks` | `{"split": ...}` |
| POST | `/{env_name}/task` | `{"split", "index"}` |
| POST | `/{env_name}/task_range` | `{"split", "start", "stop"}` |

- A missing method returns 501.
- `{env_name}` is matched case-insensitively against `create_app(env_name=...)`. When `env_name` is omitted, the class name is used. Pass it explicitly.
- Each task call builds a fresh environment instance, calls the method, then closes the instance. Task methods therefore have to work without `reset()` having run, and must not have side effects.

**Selecting a task.** Name the parameters in `reset()`. For example, `reset(self, seed=None, episode_id=None, task: str = "easy", **kwargs)`.
- The server filters incoming reset keys against the `reset()` signature. Unknown keys are dropped silently, unless the signature has `**kwargs`.
- To publish the reset parameters in `/schema`, pass `reset_cls=RecruitResetRequest(ResetRequest)`. This is only a hint; it is not used for validation.
- Wire forms: `POST /reset {"task": "hard", "seed": 3}`, or `{"type": "reset", "data": {...}}` over `/ws`.

In a toy server, `POST /reset {"task": "hard"}` reached `reset(task=...)`, and `/toy/splits` returned `[{"name": "test", "type": "test"}]`. *Run.*

The manifest's matching declarations are `capabilities.task_api: true` and `declared_task_count: {split: n}`. Runtime validation checks the count against discovery (`runtime.task_declaration_accuracy`). Static validation does not check it.

The core clients do not ship Task API helpers. The guide shows a typed client adding `requests` calls against the HTTP base URL.

---

## 4. Server endpoints and how clients connect

Routes registered by `create_fastapi_app` in 0.14.0. *Source: `http_server.py:register_routes`; Run: toy server OpenAPI path list.*

| Route | Mode | Notes |
|---|---|---|
| `WS /ws` | both | One environment instance per connection. Messages are `{"type": "reset" \| "step" \| "state" \| "close", "data": {...}}`; replies are `observation`, `state` or `error`. Optional `?session_id=` attaches to an HTTP MCP session. |
| `POST /reset`, `POST /step`, `GET /state` | simulation only | **Stateless.** Each request builds a fresh environment instance and closes it afterwards, so a `/step` does not continue the episode from a `/reset`. Use `/ws` for episodes. |
| `GET /schema` | both | `{action, observation, state, reset?, reset_observation?}` as JSON schemas |
| `GET /metadata` | both | `Environment.get_metadata()`, called on a fresh instance |
| `GET /health` | both | `{"status": "healthy"}` |
| `POST /mcp`, `WS /mcp` | both | MCP JSON-RPC (`tools/list`, `tools/call`, `openenv/session/create`, `openenv/session/close`) |
| Task API routes | both | see section 3 |
| `/docs`, `/redoc`, `/openapi.json` | both | FastAPI |
| `/web` | when `ENABLE_WEB_INTERFACE=true` | Gradio UI |
| `WS /harness` | production, harness envs only | RFC 005 |

**Production mode** comes from `mode="production"` or `OPENENV_MODE=production`. It drops `/reset`, `/step` and `/state`.

**Who needs which endpoint.** *Source: grep of the 0.14.0 client code.*
- `EnvClient` and `GenericEnvClient` use **only `/ws`**.
- `AutoEnv` and the container or uv providers poll `GET /health` to wait for readiness.
- `MCPToolClient` uses `/mcp`.
- No client code requests `/schema` or `/metadata`. Those two exist for the `--url` validator, the web UI and humans.

The 2026-11-21 audit said clients expect `/ws`, `/schema` and `/metadata`. Only `/ws` is a client requirement; the other two are validator requirements.

**Stateless HTTP is a breaking change.** The current `api/main.py` keeps a stateful episode across `/reset` and `/step`. The OpenEnv HTTP routes do not. Any HTTP-driven script or test in this repo that assumes `/reset` then `/step` continue one episode has to move to `/ws` or the in-process core. *Source: `reset_handler` and `step_handler` each call `self._env_factory()` and then `_env.close()` in `finally`.*

---

## 5. TRL GRPOTrainer with environments

**Sources:**
- https://huggingface.co/docs/trl/openenv (`docs/source/openenv.md` at `v1.18.2`)
- `trl/trainer/grpo_trainer.py` at `v1.18.2`
- `tests/test_grpo_trainer.py` at `v1.18.2`

### Current integration: `environment_factory`

```python
GRPOTrainer(model, reward_funcs=None, args=None, train_dataset=None, eval_dataset=None,
            processing_class=None, reward_processing_classes=None, callbacks=None,
            optimizers=(None, None), quantization_config=None, peft_config=None,
            tools=None, rollout_func=None,
            environment_factory: EnvironmentFactory | dict[str, EnvironmentFactory] | None = None)
```

**The contract for the environment class**, which is not an OpenEnv type: TRL wraps any class with the shape below.
- `__init__(self)` takes no arguments.
- `reset(self, **kwargs) -> str | None` is required. It receives the dataset row's columns as kwargs, and a returned string is appended to the last user message.
- **Every public method other than `reset` and `get_reward` becomes a tool.** Each needs typed arguments and a Google-style docstring with an `Args:` section. TRL advises against a generic `step(action)`.
- An optional `get_reward(self) -> float`, sync or async, makes the environment own the reward. `reward_funcs` then becomes optional.
- An exception raised inside a tool is sent back to the model as the tool response.

**Rollout behavior.**
- The trainer runs the multi-turn loop itself: generate, parse tool calls, call the tools, append the results, generate again.
- The loop ends when a turn has no tool call, when the whole conversation reaches `max_completion_length` (tool output counts), or when it reaches `max_tool_calling_iterations` (default `None`).
- **Instances are pooled and reused.** The trainer probes one instance at construction and creates more only when a batch needs more concurrent instances. The docs page says "one instance per generation"; the source shows pooling. Either way, each rollout in a batch gets its own instance, and each instance should hold one OpenEnv client.

**Minimal shape** (adapted from the docs quick start and `examples/grpo_echo`; I did not run it here):

```python
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

ENV_URL = "http://127.0.0.1:7860"

class RecruitToolEnv:
    def __init__(self):
        self.client = None
        self.reward = 0.0

    def reset(self, **kwargs) -> str | None:
        if self.client is not None:
            self.client.close()
        self.client = RecruitClient(base_url=ENV_URL).sync()   # one socket per episode
        result = self.client.reset(task=kwargs.get("task", "easy"), seed=kwargs.get("seed"))
        self.reward = 0.0
        return render(result.observation)

    def reveal_github(self, candidate_id: str) -> str:
        """
        Reveal a candidate's GitHub evidence.

        Args:
            candidate_id: The candidate to inspect.

        Returns:
            The revealed evidence.
        """
        r = self.client.step(RecruitAction(kind="reveal_github", candidate_id=candidate_id))
        self.reward = r.reward or 0.0
        if r.done:
            raise ValueError("Episode over.")
        return render(r.observation)

    def get_reward(self) -> float:
        return self.reward

dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": "Triage the pipeline."}]] * 16,
                             "task": ["easy"] * 16})
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
    environment_factory=RecruitToolEnv,
    args=GRPOConfig(per_device_train_batch_size=2, num_generations=2, max_steps=3,
                    max_completion_length=1024, bf16=False, report_to="none"),
)
trainer.train()
```

### Version requirements

These come from per-tag source checks of `grpo_trainer.py`.

| Feature | First TRL tag with it | Released |
|---|---|---|
| `environment_factory` | v0.29.0 (absent in v0.28.0) | 2026-02-25 |
| `get_reward` (env-owned reward), dict factory (multi-env) | v1.8.0 (absent in v1.7.0) | 2026-07-09 |
| `train_dataset` optional with `environment_factory` (needs `max_steps`) | v1.9.0 | 2026-07-21 |

**transformers floors.** At runtime, `GRPOTrainer` requires:
- `transformers>=5.0.0` for `tools`;
- `transformers>=5.2.0` for `environment_factory`;
- `jmespath`, if `transformers<5.13`.

TRL's package metadata only requires `transformers>=4.56.2`, so pin `transformers>=5.2` yourself. Latest on PyPI on 2026-11-25: `transformers` 5.24.1 and `torch` 2.15.0, both `requires_python>=3.11`.

**Recommended pin:** `trl>=1.9,<2`, `transformers>=5.2`.

At v1.18.2, `GRPOTrainer` has a `transformers>=5.23` branch. It only mirrors `gradient_checkpointing_kwargs` to `activation_checkpointing_kwargs` for PEFT with DeepSpeed ZeRO-3 (`grpo_trainer.py:579`), so it does not affect a plain run on 5.24.1.

The `environment_factory` feature logs a warning that it is experimental; set `TRL_EXPERIMENTAL_SILENCE=1` to suppress it. *Source: `grpo_trainer.py:677`.*

### Qwen2.5-0.5B-Instruct and CPU

- **Tool-call parsing works with this model.** `add_response_schema` matches chat templates by exact string equality. The model's Hub `tokenizer_config.json` `chat_template` is byte-identical to TRL v1.18.2's `trl/chat_templates/qwen2_5.jinja`: both are 2507 characters and compare equal. *Run.* That template maps to `qwen3_schema` and `qwen3_template`, the `<tool_call>{json}</tool_call>` format.
- An unrecognized template raises `ValueError: Unrecognized chat template, failed to add response schema`. Avoid fine-tunes with modified templates.
- **CPU is possible but unverified here.**
  - TRL's own CI test `test_train_with_environment_factory` runs `environment_factory` on a tiny Qwen3 model (`trl-internal-testing/tiny-Qwen3MoeForCausalLM`), with `max_steps=5` and `num_generations=3`. It has no accelerator marker, and it patches `generate` with canned tool calls.
  - `GRPOConfig` defaults `bf16=True` unless `fp16` is set, so set `bf16=False` explicitly for CPU. `use_vllm` defaults to False.
  - `beta` defaults to 0.0, so no reference model is loaded.
  - Rough memory: 0.5B params in fp32 is about 2 GB of weights, plus about 4 GB of AdamW state, plus gradients and activations: about 8 to 10 GB in total. This is an estimate, not a measurement.
  - **I did not run it here**, to protect the shared 16 GB host (7 GB available, swap nearly full, RAM-backed `/tmp`).
  - Expect minutes per step on CPU with `num_generations=2` and short completions. A real end-to-end run belongs in Colab or on a GPU.
- A 0.5B base model will rarely call tools correctly at first. TRL's docs recommend an SFT warm-up when groups get identical rewards (see "Make sure groups differ").

### Alternatives

- `rollout_func`: you own generation and return `prompt_ids`, `completion_ids` and `logprobs`.
- `AsyncGRPOTrainer` plus OpenEnv Harbor, for black-box agents; it needs `openenv>=0.7.0` and vLLM.

Neither is needed for a minimal example.

### Server concurrency for training

From the TRL docs: `max_concurrent_envs` must be at least `generation_batch_size`, which defaults to `per_device_train_batch_size × gradient_accumulation_steps`. Old sessions are released asynchronously, so leave headroom and open a fresh client in each `reset()`.

---

## 6. Python and dependency constraints

**`openenv` 0.14.0 hard dependencies** (*Source: PyPI `requires_dist`*):
- `fastapi>=0.116`, `pydantic>=2`, `uvicorn>=0.24`, `requests`, `httpx>=0.28.1`, `websockets>=15.0.1`
- `typer`, `rich`, `pyyaml`, `tomli`, `tomli-w`, `jsonschema>=4.20`, `packaging`, `typing_extensions`
- `huggingface_hub>=1.29.0`, `openai>=2.7.2`, `fastmcp>=3,<4`, **`gradio>=4.0.0`**

**Gradio is a hard dependency, not an extra.**
- A clean install resolved 113 packages and took 319 MB.
- `gradio` 6.32.0 alone is 82 MB.
- `import openenv.core.env_server` loads `gradio` (`'gradio' in sys.modules` is True) and takes about 2 s.

*Run.* This is why OpenEnv must stay an optional extra (`recruitenv[openenv]`) and why the core must not import it.

- **No `torch`.** It is not in `requires_dist` and was not installed. *Run.*
- **Python.** `openenv` needs `>=3.10`. TRL, transformers and torch need `>=3.11`. The repo's `requires-python = ">=3.11"` is compatible with all of them.
- **Optional extras on `openenv`:** `daytona`, `aca`, `modal`, `kubernetes`, `novita`, `inspect`, `security` (`fastapi-guard`), and `harbor` (Python >= 3.12).
- **Dependency drift to watch.** `main` is migrating to FastMCP 4, while 0.14.0 pins `fastmcp<4`. This is a reason to cap `openenv<0.15`.

---

## 7. Corrections to the 2026-11-21 audit notes

| Audit claim | Status on 2026-11-25 |
|---|---|
| `openenv` released 0.14.0 | **Confirmed.** Still the latest on 2026-11-25. |
| `openenv-core` is the old name | **Confirmed.** It stopped at 0.3.0, and `openenv_core` is a deprecation shim inside `openenv`. Both distributions ship the same top-level packages, so replace one with the other rather than installing both. |
| The validator wants a `validation:` block | **Confirmed**, and the block now has a precise schema (above). Static passes as `WARN` with exit 0. Only `name` and `validation` are read; `tasks`, `entry_point` and `space_url` are ignored. |
| Clients expect `/ws`, `/schema`, `/metadata` | **Partly wrong.** `EnvClient` uses only `/ws`; `AutoEnv` and the providers poll `/health`. `/schema` and `/metadata`, together with `/openapi.json`, `/health` and `POST /mcp`, are what `openenv validate --url` checks. |
| Repo at `meta-pytorch/OpenEnv` | **Moved** to `openenv/OpenEnv`. The old URL redirects. |
| (new) HTTP `/reset`, `/step` and `/state` are stateless | Each request uses a fresh environment instance, so episodes need `/ws` or the in-process core. |
| (new) `environment_factory` floors | Basic support since TRL v0.29.0; `get_reward` and multi-env since v1.8.0; no-dataset since v1.9.0. Needs `transformers>=5.2`. |
| (new) Concurrency guard | `max_concurrent_envs > 1` without `SUPPORTS_CONCURRENT_SESSIONS = True` raises `ConcurrencyConfigurationError`. |

## Unverified items

- A CPU GRPO run with Qwen2.5-0.5B-Instruct: not run here. The evidence is the CI test, the template match and a memory estimate.
- That the current `api/main.py` fails four `--url` criteria: inferred from its routes and its `/health` body, not run.
- What 0.15.0 will change: `main` has unreleased changes to the server, the client and MCP.
