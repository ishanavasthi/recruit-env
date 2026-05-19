# 11 — Deployment (Docker + Hugging Face Spaces)

How the project actually runs in the wild. Three layers: the Dockerfile, the `openenv.yaml` manifest, and the HF Spaces front-matter inside `README.md`.

## The Dockerfile (`Dockerfile`)

```dockerfile
FROM python:3.11-slim

# Create non-root user (HF Spaces requirement)
RUN useradd --create-home appuser

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project
COPY . .

# Switch to non-root user
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

Walked through:

### `FROM python:3.11-slim`

- Python 3.11 is the project's required version (`pyproject.toml: requires-python = ">=3.11"`).
- `-slim` cuts ~700MB of OS tooling vs `python:3.11`. Trade-off: you have to `apt-get install` any system tool you need (like `curl` below).

### Non-root user

```dockerfile
RUN useradd --create-home appuser
...
USER appuser
```

**HF Spaces requirement.** Containers run as a non-root user there. Build sets things up; the final `USER appuser` switch happens *after* `pip install` so root can write to system site-packages.

### Layer caching trick

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
```

`requirements.txt` is copied **separately** before the rest of the source. Docker caches each layer; if you change a `.py` file but not `requirements.txt`, the heavy `pip install` step is reused from cache. Standard Python-Docker hygiene.

`--no-cache-dir` skips pip's own cache to keep the image small.

### Healthcheck

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1
```

- `--start-period=10s` — gives uvicorn 10 seconds to bind before the first check.
- `--interval=30s` — pings `/health` every 30 seconds.
- `--retries=3` — needs 3 consecutive failures to declare unhealthy.
- `curl -f` — `-f` makes curl exit non-zero on HTTP errors.

This is what HF Spaces and `docker ps` use to flag a container as `(healthy)` vs `(unhealthy)`.

### CMD

```dockerfile
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

- **Exec form (`[...]`)** so the process becomes PID 1 directly (no extra shell). Means SIGTERM is delivered to uvicorn cleanly on container stop.
- `--host 0.0.0.0` is mandatory inside containers (loopback-only won't be reachable from the host).
- `--port 7860` is the HF Spaces convention.
- `--workers 1` because the env is a **singleton**. Multiple workers would each get their own copy of the env and clients would land on whichever, breaking episode continuity.

### Image build

```bash
docker build -t recruitenv .
docker run -p 7860:7860 recruitenv
```

## `docker-compose.yml`

```yaml
services:
  recruitenv:
    build: .
    ports:
      - "7860:7860"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}
    restart: unless-stopped
```

For local development. Forwards two env vars (the API key + an optional model override) into the container. `restart: unless-stopped` so the container survives reboots in dev environments.

Note: the **server itself** doesn't use `OPENAI_API_KEY` — only the external `baseline/run_baseline.py` script does. The env var is forwarded into the container anyway so you can `docker exec` in and run the baseline from inside the container too.

## `openenv.yaml` — OpenEnv Manifest

```yaml
name: RecruitEnv
version: "0.1.0"
description: >
  OpenEnv-compliant RL environment simulating candidate pipeline triage.
  An agent reviews synthetic candidate profiles and makes shortlisting
  decisions against a job description.

entry_point: api.main:app
port: 7860
space_url: "https://huggingface.co/spaces/heyavasthi/recruitenv"

tasks:
  - id: easy
    description: Screen 10 candidates, pick top 3
  - id: medium
    description: Conflicting signals evaluation (15 candidates, pick 5)
  - id: hard
    description: Batch triage 20 candidates, tight budget (pick 5)
```

OpenEnv-core uses this file to discover what the env looks like *without* actually running it. Critical fields:

- **`entry_point`** — `api.main:app` (the FastAPI app object). When `openenv-core` validates the env, it imports this and confirms it's an ASGI app.
- **`port`** — must match what the Dockerfile and uvicorn say (7860).
- **`tasks`** — the manifest's view of available tasks (matches `TASK_REGISTRY` in the code).
- **`space_url`** — the deployed Space URL. This is what `client.py:RecruitEnvClient` uses by default.

## HF Spaces Front-Matter (in `README.md`)

```yaml
---
title: RecruitEnv
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

This YAML block lives at the very top of `README.md`. Hugging Face Spaces reads it to configure the deployment:

| Field           | Effect                                                      |
| --------------- | ----------------------------------------------------------- |
| `title`         | Displayed on the Space landing page                         |
| `emoji`         | Card icon                                                   |
| `colorFrom/To`  | Gradient on the card                                        |
| `sdk: docker`   | Tells HF to build/run a Dockerfile (vs Gradio/Streamlit)    |
| `app_port: 7860`| Which container port to expose to the public                |
| `pinned: false` | Don't pin to the user's profile                             |

The deployment flow: push to the HF git remote → HF reads the front-matter → builds the Dockerfile → publishes the container → exposes port 7860 at `https://heyavasthi-recruitenv.hf.space`.

## The Three Inference Entrypoints

The repo has three different ways to "run an agent against the env" — easy to confuse them:

| Script                       | Calls LLM?           | Talks to                  | Purpose                                            |
| ---------------------------- | -------------------- | ------------------------- | -------------------------------------------------- |
| `api/main.py:_run_rule_baseline` | No (pure heuristic) | Local env directly        | Powers `GET /baseline` so judges can ping it       |
| `baseline/run_baseline.py`   | Yes (OpenAI SDK)     | localhost:7860 over HTTP  | The official "baseline inference script"          |
| `inference.py`               | Yes (HF Router)      | HF Space URL over HTTP    | Standalone validator-format runner with `[START]/[STEP]/[END]` log format |

`inference.py` is what an external automated evaluator runs against the deployed Space. It expects env vars `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, hits `ENV_URL = https://heyavasthi-recruitenv.hf.space`, and prints structured per-step logs in the exact format the OpenEnv judging pipeline parses.

## `client.py` — The OpenEnv-Core Wrapper

```python
from openenv_core import GenericEnvClient

class RecruitEnvClient(GenericEnvClient):
    def __init__(self, base_url: str = "https://heyavasthi-recruitenv.hf.space"):
        super().__init__(base_url=base_url)
```

Lets external Python callers do:

```python
from client import RecruitEnvClient
env = RecruitEnvClient()
obs = env.reset(task_id="easy", seed=42)
obs, reward, done, info = env.step(action_dict)
```

This wraps the HTTP plumbing in a Gym-style API. `openenv-core`'s `GenericEnvClient` knows how to translate `reset`/`step`/`state` Python calls into the corresponding HTTP requests.

## `validate_submission.sh`

The repo's pre-submission validator (shell script, ~6KB). Runs through the OpenEnv compliance gate:

- Builds the Docker image
- Starts a container
- Hits `/health`, `/reset`, `/step`, `/grader`, `/baseline`
- Verifies `reset(seed=42)` twice gives identical responses (determinism)
- Verifies all three graders produce scores in `[0, 1]`

If anything fails, exits non-zero. This is what you run before submitting; if it passes locally, judging should also pass.

## Deployment Checklist (for the Interview)

| Requirement                | Met by                                                  |
| -------------------------- | ------------------------------------------------------- |
| Containerised              | Dockerfile (Python 3.11-slim, non-root, healthcheck)    |
| Port 7860                  | EXPOSE + CMD + openenv.yaml + HF front-matter all agree |
| HF Spaces compatible       | `sdk: docker` + non-root user + `app_port: 7860`        |
| OpenEnv spec compliant     | `openenv.yaml` + typed models + reset/step/state        |
| 3+ tasks with graders      | TASK_REGISTRY + GRADER_REGISTRY                         |
| Baseline reproducible      | `baseline/run_baseline.py` + seed=42 + baseline_scores.json |
| Tests pass                 | 35 pytest tests                                         |
| Healthcheck                | `curl -f /health` every 30s                             |

## Soundbites

- *"Single-worker uvicorn because the environment is a singleton. Multiple workers would mean each gets its own env and clients land on whichever — episode state would fragment."*
- *"Non-root user, `sdk: docker` in the README front-matter, port 7860 everywhere — all HF Spaces requirements."*
- *"Three inference scripts: in-server rule-based baseline at `GET /baseline`, OpenAI-based baseline in `baseline/run_baseline.py`, and an HF-router runner at `inference.py` that uses the validator's log format."*
- *"`validate_submission.sh` is the pre-submission gate — Docker build, full HTTP smoke test including a `reset` determinism check and a `/grader` score-range check."*
