# RecruitEnv

OpenEnv-compliant RL environment simulating candidate pipeline triage.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload
```

## Running the Baseline

```bash
export OPENAI_API_KEY=your_key_here
python baseline/run_baseline.py
```

## Docker

```bash
docker build -t recruitenv .
docker run -p 7860:7860 recruitenv
```

## Tests

```bash
pytest tests/ -v
```
