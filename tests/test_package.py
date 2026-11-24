"""The core is a plain library: no web, OpenEnv or model-client imports."""

from __future__ import annotations

import subprocess
import sys

import recruitenv


def test_core_imports_no_web_or_openenv():
    code = (
        "import sys, recruitenv, recruitenv.trajectory, recruitenv.tools, recruitenv.policies, "
        "recruitenv.fairness, recruitenv.tasks\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'fastapi', 'starlette', 'uvicorn', 'openenv', 'openai', 'httpx', 'gradio'})\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_version_matches_pyproject():
    import tomllib
    from pathlib import Path

    meta = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    assert meta["project"]["version"] == recruitenv.__version__
    assert set(meta["project"]["dependencies"]) == {"pydantic>=2", "numpy"}
