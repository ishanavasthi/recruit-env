"""OpenAI-powered baseline agent for RecruitEnv.

Connects to the RecruitEnv API and plays a full episode using an LLM
to choose actions.  Requires OPENAI_API_KEY in the environment.

Usage: python baseline/run_baseline.py
"""

from __future__ import annotations

import os

API_BASE = os.getenv("RECRUITENV_API_BASE", "http://localhost:7860")


def run_baseline(task_id: str = "easy", seed: int = 42) -> None:
    """Run one full episode with the OpenAI-powered baseline agent.

    Steps:
    1. POST /reset to start a new episode.
    2. Loop: ask the LLM what action to take, POST /step.
    3. Print final reward when episode ends.
    """
    raise NotImplementedError


if __name__ == "__main__":
    run_baseline()
