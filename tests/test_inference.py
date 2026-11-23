"""End-to-end test of inference.py's episode loop against the real API app.

A scripted chooser stands in for the LLM, so no API key is needed.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

import inference
from api.main import app


def _scripted_chooser():
    """Check GitHub, decide on stars, and send one invalid action first."""
    sent_invalid = False

    def choose(observation: dict, history: list[str]) -> dict:
        nonlocal sent_invalid
        if not sent_invalid:
            sent_invalid = True
            return {"type": "score_dimension", "candidate_id": "x", "dimension": "technical", "score": 1}
        revealed = observation["revealed_data"]
        for c in observation["candidates_summary"]:
            cid = c["id"]
            if cid in observation["decisions_made"]:
                continue
            github = revealed.get(cid, {}).get("platforms", {}).get("github")
            if github is None:
                return {"type": "check_platform", "candidate_id": cid, "platform": "github"}
            stars = github["stars_received"]
            decision = "shortlist" if stars > 200 else "reject" if stars < 20 else "hold"
            return {"type": "make_decision", "candidate_id": cid, "decision": decision}
        raise AssertionError("episode should have ended")

    return choose


@pytest.mark.parametrize("task", ["easy", "medium"])
def test_inference_reports_the_grader_score(task: str) -> None:
    async def run() -> dict:
        env = inference.EnvHTTPClient("http://test", transport=httpx.ASGITransport(app=app))
        try:
            return await inference.run_episode(env, _scripted_chooser(), task, seed=42)
        finally:
            await env.close()

    result = asyncio.run(run())
    assert result["score"] == result["grade"]["score"]
    assert sum(result["rewards"]) == pytest.approx(result["score"])
    assert result["rewards"][-1] == result["score"]
    assert result["score"] >= 0.99
    assert result["success"] is True


def test_build_user_prompt_contains_revealed_values() -> None:
    observation = {
        "steps_remaining": 10,
        "job_description": {"role": "ML Engineer"},
        "candidates_summary": [{"id": "candidate_001", "name": "A"}],
        "revealed_data": {"candidate_001": {"platforms": {"github": {"stars_received": 321}}}},
        "decisions_made": {},
    }
    prompt = inference.build_user_prompt(1, observation, [])
    assert '"stars_received":321' in prompt
    assert "ML Engineer" in prompt


def test_parse_action() -> None:
    assert inference.parse_action('```json\n{"type": "make_decision"}\n```') == {"type": "make_decision"}
    assert inference.parse_action("I pick {\"a\": 1} now") == {"a": 1}
    assert inference.parse_action("nothing") is None
