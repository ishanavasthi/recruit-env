"""End-to-end test of inference.py's episode loop against the real API app.

A scripted chooser stands in for the LLM, so no API key is needed.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

import inference
from api.main import app
from env import rubric
from env.models import JobDescription
from env.profile_factory import fill_quotas


def _scripted_chooser():
    """Reveal every rubric source, then decide by the published rubric.

    Sends one invalid action first to exercise the rejection path.
    """
    sent_invalid = False
    actions = {
        "github": ("check_platform", "platform"),
        "leetcode": ("check_platform", "platform"),
        "kaggle": ("check_platform", "platform"),
        "experience": ("read_resume_section", "section"),
        "skills": ("read_resume_section", "section"),
    }

    def choose(observation: dict, history: list[str]) -> dict:
        nonlocal sent_invalid
        if not sent_invalid:
            sent_invalid = True
            return {"type": "score_dimension", "candidate_id": "x", "dimension": "technical", "score": 1}
        revealed = observation["revealed_data"]
        for c in observation["candidates_summary"]:
            have = revealed.get(c["id"], {})
            for src in rubric.SOURCES:
                if src not in have.get("platforms", {}) and src not in have.get("resume_sections", {}):
                    kind, key = actions[src]
                    return {"type": kind, "candidate_id": c["id"], key: src}
        jd = JobDescription(**observation["job_description"])
        est = {c["id"]: rubric.estimate(revealed.get(c["id"]), jd).total for c in observation["candidates_summary"]}
        labels = fill_quotas(est, observation["quotas"])
        for cid, label in labels.items():
            if cid not in observation["decisions_made"]:
                return {"type": "make_decision", "candidate_id": cid, "decision": label}
        raise AssertionError("episode should have ended")

    return choose


@pytest.mark.parametrize("task", ["easy", "medium"])
def test_inference_reports_the_grader_score(task: str) -> None:
    """The reported score is /grader's. Medium's budget fits a full read."""
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
    if task == "medium":
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
