"""inference.py and the decide-only baseline: episode loop, no silent fallback,
retries, logging, and the websocket driver. A fake model stands in for the API."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import inference
from baseline import run_baseline
from recruitenv import rubric
from recruitenv.models import JobDescription
from recruitenv.profile_factory import fill_quotas
from recruitenv.trajectory import read_log, replay


def rubric_chooser(send_invalid_first: bool = True):
    """Reveal every rubric source, then decide by the published rubric."""
    sent = not send_invalid_first
    actions = {
        "github": ("check_platform", "platform"), "leetcode": ("check_platform", "platform"),
        "kaggle": ("check_platform", "platform"), "experience": ("read_resume_section", "section"),
        "skills": ("read_resume_section", "section"),
    }

    def choose(observation, history, usage):
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "score_dimension", "candidate_id": "x"}
        revealed = observation["revealed_data"]
        for c in observation["candidates_summary"]:
            have = revealed.get(c["id"], {})
            for src in rubric.SOURCES:
                if src not in have.get("platforms", {}) and src not in have.get("resume_sections", {}):
                    kind, key = actions[src]
                    return {"type": kind, "candidate_id": c["id"], key: src}
        jd = JobDescription(**observation["job_description"])
        est = {c["id"]: rubric.estimate(revealed.get(c["id"]), jd).total for c in observation["candidates_summary"]}
        for cid, label in fill_quotas(est, observation["quotas"]).items():
            if cid not in observation["decisions_made"]:
                return {"type": "make_decision", "candidate_id": cid, "decision": label}
        raise AssertionError("episode should have ended")

    return choose


@pytest.mark.parametrize("task", ["easy", "medium"])
def test_local_episode_reports_grader_score_and_logs(task, tmp_path):
    log = tmp_path / "ep.jsonl"
    driver = inference.LocalDriver(log, model="fake")
    result = inference.run_episode(driver, rubric_chooser(), task, seed=42, verbose=False)
    driver.close()
    assert result["incomplete"] is False
    assert result["score"] == result["grade"]["score"] == result["rewards"][-1]
    assert result["usage"]["rejections"] == 1
    assert result["score"] >= 0.99 and result["success"] is True
    assert read_log(log)[-1]["grade"]["score"] == result["score"]
    replay(log)


def test_remote_driver_over_websocket():
    pytest.importorskip("openenv")
    import threading

    import uvicorn

    from recruitenv.server import app

    config = uvicorn.Config(app, host="127.0.0.1", port=7893, log_level="warning")
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    import time
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        driver = inference.RemoteDriver("http://127.0.0.1:7893")
        result = inference.run_episode(driver, rubric_chooser(), "medium", seed=42, verbose=False)
        driver.close()
    finally:
        server.should_exit = True
        th.join(timeout=5)
    local = inference.run_episode(inference.LocalDriver(), rubric_chooser(), "medium", seed=42, verbose=False)
    assert result["incomplete"] is False
    assert result["score"] == local["score"]
    assert result["grade"]["checks"] == json.loads(json.dumps(local["grade"]["checks"]))


class FakeLLM:
    """OpenAI-shaped client. ``script`` items: an action dict, "error", "cutoff" or "junk"."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0) if self.script else "junk"
        if item == "error":
            raise RuntimeError("provider down")
        if item == "cutoff":
            text, reason = "", "length"
        elif item == "junk":
            text, reason = "I think candidate 3 looks strong", "stop"
        else:
            text, reason = json.dumps(item), "stop"
        msg = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=reason)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))


def _first_action(task="easy", seed=1):
    d = inference.LocalDriver()
    obs = d.reset(task, seed)
    return {"type": "check_platform", "candidate_id": obs["candidates_summary"][0]["id"], "platform": "github"}


def test_errors_and_cutoffs_are_retried_with_a_growing_cap():
    action = _first_action()
    llm = FakeLLM(["error", "cutoff", "junk", action])
    chooser = inference.ModelChooser(llm, "m", max_tokens=1000, backoff_s=0)
    usage = inference.Usage()
    d = inference.LocalDriver()
    obs = d.reset("easy", 1)
    assert chooser(obs, [], usage) == action
    assert (usage.api_errors, usage.length_cutoffs, usage.parse_failures, usage.calls) == (1, 1, 1, 3)
    assert [c["max_tokens"] for c in llm.calls] == [1000, 1000, 2000, 2000]


def test_no_fallback_episode_is_incomplete():
    llm = FakeLLM(["cutoff"] * 10)
    chooser = inference.ModelChooser(llm, "m", max_tokens=1000, max_tokens_ceiling=4000, backoff_s=0)
    result = inference.run_episode(inference.LocalDriver(), chooser, "easy", 1, verbose=False)
    assert result["incomplete"] is True and "cut off" in result["reason"]
    assert "score" not in result
    assert [c["max_tokens"] for c in llm.calls] == [1000, 2000, 4000, 4000]


def test_endless_rejections_end_incomplete():
    def bad(observation, history, usage):
        return {"type": "make_decision", "candidate_id": "nobody", "decision": "hold"}
    result = inference.run_episode(inference.LocalDriver(), bad, "easy", 1, verbose=False)
    assert result["incomplete"] is True and "rejected" in result["reason"]


def test_build_user_prompt_contains_revealed_values():
    observation = {
        "steps_remaining": 10,
        "job_description": {"role": "ML Engineer"},
        "candidates_summary": [{"id": "candidate_001", "name": "A"}],
        "revealed_data": {"candidate_001": {"platforms": {"github": {"stars_received": 321}}}},
        "decisions_made": {},
    }
    prompt = inference.build_user_prompt(1, observation, [])
    assert '"stars_received":321' in prompt and "ML Engineer" in prompt


def test_parse_action():
    assert inference.parse_action('```json\n{"type": "make_decision"}\n```') == {"type": "make_decision"}
    assert inference.parse_action('I pick {"a": 1} now') == {"a": 1}
    assert inference.parse_action("nothing") is None


# -- decide-only baseline ---------------------------------------------------


def _rubric_answer(task, seed):
    from recruitenv.environment import RecruitmentEnvironment

    env = RecruitmentEnvironment()
    obs = env.reset(task, seed).model_dump(mode="json")
    return run_baseline.rubric_decisions(run_baseline.gather(env, obs))


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_decide_only_baseline(task, tmp_path):
    llm = FakeLLM(["cutoff", _rubric_answer(task, 42)])
    log = tmp_path / f"{task}.jsonl"
    r = run_baseline.run_task(llm, "m", task, 42, log_path=log, backoff_s=0)
    assert r["incomplete"] is False and r["usage"]["length_cutoffs"] == 1
    assert r["score"] == r["grade"]["score"]
    if task in ("easy", "medium"):
        assert r["score"] >= 0.99
    replay(log)


def test_decide_only_has_no_fallback():
    r = run_baseline.run_task(FakeLLM(["junk"] * 10), "m", "medium", 42, backoff_s=0)
    assert r["incomplete"] is True and "unusable" in r["reason"]


def test_api_errors_have_their_own_budget():
    action = _first_action()
    llm = FakeLLM(["error"] * 6 + [action])
    chooser = inference.ModelChooser(llm, "m", attempts=2, api_attempts=8, backoff_s=0)
    usage = inference.Usage()
    d = inference.LocalDriver()
    assert chooser(d.reset("easy", 1), [], usage) == action
    assert usage.api_errors == 6 and usage.calls == 1


def test_api_outage_ends_incomplete_with_api_reason():
    chooser = inference.ModelChooser(FakeLLM(["error"] * 20), "m", api_attempts=3, backoff_s=0)
    result = inference.run_episode(inference.LocalDriver(), chooser, "easy", 1, verbose=False)
    assert result["incomplete"] and result["reason"].startswith(inference.API_ERROR_PREFIX)


def test_rate_limit_detection_ignores_reply_text():
    from scripts.run_benchmark import is_rate_limited

    assert is_rate_limited("model request failed: Error code: 429 - rate limit exceeded")
    assert not is_rate_limited("unparseable reply: 'an accurate rate estimate'")
    assert not is_rate_limited("reply cut off at max_tokens=32000")
