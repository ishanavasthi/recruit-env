"""Concurrent sessions do not share mutable state.

Pools are cached per (seed, task shape), so two sessions on the same seed are
the case that would expose shared candidates: interleave them with different
name seeds and check each matches a solo run.
"""

from __future__ import annotations

import json
import threading

import pytest

from recruitenv import policies
from recruitenv.environment import RecruitmentEnvironment
from recruitenv.models import CheckPlatformAction, MakeDecisionAction, ReadResumeSectionAction


def _script(env: RecruitmentEnvironment) -> list:
    """A fixed action list touching reveals and decisions for every candidate."""
    ids = [c["id"] for c in env.state().candidates_summary]
    acts = []
    for i, cid in enumerate(ids):
        acts.append(CheckPlatformAction(type="check_platform", candidate_id=cid,
                                        platform=("github", "leetcode", "kaggle")[i % 3]))
        acts.append(ReadResumeSectionAction(type="read_resume_section", candidate_id=cid, section="skills"))
    acts += [MakeDecisionAction(type="make_decision", candidate_id=cid,
                                decision=("shortlist", "hold", "reject")[i % 3]) for i, cid in enumerate(ids)]
    return acts


def _solo(task, seed, name_seed):
    env = RecruitmentEnvironment()
    env.reset(task, seed, name_seed=name_seed)
    trace = []
    for a in _script(env):
        obs, r, done, _ = env.step(a)
        trace.append(obs.model_dump(mode="json"))
        if done:
            break
    return trace, env.grade_report()


@pytest.mark.parametrize("task", ["medium", "hard"])
def test_interleaved_same_seed_sessions(task):
    seed = 9
    envs = [RecruitmentEnvironment() for _ in range(2)]
    for ns, env in enumerate(envs):
        env.reset(task, seed, name_seed=ns)
    scripts = [_script(env) for env in envs]
    traces = [[], []]
    done = [False, False]
    for i in range(max(map(len, scripts))):
        for k, env in enumerate(envs):
            if done[k] or i >= len(scripts[k]):
                continue
            obs, _, d, _ = env.step(scripts[k][i])
            traces[k].append(obs.model_dump(mode="json"))
            done[k] = d
    for ns, env in enumerate(envs):
        solo_trace, solo_grade = _solo(task, seed, ns)
        assert traces[ns] == solo_trace
        assert env.grade_report() == solo_grade
    names = [[c["name"] for c in e.state().candidates_summary] for e in envs]
    assert names[0] != names[1]


def test_mutating_one_session_does_not_touch_another():
    a, b = RecruitmentEnvironment(), RecruitmentEnvironment()
    a.reset("easy", 3)
    b.reset("easy", 3)
    before = json.dumps([c.model_dump(mode="json") for c in b.verifier_state().candidates], sort_keys=True)
    a._state.candidates[0].github.repos = -999   # mutate a's live state directly
    after = json.dumps([c.model_dump(mode="json") for c in b.verifier_state().candidates], sort_keys=True)
    assert before == after
    c = RecruitmentEnvironment()
    c.reset("easy", 3)
    assert c.verifier_state().candidates[0].github.repos != -999


def test_threaded_policies_match_sequential():
    jobs = [(t, s, ns) for t in ("easy", "medium", "hard") for s in (1, 2) for ns in (0, 1)]

    def run(job):
        t, s, ns = job
        env = RecruitmentEnvironment()
        env.reset(t, s, name_seed=ns)
        policies.rubric_heuristic(env)
        return env.grade_report()

    expected = {j: run(j) for j in jobs}
    got: dict = {}
    errors: list = []

    def worker(job):
        try:
            got[job] = run(job)
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(j,)) for j in jobs for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors
    assert got == expected


def test_websocket_sessions_do_not_interfere():
    pytest.importorskip("openenv")
    from fastapi.testclient import TestClient

    from recruitenv.server import app

    client = TestClient(app)
    solo = {ns: _solo("medium", 9, ns) for ns in (0, 1)}
    with client.websocket_connect("/ws") as w0, client.websocket_connect("/ws") as w1:
        socks = {0: w0, 1: w1}
        obs = {}
        for ns, w in socks.items():
            w.send_text(json.dumps({"type": "reset", "data": {"task": "medium", "seed": 9, "name_seed": ns}}))
            obs[ns] = json.loads(w.receive_text())["data"]["observation"]
        ref = RecruitmentEnvironment()
        ref.reset("medium", 9)
        script = [a.model_dump(mode="json") for a in _script(ref)]
        last = {}
        for a in script:
            for ns, w in socks.items():
                if last.get(ns, {}).get("done"):
                    continue
                w.send_text(json.dumps({"type": "step", "data": a}))
                last[ns] = json.loads(w.receive_text())["data"]
        for ns in socks:
            assert last[ns]["done"] is True
            assert last[ns]["metadata"]["grade"] == json.loads(json.dumps(solo[ns][1]))
        assert [c["name"] for c in obs[0]["candidates_summary"]] != [c["name"] for c in obs[1]["candidates_summary"]]
