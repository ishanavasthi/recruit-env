"""The OpenEnv server (recruitenv/server.py): routes, websocket episodes, Task API,
fairness controls and the hidden verifier."""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("openenv")
from fastapi.testclient import TestClient  # noqa: E402

from recruitenv import __version__  # noqa: E402
from recruitenv.environment import RecruitmentEnvironment  # noqa: E402
from recruitenv.names import NAME_TABLE  # noqa: E402
from recruitenv.server import app  # noqa: E402

client = TestClient(app)


class Session:
    """One websocket episode session."""

    def __init__(self, ws):
        self.ws = ws

    def send(self, kind: str, data: dict | None = None) -> dict:
        msg = {"type": kind} if kind == "state" else {"type": kind, "data": data or {}}
        self.ws.send_text(json.dumps(msg))
        return json.loads(self.ws.receive_text())

    def reset(self, **kw) -> dict:
        msg = self.send("reset", kw)
        assert msg["type"] == "observation", msg
        return msg["data"]

    def step(self, **action) -> dict:
        msg = self.send("step", action)
        assert msg["type"] == "observation", msg
        return msg["data"]

    def state(self) -> dict:
        msg = self.send("state")
        assert msg["type"] == "state", msg
        return msg["data"]


def session():
    return client.websocket_connect("/ws")


# -- routes ---------------------------------------------------------------


def test_health_and_metadata():
    assert client.get("/health").json()["status"] == "healthy"
    meta = client.get("/metadata").json()
    assert meta["name"] == "RecruitEnv" and meta["version"] == __version__
    about = client.get("/").json()
    assert about["version"] == __version__


def test_schema_has_flat_action_and_reset_params():
    schema = client.get("/schema").json()
    props = schema["action"]["properties"]
    assert set(props["type"]["enum"]) == {"read_resume_section", "check_platform", "make_decision"}
    assert {"task", "name_seed", "name_blind"} <= set(schema["reset"]["properties"])


def test_tasks_route_and_task_api():
    body = client.get("/tasks").json()
    assert [t["id"] for t in body["tasks"]] == ["easy", "medium", "hard"]
    assert "properties" in body["action_schema"]
    assert client.get("/recruitenv/splits").json()
    assert client.post("/recruitenv/num_tasks", json={"split": "test"}).json() in (3, {"num_tasks": 3})
    task = client.post("/recruitenv/task", json={"split": "test", "index": 2}).json()
    assert json.dumps(task).count('"hard"') >= 1
    assert client.post("/recruitenv/task", json={"split": "test", "index": 9}).status_code == 400


def test_baseline_route():
    body = client.get("/baseline").json()
    assert set(body["scores"]) == {"easy_task", "medium_task", "hard_task"}
    assert body["scores"]["easy_task"] >= 0.75


# -- episodes over the websocket -----------------------------------------


def test_full_episode_matches_core_and_reports_grade():
    ref = RecruitmentEnvironment()
    ref.reset("easy", 5)
    truth = {c.id: c.ground_truth_label for c in ref.verifier_state().candidates}
    with session() as ws:
        s = Session(ws)
        obs = s.reset(task="easy", seed=5)
        assert obs["observation"]["task_id"] == "easy" and obs["done"] is False
        for cid, label in truth.items():
            last = s.step(type="make_decision", candidate_id=cid, decision=label)
        assert last["done"] is True
        grade = last["metadata"]["grade"]
        assert grade["score"] == last["reward"] >= 0.99
        assert grade["ground_truth"] == truth
        assert {"checks", "gates", "score_before_gates"} <= set(grade)
        st = s.state()
        assert st["is_done"] is True and st["episode_score"] == grade["score"]


def test_rejected_action_costs_nothing():
    with session() as ws:
        s = Session(ws)
        obs = s.reset(task="medium", seed=2)
        cid = obs["observation"]["candidates_summary"][0]["id"]
        s.step(type="make_decision", candidate_id=cid, decision="hold")
        again = s.step(type="make_decision", candidate_id=cid, decision="reject")
        assert again["observation"]["error"] and again["observation"]["step_number"] == 1
        bad = s.send("step", {"type": "score_dimension", "candidate_id": cid})
        assert bad["type"] == "error"


def test_reveal_appears_in_observation():
    with session() as ws:
        s = Session(ws)
        cid = s.reset(task="hard", seed=1)["observation"]["candidates_summary"][3]["id"]
        r = s.step(type="check_platform", candidate_id=cid, platform="kaggle")
        assert r["observation"]["revealed_data"][cid]["platforms"]["kaggle"]
        assert r["observation"]["last_action_result"]["data"]


# -- fairness controls ----------------------------------------------------


def test_name_blind_hides_names():
    with session() as ws:
        s = Session(ws)
        obs = s.reset(task="hard", seed=11, name_blind=True)
        cid = obs["observation"]["candidates_summary"][0]["id"]
        step = s.step(type="read_resume_section", candidate_id=cid, section="education")
        text = json.dumps([obs, step, s.state()])
        assert not [n for n, _, _ in NAME_TABLE if n in text]
        assert all(re.fullmatch(r"Candidate \d\d", c["name"]) for c in obs["observation"]["candidates_summary"])


def test_name_seed_redraws_only_names():
    with session() as ws:
        s = Session(ws)
        a = s.reset(task="easy", seed=4)["observation"]
        b = s.reset(task="easy", seed=4, name_seed=2)["observation"]
    assert [c["id"] for c in a["candidates_summary"]] == [c["id"] for c in b["candidates_summary"]]
    assert [c["name"] for c in a["candidates_summary"]] != [c["name"] for c in b["candidates_summary"]]
    assert a["job_description"] == b["job_description"] and a["quotas"] == b["quotas"]


# -- hidden verifier: nothing an agent can reach mid-episode leaks answers --

_FORBIDDEN_KEYS = {
    "ground_truth", "ground_truth_label", "ground_truth_scores", "candidates",
    "origin", "gender", "conflicts", "twins", "decision_context", "grade",
}


def _keys(obj) -> set[str]:
    if isinstance(obj, dict):
        out = set(obj)
        for v in obj.values():
            out |= _keys(v)
        return out
    if isinstance(obj, list):
        out: set[str] = set()
        for v in obj:
            out |= _keys(v)
        return out
    return set()


def test_no_ground_truth_reachable_mid_episode():
    bodies = []
    with session() as ws:
        s = Session(ws)
        obs = s.reset(task="medium", seed=7)
        cid = obs["observation"]["candidates_summary"][0]["id"]
        bodies.append(obs)
        for action in (
            {"type": "check_platform", "candidate_id": cid, "platform": "github"},
            {"type": "read_resume_section", "candidate_id": cid, "section": "skills"},
            {"type": "make_decision", "candidate_id": cid, "decision": "hold"},
            {"type": "make_decision", "candidate_id": cid, "decision": "hold"},   # rejected
        ):
            bodies.append(s.step(**action))
        bodies.append(s.state())

    get_paths = [
        route.path for route in app.routes
        if "GET" in getattr(route, "methods", set())
        and "{" not in route.path
        and route.path not in {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
    ]
    assert {"/state", "/schema", "/metadata", "/tasks"} <= set(get_paths)
    for path in get_paths:
        r = client.get(path)
        assert r.status_code == 200, path
        bodies.append(r.json())
    bodies.append(client.post("/reset", json={"task": "hard", "seed": 3}).json())
    for route in ("splits",):
        bodies.append(client.get(f"/recruitenv/{route}").json())
    for route, payload in (("tasks", {"split": "test"}), ("task", {"split": "test", "index": 0}),
                           ("task_range", {"split": "test"})):
        bodies.append(client.post(f"/recruitenv/{route}", json=payload).json())

    for body in bodies:
        leaked = _keys(body) & _FORBIDDEN_KEYS
        assert not leaked, f"leaked {leaked}"
