"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# -----------------------------------------------------------------------
# health / root
# -----------------------------------------------------------------------


class TestHealth:
    def test_root(self) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_returns_200(self) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"] == "2.0.0"


# -----------------------------------------------------------------------
# /tasks
# -----------------------------------------------------------------------


class TestTasks:
    def test_tasks_endpoint_returns_3_tasks(self) -> None:
        r = client.get("/tasks")
        assert r.status_code == 200
        body = r.json()
        assert len(body["tasks"]) == 3
        ids = {t["id"] for t in body["tasks"]}
        assert ids == {"easy", "medium", "hard"}

    def test_tasks_includes_action_schema(self) -> None:
        r = client.get("/tasks")
        body = r.json()
        schema = body["action_schema"]
        # Discriminated union schema should have oneOf or $defs
        assert "oneOf" in schema or "$defs" in schema


# -----------------------------------------------------------------------
# /reset
# -----------------------------------------------------------------------


class TestReset:
    def test_reset_valid_task(self) -> None:
        r = client.post("/reset", json={"task_id": "easy", "seed": 42})
        assert r.status_code == 200
        obs = r.json()
        assert obs["task_id"] == "easy"
        assert obs["step_number"] == 0
        assert obs["steps_remaining"] == 40
        assert len(obs["candidates_summary"]) == 10
        assert obs["done"] is False

    def test_reset_invalid_task_400(self) -> None:
        r = client.post("/reset", json={"task_id": "nonexistent", "seed": 42})
        assert r.status_code == 400
        assert "Unknown task_id" in r.json()["detail"]


# -----------------------------------------------------------------------
# /step
# -----------------------------------------------------------------------


class TestStep:
    def test_step_without_reset_400(self) -> None:
        """Stepping on a fresh (no-episode) env returns 400.

        NOTE: the test client shares state with other tests, so we first
        ensure there is no active episode by resetting + completing one,
        then attempting a step after done.  Alternatively, we check that
        an obviously invalid step (after done) returns 400.
        """
        # Reset and immediately finish the episode
        client.post("/reset", json={"task_id": "easy", "seed": 99})
        state = client.get("/state").json()
        for c in state["candidates_summary"]:
            client.post(
                "/step",
                json={
                    "type": "make_decision",
                    "candidate_id": c["id"],
                    "decision": "hold",
                },
            )
        # Now episode is done → next step should 400
        r = client.post(
            "/step",
            json={
                "type": "read_resume_section",
                "candidate_id": state["candidates_summary"][0]["id"],
                "section": "education",
            },
        )
        assert r.status_code == 400

    def test_step_returns_correct_shape(self) -> None:
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        state = client.get("/state").json()
        cid = state["candidates_summary"][0]["id"]

        r = client.post(
            "/step",
            json={
                "type": "check_platform",
                "candidate_id": cid,
                "platform": "leetcode",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "observation" in body
        assert "reward" in body
        assert "done" in body
        assert "info" in body
        assert body["reward"]["step_reward"] == 0.0
        stats = body["observation"]["revealed_data"][cid]["platforms"]["leetcode"]
        assert stats == body["info"]["data"]


# -----------------------------------------------------------------------
# full episode — reset → step loop → grader
# -----------------------------------------------------------------------


class TestFullEpisode:
    def test_full_easy_episode(self) -> None:
        """Run a complete easy episode through the API and verify grading."""
        # Ground truth comes from the verifier side, never from the API
        from env.environment import RecruitmentEnvironment

        ref = RecruitmentEnvironment()
        ref.reset("easy", 42)
        gt = {c.id: c.ground_truth_label for c in ref.verifier_state().candidates}

        r = client.post("/reset", json={"task_id": "easy", "seed": 42})
        assert r.status_code == 200

        # Decide all candidates with correct labels
        for cid, label in gt.items():
            r = client.post(
                "/step",
                json={
                    "type": "make_decision",
                    "candidate_id": cid,
                    "decision": label,
                },
            )
            assert r.status_code == 200

        last = r.json()
        assert last["done"] is True

        # Grade
        r = client.post("/grader")
        assert r.status_code == 200
        grade = r.json()
        assert grade["score"] >= 0.99
        assert grade["score"] == last["reward"]["cumulative_reward"]
        assert grade["task_id"] == "easy"
        assert grade["breakdown"]["correct"] == 10

    def test_grader_rejects_incomplete_episode(self) -> None:
        """Grading before episode ends returns 400."""
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.post("/grader")
        assert r.status_code == 400
        assert "not complete" in r.json()["detail"]


# -----------------------------------------------------------------------
# /state
# -----------------------------------------------------------------------


class TestState:
    def test_state_after_reset(self) -> None:
        client.post("/reset", json={"task_id": "medium", "seed": 42})
        r = client.get("/state")
        assert r.status_code == 200
        body = r.json()
        assert body["task_id"] == "medium"
        assert len(body["candidates_summary"]) == 5


# -----------------------------------------------------------------------
# hidden verifier: nothing an agent can call mid-episode leaks the answers
# -----------------------------------------------------------------------

_FORBIDDEN_KEYS = {"ground_truth", "ground_truth_label", "ground_truth_scores", "candidates"}


def _keys(obj) -> set[str]:
    """All dict keys anywhere in a decoded JSON value."""
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


class TestHiddenVerifier:
    def test_no_ground_truth_reachable_mid_episode(self) -> None:
        """Every GET route, /step and /grader return no ground-truth field mid-episode."""
        obs = client.post("/reset", json={"task_id": "medium", "seed": 7}).json()
        cid = obs["candidates_summary"][0]["id"]
        bodies = [obs]
        for action in (
            {"type": "check_platform", "candidate_id": cid, "platform": "github"},
            {"type": "read_resume_section", "candidate_id": cid, "section": "skills"},
            {"type": "make_decision", "candidate_id": cid, "decision": "hold"},
        ):
            r = client.post("/step", json=action)
            assert r.status_code == 200
            bodies.append(r.json())

        get_paths = [
            route.path for route in app.routes
            if "GET" in getattr(route, "methods", set())
            and "{" not in route.path
            and route.path not in {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
        ]
        assert "/state" in get_paths
        for path in get_paths:
            r = client.get(path)
            assert r.status_code == 200, path
            bodies.append(r.json())

        r = client.post("/grader")
        assert r.status_code == 400
        bodies.append(r.json())

        for body in bodies:
            leaked = _keys(body) & _FORBIDDEN_KEYS
            assert not leaked, f"leaked {leaked}"

    def test_state_is_the_agent_view(self) -> None:
        client.post("/reset", json={"task_id": "easy", "seed": 3})
        state = client.get("/state").json()
        assert state["task_id"] == "easy"
        assert len(state["candidates_summary"]) == 10
        assert state["is_done"] is False
        assert state["episode_score"] is None

    def test_grader_reports_ground_truth_after_episode(self) -> None:
        obs = client.post("/reset", json={"task_id": "easy", "seed": 5}).json()
        for c in obs["candidates_summary"]:
            r = client.post("/step", json={"type": "make_decision", "candidate_id": c["id"], "decision": "hold"})
        last = r.json()
        assert last["done"] is True
        grade = client.post("/grader").json()
        assert set(grade["breakdown"]["ground_truth"]) == {c["id"] for c in obs["candidates_summary"]}
        assert grade["score"] == last["reward"]["cumulative_reward"]

    def test_score_dimension_action_is_gone(self) -> None:
        obs = client.post("/reset", json={"task_id": "easy", "seed": 42}).json()
        r = client.post("/step", json={
            "type": "score_dimension",
            "candidate_id": obs["candidates_summary"][0]["id"],
            "dimension": "technical",
            "score": 0.5,
        })
        assert r.status_code == 422
