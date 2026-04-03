"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthCheck:
    """GET / returns ok status."""

    def test_root(self) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestResetEndpoint:
    """POST /reset starts a new episode."""

    def test_reset(self) -> None:
        """Reset returns an observation."""
        pytest.skip("Not implemented yet")


class TestStepEndpoint:
    """POST /step processes an action."""

    def test_step(self) -> None:
        """Step returns observation and reward."""
        pytest.skip("Not implemented yet")


class TestStateEndpoint:
    """GET /state returns full state."""

    def test_state(self) -> None:
        """State endpoint returns serialised state."""
        pytest.skip("Not implemented yet")


class TestTasksEndpoint:
    """GET /tasks lists available tasks."""

    def test_list_tasks(self) -> None:
        """Tasks endpoint returns task IDs."""
        pytest.skip("Not implemented yet")
