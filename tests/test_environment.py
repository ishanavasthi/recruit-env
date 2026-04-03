"""Tests for the core environment (reset, step, determinism)."""

import pytest

from env.environment import RecruitmentEnvironment
from env.models import ReadResumeSectionAction, CheckPlatformAction, MakeDecisionAction


class TestReset:
    """reset() must be deterministic and return a valid observation."""

    def test_reset_returns_observation(self) -> None:
        """reset() returns an Observation with correct initial fields."""
        pytest.skip("Not implemented yet")

    def test_reset_determinism(self) -> None:
        """Two resets with the same seed produce identical state."""
        pytest.skip("Not implemented yet")


class TestStep:
    """step() processes each action type correctly."""

    def test_read_resume_section(self) -> None:
        """read_resume_section returns the requested section data."""
        pytest.skip("Not implemented yet")

    def test_check_platform(self) -> None:
        """check_platform returns platform-specific data."""
        pytest.skip("Not implemented yet")

    def test_make_decision_ends_candidate(self) -> None:
        """make_decision marks the candidate as decided."""
        pytest.skip("Not implemented yet")

    def test_episode_ends_when_all_decided(self) -> None:
        """Episode done flag is set when every candidate has a decision."""
        pytest.skip("Not implemented yet")

    def test_step_budget_exhaustion(self) -> None:
        """Episode ends when step budget is exhausted."""
        pytest.skip("Not implemented yet")
