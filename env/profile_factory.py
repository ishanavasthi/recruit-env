"""Synthetic candidate profile generator.

Uses numpy seeded RNG for full determinism.  NO live HTTP calls.
"""

from __future__ import annotations

from typing import Literal

from env.models import CandidateProfile, JobDescription


class ProfileFactory:
    """Generates synthetic candidate profiles and job descriptions from a seed."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def generate_job_description(self) -> JobDescription:
        """Create a synthetic job description deterministically from the seed."""
        raise NotImplementedError

    def generate_candidates(self, count: int) -> list[CandidateProfile]:
        """Generate *count* synthetic candidate profiles.

        Profile distributions follow CLAUDE.md tiers:
        - Junior: leetcode <100, github <10 repos, kaggle unranked
        - Mid:    leetcode 100-250, github 10-30 repos, kaggle Contributor
        - Senior: leetcode 250+, github 30+ repos, kaggle Expert+

        Ground-truth labels are consistent with tier + controlled noise.
        """
        raise NotImplementedError

    def _random_tier(self) -> Literal["junior", "mid", "senior"]:
        """Pick a candidate tier using the seeded RNG."""
        raise NotImplementedError
