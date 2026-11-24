"""Acceptance criteria for the 3.0.0 task design (spec sections 7 and 10).

Measured with the committed constants over evaluation seeds 0-199; the
single-source adversary is chosen per role on training seeds 1000-1199.
The same numbers are printed by ``python scripts/difficulty_sweep.py --seeds 200 --current``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import difficulty_sweep  # noqa: E402

N = 200


@pytest.fixture(scope="module")
def criteria() -> dict:
    return difficulty_sweep.measure(N)


def test_c2_no_single_source_solves_medium_or_hard(criteria) -> None:
    assert criteria["C2_medium_best_single_pass"] <= 0.20
    assert criteria["C2_hard_best_single_pass"] <= 0.20


def test_c3_full_read_beats_best_single_source_on_medium(criteria) -> None:
    assert criteria["C3_medium_full_minus_best_single"] >= 0.2


def test_c4_hard_rewards_prioritising(criteria) -> None:
    diff, lo, _ = criteria["C4_hard_diff_ci"]
    assert diff >= 0.05 and lo > 0
    assert 0.5 <= criteria["C4_hard_prioritising_mean"] <= 0.85
    assert criteria["C4_hard_random_pass"] < 0.05


def test_c17_easy_heuristic_passes(criteria) -> None:
    assert criteria["C17_easy_heuristic_pass"] >= 0.90


def test_generation_is_cheap(criteria) -> None:
    mean, worst = criteria["attempts_mean_max"]
    assert mean <= 5 and worst < 100
