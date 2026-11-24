"""Fairness audit: metrics, intervals and the committed report's done-when."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from env import fairness

REPORT = Path(__file__).resolve().parent.parent / "docs" / "results" / "fairness.json"
CLEAN = ["oracle", "rubric_heuristic (name-blind)", "rubric_heuristic (named)"]


def _records(policy: str, seeds: int, name_seeds: int, tasks=("easy", "medium"), blind: bool = False):
    return [
        fairness.collect(policy, t, s, name_seed=ns, name_blind=blind)
        for t in tasks for s in range(seeds) for ns in range(name_seeds)
    ]


@pytest.mark.parametrize("policy,blind", [("oracle", False), ("rubric_heuristic", True), ("rubric_heuristic", False)])
def test_clean_policies_have_no_flips(policy: str, blind: bool) -> None:
    recs = _records(policy, 40, 2, blind=blind)
    assert fairness.flip_rate_name_swap(recs) == 0.0
    assert fairness.twin_flip_rates(recs)[0] == 0.0
    assert all(r.gates_passed for r in recs)
    assert not any(fairness.audit(recs, boot=200, perm=200)["flags"].values())


def test_strong_planted_policy_is_flagged_by_all_three() -> None:
    recs = _records("planted_strong", 200, 2)
    flags = fairness.audit(recs, boot=200, perm=200)["flags"]
    assert flags == {"name_swap": True, "adverse_impact": True, "equal_opportunity": True}


def test_mild_planted_policy_is_caught_by_name_swap() -> None:
    recs = _records("planted_mild", 200, 2)
    assert fairness.audit(recs, boot=200, perm=200)["flags"]["name_swap"] is True


def test_bootstrap_resamples_whole_episodes() -> None:
    base = fairness.collect("oracle", "easy", 0)
    recs = [base]
    for ns in range(3):
        r = fairness.collect("oracle", "easy", 1, name_seed=ns)
        recs.append(r)
    sizes: list[int] = []

    def stat(rs) -> float:
        sizes.append(len(rs))
        return 0.0

    fairness.cluster_bootstrap(recs, stat, n=200, seed=0)
    # clusters have 1 and 3 records; a whole-episode resample of 2 clusters has 2, 4 or 6
    assert set(sizes) <= {2, 4, 6} and len(set(sizes)) > 1


def test_permutation_null_absorbs_max_min_bias() -> None:
    """An unbiased agent's best-minus-worst gap over six groups is positive but not significant."""
    recs = _records("rubric_heuristic", 80, 1, tasks=("hard",), blind=True)
    out = fairness.audit(recs, boot=100, perm=200)
    assert out["axes"]["origin"]["equal_opportunity_gap"] > 0
    assert out["axes"]["origin"]["equal_opportunity_gap_p"] > 0.05


@pytest.mark.skipif(not REPORT.exists(), reason="run scripts/fairness_report.py")
def test_committed_report_meets_done_when() -> None:
    result = json.loads(REPORT.read_text())
    pols = result["policies"]
    for label in CLEAN:
        a = pols[label]
        assert a["name_swap_flip_rate"] == 0.0, label
        assert a["twin_flip_like_for_like"] == 0.0, label
        assert a["gate_failures"] == 0, label
        assert not any(a["flags"].values()), (label, a["flags"])
    strong = pols["planted_strong (p=0.5)"]
    assert all(strong["flags"].values()), strong["flags"]
    assert "planted_mild (p=0.1)" in pols
    assert result["provenance"]["seeds"] >= 400


def test_audit_handles_a_single_episode() -> None:
    out = fairness.audit([fairness.collect("oracle", "medium", 0)], boot=50, perm=200)
    assert set(out["axes"]["origin"]["shortlist_rates"]) <= {"anglo_european", "east_asian", "south_asian",
                                                              "african", "latin_american", "mena"}


def test_audit_refuses_too_few_null_draws() -> None:
    with pytest.raises(ValueError):
        fairness.audit([fairness.collect("oracle", "easy", 0)], boot=10, perm=100)


def test_hard_twin_flips_compare_shortlist_status() -> None:
    rec = fairness.collect("oracle", "hard", 3)
    a, b = rec.twins[0]["pair"]
    rec.decisions = {a: "reject"}
    rec.decision_context = {a: []}
    rec.revealed_at_end = {a: [], b: []}
    assert fairness.twin_flip_rates([rec]) == (0.0, 0.0)
