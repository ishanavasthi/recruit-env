"""Generator v3: quotas, rubric labels, margins, conflicts, twins and names."""

from __future__ import annotations

import hashlib
import json
import statistics as st
from collections import Counter, defaultdict

import pytest

from recruitenv import rubric
from recruitenv.environment import RecruitmentEnvironment
from recruitenv.names import group_of
from recruitenv.profile_factory import (
    MARGIN,
    ROLES,
    ProfileFactory,
    conflicted_candidates,
    fill_quotas,
)
from recruitenv.tasks import TASK_REGISTRY

TASKS = ["easy", "medium", "hard"]
SEEDS = range(100)
N_CONVERGE = 1000


def _episode(task: str, seed: int, name_seed: int = 0):
    cfg = TASK_REGISTRY[task]
    return ProfileFactory().generate_episode(seed, cfg.quotas, cfg.required_conflicts, name_seed=name_seed)


@pytest.fixture(scope="session")
def converge() -> dict[str, list[int]]:
    return {t: [_episode(t, s).attempts for s in range(N_CONVERGE)] for t in TASKS}


@pytest.mark.parametrize("task", TASKS)
def test_generation_converges(task: str, converge) -> None:
    attempts = converge[task]
    assert st.mean(attempts) <= 5.0, st.mean(attempts)
    assert max(attempts) < 100


@pytest.mark.parametrize("task", TASKS)
def test_quotas_and_counts(task: str) -> None:
    cfg = TASK_REGISTRY[task]
    for seed in SEEDS:
        ep = _episode(task, seed)
        assert len(ep.candidates) == cfg.candidate_count
        assert Counter(c.ground_truth_label for c in ep.candidates) == Counter(cfg.quotas)
        assert len({c.id for c in ep.candidates}) == cfg.candidate_count


@pytest.mark.parametrize("task", TASKS)
def test_labels_follow_rubric_with_margin(task: str) -> None:
    cfg = TASK_REGISTRY[task]
    for seed in SEEDS:
        ep = _episode(task, seed)
        totals = {c.id: rubric.score(c, ep.jd).total for c in ep.candidates}
        assert fill_quotas(totals, cfg.quotas) == {c.id: c.ground_truth_label for c in ep.candidates}
        by = defaultdict(list)
        for c in ep.candidates:
            by[c.ground_truth_label].append(totals[c.id])
        assert min(by["shortlist"]) - max(by["hold"]) >= MARGIN
        assert min(by["hold"]) - max(by["reject"]) >= MARGIN


def test_role_rotates_and_shapes_the_jd() -> None:
    for seed in SEEDS:
        jd = _episode("easy", seed).jd
        assert jd.role_type == ROLES[seed % 4]
        assert abs(jd.weight_technical + jd.weight_experience + jd.weight_growth - 1) < 1e-9


@pytest.mark.parametrize("task", TASKS)
def test_conflict_predicate_matches_record(task: str) -> None:
    """The recorded conflicts are exactly the candidates the predicate flags."""
    cfg = TASK_REGISTRY[task]
    for seed in range(40):
        ep = _episode(task, seed)
        records = {c.id: rubric.reveal_record(c, rubric.SOURCES) for c in ep.candidates}
        labels = {c.id: c.ground_truth_label for c in ep.candidates}
        assert conflicted_candidates(records, labels, ep.jd, cfg.quotas) == ep.conflicts
        if task == "medium":
            assert len(ep.conflicts) == 2


@pytest.mark.parametrize("task", TASKS)
def test_twin_pair_is_identical_on_rubric_inputs(task: str) -> None:
    for seed in SEEDS:
        ep = _episode(task, seed)
        (twin,) = ep.twins
        a, b = (next(c for c in ep.candidates if c.id == cid) for cid in twin["pair"])
        def numbers(c):
            gh = c.github.model_dump(exclude={"notable_projects"})
            return (gh, c.leetcode.model_dump(), c.kaggle.model_dump(),
                    c.resume_sections["experience"].years_experience, c.resume_sections["skills"].skills)

        assert numbers(a) == numbers(b)
        assert len(a.github.notable_projects) == len(b.github.notable_projects)
        assert rubric.score(a, ep.jd) == rubric.score(b, ep.jd)
        assert a.ground_truth_label == b.ground_truth_label == twin["label"]
        assert a.resume_sections["education"].content != b.resume_sections["education"].content
        assert a.id not in ep.conflicts and b.id not in ep.conflicts
        ga, gb = group_of(a.name), group_of(b.name)
        assert ga is not None and gb is not None
        if twin["contrast"] == "origin":
            assert ga[0] != gb[0] and ga[1] == gb[1]
        elif twin["contrast"] == "gender":
            assert ga[0] == gb[0] and ga[1] != gb[1]
        else:
            assert ga[0] != gb[0] and ga[1] != gb[1]


def test_twin_label_and_contrast_rates() -> None:
    eps = [_episode("easy", s) for s in range(N_CONVERGE)]
    labels = Counter(e.twins[0]["label"] for e in eps)
    contrasts = Counter(e.twins[0]["contrast"] for e in eps)
    n = len(eps)
    assert abs(labels["shortlist"] / n - 0.50) < 0.06
    assert abs(labels["hold"] / n - 0.35) < 0.06
    assert abs(labels["reject"] / n - 0.15) < 0.05
    for c in ("origin", "gender", "both"):
        assert abs(contrasts[c] / n - 1 / 3) < 0.06


@pytest.mark.parametrize("task", TASKS)
def test_name_seed_changes_only_names(task: str) -> None:
    for seed in range(30):
        a, b = _episode(task, seed, 0), _episode(task, seed, 1)
        assert [c.id for c in a.candidates] == [c.id for c in b.candidates]
        strip = lambda c: c.model_dump(exclude={"name", "origin", "gender"})  # noqa: E731
        assert [strip(c) for c in a.candidates] == [strip(c) for c in b.candidates]
        assert a.twins == b.twins and a.conflicts == b.conflicts
        assert [c.name for c in a.candidates] != [c.name for c in b.candidates]
        assert len({c.name for c in b.candidates}) == len(b.candidates)


def _pool_hash(task: str, seed: int) -> str:
    ep = _episode(task, seed)
    blob = json.dumps([c.model_dump() for c in ep.candidates], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


PINNED: dict[tuple[str, int], str] = {
    ("easy", 0): "75f628a2731cbbc4",
    ("easy", 1): "f4c4d057d479fe77",
    ("easy", 2): "255cd4c4b519bdc6",
    ("medium", 0): "eff9966e2de445d7",
    ("medium", 1): "b53a0ad13254024d",
    ("medium", 2): "2780477f3853028b",
    ("hard", 0): "344019951849f2c0",
    ("hard", 1): "ebbcbbdb513a87a0",
    ("hard", 2): "2085fd364ac88213",
}


@pytest.mark.parametrize("task", TASKS)
def test_deterministic_and_pinned(task: str) -> None:
    for seed in (0, 1, 2):
        h = _pool_hash(task, seed)
        assert h == _pool_hash(task, seed)
        if (task, seed) in PINNED:
            assert h == PINNED[(task, seed)], "generator output changed: update PINNED deliberately"


def test_priors_match_generator_means() -> None:
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for seed in range(5000, 5300):
        ep = ProfileFactory().generate_episode(seed, {"shortlist": 5, "hold": 8, "reject": 7})
        for c in ep.candidates:
            bd = rubric.score(c, ep.jd).per_source
            d = acc[ep.jd.role_type]
            for k in ("github", "leetcode", "kaggle", "github_activity", "leetcode_practice"):
                d[k].append(bd[k])
            d["skill_match"].append(bd["skills"])
    for role, d in acc.items():
        for k, v in d.items():
            assert abs(st.mean(v) - rubric.PRIORS[role][k]) < 0.02, (role, k, st.mean(v), rubric.PRIORS[role][k])


def test_observation_has_quotas_and_no_secrets() -> None:
    env = RecruitmentEnvironment()
    obs = env.reset("medium", 3)
    assert obs.quotas == {"shortlist": 3, "hold": 3, "reject": 2}
    blob = obs.model_dump_json()
    for secret in ("ground_truth", "conflicts", "twins", "origin", "gender", "decision_context"):
        assert secret not in blob


def test_tasks_sharing_a_seed_share_no_candidates() -> None:
    """Reading one task's pool must not reveal another task's pool for the same seed."""
    def numbers(c):
        return json.dumps([c.github.model_dump(exclude={"notable_projects", "top_languages"}),
                           c.leetcode.model_dump(), c.kaggle.model_dump()], sort_keys=True)

    for seed in range(50):
        pools = {t: {numbers(c) for c in _episode(t, seed).candidates} for t in TASKS}
        assert not pools["easy"] & pools["hard"]
        assert not pools["easy"] & pools["medium"]
        assert not pools["medium"] & pools["hard"]
