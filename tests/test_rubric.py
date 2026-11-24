"""The published hiring rubric and the name table."""

from __future__ import annotations

import math

import pytest

from env import rubric
from env.models import (
    CandidateProfile,
    GitHubStats,
    JobDescription,
    KaggleStats,
    LeetCodeStats,
    ResumeSection,
)
from env.names import GENDERS, NAME_TABLE, ORIGINS, blind_label, group_of
from env.profile_factory import ProfileFactory


def _jd(role: str) -> JobDescription:
    return ProfileFactory().generate_job_description(seed=0, role_type=role)


def _profile(
    *,
    stars: int = 100,
    repos: int = 15,
    streak: int = 30,
    contributions: int = 300,
    years_active: float = 4.0,
    rating: int = 1500,
    hard: int = 10,
    percentile: float = 50.0,
    kaggle_pct: float = 50.0,
    kaggle_rank: str = "Contributor",
    medals: dict[str, int] | None = None,
    claimed_years: float = 4.0,
    skills: list[str] | None = None,
) -> CandidateProfile:
    return CandidateProfile(
        id="candidate_001",
        name="Test Person",
        resume_sections={
            "education": ResumeSection(section_name="education", content="B.S. CS", years_experience=0.0),
            "experience": ResumeSection(section_name="experience", content="x", years_experience=claimed_years),
            "skills": ResumeSection(
                section_name="skills", content=", ".join(skills or []), skills=list(skills or [])
            ),
        },
        github=GitHubStats(
            repos=repos,
            stars_received=stars,
            commit_streak_days=streak,
            contributions_last_year=contributions,
            years_active=years_active,
        ),
        leetcode=LeetCodeStats(
            problems_solved=100 + hard, hard=hard, contest_rating=rating, global_rank_percentile=percentile
        ),
        kaggle=KaggleStats(rank=kaggle_rank, best_finish_percentile=kaggle_pct, medals=medals or {}),
        ground_truth_label="hold",
    )


ROLES = ["ml_engineer", "frontend_dev", "backend_dev", "data_scientist"]


@pytest.mark.parametrize("role", ROLES)
def test_score_equals_estimate_on_full_reveal(role: str) -> None:
    jd = _jd(role)
    for i in range(50):
        p = _profile(
            stars=7 * i, repos=i, streak=3 * i, contributions=20 * i, years_active=i / 5,
            rating=1000 + 20 * i, hard=i, percentile=float(2 * i), kaggle_pct=float(i),
            claimed_years=i / 4, skills=jd.required_skills[: i % 6],
        )
        full = rubric.estimate(rubric.reveal_record(p, rubric.SOURCES), jd)
        assert full.total == rubric.score(p, jd).total


def test_education_is_ignored() -> None:
    jd = _jd("backend_dev")
    p = _profile(skills=jd.required_skills[:3])
    without = rubric.reveal_record(p, ["github", "skills"])
    with_edu = rubric.reveal_record(p, ["github", "skills"])
    with_edu["resume_sections"]["education"] = p.resume_sections["education"].model_dump()
    assert rubric.estimate(without, jd) == rubric.estimate(with_edu, jd)


@pytest.mark.parametrize("role", ROLES)
def test_empty_record_gives_prior_estimate(role: str) -> None:
    est = rubric.estimate({}, _jd(role))
    assert math.isfinite(est.total) and 0.0 < est.total < 1.0


def test_jd_changes_ranking() -> None:
    """The same two candidates swap order between an ML and a frontend job."""
    ml, fe = _jd("ml_engineer"), _jd("frontend_dev")
    ml_person = _profile(skills=ml.required_skills, kaggle_pct=80.0, kaggle_rank="Expert")
    fe_person = _profile(skills=fe.required_skills, kaggle_pct=40.0)
    assert rubric.score(ml_person, ml).total > rubric.score(fe_person, ml).total
    assert rubric.score(ml_person, fe).total < rubric.score(fe_person, fe).total


def test_kaggle_weighs_more_for_ml_than_frontend() -> None:
    assert rubric.TECH_WEIGHTS["ml_engineer"]["kaggle"] > rubric.TECH_WEIGHTS["frontend_dev"]["kaggle"]


def test_experience_capped_by_activity() -> None:
    jd = _jd("backend_dev")
    inflated = _profile(claimed_years=9.0, years_active=1.0)
    honest = _profile(claimed_years=9.0, years_active=8.0)
    assert rubric.score(inflated, jd).experience < rubric.score(honest, jd).experience


def test_skill_factor_multiplicative() -> None:
    jd = _jd("data_scientist")
    none = rubric.score(_profile(skills=[]), jd)
    base = jd.weight_technical * none.technical + jd.weight_experience * none.experience + jd.weight_growth * none.growth
    assert none.skill_match == 0.0
    assert none.total == pytest.approx(base * rubric.SKILL_FLOOR)


def test_no_saturation_ties() -> None:
    jd = _jd("frontend_dev")
    subs = [rubric.score(_profile(stars=s), jd).per_source["github"] for s in range(0, 20000, 500)]
    assert all(a < b for a, b in zip(subs, subs[1:]))


def test_name_table_balanced_and_unambiguous() -> None:
    names = [n for n, _, _ in NAME_TABLE]
    assert len(names) == len(set(names)) == 60
    for o in ORIGINS:
        for g in GENDERS:
            assert sum(1 for _, oo, gg in NAME_TABLE if oo == o and gg == g) == 5
    for banned in ("David Kim", "Naomi Watanabe", "Zara Hussain", "Hana Yilmaz", "Sara Johansson"):
        assert banned not in names
    assert group_of(NAME_TABLE[0][0]) == NAME_TABLE[0][1:]
    assert group_of("Nobody Here") is None
    assert blind_label(6) == "Candidate 07"


@pytest.mark.parametrize("role", ROLES)
def test_rubric_is_monotone_in_every_platform(role: str) -> None:
    """Better evidence on any platform never lowers the total."""
    jd = _jd(role)
    base = dict(stars=100, rating=1500, kaggle_pct=50.0, skills=jd.required_skills[:2])
    for field, values in (("stars", [0, 50, 200, 2000]), ("rating", [900, 1300, 1700, 2300]),
                          ("kaggle_pct", [1.0, 30.0, 70.0, 99.0])):
        totals = [rubric.score(_profile(**{**base, field: v}), jd).total for v in values]
        assert all(a < b for a, b in zip(totals, totals[1:])), (field, totals)
    for p in (_profile(kaggle_pct=1.0), _profile(stars=5000, rating=2400)):
        low = rubric.score(p, jd).total
        p.kaggle.best_finish_percentile = 99.0
        assert rubric.score(p, jd).total > low


def test_rubric_is_monotone_in_every_input_on_generated_profiles() -> None:
    """Property: raising any rubric input of a generated candidate never lowers its total."""
    bumps = {
        ("github", "stars_received"): 50, ("github", "repos"): 3, ("github", "commit_streak_days"): 10,
        ("github", "contributions_last_year"): 80, ("github", "years_active"): 1.0,
        ("leetcode", "contest_rating"): 100, ("leetcode", "hard"): 5, ("leetcode", "global_rank_percentile"): 5.0,
        ("leetcode", "problems_solved"): 30, ("kaggle", "best_finish_percentile"): 5.0,
    }
    factory = ProfileFactory()
    for seed in range(40):
        ep = factory.generate_episode(seed, {"shortlist": 5, "hold": 8, "reject": 7})
        for c in ep.candidates:
            base = rubric.score(c, ep.jd).total
            for (platform, field), delta in bumps.items():
                bumped = c.model_copy(deep=True)
                stats = getattr(bumped, platform)
                value = getattr(stats, field)
                if field.endswith("percentile"):
                    value = min(99.9, value + delta)
                else:
                    value = value + delta
                setattr(stats, field, type(getattr(stats, field))(value))
                assert rubric.score(bumped, ep.jd).total >= base - 1e-12, (seed, c.id, field)
            more = c.model_copy(deep=True)
            more.resume_sections["experience"].years_experience += 1.0
            assert rubric.score(more, ep.jd).total >= base - 1e-12
