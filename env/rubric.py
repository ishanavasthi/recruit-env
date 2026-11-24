"""The published hiring rubric.

Every ground-truth label in RecruitEnv comes from this module. The generator
ranks each pool by :func:`score` and fills the task's quotas (top ``k``
shortlist, bottom ``r`` reject, the rest hold). The formula is public on
purpose: an agent that reveals every source and applies it scores the
maximum, so the task measures how well an agent gathers and weighs evidence
under a step budget, not how well it guesses a hidden formula.

Sources (one action reveals one source):

==========  ====================================  =============================
source      action                                rubric inputs
==========  ====================================  =============================
github      check_platform github                 stars, repos, streak,
                                                  contributions, years_active
leetcode    check_platform leetcode               contest rating, hard solved,
                                                  global percentile, problems
                                                  solved
kaggle      check_platform kaggle                 rank, best finish percentile,
                                                  medals
experience  read_resume_section experience        claimed years
skills      read_resume_section skills            skills list
==========  ====================================  =============================

The education section is not an input. It is a distractor, and a known
prestige-bias proxy, so it never affects a label.

Formula (``h(x, s) = x / (x + s)`` is a smooth, never-saturating curve, so no
two different inputs tie):

- **Per-platform sub-scores**, each in [0, 1]:

  - ``leetcode = 0.5*sigmoid((rating - 1500)/200) + 0.25*h(hard, 15) + 0.25*percentile/100``
  - ``github   = 0.6*h(ln(1+stars), ln(1+150)) + 0.4*h(repos, 15)``
  - ``kaggle   = 0.5*percentile/100 + 0.3*rank_value + 0.2*h(3*gold + 2*silver + bronze, 6)``

- **technical** = weighted power mean of the three platform sub-scores with
  exponent ``TECH_POWER`` (-2) and the role's ``TECH_WEIGHTS``:
  ``(sum w_k * sub_k**-2) ** (-1/2)``. It rises with every sub-score but sits
  below the plain mean when they disagree, so a strong claim on one platform
  needs corroboration from the others.
- **growth** = ``0.5*activity + 0.5*practice``: GitHub activity
  ``0.5*h(streak, 45) + 0.5*h(contributions, 300)`` and LeetCode practice
  volume ``h(problems_solved, 150)``.
- **experience** = ``h(supported_years, jd.experience_years)``, where
  ``supported_years = min(claimed_years, years_active + SLACK_YEARS)`` once
  both the experience section and GitHub are revealed (claimed years count
  only as far as public activity supports them).
- **skill_match** = ``0.85 * required_share + 0.15 * nice_to_have_share``.
- **total** = ``(w_t*technical + w_e*experience + w_g*growth) * (SKILL_FLOOR + (1 - SKILL_FLOOR)*skill_match)``
  with the job description's weights.

:func:`estimate` applies the same formula to partially revealed data. Each
unrevealed input takes the role's published prior (``PRIORS``), the mean over
generated candidates, so an agent can rank candidates before it has seen
everything.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from env.models import CandidateProfile, JobDescription

SOURCES: tuple[str, ...] = ("github", "leetcode", "kaggle", "experience", "skills")
PLATFORMS: tuple[str, ...] = ("github", "leetcode", "kaggle")
RESUME_SOURCES: tuple[str, ...] = ("experience", "skills")

TECH_POWER = -2.0  # weighted power mean: below 1, a weak platform pulls technical down
SLACK_YEARS = 1.5
SKILL_FLOOR = 0.75

TECH_WEIGHTS: dict[str, dict[str, float]] = {
    "ml_engineer": {"leetcode": 0.30, "github": 0.30, "kaggle": 0.40},
    "data_scientist": {"leetcode": 0.30, "github": 0.25, "kaggle": 0.45},
    "backend_dev": {"leetcode": 0.40, "github": 0.35, "kaggle": 0.25},
    "frontend_dev": {"leetcode": 0.35, "github": 0.40, "kaggle": 0.25},
}

KAGGLE_RANK_VALUE: dict[str, float] = {
    "Novice": 0.0,
    "Contributor": 0.25,
    "Expert": 0.5,
    "Master": 0.75,
    "Grandmaster": 1.0,
}

# Role priors for unrevealed inputs: generator means over seeds 5000-5999,
# measured by `python scripts/difficulty_sweep.py --priors`. Platform entries
# are sub-scores; claimed_years is in years; skill_match is a share.
PRIORS: dict[str, dict[str, float]] = {
    "backend_dev": {"github": 0.449, "leetcode": 0.478, "kaggle": 0.426, "github_activity": 0.414,
                    "leetcode_practice": 0.399, "claimed_years": 4.229, "skill_match": 0.475},
    "data_scientist": {"github": 0.45, "leetcode": 0.48, "kaggle": 0.439, "github_activity": 0.416,
                       "leetcode_practice": 0.402, "claimed_years": 4.227, "skill_match": 0.478},
    "frontend_dev": {"github": 0.45, "leetcode": 0.476, "kaggle": 0.432, "github_activity": 0.414,
                     "leetcode_practice": 0.397, "claimed_years": 4.318, "skill_match": 0.48},
    "ml_engineer": {"github": 0.454, "leetcode": 0.48, "kaggle": 0.434, "github_activity": 0.412,
                    "leetcode_practice": 0.399, "claimed_years": 4.26, "skill_match": 0.481},
}


@dataclass(frozen=True)
class RubricBreakdown:
    """Rubric components and total for one candidate."""

    technical: float
    experience: float
    growth: float
    skill_match: float
    total: float
    per_source: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "technical": self.technical,
            "experience": self.experience,
            "growth": self.growth,
            "skill_match": self.skill_match,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------


def _h(x: float, s: float) -> float:
    x = max(0.0, float(x))
    return x / (x + s)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# per-source sub-scores (inputs are the revealed dicts, as agents see them)
# ---------------------------------------------------------------------------


def leetcode_subscore(stats: dict[str, Any]) -> float:
    return (
        0.5 * _sigmoid((stats["contest_rating"] - 1500) / 200)
        + 0.25 * _h(stats["hard"], 15)
        + 0.25 * stats["global_rank_percentile"] / 100
    )


def github_subscore(stats: dict[str, Any]) -> float:
    return 0.6 * _h(math.log1p(stats["stars_received"]), math.log1p(150)) + 0.4 * _h(stats["repos"], 15)


def github_activity(stats: dict[str, Any]) -> float:
    return 0.5 * _h(stats["commit_streak_days"], 45) + 0.5 * _h(stats["contributions_last_year"], 300)


def leetcode_practice(stats: dict[str, Any]) -> float:
    return _h(stats["problems_solved"], 150)


def growth_subscore(github: dict[str, Any] | None, leetcode: dict[str, Any] | None, jd: JobDescription) -> float:
    """Half GitHub activity, half LeetCode practice volume; priors fill a missing half."""
    prior = PRIORS[jd.role_type]
    a = github_activity(github) if github else prior["github_activity"]
    p = leetcode_practice(leetcode) if leetcode else prior["leetcode_practice"]
    return 0.5 * a + 0.5 * p


def kaggle_subscore(stats: dict[str, Any]) -> float:
    medals = stats.get("medals") or {}
    points = 3 * medals.get("gold", 0) + 2 * medals.get("silver", 0) + medals.get("bronze", 0)
    return (
        0.5 * stats["best_finish_percentile"] / 100
        + 0.3 * KAGGLE_RANK_VALUE.get(stats["rank"], 0.0)
        + 0.2 * _h(points, 6)
    )


def skill_match(skills: Iterable[str], jd: JobDescription) -> float:
    have = {s.lower() for s in skills}
    req = [s.lower() for s in jd.required_skills]
    nice = [s.lower() for s in jd.nice_to_have]
    req_share = sum(s in have for s in req) / len(req) if req else 0.0
    nice_share = sum(s in have for s in nice) / len(nice) if nice else 0.0
    return 0.85 * req_share + 0.15 * nice_share


def experience_score(claimed_years: float, years_active: float | None, jd: JobDescription) -> float:
    supported = claimed_years if years_active is None else min(claimed_years, years_active + SLACK_YEARS)
    return _h(supported, max(1, jd.experience_years))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def estimate(revealed: dict[str, Any] | None, jd: JobDescription) -> RubricBreakdown:
    """Apply the rubric to one candidate's revealed record.

    *revealed* has the observation's shape:
    ``{"resume_sections": {name: section}, "platforms": {name: stats}}``.
    Unrevealed inputs take ``PRIORS[jd.role_type]``. On a record with every
    source revealed this equals :func:`score` exactly.
    """
    revealed = revealed or {}
    platforms = revealed.get("platforms") or {}
    sections = revealed.get("resume_sections") or {}
    prior = PRIORS[jd.role_type]

    gh = platforms.get("github")
    sub = {
        "github": github_subscore(gh) if gh else prior["github"],
        "leetcode": leetcode_subscore(platforms["leetcode"]) if "leetcode" in platforms else prior["leetcode"],
        "kaggle": kaggle_subscore(platforms["kaggle"]) if "kaggle" in platforms else prior["kaggle"],
    }
    growth = growth_subscore(gh, platforms.get("leetcode"), jd)

    w = TECH_WEIGHTS[jd.role_type]
    technical = sum(w[k] * max(sub[k], 1e-6) ** TECH_POWER for k in PLATFORMS) ** (1.0 / TECH_POWER)

    exp_sec = sections.get("experience")
    if exp_sec:
        experience = experience_score(exp_sec["years_experience"], gh["years_active"] if gh else None, jd)
    else:  # claimed years unknown: the prior, uncapped (GitHub alone says nothing about the claim)
        experience = experience_score(prior["claimed_years"], None, jd)

    sk_sec = sections.get("skills")
    skills = skill_match(sk_sec.get("skills") or [], jd) if sk_sec else prior["skill_match"]

    base = jd.weight_technical * technical + jd.weight_experience * experience + jd.weight_growth * growth
    total = base * (SKILL_FLOOR + (1.0 - SKILL_FLOOR) * skills)
    per_source = {**sub, "experience": experience, "skills": skills, "growth": growth}
    if gh:
        per_source["github_activity"] = github_activity(gh)
    if "leetcode" in platforms:
        per_source["leetcode_practice"] = leetcode_practice(platforms["leetcode"])
    return RubricBreakdown(
        technical=technical,
        experience=experience,
        growth=growth,
        skill_match=skills,
        total=total,
        per_source=per_source,
    )


def reveal_record(profile: CandidateProfile, sources: Iterable[str]) -> dict[str, Any]:
    """The observation-shaped record an agent holds after revealing *sources*."""
    record: dict[str, Any] = {"resume_sections": {}, "platforms": {}}
    for src in sources:
        if src in PLATFORMS:
            record["platforms"][src] = getattr(profile, src).model_dump()
        elif src in profile.resume_sections:
            record["resume_sections"][src] = profile.resume_sections[src].model_dump()
    return record


def reveal_from_record(full: dict[str, Any], sources: Iterable[str]) -> dict[str, Any]:
    """Restrict a full record to *sources* (same shape as :func:`reveal_record`)."""
    record: dict[str, Any] = {"resume_sections": {}, "platforms": {}}
    for src in sources:
        if src in PLATFORMS and src in full.get("platforms", {}):
            record["platforms"][src] = full["platforms"][src]
        elif src in full.get("resume_sections", {}):
            record["resume_sections"][src] = full["resume_sections"][src]
    return record


def score(profile: CandidateProfile, jd: JobDescription) -> RubricBreakdown:
    """The exact rubric over a full profile (every source revealed)."""
    return estimate(reveal_record(profile, SOURCES), jd)


def source_of(action: Any) -> str | None:
    """The rubric source an action reveals, or ``None`` (decisions, education)."""
    kind = getattr(action, "type", None)
    if kind == "check_platform":
        return action.platform
    if kind == "read_resume_section" and action.section in RESUME_SOURCES:
        return action.section
    return None
