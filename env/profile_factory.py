"""Synthetic candidate profile generator.

Uses numpy seeded RNG for full determinism.  NO live HTTP calls.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.random import Generator

from env.models import (
    CandidateProfile,
    GitHubStats,
    JobDescription,
    KaggleStats,
    LeetCodeStats,
    ResumeSection,
)

# ---------------------------------------------------------------------------
# Hardcoded diverse name pool (50 names)
# ---------------------------------------------------------------------------

_NAMES: list[str] = [
    "Aisha Patel",
    "Carlos Rivera",
    "Wei Zhang",
    "Fatima Al-Rashid",
    "Dmitri Volkov",
    "Priya Sharma",
    "James O'Brien",
    "Yuki Tanaka",
    "Amara Okafor",
    "Lucas Fernandez",
    "Sara Johansson",
    "Raj Krishnamurthy",
    "Elena Popescu",
    "Mohammed Hassan",
    "Sofia Garcia",
    "Kenji Nakamura",
    "Olga Petrova",
    "David Kim",
    "Ines Moreau",
    "Tariq Mahmoud",
    "Lena Fischer",
    "Andrei Novak",
    "Maria Santos",
    "Chen Wei-Lin",
    "Nadia Bousaid",
    "Tomasz Kowalski",
    "Ananya Gupta",
    "Erik Lindqvist",
    "Zara Hussain",
    "Marco Rossi",
    "Hana Yilmaz",
    "Sanjay Mehta",
    "Chloe Dubois",
    "Kwame Asante",
    "Mika Leinonen",
    "Rania Khoury",
    "Ivan Petrov",
    "Leila Ahmadi",
    "Oscar Bergstrom",
    "Mei-Ling Wu",
    "Alejandro Cruz",
    "Naomi Watanabe",
    "Hassan El-Sayed",
    "Julia Nowak",
    "Vikram Singh",
    "Clara Hoffmann",
    "Emeka Nwosu",
    "Sakura Ito",
    "Rafael Torres",
    "Lin Xiaoming",
]

# ---------------------------------------------------------------------------
# Language / skill / project pools per role type
# ---------------------------------------------------------------------------

_LANGUAGES_BY_ROLE: dict[str, list[str]] = {
    "ml_engineer": ["Python", "C++", "Julia", "R", "CUDA"],
    "frontend_dev": ["TypeScript", "JavaScript", "CSS", "HTML", "Dart"],
    "backend_dev": ["Go", "Python", "Java", "Rust", "C++"],
    "data_scientist": ["Python", "R", "SQL", "Julia", "Scala"],
}

_REQUIRED_SKILLS: dict[str, list[str]] = {
    "ml_engineer": ["PyTorch", "TensorFlow", "scikit-learn", "MLOps", "Docker"],
    "frontend_dev": ["React", "TypeScript", "CSS", "Webpack", "Testing"],
    "backend_dev": ["REST APIs", "SQL", "Docker", "CI/CD", "System Design"],
    "data_scientist": ["pandas", "SQL", "statistics", "visualization", "A/B testing"],
}

_NICE_TO_HAVE: dict[str, list[str]] = {
    "ml_engineer": ["Triton", "ONNX", "Ray", "Kubernetes", "Spark"],
    "frontend_dev": ["Next.js", "GraphQL", "Storybook", "Figma", "A11y"],
    "backend_dev": ["gRPC", "Kafka", "Redis", "Terraform", "Kubernetes"],
    "data_scientist": ["Spark", "dbt", "Airflow", "Tableau", "Bayesian methods"],
}

_NOTABLE_PROJECTS: list[str] = [
    "distributed-cache",
    "ml-pipeline-framework",
    "real-time-dashboard",
    "api-gateway",
    "image-classifier",
    "search-engine",
    "chat-application",
    "recommendation-engine",
    "data-lake-etl",
    "cli-toolkit",
    "compiler-frontend",
    "graph-database",
    "neural-style-transfer",
    "ci-cd-toolkit",
    "event-sourcing-lib",
]

_UNIVERSITIES: list[str] = [
    "MIT",
    "Stanford University",
    "IIT Bombay",
    "ETH Zurich",
    "University of Toronto",
    "Tsinghua University",
    "UC Berkeley",
    "National University of Singapore",
    "University of Waterloo",
    "Georgia Tech",
]

_DEGREES: list[str] = [
    "B.S. Computer Science",
    "M.S. Computer Science",
    "B.S. Software Engineering",
    "M.S. Data Science",
    "B.Tech Computer Engineering",
    "M.S. Machine Learning",
    "B.S. Mathematics & CS",
    "Ph.D. Computer Science",
]

# ---------------------------------------------------------------------------
# Role metadata for job description generation
# ---------------------------------------------------------------------------

_ROLE_META: dict[str, dict] = {
    "ml_engineer": {
        "role": "Machine Learning Engineer",
        "seniority": "senior",
        "experience_years": 4,
        "weight_technical": 0.50,
        "weight_experience": 0.25,
        "weight_growth": 0.25,
    },
    "frontend_dev": {
        "role": "Frontend Developer",
        "seniority": "mid",
        "experience_years": 2,
        "weight_technical": 0.35,
        "weight_experience": 0.35,
        "weight_growth": 0.30,
    },
    "backend_dev": {
        "role": "Backend Developer",
        "seniority": "mid",
        "experience_years": 3,
        "weight_technical": 0.45,
        "weight_experience": 0.30,
        "weight_growth": 0.25,
    },
    "data_scientist": {
        "role": "Data Scientist",
        "seniority": "mid",
        "experience_years": 2,
        "weight_technical": 0.40,
        "weight_experience": 0.30,
        "weight_growth": 0.30,
    },
}


# ---------------------------------------------------------------------------
# ProfileFactory
# ---------------------------------------------------------------------------

class ProfileFactory:
    """Generates synthetic candidate profiles and job descriptions from a seed.

    All randomness flows through ``numpy.random.default_rng(seed)`` so that
    identical seeds always produce identical output.  NO live HTTP calls.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng: Generator = np.random.default_rng(seed)

    # -- public API ---------------------------------------------------------

    def generate_pool(
        self,
        seed: int,
        count: int,
        label_distribution: dict[str, int],
    ) -> list[CandidateProfile]:
        """Generate a pool of candidates matching *label_distribution*.

        Parameters
        ----------
        seed:
            RNG seed — reseeds the internal generator for determinism.
        count:
            Total number of candidates (must equal sum of distribution values).
        label_distribution:
            Mapping of label → count, e.g. ``{"shortlist": 3, "hold": 4, "reject": 3}``.

        Returns
        -------
        list[CandidateProfile]
            Shuffled list of *count* fully-populated candidate profiles.
        """
        dist_total = sum(label_distribution.values())
        if dist_total != count:
            raise ValueError(
                f"label_distribution sums to {dist_total}, expected {count}"
            )

        self._rng = np.random.default_rng(seed)

        # Pick names deterministically without replacement
        name_indices = self._rng.choice(len(_NAMES), size=count, replace=False)
        names = [_NAMES[i] for i in name_indices]

        # Build flat label list, then shuffle
        labels: list[Literal["shortlist", "hold", "reject"]] = []
        for label, n in label_distribution.items():
            labels.extend([label] * n)  # type: ignore[arg-type]
        self._rng.shuffle(labels)  # type: ignore[arg-type]

        profiles: list[CandidateProfile] = []
        for idx, (name, label) in enumerate(zip(names, labels)):
            cid = f"candidate_{idx + 1:03d}"
            profile = self._build_profile(cid, name, label)
            profiles.append(profile)

        return profiles

    def generate_job_description(
        self,
        seed: int,
        role_type: Literal["ml_engineer", "frontend_dev", "backend_dev", "data_scientist"],
    ) -> JobDescription:
        """Create a synthetic job description for *role_type*.

        Parameters
        ----------
        seed:
            RNG seed — reseeds the internal generator for determinism.
        role_type:
            One of ``"ml_engineer"``, ``"frontend_dev"``, ``"backend_dev"``,
            ``"data_scientist"``.
        """
        self._rng = np.random.default_rng(seed)
        meta = _ROLE_META[role_type]
        required = list(_REQUIRED_SKILLS[role_type])
        nice = list(_NICE_TO_HAVE[role_type])
        # Shuffle nice-to-haves so different seeds surface different subsets
        self._rng.shuffle(nice)
        nice = nice[: self._rng.integers(2, len(nice) + 1)]

        return JobDescription(
            role=meta["role"],
            required_skills=required,
            nice_to_have=list(nice),
            experience_years=meta["experience_years"],
            seniority=meta["seniority"],
            weight_technical=meta["weight_technical"],
            weight_experience=meta["weight_experience"],
            weight_growth=meta["weight_growth"],
        )

    # -- internal builders --------------------------------------------------

    def _build_profile(
        self,
        cid: str,
        name: str,
        label: Literal["shortlist", "hold", "reject"],
    ) -> CandidateProfile:
        """Build a single fully-populated CandidateProfile."""
        if label == "shortlist":
            return self._build_shortlist(cid, name)
        if label == "hold":
            return self._build_hold(cid, name)
        return self._build_reject(cid, name)

    # ---- shortlist (senior tier) ------------------------------------------

    def _build_shortlist(self, cid: str, name: str) -> CandidateProfile:
        rng = self._rng

        leetcode = self._senior_leetcode(rng)
        github = self._senior_github(rng)
        kaggle = self._senior_kaggle(rng)
        years = float(rng.uniform(5.0, 10.0))
        resume = self._build_resume(rng, years, skill_coverage=1.0)

        gt_scores = self._compute_ground_truth_scores(
            leetcode, github, kaggle, years, resume
        )

        return CandidateProfile(
            id=cid,
            name=name,
            resume_sections=resume,
            github=github,
            leetcode=leetcode,
            kaggle=kaggle,
            ground_truth_label="shortlist",
            ground_truth_scores=gt_scores,
        )

    # ---- hold (mid tier, mixed signals) -----------------------------------

    def _build_hold(self, cid: str, name: str) -> CandidateProfile:
        rng = self._rng

        leetcode = self._mid_leetcode(rng)
        github = self._mid_github(rng)
        kaggle = self._mid_kaggle(rng)
        years = float(rng.uniform(2.0, 5.0))
        resume = self._build_resume(rng, years, skill_coverage=0.7)

        gt_scores = self._compute_ground_truth_scores(
            leetcode, github, kaggle, years, resume
        )

        return CandidateProfile(
            id=cid,
            name=name,
            resume_sections=resume,
            github=github,
            leetcode=leetcode,
            kaggle=kaggle,
            ground_truth_label="hold",
            ground_truth_scores=gt_scores,
        )

    # ---- reject (junior tier or mismatched) -------------------------------

    def _build_reject(self, cid: str, name: str) -> CandidateProfile:
        rng = self._rng

        leetcode = self._junior_leetcode(rng)
        github = self._junior_github(rng)
        kaggle = self._junior_kaggle(rng)
        years = float(rng.uniform(0.0, 2.0))
        resume = self._build_resume(rng, years, skill_coverage=0.3)

        gt_scores = self._compute_ground_truth_scores(
            leetcode, github, kaggle, years, resume
        )

        return CandidateProfile(
            id=cid,
            name=name,
            resume_sections=resume,
            github=github,
            leetcode=leetcode,
            kaggle=kaggle,
            ground_truth_label="reject",
            ground_truth_scores=gt_scores,
        )

    # -- platform stat builders per tier ------------------------------------

    @staticmethod
    def _senior_leetcode(rng: Generator) -> LeetCodeStats:
        hard = int(rng.integers(20, 60))
        medium = int(rng.integers(80, 180))
        easy = int(rng.integers(60, 160))
        total = easy + medium + hard
        return LeetCodeStats(
            problems_solved=total,
            easy=easy,
            medium=medium,
            hard=hard,
            contest_rating=int(rng.integers(1600, 2001)),
            global_rank_percentile=round(float(rng.uniform(85.0, 99.5)), 1),
        )

    @staticmethod
    def _mid_leetcode(rng: Generator) -> LeetCodeStats:
        hard = int(rng.integers(5, 21))
        medium = int(rng.integers(40, 120))
        easy = int(rng.integers(30, 110))
        total = easy + medium + hard
        return LeetCodeStats(
            problems_solved=total,
            easy=easy,
            medium=medium,
            hard=hard,
            contest_rating=int(rng.integers(1200, 1601)),
            global_rank_percentile=round(float(rng.uniform(50.0, 85.0)), 1),
        )

    @staticmethod
    def _junior_leetcode(rng: Generator) -> LeetCodeStats:
        hard = int(rng.integers(0, 5))
        medium = int(rng.integers(5, 40))
        easy = int(rng.integers(10, 60))
        total = easy + medium + hard
        return LeetCodeStats(
            problems_solved=total,
            easy=easy,
            medium=medium,
            hard=hard,
            contest_rating=int(rng.integers(800, 1200)),
            global_rank_percentile=round(float(rng.uniform(5.0, 50.0)), 1),
        )

    @staticmethod
    def _senior_github(rng: Generator) -> GitHubStats:
        repos = int(rng.integers(30, 61))
        all_langs = ["Python", "TypeScript", "Go", "Rust", "Java", "C++", "Kotlin"]
        lang_idx = rng.choice(len(all_langs), size=min(4, len(all_langs)), replace=False)
        top_languages = [all_langs[i] for i in lang_idx]
        proj_idx = rng.choice(len(_NOTABLE_PROJECTS), size=int(rng.integers(2, 5)), replace=False)
        notable = [_NOTABLE_PROJECTS[i] for i in proj_idx]
        return GitHubStats(
            repos=repos,
            top_languages=top_languages,
            commit_streak_days=int(rng.integers(60, 181)),
            stars_received=int(rng.integers(200, 1001)),
            notable_projects=notable,
            contributions_last_year=int(rng.integers(500, 1500)),
        )

    @staticmethod
    def _mid_github(rng: Generator) -> GitHubStats:
        repos = int(rng.integers(10, 31))
        all_langs = ["Python", "JavaScript", "Java", "Go", "TypeScript", "C#"]
        lang_idx = rng.choice(len(all_langs), size=min(3, len(all_langs)), replace=False)
        top_languages = [all_langs[i] for i in lang_idx]
        proj_idx = rng.choice(len(_NOTABLE_PROJECTS), size=int(rng.integers(1, 3)), replace=False)
        notable = [_NOTABLE_PROJECTS[i] for i in proj_idx]
        return GitHubStats(
            repos=repos,
            top_languages=top_languages,
            commit_streak_days=int(rng.integers(10, 61)),
            stars_received=int(rng.integers(20, 201)),
            notable_projects=notable,
            contributions_last_year=int(rng.integers(100, 500)),
        )

    @staticmethod
    def _junior_github(rng: Generator) -> GitHubStats:
        repos = int(rng.integers(1, 10))
        all_langs = ["Python", "JavaScript", "HTML", "CSS", "Java"]
        lang_idx = rng.choice(len(all_langs), size=min(2, len(all_langs)), replace=False)
        top_languages = [all_langs[i] for i in lang_idx]
        return GitHubStats(
            repos=repos,
            top_languages=top_languages,
            commit_streak_days=int(rng.integers(0, 10)),
            stars_received=int(rng.integers(0, 20)),
            notable_projects=[],
            contributions_last_year=int(rng.integers(10, 100)),
        )

    @staticmethod
    def _senior_kaggle(rng: Generator) -> KaggleStats:
        rank = str(rng.choice(["Expert", "Master", "Grandmaster"]))
        return KaggleStats(
            rank=rank,
            competitions_entered=int(rng.integers(10, 40)),
            best_finish_percentile=round(float(rng.uniform(85.0, 99.0)), 1),
            medals={
                "gold": int(rng.integers(1, 6)),
                "silver": int(rng.integers(2, 8)),
                "bronze": int(rng.integers(3, 12)),
            },
        )

    @staticmethod
    def _mid_kaggle(rng: Generator) -> KaggleStats:
        rank = str(rng.choice(["Contributor", "Expert"]))
        return KaggleStats(
            rank=rank,
            competitions_entered=int(rng.integers(3, 12)),
            best_finish_percentile=round(float(rng.uniform(40.0, 80.0)), 1),
            medals={
                "silver": int(rng.integers(0, 3)),
                "bronze": int(rng.integers(0, 4)),
            },
        )

    @staticmethod
    def _junior_kaggle(rng: Generator) -> KaggleStats:
        rank = str(rng.choice(["Novice", "Contributor"]))
        return KaggleStats(
            rank=rank,
            competitions_entered=int(rng.integers(0, 3)),
            best_finish_percentile=round(float(rng.uniform(0.0, 40.0)), 1),
            medals={},
        )

    # -- resume builder -----------------------------------------------------

    def _build_resume(
        self,
        rng: Generator,
        years: float,
        skill_coverage: float,
    ) -> dict[str, ResumeSection]:
        """Build resume sections.

        *skill_coverage* in [0, 1] controls what fraction of a generic skill
        list appears in the skills section content.
        """
        # Education
        uni = _UNIVERSITIES[int(rng.integers(0, len(_UNIVERSITIES)))]
        degree = _DEGREES[int(rng.integers(0, len(_DEGREES)))]
        grad_year = int(2024 - years)

        # Skills (draw from a generic pool; coverage controls count)
        all_skills = [
            "Python", "JavaScript", "SQL", "Docker", "Kubernetes",
            "React", "Node.js", "AWS", "Git", "CI/CD",
            "TensorFlow", "PyTorch", "REST APIs", "GraphQL", "Linux",
        ]
        n_skills = max(1, int(len(all_skills) * skill_coverage))
        skill_idx = rng.choice(len(all_skills), size=n_skills, replace=False)
        chosen_skills = [all_skills[i] for i in sorted(skill_idx)]

        # Experience blurb
        role_choices = [
            "Software Engineer", "ML Engineer", "Data Analyst",
            "Backend Developer", "Frontend Developer", "Full-Stack Developer",
        ]
        role = role_choices[int(rng.integers(0, len(role_choices)))]
        company_choices = [
            "a Series-B startup", "a Fortune 500 company", "a mid-size SaaS firm",
            "an open-source foundation", "a consulting agency", "a fintech company",
        ]
        company = company_choices[int(rng.integers(0, len(company_choices)))]

        sections: dict[str, ResumeSection] = {
            "education": ResumeSection(
                section_name="education",
                content=f"{degree} from {uni}, graduated {grad_year}.",
                years_experience=0.0,
            ),
            "experience": ResumeSection(
                section_name="experience",
                content=(
                    f"{role} at {company} for {years:.1f} years. "
                    f"Worked on production systems handling real-world traffic."
                ),
                years_experience=round(years, 1),
            ),
            "skills": ResumeSection(
                section_name="skills",
                content=", ".join(chosen_skills),
                years_experience=0.0,
            ),
        }
        return sections

    # -- ground truth score computation -------------------------------------

    @staticmethod
    def _compute_ground_truth_scores(
        leetcode: LeetCodeStats,
        github: GitHubStats,
        kaggle: KaggleStats,
        years: float,
        resume: dict[str, ResumeSection],
    ) -> dict[str, float]:
        """Derive ground-truth dimension scores from raw profile data.

        - technical: weighted average of normalised leetcode + github signals
        - experience: years_experience / 10, capped at 1.0
        - growth: streak consistency + recent activity score
        """
        # -- technical --
        # Normalise leetcode: problems_solved / 400 (senior max ~400)
        lc_norm = min(leetcode.problems_solved / 400.0, 1.0)
        # Normalise hard ratio: hard / problems_solved (if any)
        hard_ratio = (
            leetcode.hard / leetcode.problems_solved
            if leetcode.problems_solved > 0
            else 0.0
        )
        # Normalise github repos: repos / 60
        gh_norm = min(github.repos / 60.0, 1.0)
        # Normalise stars: stars / 1000
        stars_norm = min(github.stars_received / 1000.0, 1.0)

        technical = 0.30 * lc_norm + 0.20 * hard_ratio + 0.30 * gh_norm + 0.20 * stars_norm
        technical = round(min(max(technical, 0.0), 1.0), 4)

        # -- experience --
        experience = round(min(years / 10.0, 1.0), 4)

        # -- growth --
        # Streak normalised to 180 days (senior max)
        streak_norm = min(github.commit_streak_days / 180.0, 1.0)
        # Recent activity normalised to 1500 contributions
        activity_norm = min(github.contributions_last_year / 1500.0, 1.0)
        # Contest rating as growth proxy, normalised to 2000
        contest_norm = min(leetcode.contest_rating / 2000.0, 1.0)

        growth = 0.40 * streak_norm + 0.35 * activity_norm + 0.25 * contest_norm
        growth = round(min(max(growth, 0.0), 1.0), 4)

        return {
            "technical": technical,
            "experience": experience,
            "growth": growth,
        }
