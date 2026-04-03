"""Core RecruitEnv environment: reset(), step(), state()."""

from __future__ import annotations

import threading
from typing import Any

from env.fairness import FairnessChecker
from env.models import (
    Action,
    CandidateProfile,
    CheckPlatformAction,
    EpisodeState,
    MakeDecisionAction,
    Observation,
    ReadResumeSectionAction,
    Reward,
    ScoreDimensionAction,
)
from env.profile_factory import ProfileFactory
from env.rewards import RewardCalculator
from tasks import TASK_REGISTRY


class RecruitmentEnvironment:
    """OpenEnv-compliant RL environment for candidate pipeline triage.

    Deterministic: reset(seed=42) always produces identical state.
    Thread-safe: all public methods are guarded by a lock.
    """

    def __init__(self) -> None:
        self._state: EpisodeState | None = None
        self._factory = ProfileFactory()
        self._reward_calc = RewardCalculator()
        self._fairness = FairnessChecker()
        self._task_registry = dict(TASK_REGISTRY)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: str, seed: int = 42) -> Observation:
        """Reset the environment for a new episode.

        Generates a job description and candidate pool deterministically
        from the given seed, then returns the initial observation.

        Raises ``ValueError`` if *task_id* is not in the task registry.
        """
        with self._lock:
            if task_id not in self._task_registry:
                raise ValueError(
                    f"Unknown task_id '{task_id}'. "
                    f"Available: {list(self._task_registry.keys())}"
                )

            task = self._task_registry[task_id]

            # Derive label distribution from task config.
            # select_quota  → shortlist count
            # remaining     → split roughly between hold and reject
            shortlist_n = task.select_quota
            remaining = task.num_candidates - shortlist_n
            hold_n = remaining // 2
            reject_n = remaining - hold_n
            label_distribution = {
                "shortlist": shortlist_n,
                "hold": hold_n,
                "reject": reject_n,
            }

            # Pick a role type deterministically from the seed so the same
            # seed always produces the same job description.
            role_types = ["ml_engineer", "frontend_dev", "backend_dev", "data_scientist"]
            role_type = role_types[seed % len(role_types)]

            jd = self._factory.generate_job_description(seed=seed, role_type=role_type)
            pool = self._factory.generate_pool(
                seed=seed,
                count=task.num_candidates,
                label_distribution=label_distribution,
            )

            summary = [{"id": c.id, "name": c.name} for c in pool]

            self._state = EpisodeState(
                task_id=task_id,
                seed=seed,
                step_number=0,
                max_steps=task.step_budget,
                job_description=jd,
                candidates=pool,
                candidates_summary=summary,
                revealed_data={},
                decisions_made={},
                scores_recorded={},
                is_done=False,
                episode_score=None,
            )

            return self._build_observation()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self, action: Action
    ) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """Apply an agent action and return (observation, reward, done, info).

        Raises ``RuntimeError`` if no active episode.
        Raises ``ValueError`` on illegal actions (unknown candidate, duplicate
        decision, etc.).
        """
        with self._lock:
            return self._step_locked(action)

    def _step_locked(
        self, action: Action
    ) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """Unlocked implementation of step()."""
        s = self._state
        if s is None:
            raise RuntimeError("No active episode. Call reset() first.")
        if s.is_done:
            raise RuntimeError("Episode is already done. Call reset().")

        cid = action.candidate_id
        valid_ids = {c.id for c in s.candidates}
        if cid not in valid_ids:
            raise ValueError(
                f"Unknown candidate_id '{cid}'. Valid: {sorted(valid_ids)}"
            )

        if isinstance(action, MakeDecisionAction) and cid in s.decisions_made:
            raise ValueError(
                f"Candidate '{cid}' already has a decision: "
                f"'{s.decisions_made[cid]}'"
            )

        # 1. Compute step reward BEFORE mutating state (novelty check)
        step_reward = self._reward_calc.calculate_step_reward(action, s)

        # 2. Execute action (mutate state)
        info: dict[str, Any] = {}
        if isinstance(action, ReadResumeSectionAction):
            self._exec_read_resume(action, s, info)
        elif isinstance(action, CheckPlatformAction):
            self._exec_check_platform(action, s, info)
        elif isinstance(action, ScoreDimensionAction):
            self._exec_score_dimension(action, s, info)
        elif isinstance(action, MakeDecisionAction):
            self._exec_make_decision(action, s, info)

        # 3. Increment step counter
        s.step_number += 1

        # 4. Check terminal conditions
        all_decided = len(s.decisions_made) == len(s.candidates)
        budget_exhausted = s.step_number >= s.max_steps
        s.is_done = all_decided or budget_exhausted
        info["termination_reason"] = (
            "all_decided" if all_decided
            else "budget_exhausted" if budget_exhausted
            else None
        )

        # 5. Build reward
        if s.is_done:
            terminal = self._reward_calc.calculate_terminal_reward(
                s.decisions_made, s.candidates, s.job_description,
            )
            gt_map = {c.id: c.ground_truth_label for c in s.candidates}
            correct = sum(
                1 for cid_, d in s.decisions_made.items()
                if gt_map.get(cid_) == d
            )
            efficiency = self._reward_calc.calculate_efficiency_bonus(
                s.step_number, s.max_steps, correct, len(s.candidates),
            )
            penalty = self._fairness.compute_penalty(
                s.decisions_made, s.candidates,
            )
            accuracy = terminal.get("total", 0.0)
            raw = accuracy + efficiency - penalty
            clamped = max(0.0, min(1.0, raw))
            s.episode_score = round(clamped, 6)

            breakdown = {k: v for k, v in terminal.items() if k != "total"}
            breakdown["accuracy_total"] = accuracy
            breakdown["efficiency_bonus"] = efficiency
            breakdown["fairness_penalty"] = penalty

            reward = Reward(
                step_reward=step_reward,
                cumulative_reward=s.episode_score,
                fairness_penalty=round(max(0.0, min(1.0, penalty)), 6),
                accuracy_bonus=round(max(0.0, min(1.0, accuracy)), 6),
                breakdown=breakdown,
            )
        else:
            reward = Reward(step_reward=step_reward)

        obs = self._build_observation()
        return obs, reward, s.is_done, info

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> EpisodeState:
        """Return a copy of the current state (never exposes the mutable original).

        Raises ``ValueError`` if no active episode.
        """
        with self._lock:
            if self._state is None:
                raise ValueError("No active episode. Call reset() first.")
            return self._state.model_copy(deep=True)

    # ------------------------------------------------------------------
    # action executors
    # ------------------------------------------------------------------

    @staticmethod
    def _exec_read_resume(
        action: ReadResumeSectionAction,
        s: EpisodeState,
        info: dict[str, Any],
    ) -> None:
        cid = action.candidate_id
        candidate = _find_candidate(s, cid)

        if cid not in s.revealed_data:
            s.revealed_data[cid] = {"resume_sections": [], "platforms": []}

        section_name = action.section
        revealed = s.revealed_data[cid]

        if section_name in revealed["resume_sections"]:
            info["already_revealed"] = True
            info["data"] = {}
            return

        section = candidate.resume_sections.get(section_name)
        if section is None:
            info["section_found"] = False
            info["data"] = {}
            info["available_sections"] = list(candidate.resume_sections.keys())
            return

        revealed["resume_sections"].append(section_name)
        info["section_found"] = True
        info["data"] = section.model_dump()

    @staticmethod
    def _exec_check_platform(
        action: CheckPlatformAction,
        s: EpisodeState,
        info: dict[str, Any],
    ) -> None:
        cid = action.candidate_id
        candidate = _find_candidate(s, cid)

        if cid not in s.revealed_data:
            s.revealed_data[cid] = {"resume_sections": [], "platforms": []}

        platform = action.platform
        revealed = s.revealed_data[cid]

        if platform in revealed["platforms"]:
            info["already_revealed"] = True
            info["data"] = {}
            return

        revealed["platforms"].append(platform)

        if platform == "github":
            info["data"] = candidate.github.model_dump()
        elif platform == "leetcode":
            info["data"] = candidate.leetcode.model_dump()
        elif platform == "kaggle":
            info["data"] = candidate.kaggle.model_dump()

    @staticmethod
    def _exec_score_dimension(
        action: ScoreDimensionAction,
        s: EpisodeState,
        info: dict[str, Any],
    ) -> None:
        cid = action.candidate_id
        if cid not in s.scores_recorded:
            s.scores_recorded[cid] = {}
        s.scores_recorded[cid][action.dimension] = action.score
        info["recorded"] = True

    @staticmethod
    def _exec_make_decision(
        action: MakeDecisionAction,
        s: EpisodeState,
        info: dict[str, Any],
    ) -> None:
        s.decisions_made[action.candidate_id] = action.decision
        info["decision_recorded"] = True

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Build an Observation from the current state."""
        s = self._state
        assert s is not None
        return Observation(
            task_id=s.task_id,
            step_number=s.step_number,
            steps_remaining=s.max_steps - s.step_number,
            job_description=s.job_description.model_copy(deep=True),
            candidates_summary=list(s.candidates_summary),
            revealed_data={k: _deep_copy_dict(v) for k, v in s.revealed_data.items()},
            decisions_made=dict(s.decisions_made),
            scores_recorded={
                k: dict(v) for k, v in s.scores_recorded.items()
            },
            done=s.is_done,
        )


# ------------------------------------------------------------------
# module-level helpers
# ------------------------------------------------------------------


def _find_candidate(s: EpisodeState, cid: str) -> CandidateProfile:
    """Find a candidate by id in the state's candidate list."""
    for c in s.candidates:
        if c.id == cid:
            return c
    raise ValueError(f"Candidate '{cid}' not found")


def _deep_copy_dict(d: dict) -> dict:
    """Simple recursive copy for nested dicts/lists (no custom objects)."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out
