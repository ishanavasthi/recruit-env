"""Core RecruitEnv environment: reset(), step(), state()."""

from __future__ import annotations

import copy
import threading
from typing import Any

from env.models import (
    Action,
    CandidateProfile,
    CheckPlatformAction,
    EpisodeState,
    MakeDecisionAction,
    Observation,
    PublicState,
    ReadResumeSectionAction,
    Reward,
)
from env import rubric
from env.profile_factory import GENERATOR_VERSION, ProfileFactory, display_names


def _load_registries() -> tuple[dict, dict]:
    """Lazy import to break the env ↔ tasks circular dependency."""
    from tasks import GRADER_REGISTRY, TASK_REGISTRY
    return dict(TASK_REGISTRY), dict(GRADER_REGISTRY)


class RecruitmentEnvironment:
    """RL environment for candidate pipeline triage.

    Deterministic: reset(seed=42) always produces identical state.
    Thread-safe: all public methods are guarded by a lock.

    Scoring: the reward is sparse. Every step before the last earns 0.0; the
    terminal step earns the task grader's score, which is also the episode
    score and what ``grade_report()`` returns.

    Agents see ``Observation`` (from ``reset``/``step``) and ``state()``.
    ``verifier_state()`` holds the ground truth and is for graders and tests.
    """

    def __init__(self) -> None:
        self._state: EpisodeState | None = None
        self._initial_obs: Observation | None = None
        self._grade_result: Any = None
        self._factory = ProfileFactory()
        self._task_registry, self._grader_registry = _load_registries()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str,
        seed: int = 42,
        name_seed: int = 0,
        name_blind: bool = False,
    ) -> Observation:
        """Reset the environment for a new episode.

        Generates a job description and candidate pool deterministically
        from *seed*. *name_seed* redraws only the names (a name-swap replay:
        every signal and label stays the same). With *name_blind* the agent
        sees neutral labels ("Candidate 07") instead of names.

        Raises ``ValueError`` if *task_id* is not in the task registry.
        """
        with self._lock:
            if task_id not in self._task_registry:
                raise ValueError(
                    f"Unknown task_id '{task_id}'. "
                    f"Available: {list(self._task_registry.keys())}"
                )

            task = self._task_registry[task_id]
            ep = self._factory.generate_episode(
                seed=seed,
                quotas=task.quotas,
                required_conflicts=task.required_conflicts,
                name_seed=name_seed,
            )

            self._state = EpisodeState(
                task_id=task_id,
                seed=seed,
                step_number=0,
                max_steps=task.max_steps,
                job_description=ep.jd,
                candidates=ep.candidates,
                candidates_summary=display_names(ep.candidates, name_blind),
                revealed_data={},
                decisions_made={},
                quotas=task.quotas,
                conflicts=ep.conflicts,
                twins=ep.twins,
                decision_context={},
                name_seed=name_seed,
                name_blind=name_blind,
                generator_version=GENERATOR_VERSION,
                is_done=False,
                episode_score=None,
            )

            obs = self._build_observation()
            self._initial_obs = obs.model_copy(deep=True)
            return obs

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

        # 1. Execute action (mutate state)
        info: dict[str, Any] = {}
        if isinstance(action, ReadResumeSectionAction):
            self._exec_read_resume(action, s, info)
        elif isinstance(action, CheckPlatformAction):
            self._exec_check_platform(action, s, info)
        elif isinstance(action, MakeDecisionAction):
            self._exec_make_decision(action, s, info)

        # 2. Increment step counter
        s.step_number += 1

        # 3. Check terminal conditions
        all_decided = len(s.decisions_made) == len(s.candidates)
        budget_exhausted = s.step_number >= s.max_steps
        s.is_done = all_decided or budget_exhausted
        info["termination_reason"] = (
            "all_decided" if all_decided
            else "budget_exhausted" if budget_exhausted
            else None
        )

        # 4. Build reward: sparse, the grader's score on the terminal step
        if s.is_done:
            score = self._grade_locked()
            s.episode_score = score
            reward = Reward(
                step_reward=score,
                cumulative_reward=score,
                breakdown={"grader_score": score},
            )
        else:
            reward = Reward()

        obs = self._build_observation()
        return obs, reward, s.is_done, info

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> PublicState:
        """Return the agent-visible episode state (no ground truth).

        Raises ``ValueError`` if no active episode.
        """
        with self._lock:
            s = self._require_state()
            return PublicState(
                task_id=s.task_id,
                seed=s.seed,
                step_number=s.step_number,
                max_steps=s.max_steps,
                job_description=s.job_description.model_copy(deep=True),
                candidates_summary=[dict(c) for c in s.candidates_summary],
                revealed_data=_copy_revealed(s.revealed_data),
                decisions_made=dict(s.decisions_made),
                quotas=dict(s.quotas),
                name_blind=s.name_blind,
                generator_version=s.generator_version,
                is_done=s.is_done,
                episode_score=s.episode_score,
            )

    def verifier_state(self) -> EpisodeState:
        """Return a copy of the full internal state, including ground truth.

        For graders, tests and post-episode analysis only. Never hand it to
        an agent during an episode.

        Raises ``ValueError`` if no active episode.
        """
        with self._lock:
            return self._require_state().model_copy(deep=True)

    def grade_report(self) -> dict[str, Any]:
        """Return the score and a per-candidate breakdown of a finished episode.

        The breakdown includes ground-truth labels, so it is only available
        once the episode is done.

        Includes the verifier's named checks (``checks``), gates (``gates``)
        and the score before gates were applied.

        Raises ``ValueError`` if no active episode, ``RuntimeError`` if the
        episode is still running.
        """
        with self._lock:
            s = self._require_state()
            if not s.is_done:
                raise RuntimeError(
                    "Episode is not complete. Finish the episode before grading."
                )
            assert s.episode_score is not None
            task = self._task_registry[s.task_id]
            gt_map = {c.id: c.ground_truth_label for c in s.candidates}
            return {
                "task_id": s.task_id,
                "score": s.episode_score,
                "success": s.episode_score >= task.success_threshold,
                "threshold": task.success_threshold,
                "decisions": dict(s.decisions_made),
                "ground_truth": gt_map,
                "correct": sum(
                    1 for cid, d in s.decisions_made.items() if gt_map.get(cid) == d
                ),
                "total": len(s.candidates),
                "steps_used": s.step_number,
                "max_steps": s.max_steps,
                "score_before_gates": round(self._grade_result.score_before_gates, 6),
                "checks": [c.model_dump() for c in self._grade_result.checks],
                "gates": [g.model_dump() for g in self._grade_result.gates],
            }

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
        revealed = _revealed_for(s, cid)

        section_name = action.section
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

        data = section.model_dump()
        revealed["resume_sections"][section_name] = data
        info["section_found"] = True
        info["data"] = dict(data)

    @staticmethod
    def _exec_check_platform(
        action: CheckPlatformAction,
        s: EpisodeState,
        info: dict[str, Any],
    ) -> None:
        cid = action.candidate_id
        candidate = _find_candidate(s, cid)
        revealed = _revealed_for(s, cid)

        platform = action.platform
        if platform in revealed["platforms"]:
            info["already_revealed"] = True
            info["data"] = {}
            return

        stats = getattr(candidate, platform).model_dump()
        revealed["platforms"][platform] = stats
        info["data"] = dict(stats)

    @staticmethod
    def _exec_make_decision(
        action: MakeDecisionAction,
        s: EpisodeState,
        info: dict[str, Any],
    ) -> None:
        cid = action.candidate_id
        s.decisions_made[cid] = action.decision
        s.decision_context[cid] = revealed_sources(s.revealed_data.get(cid))
        info["decision_recorded"] = True

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _require_state(self) -> EpisodeState:
        if self._state is None:
            raise ValueError("No active episode. Call reset() first.")
        return self._state

    def _grade_locked(self) -> float:
        s = self._state
        assert s is not None and self._initial_obs is not None
        grader = self._grader_registry[s.task_id]
        self._grade_result = grader.evaluate(self._initial_obs, s.model_copy(deep=True))
        return round(self._grade_result.score, 6)

    def _build_observation(self) -> Observation:
        """Build an Observation from the current state."""
        s = self._state
        assert s is not None
        return Observation(
            task_id=s.task_id,
            step_number=s.step_number,
            steps_remaining=s.max_steps - s.step_number,
            job_description=s.job_description.model_copy(deep=True),
            candidates_summary=[dict(c) for c in s.candidates_summary],
            revealed_data=_copy_revealed(s.revealed_data),
            decisions_made=dict(s.decisions_made),
            quotas=dict(s.quotas),
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


def _revealed_for(s: EpisodeState, cid: str) -> dict[str, dict[str, Any]]:
    """Return (creating if needed) the revealed-data record for a candidate."""
    return s.revealed_data.setdefault(cid, {"resume_sections": {}, "platforms": {}})


def revealed_sources(record: dict[str, Any] | None) -> list[str]:
    """Sorted rubric sources present in one candidate's revealed record."""
    if not record:
        return []
    have = set(record.get("platforms", {})) | set(record.get("resume_sections", {}))
    return sorted(s for s in rubric.SOURCES if s in have)


def _copy_revealed(revealed: dict[str, Any]) -> dict[str, Any]:
    """Deep copy of revealed data (nested dicts and lists of plain values)."""
    return copy.deepcopy(revealed)
