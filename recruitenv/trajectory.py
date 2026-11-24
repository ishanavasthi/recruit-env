"""Transition logs: one JSONL record per step, replayable exactly.

A log has three kinds of record, each a JSON object with a ``kind`` field:

``header``
    ``episode_id``, ``task``, ``seed``, ``name_seed``, ``name_blind``,
    ``generator_version``, ``package_version``, ``agent``, ``model`` and
    ``started_at``.
``step``
    ``step`` (the step number after the action, or the unchanged number for a
    rejected action), ``action``, ``revealed`` (only what this step revealed),
    ``obs_sha256`` (hash of the canonical observation JSON), ``reward``,
    ``steps_remaining``, ``done`` and ``error`` (the rejection message, or
    null). A rejected action costs no step and changes nothing.
``final``
    ``grade``: the full grade report, with checks and gates.

Wrap an environment to log it::

    env = LoggedEnvironment(RecruitmentEnvironment(), "outputs/run/ep.jsonl",
                            agent="rubric_heuristic")
    env.reset("hard", seed=3)
    policies.rubric_heuristic(env)   # every step is logged
    env.close()

``replay(path)`` checks that a log reproduces: same observations, rewards and
grade. ``scripts/replay.py`` is its command line.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from recruitenv.environment import RecruitmentEnvironment
from recruitenv.models import Action, Observation
from recruitenv.profile_factory import GENERATOR_VERSION

_ACTION = TypeAdapter(Action)


def package_version() -> str:
    from recruitenv import __version__
    return __version__


def canonical(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def obs_hash(obs: Observation | dict[str, Any]) -> str:
    data = obs.model_dump(mode="json") if isinstance(obs, Observation) else obs
    return hashlib.sha256(canonical(data).encode()).hexdigest()


def _jsonable(obj: Any) -> Any:
    """Round-trip through JSON so logged and recomputed values compare equal."""
    return json.loads(canonical(obj))


class LoggedEnvironment:
    """Wraps a ``RecruitmentEnvironment`` and writes a transition log.

    Every other attribute is passed through, so reference policies and agents
    use it exactly like the plain environment. One wrapper writes one episode;
    call ``reset`` once. The file is closed when the episode ends, or by
    ``close()``.
    """

    def __init__(
        self,
        env: RecruitmentEnvironment,
        path: str | Path,
        *,
        agent: str,
        model: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._env = env
        self.path = Path(path)
        self.agent = agent
        self.model = model
        self.extra = extra or {}
        self._fh = None
        self.episode_id: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _write(self, record: dict[str, Any]) -> None:
        assert self._fh is not None, "reset() first"
        self._fh.write(canonical(record) + "\n")
        self._fh.flush()

    def reset(self, task_id: str, seed: int = 42, name_seed: int = 0, name_blind: bool = False) -> Observation:
        if self._fh is not None:
            raise RuntimeError("LoggedEnvironment writes one episode; make a new one per episode")
        obs = self._env.reset(task_id, seed, name_seed=name_seed, name_blind=name_blind)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.episode_id = uuid.uuid4().hex
        self._write({
            "kind": "header",
            "episode_id": self.episode_id,
            "task": task_id,
            "seed": seed,
            "name_seed": name_seed,
            "name_blind": name_blind,
            "generator_version": GENERATOR_VERSION,
            "package_version": package_version(),
            "agent": self.agent,
            "model": self.model,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "initial_obs_sha256": obs_hash(obs),
            **({"extra": self.extra} if self.extra else {}),
        })
        return obs

    def step(self, action: Any):
        if not isinstance(action, dict):
            action_dict = action.model_dump(mode="json")
        else:
            action_dict = action
            action = _ACTION.validate_python(action)
        try:
            obs, reward, done, info = self._env.step(action)
        except (ValueError, RuntimeError) as exc:
            state = self._env.state()
            self._write({
                "kind": "step", "step": state.step_number, "action": action_dict,
                "revealed": None, "obs_sha256": None, "reward": None,
                "steps_remaining": state.max_steps - state.step_number,
                "done": state.is_done, "error": str(exc),
            })
            raise
        self._write({
            "kind": "step",
            "step": obs.step_number,
            "action": action_dict,
            "revealed": _jsonable(info.get("data")) if "data" in info else None,
            "obs_sha256": obs_hash(obs),
            "reward": reward.step_reward,
            "steps_remaining": obs.steps_remaining,
            "done": done,
            "error": None,
        })
        if done:
            self._write({"kind": "final", "grade": _jsonable(self._env.grade_report())})
            self.close()
        return obs, reward, done, info

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()


def read_log(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class ReplayError(Exception):
    """The log does not reproduce."""


def replay(path: str | Path, *, check_versions: bool = True) -> dict[str, Any]:
    """Replay a log against a fresh environment and check it reproduces.

    Returns the replayed grade report. Raises ``ReplayError`` on the first
    difference: any observation hash, reward, done flag, a rejection that no
    longer happens, or the final grade. A log from another generator version
    is refused up front (``check_versions=False`` skips that); a log from
    another package version is replayed, and its version is named if it
    diverges.
    """
    records = read_log(path)
    if not records or records[0].get("kind") != "header":
        raise ReplayError("log has no header")
    head = records[0]
    if check_versions and head["generator_version"] != GENERATOR_VERSION:
        raise ReplayError(
            f"generator version {head['generator_version']} in log, {GENERATOR_VERSION} installed"
        )
    note = ""
    if head.get("package_version") != package_version():
        note = f" (log written by recruitenv {head.get('package_version')}, replayed with {package_version()})"
    try:
        return _replay_records(head, records)
    except ReplayError as exc:
        raise ReplayError(f"{exc}{note}") from None


def _replay_records(head: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    env = RecruitmentEnvironment()
    obs = env.reset(head["task"], head["seed"], name_seed=head["name_seed"], name_blind=head["name_blind"])
    if "initial_obs_sha256" in head and obs_hash(obs) != head["initial_obs_sha256"]:
        raise ReplayError("initial observation differs")
    final = None
    for i, rec in enumerate(records[1:], start=1):
        if rec["kind"] == "final":
            final = rec
            break
        if rec["kind"] != "step":
            raise ReplayError(f"record {i}: unknown kind {rec['kind']!r}")
        action = _ACTION.validate_python(rec["action"])
        try:
            obs, reward, done, _ = env.step(action)
        except (ValueError, RuntimeError) as exc:
            if rec["error"] is None:
                raise ReplayError(f"record {i}: action rejected on replay: {exc}") from exc
            continue
        if rec["error"] is not None:
            raise ReplayError(f"record {i}: logged a rejection, replay accepted the action")
        if obs_hash(obs) != rec["obs_sha256"]:
            raise ReplayError(f"record {i} (step {rec['step']}): observation differs")
        if reward.step_reward != rec["reward"] or done != rec["done"]:
            raise ReplayError(f"record {i} (step {rec['step']}): reward or done differs")
    if final is None:
        if env.state().is_done:
            raise ReplayError("episode finished on replay but the log has no final record")
        return {}
    grade = _jsonable(env.grade_report())
    if grade != final["grade"]:
        diff = sorted(k for k in set(grade) | set(final["grade"]) if grade.get(k) != final["grade"].get(k))
        raise ReplayError(f"final grade differs in {diff}")
    return grade
