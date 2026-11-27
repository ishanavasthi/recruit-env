# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Summarise a GRPO run: reward curve plot, early-versus-late comparison, behaviour.

Reads the per-arm output folders written by ``examples/train_grpo.py``
(``reward_curve.csv``, ``train_log.jsonl``, ``run_info.json``) and the
reference scores from ``scripts/training_references.py``. Writes a PNG and
a JSON summary.

    uv run scripts/training_report.py --run outputs/kaggle-run/grpo \\
        --references docs/results/training/references.json --out docs/results/training

The early-versus-late comparison treats each training step as one unit:
every step is a fresh training seed with its own group of rollouts. Steps
are not fully independent (the policy changes slowly), so the 95% bootstrap
interval over steps is somewhat optimistic.

"Reveal calls" and "decision calls" count tool calls that the environment
accepted, per episode. A reveal of something already revealed is accepted
(it returns "already revealed") and is counted; rejected calls are not.

"Blind-score steps" are steps where all rollouts scored exactly the score of
shortlisting everyone without reading. On easy that score is the same on
every seed (0.3692 on seeds 10000-10999), so such a step is a strong sign
that every rollout labelled the pool blind.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

WINDOW = 20
REFERENCE_LINES = ("idle", "shortlist everyone", "random labels", "rubric heuristic")


def load_arm(folder: Path) -> dict:
    rows = list(csv.DictReader((folder / "reward_curve.csv").open()))
    log = [json.loads(line) for line in (folder / "train_log.jsonl").open()]
    return {
        "name": folder.name,
        "info": json.loads((folder / "run_info.json").read_text()),
        "step": np.array([int(r["step"]) for r in rows]),
        "reward": np.array([float(r["reward"]) for r in rows]),
        "reward_std": np.array([float(r["reward_std"] or 0) for r in rows]),
        "reveal_calls": np.array([float(r.get("reveals_per_episode") or 0) for r in rows]),
        "decision_calls": np.array([float(r.get("decisions_per_episode") or 0) for r in rows]),
        "peak_gpu_gb": max((r.get("peak_gpu_gb", 0) for r in log), default=0),
    }


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of ``b`` minus mean of ``a`` with a 95% percentile bootstrap interval."""
    rng = np.random.default_rng(seed)
    diffs = rng.choice(b, (n, len(b))).mean(1) - rng.choice(a, (n, len(a))).mean(1)
    return float(b.mean() - a.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def window_stats(arm: dict, sl: slice, blind: float | None) -> dict:
    out = {"steps": [int(arm["step"][sl][0]), int(arm["step"][sl][-1])],
           "reward": round(float(arm["reward"][sl].mean()), 4),
           "reward_spread": round(float(arm["reward_std"][sl].mean()), 4),
           "reveal_calls": round(float(arm["reveal_calls"][sl].mean()), 2),
           "decision_calls": round(float(arm["decision_calls"][sl].mean()), 2),
           "zero_spread_steps": int((arm["reward_std"][sl] == 0).sum())}
    if blind is not None:
        out["blind_score_steps"] = int(((np.abs(arm["reward"][sl] - blind) < 1e-3)
                                        & (arm["reward_std"][sl] == 0)).sum())
    return out


def summarise(arm: dict, window: int, blind: float | None) -> dict:
    n = len(arm["reward"])
    w = min(window, n // 2)
    first, last = slice(0, w), slice(n - w, None)
    diff, lo, hi = bootstrap_diff(arm["reward"][first], arm["reward"][last])
    return {
        "arm": arm["name"],
        "learning_rate": arm["info"]["args"].get("learning_rate"),
        "steps": int(arm["step"][-1]) if len(arm["step"]) else 0,
        "minutes": arm["info"].get("minutes"),
        "status": arm["info"].get("status"),
        "window_steps": w,
        "windows": [window_stats(arm, slice(i, min(i + window, n)), blind) for i in range(0, n, window)] if w else [],
        "first_window": window_stats(arm, first, blind) if w else None,
        "last_window": window_stats(arm, last, blind) if w else None,
        "late_minus_early": {"diff": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)],
                             "excludes_zero": lo > 0 or hi < 0},
        "peak_gpu_gb": arm["peak_gpu_gb"],
    }


def rolling(x: np.ndarray, k: int) -> np.ndarray:
    """Trailing mean over up to ``k`` steps; same length as ``x``."""
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - k, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def plot(arms: list[dict], refs: dict, path: Path, task: str, smooth: int = 20) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    colours = ["#1f6feb", "#d1242f"]
    for arm, c in zip(arms, colours):
        lr = arm["info"]["args"].get("learning_rate")
        status = arm["info"].get("status", "")
        tag = f"lr {lr:g}" + (f", crashed at step {int(arm['step'][-1])} (out of GPU memory)"
                              if "OutOfMemory" in status else
                              f", crashed at step {int(arm['step'][-1])}" if status.startswith("failed") else "")
        ax.plot(arm["step"], arm["reward"], color=c, alpha=0.18, lw=0.8)
        ax.plot(arm["step"], rolling(arm["reward"], smooth), color=c, lw=2,
                label=f"{tag} (trailing {smooth}-step mean)")
        bx.plot(arm["step"], rolling(arm["decision_calls"], 10), color=c, lw=1.5, label=f"decision calls, lr {lr:g}")
        bx.plot(arm["step"], rolling(arm["reveal_calls"], 10), color=c, lw=1.5, ls="--",
                label=f"reveal calls, lr {lr:g}")
    for name in REFERENCE_LINES:
        if name in refs:
            ax.axhline(refs[name]["mean"], color="#57606a", ls=":", lw=1)
            ax.text(ax.get_xlim()[0], refs[name]["mean"], f" {name} {refs[name]['mean']:.2f}", va="bottom",
                    fontsize=8, color="#57606a")
    ax.set_ylabel("episode score (training seeds)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"GRPO on RecruitEnv {task}: Qwen2.5-0.5B-Instruct, one Kaggle T4 per arm")
    ax.legend(loc="center right", fontsize=8)
    bx.set_ylabel("accepted calls per episode\n(trailing 10-step mean)")
    bx.set_xlabel("training step (one fresh seed, 8 rollouts)")
    bx.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="folder holding one sub-folder per arm")
    ap.add_argument("--references", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=WINDOW)
    args = ap.parse_args()

    arms = [load_arm(p) for p in sorted(args.run.iterdir()) if (p / "reward_curve.csv").exists()]
    refs = json.loads(args.references.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    plot(arms, refs["references"], args.out / "reward_curve.png", refs["task"])
    blind = refs["references"].get("shortlist everyone", {}).get("mean")
    summary = {"task": refs["task"], "references": refs["references"],
               "arms": [summarise(a, args.window, blind) for a in arms],
               "run_info": {a["name"]: {k: a["info"][k] for k in ("gpus", "precision", "versions", "python",
                                                                   "train_seeds", "args")}
                            for a in arms}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for s in summary["arms"]:
        d = s["late_minus_early"]
        print(f"{s['arm']}: {s['steps']} steps, {s['minutes']} min, {s['status'][:60]}; "
              f"first {s['window_steps']} {s['first_window']['reward']:.3f} -> last {s['last_window']['reward']:.3f} "
              f"(diff {d['diff']:+.3f}, 95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f})")
        for w in s["windows"]:
            print(f"  steps {w['steps'][0]:>3}-{w['steps'][1]:>3}: reward {w['reward']:.3f} spread {w['reward_spread']:.3f} "
                  f"reveal calls {w['reveal_calls']:5.2f} decision calls {w['decision_calls']:5.2f} "
                  f"blind-score steps {w.get('blind_score_steps', '-')}")


if __name__ == "__main__":
    main()
