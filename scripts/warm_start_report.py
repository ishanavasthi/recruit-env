# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Summarise a warm-start run: held-out scores by stage, behaviour, GRPO curve.

Reads the per-arm folders written by ``examples/warm_start.py pipeline``
(``pipeline.json``, ``eval_base/``, ``eval_sft/``, ``eval_grpo/``,
``sft/sft_log.jsonl``, ``grpo/reward_curve.csv``) and reference scores on the
evaluation seeds from ``scripts/training_references.py``. Writes a PNG and a
JSON summary.

    uv run scripts/warm_start_report.py --run outputs/kaggle-warm/grpo \\
        --references docs/results/training/references_eval.json --out docs/results/training

Every evaluation plays the same held-out seeds, one greedy episode each, so
stage-to-stage differences are paired by seed. Intervals are 95% percentile
bootstraps over seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STAGES = (("eval_base", "instruct model"), ("eval_sft", "after SFT"), ("eval_grpo", "after SFT + GRPO"))
COLOURS = ("#2a78d6", "#eb6834")     # categorical slots 1 and 2
REFERENCE_LINES = ("shortlist everyone", "random labels", "rubric heuristic")


def load_eval(folder: Path) -> dict | None:
    if not (folder / "summary.json").exists():
        return None
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").open()]
    return {"summary": json.loads((folder / "summary.json").read_text()),
            "seeds": np.array([r["seed"] for r in rows]),
            "score": np.array([r["score"] for r in rows]),
            "reveals": np.array([r["reveals"] for r in rows], dtype=float),
            "correct": np.array([r["correct"] for r in rows], dtype=float)}


def bootstrap(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(a: dict, b: dict) -> dict | None:
    """``b`` minus ``a`` on the seeds both played."""
    common = np.intersect1d(a["seeds"], b["seeds"])
    if not len(common):
        return None
    ia = {s: i for i, s in enumerate(a["seeds"])}
    ib = {s: i for i, s in enumerate(b["seeds"])}
    d = np.array([b["score"][ib[s]] - a["score"][ia[s]] for s in common])
    m, lo, hi = bootstrap(d)
    return {"seeds": int(len(common)), "diff": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": lo > 0 or hi < 0}


def load_arm(folder: Path) -> dict:
    arm = {"name": folder.name, "pipeline": json.loads((folder / "pipeline.json").read_text()),
           "evals": {key: load_eval(folder / key) for key, _ in STAGES}}
    curve = folder / "grpo" / "reward_curve.csv"
    if curve.exists():
        rows = list(csv.DictReader(curve.open()))
        arm["grpo_step"] = np.array([int(r["step"]) for r in rows])
        arm["grpo_reward"] = np.array([float(r["reward"]) for r in rows])
    sft_log = folder / "sft" / "sft_log.jsonl"
    if sft_log.exists():
        logs = [json.loads(line) for line in sft_log.open()]
        losses = [r["loss"] for r in logs if "loss" in r]
        arm["sft_loss"] = [losses[0], losses[-1]] if losses else None
        arm["sft_peak_gpu_gb"] = max((r.get("peak_gpu_gb", 0) for r in logs), default=0)
    return arm


def summarise(arm: dict, base: dict | None, blind: float | None) -> dict:
    out: dict = {"arm": arm["name"], "ranking": arm["pipeline"]["args"].get("ranking"),
                 "stages": {k: v["status"] if isinstance(v, dict) and "status" in v else v
                            for k, v in arm["pipeline"]["stages"].items()},
                 "sft_loss_first_last": arm.get("sft_loss"), "sft_peak_gpu_gb": arm.get("sft_peak_gpu_gb"),
                 "evals": {}}
    for key, label in STAGES:
        ev = arm["evals"][key] or (base if key == "eval_base" else None)
        if ev is None:
            continue
        m, lo, hi = bootstrap(ev["score"])
        row = {"label": label, **ev["summary"], "score_ci95": [round(lo, 4), round(hi, 4)]}
        if blind is not None:
            d = ev["score"] - blind
            dm, dlo, dhi = bootstrap(d)
            row["minus_blind"] = {"diff": round(dm, 4), "ci95": [round(dlo, 4), round(dhi, 4)]}
        out["evals"][key] = row
    evs = {k: arm["evals"][k] or (base if k == "eval_base" else None) for k, _ in STAGES}
    if evs["eval_base"] and evs["eval_sft"]:
        out["sft_minus_base"] = paired(evs["eval_base"], evs["eval_sft"])
    if evs["eval_sft"] and evs["eval_grpo"]:
        out["grpo_minus_sft"] = paired(evs["eval_sft"], evs["eval_grpo"])
    if "grpo_step" in arm:
        out["grpo_steps"] = int(arm["grpo_step"][-1])
        w = min(10, len(arm["grpo_reward"]) // 2) or 1
        out["grpo_reward_first_last"] = [round(float(arm["grpo_reward"][:w].mean()), 4),
                                         round(float(arm["grpo_reward"][-w:].mean()), 4)]
    return out


def plot(arms: list[dict], base: dict | None, refs: dict, path: Path, task: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [(k, label) for k, label in STAGES
              if (k == "eval_base" and base) or any(a["evals"][k] for a in arms if k != "eval_base")]
    has_grpo = any("grpo_step" in a for a in arms)
    panels = 4 if has_grpo else 3
    fig, axes = plt.subplots(1, panels, figsize=(4.3 * panels, 4.6))
    xs = np.arange(len(stages))
    width = 0.2
    metrics = (("score", "Held-out score", "mean episode score, 95% CI"),
               ("reveals", "Evidence read", "accepted reveal calls per episode"),
               ("correct", "Labels correct", "correct labels per episode (of 10), 95% CI"))
    seen = set()
    for (metric, title, ylabel), ax in zip(metrics, axes):
        for k, (key, _) in enumerate(stages):
            if key == "eval_base":
                points = [(0.0, base, "#6b6b66", "instruct model (shared start)")]
            else:
                points = [((j - (len(arms) - 1) / 2) * width, a["evals"][key], c,
                           "SFT with ranking line" if a["pipeline"]["args"].get("ranking") else "plain SFT")
                          for j, (a, c) in enumerate(zip(arms, COLOURS))]
            for off, ev, c, label in points:
                if ev is None:
                    continue
                m, lo, hi = bootstrap(ev[metric])
                show = metric == "score" and label not in seen
                seen.add(label) if show else None
                ax.errorbar(xs[k] + off, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=8, color=c, capsize=3, lw=1.5,
                            mec="white", mew=1.5, label=label if show else None)
        ax.set_xticks(xs, [s for _, s in stages], fontsize=9)
        ax.set_xlim(-0.6, len(stages) - 0.4)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e3", lw=0.8)
        ax.set_axisbelow(True)

    def ref(ax, y, text):
        ax.axhline(y, color="#8a8a85", ls=":", lw=1, zorder=0)
        ax.text(ax.get_xlim()[0], y, f" {text}", va="bottom", ha="left", fontsize=8, color="#5f5f5a")

    ax, bx, lx = axes[0], axes[1], axes[2]
    ax.set_ylim(-0.02, 1.08)
    for name in REFERENCE_LINES:
        if name in refs:
            ref(ax, refs[name]["mean"], f"{name} {refs[name]['mean']:.2f}")
    ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.62), frameon=False)
    bx.set_ylim(-1, 56)
    ref(bx, 50, "full read: 50")
    lx.set_ylim(-0.2, 10.5)
    ref(lx, 10, "all correct")
    ref(lx, 10 / 3, "uniform random labels: 3.3")
    if has_grpo:
        cx = axes[3]
        for arm, c in zip(arms, COLOURS):
            if "grpo_step" not in arm:
                continue
            r = arm["grpo_reward"]
            c_ = np.cumsum(np.insert(r, 0, 0.0))
            idx = np.arange(1, len(r) + 1)
            smooth = (c_[idx] - c_[np.maximum(idx - 10, 0)]) / (idx - np.maximum(idx - 10, 0))   # trailing 10-step mean
            cx.plot(arm["grpo_step"], r, color=c, alpha=0.25, lw=0.8)
            cx.plot(arm["grpo_step"], smooth, color=c, lw=2)
        cx.set_title("GRPO training reward after SFT", fontsize=10)
        cx.set_xlabel("step (one fresh training seed, 8 rollouts)", fontsize=9)
        cx.set_ylabel("episode score (trailing 10-step mean)", fontsize=9)
        cx.set_ylim(-0.02, 1.05)
        cx.spines[["top", "right"]].set_visible(False)
        cx.grid(axis="y", color="#e6e6e3", lw=0.8)
    n = sorted({len(ev["score"]) for a in arms for ev in a["evals"].values() if ev is not None})
    fig.suptitle(f"Warm start on RecruitEnv {task}: Qwen2.5-0.5B-Instruct, SFT on full-read demonstrations. "
                 f"Held-out seeds from 0, greedy, {n[0]}-{n[-1]} episodes per point", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="folder holding one sub-folder per arm")
    ap.add_argument("--references", type=Path, required=True, help="references on the evaluation seeds")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms = [load_arm(p) for p in sorted(args.run.iterdir()) if (p / "pipeline.json").exists()]
    base = next((a["evals"]["eval_base"] for a in arms if a["evals"]["eval_base"]), None)
    refs = json.loads(args.references.read_text())
    blind = refs["references"].get("shortlist everyone", {}).get("mean")
    args.out.mkdir(parents=True, exist_ok=True)
    plot(arms, base, refs["references"], args.out / "warm_start.png", refs["task"])
    summary = {"task": refs["task"], "reference_seeds": refs.get("seeds"), "references": refs["references"],
               "arms": [summarise(a, base, blind) for a in arms]}
    if len(arms) == 2:
        for key, label in STAGES[1:]:
            a, b = arms[0]["evals"][key], arms[1]["evals"][key]
            if a and b:
                summary.setdefault("between_arms", {})[key] = {"difference": f"{arms[1]['name']} minus {arms[0]['name']}",
                                                               **paired(a, b)}
    (args.out / "warm_start_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for s in summary["arms"]:
        print(f"{s['arm']} (ranking {s['ranking']}): stages {s['stages']}")
        for key, ev in s["evals"].items():
            print(f"  {ev['label']:>18}: score {ev['mean_score']:.3f} {ev['score_ci95']} pass {ev['pass_rate']:.2f} "
                  f"correct {ev['correct_labels']:.1f}/10 reveals {ev['reveals']:.1f} decided {ev['decided']:.1f} "
                  f"failed calls {ev['failed_call_share']:.2f} stops {ev['stops']}")
        for k in ("sft_minus_base", "grpo_minus_sft"):
            if s.get(k):
                print(f"  {k}: {s[k]}")
    for key, d in summary.get("between_arms", {}).items():
        print(f"{key}: {d}")


if __name__ == "__main__":
    main()
