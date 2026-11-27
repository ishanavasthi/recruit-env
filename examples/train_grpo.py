# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "recruitenv @ git+https://github.com/ishanavasthi/recruit-env",
#     "trl>=1.9,<2",
#     "transformers>=5.2",
#     "datasets",
#     "accelerate",
# ]
# ///
"""GRPO on RecruitEnv with TRL's environment_factory (tool-calling rollouts).

The model plays whole episodes through the tools in ``recruitenv.tools``
(``check_github``, ``read_resume``, ``shortlist``, ...). The reward is the
environment's own grader score (``get_reward``). Each training prompt is one
(task, seed) episode. Training seeds start at 10,000, so they never overlap
the evaluation seeds (0-199).

Run on a GPU:

    uv run examples/train_grpo.py --task easy --max-steps 200
    hf jobs uv run --flavor a10g-small examples/train_grpo.py --max-steps 200

On Kaggle's free GPUs, ``scripts/kaggle_train.py`` builds and pushes a
private notebook that runs this script (see docs/results/training.md).

Precision: ``--precision auto`` uses bf16 only on GPUs with native bf16
(compute capability 8.0 or newer) and fp32 otherwise. ``torch`` reports bf16
as supported on a T4, which has no native bf16, so the check uses the
compute capability instead.

CPU smoke test of the whole pipeline (no learning; a random tiny model makes
no tool calls, so the reward stays at the floor). Last checked 2026-11-27
with TRL 1.19.0 and torch 2.15 (CPU), in about 30 seconds:

    python examples/train_grpo.py --cpu --episodes 8 \\
        --model trl-internal-testing/tiny-Qwen3MoeForCausalLM

TRL must recognise the model's chat template to parse tool calls:
Qwen2.5-Instruct and TRL's tiny Qwen3 MoE test model work;
``tiny-Qwen3ForCausalLM`` does not.

Outputs, in ``--output-dir``:

- ``reward_curve.csv``: step, mean reward, within-group reward spread, and
  reveals and decisions per episode (what the policy actually does).
- ``train_log.jsonl``: every logged metric dict, one line per step, plus
  peak GPU memory.
- ``run_info.json``: versions, GPU, precision, arguments and wall clock.

A 0.5B model rarely calls tools well at first; expect a flat start, and see
TRL's notes on a short SFT warm-up if every group gets identical rewards.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

TRAIN_SEEDS_FROM = 10_000   # evaluation uses seeds 0-199


def pick_precision(requested: str, cpu: bool) -> str:
    """Resolve ``--precision``: native bf16 needs compute capability >= 8.0."""
    if cpu:
        return "fp32"
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return "bf16"
    return "fp32"


def counting_env(base):
    """``base`` with per-episode counts of reveals and decisions.

    TRL turns every public method into a tool, so the counters are private
    class attributes and the subclass adds no public method.
    """

    class Counting(base):
        _episodes = 0
        _reveals = 0
        _decisions = 0

        def reset(self, *a, **kw):
            type(self)._episodes += 1
            return super().reset(*a, **kw)

        def _act(self, action):
            out = super()._act(action)          # raises on an illegal action, which is not counted
            if getattr(action, "type", "") == "make_decision":
                type(self)._decisions += 1
            else:
                type(self)._reveals += 1
            return out

        @classmethod
        def _take_counts(cls) -> tuple[float, float]:
            n = max(cls._episodes, 1)
            out = (cls._reveals / n, cls._decisions / n)
            cls._episodes = cls._reveals = cls._decisions = 0
            return out

    Counting.__name__ = base.__name__   # TRL names the reward column after the class
    return Counting


def run_info(args: argparse.Namespace, precision: str) -> dict:
    import torch
    import transformers
    import trl

    import recruitenv

    gpus = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [])
    return {
        "args": vars(args),
        "precision": precision,
        "gpus": gpus,
        "python": platform.python_version(),
        "versions": {
            "recruitenv": recruitenv.__version__,
            "trl": trl.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        "train_seeds": [TRAIN_SEEDS_FROM, TRAIN_SEEDS_FROM + args.episodes - 1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    ap.add_argument("--episodes", type=int, default=512, help="distinct training seeds")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4, help="per-device micro-batch of completions")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step; one step's completions = batch size x this")
    ap.add_argument("--max-completion-length", type=int, default=4096,
                    help="tokens for the whole episode: every turn and every tool result")
    ap.add_argument("--max-tool-calls", type=int, default=40,
                    help="tool-calling turns per episode (TRL's max_tool_calling_iterations)")
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop cleanly after this much training wall clock (0 = no limit)")
    ap.add_argument("--stop-flat-steps", type=int, default=0,
                    help="stop after this many consecutive steps with zero reward spread (0 = never)")
    ap.add_argument("--log-completions", action="store_true", help="print one sample rollout per step")
    ap.add_argument("--output-dir", default="outputs/grpo")
    ap.add_argument("--cpu", action="store_true", help="smoke test on CPU (slow; tiny settings)")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from recruitenv.tools import RecruitToolEnv

    EnvCls = counting_env(RecruitToolEnv)

    precision = pick_precision(args.precision, args.cpu)
    prompt = [{"role": "user", "content": "Triage this candidate pool. Use the tools, then decide everyone."}]
    seeds = list(range(TRAIN_SEEDS_FROM, TRAIN_SEEDS_FROM + args.episodes))
    dataset = Dataset.from_dict({
        "prompt": [prompt] * len(seeds),
        "task": [args.task] * len(seeds),
        "seed": seeds,
    })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    curve = out / "reward_curve.csv"
    log_jsonl = out / "train_log.jsonl"
    info = run_info(args, precision)
    started = time.monotonic()

    def write_info(**extra) -> None:
        info.update(extra, minutes=round((time.monotonic() - started) / 60, 2))
        (out / "run_info.json").write_text(json.dumps(info, indent=2))

    write_info(status="starting")

    flat = {"run": 0}

    class CurveLogger(TrainerCallback):
        """Write the curve and every metric as they arrive; stop at the wall-clock limit or when flat."""

        def on_log(self, _args, state, control, logs=None, **kw):
            if not logs or "reward" not in logs:
                return
            import torch

            reveals, decisions = EnvCls._take_counts()
            row = {"step": state.global_step, "minutes": round((time.monotonic() - started) / 60, 2),
                   "reveals_per_episode": round(reveals, 3), "decisions_per_episode": round(decisions, 3),
                   **logs}
            if torch.cuda.is_available():
                row["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            with log_jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            new = not curve.exists()
            with curve.open("a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["step", "reward", "reward_std", "reveals_per_episode", "decisions_per_episode"])
                w.writerow([state.global_step, logs["reward"], logs.get("reward_std", ""),
                            row["reveals_per_episode"], row["decisions_per_episode"]])
            print(f"[curve] step {state.global_step} reward {logs['reward']:.4f} "
                  f"std {logs.get('reward_std', float('nan')):.4f} reveals {reveals:.1f} "
                  f"decisions {decisions:.1f} gpu {row.get('peak_gpu_gb', 0)}GB minutes {row['minutes']}",
                  flush=True)
            flat["run"] = flat["run"] + 1 if logs.get("reward_std", 1) == 0 else 0
            if args.stop_flat_steps and flat["run"] >= args.stop_flat_steps:
                print(f"[curve] {flat['run']} consecutive steps with no reward spread; stopping", flush=True)
                write_info(status="stopping: flat")
                control.should_training_stop = True

        def on_step_end(self, _args, state, control, **kw):
            if args.max_minutes and (time.monotonic() - started) / 60 >= args.max_minutes:
                print(f"[curve] wall-clock limit of {args.max_minutes} min reached at step {state.global_step}",
                      flush=True)
                control.should_training_stop = True

    config = GRPOConfig(
        output_dir=str(out),
        max_steps=args.max_steps if not args.cpu else 2,
        per_device_train_batch_size=args.batch_size if not args.cpu else 2,
        gradient_accumulation_steps=args.grad_accum if not args.cpu else 1,
        num_generations=args.num_generations if not args.cpu else 2,
        max_completion_length=args.max_completion_length if not args.cpu else 512,
        max_tool_calling_iterations=args.max_tool_calls,
        learning_rate=args.learning_rate,
        logging_steps=1,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        gradient_checkpointing=not args.cpu,
        log_completions=args.log_completions,
        num_completions_to_print=1,
        report_to="none",
        save_strategy="no",
    )
    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        environment_factory=EnvCls,
        args=config,
        callbacks=[CurveLogger()],
    )
    try:
        trainer.train()
    except BaseException as exc:
        write_info(status=f"failed: {type(exc).__name__}: {exc}")
        raise
    write_info(status="finished", steps=trainer.state.global_step)
    print(f"reward curve: {curve}")


if __name__ == "__main__":
    main()
