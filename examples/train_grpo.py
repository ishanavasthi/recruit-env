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
(task, seed) episode; evaluation seeds (0-199) are kept out of training.

Run on a GPU:

    uv run examples/train_grpo.py --task easy --max-steps 200
    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN examples/train_grpo.py --max-steps 200

or in Colab: ``!pip install uv && uv run examples/train_grpo.py``.

CPU smoke test of the whole pipeline (no learning; a random tiny model makes
no tool calls, so the reward stays at the floor). Checked on 2026-11-25 with
TRL 1.18.2, transformers 5.24.1 and torch 2.15 (CPU), about 30 seconds:

    uv run examples/train_grpo.py --cpu --episodes 8 \
        --model trl-internal-testing/tiny-Qwen3MoeForCausalLM

TRL must recognise the model's chat template to parse tool calls:
Qwen2.5-Instruct and TRL's tiny Qwen3 MoE test model work;
``tiny-Qwen3ForCausalLM`` does not.

It writes ``reward_curve.csv`` (step, mean reward) to the output directory.
A 0.5B model rarely calls tools well at first; expect a flat start, and see
TRL's notes on a short SFT warm-up if every group gets identical rewards.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

TRAIN_SEEDS_FROM = 10_000   # evaluation uses seeds 0-199


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    ap.add_argument("--episodes", type=int, default=512, help="distinct training seeds")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=4096)
    ap.add_argument("--output-dir", default="outputs/grpo")
    ap.add_argument("--cpu", action="store_true", help="smoke test on CPU (slow; tiny settings)")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from recruitenv.tools import RecruitToolEnv

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

    class CurveLogger(TrainerCallback):
        def on_log(self, _args, state, control, logs=None, **kw):
            if logs and "reward" in logs:
                new = not curve.exists()
                with curve.open("a", newline="") as fh:
                    w = csv.writer(fh)
                    if new:
                        w.writerow(["step", "reward", "reward_std"])
                    w.writerow([state.global_step, logs["reward"], logs.get("reward_std", "")])

    config = GRPOConfig(
        output_dir=str(out),
        max_steps=args.max_steps if not args.cpu else 2,
        per_device_train_batch_size=args.batch_size if not args.cpu else 2,
        num_generations=args.num_generations if not args.cpu else 2,
        max_completion_length=args.max_completion_length if not args.cpu else 512,
        learning_rate=1e-6,
        logging_steps=1,
        bf16=not args.cpu,
        report_to="none",
        save_strategy="no",
    )
    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        environment_factory=RecruitToolEnv,
        args=config,
        callbacks=[CurveLogger()],
    )
    trainer.train()
    print(f"reward curve: {curve}")


if __name__ == "__main__":
    main()
