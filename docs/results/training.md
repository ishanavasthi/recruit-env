# Training: a first GRPO reward curve

One GRPO run of Qwen2.5-0.5B-Instruct on the **easy** task, on a free Kaggle T4, 2026-11-26/27. All numbers are **training reward on training seeds** (10,000 onwards; every step draws a fresh seed, so each step's episode is unseen when it is played). This is not a held-out evaluation.

![Reward curve](training/reward_curve.png)

## Result

**The model learned to label the pool. It did not learn to triage.** Its score rose from 0.06 to about 0.33, close to 0.369, the score of shortlisting every candidate without reading anything. Every 20-step window averages below that line; 3 single steps of 168 went above it (at most 0.385), with 0–1 accepted reveal calls per episode. By the end the model made no accepted reveal calls.

| Steps | Mean score | Within-group spread | Accepted reveal calls per episode | Accepted decision calls per episode | Steps where all 8 rollouts scored exactly 0.3692 |
|---|---|---|---|---|---|
| 1–20 | 0.056 | 0.081 | 3.64 | 1.46 | 0 |
| 21–40 | 0.266 | 0.125 | 0.30 | 7.22 | 0 |
| 41–60 | 0.304 | 0.108 | 0.02 | 8.21 | 2 |
| 61–80 | 0.310 | 0.099 | 0.00 | 8.49 | 5 |
| 81–100 | 0.316 | 0.113 | 0.04 | 8.66 | 1 |
| 101–120 | 0.265 | 0.133 | 2.57 | 8.46 | 0 |
| 121–140 | 0.264 | 0.127 | 13.72 | 7.80 | 0 |
| 141–160 | 0.316 | 0.100 | 0.06 | 8.61 | 3 |
| 161–168 | 0.348 | 0.049 | 0.00 | 9.45 | 3 |

- **Early versus late.** The last 20 steps average 0.271 more than the first 20 (95% bootstrap interval over steps +0.242 to +0.298). The interval treats steps as independent, which they are not quite, so it is somewhat optimistic. The rise itself is not in doubt.
- **Blind labelling.** Shortlisting everyone scores exactly 0.3692 on every one of the 1,000 training seeds. In 14 of the 168 steps, all 8 rollouts scored exactly that, including 5 of the last 20.
- **A reveal burst.** Around steps 110–145, accepted reveal calls rose to about 20 per episode, the score dipped to about 0.25, and then they went back to zero. No rollouts were saved in this run, so the logs cannot say what the model did with what it read.
- **Many failed calls.** From step 40 on, the model made a median of 24 tool calls per episode (10–60), and a median of 59% of them failed (22–79%). TRL counts as failed both calls the environment refused and malformed calls that never reached it; the logs record the rate, not the reasons. Completions grew to a median of about 900 tokens, and a step took a median of 180 s, against 74 s for the first step.

### What this says about the environment

- The reward gives GRPO a usable signal from a weak start. Within-group spread was above zero on 150 of 168 steps.
- Blind labelling does poorly on easy. The blind policies measured here average at most 0.369 and pass (score ≥ 0.5) on 0–3% of seeds, while the rubric heuristic, which reads, scores 0.999 and passes every seed. A model that stops near the blind line has learned the format, not the job, and the score shows it.
- In this one run, a pretrained 0.5B instruct model fine-tuned with GRPO, without a supervised warm-up, settled just below the blind line (0.30–0.35) within about 40 steps and did not get past it in 168.

## Reference scores on the training seeds

From `scripts/training_references.py --task easy --seeds 10000:11000` ([references.json](training/references.json)):

| Policy | Mean score | Pass rate (≥ 0.5) |
|---|---|---|
| Idle (no decisions) | 0.001 | 0.00 |
| Shortlist everyone | 0.369 | 0.00 |
| Hold everyone | 0.246 | 0.00 |
| Reject everyone | 0.185 | 0.00 |
| Blind quota fill (id order) | 0.267 | 0.03 |
| Random labels | 0.269 | 0.02 |
| Rubric heuristic | 0.999 | 1.00 |
| Oracle | 0.999 | 1.00 |

## Setup

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, full fine-tune, fp32, gradient checkpointing |
| Precision note | The T4 has no native bf16, though `torch.cuda.is_bf16_supported()` returns True on it. bf16 was not tried; `--precision auto` uses fp32 below compute capability 8.0. |
| Algorithm | TRL 1.19.0 `GRPOTrainer` with `environment_factory=RecruitToolEnv` (tool-calling rollouts) |
| Per step | one fresh seed, 8 rollouts (`num_generations 8`, micro-batch 4, gradient accumulation 2) |
| Limits | 2,560 completion tokens per episode (all turns and tool results), 40 tool-calling turns |
| Learning rate | 1e-6 (the 3e-6 arm crashed, see below) |
| Hardware | Kaggle notebook, 2 × Tesla T4 (one arm per GPU), private |
| Software | Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2, TRL 1.19.0, recruitenv 4.0.0 |
| Wall clock | 600 minutes of training (the `--max-minutes` limit), 168 steps |
| Peak GPU memory | 11.5 GB |

The script that ran is byte-identical to `examples/train_grpo.py` at commit `47fc093`. The launch command was:

```
python scripts/kaggle_train.py push --slug recruitenv-grpo-easy \
  --arm "lr1e-6=--learning-rate 1e-6" --arm "lr3e-6=--learning-rate 3e-6" -- \
  --task easy --max-steps 1000 --episodes 1000 --num-generations 8 --batch-size 4 \
  --grad-accum 2 --max-completion-length 2560 --max-tool-calls 40 \
  --stop-flat-steps 30 --max-minutes 600
```

Raw outputs for both arms (curve CSV, every logged metric, run info, console log) are in [`training/run-2026-11-27/`](training/run-2026-11-27/). The plot and [summary.json](training/summary.json) are regenerated with:

```
uv run scripts/training_report.py --run docs/results/training/run-2026-11-27 \
  --references docs/results/training/references.json --out docs/results/training
```

## What did not work

- **The 3e-6 arm ran out of GPU memory at step 8**, 46 minutes in. GPU memory grows with the longest episode in a step, and episodes grew later, after the 5-step smoke test had passed at 10.2 GB. It says nothing about the learning rate.
- **1e-5 collapsed in a 5-step smoke test**: by step 4 all 8 rollouts made the same single decision and scored the same, so there was no gradient. That is one short run, not a sweep.

## Next steps (not done here)

- An SFT warm-up on rubric-heuristic episodes, so the model starts from a policy that reads before deciding.
- A larger model (1.5B–3B with LoRA), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or a lower completion cap to avoid the out-of-memory crash.
- Find out why most late tool calls were rejected (save rollouts with `--log-completions`), and if they are repeats, end the rollout after a run of rejected calls.
- A held-out evaluation of the trained checkpoint on seeds 0–199 (no checkpoint was saved in this run).
