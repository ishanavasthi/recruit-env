# Reward curve on a free GPU: design and record

Date: 2026-11-26/27. Roadmap: Phase 5, "show learning" (`docs/research/2026-11-21-findings-and-roadmap.md`). Issue: #12 (part; the 50+ seed baseline table remains).

Decisions marked **(owner)** were the owner's. Decisions marked **(delegate)** came from the owner's Fable delegate, through the advisor tool. Decisions marked **(builder)** were made by the builder from measurements and can be questioned.

## Intent

Publish a reward curve from a real training run of a model on RecruitEnv, with enough context to read it honestly: what blind policies score, what the model actually did, and what went wrong.

## Decisions

1. **Kaggle, not HF Jobs or Colab (owner).** Kaggle's free T4 notebooks cost nothing, and the Kaggle CLI can push, poll and download a notebook from the development server, so no one has to run anything by hand. The API token stays on the development server and never enters a notebook.
2. **A single-file notebook built from the repo (builder).** `scripts/kaggle_train.py` embeds `examples/train_grpo.py` in the notebook and installs `recruitenv` at a pinned commit (`2bd9e4a`, 4.0.0). Pilot iterations then need no GitHub push and no CI run. The notebook is private.
3. **Probe before training (delegate).** A two-minute notebook checked the GPU, the Python version (3.13), the preinstalled torch (2.11, CUDA 13), internet access, and that installing TRL 1.19 and transformers 5.20 leaves torch alone. It also found that `torch.cuda.is_bf16_supported()` returns True on a T4, which has no native bf16. So `--precision auto` uses the compute capability instead (builder). bf16 was not tried.
4. **Pilot, then gate (delegate).** A 10-step pilot measured about 50 s a step with groups of 4 and showed that rewards varied within groups, so GRPO had a gradient. The full run depended on that check, because a reward stuck at the floor would have burned the quota with nothing to learn from.
5. **Reference lines on the plot (delegate).** `scripts/training_references.py` scores idle, blind constant labels, blind quota fill, random labels, the rubric heuristic and the oracle on the training seeds. The key line is "shortlist everyone" at 0.369. The pilot's rollouts were already heading there.
6. **Groups of 8, two learning rates on two GPUs (delegate, then builder).** The group size went up from 4 to 8 through gradient accumulation, which kept the forward pass at the size the pilot survived. Kaggle's GPU notebooks have two T4s, so a second arm cost no extra quota. The delegate proposed 1e-6 against 1e-5. A 5-step smoke test of the final configuration showed 1e-5 collapsing to identical rollouts by step 4, so the builder replaced it with 3e-6.
7. **Guards for an unattended run (delegate).**
   - `--max-minutes 600` stops cleanly inside Kaggle's 12-hour session.
   - `--stop-flat-steps 30` stops an arm whose groups all score the same.
   - A deliberately failing notebook confirmed that Kaggle keeps output files from a failed run.
8. **Honest claims (delegate).**
   - The curve is training reward on training seeds.
   - Improvement is claimed from a first-versus-last window comparison with a bootstrap interval, noting that steps are not independent.
   - Behaviour is reported as counts of accepted reveal and decision calls.
   - Nothing is explained beyond what the logs show.

## Result

See `docs/results/training.md`. In short:
- The 1e-6 arm ran 168 steps in 600 minutes. Training reward rose from 0.06 to 0.33, close to the 0.369 blind line (every 20-step window below it, 3 single steps above, at most 0.385), and by the end the model made no accepted reveal calls.
- The 3e-6 arm ran out of GPU memory at step 8. Episodes grew longer after the smoke test, and GPU memory grows with the longest episode.

## Not done

- An SFT warm-up, a larger model, saved checkpoints and a held-out evaluation. These are the next steps listed in the write-up.
- The 50+ seed baseline table (#12).
- Phase 6 (results page, report, demo, PyPI).
