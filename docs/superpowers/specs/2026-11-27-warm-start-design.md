# Warm start: supervised demonstrations before RL, with a held-out evaluation

Date: 2026-11-27. Roadmap: Phase 5, "show learning". Follows the first reward curve (`2026-11-27-reward-curve-design.md`). Issue: #12 (part).

Decisions marked **(owner)** were the owner's. Decisions marked **(delegate)** came from the owner's Fable delegate, through the advisor tool. Decisions marked **(builder)** were made by the builder from measurements and can be questioned.

## Intent

The first GRPO run ended near the score of labelling the pool blind. The question for this run: if a 0.5B model is first shown episodes that read every candidate's evidence, does it read on held-out seeds, and do its labels beat the blind line?

## Decisions

1. **A second free Kaggle run of about 10 GPU hours (owner).** Approved after the first run's write-up proposed a warm-up or a larger model.
2. **Measure what run 1 could reach before building (delegate).** A full read is 60 tool calls and about 6,000 completion tokens; run 1 allowed 2,560. A rubric heuristic limited to the reveals that fit (about 18) scores about 0.5. That correction went into the run-1 write-up.
3. **Supervised fine-tuning (SFT) and a held-out evaluation are the deliverable; GRPO after SFT is conditional (delegate).** A pilot then measured one GRPO step at 21 minutes, and the second step ran out of memory, so this run has no GRPO stage (builder). That pilot started from a 16-demonstration model that did not end its turns and ran before the attention fix (decision 9), so it says little about GRPO from the final model; it was not re-measured. The pipeline keeps a `--grpo` option, untested beyond that one step.
4. **Demonstrations from `rubric_full_read`, not the adaptive heuristic (delegate).** Its read order is fixed per role, so only the final labels need judgement. Every demonstration scores 0.999.
5. **Batched calls (builder).** TRL runs every tool call in one assistant turn, so a demonstration reveals all five sources of a candidate in one turn and makes the ten decisions in one turn: 11 tool turns instead of 60, and far fewer generation calls.
6. **Two arms, one per T4 (delegate).**
   - **plain:** the demonstrations as above.
   - **ranking:** the decision turn first writes the candidates ranked by the rubric estimate. This tests whether showing the aggregation step helps the labels.
7. **Token parity is a test (delegate).** `tests/test_warm_start.py` replays a demonstration through the evaluation loop and checks that the tokens are exactly those SFT trains on, and that SFT's loss covers exactly the tokens the model generates. The loop reuses TRL's own prompt rendering, response parser and tool-result tokens. It needs the Qwen tokenizer locally, so CI skips it.
8. **Evaluation (delegate).**
   - Seeds 0–99 (evaluation seeds; training and demonstration seeds start at 10,000), greedy, one episode per seed.
   - The instruct model before SFT is the control.
   - The limits are 20 tool turns and 7,000 completion tokens, so a full read fits.
   - Behaviour is reported separately from the score: reveals, decisions, correct labels. A model that reads everything but labels at chance scores below the blind line, and the report must show that this is what happened rather than look like a regression.
9. **Memory fixes found by pilots (builder).**
   - Full fine-tuning with AdamW ran out of memory on a T4 at ~6,600 tokens. Adafactor keeps almost no optimiser state.
   - The remaining out-of-memory came from attention. Qwen2.5's grouped-query attention without a padding mask made transformers ask SDPA for `enable_gqa`, which a T4 can only run with the math kernel and its full score matrix (about 2.3 GB per layer). The script now makes transformers repeat the key/value heads, and allows only the memory-efficient kernel during SFT so any fallback fails at once.
   - Peak memory went from 13.9 GB (then out of memory) to 5.4 GB.
10. **fp16 mixed precision for SFT (builder).** About 4 s per demonstration against 17 s in fp32, with no overflow in the pilot. Evaluation stays fp32.
11. **Size (builder).** 1,000 demonstrations (seeds 10,000–10,999), 2 epochs, learning rate 1e-5 with cosine decay, effective batch 8.

## Result

See `docs/results/training.md`, "Run 2". In short:
- After SFT, every held-out episode made 50 accepted reveal calls (in the six kept transcripts, the 50 distinct rubric sources), decided all 10 candidates and made no failed tool call. The instruct model made 2 reveals and 0.8 decisions per episode.
- Labels were at chance (3.5 and 3.3 of 10 correct), so the scores (0.267 and 0.269) sit on the random-labels line and below shortlisting everyone (0.369).
- The ranking arm wrote well-formed rankings unrelated to the evidence (rank correlation −0.02 to 0.16 in the three kept transcripts).
- Each evaluation hit its 120-minute limit after 97 and 92 of 100 seeds (about 75 s an episode).

## Not done

- GRPO after SFT (see decision 3).
- A larger model.
- Medium and hard.
