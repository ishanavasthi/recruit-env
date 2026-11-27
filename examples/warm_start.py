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
"""Supervised warm-up on reading demonstrations, held-out evaluation, then GRPO.

The first GRPO run (docs/results/training.md) started from the instruct model
and learned to label the pool without reading. This script first teaches the
format of an episode that reads, by supervised fine-tuning (SFT) on episodes
played by ``rubric_full_read``, a reference policy that reads every rubric
source of every candidate and then labels everyone by the rubric. It then
measures the model on held-out seeds and can continue with GRPO.

A demonstration is one conversation in exactly the form TRL's GRPOTrainer
builds during rollouts: the same prompt, the same tool schemas, the same tool
results. Calls are batched: one assistant turn reveals all five sources of
one candidate, and one turn makes all ten decisions (TRL runs every tool call
in a turn). With ``--ranking`` the decision turn first writes the candidates
ranked by the rubric estimate, so the model is shown the aggregation step.

    # demonstrations only (prints one, checks they score 0.999)
    python examples/warm_start.py demos --episodes 20

    # held-out evaluation of a model, greedy, one episode per seed
    python examples/warm_start.py eval --model Qwen/Qwen2.5-0.5B-Instruct --seeds 0:100 --out outputs/eval_base

    # the whole run: [base eval] -> SFT -> eval -> [GRPO -> eval]
    python examples/warm_start.py pipeline --out outputs/warm --base-eval --deadline-minutes 600

The evaluation loop follows TRL's rollout loop (prompt rendering, tool-call
parsing, tool execution and tool-result tokens use TRL's own functions), so a
model is evaluated on the same token sequences it is trained on.
``tests/test_warm_start.py`` checks that a demonstration replayed through the
evaluation loop gives exactly the tokens that SFT trains on.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grpo import PROMPT, TRAIN_SEEDS_FROM, pick_precision  # noqa: E402

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

FINAL = "All candidates are decided."
SFT_SEEDS_FROM = TRAIN_SEEDS_FROM          # SFT demonstrations: 10,000 onwards
GRPO_SEEDS_FROM = 20_000                   # GRPO after SFT: fresh seeds
PASS = 0.5                                 # the pass line used by scripts/training_references.py


# -- demonstrations ------------------------------------------------------


def env_tools(env: Any) -> list[Callable]:
    """The tools TRL gives the model for ``env``, in TRL's order."""
    return [getattr(env, name) for name, _ in inspect.getmembers(type(env), predicate=inspect.isfunction)
            if name not in ("reset", "get_reward") and not name.startswith("_")]


def _tool_call(action: Any) -> tuple[str, dict[str, str]]:
    if action.type == "make_decision":
        return action.decision, {"candidate_id": action.candidate_id}
    if action.type == "check_platform":
        return f"check_{action.platform}", {"candidate_id": action.candidate_id}
    return "read_resume", {"candidate_id": action.candidate_id, "section": action.section}


def ranking_line(env: Any) -> str:
    """The candidates ranked by the rubric estimate on what has been revealed."""
    from recruitenv import rubric

    st = env.env.state()
    est = {c["id"]: rubric.estimate(st.revealed_data.get(c["id"]), st.job_description).total
           for c in st.candidates_summary}
    order = sorted(est, key=lambda c: (-est[c], c))
    return "Ranking by the rubric: " + ", ".join(f"{c} {est[c]:.2f}" for c in order) + "."


def demonstration(task: str, seed: int, ranking: bool = False) -> tuple[list[dict], float]:
    """``rubric_full_read`` on (task, seed) as a batched tool-calling conversation, and its score."""
    from recruitenv import policies
    from recruitenv.environment import RecruitmentEnvironment
    from recruitenv.tools import RecruitToolEnv

    core = RecruitmentEnvironment()
    core.reset(task, seed)
    actions: list[Any] = []
    step = core.step
    core.step = lambda a: (actions.append(a), step(a))[1]
    policies.rubric_full_read(core)

    turns: list[tuple[str, list[Any]]] = []
    for a in actions:
        key = "decide" if a.type == "make_decision" else a.candidate_id
        if turns and turns[-1][0] == key:
            turns[-1][1].append(a)
        else:
            turns.append((key, [a]))

    env = RecruitToolEnv()
    messages: list[dict] = [{"role": "user", "content": PROMPT + env.reset(task=task, seed=seed)}]
    for key, acts in turns:
        calls = [_tool_call(a) for a in acts]
        content = ranking_line(env) if ranking and key == "decide" else ""
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}}
                                        for n, a in calls]})
        for name, args in calls:
            messages.append({"role": "tool", "name": name, "content": str(getattr(env, name)(**args))})
    messages.append({"role": "assistant", "content": FINAL})
    return messages, env.get_reward()


# -- rendering, as TRL does it ----------------------------------------------


def load_tokenizer(model: str):
    """The tokenizer with TRL's response parser and, if needed, TRL's prefix-preserving template."""
    from transformers import AutoTokenizer
    from trl.chat_template_utils import (add_response_schema, get_training_chat_template,
                                         is_chat_template_prefix_preserving)

    tok = AutoTokenizer.from_pretrained(model)
    if getattr(tok, "response_template", None) is None and getattr(tok, "response_schema", None) is None:
        tok = add_response_schema(tok)
    rollout_template = None if is_chat_template_prefix_preserving(tok) else get_training_chat_template(tok)
    return tok, rollout_template


def prompt_ids(tok, template, messages: list[dict], tools: list) -> list[int]:
    """GRPOTrainer._tokenize_prompts for one environment prompt."""
    out = tok.apply_chat_template(conversation=[messages], tools=tools or None, chat_template=template,
                                  add_generation_prompt=True, tokenize=True, return_dict=True)
    return list(out["input_ids"][0])


def eos_ids(tok, model_path: str | None = None) -> list[int]:
    ids = {tok.eos_token_id}
    end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(end, int) and end != tok.unk_token_id:
        ids.add(end)
    return sorted(i for i in ids if i is not None)


def tool_suffix_ids(tok, template, tool_messages: list[dict]) -> list[int]:
    """TRL's own ``GRPOTrainer._get_tool_suffix_ids``, called on a stand-in for the trainer."""
    from trl import GRPOTrainer

    shim = SimpleNamespace(processing_class=tok, _tokenizer=tok, chat_template=template, chat_template_kwargs={},
                           _is_vlm=False, eos_token_ids=eos_ids(tok))
    return GRPOTrainer._get_tool_suffix_ids(shim, tool_messages)


def training_ids(tok, messages: list[dict], tools: list) -> tuple[list[int], list[int]]:
    """Token ids of a whole conversation and its labels (-100 outside assistant turns).

    Uses TRL's training template (``{% generation %}`` markers), as SFTTrainer
    does with ``assistant_only_loss``. Pre-tokenizing here avoids storing the
    tool calls in an Arrow table, which would add every argument of every tool
    to every call.
    """
    from trl.chat_template_utils import get_training_chat_template

    template = get_training_chat_template(tok, require_prefix_preservation=False)
    out = tok.apply_chat_template(messages, tools=tools, chat_template=template, tokenize=True, return_dict=True,
                                  return_assistant_tokens_mask=True)
    ids = list(out["input_ids"])
    labels = [t if m else -100 for t, m in zip(ids, out["assistant_masks"])]
    return ids, labels


# -- rollouts ----------------------------------------------------------------


def rollout(tok, template, generate: Callable[[list[int], int], list[int]], task: str, seed: int,
            max_turns: int = 20, max_completion: int = 7000) -> dict:
    """One episode the way TRL's tool loop plays it. ``generate(ids, limit)`` returns new token ids."""
    from trl.chat_template_utils import parse_response

    from recruitenv.tools import RecruitToolEnv

    env = RecruitToolEnv()
    tools = env_tools(env)
    by_name = {t.__name__: t for t in tools}
    prompt = prompt_ids(tok, template, [{"role": "user", "content": PROMPT + env.reset(task=task, seed=seed)}],
                        tools)
    completion: list[int] = []
    turns = calls = failed = reveals = decisions = 0
    texts: list[str] = []
    stop = "no tool call"
    while True:
        new = generate(prompt + completion, max_completion - len(completion))
        msg = parse_response(tok, new, prefix=prompt + completion, tools=tools) if new else {}
        completion += new
        texts.append(tok.decode(new))
        tool_calls = msg.get("tool_calls") if msg else None
        if not tool_calls:
            stop = ("length" if not new or len(completion) >= max_completion
                    else "no tool call" if new[-1] in eos_ids(tok) else "turn cut off")
            break
        if turns >= max_turns:
            stop = "turn limit"
            break
        results = []
        for call in tool_calls:
            calls += 1
            if call.get("type") != "function":
                failed += 1
                results.append((call.get("name", "unknown"), {"error": f"Unsupported tool call type: {call.get('type')}"}))
                continue
            name, args = call["function"]["name"], call["function"].get("arguments") or {}
            try:
                if name not in by_name:
                    raise ValueError(f"Tool {name} not found.")
                results.append((name, by_name[name](**args)))
                if name in ("shortlist", "hold", "reject"):
                    decisions += 1
                else:
                    reveals += 1
            except Exception as exc:     # noqa: BLE001 - as TRL: any tool error goes back to the model
                failed += 1
                results.append((name, {"error": str(exc)}))
        turns += 1
        messages = [{"role": "tool", "name": n, "content": r if isinstance(r, list) else str(r)} for n, r in results]
        suffix = tool_suffix_ids(tok, template, messages)
        if len(completion) + len(suffix) > max_completion:
            stop = "length"
            break
        completion += suffix
        texts.append(tok.decode(suffix))

    score = env.get_reward()
    report = env.env.grade_report()
    gt = report["ground_truth"]
    sl_true = {c for c, label in gt.items() if label == "shortlist"}
    sl_agent = {c for c, d in report["decisions"].items() if d == "shortlist"}
    return {
        "seed": seed, "score": score, "correct": report["correct"], "decided": len(report["decisions"]),
        "shortlist_hits": len(sl_true & sl_agent), "reveals": reveals, "decisions": decisions,
        "calls": calls, "failed_calls": failed, "turns": turns, "completion_tokens": len(completion),
        "stop": stop, "ids": prompt + completion, "prompt_len": len(prompt), "text": "".join(texts),
    }


def greedy_generator(model, tok, max_new_tokens: int):
    """Greedy decoding with TRL's stop tokens and no repetition penalty."""
    import torch
    from transformers import GenerationConfig

    cfg = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=eos_ids(tok),
                           pad_token_id=tok.pad_token_id or tok.eos_token_id, repetition_penalty=1.0)

    @torch.no_grad()
    def generate(ids: list[int], limit: int) -> list[int]:
        if limit <= 0:
            return []
        x = torch.tensor([ids], device=model.device)
        cfg.max_new_tokens = min(max_new_tokens, limit)
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=cfg)
        return out[0, len(ids):].tolist()

    return generate


def summarise_eval(rows: list[dict]) -> dict:
    n = len(rows)
    mean = lambda k: round(statistics.mean(r[k] for r in rows), 4)  # noqa: E731
    return {
        "episodes": n, "mean_score": mean("score"),
        "pass_rate": round(sum(r["score"] >= PASS for r in rows) / n, 4),
        "success_rate": round(sum(r["score"] >= 0.75 for r in rows) / n, 4),
        "correct_labels": mean("correct"), "decided": mean("decided"), "shortlist_hits": mean("shortlist_hits"),
        "reveals": mean("reveals"), "decision_calls": mean("decisions"),
        "failed_call_share": round(sum(r["failed_calls"] for r in rows) / max(sum(r["calls"] for r in rows), 1), 4),
        "all_decided": round(sum(r["decided"] == 10 for r in rows) / n, 4),
        "stops": {s: sum(r["stop"] == s for r in rows) for s in sorted({r["stop"] for r in rows})},
        "completion_tokens_median": statistics.median(r["completion_tokens"] for r in rows),
    }


def evaluate(model_path: str, task: str, seeds: range, out: Path, max_turns: int, max_completion: int,
             max_new_tokens: int, cpu: bool = False, keep_text: int = 3, max_minutes: float = 0) -> dict:
    """Greedy episodes on ``seeds`` in order; stops early after ``max_minutes`` (0 = no limit)."""
    import torch
    from transformers import AutoModelForCausalLM

    out.mkdir(parents=True, exist_ok=True)
    expand_kv_heads_for_sdpa()                 # efficient attention for the ~6k-token prefills
    tok, template = load_tokenizer(model_path)
    precision = pick_precision("auto", cpu)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.to("cpu" if cpu or not torch.cuda.is_available() else "cuda").eval()
    generate = greedy_generator(model, tok, max_new_tokens)
    started = time.monotonic()
    rows = []
    with (out / "episodes.jsonl").open("w") as fh:
        for i, seed in enumerate(seeds):
            if max_minutes and (time.monotonic() - started) / 60 >= max_minutes:
                print(f"[eval] {out.name}: time limit of {max_minutes} min reached after {i} episodes", flush=True)
                break
            r = rollout(tok, template, generate, task, seed, max_turns, max_completion)
            r.pop("ids")
            if i >= keep_text:
                r.pop("text")
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"[eval] {out.name} seed {seed} score {r['score']:.3f} correct {r['correct']} "
                  f"reveals {r['reveals']} decided {r['decided']} failed {r['failed_calls']}/{r['calls']} "
                  f"stop {r['stop']} tokens {r['completion_tokens']}", flush=True)
    summary = {"model": model_path, "task": task, "seeds": [seeds.start, seeds.start + len(rows) - 1],
               "minutes": round((time.monotonic() - started) / 60, 2), **summarise_eval(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[eval] summary", json.dumps(summary), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


# -- SFT ---------------------------------------------------------------------


def expand_kv_heads_for_sdpa() -> None:
    """Make transformers repeat the key/value heads before SDPA instead of passing ``enable_gqa``.

    Qwen2.5 has 14 query heads and 2 key/value heads. Without a padding mask,
    transformers asks SDPA for grouped-query attention, which on a T4 (no
    flash attention; the memory-efficient kernel does not support it) falls
    back to the math kernel and its full score matrix.
    """
    from transformers.integrations import sdpa_attention

    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


def attention_without_score_matrix():
    """Only SDPA's memory-efficient kernel, on a GPU.

    A T4 has no flash attention, and when the memory-efficient kernel cannot
    be used PyTorch silently falls back to the math kernel, which builds the
    full score matrix: about 2.3 GB per layer at 6,600 tokens, which ran a
    pilot out of memory. Allowing only the efficient kernel turns that
    fallback into an immediate error.
    """
    import contextlib

    import torch

    expand_kv_heads_for_sdpa()
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])


def sft(model: str, task: str, episodes: int, ranking: bool, out: Path, epochs: float, learning_rate: float,
        grad_accum: int, optim: str = "adafactor", precision: str = "auto", cpu: bool = False) -> dict:
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from recruitenv.tools import RecruitToolEnv

    tok, _ = load_tokenizer(model)
    tools = env_tools(RecruitToolEnv())
    rows, scores = [], []
    for seed in range(SFT_SEEDS_FROM, SFT_SEEDS_FROM + episodes):
        messages, score = demonstration(task, seed, ranking)
        ids, labels = training_ids(tok, messages, tools)
        rows.append({"input_ids": ids, "labels": labels})
        scores.append(score)
    lengths = [len(r["input_ids"]) for r in rows]
    print(f"[sft] {episodes} demonstrations, mean score {statistics.mean(scores):.3f}, "
          f"tokens {min(lengths)}-{max(lengths)}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    log = out / "sft_log.jsonl"
    started = time.monotonic()

    class Log(TrainerCallback):
        def on_log(self, _a, state, _c, logs=None, **kw):
            if not logs:
                return
            row = {"step": state.global_step, "minutes": round((time.monotonic() - started) / 60, 2), **logs}
            if torch.cuda.is_available():
                row["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            with log.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print("[sft]", json.dumps(row), flush=True)

    precision = pick_precision(precision, cpu)
    config = SFTConfig(
        output_dir=str(out / "sft_checkpoints"),
        num_train_epochs=epochs if not cpu else 1,
        max_steps=-1 if not cpu else 2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum if not cpu else 1,
        learning_rate=learning_rate,
        optim=optim,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        max_length=None,
        pad_to_multiple_of=64,             # an aligned mask lets SDPA use its memory-efficient kernel
        logging_steps=1,
        bf16=precision == "bf16",
        fp16=precision == "fp16",          # mixed precision: fp32 weights, fp16 compute (T4 tensor cores)
        gradient_checkpointing=not cpu,
        report_to="none",
        save_strategy="no",
        use_cpu=cpu,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=Dataset.from_list(rows), processing_class=tok,
                         callbacks=[Log()])
    with attention_without_score_matrix():
        trainer.train()
    model_dir = out / "model"
    trainer.save_model(str(model_dir))
    tok.save_pretrained(str(model_dir))
    info = {"episodes": episodes, "demo_seeds": [SFT_SEEDS_FROM, SFT_SEEDS_FROM + episodes - 1],
            "demo_mean_score": round(statistics.mean(scores), 4), "ranking": ranking,
            "tokens": [min(lengths), max(lengths)], "epochs": epochs, "learning_rate": learning_rate, "optim": optim, "precision": precision,
            "steps": trainer.state.global_step, "minutes": round((time.monotonic() - started) / 60, 2)}
    (out / "sft_info.json").write_text(json.dumps(info, indent=2) + "\n")
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return info


# -- the whole run -----------------------------------------------------------


def pipeline(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    left = lambda: args.deadline_minutes - (time.monotonic() - started) / 60  # noqa: E731
    seeds = range(*map(int, args.seeds.split(":")))
    status: dict[str, Any] = {"args": vars(args), "stages": {}}

    def record(stage: str, **info) -> None:
        status["stages"][stage] = {**info, "at_minute": round((time.monotonic() - started) / 60, 1)}
        (out / "pipeline.json").write_text(json.dumps(status, indent=2) + "\n")
        print(f"[pipeline] {stage}: {json.dumps(info)}", flush=True)

    ev = dict(task=args.task, seeds=seeds, max_turns=args.max_turns, max_completion=args.max_completion_length,
              max_new_tokens=args.max_new_tokens, cpu=args.cpu, max_minutes=args.eval_minutes)
    try:
        if args.base_eval:
            record("base_eval", status="done", **evaluate(args.model, out=out / "eval_base", **ev))
        record("sft", status="done", **sft(args.model, args.task, args.sft_episodes, args.ranking, out / "sft",
                                            args.sft_epochs, args.sft_learning_rate, args.sft_grad_accum,
                                            args.optim, args.sft_precision, args.cpu))
        sft_model = str(out / "sft" / "model")
        summary = evaluate(sft_model, out=out / "eval_sft", **ev)
        record("sft_eval", status="done", **summary)
    except Exception as exc:
        record("failed", status=f"{type(exc).__name__}: {exc}")
        raise

    reserve = summary["minutes"] * 1.3 + 5
    grpo_minutes = left() - reserve
    if not args.grpo or grpo_minutes < args.min_grpo_minutes:
        record("grpo", status="skipped", minutes_available=round(grpo_minutes, 1))
        return
    argv = [sys.executable, str(Path(__file__).resolve().parent / "train_grpo.py"), "--model", sft_model,
            "--task", args.task, "--output-dir", str(out / "grpo"), "--seed-start", str(GRPO_SEEDS_FROM),
            "--max-tool-calls", str(args.max_turns), "--max-completion-length", str(args.max_completion_length),
            "--max-minutes", str(round(grpo_minutes, 1)), "--optim", args.optim, "--save-model",
            *args.grpo_args.split()]
    print("[pipeline] running:", " ".join(argv), flush=True)
    code = subprocess.run(argv).returncode
    record("grpo", status="done" if code == 0 else f"exit {code}", minutes_available=round(grpo_minutes, 1))
    if code == 0 and (out / "grpo" / "model").exists():
        record("grpo_eval", status="done", **evaluate(str(out / "grpo" / "model"), out=out / "eval_grpo", **ev))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demos", help="build demonstrations and check their scores")
    d.add_argument("--task", default="easy")
    d.add_argument("--episodes", type=int, default=20)
    d.add_argument("--ranking", action="store_true")

    def eval_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
        p.add_argument("--seeds", default="0:100", help="held-out seeds start:stop (evaluation seeds are 0-199)")
        p.add_argument("--max-turns", type=int, default=20, help="tool-calling turns per episode")
        p.add_argument("--max-completion-length", type=int, default=7000,
                       help="tokens for the whole episode after the prompt, tool results included")
        p.add_argument("--max-new-tokens", type=int, default=512,
                       help="tokens per assistant turn (a demonstration turn is under 400)")
        p.add_argument("--eval-minutes", type=float, default=45, help="time limit per evaluation (0 = none)")
        p.add_argument("--cpu", action="store_true")

    e = sub.add_parser("eval", help="evaluate a model on held-out seeds")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    eval_args(e)

    p = sub.add_parser("pipeline", help="[base eval] -> SFT -> eval -> [GRPO -> eval]")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--base-eval", action="store_true", help="evaluate the starting model first")
    p.add_argument("--ranking", action="store_true", help="demonstrations write a rubric ranking before deciding")
    p.add_argument("--sft-episodes", type=int, default=400)
    p.add_argument("--sft-epochs", type=float, default=2)
    p.add_argument("--sft-learning-rate", type=float, default=1e-5)
    p.add_argument("--sft-grad-accum", type=int, default=8)
    p.add_argument("--sft-precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                   help="fp16 is mixed precision with fp32 weights; auto is fp32 on a T4")
    p.add_argument("--optim", default="adafactor",
                   help="optimizer for SFT and GRPO; full fine-tuning with AdamW does not fit a T4 at 6-7k tokens")
    p.add_argument("--grpo", action="store_true", help="continue with GRPO from the SFT model while time is left")
    p.add_argument("--grpo-args", default="", help="extra arguments for train_grpo.py")
    p.add_argument("--min-grpo-minutes", type=float, default=60)
    p.add_argument("--deadline-minutes", type=float, default=600, help="wall clock for the whole pipeline")
    eval_args(p)

    args = ap.parse_args()
    if args.cmd == "demos":
        scores = []
        for i, seed in enumerate(range(SFT_SEEDS_FROM, SFT_SEEDS_FROM + args.episodes)):
            messages, score = demonstration(args.task, seed, args.ranking)
            scores.append(score)
            if i == 0:
                print(json.dumps(messages[:3] + messages[-4:], indent=1))
        print(f"{len(scores)} demonstrations, mean score {statistics.mean(scores):.4f}, min {min(scores):.4f}")
    elif args.cmd == "eval":
        evaluate(args.model, args.task, range(*map(int, args.seeds.split(":"))), Path(args.out), args.max_turns,
                 args.max_completion_length, args.max_new_tokens, args.cpu, max_minutes=args.eval_minutes)
    else:
        pipeline(args)


if __name__ == "__main__":
    main()
