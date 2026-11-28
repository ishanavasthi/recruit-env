"""Demonstrations score the maximum; SFT tokens, rollout tokens and TRL's rendering agree.

The token-parity test needs TRL, transformers and the Qwen2.5 tokenizer in the
local Hugging Face cache, so it is skipped in CI. Run it after changing the
prompt, the tools or the demonstrations:

    uv run --with "trl>=1.9,<2" --with "transformers>=5.2" pytest tests/test_warm_start.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


warm = _load("warm_start", ROOT / "examples" / "warm_start.py")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.parametrize("ranking", [False, True])
def test_demonstrations_read_everything_and_score_the_maximum(ranking):
    for seed in (10_000, 10_007):
        messages, score = warm.demonstration("easy", seed, ranking)
        assert score == pytest.approx(0.999)
        assistant = [m for m in messages if m["role"] == "assistant"]
        calls = [c["function"]["name"] for m in assistant for c in m.get("tool_calls", [])]
        assert len(assistant) == 12                       # 10 candidates' reads, the decisions, a closing line
        assert len(calls) == 60 and calls.count("read_resume") == 20
        assert sum(n in ("shortlist", "hold", "reject") for n in calls) == 10
        assert all(m["content"].startswith("Error") is False for m in messages if m["role"] == "tool")
        assert assistant[-2]["content"].startswith("Ranking by the rubric") == ranking
        assert messages[0]["content"].startswith(warm.PROMPT + "You are triaging")


def _tokenizer():
    pytest.importorskip("trl")
    transformers = pytest.importorskip("transformers")
    try:
        transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer not in the local cache")
    return warm.load_tokenizer(MODEL)


@pytest.mark.parametrize("ranking", [False, True])
def test_rollout_of_a_demonstration_gives_the_sft_tokens(ranking):
    from recruitenv.tools import RecruitToolEnv

    tok, template = _tokenizer()
    messages, _ = warm.demonstration("easy", 10_003, ranking)
    ids, labels = warm.training_ids(tok, messages, warm.env_tools(RecruitToolEnv()))
    end = tok.convert_tokens_to_ids("<|im_end|>")
    spans: list[tuple[int, int]] = []

    def forced(prefix: list[int], limit: int) -> list[int]:
        assert ids[: len(prefix)] == prefix, "the rollout diverged from the SFT tokens"
        stop = ids.index(end, len(prefix)) + 1
        spans.append((len(prefix), stop))
        return ids[len(prefix):stop]

    r = warm.rollout(tok, template, forced, "easy", 10_003, max_turns=20, max_completion=8000)
    assert r["score"] == pytest.approx(0.999)
    assert (r["calls"], r["failed_calls"], r["turns"], r["reveals"], r["decided"]) == (60, 0, 11, 50, 10)
    assert r["stop"] == "no tool call"
    seq = r["ids"]
    assert ids[: len(seq)] == seq and not tok.decode(ids[len(seq):]).strip()

    # SFT trains on exactly the tokens the model generates in a rollout (each turn ends with <|im_end|>).
    generated = {i for a, b in spans for i in range(a, b)}
    trained = {i for i, label in enumerate(labels) if label != -100}
    assert generated <= trained
    assert not tok.decode([ids[i] for i in sorted(trained - generated)]).strip()
    assert len(spans) == 12 and all(ids[b - 1] == end for _, b in spans)
