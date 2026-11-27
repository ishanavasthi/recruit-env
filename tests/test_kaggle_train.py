"""The Kaggle runner builds a private, self-contained notebook (no TRL or network needed)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kaggle_train = _load("kaggle_train", ROOT / "scripts" / "kaggle_train.py")
train_grpo = _load("train_grpo", ROOT / "examples" / "train_grpo.py")


def test_build_embeds_script_and_is_private(tmp_path):
    folder = kaggle_train.build("recruitenv-test", "someone", "abc1234",
                                ["--max-steps", "3", "--task", "easy"], tmp_path / "k")
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/recruitenv-test"
    assert meta["is_private"] == "true"
    assert meta["enable_gpu"] == "true" and meta["enable_internet"] == "true"
    kernel = (folder / "kernel.py").read_text()
    compile(kernel, "kernel.py", "exec")
    assert "recruit-env@abc1234" in kernel
    assert repr((ROOT / "examples" / "train_grpo.py").read_text()) in kernel
    assert "'--max-steps', '3'" in kernel
    for secret in ("HF_TOKEN", "KAGGLE_API_TOKEN", "KGAT_", "hf_"):
        assert secret not in kernel


def test_precision_never_emulates_bf16():
    assert train_grpo.pick_precision("auto", cpu=True) == "fp32"
    assert train_grpo.pick_precision("fp16", cpu=False) == "fp16"
    # without a GPU (as in CI) auto falls back to fp32, never bf16
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        assert train_grpo.pick_precision("auto", cpu=False) == "fp32"


def test_arms_run_one_per_gpu(tmp_path):
    arms = [kaggle_train.parse_arm("lr1e-6=--learning-rate 1e-6"),
            kaggle_train.parse_arm("lr1e-5=--learning-rate 1e-5")]
    assert arms[1] == ("lr1e-5", ["--learning-rate", "1e-5"])
    kernel = (kaggle_train.build("x", "o", "abc", ["--max-steps", "3"], tmp_path, arms) / "kernel.py").read_text()
    compile(kernel, "kernel.py", "exec")
    assert "CUDA_VISIBLE_DEVICES" in kernel and "'lr1e-5', ['--learning-rate', '1e-5']" in kernel


def test_build_runs_the_warm_start_pipeline(tmp_path):
    kernel = (kaggle_train.build("x", "o", "abc", ["--grpo"], tmp_path, [("plain", [])], "warm_start.py")
              / "kernel.py").read_text()
    compile(kernel, "kernel.py", "exec")
    assert "['warm_start.py', 'pipeline', '--out']" in kernel
    assert repr((ROOT / "examples" / "warm_start.py").read_text()) in kernel
    import pytest
    with pytest.raises(ValueError):
        kaggle_train.build("x", "o", "abc", [], tmp_path, script="other.py")


def test_counting_env_adds_no_tools_and_counts():
    from recruitenv.tools import RecruitToolEnv

    Env = train_grpo.counting_env(RecruitToolEnv)
    assert Env.__name__ == "RecruitToolEnv"
    public = lambda c: {n for n in dir(c) if not n.startswith("_")}
    assert public(Env) == public(RecruitToolEnv)
    e = Env()
    e.reset(task="easy", seed=3)
    e.check_github("candidate_001")
    e.shortlist("candidate_001")
    try:
        e.shortlist("candidate_001")      # illegal: already decided; not counted
    except ValueError:
        pass
    assert Env._take_counts() == (1.0, 1.0)
    assert Env._take_counts() == (0.0, 0.0)
