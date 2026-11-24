"""scripts/run_benchmark.py and scripts/baselines_report.py."""

from __future__ import annotations

import json

import inference
from recruitenv.trajectory import replay
from scripts import baselines_report, run_benchmark
from tests.test_inference import FakeLLM


def test_parse_seeds():
    assert run_benchmark.parse_seeds("0-3,7") == [0, 1, 2, 3, 7]


def test_policy_run_is_resumable_and_replayable(tmp_path, capsys):
    args = ["--agent", "rubric_heuristic", "--seeds", "0-1", "--tasks", "easy,hard", "--out", str(tmp_path)]
    assert run_benchmark.main(args) == 0
    run = tmp_path / "rubric_heuristic"
    assert len(list(run.glob("*.json"))) == 5          # 4 results + run.json
    for log in run.glob("*.jsonl"):
        replay(log)
    capsys.readouterr()
    assert run_benchmark.main(args) == 0
    assert "0 episodes to run (4 already done)" in capsys.readouterr().out


def test_model_run_counts_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "make_client", lambda *a, **k: FakeLLM(["cutoff"] * 100))
    monkeypatch.setattr(inference.ModelChooser, "backoff_s", 0.0)
    args = ["--agent", "model:fake", "--seeds", "0", "--tasks", "easy", "--out", str(tmp_path), "--max-tokens", "100"]
    assert run_benchmark.main(args) == 0
    res = json.loads((tmp_path / "model__fake" / "easy-s0.json").read_text())
    assert res["incomplete"] is True and res["usage"]["length_cutoffs"] == 4


def test_report(tmp_path):
    run_benchmark.main(["--agent", "oracle", "--seeds", "0-2", "--tasks", "easy,medium,hard", "--out", str(tmp_path)])
    run_benchmark.main(["--agent", "random", "--seeds", "0-2", "--tasks", "easy,medium,hard", "--out", str(tmp_path)])
    out = tmp_path / "report"
    assert baselines_report.main([str(tmp_path), "--out", str(out), "--common-seeds"]) == 0
    rep = json.loads(out.with_suffix(".json").read_text())
    oracle = rep["agents"]["oracle"]["all_seeds"]["hard"]
    assert oracle["episodes"] == 3 and oracle["mean"] >= 0.99 and oracle["pass_rate"] == 1.0
    assert rep["agents"]["random"]["all_seeds"]["hard"]["mean"] < 0.7
    md = out.with_suffix(".md").read_text()
    assert "| oracle |" in md and "n=3" in md


def test_bootstrap_ci_brackets_mean():
    lo, hi = baselines_report.bootstrap_ci([0.2, 0.4, 0.6, 0.8])
    assert lo < 0.5 < hi
