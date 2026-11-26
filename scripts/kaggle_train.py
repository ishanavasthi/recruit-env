"""Run ``examples/train_grpo.py`` on a free Kaggle GPU as a private notebook.

Kaggle script notebooks are a single file, so this builds one that embeds the
local ``examples/train_grpo.py``, installs ``recruitenv`` at a pinned commit
plus TRL, and runs the script with the arguments given after ``--``. Outputs
land in ``/kaggle/working/grpo`` and are fetched with ``kaggle kernels output``.

    # build only (inspect the folder before pushing)
    python scripts/kaggle_train.py build --slug recruitenv-grpo-pilot -- --max-steps 10

    # build and push (needs the Kaggle CLI and an API token)
    python scripts/kaggle_train.py push --slug recruitenv-grpo-pilot -- --max-steps 10

    # two arms in parallel, one per T4 (Kaggle's GPU notebooks have two)
    python scripts/kaggle_train.py push --slug recruitenv-grpo-easy \
        --arm "lr1e-6=--learning-rate 1e-6" --arm "lr1e-5=--learning-rate 1e-5" -- --max-minutes 600

With arms, the arguments after ``--`` are shared and each arm adds its own;
arm ``N`` runs on GPU ``N`` and writes to ``/kaggle/working/grpo/<name>``,
with its console output in ``stdout.log`` there.

The notebook is private. It needs a phone-verified Kaggle account (for
internet access) and uses the T4 GPU. No tokens go into the notebook: the
model and the repository are public.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "examples" / "train_grpo.py"
DEFAULT_REF = "2bd9e4a"     # 4.0.0; the package the notebook installs
PACKAGES = ["trl>=1.9,<2", "transformers>=5.2", "datasets", "accelerate"]

KERNEL_TEMPLATE = '''\
# Built by scripts/kaggle_train.py in the recruit-env repository. Do not edit.
import subprocess, sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", {install!r}, *{packages!r}], check=True)
Path("train_grpo.py").write_text({source!r})
arms = {arms!r}
if not arms:
    argv = [sys.executable, "train_grpo.py", "--output-dir", "/kaggle/working/grpo", *{train_args!r}]
    print("running:", " ".join(argv), flush=True)
    sys.exit(subprocess.run(argv).returncode)

import os
procs = []
for gpu, (name, extra) in enumerate(arms):
    out = Path("/kaggle/working/grpo") / name
    out.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "train_grpo.py", "--output-dir", str(out), *{train_args!r}, *extra]
    print(f"arm {{name}} on GPU {{gpu}}:", " ".join(argv), flush=True)
    log = open(out / "stdout.log", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    procs.append((name, subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env), log))
codes = {{}}
for name, proc, log in procs:
    codes[name] = proc.wait()
    log.close()
    print(f"arm {{name}} exited with {{codes[name]}}", flush=True)
sys.exit(max(codes.values()))
'''


def parse_arm(spec: str) -> tuple[str, list[str]]:
    """``"name=--learning-rate 1e-5"`` -> ``("name", ["--learning-rate", "1e-5"])``."""
    name, _, extra = spec.partition("=")
    if not name or not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"bad arm name in {spec!r}")
    return name, shlex.split(extra)


def build(slug: str, owner: str, ref: str, train_args: list[str], folder: Path,
          arms: list[tuple[str, list[str]]] | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    kernel = KERNEL_TEMPLATE.format(
        install=f"recruitenv @ git+https://github.com/ishanavasthi/recruit-env@{ref}",
        packages=PACKAGES,
        source=TRAIN_SCRIPT.read_text(),
        train_args=train_args,
        arms=[list(a) for a in arms or []],
    )
    compile(kernel, "kernel.py", "exec")
    (folder / "kernel.py").write_text(kernel)
    meta = {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return folder


def main() -> None:
    argv = sys.argv[1:]
    train_args: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, train_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["build", "push"])
    ap.add_argument("--slug", required=True, help="notebook slug, e.g. recruitenv-grpo-easy")
    ap.add_argument("--owner", default="ishanavasthi", help="Kaggle username")
    ap.add_argument("--ref", default=DEFAULT_REF, help="recruit-env commit to install")
    ap.add_argument("--folder", type=Path, default=None, help="build folder (default outputs/kaggle/<slug>)")
    ap.add_argument("--kaggle", default="kaggle", help="Kaggle CLI command")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=ARGS",
                    help="run several configurations in parallel, one per GPU (at most 2 on Kaggle)")
    args = ap.parse_args(argv)

    arms = [parse_arm(a) for a in args.arm]
    if len(arms) > 2:
        ap.error("Kaggle GPU notebooks have two GPUs; use at most two arms")
    folder = build(args.slug, args.owner, args.ref, train_args,
                   args.folder or ROOT / "outputs" / "kaggle" / args.slug, arms)
    print(f"built {folder}")
    if args.action == "push":
        cmd = [*shlex.split(args.kaggle), "kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4"]
        sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
