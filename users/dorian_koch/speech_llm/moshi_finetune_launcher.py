"""
moshi_finetune_launcher.py — drop-in replacement entry point for moshi-finetune.

Imports moshi_finetune.train, swaps out its data loader for our arrow-capable
version, then runs the same fire.Fire(train) CLI.  Zero edits to the fork.

Usage:
    python -m i6_experiments.users.dorian_koch.speech_llm.moshi_finetune_launcher \
        /path/to/config.yaml
"""

import sys
import importlib
import os

import fire

# Pin the fork's own directory on sys.path so `finetune` -- the fork's internal top-level
# package, imported as `from finetune.args import ...` inside moshi_finetune/train.py --
# resolves to the fork SOURCE (recipe/moshi_finetune/finetune, which ships the `data` and
# `monitoring` subpackages) rather than a stale/partial `finetune` copy that can sit in the
# job venv's site-packages. That copy is built from `packages=["finetune"]` (moshi_finetune's
# pyproject), which omits `finetune.data`, so importing it dies with
# `ModuleNotFoundError: No module named 'finetune.data'`. Inserting at position 0 makes the
# source win over any such shadowing copy. `import moshi_finetune` is cheap (empty __init__)
# and does not itself trigger the `finetune` import.
import moshi_finetune  # noqa: E402  (needed before .train to pin the path below)

_fork_dir = os.path.dirname(os.path.abspath(moshi_finetune.__file__))
if _fork_dir not in sys.path:
    sys.path.insert(0, _fork_dir)

# Import the fork's train module so we can patch it before fire runs it.
import moshi_finetune.train as train_module  # type: ignore

# Guard: fail loudly if the fork's internal structure changed.
assert hasattr(train_module, "build_data_loader"), (
    "moshi_finetune.train no longer exports 'build_data_loader'; "
    "check the fork for API changes and update this launcher."
)

# Import our arrow-capable loader.
from i6_experiments.users.dorian_koch.speech_llm.moshi_arrow_dataset import (
    build_data_loader as arrow_build_data_loader,
)

# Monkeypatch: replace the name in the fork's module namespace.
# train.py calls build_data_loader at L168 using the module-level name,
# so rebinding it here takes effect for the entire training run.
train_module.build_data_loader = arrow_build_data_loader


def main():
    """Entry point; installs our data config, then delegates to fire.Fire(train)."""
    # The config.yaml path is the sole CLI arg (see MoshiFinetune.run). Load the
    # ArrowDataConfig sidecar written beside it and install it before fire builds
    # the data loader. No-op default if the sidecar is absent.
    from i6_experiments.users.dorian_koch.speech_llm.moshi_arrow_config import (
        ArrowDataConfig,
    )
    from i6_experiments.users.dorian_koch.speech_llm import moshi_arrow_dataset

    if len(sys.argv) > 1:
        cfg = ArrowDataConfig.load_beside(sys.argv[1])
        moshi_arrow_dataset.set_active_config(cfg)
        print(f"[launcher] installed ArrowDataConfig: {cfg}", flush=True)

    fire.Fire(train_module.train)


if __name__ == "__main__":
    main()
