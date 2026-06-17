"""Generic, architecture-parameterized finetuning harness.

The Sisyphus harness for a LoRA/full finetune is identical across speech-LLM
architectures -- only three things differ: the ``config.yaml`` schema/template, the
training entry-point module, and the fork package that must be on ``PYTHONPATH``.
We capture exactly those in a small frozen :class:`FinetuneAdapter` and keep one
generic job, :class:`SpeechFinetune`, that consumes an adapter. "Add an
architecture" therefore means "write one adapter", not "copy a Job".

This is the de-duplicated home of the harness logic that used to live inline in
``moshi.py:MoshiFinetune``. ``MoshiFinetune`` stays a thin, hash-frozen shim over
these helpers (its public ``__init__`` / ``__sis_hash_exclude__`` are unchanged, so
every existing Moshi finetune keeps its exact hash and is not re-run); all *new*
architectures (PersonaPlex, RAG variants, ablation sweeps) go through
``SpeechFinetune(adapter=...)``.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

from sisyphus import Job, Task

from .common import HF_CACHE_DIR, last_jsonl_value
from .moshi_arrow_config import ArrowDataConfig


# --------------------------------------------------------------------------- #
# Adapter: everything architecture-specific, bundled as data.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FinetuneAdapter:
    """Per-architecture training recipe consumed by :class:`SpeechFinetune`.

    Args:
        name: stable identifier; this is the *only* thing the adapter contributes
            to a job's Sisyphus hash (see ``SpeechFinetune.hash``), so refactoring
            the callables below never re-hashes existing runs.
        batch_size: training batch size (also used to size ``max_steps`` per epoch).
        render_config: ``(job, batch_size, max_steps) -> str`` returning the full
            ``config.yaml`` text for this architecture's trainer.
        launcher_module: dotted module run as ``python -m <module> <config.yaml>``.
        fork_module: importable package whose parent dir is prepended to
            ``PYTHONPATH`` at launch (e.g. ``"moshi_finetune"``); imported lazily on
            the compute node so the manager env need not have it.
        progress: ``(metrics_file_relpath, json_field)`` tail-read for
            ``completed_fraction`` (defaults to moshi-finetune's metrics file).
    """

    name: str
    batch_size: int
    render_config: Callable[["SpeechFinetune", int, int], str]
    launcher_module: str
    fork_module: str
    progress: tuple[str, str] = ("metrics.train.jsonl", "percent_done")


# --------------------------------------------------------------------------- #
# Shared harness helpers (used by both SpeechFinetune and the MoshiFinetune shim).
# --------------------------------------------------------------------------- #
def resolve_max_steps(*, train_data, duration_sec: int, num_epochs, max_steps: int, batch_size: int) -> int:
    """Sanity-check durations and, if ``num_epochs`` is set, size ``max_steps`` to
    cover that many epochs over the whole dataset.

    Raises if >1% of dialogues are longer than ``duration_sec`` (otherwise the
    loader would silently truncate content and the window count below would be
    wrong). Assumes single-GPU training (world_size == 1).
    """
    import numpy as np
    from datasets import load_from_disk

    durations = np.asarray(load_from_disk(train_data.get())["duration"], dtype=float)
    over_frac = float((durations > duration_sec).mean())
    if over_frac > 0.01:
        raise ValueError(
            f"{over_frac:.2%} of audios exceed duration_sec={duration_sec} "
            f"(>1% not allowed; p99={np.percentile(durations, 99):.1f}s). "
            f"Increase duration_sec."
        )
    if num_epochs is None:
        return max_steps
    # windows per row = ceil(duration / duration_sec); matches the loader.
    windows = int(np.ceil(durations / duration_sec).sum())
    steps_per_epoch = int(np.ceil(windows / batch_size))
    return steps_per_epoch * num_epochs


def prepare_run_dir(run_dir: str) -> None:
    """Move aside a pre-existing, non-empty run_dir so a re-run starts clean."""
    if os.path.exists(run_dir) and os.listdir(run_dir):
        print(f"Warning: run_dir {run_dir} already exists and is not empty.")
        new_dir = os.path.join(os.getcwd(), "moshi_finetune_old_runs")
        os.makedirs(new_dir, exist_ok=True)
        cand = os.path.join(new_dir, "0001")
        while os.path.exists(cand):
            cand = os.path.join(new_dir, f"{int(os.path.basename(cand)) + 1:04d}")
        print(f"Moving existing contents to {cand}")
        os.rename(run_dir, cand)


def write_finetune_config(job: "SpeechFinetune", adapter: FinetuneAdapter) -> None:
    """Shared ``write_config`` body: prep run_dir, size steps, save the data-aug
    sidecar, render the architecture's config.yaml."""
    prepare_run_dir(job.out_rundir.get())
    batch_size = adapter.batch_size
    max_steps = resolve_max_steps(
        train_data=job.train_data,
        duration_sec=job.duration_sec,
        num_epochs=job.num_epochs,
        max_steps=job.max_steps,
        batch_size=batch_size,
    )
    # Persist our data/augmentation config beside config.yaml; the launcher loads it
    # at training time (avoids touching the fork's TrainArgs schema).
    ArrowDataConfig(jitter_max_sec=job.audio_jitter_sec).save_beside(job.out_config.get())
    text = adapter.render_config(job, batch_size, max_steps)
    with open(job.out_config, "w") as f:
        f.write(text)


def finetune_completed_fraction(job: "SpeechFinetune", adapter: FinetuneAdapter):
    """Shared ``completed_fraction``: tail-read the trainer's metrics file."""
    rel, field_name = adapter.progress
    pct = last_jsonl_value(os.path.join(job.out_rundir.get_path(), rel), field_name)
    if pct is None:
        return None
    return max(0.0, min(1.0, pct / 100.0))


def launch_training(job: "SpeechFinetune", adapter: FinetuneAdapter) -> None:
    """Shared ``run`` body: single-node ``torch.distributed.run`` of the adapter's
    launcher module, with the fork on ``PYTHONPATH`` and the HF cache wired."""
    import hashlib
    import subprocess

    # Deterministic per-GPU MASTER_PORT (PYTHONHASHSEED-independent) so concurrent
    # single-node trainings on one machine don't collide.
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "-1")
    port_offset = int(hashlib.md5(cuda_devices.encode()).hexdigest()[:4], 16) % 100

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HOME"] = HF_CACHE_DIR.get()
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = str(29600 + port_offset)  # avoid conflicts if multiple run on one machine
    print(f"Set MASTER_PORT to {env['MASTER_PORT']} based on hash of CUDA_VISIBLE_DEVICES")

    command = [
        job.venv_python_path.get(),
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(job.rqmt["gpu"]),
        f"--rdzv_endpoint={env['MASTER_ADDR']}:{env['MASTER_PORT']}",
        "-m",
        adapter.launcher_module,
        job.out_config.get(),
    ]

    # Import the fork lazily (on the compute node, where its venv has it) to get its
    # path for PYTHONPATH; both the package dir and its parent are added, matching the
    # original MoshiFinetune behaviour.
    fork = importlib.import_module(adapter.fork_module)
    top_level_file = fork.__file__
    assert top_level_file is not None, f"Could not find {adapter.fork_module} module file"
    package_base_dir = f"{str(Path(top_level_file).parent.parent)}{os.pathsep}{str(Path(top_level_file).parent)}"
    env["PYTHONPATH"] = (
        f"{package_base_dir}{os.pathsep}{env['PYTHONPATH']}" if "PYTHONPATH" in env else package_base_dir
    )
    print(f"Running {adapter.name} training with command: {' '.join(command)}", flush=True)
    print(f"Using HF cache directory: {HF_CACHE_DIR}")
    subprocess.run(command, env=env, check=True)


# --------------------------------------------------------------------------- #
# Moshi adapter: byte-identical to the original MoshiFinetune config template.
# --------------------------------------------------------------------------- #
def _render_moshi_finetune_config(job: "SpeechFinetune", batch_size: int, max_steps: int, *, hf_repo_id: str) -> str:
    """Render a moshi-finetune ``config.yaml``. Architecture-agnostic except the base
    model repo, so Moshi and (scaffolded) PersonaPlex share it."""
    run_dir = job.out_rundir.get()
    return f"""
# data
data:
  eval_data: '{job.eval_data.get() if job.eval_data is not None else ""}' # Fill
  shuffle: true
  train_data: '{job.train_data.get()}' # Fill

# model
moshi_paths:
  hf_repo_id: "{hf_repo_id}"

full_finetuning: false # Activate lora.enable if partial finetuning
lora:
  enable: true # Set to False if full_finetuning is True
  rank: {job.lora_rank}
  scaling: 2.
  ft_embed: false # Optional, set to True if you want to finetune the embedding layer

first_codebook_weight_multiplier: 100.
text_padding_weight: .5

# optim
duration_sec: {job.duration_sec}
batch_size: {batch_size}
max_steps: {max_steps}
gradient_checkpointing: true
optim:
  lr: 2e-6
  weight_decay: 0.1
  pct_start: 0.05

# other
seed: {getattr(job, "seed", 0)}
log_freq: 1
eval_freq: 100
do_eval: {"true" if job.eval_data is not None else "false"}
do_ckpt: true
ckpt_freq: 100
overwrite_run_dir: true

save_adapters: true # Must be False if full_finetuning is True

run_dir: "{run_dir}"  # Fill
"""


MOSHI_ADAPTER = FinetuneAdapter(
    name="moshi",
    batch_size=16,
    render_config=partial(_render_moshi_finetune_config, hf_repo_id="kyutai/moshiko-pytorch-bf16"),
    launcher_module="i6_experiments.users.dorian_koch.speech_llm.moshi_finetune_launcher",
    fork_module="moshi_finetune",
)


# --------------------------------------------------------------------------- #
# PersonaPlex adapter (SCAFFOLD -- see projects/2026-01-speech-llm/personaplex.md).
# PersonaPlex ships an *inference* fork; there is (likely) no released finetuning code,
# so the launcher + fork below are placeholders to finalize once a PersonaPlex-aware
# training loop exists. The harness + data pipeline are already reusable; only this
# adapter (and the launcher it names) need filling in. Wire via
# ``SpeechFinetune(adapter=PERSONAPLEX_ADAPTER, ...)``.
# --------------------------------------------------------------------------- #
PERSONAPLEX_ADAPTER = FinetuneAdapter(
    name="personaplex",
    batch_size=16,
    # VERIFY(personaplex): config schema for the PersonaPlex trainer (assumed
    # moshi-finetune-compatible here; 7B may need multi-GPU + schema tweaks).
    render_config=partial(_render_moshi_finetune_config, hf_repo_id="nvidia/personaplex-7b-v1"),
    # VERIFY(personaplex): training entry point + fork package (no released code yet).
    launcher_module="i6_experiments.users.dorian_koch.speech_llm.personaplex_finetune_launcher",
    fork_module="moshi_finetune",
)


# --------------------------------------------------------------------------- #
# MoshiRAG adapter (SCAFFOLD -- see projects/2026-01-speech-llm/moshirag.md).
# MoshiRAG (kyutai-labs/moshi-rag, arXiv 2604.12928) adds an ARC-Encoder reference
# conditioner + a <ret> retrieval-trigger token to Moshi. The released fork is
# *inference-only*; training requires wiring that conditioner's forward/collate +
# <ret>/reference-dropout/retrieval-delay-sim into the loop (the launcher stub raises until
# then). We design for BOTH init paths but only checkpoint-init is wired now:
#   * init_from = "kyutai/moshika-rag-pytorch-bf16"  (LoRA on top; conditioner + <ret>
#     already trained) -- the path we'd actually run.  [WIRED as the config base below]
#   * init_from = None / base moshiko  (train the conditioner from scratch -- the biggest
#     run we'd ever do, base-moshi -> moshirag) -- SEAM ONLY, multi-month; see D3 in
#     moshirag.md.  To enable, swap hf_repo_id below + extend the launcher.
# Wire via ``SpeechFinetune(adapter=MOSHIRAG_ADAPTER, venv_python_path=moshirag_venv(), ...)``.
# --------------------------------------------------------------------------- #
MOSHIRAG_ADAPTER = FinetuneAdapter(
    name="moshirag",
    batch_size=16,
    # Checkpoint-init: LoRA-finetune on top of the released MoshiRAG checkpoint (conditioner
    # + <ret> already present). VERIFY(moshirag): the RAG trainer needs extra config
    # (reference conditioner on, reference-dropout 0.2, retrieval-delay sim) the moshi-finetune
    # schema does not express -- the launcher must inject those; see moshirag.md.
    render_config=partial(_render_moshi_finetune_config, hf_repo_id="kyutai/moshika-rag-pytorch-bf16"),
    launcher_module="i6_experiments.users.dorian_koch.speech_llm.moshirag_finetune_launcher",
    fork_module="moshi",  # the moshi-rag fork (import name `moshi`); installed by moshirag_venv()
)


# --------------------------------------------------------------------------- #
# Generic job for *new* architectures (Moshi keeps its own frozen class).
# --------------------------------------------------------------------------- #
class SpeechFinetune(Job):
    """Architecture-parameterized finetune job; pass a :class:`FinetuneAdapter`.

    Same knobs as the legacy ``MoshiFinetune`` (so behaviour matches when given
    ``MOSHI_ADAPTER``), plus ``adapter`` which selects the model/trainer. The
    adapter contributes only its ``name`` to the hash.
    """

    __sis_hash_exclude__ = {
        "duration_sec": 100,
        "audio_jitter_sec": 0.0,
        "num_epochs": None,
        "max_steps": 2000,
        "eval_data": None,
        "lora_rank": 128,
    }

    def __init__(
        self,
        *,
        adapter: FinetuneAdapter,
        venv_python_path,
        train_data,
        seed: int = 0,
        duration_sec: int = 100,
        audio_jitter_sec: float = 0.0,
        num_epochs: int | None = None,
        max_steps: int = 2000,
        eval_data=None,
        lora_rank: int = 128,
    ):
        self.adapter = adapter
        self.train_data = train_data
        self.eval_data = eval_data
        self.venv_python_path = venv_python_path
        self.seed = seed
        self.duration_sec = duration_sec
        self.audio_jitter_sec = audio_jitter_sec
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.lora_rank = lora_rank
        self.out_config = self.output_path("config.yaml")
        self.out_rundir = self.output_path("run_dir", directory=True)
        self.rqmt = {"gpu": 1, "cpu": 6, "mem": 24, "time": 23}

    @classmethod
    def hash(cls, parsed_args):
        # The adapter is bundled callables; hash it by its stable ``name`` only, so
        # refactoring the adapter's functions never re-hashes existing runs.
        d = dict(parsed_args)
        d["adapter"] = d["adapter"].name
        return super().hash(d)

    def tasks(self):
        yield Task("write_config", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        return finetune_completed_fraction(self, self.adapter)

    def write_config(self):
        write_finetune_config(self, self.adapter)

    def run(self):
        launch_training(self, self.adapter)
