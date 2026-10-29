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
import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

from sisyphus import Job, Task

from . import common as _common
from .common import HF_CACHE_DIR, HF_HOME_DIR, last_jsonl_value, moshi_family_lib_parent, recipe_root, worker_env
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
        pythonpath_package: importable package whose dir + parent are prepended to the
            training subprocess's ``PYTHONPATH`` (e.g. ``"moshi_family"``, or a fork's
            ``"moshi_finetune"``); located lazily on the compute node -- in the JOB's venv,
            not the manager's -- so the manager env need not have it installed.
        progress: ``(metrics_file_relpath, json_field)`` tail-read for
            ``completed_fraction`` (defaults to moshi-finetune's metrics file).
    """

    name: str
    batch_size: int
    render_config: Callable[["SpeechFinetune", int, int], str]
    launcher_module: str
    pythonpath_package: str
    progress: tuple[str, str] = ("metrics.train.jsonl", "percent_done")

    # --- what DOWNSTREAM work needs to know about this architecture -------------------------
    # These were previously restated by hand at every consumer, or buried where nothing could read
    # them: `overlay_kind` was a string literal repeated at each ResolveOverlayCheckpoint call, and
    # `base_model` was trapped inside render_config's functools.partial, invisible to the recipe.
    # Declaring them on the adapter means a training run can describe itself (see
    # sis_recipe/doriank/runs.py) and a consumer never has to know which family it is holding.
    #
    # Safe to add: SpeechFinetune.hash() replaces the adapter with adapter.name, so extra fields
    # here cannot move any existing job's hash.
    #: Discriminator for ResolveOverlayCheckpoint: how the trained delta sits on the base model.
    overlay_kind: str = "lora"
    #: HF repo of the frozen base this architecture finetunes.
    base_model: str = ""
    #: Filename of the trained weights inside a checkpoint dir.
    weights_name: str = "lora.safetensors"


# --------------------------------------------------------------------------- #
# Shared harness helpers (used by both SpeechFinetune and the MoshiFinetune shim).
# --------------------------------------------------------------------------- #
def train_data_specs(train_data) -> list[tuple[object, float]]:
    """Normalise ``SpeechFinetune.train_data`` to ``[(path, weight), ...]``.

    Accepts either a single ``tk.Path`` (weight 1.0) or a **list of ``(path, weight)`` tuples** for
    an on-the-fly weighted mix. Weights are relative; the loader normalises them.

    ⚠ Must be a list of tuples, NOT a ``{path: weight}`` dict. Sisyphus discovers a job's inputs by
    traversing its constructor arguments and does **not** walk dict *keys*, so a Path in key position
    creates no dependency edge: the job looks runnable immediately and starts before its training
    data exists (observed as `write_config` failing with "neither a `Dataset` directory nor a
    `DatasetDict` directory" while the corpus was still being written).
    """
    assert not hasattr(train_data, "items"), (
        "train_data must be a single Path or a list of (path, weight) tuples -- a dict keyed by "
        "Path silently loses the Sisyphus dependency edge (see docstring)"
    )
    if isinstance(train_data, (list, tuple)):
        # each entry is (path, weight) or (path, weight, window_sec); a set window_sec marks a
        # full-conversation corpus the loader slices to a random window_sec window per draw.
        specs = [(t[0], float(t[1]), (float(t[2]) if len(t) > 2 and t[2] else None)) for t in train_data]
        assert specs, "train_data mix is empty"
        assert all(w > 0 for _, w, _ in specs), f"train_data weights must be positive, got {[w for _, w, _ in specs]}"
        return specs
    return [(train_data, 1.0, None)]


def resolve_max_steps(*, train_data, duration_sec: int, num_epochs, max_steps: int, batch_size: int) -> int:
    """Sanity-check durations and, if ``num_epochs`` is set, size ``max_steps`` to
    cover that many epochs over the whole dataset.

    Raises if >1% of dialogues are longer than ``duration_sec`` (otherwise the
    loader would silently truncate content and the window count below would be
    wrong). Assumes single-GPU training (world_size == 1).
    """
    import numpy as np
    from datasets import load_from_disk

    # Validate EVERY corpus in the mix: a row longer than duration_sec is silently truncated by
    # build_codes, so this guard is the only thing standing between a too-small window and quietly
    # training on cut-off dialogues.
    all_durations = []
    for path, _weight, window_sec in train_data_specs(train_data):
        durations = np.asarray(load_from_disk(path.get())["duration"], dtype=float)
        if window_sec:
            # full-conversation corpus: rows are longer than the window BY DESIGN (the loader cuts a
            # random window_sec window per draw), so the truncation guard does not apply; count each
            # row as one window's worth for the epoch-size estimate.
            all_durations.append(np.minimum(durations, window_sec))
            continue
        over_frac = float((durations > duration_sec).mean())
        if over_frac > 0.01:
            raise ValueError(
                f"{over_frac:.2%} of audios in {path.get()} exceed duration_sec={duration_sec} "
                f"(>1% not allowed; p99={np.percentile(durations, 99):.1f}s). "
                f"Increase duration_sec."
            )
        all_durations.append(durations)
    if num_epochs is None:
        return max_steps
    # windows per row = ceil(duration / duration_sec); matches the loader.
    # For a mix, "an epoch" is one pass over the pooled rows -- approximate, since the loader samples
    # by weight rather than sweeping each corpus once. Prefer an explicit max_steps for mixed runs.
    durations = np.concatenate(all_durations)
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
    # lr_schedule (2026-10-13) is rendered by the PersonaPlex LoRA / full renderer only: refuse it elsewhere rather than
    # train a silent cosine
    if (getattr(job, "hparams", None) or {}).get("lr_schedule") not in (None, "cosine") and adapter.name not in (
        "personaplex_lib_lora",
        "personaplex_lib_full",
    ):
        raise ValueError(f"{adapter.name}: lr_schedule is implemented for the PersonaPlex launcher only")
    text = adapter.render_config(job, batch_size, max_steps)
    with open(job.out_config, "w") as f:
        f.write(text)


class _ReadTrackingDict(dict):
    """A dict that records every key read through ``[]``, ``get`` or ``in``."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.read: set = set()

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self.read.add(k)
        return super().get(k, default)

    def __contains__(self, k):
        self.read.add(k)
        return super().__contains__(k)


#: hparams keys read outside the renderers: scheduling (``SpeechFinetune.__init__``, ``declared_train_gpus``), the
#: step plan evals check against (``speech_inference.assert_checkpoint_will_exist``), and ``data_pipeline_version``,
#: a marker whose only job is to move the hash (``train_config.merge_hparams``).
HPARAMS_READ_ELSEWHERE = frozenset({"gpu", "rqmt_time_h", "max_steps", "num_epochs", "data_pipeline_version"})


def unread_hparams(job: "SpeechFinetune", adapter: FinetuneAdapter, max_steps: int = 1000) -> list[str]:
    """``job.hparams`` keys its adapter's renderer never reads: hashed knobs that configure nothing (the RL side's
    ``check_knobs_rendered``). Renders at graph time from a copy; the job is untouched."""
    hp = _ReadTrackingDict(getattr(job, "hparams", None) or {})
    if not hp:
        return []
    saved = job.hparams
    try:
        job.hparams = hp
        adapter.render_config(job, adapter.batch_size, max_steps)
    finally:
        job.hparams = saved
    return sorted(set(hp) - hp.read - HPARAMS_READ_ELSEWHERE)


def finetune_completed_fraction(job: "SpeechFinetune", adapter: FinetuneAdapter):
    """Shared ``completed_fraction``: tail-read the trainer's metrics file."""
    rel, field_name = adapter.progress
    pct = last_jsonl_value(os.path.join(job.out_rundir.get_path(), rel), field_name)
    if pct is None:
        return None
    return max(0.0, min(1.0, pct / 100.0))


def declared_train_gpus(job) -> int:
    """The GPU count a training job declares: ``compute["gpu"]``, else ``hparams["gpu"]``, else 1. Jobs without
    ``hparams`` (``RLFinetune``, ``AudexDuplexFinetune``) declare 1 -- reading ``job.hparams`` unguarded crashed the
    first RLFinetune after the 2026-09-23 fix, after its judge had loaded (guard ``check_declared_train_gpus.py``)."""
    return int((getattr(job, "compute", None) or {}).get("gpu", (getattr(job, "hparams", None) or {}).get("gpu", 1)))


def training_hf_env(job: "SpeechFinetune") -> dict:
    """The worker env's HF part: the shared cache, or the job's own hub cache (``SpeechFinetune(hf_cache_dir=)``).

    Before 2026-10-17 a caller could swap the cache by patching this module's ``HF_CACHE_DIR`` / ``HF_HOME_DIR``
    (``mock.patch.multiple``). The env is now built by ``common.worker_env``, which reads ``common``'s globals, so such a
    patch would be silently ignored and the run would train from the shared cache. It is refused instead."""
    if HF_CACHE_DIR is not _common.HF_CACHE_DIR or HF_HOME_DIR is not _common.HF_HOME_DIR:
        raise RuntimeError(
            "speech_llm.finetune.HF_CACHE_DIR / HF_HOME_DIR were patched. That no longer reaches the training env "
            "(common.worker_env builds it), so the run would silently use the shared HF cache. Pass the hub cache as "
            "SpeechFinetune(hf_cache_dir=<tk.Path of the hub cache dir>) instead (HF_HOME then stays the job's own). "
            "A subclass with its own cache argument keeps its hash by forwarding it, e.g. "
            "super().__init__(hf_cache_dir=hf_hub_cache, **kwargs), and drops the patch from run()."
        )
    hf_cache_dir = getattr(job, "hf_cache_dir", None)  # getattr: jobs pickled before the argument existed
    env = worker_env(offline=True, pythonpath=None, hf_cache=hf_cache_dir is None)
    if hf_cache_dir is not None:
        env["HF_HUB_CACHE"] = hf_cache_dir.get()
    return env


def launch_training(job: "SpeechFinetune", adapter: FinetuneAdapter) -> None:
    """Shared ``run`` body: single-node ``torch.distributed.run`` of the adapter's
    launcher module, with the fork on ``PYTHONPATH`` and the HF cache wired."""
    import hashlib
    import subprocess

    # Deterministic MASTER_PORT (PYTHONHASHSEED-independent) so concurrent single-node trainings on one
    # machine don't collide. Key off SLURM_JOB_ID (unique per job, identical across a job's ranks): under
    # SLURM each single-GPU job's cgroup renumbers its visible GPU to "0", so hashing CUDA_VISIBLE_DEVICES
    # gave two co-located jobs the SAME port -> EADDRINUSE. Fall back to CUDA_VISIBLE_DEVICES off-SLURM.
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "-1")  # also used below for nproc
    port_key = os.environ.get("SLURM_JOB_ID") or cuda_devices
    port_offset = int(hashlib.md5(port_key.encode()).hexdigest()[:4], 16) % 4000

    # The base model + tokenizer are pre-staged into the HF cache by the eval graph, so training runs HF-offline: no
    # HF download / Xet re-verification, which (a) avoids re-fetching a 16 GB checkpoint every run and (b) does not
    # depend on writable HF cache space (the shared hpcwork cache can be full -- a Xet "Background writer channel
    # closed" download error is that symptom). HF_HUB_CACHE points AT the hub dir (HF appends "/hub" to HF_HOME;
    # common.HF_HOME_DIR), or at the job's own hf_cache_dir (training_hf_env). PYTHONPATH is built below (the adapter's
    # fork dirs).
    env = training_hf_env(job)
    # The in-loop knowledge probe generates on the SAME model training uses. moshi's generation and
    # training forward share torch.compile'd fns (``torch_compile_lazy``, e.g. ``apply_rope``) + CUDA
    # graphs; letting the probe's generation shapes into the compile cache poisons the guards
    # training's forward recompiles against -> ``torch._dynamo`` fake-tensor crash at the step AFTER a
    # probe (``s0`` vs ``s0*s98`` in apply_rope). ``apply_rope`` is a pure fn, so nothing is really
    # wrong -- it is purely a compile artifact. Disable compile + CUDA graphs for the whole run when a
    # probe is attached (eager is correct, slightly slower); run()-side env, not hashed.
    if getattr(job, "knowledge_probe_data", None) is not None:
        env["NO_TORCH_COMPILE"] = "1"
        env["NO_CUDA_GRAPH"] = "1"
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = str(20000 + port_offset)  # unique per SLURM job -> no EADDRINUSE across co-located jobs
    print(f"Set MASTER_PORT to {env['MASTER_PORT']} based on hash of SLURM_JOB_ID/CUDA_VISIBLE_DEVICES")

    # nproc-per-node = the GPUs actually handed to *training*, NOT job.rqmt["gpu"] (which counts the
    # whole allocation). When a vLLM judge is co-launched, the retrieval seam pins the trainer to a
    # single GPU via CUDA_VISIBLE_DEVICES (the judge takes the other), so rqmt["gpu"]==2 while only one
    # GPU is visible here. This launcher runs one single-GPU rank per proc and never maps rank->device,
    # so a larger nproc piles every rank onto the one visible GPU and OOMs (observed: 2x46GiB on a 93GiB
    # card). Count the visible devices; fall back to rqmt["gpu"] only when CUDA_VISIBLE_DEVICES is unset.
    _visible_gpus = [d for d in cuda_devices.split(",") if d.strip() not in ("", "-1")]
    n_train_gpus = len(_visible_gpus) if _visible_gpus else int(job.rqmt["gpu"])
    # ...but never MORE ranks than the run declared. The cluster may hand a job more GPUs than it
    # asked for (settings.MIN_GPUS_PER_JOB on a cluster whose smallest node is 4 cards); for an
    # inference job that is free speed, for a training run it would quadruple the effective batch
    # and silently make it a different experiment under the same hash. So the declared count wins
    # and the surplus idles -- an arm that should use 4 GPUs says so (Compute(gpus=4)).
    # `compute` first: from RULE_RQMT_NOT_HASHED (2026-09-17) the GPU count travels there, not in `hparams`,
    # and reading only `hparams` trained such a run on ONE rank of its N-GPU allocation (found 2026-09-23 on
    # p5 before it ran; no earlier multi-GPU arm used `compute`). Same fallback order as `SpeechFinetune.rqmt`.
    declared = declared_train_gpus(job)
    if n_train_gpus > declared:
        print(
            f"[launch_training] {n_train_gpus} GPUs visible but the run declares gpu={declared}; "
            f"training on {declared} rank(s), {n_train_gpus - declared} idle (declare Compute(gpus=N) "
            f"to use them -- it changes the effective batch).",
            flush=True,
        )
        n_train_gpus = declared

    command = [
        job.venv_python_path.get(),
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(n_train_gpus),
        f"--rdzv_endpoint={env['MASTER_ADDR']}:{env['MASTER_PORT']}",
        "-m",
        adapter.launcher_module,
        job.out_config.get(),
    ]

    # Locate the fork's install dir to prepend to the torchrun subprocess's PYTHONPATH (both the
    # package dir and its parent, matching the original MoshiFinetune behaviour). The fork lives in
    # the JOB's venv (job.venv_python_path), NOT necessarily this worker's .venv -- so query the job
    # venv for the module file rather than importing it here (the setup .venv has no moshi/
    # moshi_finetune; importing here crashed personaplex). If the fork is a normal site-packages
    # install (e.g. the personaplex `moshi`), the launcher imports it directly anyway, so a failed
    # lookup is non-fatal: skip the prepend with a warning.
    top_level_file = None
    try:
        package = importlib.import_module(adapter.pythonpath_package)  # fast path: worker venv has it
        top_level_file = package.__file__
    except ModuleNotFoundError:
        probe = subprocess.run(
            [job.venv_python_path.get(), "-c", f"import {adapter.pythonpath_package} as m; print(m.__file__)"],
            capture_output=True,
            text=True,
            env=env,
        )
        top_level_file = probe.stdout.strip() or None

    # Build the torchrun subprocess's PYTHONPATH. It is a fresh job-venv python, so neither the recipe
    # tree nor (for non-site-packages forks) the fork dir is on its path:
    #  * The launcher (adapter.launcher_module) is an i6_experiments module. Sisyphus puts the recipe
    #    root on the WORKER's sys.path programmatically -- not via PYTHONPATH -- so the subprocess
    #    can't import i6_experiments unless we add the recipe root explicitly. Resolve it from the
    #    already-imported i6_experiments package (recipe/i6_experiments/__init__.py -> recipe root).
    #  * The fork's package dir + parent are prepended too (matching the original MoshiFinetune
    #    behaviour). The launcher's own sys.path guard still wins for `import moshi` (site-packages
    #    fork beats recipe/moshi), so the recipe root on the path does not reintroduce shadowing.
    extra_paths: list[str] = []
    if top_level_file:
        extra_paths += [str(Path(top_level_file).parent.parent), str(Path(top_level_file).parent)]
    else:
        print(
            f"[launch_training] package {adapter.pythonpath_package!r} not locatable for PYTHONPATH; "
            f"relying on the job venv site-packages + launcher sys.path guard",
            flush=True,
        )
    # recipe root = the dir holding the i6_experiments (+ fork) symlinks. Walk up UNRESOLVED from this
    # file (recipe/i6_experiments/.../finetune.py): recipe/ is a symlink tree, so
    # ``i6_experiments.__file__.resolve()`` would land in projects/ -- which lacks the recipe/<fork>
    # symlinks (e.g. recipe/moshi_finetune -> projects/moshi-finetune; there is no projects/
    # moshi_finetune), breaking the launcher's ``import <fork>``. Keep it unresolved.
    root = recipe_root()
    extra_paths.append(root)
    # Lib launchers (``-m moshi_family.<...>``) live under recipe/speech_llm/full_duplex, which is
    # NOT the recipe root; add it so the owned moshi_family package imports (harmless for fork
    # launchers -- it is just one more importable dir on the path).
    lib_parent = moshi_family_lib_parent()
    if os.path.isdir(lib_parent):
        extra_paths.append(lib_parent)
    if env.get("PYTHONPATH"):
        extra_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    print(f"Running {adapter.name} training with command: {' '.join(command)}", flush=True)
    print(f"Using HF cache directory: {env['HF_HUB_CACHE']}")
    subprocess.run(command, env=env, check=True)


# --------------------------------------------------------------------------- #
# Moshi adapter: byte-identical to the original MoshiFinetune config template.
# --------------------------------------------------------------------------- #
def _render_moshi_finetune_config(job: "SpeechFinetune", batch_size: int, max_steps: int, *, hf_repo_id: str) -> str:
    """Render a moshi-finetune ``config.yaml``. Architecture-agnostic except the base
    model repo, so Moshi and (scaffolded) PersonaPlex share it.

    Extra knobs come from ``job.hparams`` (a plain dict) so the job signature never grows: ``lr``
    (default 2e-6) and ``full_finetuning`` (default False -> LoRA; True -> full-FT: LoRA off +
    save the full model). Text-replay is NOT supported by the fork (owned-launcher feature)."""
    hp = getattr(job, "hparams", None) or {}
    full_ft = bool(hp.get("full_finetuning", False))
    run_dir = job.out_rundir.get()
    return f"""
# data
data:
  eval_data: '{job.eval_data.get() if job.eval_data is not None else ""}' # Fill
  shuffle: true
  train_data: '{_single_train_data_path(job)}' # single corpus only (fork schema has no mix key)

# model
moshi_paths:
  hf_repo_id: "{hf_repo_id}"

full_finetuning: {str(full_ft).lower()}
lora:
  enable: {str(not full_ft).lower()} # False when full_finetuning
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
  lr: {hp.get("lr", 2e-6)}
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

save_adapters: {str(not full_ft).lower()} # False when full_finetuning

run_dir: "{run_dir}"  # Fill
"""


def _render_personaplex_config(job: "SpeechFinetune", batch_size: int, max_steps: int, *, hf_repo_id: str) -> str:
    """Render the PersonaPlex training config (consumed by moshi_family.personaplex.finetune_launcher, NOT the
    moshi_finetune schema). Paper values (arXiv 2602.06053): Adam+cosine, depformer LR 4e-6 /
    temporal 2e-6, batch 32, 24,576 steps, 163.84 s seq, full finetune; loss down-weights + system
    -prompt masking live in the launcher's personaplex_loss. ``system_prompt_key`` lets service rows
    carry a per-row role prompt (column "context"); QA rows fall back to the default persona."""
    return f"""# PersonaPlex finetune config (personaplex_finetune_launcher schema)
hf_repo_id: "{hf_repo_id}"
{_train_data_yaml(job, supports_mix=False)}
out_dir: "{job.out_rundir.get()}"
max_steps: {max_steps}
duration_sec: {job.duration_sec}
# Single-GPU (this cluster, for now): freeze the backbone, full-FT the depformer + heads; small
# per-GPU batch + grad-accum (effective batch 8). For the paper recipe use train_scope=full +
# multi-GPU torchrun + grad_accum to effective batch 32.
train_scope: "heads"
per_gpu_batch: 1
grad_accum: 8
lr_temporal: 2e-6
lr_depformer: 4e-6
warmup_steps: 200
grad_clip: 1.0
save_every: 500
log_every: 5
seed: {getattr(job, "seed", 0)}
system_prompt_key: "context"
"""


MOSHI_ADAPTER = FinetuneAdapter(
    name="moshi",
    batch_size=16,
    overlay_kind="lora",
    base_model="kyutai/moshiko-pytorch-bf16",
    weights_name="lora.safetensors",
    render_config=partial(_render_moshi_finetune_config, hf_repo_id="kyutai/moshiko-pytorch-bf16"),
    launcher_module="i6_experiments.users.dorian_koch.speech_llm.moshi_finetune_launcher",
    pythonpath_package="moshi_finetune",
)


# PersonaPlex on the OWNED moshi_family lib (Phase 5): same paper recipe + config schema, but the
# launcher builds on moshi_family.personaplex (no fork) via the SHARED moshi_family.train_loop, and
# runs in moshi_family_venv. New ``name`` -> fresh hash (intended; this is the fork->lib migration).
# pythonpath_package="moshi_family" is located via the lib path launch_training now adds to PYTHONPATH.
PERSONAPLEX_LIB_ADAPTER = FinetuneAdapter(
    name="personaplex_lib",
    batch_size=32,
    overlay_kind="personaplex_heads",
    base_model="nvidia/personaplex-7b-v1",
    weights_name="trained_heads.safetensors",
    render_config=partial(_render_personaplex_config, hf_repo_id="nvidia/personaplex-7b-v1"),
    launcher_module="moshi_family.personaplex.finetune_launcher",
    pythonpath_package="moshi_family",
)


#: Every training-speed launcher key (``moshi_family.train_perf``)
#: and its YAML type, in render order. A launcher reads only the keys that apply to its model; each
#: renderer names the subset it supports and refuses the rest (see :func:`render_train_perf_keys`).
TRAIN_PERF_KEYS = {
    "contiguous_step_inputs": bool,
    "fused_causal_attention": bool,
    "fused_lora_add": bool,
    "depformer_skip_masked": bool,
    "prefetch_batches": int,
    "length_bucket_pool": int,
}


def render_train_perf_keys(hp: dict, supported, *, who: str, rendered_elsewhere=()) -> str:
    """YAML lines for the training-speed keys ``hp`` sets, in :data:`TRAIN_PERF_KEYS` order; "" when none.

    Rendered ONLY when set, so an arm without them renders a config byte-identical to before the keys
    existed (and its launcher falls back to the historical kernels). A key this launcher cannot act on
    raises instead of vanishing: it is in the hashed ``hparams``, so ignoring it would make a new job
    that trains exactly like the old one -- the "knob that does nothing" trap.
    ``rendered_elsewhere``: keys the template already renders itself (base Moshi's ``prefetch_batches``)."""
    unsupported = sorted(
        k for k in TRAIN_PERF_KEYS if k not in supported and k not in rendered_elsewhere and hp.get(k) is not None
    )
    if unsupported:
        raise ValueError(f"{who}: training-speed key(s) {unsupported} do not apply to this launcher")
    out = ""
    for k, typ in TRAIN_PERF_KEYS.items():
        if k in supported and hp.get(k) is not None:
            out += f"{k}: {str(bool(hp[k])).lower()}\n" if typ is bool else f"{k}: {int(hp[k])}\n"
    return out


def _render_personaplex_lora_config(
    job: "SpeechFinetune", batch_size: int, max_steps: int, *, hf_repo_id: str, train_scope: str = "lora"
) -> str:
    """Render a PersonaPlex LoRA config (``moshi_family.personaplex.finetune_launcher``, train_scope=lora).

    ``train_scope="full"`` (``PERSONAPLEX_LIB_FULL_ADAPTER``, 2026-09-23): the same template for a FULL
    finetune -- no ``lora_*`` keys, and no ``fused_lora_add`` (refused). On more than one GPU the launcher
    shards it with FSDP2 and fp32 master weights. The LoRA output is byte-identical to before the kwarg.

    Unlike :func:`_render_personaplex_config` (which predates per-arm knobs and hardcodes everything),
    every training knob here comes from ``job.hparams``, with defaults equal to the base-Moshi A-arms
    so a PersonaPlex arm and its Moshi counterpart differ only where declared. Emits ONLY keys the
    launcher reads -- ``report_unread_config`` fails the run on anything else. The text-stream keys
    (``emit_epad``, ``onset_floor``, ``text_lead_frames``) render as ``null`` when unset, so the
    launcher falls back to ``interleave_text``'s defaults.

    ``system_prompt_key`` / ``voice_codes_key`` name the per-row prompt columns
    (``AttachPersonaPrompts`` writes ``context`` and ``voice_codes``)."""
    hp = getattr(job, "hparams", None) or {}
    _lr = hp.get("lr", 1e-6)

    def _opt_bool(k):
        return "null" if hp.get(k) is None else str(bool(hp[k])).lower()

    _tlf = "null" if hp.get("text_lead_frames") is None else int(hp["text_lead_frames"])
    # Training-speed switches (PERSONAPLEX_FAST_HPARAMS), rendered
    # ONLY when the arm sets them: an arm without them gets a config byte-identical to before they
    # existed, and the launcher's defaults are the historical kernels and batching.
    assert train_scope in ("lora", "full"), train_scope
    full = train_scope == "full"
    who = "PersonaPlex full FT" if full else "PersonaPlex LoRA"
    perf = render_train_perf_keys(hp, [k for k in TRAIN_PERF_KEYS if not (full and k == "fused_lora_add")], who=who)
    # EPAD's own text-loss weight (2026-09-23, ablation ab_epad1): rendered ONLY when set, so every
    # other arm's config is byte-identical to before the key existed (EPAD weighted as PAD).
    if hp.get("text_epad_weight") is not None:
        if not hp.get("emit_epad"):
            raise ValueError(f"{who}: text_epad_weight needs emit_epad=True")
        perf = f"text_epad_weight: {_yaml_float(hp['text_epad_weight'])}\n" + perf
    # Loss-weight overrides (2026-09-25, ablations ab2_pad05 / ab2_usersem): rendered ONLY when set, like
    # text_epad_weight; absent = PAD at the paper's 0.3 and the user stream's semantic codebook weighted as acoustic.
    if hp.get("text_pad_weight") is not None:
        perf = f"text_pad_weight: {_yaml_float(hp['text_pad_weight'])}\n" + perf
    if hp.get("user_semantic_weight") is not None:
        perf = f"user_semantic_weight: {_yaml_float(hp['user_semantic_weight'])}\n" + perf
    # The user-text head (2026-10-12, ablation ab2_usertext; ``train_data_common.user_text_loss``): its loss weight, and
    # optionally its target delay (frames) and LR. Rendered ONLY when set (absent = off, every earlier config
    # byte-identical); the launcher refuses the delay / LR without the weight.
    if hp.get("user_text_loss_weight") is not None:
        if float(hp["user_text_loss_weight"]) <= 0:
            raise ValueError(f"{who}: user_text_loss_weight must be > 0, got {hp['user_text_loss_weight']!r}")
        if full or not hp.get("swap_channels"):
            raise ValueError(f"{who}: user_text_loss_weight needs LoRA and a two-sided (swap_channels) codes corpus")
        ut = f"user_text_loss_weight: {_yaml_float(hp['user_text_loss_weight'])}\n"
        if hp.get("user_text_delay_frames") is not None:
            ut += f"user_text_delay_frames: {int(hp['user_text_delay_frames'])}\n"
        if hp.get("user_text_lr") is not None:
            ut += f"user_text_lr: {_yaml_float(hp['user_text_lr'])}\n"
        if hp.get("user_text_head_warmup_steps"):
            ut += f"user_text_head_warmup_steps: {int(hp['user_text_head_warmup_steps'])}\n"
        # the head's input layer (2026-10-13, ablation ab2_usertext_kn_w02_mid): rendered ONLY when not the default
        # "last" (= text_linear's input); an int = the residual stream after transformer layer k (0-based)
        if hp.get("user_text_layer") not in (None, "last"):
            ut += f"user_text_layer: {int(hp['user_text_layer'])}\n"
        # the head's loss (2026-10-13, ablation ab2_usertext_kn_w02_ctc): rendered ONLY when not the default "ce"
        # (framewise CE); "ctc" = CTC within each user turn span + PAD CE outside (train_data_common.user_text_ctc_loss)
        if hp.get("user_text_loss_kind") not in (None, "ce"):
            if hp["user_text_loss_kind"] != "ctc":
                raise ValueError(f"{who}: user_text_loss_kind must be 'ce' or 'ctc', got {hp['user_text_loss_kind']!r}")
            ut += f'user_text_loss_kind: "{hp["user_text_loss_kind"]}"\n'
        # the knowledge items' user words too (2026-10-12; rendered ONLY when set: every earlier config byte-identical)
        if hp.get("knowledge_user_text"):
            if not hp.get("knowledge_data"):
                raise ValueError(f"{who}: knowledge_user_text needs knowledge_data")
            ut += "knowledge_user_text: true\n"
        # ... fed back as an INPUT stream (2026-10-18; moshi_family.user_text_stream): rendered ONLY when set (absent =
        # off, every earlier config byte-identical). The stream feeds the CE head's argmax: CE, last layer only.
        if hp.get("user_text_input_stream"):
            if hp.get("user_text_loss_kind") not in (None, "ce") or hp.get("user_text_layer") not in (None, "last"):
                raise ValueError(f"{who}: user_text_input_stream needs the CE head on the last layer")
            if hp.get("user_text_input_lr") is None:
                raise ValueError(f"{who}: user_text_input_stream needs user_text_input_lr")
            if hp.get("knowledge_data") and not hp.get("knowledge_user_text"):
                raise ValueError(f"{who}: user_text_input_stream with knowledge_data needs knowledge_user_text")
            ut += f"user_text_input_stream: true\nuser_text_input_lr: {_yaml_float(hp['user_text_input_lr'])}\n"
        elif hp.get("user_text_input_lr") is not None:
            raise ValueError(f"{who}: user_text_input_lr without user_text_input_stream")
        perf = ut + perf
    elif any(
        hp.get(k)
        for k in (
            "user_text_delay_frames",
            "user_text_lr",
            "user_text_head_warmup_steps",
            "knowledge_user_text",
            "user_text_layer",
            "user_text_input_stream",
            "user_text_input_lr",
        )
    ) or hp.get("user_text_loss_kind") not in (None, "ce"):
        raise ValueError(
            f"{who}: user_text_delay_frames / user_text_lr / user_text_head_warmup_steps / knowledge_user_text without "
            "user_text_loss_weight"
        )
    # AdamW decoupled weight decay (2026-09-25, ablation ab2_wd): rendered ONLY when set; absent = 0.0 in the
    # launcher, so every earlier config is byte-identical.
    if hp.get("weight_decay") is not None:
        perf = f"weight_decay: {_yaml_float(hp['weight_decay'])}\n" + perf
    # The LR schedule of every param group (2026-10-13, ablation ab2_constlr; train_loop.lr_scale): rendered ONLY when
    # not the default cosine, so every earlier config is byte-identical.
    if hp.get("lr_schedule") not in (None, "cosine"):
        if hp["lr_schedule"] != "constant":
            raise ValueError(f"{who}: lr_schedule must be 'cosine' or 'constant', got {hp['lr_schedule']!r}")
        perf = f'lr_schedule: "{hp["lr_schedule"]}"\n' + perf
    # Assistant-side loss down-weight on long turns (2026-09-25, ablation ab2_turntail03): turn_tail_weight on the
    # assistant's text + audio targets more than turn_tail_s (default 10) into a continuous assistant turn. Rendered
    # ONLY when set (absent = off, every earlier config byte-identical); turn_tail_s always with it, turn_tail_gap_s
    # only when set (absent = the launcher's 1.0 s).
    if hp.get("turn_tail_weight") is not None:
        if hp.get("floor_weight") is not None or hp.get("knowledge_data") is not None:
            raise ValueError(f"{who}: turn_tail_weight goes with neither floor_weight nor knowledge_data")
        gap = hp.get("turn_tail_gap_s")
        gap = "" if gap is None else f"turn_tail_gap_s: {_yaml_float(gap)}\n"
        perf = (
            f"turn_tail_weight: {_yaml_float(hp['turn_tail_weight'])}\n"
            f"turn_tail_s: {_yaml_float(hp.get('turn_tail_s', 10.0))}\n" + gap + perf
        )
    elif hp.get("turn_tail_s") is not None or hp.get("turn_tail_gap_s") is not None:
        raise ValueError(f"{who}: turn_tail_s / turn_tail_gap_s without turn_tail_weight")
    # Context weight at floor-available frames (2026-09-24, ablation ab_epad1_floor): rendered ONLY when set,
    # like text_epad_weight. The launcher asserts onset_floor and no text lead.
    if hp.get("floor_weight") is not None:
        if not hp.get("onset_floor") or hp.get("text_lead_frames"):
            raise ValueError(f"{who}: floor_weight needs onset_floor=True and no text_lead_frames")
        perf = (
            f"floor_weight: {_yaml_float(hp['floor_weight'])}\n"
            f"floor_quiet_ms: {_yaml_float(hp.get('floor_quiet_ms', 200.0))}\n" + perf
        )
    # Synthetic knowledge dialogues at a stratified in-batch share (2026-09-25, ablation ab_epad1_know30): rendered
    # ONLY when set, like text_epad_weight. ``knowledge_data`` is a ``KnowledgeDialogueCodes`` output (a tk.Path in
    # hparams, so the run depends on it); the launcher refuses it together with floor_weight / length bucketing.
    if hp.get("knowledge_data") is not None:
        share = float(hp["knowledge_share"])
        # 1.0 = knowledge items only (ab2_knowonly_3p, 2026-09-29); the launcher refuses 1.0 together with aux_data.
        if not 0.0 < share <= 1.0:
            raise ValueError(f"{who}: knowledge_share must be in (0, 1], got {share}")
        if hp.get("floor_weight") is not None or hp.get("length_bucket_pool"):
            raise ValueError(f"{who}: the knowledge mix has no floor weight and no length bucketing")
        kd = hp["knowledge_data"]
        kd = kd.get() if hasattr(kd, "get") else str(kd)
        concat = ""
        # ... each knowledge item = 2-4 dialogues back to back (2026-09-25, ablation ab_epad1_know30_concat): rendered
        # ONLY when set. ``knowledge_meta`` = ``KnowledgeConcatMeta.out_meta`` (a tk.Path: the run depends on it),
        # ``knowledge_concat`` = the scheme's knobs (``moshi_family.knowledge_concat.ConcatCfg``), as one JSON line.
        if (hp.get("knowledge_meta") is None) != (hp.get("knowledge_concat") is None):
            raise ValueError(f"{who}: knowledge_meta and knowledge_concat go together")
        if hp.get("knowledge_meta") is not None:
            km = hp["knowledge_meta"]
            km = km.get() if hasattr(km, "get") else str(km)
            concat = f'knowledge_meta: "{km}"\nknowledge_concat: {json.dumps(dict(hp["knowledge_concat"]), sort_keys=True)}\n'
        # ... and the knowledge rows' user stream from their stored augmented variants (2026-09-28, knowledge corpus v7):
        # rendered ONLY when set, so every earlier config is byte-identical.
        if hp.get("knowledge_user_augment"):
            concat += "knowledge_user_augment: true\n"
        # ... and only these knowledge-corpus rows (2026-09-29, ablation ab2_know_known: ``KnowledgeKnownScore.out_rows``, a
        # tk.Path: the run depends on it): rendered ONLY when set, so every earlier config is byte-identical.
        if hp.get("knowledge_train_rows") is not None:
            kr = hp["knowledge_train_rows"]
            kr = kr.get() if hasattr(kr, "get") else str(kr)
            concat += f'knowledge_train_rows: "{kr}"\n'
        perf = f'knowledge_data: "{kd}"\nknowledge_share: {_yaml_float(share)}\n' + concat + perf
    elif hp.get("knowledge_user_augment"):
        raise ValueError(f"{who}: knowledge_user_augment without knowledge_data")
    elif hp.get("knowledge_train_rows") is not None:
        raise ValueError(f"{who}: knowledge_train_rows without knowledge_data")
    elif hp.get("knowledge_share") is not None or hp.get("knowledge_meta") is not None:
        raise ValueError(f"{who}: knowledge_share / knowledge_meta without knowledge_data")
    # A second conversational corpus in the knowledge mix at its own stratified share (2026-09-28, ablation
    # ab2_mix_jre_oy_know: JRE + Open Yap + knowledge; ``train_data.build_knowledge_mix_loader(aux_path=)``). ``aux_data`` /
    # ``aux_train_rows`` are tk.Paths in hparams (the run depends on them). Rendered ONLY when set: every earlier config is
    # byte-identical.
    if hp.get("aux_data") is not None:
        if hp.get("knowledge_data") is None:
            raise ValueError(f"{who}: aux_data rides on the knowledge mix: it needs knowledge_data")
        aux_share = float(hp["aux_share"])
        if not (0.0 < aux_share and aux_share + float(hp["knowledge_share"]) < 1.0):
            raise ValueError(f"{who}: aux_share must be > 0 with knowledge_share + aux_share < 1, got {aux_share}")
        ad = hp["aux_data"]
        ad = ad.get() if hasattr(ad, "get") else str(ad)
        aux = f'aux_data: "{ad}"\naux_share: {_yaml_float(aux_share)}\n'
        if hp.get("aux_train_rows") is not None:
            ar = hp["aux_train_rows"]
            ar = ar.get() if hasattr(ar, "get") else str(ar)
            aux += f'aux_train_rows: "{ar}"\n'
        # ... and the aux corpus's own row weights / flagged-word spans (POLICY_2026_09_29; tk.Paths, the run depends on
        # them): each exactly when the primary has the matching knob (the loader refuses a mismatch; refused here first,
        # at render time). Rendered ONLY when set.
        for key, primary in (
            ("aux_train_row_weights", "train_row_weights"),
            ("aux_train_word_spans", "train_word_spans"),
        ):
            if (hp.get(key) is None) != (hp.get(primary) is None):
                raise ValueError(f"{who}: {key} and {primary} go together (each corpus of the mix needs its own)")
            if hp.get(key) is not None:
                v = hp[key]
                v = v.get() if hasattr(v, "get") else str(v)
                aux += f'{key}: "{v}"\n'
        perf = aux + perf
    elif any(
        hp.get(k) is not None for k in ("aux_share", "aux_train_rows", "aux_train_row_weights", "aux_train_word_spans")
    ):
        raise ValueError(
            f"{who}: aux_share / aux_train_rows / aux_train_row_weights / aux_train_word_spans without aux_data"
        )
    # A row subset of train_data (2026-09-28, the episode-edge filter; ``persona_prompts.EpisodeEdgeRows.out_rows``, a
    # tk.Path in hparams, so the run depends on it). Rendered ONLY when set: every earlier config is byte-identical.
    if hp.get("train_rows") is not None:
        tr = hp["train_rows"]
        tr = tr.get() if hasattr(tr, "get") else str(tr)
        perf = f'train_rows: "{tr}"\n' + perf
    # Per-row loss weights (2026-09-28, the monologue down-weight; ``row_filters.ThresholdRowWeights.out_weights``, a
    # tk.Path in hparams, so the run depends on it). Rendered ONLY when set: every earlier config is byte-identical.
    if hp.get("train_row_weights") is not None:
        if hp.get("floor_weight") is not None or hp.get("turn_tail_weight") is not None:
            raise ValueError(f"{who}: train_row_weights goes with neither floor_weight nor turn_tail_weight")
        rw = hp["train_row_weights"]
        rw = rw.get() if hasattr(rw, "get") else str(rw)
        perf = f'train_row_weights: "{rw}"\n' + perf
    # Flagged-word loss weights / the censored-assistant sidecar (2026-09-28b, ablations ab2_wordmask / ab2_wordw01 /
    # ab2_beep; ``row_filters.FlaggedWordSpans.out_spans`` / ``CensorFlaggedWords.out_dir``, tk.Paths in hparams, so the
    # run depends on them). Rendered ONLY when set: every earlier config is byte-identical.
    if (hp.get("train_word_spans") is None) != (hp.get("train_word_weight") is None):
        raise ValueError(f"{who}: train_word_spans and train_word_weight go together")
    if hp.get("train_word_spans") is not None or hp.get("train_censor") is not None:
        if hp.get("floor_weight") is not None or hp.get("turn_tail_weight") is not None:
            raise ValueError(
                f"{who}: train_word_spans / train_censor go with neither floor_weight nor turn_tail_weight"
            )
        if hp.get("aux_data") is not None and hp.get("train_censor") is not None:
            raise ValueError(f"{who}: train_censor is not implemented with aux_data")
    if hp.get("train_word_spans") is not None:
        ww = float(hp["train_word_weight"])
        if not 0.0 <= ww < 1.0:
            raise ValueError(f"{who}: train_word_weight must be in [0, 1), got {ww}")
        ws = hp["train_word_spans"]
        ws = ws.get() if hasattr(ws, "get") else str(ws)
        pad = ""
        # ... and each flagged word's AUDIO span widened by this many frames on both sides (POLICY_2026_09_29:
        # e.g. 1 = one frame in each direction). Rendered ONLY when set (> 0).
        if hp.get("train_word_pad_frames"):
            pf = hp["train_word_pad_frames"]
            if isinstance(pf, bool) or int(pf) != pf or int(pf) < 0:
                raise ValueError(f"{who}: train_word_pad_frames must be a non-negative int, got {pf!r}")
            pad = f"train_word_pad_frames: {int(pf)}\n"
        perf = f'train_word_spans: "{ws}"\ntrain_word_weight: {_yaml_float(ww)}\n' + pad + perf
    elif hp.get("train_word_pad_frames"):
        raise ValueError(f"{who}: train_word_pad_frames needs train_word_spans")
    if hp.get("train_censor") is not None:
        tc = hp["train_censor"]
        tc = tc.get() if hasattr(tc, "get") else str(tc)
        perf = f'train_censor: "{tc}"\n' + perf
    # Held-out validation loss (2026-09-25, p6): rendered ONLY when set, like text_epad_weight. ``job.eval_data`` (the
    # SpeechFinetune constructor argument, so the run depends on it; never an hparam, check_hparams_destination) is a
    # fixed ``heldout_set`` window set (e.g. ``PersonaHeldoutSet`` "heldout"), ``eval_every`` the cadence in
    # optimizer steps (the launcher also evaluates step 0).
    _eval_data = getattr(job, "eval_data", None)
    if _eval_data is not None:
        every = int(hp.get("eval_every") or 0)
        if every <= 0:
            raise ValueError(f"{who}: eval_data needs hparams eval_every > 0")
        ed = _eval_data.get() if hasattr(_eval_data, "get") else str(_eval_data)
        # The cheap cadence (2026-09-25): a fixed ``eval_subset``-window subset every ``eval_every`` steps, the whole
        # set every ``eval_full_every``; rendered ONLY when set (p6's first config stays byte-identical without it).
        sub = ""
        if (hp.get("eval_subset") is None) != (hp.get("eval_full_every") is None):
            raise ValueError(f"{who}: eval_subset and eval_full_every go together")
        if hp.get("eval_subset") is not None:
            n_sub, full_every = int(hp["eval_subset"]), int(hp["eval_full_every"])
            if n_sub <= 0 or full_every <= 0 or full_every % every:
                raise ValueError(f"{who}: eval_subset > 0 and eval_full_every a multiple of eval_every ({every})")
            sub = f"eval_subset: {n_sub}\neval_full_every: {full_every}\n"
        perf = f'eval_data: "{ed}"\neval_every: {every}\n' + sub + perf
    elif hp.get("eval_every") is not None or hp.get("eval_subset") is not None or hp.get("eval_full_every") is not None:
        raise ValueError(f"{who}: eval_every / eval_subset / eval_full_every without eval_data")
    scope_block = (
        'train_scope: "full"\n' if full else f'train_scope: "lora"\nlora_rank: {job.lora_rank}\nlora_scaling: 2.0\n'
    )
    return (
        f"""# PersonaPlex {"full" if full else "LoRA"} finetune config (moshi_family.personaplex.finetune_launcher, train_scope={train_scope})
hf_repo_id: "{hf_repo_id}"
{_train_data_yaml(job, supports_mix=False)}
out_dir: "{job.out_rundir.get()}"
max_steps: {max_steps}
duration_sec: {job.duration_sec}
{scope_block}gradient_checkpointing: {str(bool(hp.get("gradient_checkpointing", True))).lower()}
per_gpu_batch: {hp.get("per_gpu_batch", 1)}
grad_accum: {hp.get("grad_accum", 4)}
lr_temporal: {_yaml_float(hp.get("temporal_lr", _lr))}
lr_depformer: {_yaml_float(hp.get("depth_lr", 2 * _lr))}
warmup_steps: {hp.get("warmup_steps", 500)}
grad_clip: {_yaml_float(hp.get("grad_clip", 1.0))}
save_every: {hp.get("save_every", 100)}
log_every: 10
seed: {getattr(job, "seed", 0)}
system_prompt_key: "{hp.get("system_prompt_key", "context")}"
voice_codes_key: "{hp.get("voice_codes_key") or ""}"
silence_frames: {int(hp.get("silence_frames", 6))}
swap_channels: {"true" if hp.get("swap_channels") else "false"}
stored_user_augment: {"true" if hp.get("stored_user_augment") else "false"}
emit_epad: {_opt_bool("emit_epad")}
onset_floor: {_opt_bool("onset_floor")}
text_lead_frames: {_tlf}
word_overlap: {"null" if hp.get("word_overlap") is None else chr(34) + hp["word_overlap"] + chr(34)}
loss_norm: "{hp.get("loss_norm", "weight")}"
"""
        + perf
        + render_launcher_fix_keys(
            hp, ("resume_data_position", "fp32_master_weights"), who=who, rendered_elsewhere=("loss_norm",)
        )
    )


#: Training-speed preset for FUTURE PersonaPlex LoRA arms (2026-09-23).
#: Merge it into an arm's ``hparams`` -- that is a new hash, i.e. a new arm, so
#: it is never applied to a running or finished one (p2/p3 do not use it). Every entry computes the same
#: training step as without it, to bf16 rounding (guard ``check_train_perf.py``); prefetching yields the
#: same batches bit for bit. Not in here, because they change the EXPERIMENT and not just its speed:
#: ``length_bucket_pool`` (batch composition / data order), ``per_gpu_batch`` and ``gradient_checkpointing``.
PERSONAPLEX_FAST_HPARAMS = {
    "contiguous_step_inputs": True,
    "fused_causal_attention": True,
    "fused_lora_add": True,
    "depformer_skip_masked": True,
    "prefetch_batches": 2,
}

#: A PersonaPlex FULL finetune (2026-09-23, p5): the same switches minus the LoRA one.
PERSONAPLEX_FULL_FAST_HPARAMS = {k: v for k, v in PERSONAPLEX_FAST_HPARAMS.items() if k != "fused_lora_add"}

#: The same for every other family (2026-09-23), each restricted to what its launcher can act on
#: (``moshi_family.train_perf`` has the table). New arms get these from policy epoch POLICY_2026_09_23
#: (``runs.RULE_FAST_TRAIN`` / ``runs.fast_train_hparams``); never merge them into an existing arm.
#: base Moshi: prefetch is already on (``prefetch_batches`` renders 2 for every arm).
MOSHI_FAST_HPARAMS = {
    "contiguous_step_inputs": True,
    "fused_causal_attention": True,
    "fused_lora_add": True,
    "depformer_skip_masked": True,
}
#: A full finetune has no LoRA to fuse.
MOSHI_FULL_FAST_HPARAMS = {k: v for k, v in MOSHI_FAST_HPARAMS.items() if k != "fused_lora_add"}
#: batch 1 with an all-ones loss mask: no depformer row to skip; no prefetch (the batch builder runs the
#: model's ARC conditioner on the GPU).
MOSHIRAG_FAST_HPARAMS = {"contiguous_step_inputs": True, "fused_causal_attention": True, "fused_lora_add": True}
#: HF trunk (no fused causal attention); stage0 has no LoRA.
AUDEX_STAGE0_FAST_TRAIN_PERF = {"contiguous_step_inputs": True, "depformer_skip_masked": True, "prefetch_batches": 2}
AUDEX_STAGE1_FAST_TRAIN_PERF = {**AUDEX_STAGE0_FAST_TRAIN_PERF, "fused_lora_add": True}
#: depformer_skip_masked = text-only log-prob forwards (the GRPO loss reads text only); no loader.
RL_FAST_TRAIN_PERF = {
    "contiguous_step_inputs": True,
    "fused_causal_attention": True,
    "fused_lora_add": True,
    "depformer_skip_masked": True,
}
#: family -> preset. SpeechFinetune families are keyed by adapter name; Audex by stage.
FAST_TRAIN_PRESETS = {
    "moshi_lib": MOSHI_FAST_HPARAMS,
    "moshi_lib_full": MOSHI_FULL_FAST_HPARAMS,
    "moshirag_lib": MOSHIRAG_FAST_HPARAMS,
    "personaplex_lib_lora": PERSONAPLEX_FAST_HPARAMS,
    "personaplex_lib_full": PERSONAPLEX_FULL_FAST_HPARAMS,
    "audex_stage0": AUDEX_STAGE0_FAST_TRAIN_PERF,
    "audex_stage1": AUDEX_STAGE1_FAST_TRAIN_PERF,
    "rl_grpo": RL_FAST_TRAIN_PERF,
}

#: Launcher setup fixes from the 2026-09-23 code-sharing review (code-map §9), per family. Each launcher
#: reads its keys with a default that reproduces the old behaviour, and the renderers emit them ONLY
#: when set, so an existing arm renders the same config and keeps its hash. New arms get them from
#: policy epoch POLICY_2026_09_23 (``runs.RULE_LAUNCHER_FIXES`` / ``runs.launcher_fix_hparams``); never
#: merge them into an existing arm. Base Moshi has every piece already (fp32 masters from
#: RULE_FP32_MASTER_WEIGHTS, resume unconditionally since 2026-09-16) except one:
#:  * ``ddp_static_graph=False`` -- DDP(static_graph=True) never all-reduces a module returning the
#:    LMOutput dataclass, so every multi-GPU base-Moshi LoRA arm trained unsynchronised replicas
#:    (found 2026-09-23; a no-op on one GPU, and absent from full finetunes, which use FSDP);
#:  * ``fp32_master_weights`` -- AdamW on fp32 masters of the bf16 adapter (not for Audex stage0, which
#:    trains the embeddings + depformer outright, like a full finetune);
#:  * ``resume_data_position`` -- a preempted run continues its data stream instead of restarting it;
#:  * ``loss_norm="weight"`` -- MoshiRAG's loss at the paper's eq. 7 denominator, like every other family;
#:  * ``text_pad_weight_in_full_stage1`` -- Audex's replay/curriculum branch honours ``text_pad_weight``.
LAUNCHER_FIX_PRESETS = {
    "moshi_lib": {"ddp_static_graph": False},
    "moshi_lib_full": {},
    "moshirag_lib": {"fp32_master_weights": True, "loss_norm": "weight", "resume_data_position": True},
    "personaplex_lib_lora": {"resume_data_position": True},
    "personaplex_lib_full": {"resume_data_position": True},
    "audex_stage0": {"resume_data_position": True},
    "audex_stage1": {
        "fp32_master_weights": True,
        "resume_data_position": True,
        "text_pad_weight_in_full_stage1": True,
    },
    "rl_grpo": {"fp32_master_weights": True},
}

#: Every launcher-fix key and its YAML type, in render order (see :data:`LAUNCHER_FIX_PRESETS`).
LAUNCHER_FIX_KEYS = {
    "fp32_master_weights": bool,
    "resume_data_position": bool,
    "loss_norm": str,
    "text_pad_weight_in_full_stage1": bool,
    "ddp_static_graph": bool,
    "eval_fixed_batches": bool,
}


def render_launcher_fix_keys(hp: dict, supported, *, who: str, rendered_elsewhere=()) -> str:
    """YAML lines for the launcher-fix keys ``hp`` sets, in :data:`LAUNCHER_FIX_KEYS` order; "" when none.

    Rendered ONLY when set, as :func:`render_train_perf_keys`: an arm without them renders a config
    byte-identical to before the keys existed. A key this launcher does not read raises instead of
    vanishing (a hashed knob that does nothing). ``rendered_elsewhere``: keys the template already
    renders itself (PersonaPlex LoRA's ``loss_norm``)."""
    hp = hp or {}
    unsupported = sorted(
        k for k in LAUNCHER_FIX_KEYS if k not in supported and k not in rendered_elsewhere and hp.get(k) is not None
    )
    if unsupported:
        raise ValueError(f"{who}: launcher-fix key(s) {unsupported} do not apply to this launcher")
    out = ""
    for k, typ in LAUNCHER_FIX_KEYS.items():
        if k in supported and hp.get(k) is not None:
            if typ is bool:
                out += f"{k}: {str(bool(hp[k])).lower()}\n"
            else:
                out += f'{k}: "{hp[k]}"\n'
    return out


# PersonaPlex LoRA on the owned lib: the SAME uniform LoRA as base Moshi (moshi_family.build.apply_lora_to)
# and the base-Moshi checkpoint layout, so ResolveOverlayCheckpoint(overlay_kind="lora") resolves it and
# `personaplex_family_backend_spec(lora_rank=R)` re-wraps the layers at inference. New ``name`` -> the
# existing heads-only arms keep their hashes.
PERSONAPLEX_LIB_LORA_ADAPTER = FinetuneAdapter(
    name="personaplex_lib_lora",
    batch_size=32,
    overlay_kind="lora",
    base_model="nvidia/personaplex-7b-v1",
    weights_name="lora.safetensors",
    render_config=partial(_render_personaplex_lora_config, hf_repo_id="nvidia/personaplex-7b-v1"),
    launcher_module="moshi_family.personaplex.finetune_launcher",
    pythonpath_package="moshi_family",
)


# PersonaPlex FULL finetune on the owned lib (2026-09-23, p5; the paper's own recipe): every parameter
# trained, FSDP2 with fp32 master weights on more than one GPU (moshi_family.fsdp_full_ft). Checkpoints in
# the base-Moshi full-FT layout ``checkpoints/checkpoint_<step>/consolidated/model.safetensors`` (bf16), so
# ResolveOverlayCheckpoint(overlay_kind="full") resolves them and the PersonaPlex engine overlays the whole
# state dict with no LoRA. A new ``name`` -> no existing arm is touched.
PERSONAPLEX_LIB_FULL_ADAPTER = FinetuneAdapter(
    name="personaplex_lib_full",
    batch_size=32,
    overlay_kind="full",
    base_model="nvidia/personaplex-7b-v1",
    weights_name="model.safetensors",
    render_config=partial(_render_personaplex_lora_config, hf_repo_id="nvidia/personaplex-7b-v1", train_scope="full"),
    launcher_module="moshi_family.personaplex.finetune_launcher",
    pythonpath_package="moshi_family",
)


def _single_train_data_path(job: "SpeechFinetune") -> str:
    """The one training corpus path, asserting there is exactly one.

    For the moshi-finetune fork's YAML schema, which has no mixed-corpus key at all. Taking
    ``specs[0]`` silently would train on half the intended data.
    """
    specs = train_data_specs(job.train_data)
    assert len(specs) == 1, (
        f"the moshi-finetune fork schema has no mixed-corpus key, but train_data is a "
        f"{len(specs)}-corpus mix -- use MOSHI_LIB_ADAPTER for mixed corpora"
    )
    return specs[0][0].get()


def _train_data_yaml(job: "SpeechFinetune", *, supports_mix: bool) -> str:
    """Render the training-corpus YAML for a launcher config -- the ONE place that does this.

    Emits the plain ``train_data:`` scalar for a single corpus (so single-corpus configs stay
    byte-identical to those written before mixing existed) and a ``train_data_mix:`` list of
    ``{path, weight}`` rows otherwise.

    ``supports_mix`` says whether this adapter's *launcher* actually reads ``train_data_mix``.
    Only the base-Moshi lib launcher does; the others do ``cfg["train_data"]`` and would die with a
    KeyError once the GPU job is already running. Rendering happens in a login-node mini_task, so
    refusing here turns a wasted allocation into an instant, explanatory failure.
    """
    specs = train_data_specs(job.train_data)
    if len(specs) == 1:
        p, w, window_sec = specs[0]
        line = f'train_data: "{p.get()}"'
        if window_sec:  # single full-conversation corpus -> loader windows it (top-level key)
            line += f"\nwindow_sec: {window_sec}"
        return line
    assert supports_mix, (
        f"train_data is a {len(specs)}-corpus mix, but this architecture's launcher only reads a "
        f"single `train_data` key -- it would fail at runtime. Use MOSHI_LIB_ADAPTER for mixed "
        f"corpora, or teach this launcher to read `train_data_mix` (see moshi_finetune_launcher)."
    )
    rows = "\n".join(
        f'  - {{path: "{p.get()}", weight: {w}' + (f", window_sec: {ws}" if ws else "") + "}" for p, w, ws in specs
    )
    return f"train_data_mix:\n{rows}"


def _yaml_float(x) -> str:
    """Format a number so YAML parses it as a float, not a string.

    YAML 1.1 only recognises a float in exponent form when it has both a decimal point and a signed
    exponent, so ``1e-06`` loads as the *string* ``"1e-06"`` while ``1.0e-06`` loads as a float. Our
    launcher wraps these in ``float()`` so the difference has been harmless, but anything reading the
    config without that coercion would silently get a string.
    """
    return f"{float(x):.10e}"


def _render_moshi_lib_config(job: "SpeechFinetune", batch_size: int, max_steps: int, *, hf_repo_id: str) -> str:
    """Render the base-Moshi LoRA config (consumed by ``moshi_family.moshi_finetune_launcher``, NOT
    the moshi-finetune fork schema). LoRA over the whole trunk via the shared ``train_loop``; the
    fork's loss weighting is replicated in ``moshi_train_data.moshi_loss``. SINGLE-GPU: per-gpu batch
    1 x grad_accum 16 = effective batch 16 (matching the fork's ``batch_size=16``); gradient
    checkpointing on so LoRA-over-backbone fits one 24 GB GPU. lr 2e-6 = the fork's ``optim.lr``.

    Extra knobs (lr, sample_every, eval_batches, general_eval_data, knowledge_probe_*) come from
    ``job.hparams``; ``knowledge_probe_data`` (the held-out probe set) is a first-class job arg."""
    hp = getattr(job, "hparams", None) or {}
    _gen = hp.get("general_eval_data")
    _gen = _gen.get() if hasattr(_gen, "get") else (_gen or "")
    _data = _train_data_yaml(job, supports_mix=True)  # moshi_finetune_launcher reads train_data_mix
    _lr = hp.get("lr", 2e-6)
    return (
        f"""# base-Moshi LoRA finetune config (moshi_finetune_launcher schema)
hf_repo_id: "{hf_repo_id}"
{_data}
out_dir: "{job.out_rundir.get()}"
max_steps: {max_steps}
duration_sec: {job.duration_sec}
audio_jitter_sec: {getattr(job, "audio_jitter_sec", 0.0)}
emit_epad: {str(bool(hp.get("emit_epad", False))).lower()}
clamp_text_to_speech: {str(bool(hp.get("clamp_text_to_speech", False))).lower()}
onset_floor: {str(bool(hp.get("onset_floor", False))).lower()}
text_lead_frames: {int(hp.get("text_lead_frames", 0))}
word_overlap: "{hp.get("word_overlap", "overwrite")}"
aug_gain_prob: {float(hp.get("aug_gain_prob", 0.0))}
stored_user_augment: {"true" if hp.get("stored_user_augment") else "false"}
aug_noise_prob: {float(hp.get("aug_noise_prob", 0.0))}
aug_echo_reverb_prob: {float(hp.get("aug_echo_reverb_prob", 0.0))}
aug_clip_peak: {hp.get("aug_clip_peak", "null")}
token_corrupt_audio_prob: {float(hp.get("token_corrupt_audio_prob", 0.0))}
text_pad_target_share: {hp.get("text_pad_target_share", "null")}
pack_to_window: {str(bool(hp.get("pack_to_window", False))).lower()}
pack_gap_min_sec: {float(hp.get("pack_gap_min_sec", 0.5))}
pack_gap_max_sec: {float(hp.get("pack_gap_max_sec", 3.0))}
lora_rank: {job.lora_rank}
lora_scaling: 2.0
full_finetuning: {str(job.adapter.overlay_kind == "full").lower()}
fp32_master_weights: {"true" if hp.get("fp32_master_weights") else "false"}
per_gpu_batch: {hp.get("per_gpu_batch", 1)}
grad_accum: {hp.get("grad_accum", 16)}
gradient_checkpointing: {str(bool(hp.get("gradient_checkpointing", True))).lower()}
prefetch_batches: {int(hp.get("prefetch_batches", 2))}
lr: {_yaml_float(_lr)}
depth_lr: {_yaml_float(hp.get("depth_lr", _lr))}
temporal_lr: {_yaml_float(hp.get("temporal_lr", _lr))}
audio_other_weight: {_yaml_float(hp.get("audio_other_weight", 0.01))}
text_pad_weight: {_yaml_float(hp.get("text_pad_weight", 0.5))}
text_loss_weight: {_yaml_float(hp.get("text_loss_weight", 1.0))}
loss_norm: {hp.get("loss_norm", "count")}
warmup_steps: {hp.get("warmup_steps", 200)}
grad_clip: {_yaml_float(hp.get("grad_clip", 1.0))}
kl_weight: {_yaml_float(hp.get("kl_weight", 0.0))}
kl_temperature: {_yaml_float(hp.get("kl_temperature", 1.0))}
l2sp_weight: {_yaml_float(hp.get("l2sp_weight", 0.0))}
text_lr_mult: {_yaml_float(hp.get("text_lr_mult", 1.0))}
save_every: {hp.get("save_every", 500)}
save_steps: {list(hp.get("save_steps", []))}
log_every: 10
seed: {getattr(job, "seed", 0)}
sample_every: {hp.get("sample_every", 100)}
eval_batches: {hp.get("eval_batches", 8)}
holdout_every: {hp.get("holdout_every", 0)}
general_eval_data: "{_gen}"
do_eval: {str(getattr(job, "eval_data", None) is not None).lower()}
eval_data: "{job.eval_data.get() if getattr(job, "eval_data", None) is not None else ""}"
eval_freq: {hp.get("eval_freq", 100)}
knowledge_probe_data: "{job.knowledge_probe_data.get() if getattr(job, "knowledge_probe_data", None) is not None else ""}"
knowledge_probe_every: {hp.get("knowledge_probe_every", 100)}
knowledge_probe_batch_size: {hp.get("knowledge_probe_batch_size", 4)}
knowledge_probe_capture_s: {_yaml_float(hp.get("knowledge_probe_capture_s", 20.0))}
knowledge_probe_n: {hp.get("knowledge_probe_n", 0)}
knowledge_probe_transcripts: {str(bool(hp.get("knowledge_probe_transcripts", True))).lower()}
knowledge_probe_audio_n: {hp.get("knowledge_probe_audio_n", 0)}
"""
        + render_train_perf_keys(
            hp,
            # a full finetune has no LoRA linears to fuse: refuse the key, as the PersonaPlex renderer does
            [k for k in MOSHI_LIB_PERF_KEYS if not (job.adapter.overlay_kind == "full" and k == "fused_lora_add")],
            who="base-Moshi lib",
            rendered_elsewhere=("prefetch_batches",),
        )
        + render_launcher_fix_keys(
            hp,
            ("ddp_static_graph", "eval_fixed_batches"),
            who="base-Moshi lib",
            rendered_elsewhere=("fp32_master_weights", "loss_norm"),
        )
    )


#: Training-speed keys the base-Moshi launcher reads (``prefetch_batches`` is rendered above, default 2
#: since before these switches existed; the loader has no length bucketing).
MOSHI_LIB_PERF_KEYS = ("contiguous_step_inputs", "fused_causal_attention", "fused_lora_add", "depformer_skip_masked")


# Base Moshi on the OWNED moshi_family lib (Phase 5): fork-free LoRA finetune of moshiko via
# ``moshi_family.moshi_finetune_launcher`` on the SHARED ``train_loop`` + uniform ``modules.lora``,
# in moshi_family_venv. New ``name`` -> fresh hash (intended fork->lib migration). Eval routes the
# saved adapter through ``moshi_family_backend_spec(lora_rank=...)`` (--overlay + --lora_rank/scaling).
MOSHI_LIB_ADAPTER = FinetuneAdapter(
    name="moshi_lib",
    batch_size=16,
    overlay_kind="lora",
    base_model="kyutai/moshiko-pytorch-bf16",
    weights_name="lora.safetensors",
    render_config=partial(_render_moshi_lib_config, hf_repo_id="kyutai/moshiko-pytorch-bf16"),
    launcher_module="moshi_family.moshi_finetune_launcher",
    pythonpath_package="moshi_family",
)


# Full-parameter finetune of the same base, through the SAME launcher and the same rendered config --
# the launcher branches on `full_finetuning`, which the renderer derives from `overlay_kind` above, so
# the adapter is the single place that decides and the two cannot drift apart. The checkpoint is a
# plain state dict over the base module names rather than a LoRA adapter, which is why the overlay
# kind and weights name differ; `overlay_state_dict` already applies a full state dict (its docstring
# names the case) and `moshi_family_backend_spec(lora_rank=None)` wraps no LoRA at inference, so the
# eval path needs no new branch either. Sharding is FSDP2 -- see moshi_family/fsdp_full_ft.py.
MOSHI_LIB_FULL_ADAPTER = FinetuneAdapter(
    name="moshi_lib_full",
    batch_size=16,
    overlay_kind="full",
    base_model="kyutai/moshiko-pytorch-bf16",
    weights_name="model.safetensors",
    render_config=partial(_render_moshi_lib_config, hf_repo_id="kyutai/moshiko-pytorch-bf16"),
    launcher_module="moshi_family.moshi_finetune_launcher",
    pythonpath_package="moshi_family",
)


def _render_moshirag_lib_config(job: "SpeechFinetune", batch_size: int, max_steps: int, *, hf_repo_id: str) -> str:
    """Render the MoshiRAG LoRA config (consumed by ``moshi_family.moshirag_finetune_launcher``). Same
    single-GPU LoRA recipe as base Moshi (eff batch 16, grad-ckpt on, lr 2e-6), plus the RAG-only knobs
    the reference path needs: ``ref_dropout`` 0.2 (paper) and ``reference_key`` (the arrow column with
    the retrieved passage; ``ragify_dialogue`` writes ``reference_text``). The launcher builds the
    released moshika-rag (LoRA fused) + a fresh trainable LoRA over the trunk and injects the reference
    as a per-frame ``ref_schedule`` (the training mirror of inference streaming-sum conditioning).
    ``job.hparams`` is read ONLY for the training-speed keys (rendered when set, MOSHIRAG_FAST_HPARAMS)."""
    hp = getattr(job, "hparams", None) or {}
    return (
        f"""# MoshiRAG LoRA finetune config (moshirag_finetune_launcher schema)
hf_repo_id: "{hf_repo_id}"
{_train_data_yaml(job, supports_mix=False)}
out_dir: "{job.out_rundir.get()}"
max_steps: {max_steps}
duration_sec: {job.duration_sec}
lora_rank: {job.lora_rank}
lora_scaling: 2.0
ref_dropout: 0.2
reference_key: "reference_text"
per_gpu_batch: 1
grad_accum: 16
lr: 2e-6
warmup_steps: 200
grad_clip: 1.0
gradient_checkpointing: true
save_every: 500
log_every: 10
seed: {getattr(job, "seed", 0)}
"""
        + render_train_perf_keys(
            hp, ("contiguous_step_inputs", "fused_causal_attention", "fused_lora_add"), who="MoshiRAG lib"
        )
        + render_launcher_fix_keys(hp, ("fp32_master_weights", "loss_norm", "resume_data_position"), who="MoshiRAG lib")
    )


# MoshiRAG on the OWNED moshi_family lib (Phase 5): fork-free LoRA finetune of the released moshika-rag
# via ``moshi_family.moshirag_finetune_launcher`` on the SHARED ``train_loop``. The reference is injected
# as a per-frame ``ref_schedule`` (see moshirag_train_data.py). New ``name`` -> fresh hash. Needs RAG
# training data (a ``reference_text`` column); smokes on plain QA data run ungrounded (mechanism only).
MOSHIRAG_LIB_ADAPTER = FinetuneAdapter(
    name="moshirag_lib",
    batch_size=16,
    overlay_kind="lora",
    base_model="kyutai/moshika-rag-pytorch-bf16",
    weights_name="lora.safetensors",
    render_config=partial(_render_moshirag_lib_config, hf_repo_id="kyutai/moshika-rag-pytorch-bf16"),
    launcher_module="moshi_family.moshirag_finetune_launcher",
    pythonpath_package="moshi_family",
)


# MoshiRAG finetuning lives on MOSHIRAG_LIB_ADAPTER above (the vendored moshi_family.moshirag
# stack). A second, fork-based MOSHIRAG_ADAPTER used to sit here pointing at a launcher stub that
# only raised: the released moshi-rag fork is inference-only, so its training seams (the ARC-Encoder
# reference conditioner's forward/collate, the <ret> trigger supervision, reference dropout and the
# retrieval-delay simulation) were never wired, and its venv was retired in the 2026-07-31 dedup. It
# was constructible but could not run, so it is gone. The remaining RAG-training
# seams apply to the lib adapter now.


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
        # Held-out probe set for the in-training LIVE knowledge probe (MOSHI_LIB_ADAPTER only).
        # Excluded at None -> existing arms keep their hash; set it and the finetune re-hashes (a
        # genuinely new run that now self-monitors factual recall).
        "knowledge_probe_data": None,
        # Free-form extra hyper-params (lr, full_finetuning, text_replay_frac, sample_every, ...).
        # A single bag so new knobs never touch this signature; excluded at None -> existing runs
        # keep their hash, a caller that passes a dict gets a fresh hash from the dict contents.
        "hparams": None,
        # What this run ASKS THE SCHEDULER FOR -- walltime hours and GPU count. A SEPARATE channel
        # from `hparams` on purpose: `hparams` is a hashed constructor argument, so a scheduling key
        # routed through it lowers into the Sisyphus hash. That is not hypothetical -- raising an
        # arm's `hours` 24 -> 48 to reach a longer-walltime partition re-hashed it
        # (ajEFO4gPOlEj -> 6IeG9qqj4s3S), which would have orphaned ~3,000 completed steps and
        # restarted a 35 h run from zero. Excluded at None, so what a run asks for can be changed
        # freely without moving its identity. Guarded by `check_rqmt_not_hashed.py`.
        "compute": None,
        # Another HF hub cache for the base model (training_hf_env). Excluded at None -> every existing run keeps its
        # hash; a run that sets it hashes apart (it may train a different snapshot).
        "hf_cache_dir": None,
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
        knowledge_probe_data=None,
        lora_rank: int = 128,
        hparams: dict | None = None,
        compute: dict | None = None,
        card: dict | None = None,
        hf_cache_dir=None,
    ):
        from .experiment_card import check_card

        self.adapter = adapter
        # Experiment card (why / the one change / baseline / ...; speech_llm.experiment_card): NOT hashed (popped in
        # hash()), written by write_config to output/experiment_card.{json,md}.
        self.card = check_card(card)
        if self.card is not None:
            self.out_card = self.output_path("experiment_card.json")
            self.out_card_md = self.output_path("experiment_card.md")
        self.train_data = train_data
        self.eval_data = eval_data
        # Held-out knowledge probe set (probe.jsonl). Honored ONLY by MOSHI_LIB_ADAPTER (its launcher
        # runs the in-loop generate+score probe); a first-class arg so Sisyphus makes the dep edge.
        self.knowledge_probe_data = knowledge_probe_data
        self.venv_python_path = venv_python_path
        self.seed = seed
        self.duration_sec = duration_sec
        self.audio_jitter_sec = audio_jitter_sec
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.lora_rank = lora_rank
        self.hparams = hparams or {}
        self.out_config = self.output_path("config.yaml")
        self.out_rundir = self.output_path("run_dir", directory=True)
        # Walltime (default 23h -> c23g); <=12 routes to the fast c25g queue, safe for owned-launcher
        # runs because resume() continues across the cap. gpu>1 -> single-node DDP (torchrun
        # --nproc-per-node = visible GPUs); scale cpu/mem per GPU so 4 dataloaders + 4 ranks have
        # headroom.
        #
        # ⚠ READ ORDER IS THE WHOLE POINT. `self.compute` is hash-EXCLUDED, `self.hparams` is hashed.
        # Runs from RULE_RQMT_NOT_HASHED onwards carry these two facts in `compute`, so editing them
        # cannot move the run's hash. Runs older than that epoch carry them inside `hparams` and must
        # keep being read from there -- dropping the fallback would re-hash every one of them, i.e.
        # exactly the damage this split exists to prevent. Do not "tidy" the fallback away; it retires
        # when the pre-2026-09-17 runs do.
        self.compute = dict(compute or {})
        self.hf_cache_dir = hf_cache_dir
        _gpu = int(self.compute.get("gpu", self.hparams.get("gpu", 1)))
        _time_h = int(self.compute.get("rqmt_time_h", self.hparams.get("rqmt_time_h", 23)))
        self.rqmt = {
            "gpu": _gpu,
            "cpu": 6 * _gpu,
            "mem": 24 * _gpu,
            "time": _time_h,
        }

    @classmethod
    def hash(cls, parsed_args):
        # The adapter is bundled callables; hash it by its stable ``name`` only, so
        # refactoring the adapter's functions never re-hashes existing runs.
        d = dict(parsed_args)
        d["adapter"] = d["adapter"].name
        # Scheduling facts are NOT part of a run's identity -- dropped unconditionally so changing
        # what a run asks the scheduler for cannot orphan it.
        #
        # ⚠ This pop is load-bearing and `__sis_hash_exclude__` cannot replace it: that mechanism
        # excludes an argument only while it equals the listed value, which is why `hparams: None`
        # works. A populated `compute={"rqmt_time_h": 48, ...}` is not the default, so it would be
        # hashed like any other dict -- i.e. the split would move the leak from one hashed channel
        # to another and look fixed. Asserted by `check_rqmt_not_hashed.py`, which failed on exactly
        # that before this line existed. Popping a key the old runs never had is hash-neutral for
        # them (verified: 2,100 hashes unchanged).
        d.pop("compute", None)
        # The experiment card is documentation, never identity (hash-neutral for every run: none had the key).
        d.pop("card", None)
        return super().hash(d)

    def tasks(self):
        yield Task("write_config", mini_task=True)
        # resume="run": an interrupted run (preemption / 12h-cap timeout) is rescheduled and re-runs
        # run(), which continues from the latest checkpoint (owned launcher _resume_step). Preemption-safe.
        yield Task("run", resume="run", rqmt=self.rqmt)

    def completed_fraction(self):
        return finetune_completed_fraction(self, self.adapter)

    def write_config(self):
        if getattr(self, "card", None) is not None:
            from .experiment_card import write_card

            write_card(self, self.out_card.get_path(), self.out_card_md.get_path())
        write_finetune_config(self, self.adapter)

    def run(self):
        launch_training(self, self.adapter)
