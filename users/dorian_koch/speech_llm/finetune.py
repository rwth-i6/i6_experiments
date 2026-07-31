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

    # Deterministic MASTER_PORT (PYTHONHASHSEED-independent) so concurrent single-node trainings on one
    # machine don't collide. Key off SLURM_JOB_ID (unique per job, identical across a job's ranks): under
    # SLURM each single-GPU job's cgroup renumbers its visible GPU to "0", so hashing CUDA_VISIBLE_DEVICES
    # gave two co-located jobs the SAME port -> EADDRINUSE. Fall back to CUDA_VISIBLE_DEVICES off-SLURM.
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "-1")  # also used below for nproc
    port_key = os.environ.get("SLURM_JOB_ID") or cuda_devices
    port_offset = int(hashlib.md5(port_key.encode()).hexdigest()[:4], 16) % 4000

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
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
    env["HF_HOME"] = HF_CACHE_DIR.get()
    # The base model + tokenizer are pre-staged into HF_HOME by the eval graph, so force offline
    # loading: this skips any HF download / Xet re-verification, which (a) avoids re-fetching a 16 GB
    # checkpoint every run and (b) does not depend on writable HF cache space (the shared hpcwork
    # cache can be full -- a Xet "Background writer channel closed" download error is that symptom).
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # HF_CACHE_DIR already points AT the hub cache dir (".../common_hf_home/hub", which holds the
    # models--* trees), so it is HF_HUB_CACHE -- NOT HF_HOME. HF appends "/hub" to HF_HOME, so setting
    # HF_HOME to it makes HF look in ".../hub/hub/models--*" and miss the pre-staged model. Point
    # HF_HUB_CACHE straight at the real dir so offline resolve (refs/main -> snapshot -> blob) works.
    env["HF_HUB_CACHE"] = HF_CACHE_DIR.get()
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
    recipe_root = next((str(p) for p in Path(__file__).parents if (p / "i6_experiments").exists()), None)
    if recipe_root:
        extra_paths.append(recipe_root)
        # Lib launchers (``-m moshi_family.<...>``) live under recipe/speech_llm/full_duplex, which is
        # NOT recipe_root; add it so the owned moshi_family package imports (harmless for fork
        # launchers -- it is just one more importable dir on the path).
        lib_parent = os.path.join(recipe_root, "speech_llm", "full_duplex")
        if os.path.isdir(lib_parent):
            extra_paths.append(lib_parent)
    if env.get("PYTHONPATH"):
        extra_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    print(f"Running {adapter.name} training with command: {' '.join(command)}", flush=True)
    print(f"Using HF cache directory: {HF_CACHE_DIR}")
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
    """Render the PersonaPlex training config (consumed by personaplex_finetune_launcher, NOT the
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
    render_config=partial(_render_moshi_finetune_config, hf_repo_id="kyutai/moshiko-pytorch-bf16"),
    launcher_module="i6_experiments.users.dorian_koch.speech_llm.moshi_finetune_launcher",
    pythonpath_package="moshi_finetune",
)


# --------------------------------------------------------------------------- #
# PersonaPlex adapter (IMPLEMENTED, single-GPU -- see projects/2026-01-speech-llm/personaplex.md).
# Training base DECIDED (2026-06-19 investigation): moshi_finetune is NOT installed in the
# personaplex venv, so the launcher must drive the PersonaPlex fork's OWN model. Good news --
# unlike the moshi-rag fork, the personaplex fork ships a built-in training path:
# ``moshi.models.lm.LMModel.forward_train(codes) -> LMOutput`` (delays handled, logits+masks) +
# ``create_loss_report``, loaded via ``loaders.get_moshi_lm(model.safetensors)``. So pythonpath_package
# is "moshi" and the launcher builds a loop on forward_train (port moshi_finetune's loop onto this
# model); the config is a personaplex-specific YAML, NOT the moshi_finetune schema. The launcher
# is IMPLEMENTED and training (single-GPU: train_scope=heads, backbone frozen, full-FT depformer +
# heads); voice-prompt conditioning (the hybrid role/voice collator) is still TODO -- we condition
# on role text only. Wire via ``SpeechFinetune(adapter=PERSONAPLEX_ADAPTER, venv_python_path=personaplex_venv(), ...)``.
# --------------------------------------------------------------------------- #
PERSONAPLEX_ADAPTER = FinetuneAdapter(
    name="personaplex",
    batch_size=32,  # paper
    render_config=partial(_render_personaplex_config, hf_repo_id="nvidia/personaplex-7b-v1"),
    launcher_module="i6_experiments.users.dorian_koch.speech_llm.personaplex_finetune_launcher",
    pythonpath_package="moshi",  # the moshi-personaplex fork (import name `moshi`); installed by personaplex_venv()
    # progress defaults to ("metrics.train.jsonl", "percent_done") -- exactly what the launcher writes.
)


# PersonaPlex on the OWNED moshi_family lib (Phase 5): same paper recipe + config schema, but the
# launcher builds on moshi_family.personaplex (no fork) via the SHARED moshi_family.train_loop, and
# runs in moshi_family_venv. New ``name`` -> fresh hash (intended; this is the fork->lib migration).
# pythonpath_package="moshi_family" is located via the lib path launch_training now adds to PYTHONPATH.
PERSONAPLEX_LIB_ADAPTER = FinetuneAdapter(
    name="personaplex_lib",
    batch_size=32,
    render_config=partial(_render_personaplex_config, hf_repo_id="nvidia/personaplex-7b-v1"),
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
    return f"""# base-Moshi LoRA finetune config (moshi_finetune_launcher schema)
hf_repo_id: "{hf_repo_id}"
{_data}
out_dir: "{job.out_rundir.get()}"
max_steps: {max_steps}
duration_sec: {job.duration_sec}
audio_jitter_sec: {getattr(job, "audio_jitter_sec", 0.0)}
lora_rank: {job.lora_rank}
lora_scaling: 2.0
per_gpu_batch: 1
grad_accum: {hp.get("grad_accum", 16)}
lr: {_yaml_float(_lr)}
depth_lr: {_yaml_float(hp.get("depth_lr", _lr))}
temporal_lr: {_yaml_float(hp.get("temporal_lr", _lr))}
audio_other_weight: {_yaml_float(hp.get("audio_other_weight", 0.01))}
text_pad_weight: {_yaml_float(hp.get("text_pad_weight", 0.5))}
warmup_steps: {hp.get("warmup_steps", 200)}
grad_clip: 1.0
gradient_checkpointing: true
save_every: 500
log_every: 10
seed: {getattr(job, "seed", 0)}
sample_every: {hp.get("sample_every", 100)}
eval_batches: {hp.get("eval_batches", 8)}
general_eval_data: "{_gen}"
do_eval: {str(getattr(job, "eval_data", None) is not None).lower()}
eval_data: "{job.eval_data.get() if getattr(job, "eval_data", None) is not None else ""}"
eval_freq: {hp.get("eval_freq", 100)}
knowledge_probe_data: "{job.knowledge_probe_data.get() if getattr(job, "knowledge_probe_data", None) is not None else ""}"
knowledge_probe_every: {hp.get("knowledge_probe_every", 100)}
knowledge_probe_batch_size: {hp.get("knowledge_probe_batch_size", 4)}
knowledge_probe_capture_s: {_yaml_float(hp.get("knowledge_probe_capture_s", 20.0))}
knowledge_probe_n: {hp.get("knowledge_probe_n", 0)}
"""


# Base Moshi on the OWNED moshi_family lib (Phase 5): fork-free LoRA finetune of moshiko via
# ``moshi_family.moshi_finetune_launcher`` on the SHARED ``train_loop`` + uniform ``modules.lora``,
# in moshi_family_venv. New ``name`` -> fresh hash (intended fork->lib migration). Eval routes the
# saved adapter through ``moshi_family_backend_spec(lora_rank=...)`` (--overlay + --lora_rank/scaling).
MOSHI_LIB_ADAPTER = FinetuneAdapter(
    name="moshi_lib",
    batch_size=16,
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
    as a per-frame ``ref_schedule`` (the training mirror of inference streaming-sum conditioning)."""
    return f"""# MoshiRAG LoRA finetune config (moshirag_finetune_launcher schema)
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


# MoshiRAG on the OWNED moshi_family lib (Phase 5): fork-free LoRA finetune of the released moshika-rag
# via ``moshi_family.moshirag_finetune_launcher`` on the SHARED ``train_loop``. The reference is injected
# as a per-frame ``ref_schedule`` (see moshirag_train_data.py). New ``name`` -> fresh hash. Needs RAG
# training data (a ``reference_text`` column); smokes on plain QA data run ungrounded (mechanism only).
MOSHIRAG_LIB_ADAPTER = FinetuneAdapter(
    name="moshirag_lib",
    batch_size=16,
    render_config=partial(_render_moshirag_lib_config, hf_repo_id="kyutai/moshika-rag-pytorch-bf16"),
    launcher_module="moshi_family.moshirag_finetune_launcher",
    pythonpath_package="moshi_family",
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
# Wire via ``SpeechFinetune(adapter=MOSHIRAG_ADAPTER, ...)``. NOTE: the moshi-rag fork venv is retired; Phase 5 rebuilds MoshiRAG training on the moshi_family lib.
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
    pythonpath_package="moshi",  # legacy moshi-rag fork tag; the fork venv is retired (Phase 5 rebuilds it on the lib)
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
        # Held-out probe set for the in-training LIVE knowledge probe (MOSHI_LIB_ADAPTER only).
        # Excluded at None -> existing arms keep their hash; set it and the finetune re-hashes (a
        # genuinely new run that now self-monitors factual recall).
        "knowledge_probe_data": None,
        # Free-form extra hyper-params (lr, full_finetuning, text_replay_frac, sample_every, ...).
        # A single bag so new knobs never touch this signature; excluded at None -> existing runs
        # keep their hash, a caller that passes a dict gets a fresh hash from the dict contents.
        "hparams": None,
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
    ):
        self.adapter = adapter
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
        # time from hparams["rqmt_time_h"] (default 23h -> c23g). <=12 routes to the fast c25g queue;
        # safe for owned-launcher runs because resume() continues across the 12h cap. rqmt isn't hashed.
        # gpu>1 -> single-node DDP (torchrun --nproc-per-node = visible GPUs); scale cpu/mem per GPU so 4
        # dataloaders + 4 ranks have headroom. rqmt is NOT hashed, so this never re-hashes existing runs.
        _gpu = int(self.hparams.get("gpu", 1))
        self.rqmt = {
            "gpu": _gpu,
            "cpu": 6 * _gpu,
            "mem": 24 * _gpu,
            "time": int(self.hparams.get("rqmt_time_h", 23)),
        }

    @classmethod
    def hash(cls, parsed_args):
        # The adapter is bundled callables; hash it by its stable ``name`` only, so
        # refactoring the adapter's functions never re-hashes existing runs.
        d = dict(parsed_args)
        d["adapter"] = d["adapter"].name
        return super().hash(d)

    def tasks(self):
        yield Task("write_config", mini_task=True)
        # resume="run": an interrupted run (preemption / 12h-cap timeout) is rescheduled and re-runs
        # run(), which continues from the latest checkpoint (owned launcher _resume_step). Preemption-safe.
        yield Task("run", resume="run", rqmt=self.rqmt)

    def completed_fraction(self):
        return finetune_completed_fraction(self, self.adapter)

    def write_config(self):
        write_finetune_config(self, self.adapter)

    def run(self):
        launch_training(self, self.adapter)
