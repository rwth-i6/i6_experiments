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
from pathlib import Path

from sisyphus import Job, Task

from . import common as _common
from .common import HF_CACHE_DIR, HF_HOME_DIR, last_jsonl_value, moshi_family_lib_parent, recipe_root, worker_env
from .moshi_arrow_config import ArrowDataConfig
from .speech_inference import CheckpointOutputs
from .finetune_adapters import (  # noqa: F401 -- re-exported: callers and job pickles resolve finetune.<name>
    AUDEX_STAGE0_FAST_TRAIN_PERF,
    AUDEX_STAGE1_FAST_TRAIN_PERF,
    FAST_TRAIN_PRESETS,
    FinetuneAdapter,
    LAUNCHER_FIX_KEYS,
    LAUNCHER_FIX_PRESETS,
    MOSHIRAG_FAST_HPARAMS,
    MOSHIRAG_LIB_ADAPTER,
    MOSHI_ADAPTER,
    MOSHI_FAST_HPARAMS,
    MOSHI_FULL_FAST_HPARAMS,
    MOSHI_LIB_ADAPTER,
    MOSHI_LIB_FULL_ADAPTER,
    MOSHI_LIB_PERF_KEYS,
    PERSONAPLEX_FAST_HPARAMS,
    PERSONAPLEX_FULL_FAST_HPARAMS,
    PERSONAPLEX_LIB_ADAPTER,
    PERSONAPLEX_LIB_FULL_ADAPTER,
    PERSONAPLEX_LIB_LORA_ADAPTER,
    RL_FAST_TRAIN_PERF,
    TRAIN_PERF_KEYS,
    _render_moshi_finetune_config,
    _render_moshi_lib_config,
    _render_moshirag_lib_config,
    _render_personaplex_config,
    _render_personaplex_lora_config,
    _single_train_data_path,
    _train_data_yaml,
    _yaml_float,
    render_launcher_fix_keys,
    render_train_perf_keys,
    train_data_specs,
)


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
# Generic job for *new* architectures (Moshi keeps its own frozen class).
# --------------------------------------------------------------------------- #
class SpeechFinetune(CheckpointOutputs, Job):
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
        self.world_size_unhashed = bool(self.compute.get("world_size_unhashed", False))  # legacy pin, see hash()
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
        compute = d.pop("compute", None) or {}
        # ...except a training GPU count above 1: it is the DDP world size, which scales the effective batch and
        # each rank's data stride. 1-GPU runs (all of them until 2026-11-04) keep their hashes.
        # The two finished 4-GPU p5 runs keep their hashes via `world_size_unhashed` (legacy_pins.LEGACY_PINS).
        if int(compute.get("gpu", 1)) != 1 and not compute.get("world_size_unhashed"):
            d["train_world_size"] = int(compute["gpu"])
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
