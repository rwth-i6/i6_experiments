"""Unified speech-LLM inference job + overlay-checkpoint resolver, shared by both benchmarks.

``SpeechInference`` collapses the former ``MoshiInference`` (knowledge) and
``FullDuplexBenchEval_Inference`` (FDB) into one job: the backend access (server / offline
driver / RAG retrieval) is the shared ``BackendInferenceMixin``; the two ``mode``s differ only
in how clips are enumerated, the on-disk output layout, and FDB's inline NeMo-ASR scoring.

Attribute names are now uniform (``server`` / ``venv_python_path``) so the mixin's
name-indirection hooks are trivial -- the recipe builders map a ``BackendSpec`` onto these.

``ResolveOverlayCheckpoint`` merges the former ``ResolveLoraCheckpoint`` (Moshi LoRA: a
``config.json`` + ``lora.safetensors`` pair) and ``ResolvePersonaPlexCheckpoint`` (PersonaPlex:
a single ``trained_heads`` partial state-dict) behind one ``overlay_kind`` discriminator. Both
feed the same per-run ``lora_weights`` / ``lora_config`` overlay seam on ``SpeechInference``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from sisyphus import Job, Task, tk

from .clip_store import is_clip_dataset, materialise_clips, open_clips, write_clips
from .common import nested_shard, visible_gpus

from .inference_harness import (
    BackendInferenceMixin,
    StreamOptions,
    write_pair_manifest,
    fdb_files_for_tasks,
    FDB_TASK_MAP,
)
from .moshi_client import MoshiFileClient, _ws_url, moshi_server


#: The trainer's default checkpoint cadence. ``finetune.py``'s config template renders 500 when a run
#: does not override ``save_every``; this is the ONE copy the graph-build guard below and
#: ``attach_knowledge_evals`` validate against (``quick_knowledge_eval`` re-exports it).
SAVE_EVERY = 500

#: Overlay kinds stored as ``run_dir/checkpoints/checkpoint_<step>/`` -- the layout ``run_training``
#: writes at every ``save_every`` multiple and at ``max_steps``. PersonaPlex heads live elsewhere.
CHECKPOINT_DIR_KINDS = ("lora", "full", "audex_stage0")


def impossible_checkpoint_reason(step, *, max_steps, save_every: int = SAVE_EVERY):
    """Why ``step`` can NEVER be a checkpoint of a run that stops at ``max_steps`` and saves every
    ``save_every`` -- or ``None`` when it can.

    The trainer writes ``checkpoint_<step>`` at every positive multiple of ``save_every`` and at
    ``max_steps`` itself (the final save in ``run_training``), and nothing else. ``step=None`` names
    the latest checkpoint and is always possible. ``max_steps=None`` is a run sized at RUN time
    (``num_epochs``), so its end is unknowable here and only the cadence rule applies.

    One rule, two callers: ``attach_knowledge_evals`` applies it to a whole track when the track is
    declared, and ``assert_checkpoint_will_exist`` applies it to every single checkpoint reference
    the graph mints, whichever path built it.
    """
    if step is None:
        return None
    if not (step > 0 and step % save_every == 0) and step != max_steps:
        return (
            f"is not a checkpoint -- must be a positive multiple of save_every={save_every}, "
            f"or max_steps ({max_steps}) itself"
        )
    if max_steps is not None and step > max_steps:
        return (
            f"is past the end of the run (max_steps={max_steps}); that checkpoint will never exist "
            f"and its eval job would fail"
        )
    return None


def expected_checkpoint_glob(run_dir: str, overlay_kind: str, step) -> str:
    """Where ``ResolveOverlayCheckpoint.run`` will look for ``step`` of ``overlay_kind`` -- as a glob,
    so ``step=None`` (latest) matches any checkpoint. Kept next to the resolver so the two cannot
    disagree about the layout."""
    if overlay_kind in CHECKPOINT_DIR_KINDS:
        name = "checkpoint_*" if step is None else f"checkpoint_{step:06d}"
        return os.path.join(run_dir, "checkpoints", name)
    name = "trained_heads.safetensors" if step is None else f"trained_heads.step{step}.safetensors"
    return os.path.join(run_dir, "consolidated", name)


def assert_checkpoint_will_exist(run_dir: tk.Path, overlay_kind: str, step) -> None:
    """Graph-build guard (user request, 2026-09-09): an eval must never be wired to a checkpoint
    that cannot arrive. Refuse it HERE, on the login node while the manager loads the config, not
    hours later in a job that has already allocated a GPU -- or, worse, never at all.

    Sisyphus waits on input *paths*, not input jobs: a job whose input lives under a run that has
    already FINISHED without writing it is not an error to the manager, it is merely "waiting", and
    it waits forever with the graph looking healthy. Two earlier incidents were the milder, visible
    form of the same mistake (a8-4gpu's track named steps past the run's end, a11_full's named
    steps its 1250 cadence never wrote; both failed in ``run()`` after the graph had accepted them).

    Three sources of truth, strongest first:

    1. **The disk.** If the checkpoint is already there, it exists -- whatever the rules below think.
    2. **The run's state.** If the run has finished (its ``finished`` marker is written, or a
       creator-less ``run_dir`` already exists on disk) and the checkpoint is not there, it never
       will be. Fail, naming what the run *did* write.
    3. **The run's declared shape.** For a run still to come, its ``hparams`` (``max_steps``,
       ``save_every``) say which steps the trainer will save; ``impossible_checkpoint_reason`` is
       the rule. Only for the ``checkpoints/checkpoint_<step>`` layouts -- the PersonaPlex heads
       cadence is not declared this way, so it gets (1) and (2) only.

    Reads the creator's finished marker directly (``sisyphus.job.job_finished``) rather than calling
    ``Job._sis_finished()``, which is rate-limited and recurses into runnability checks.
    """
    import glob

    from sisyphus.job import job_finished

    run_path = run_dir.get_path()
    pattern = expected_checkpoint_glob(run_path, overlay_kind, step)
    if glob.glob(pattern):
        return
    creator = run_dir.creator
    if creator is None:
        finished = os.path.isdir(run_path)
    else:
        finished = job_finished(creator._sis_path())
    if finished:
        parent = os.path.dirname(pattern)
        have = sorted(os.listdir(parent)) if os.path.isdir(parent) else f"no {os.path.basename(parent)}/ at all"
        raise AssertionError(
            f"{os.path.basename(pattern)} of {run_path} will never exist: the run has already "
            f"finished and did not write it (it wrote {have}). An eval wired to it would wait "
            f"forever without ever showing as an error. Fix the step list / save_every of the "
            f"caller, or the run's overlay_kind ({overlay_kind!r}) if the layout is wrong."
        )
    hparams = getattr(creator, "hparams", None)
    if step is not None and overlay_kind in CHECKPOINT_DIR_KINDS and isinstance(hparams, dict):
        reason = impossible_checkpoint_reason(
            step,
            max_steps=hparams.get("max_steps"),
            save_every=int(hparams.get("save_every", SAVE_EVERY)),
        )
        assert reason is None, (
            f"checkpoint step {step} of {run_path} {reason}. The run has not finished yet, but its "
            f"declared shape already rules this step out -- the eval would fail after the run, on a "
            f"missing file, or wait forever."
        )


class ResolveOverlayCheckpoint(Job):
    """Resolve a finetune ``run_dir`` to the overlay file(s) the offline driver loads on top
    of the base model.

    ``overlay_kind``:
      * ``"lora"`` (Moshi): ``run_dir/checkpoints/checkpoint_<step>/consolidated/`` ->
        ``out_weights`` = ``lora.safetensors``, ``out_config`` = ``config.json``.
      * ``"personaplex_heads"`` (PersonaPlex): ``run_dir/consolidated/trained_heads[.step<N>].safetensors``
        -> ``out_weights`` only (``out_config`` is ``None``; PersonaPlex is a partial state-dict, no config).
    ``step=None`` -> latest checkpoint (lora) / final consolidated (personaplex).
    """

    def __init__(self, *, run_dir: tk.Path, overlay_kind: str = "lora", step: int | None = None):
        assert overlay_kind in ("lora", "personaplex_heads", "audex_stage0", "full"), overlay_kind
        # Every checkpoint reference in the graph is minted here, so this is the one place a
        # never-to-exist step can be refused at graph build. Not hashed (no new argument).
        assert_checkpoint_will_exist(run_dir, overlay_kind, step)
        self.run_dir = run_dir
        self.overlay_kind = overlay_kind
        self.step = step  # None -> latest (lora) / final consolidated (personaplex)
        _weights_name = {
            "lora": "lora.safetensors",
            "personaplex_heads": "trained_heads.safetensors",
            "audex_stage0": "stage0.safetensors",
            # A full finetune: the whole LM state dict, not an adapter. Same checkpoint layout as
            # "lora", so it resolves through the same branch below.
            "full": "model.safetensors",
        }[overlay_kind]
        self.out_weights = self.output_path(_weights_name)
        self.out_config = self.output_path("config.json") if overlay_kind == "lora" else None

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _link(target: Path, link_name: str) -> None:
        """Symlink ``target`` -> ``link_name``, refusing a target that does not exist.

        ``os.symlink`` creates a DANGLING link without complaining, so a wrong ``overlay_kind``
        used to make this job finish successfully while pointing at files the checkpoint never
        wrote. Sisyphus then marked it finished, released the downstream SpeechInference, and that
        died with a bare "Job isn't runnable, probably some inputs are not ready" -- an empty
        error.run.1, no log.run.1, and a SLURM state of COMPLETED in one second. Two weeks of the
        graph stalled on a failure that named neither the missing file nor the job that wanted it
        (a11_full, 2026-08-21). Fail here instead, where the path is still in hand.
        """
        assert target.exists(), (
            f"{target} does not exist, so symlinking it would create a dangling link that only "
            f"fails much later, in a downstream job, with no mention of this path. Check that "
            f"overlay_kind matches what this run actually wrote: "
            f"{sorted(p.name for p in target.parent.iterdir()) if target.parent.is_dir() else f'{target.parent} is not a directory'}"
        )
        os.symlink(target, link_name)

    def run(self):
        if self.overlay_kind in ("lora", "audex_stage0", "full"):
            ckpt_root = Path(self.run_dir.get()) / "checkpoints"
            ckpts = sorted(ckpt_root.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[-1]))
            assert ckpts, f"No checkpoints found in {ckpt_root}"
            if self.step is not None:
                chosen = ckpt_root / f"checkpoint_{self.step:06d}"
                assert chosen.exists(), f"{chosen} not found; have {[c.name for c in ckpts]}"
            else:
                chosen = ckpts[-1]
            consolidated = chosen / "consolidated"
            print(f"Resolved {self.overlay_kind} checkpoint: {chosen.name}", flush=True)
            if self.overlay_kind == "lora":
                self._link(consolidated / "config.json", self.out_config.get())
                self._link(consolidated / "lora.safetensors", self.out_weights.get())
            elif self.overlay_kind == "full":  # whole state dict, no LoRA config to carry
                self._link(consolidated / "model.safetensors", self.out_weights.get())
            else:  # audex_stage0/stage1: partial state-dict overlay, no config
                import glob as _glob

                cands = sorted(_glob.glob(str(consolidated / "stage*.safetensors")))
                assert len(cands) == 1, f"expected one stage*.safetensors in {consolidated}, got {cands}"
                self._link(Path(cands[0]), self.out_weights.get())
        else:  # personaplex_heads
            consolidated = Path(self.run_dir.get()) / "consolidated"
            name = (
                f"trained_heads.step{self.step}.safetensors" if self.step is not None else "trained_heads.safetensors"
            )
            chosen = consolidated / name
            assert chosen.exists(), (
                f"{chosen} not found; have {[p.name for p in consolidated.glob('trained_heads*.safetensors')]}"
            )
            print(f"Resolved PersonaPlex checkpoint: {chosen.name}", flush=True)
            self._link(chosen, self.out_weights.get())


class SpeechInference(BackendInferenceMixin, Job):
    """Unified speech-LLM inference for the knowledge benchmark and Full-Duplex-Bench.

    ``mode="knowledge"``: enumerate ``*.wav`` in ``in_dir`` (optionally sharded ``[shard::num_shards]``),
        produce one reply ``<name>.wav`` in ``out_dir``.
    ``mode="fdb"``: enumerate the FDB dataset clips for ``fdb_task`` into a manifest, produce
        ``<ind>/output.wav`` in ``out_dir``, then score with the benchmark's NeMo ASR.
    """

    __sis_hash_exclude__ = {
        # Pluggable speech-LLM backend (see speech_backends.py); excluded at the Moshi defaults
        # so a base-Moshi job's hash does not carry unused-feature noise. A non-Moshi backend's
        # server / offline_module / retrieval_llm differ and so already yield a distinct hash.
        "server": moshi_server,
        "file_client": MoshiFileClient,
        "ws_url": _ws_url,
        "unmute_llm": None,
        "offline_script": None,
        "offline_module": None,
        "offline_extra_args": (),
        "cloud_api": False,
        "retrieval_llm": None,
        "lora_weights": None,
        "lora_config": None,
        "hf_repo": None,
        "seed": None,
        # FDB-only attrs are absent (None) for knowledge jobs and vice-versa, so excluding their
        # "absent" default keeps each mode's hash clean.
        "asr_venv_python": None,
        "oracle_dataset": None,
        # Hashed code-version knob: bump to force a fresh hash (and cascade re-runs to downstream
        # transcription/grading/eval) after a lib-code fix that is NOT part of the hash. Excluded
        # at the default (1) so it is a no-op until bumped.
        "code_version": 1,
        # Clip storage layout for knowledge mode: "wav" writes one reply <i>.wav (the original),
        # "hf" writes a single arrow dataset. Excluded at the "wav" default so no existing job
        # re-hashes. fdb mode ignores it -- its nested <ind>/output.wav layout is read directly by
        # the benchmark's NeMo ASR, which we do not own.
        "storage": "wav",
    }

    def __init__(
        self,
        *,
        mode: str,
        venv_python_path: tk.AbstractPath | None = None,
        server=moshi_server,
        file_client=MoshiFileClient,
        ws_url=_ws_url,
        unmute_llm: str | None = None,
        offline_script: str | None = None,
        offline_module: str | None = None,
        offline_extra_args: tuple = (),
        cloud_api: bool = False,
        retrieval_llm: str | None = None,
        lora_weights: tk.Path | None = None,
        lora_config: tk.Path | None = None,
        hf_repo: str | None = None,
        seed: int | None = None,
        code_version: int = 1,
        # --- knowledge mode ---
        in_dir: tk.Path | None = None,
        shard: int | None = None,
        num_shards: int | None = None,
        lead_in_s: float = 2.0,
        capture_s: float = 24.0,
        batch_size: int = 32,
        oracle_dataset: tk.Path | None = None,
        storage: str = "wav",
        # --- fdb mode ---
        fdb_task: str | None = None,
        asr_venv_python: tk.AbstractPath | None = None,
    ):
        assert mode in ("knowledge", "fdb"), mode
        self.mode = mode
        # Interpreter the server/offline driver runs under (None -> the worker's own .venv).
        self.venv_python_path = venv_python_path
        # Pluggable backend: server ctx-mgr + streaming client + handle->url adapter (Moshi default).
        self.server = server
        self.file_client = file_client
        self.ws_url = ws_url
        self.unmute_llm = unmute_llm
        # Offline driver: a script under dorian_koch/ OR a ``python -m <module>`` (moshi_family lib).
        self.offline_script = offline_script
        self.offline_module = offline_module
        self.offline_extra_args = tuple(offline_extra_args)
        # Cloud realtime backend => remote model, no GPU: run as a login-node mini_task.
        self.cloud_api = cloud_api
        # RAG retrieval LLM served (vllm_server) as the offline driver's retrieval backend (MoshiRAG).
        self.retrieval_llm = retrieval_llm
        # Optional fine-tuned overlay (LoRA or PersonaPlex heads); None -> base model.
        self.lora_weights = lora_weights
        self.lora_config = lora_config
        # Override the model HF repo the moshi server loads (e.g. a released RL checkpoint like
        # kyutai/moshika-rl-seamless); None -> the server DEFAULT_REPO (base Moshi). Hashed (excluded
        # at None in __sis_hash_exclude__) so a different repo yields a distinct job.
        self.hf_repo = hf_repo
        # RNG seed forwarded to the moshi server (via the seeded-server wrapper); None -> the
        # server's hardcoded default (42424242). Hashed (excluded at None) so each seed is a
        # distinct replicate job -- used to measure generation seed-sensitivity of the metrics.
        self.seed = seed
        # Hashed code-version knob (see __sis_hash_exclude__); default 1 == current behaviour.
        self.code_version = code_version

        # knowledge-mode inputs
        self.in_dir = in_dir
        self.shard = shard
        self.num_shards = num_shards
        # Feed `lead_in_s` of silence (Moshi greets), the question, then `capture_s` of trailing
        # silence, capturing Moshi's ENTIRE reply (greeting included, nothing trimmed).
        self.lead_in_s = lead_in_s
        self.capture_s = capture_s
        self.batch_size = batch_size
        # KAME oracle source (sampled dataset; None for all other backends).
        self.oracle_dataset = oracle_dataset
        # MoshiRAG (retrieval_llm backend) is a serial per-clip retrieval pump with no batched path
        # (batching would re-introduce the fork's batched-server deadlock). Refuse B>1 at graph-build
        # time so it fails here, not after a wasted GPU allocation. See moshirag/offline_inference.py.
        assert not (self.retrieval_llm is not None and batch_size != 1), (
            f"MoshiRAG (retrieval backend) only supports batch_size=1; got {batch_size}"
        )

        # fdb-mode inputs
        self.fdb_task = fdb_task
        self.asr_venv_python = asr_venv_python
        if mode == "fdb":
            self.fdb_data = tk.Path(
                "/home/tt201262/setups/2026-01-speech-llm/projects/Full-Duplex-Bench/v1_v1.5/dataset/v1.0",
                hash_overwrite="FullDuplexBench-datasets",
            )

        assert storage in ("wav", "hf"), f"storage must be 'wav' or 'hf', got {storage!r}"
        # fdb mode always writes the nested wav layout the benchmark's own ASR expects.
        self.storage = storage if mode == "knowledge" else "wav"
        self.out_dir = self.output_path("speech_output", directory=True)

        # rqmt is NOT part of the Sisyphus hash (it is an instance attribute, not a
        # constructor arg). Callers tune walltime/GPUs per-backend by mutating job.rqmt
        # AFTER construction (see sharded_knowledge_inference / fdb.py), so a resource
        # change never re-hashes the job.
        if mode == "knowledge":
            self.rqmt = {"gpu": 1, "cpu": 4, "mem": 16, "time": 8}
        else:
            self.rqmt = {"gpu": 1, "cpu": 2, "mem": 16, "time": 4}

    def tasks(self):
        if self.cloud_api:
            yield Task("run", mini_task=True)  # remote model: no GPU, needs login-node internet
        else:
            yield Task("run", rqmt=self.rqmt)

    # --- BackendInferenceMixin hooks (now trivial: uniform attribute names) ---
    def _server_callable(self):
        return self.server

    def _python_exe(self):
        return self.venv_python_path.get() if self.venv_python_path is not None else None

    # --- Sisyphus observability (manager-side, runtime-only -> no hash change). Knowledge mode
    # only: it writes a flat ``<i>.wav`` per input so one scandir counts progress. FDB writes
    # nested ``<ind>/output.wav`` (and is single-shard/fast), so no progress estimate there. ---
    @staticmethod
    def _count_wavs(d: str) -> int:
        try:
            with os.scandir(d) as it:
                return sum(1 for e in it if e.name.endswith(".wav"))
        except OSError:
            return 0

    def _progress_dir(self) -> str:
        """Where clips are accumulating right now.

        Under storage="hf" the worker writes into the scratch dir and only packs at the very end,
        so counting output/ would report 0% for the whole run and then jump to 100% -- exactly the
        "is it working or hung?" ambiguity the progress hooks exist to remove.
        """
        if self.storage == "hf":
            return os.path.join(self._sis_path(), "clip_scratch")
        return self.out_dir.get_path()

    def _shard_total(self) -> "int | None":
        cached = getattr(self, "_total_cache", 0)
        if cached:
            return cached
        # The input may be either layout, so count through the store rather than globbing wavs.
        in_path = self.in_dir.get_path()
        n = len(open_clips(in_path)) if is_clip_dataset(in_path) else self._count_wavs(in_path)
        if n and self.shard is not None and self.num_shards:
            n = len(range(self.shard, n, self.num_shards))
        if n:
            self._total_cache = n
        return n or None

    def completed_fraction(self):
        if self.mode != "knowledge":
            return None
        try:
            total = self._shard_total()
            return max(0.0, min(1.0, self._count_wavs(self._progress_dir()) / total)) if total else None
        except Exception:
            return None

    def info(self):
        if self.mode != "knowledge":
            return None
        try:
            total = self._shard_total()
            return f"{self._count_wavs(self._progress_dir())}/{total} clips" if total else None
        except Exception:
            return None

    # --- run ---
    def run(self):
        if self.mode == "knowledge":
            self._run_knowledge()
        else:
            self._run_fdb()

    def _run_knowledge(self):
        # storage="hf": the driver / streaming path still writes loose wavs, but into a scratch dir
        # inside the job work dir; run() then packs them into ONE arrow dataset and drops the
        # scratch. That keeps all five offline drivers (and their five job venvs) untouched while
        # the job's durable output costs ~3 inodes instead of one per clip. The scratch lives beside
        # output/, so a crashed run leaves it behind for inspection rather than half-written output.
        final_dir = Path(self.out_dir.get())
        out_dir = Path("clip_scratch").absolute() if self.storage == "hf" else final_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        # Materialise an arrow input to scratch wavs HERE, for every path, rather than expecting the
        # driver to understand arrow. offline_cli.resolve_run can do it for the five moshi_family
        # drivers, but the FORK-era drivers (i6_experiments/.../moshi_offline_inference.py, used by
        # the default MOSHI_BACKEND, and personaplex_offline_inference.py) do not go through it --
        # they glob *.wav directly. Passing them an arrow dir silently yields zero pairs and an empty
        # output, which is how storage_ab_hf failed on 2026-08-02. Doing it at the job makes the
        # input contract driver-agnostic: every driver, fork or lib, always sees a dir of <i>.wav.
        in_path = Path(self.in_dir.get())
        if is_clip_dataset(in_path):
            in_path = Path("clip_input").absolute()
            materialise_clips(self.in_dir.get(), in_path)
            print(
                f"[clips] materialised arrow input -> {in_path} ({len(list(in_path.glob('*.wav')))} wavs)", flush=True
            )

        if self.offline_script is not None or self.offline_module is not None:
            common = dict(
                python_exe=self._python_exe(),
                in_dir=str(in_path),
                out_dir=str(out_dir),
                lead_in_s=self.lead_in_s,
                capture_s=self.capture_s,
                batch_size=self.batch_size,
                oracle_dataset=(self.oracle_dataset.get() if self.oracle_dataset is not None else None),
            )
            # Per-GPU fan-out (backlog G1): the driver's own sharding is strided by clip index
            # (wavs[shard::num_shards]), so worker k of n inside this job's shard is the NESTED
            # stride (shard + k*num_shards, num_shards*n) -- a partition of the job's clips with no
            # driver change. Outputs are index-disjoint wavs in the same scratch dir, packed once.
            # A retrieval-backed backend pins its driver to one card itself, so it stays single.
            plan = self._driver_shards()
            if len(plan) == 1:
                shard, num_shards, _env = plan[0]
                self._offline(shard=shard, num_shards=num_shards, **common)
            else:
                from concurrent.futures import ThreadPoolExecutor

                def one(spec):
                    shard, num_shards, env = spec
                    self._offline(shard=shard, num_shards=num_shards, extra_env=env, **common)

                with ThreadPoolExecutor(max_workers=len(plan)) as ex:
                    list(ex.map(one, plan))
            self._pack_clips(out_dir, final_dir)
            return
        wav_files = sorted(in_path.glob("*.wav"))
        if self.shard is not None and self.num_shards is not None:
            wav_files = wav_files[self.shard :: self.num_shards]
        items = [(wav, out_dir / wav.name) for wav in wav_files]
        self._stream(items, opts=StreamOptions(lead_in_s=self.lead_in_s, capture_s=self.capture_s, progress_every=50))
        self._pack_clips(out_dir, final_dir)

    def _pack_clips(self, scratch: Path, final_dir: Path) -> None:
        """Under storage="hf", fold the scratch wavs (+ monologue/trace sidecars) into one dataset."""
        if self.storage != "hf":
            return
        clips = open_clips(scratch)
        # Zero clips means the run produced nothing -- almost always an empty/missing input rather
        # than a model that legitimately said nothing for every single prompt. Writing an empty
        # arrow dataset here would put a valid-looking but contentless output into output/, and the
        # failure would then surface downstream (the shard merge blowing up inside pyarrow) far from
        # its cause. Fail here, where the input is still in view. See CLAUDE.md: assert non-empty
        # counts at job boundaries rather than best-effort.
        assert len(clips) > 0, (
            f"produced 0 clips from {self.in_dir.get()!r} -- refusing to write an empty dataset to "
            f"{final_dir}. Check that the input clip store is non-empty and that the shard "
            f"({self.shard}/{self.num_shards}) actually covers some of it."
        )
        monologues, traces = {}, {}
        for i in clips:
            mono = clips.sidecar(i, "monologue")
            if mono is not None:
                monologues[i] = mono
            trace = clips.sidecar(i, "trace")
            if trace is not None:
                traces[i] = trace
        write_clips(
            final_dir,
            [(i, *clips[i]) for i in clips],
            monologues=monologues,
            traces=traces,
        )
        print(f"[clips] packed {len(clips)} clips into {final_dir}", flush=True)
        shutil.rmtree(scratch, ignore_errors=True)

    def _driver_shards(self) -> list:
        """The (shard, num_shards, extra_env) per offline-driver worker this allocation runs.

        One entry -- this job's own Sisyphus shard, no pinning -- unless several GPUs are visible
        and the backend does not need one of them for a retrieval LLM, in which case each visible
        GPU gets a nested strided sub-shard and a CUDA_VISIBLE_DEVICES pin."""
        gpus = visible_gpus()
        if len(gpus) <= 1 or getattr(self, "retrieval_llm", None):
            return [(self.shard, self.num_shards, None)]
        n = len(gpus)
        return [(*nested_shard(self.shard, self.num_shards, k, n), {"CUDA_VISIBLE_DEVICES": gpus[k]}) for k in range(n)]

    def _run_fdb(self):
        assert os.path.exists(os.path.join(self.fdb_data, "candor_pause_handling/1/pause.json")), (
            f"Dataset not found at {self.fdb_data}"
        )
        files = fdb_files_for_tasks(Path(self.fdb_data.get_path()), [self.fdb_task])
        assert len(files) > 0, f"No files found for task {self.fdb_task} in dataset {self.fdb_data.get_path()}"

        out_root = Path(self.out_dir.get_path())
        items = [(inp, out_root / str(inp.parent.name) / "output.wav") for _task, inp in files]
        if self.offline_script is not None or self.offline_module is not None:
            assert self.venv_python_path is not None, "offline FDB needs venv_python_path (the model venv)"
            gpus = visible_gpus()
            if len(gpus) <= 1 or getattr(self, "retrieval_llm", None):
                manifest = write_pair_manifest(items, copy_sidecars=True)
                self._offline(python_exe=self._python_exe(), manifest=manifest)
            else:
                # Per-GPU fan-out (backlog G1): the clip pairs are strided over the visible GPUs,
                # one manifest each, every driver pinned to its card. Output paths are per clip, so
                # the workers never touch the same file; the ASR scoring below runs once over all.
                from concurrent.futures import ThreadPoolExecutor

                n = len(gpus)
                manifests = [
                    write_pair_manifest(items[k::n], copy_sidecars=True, name=f"offline_manifest.gpu{k}.json")
                    for k in range(n)
                ]

                def one(k):
                    self._offline(
                        python_exe=self._python_exe(),
                        manifest=manifests[k],
                        extra_env={"CUDA_VISIBLE_DEVICES": gpus[k]},
                    )

                with ThreadPoolExecutor(max_workers=n) as ex:
                    list(ex.map(one, range(n)))
        else:
            self._stream(
                items,
                opts=StreamOptions(resume=True, copy_sidecars=True, length_check=True, retry_sleep_s=1.0),
            )

        # Score the generated audio with the benchmark's NeMo ASR (asr.py writes output.json
        # next to each output.wav), in a dedicated CreateVenv (NeMo >=2.2).
        assert self.asr_venv_python is not None, "asr_venv_python is required for scoring"
        # Pin the recipe dir on sys.path so ``moshified_fdb_v1_v15`` (a recipe-root module)
        # imports regardless of the worker's cwd (Sisyphus RecipeFinder is cwd-relative).
        import i6_experiments as _i6e

        _recipe_dir = os.path.dirname(os.path.dirname(_i6e.__file__))
        if _recipe_dir not in sys.path:
            sys.path.insert(0, _recipe_dir)
        import moshified_fdb_v1_v15

        asr_script = os.path.join(os.path.dirname(moshified_fdb_v1_v15.__file__), "get_transcript", "asr.py")
        asr_task = (
            "user_interruption" if FDB_TASK_MAP.get(self.fdb_task, self.fdb_task) == "user_interruption" else "default"
        )
        cmd = [self.asr_venv_python.get(), asr_script, "--root_dir", self.out_dir.get_path(), "--task", asr_task]
        print("[asr]", " ".join(cmd), flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(cmd, env=env, check=True)
