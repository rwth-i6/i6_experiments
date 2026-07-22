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
import subprocess
import sys
from pathlib import Path

from sisyphus import Job, Task, tk

from .inference_harness import (
    BackendInferenceMixin,
    StreamOptions,
    write_pair_manifest,
    fdb_files_for_tasks,
    FDB_TASK_MAP,
)
from .moshi_client import MoshiFileClient, _ws_url, moshi_server


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
        assert overlay_kind in ("lora", "personaplex_heads"), overlay_kind
        self.run_dir = run_dir
        self.overlay_kind = overlay_kind
        self.step = step  # None -> latest (lora) / final consolidated (personaplex)
        self.out_weights = self.output_path(
            "lora.safetensors" if overlay_kind == "lora" else "trained_heads.safetensors"
        )
        self.out_config = self.output_path("config.json") if overlay_kind == "lora" else None

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        if self.overlay_kind == "lora":
            ckpt_root = Path(self.run_dir.get()) / "checkpoints"
            ckpts = sorted(ckpt_root.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[-1]))
            assert ckpts, f"No checkpoints found in {ckpt_root}"
            if self.step is not None:
                chosen = ckpt_root / f"checkpoint_{self.step:06d}"
                assert chosen.exists(), f"{chosen} not found; have {[c.name for c in ckpts]}"
            else:
                chosen = ckpts[-1]
            consolidated = chosen / "consolidated"
            print(f"Resolved LoRA checkpoint: {chosen.name}", flush=True)
            os.symlink(consolidated / "config.json", self.out_config.get())
            os.symlink(consolidated / "lora.safetensors", self.out_weights.get())
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
            os.symlink(chosen, self.out_weights.get())


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
        # FDB-only attrs are absent (None) for knowledge jobs and vice-versa, so excluding their
        # "absent" default keeps each mode's hash clean.
        "asr_venv_python": None,
        "oracle_dataset": None,
        # Hashed code-version knob: bump to force a fresh hash (and cascade re-runs to downstream
        # transcription/grading/eval) after a lib-code fix that is NOT part of the hash. Excluded
        # at the default (1) so it is a no-op until bumped.
        "code_version": 1,
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
        code_version: int = 1,
        # --- knowledge mode ---
        in_dir: tk.Path | None = None,
        shard: int | None = None,
        num_shards: int | None = None,
        lead_in_s: float = 2.0,
        capture_s: float = 24.0,
        batch_size: int = 32,
        oracle_dataset: tk.Path | None = None,
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

    def _shard_total(self) -> "int | None":
        cached = getattr(self, "_total_cache", 0)
        if cached:
            return cached
        n = self._count_wavs(self.in_dir.get_path())
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
            return max(0.0, min(1.0, self._count_wavs(self.out_dir.get_path()) / total)) if total else None
        except Exception:
            return None

    def info(self):
        if self.mode != "knowledge":
            return None
        try:
            total = self._shard_total()
            return f"{self._count_wavs(self.out_dir.get_path())}/{total} clips" if total else None
        except Exception:
            return None

    # --- run ---
    def run(self):
        if self.mode == "knowledge":
            self._run_knowledge()
        else:
            self._run_fdb()

    def _run_knowledge(self):
        out_dir = Path(self.out_dir.get())
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.offline_script is not None or self.offline_module is not None:
            self._offline(
                python_exe=self._python_exe(),
                in_dir=str(self.in_dir.get()),
                out_dir=str(out_dir),
                lead_in_s=self.lead_in_s,
                capture_s=self.capture_s,
                batch_size=self.batch_size,
                shard=self.shard,
                num_shards=self.num_shards,
                oracle_dataset=(self.oracle_dataset.get() if self.oracle_dataset is not None else None),
            )
            return
        wav_files = sorted(Path(self.in_dir.get()).glob("*.wav"))
        if self.shard is not None and self.num_shards is not None:
            wav_files = wav_files[self.shard :: self.num_shards]
        items = [(wav, out_dir / wav.name) for wav in wav_files]
        self._stream(items, opts=StreamOptions(lead_in_s=self.lead_in_s, capture_s=self.capture_s, progress_every=50))

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
            manifest = write_pair_manifest(items, copy_sidecars=True)
            self._offline(python_exe=self._python_exe(), manifest=manifest)
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
