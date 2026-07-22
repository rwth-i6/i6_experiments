"""Shared model-access harness for both benchmarks (knowledge + Full-Duplex-Bench).

Both benchmarks reduce to the *same* operation: given a list of
``(input_wav -> output_wav)`` pairs and a backend, produce one reply wav per
input -- either by

* **streaming** each clip through a realtime server (a ``server`` context manager +
  a ``FileClient`` honouring ``(url, inp, out, *, lead_in_s, capture_s).run()``), or
* running an **offline** driver script once over the whole set.

This module owns that one access pattern so ``MoshiInference`` (knowledge) and
``FullDuplexBenchEval_Inference`` (FDB) share identical model-access code; the jobs
differ only in how they *enumerate* the pairs and what they do downstream (ASR,
grading, ...). These are **pure runtime helpers** -- nothing here is hashed by
Sisyphus, so routing both jobs through this module does not change any job hash.

Backends plug in via the same callables a ``BackendSpec`` carries (``server``,
``file_client``, ``ws_url``, ``offline_script``); see ``speech_backends.py``. A cloud
realtime API (Gemini / OpenAI) is just a streaming backend whose ``server`` launches
nothing local and whose ``file_client`` talks to the remote websocket.
"""

from __future__ import annotations

import os
import json
import time
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from glob import glob
from typing import Callable, List, Sequence, Tuple


# FDB task helpers (light; defined here in the dependency-free harness so the unified
# SpeechInference can use them WITHOUT importing fdb.py, whose top-level imports pull in
# i6_core -- absent in the inference worker venv).
FDB_TASK_MAP = {
    "candor_pause_handling": "pause_handling",
    "candor_turn_taking": "smooth_turn_taking",
    "icc_backchannel": "backchannel",
    "synthetic_pause_handling": "pause_handling",
    "synthetic_user_interruption": "user_interruption",
}


def fdb_files_for_tasks(ds_path: Path, tasks: Sequence[str]) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for t in tasks:
        pattern = ds_path / f"{t}/*/input.wav"
        files += [(t, Path(p)) for p in sorted(glob(str(pattern)))]
    return files


def _moshi_family_pythonpath() -> str:
    """Dir to put on PYTHONPATH so ``python -m moshi_family.<driver>`` resolves the local
    library (its drivers import the top-level package name ``moshi_family``, which lives at
    ``recipe/speech_llm/full_duplex/moshi_family``). Derived from THIS file's location, not by
    importing ``speech_llm`` -- the recipe top-level is not importable by that name in a Sisyphus
    *worker* (jobs are imported as ``i6_experiments.users.dorian_koch.speech_llm...``, so only
    ``i6_experiments`` is on the worker's sys.path; ``import speech_llm`` raises ModuleNotFound)."""
    # Walk UP from this file's (unresolved) absolute path to the recipe root that holds
    # ``speech_llm/full_duplex/moshi_family``. Do NOT ``.resolve()``: ``i6_experiments`` is a
    # symlink into a shared checkout, so resolving leaves the recipe tree; the ``recipe/`` dir
    # (which also carries ``speech_llm/``) is only on the *unresolved* path.
    here = Path(os.path.abspath(__file__))
    for parent in here.parents:
        cand = parent / "speech_llm" / "full_duplex"
        if (cand / "moshi_family").is_dir():
            return str(cand)
    raise RuntimeError(f"could not locate speech_llm/full_duplex/moshi_family above {here}")


@dataclass
class StreamOptions:
    """Knobs for the streaming loop. Defaults match the knowledge benchmark; FDB
    flips ``resume``/``copy_sidecars``/``length_check`` on and uses the client's
    legacy capture mode (``lead_in_s=0``, ``capture_s=None``)."""

    lead_in_s: float = 0.0
    capture_s: float | None = None
    retries: int = 3
    retry_sleep_s: float = 0.0
    resume: bool = False  # skip a clip whose output.wav already exists (non-empty)
    copy_sidecars: bool = False  # copy inp.parent/*.json next to each output
    length_check: bool = False  # warn if reply << input (alignment regression guard)
    progress_every: int = 50


def stream_inference(
    *,
    server: Callable,
    file_client: type,
    ws_url: Callable[[str], str],
    items: Sequence[tuple[Path, Path]],
    server_kwargs: dict,
    opts: StreamOptions,
) -> None:
    """Bring up ``server(**server_kwargs)`` once and stream every ``(inp, out)`` pair
    through ``file_client``. Backend-agnostic: the server ctx-mgr takes the uniform
    ``(*, lora_weights, lora_config, python_exe, unmute_llm)`` signature and ignores
    what it does not use; cloud-API backends launch nothing and just yield a handle."""
    n = len(items)
    with server(**server_kwargs) as handle:
        url = ws_url(handle)
        server_name = getattr(server, "__name__", "server")
        print(f"[{server_name}] streaming {n} clips at {url}", flush=True)
        for i, (inp, out) in enumerate(items):
            inp, out = Path(inp), Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            if opts.resume and out.exists() and out.stat().st_size > 0:
                print("[SKIP]", out.parent.name, inp.name, "(already done)", flush=True)
                continue
            for attempt in range(opts.retries):
                try:
                    file_client(
                        url,
                        inp,
                        out,
                        lead_in_s=opts.lead_in_s,
                        capture_s=opts.capture_s,
                    ).run()
                    break
                except Exception as e:
                    print(f"[RETRY {attempt + 1}/{opts.retries}] {inp.name}: {e}", flush=True)
                    if opts.retry_sleep_s:
                        time.sleep(opts.retry_sleep_s)
            if opts.length_check:
                _warn_if_truncated(inp, out)
            if opts.copy_sidecars:
                for json_file in inp.parent.glob("*.json"):
                    shutil.copy(json_file, out.parent / json_file.name)
            if opts.progress_every and (i + 1) % opts.progress_every == 0:
                print(f"[inference] {i + 1}/{n} clips done", flush=True)


def _warn_if_truncated(inp: Path, out: Path) -> None:
    """A served full-duplex reply must span (roughly) the whole conversation timeline,
    not just the spoken segment -- otherwise FDB latency is measured from the wrong
    zero (the Unmute output-alignment bug). Warn loudly if the reply is far short."""
    try:
        import soundfile as _sf

        olen = _sf.info(str(out)).frames / _sf.info(str(out)).samplerate
        ilen = _sf.info(str(inp)).frames / _sf.info(str(inp)).samplerate
        if olen < 0.5 * ilen:
            print(
                f"[WARN] {out.parent.name}: reply {olen:.2f}s << input {ilen:.2f}s -- output may be "
                "truncated to the spoken segment (breaks latency alignment)",
                flush=True,
            )
    except Exception as e:
        print(f"[WARN] could not length-check {out}: {e}", flush=True)


def write_pair_manifest(items: Sequence[tuple[Path, Path]], *, copy_sidecars: bool = False) -> str:
    """Write a JSON manifest of ``[[in_wav, out_wav], ...]`` for an offline driver and
    (optionally) copy each input's sidecar ``*.json`` next to its output. Returns the
    manifest path (in the job's cwd)."""
    pairs = []
    for inp, out in items:
        inp, out = Path(inp), Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pairs.append([str(inp), str(out)])
        if copy_sidecars:
            for json_file in inp.parent.glob("*.json"):
                shutil.copy(json_file, out.parent / json_file.name)
    manifest = os.path.join(os.getcwd(), "offline_manifest.json")
    with open(manifest, "w") as f:
        json.dump(pairs, f)
    return manifest


def run_offline_driver(
    *,
    python_exe: str,
    script_path: str | None = None,
    module: str | None = None,
    pythonpath: str | None = None,
    in_dir: str | None = None,
    out_dir: str | None = None,
    manifest: str | None = None,
    lead_in_s: float | None = None,
    capture_s: float | None = None,
    batch_size: int | None = None,
    shard: int | None = None,
    num_shards: int | None = None,
    oracle_dataset: str | None = None,
    lora_weights: str | None = None,
    lora_config: str | None = None,
    extra_args: Sequence[str] = (),
    extra_env: dict | None = None,
) -> None:
    """Build + run an offline inference driver as a subprocess. Supports both the
    knowledge dir-mode (``--in_dir/--out_dir/--batch_size/--shard``) and the FDB
    manifest-mode (``--manifest``); the caller passes whichever set its driver wants.
    argparse is order-independent, so the two callers share one builder. ``extra_env``
    is merged into the subprocess env (e.g. ``CUDA_VISIBLE_DEVICES`` to pin a
    retrieval-LLM-backed driver to a different GPU than the co-running vLLM server)."""
    if module is not None:
        cmd = [python_exe, "-m", module]
    else:
        assert script_path is not None, "run_offline_driver needs script_path or module"
        cmd = [python_exe, str(script_path)]
    if manifest is not None:
        cmd += ["--manifest", manifest]
    if in_dir is not None:
        cmd += ["--in_dir", str(in_dir)]
    if out_dir is not None:
        cmd += ["--out_dir", str(out_dir)]
    if lead_in_s is not None:
        cmd += ["--lead_in_s", str(lead_in_s)]
    if capture_s is not None:
        cmd += ["--capture_s", str(capture_s)]
    if batch_size is not None:
        cmd += ["--batch_size", str(batch_size)]
    if lora_weights is not None:
        if module is not None:
            # Lib (moshi_family) drivers expose ONE uniform --overlay (a partial or full state-dict),
            # covering both a LoRA adapter and the PersonaPlex trained_heads. The fork drivers instead
            # take the --lora_weights/--lora_config pair. A LoRA adapter's config.json sits beside its
            # lora.safetensors in the resolved overlay dir, so the lib driver finds it as a sibling.
            cmd += ["--overlay", str(lora_weights)]
        else:
            cmd += ["--lora_weights", str(lora_weights)]
            if lora_config is not None:
                cmd += ["--lora_config", str(lora_config)]
    if shard is not None and num_shards is not None:
        cmd += ["--shard", str(shard), "--num_shards", str(num_shards)]
    if oracle_dataset is not None:
        cmd += ["--oracle_dataset", str(oracle_dataset)]
    cmd += list(extra_args)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if pythonpath:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = pythonpath + ((os.pathsep + existing) if existing else "")
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    print("[offline]", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def run_with_optional_retrieval(job, drive) -> None:
    """Run an offline driver, optionally behind a RAG retrieval LLM (MoshiRAG seam).

    ``drive(extra_args, extra_env)`` is a callback that invokes ``run_offline_driver`` for
    the concrete benchmark (knowledge dir-mode or FDB manifest-mode). If ``job.retrieval_llm``
    is set, this brings up our existing ``common.vllm_server`` (in the *job* process's venv,
    which has vLLM) on the first GPU and runs the driver -- pinned to the second GPU via
    ``CUDA_VISIBLE_DEVICES`` -- passing the server url through ``--llm_base_url`` /
    ``--llm_model_name``. The job must request >=2 GPUs for this (see ``rqmt_override`` on the
    backend spec). Plain offline backends (Moshi/PersonaPlex) leave ``retrieval_llm`` None and
    just run the driver once. Keeping this here (not in the jobs) means knowledge + FDB share
    the exact same retrieval wiring."""
    retrieval_llm = getattr(job, "retrieval_llm", None)
    if not retrieval_llm:
        drive((), None)
        return
    from .common import vllm_server

    # GPU split: vLLM retrieval server on the first visible GPU, the offline driver (the
    # speech model + its reference encoder) on the last one. VERIFY(moshirag): tune for the
    # actual GPU memory once the model is staged (gemma-4-31B may want >1 GPU of its own).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    driver_dev = visible[-1] if len(visible) > 1 else visible[0]
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[0]
    try:
        with vllm_server(retrieval_llm) as llm_url:
            drive(
                ("--llm_base_url", llm_url, "--llm_model_name", retrieval_llm),
                {"CUDA_VISIBLE_DEVICES": driver_dev},
            )
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved


# ---------------------------------------------------------------------------
# Shared run-body glue for the two backend-driven inference jobs
# (knowledge MoshiInference + FDB FullDuplexBenchEval_Inference).
# ---------------------------------------------------------------------------


class BackendInferenceMixin:
    """Runtime-only mixin factoring the backend access shared by both inference jobs.

    Nothing here enters a Sisyphus hash. The two jobs keep their own
    ``__init__``/``hash``/``__sis_hash_exclude__`` and differ only in attribute
    names and output layout, which they bridge via the two hooks below.
    """

    # --- hooks: attribute-name indirection between the two jobs ----------------
    def _server_callable(self):
        """The backend's server context manager (``self.server`` vs ``self.model``)."""
        raise NotImplementedError

    def _python_exe(self):
        """Interpreter the server/driver runs under, or None for the worker's own."""
        raise NotImplementedError

    # --- shared helpers --------------------------------------------------------
    def _server_kwargs(self) -> dict:
        return dict(
            lora_weights=self.lora_weights.get_path() if self.lora_weights is not None else None,
            lora_config=self.lora_config.get_path() if self.lora_config is not None else None,
            python_exe=self._python_exe(),
            unmute_llm=self.unmute_llm,
        )

    def _stream(self, items, *, opts: "StreamOptions") -> None:
        stream_inference(
            server=self._server_callable(),
            file_client=self.file_client,
            ws_url=self.ws_url,
            items=items,
            server_kwargs=self._server_kwargs(),
            opts=opts,
        )

    def _offline(self, *, python_exe, **driver_kwargs) -> None:
        """Run the backend's offline driver (optionally behind a RAG retrieval LLM).

        ``driver_kwargs`` are forwarded to ``run_offline_driver`` (dir-mode for the
        knowledge benchmark, manifest-mode for FDB); LoRA paths and the backend's
        extra args are spliced in here so both callers stay one line."""
        module = getattr(self, "offline_module", None)
        script_path = None if module else Path(__file__).resolve().parent.parent / self.offline_script
        pythonpath = _moshi_family_pythonpath() if module else None
        lora_weights = self.lora_weights.get() if self.lora_weights is not None else None
        lora_config = self.lora_config.get() if self.lora_config is not None else None

        def _drive(extra_args, extra_env=None):
            run_offline_driver(
                python_exe=python_exe,
                script_path=script_path,
                module=module,
                pythonpath=pythonpath,
                lora_weights=lora_weights,
                lora_config=lora_config,
                extra_args=tuple(self.offline_extra_args) + tuple(extra_args),
                extra_env=extra_env,
                **driver_kwargs,
            )

        run_with_optional_retrieval(self, _drive)
