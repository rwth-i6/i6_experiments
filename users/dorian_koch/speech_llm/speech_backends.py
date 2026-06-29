"""Pluggable speech-LLM backends for the benchmarks.

Both benchmarks (knowledge + Full-Duplex-Bench) operate at one clean boundary:
**input question wav -> spoken reply wav**. A backend bundles the callables needed
to drive one speech LLM across that boundary, in one of two modalities:

* **offline batched** (fast; for *causal, end-to-end* models like Moshi / PersonaPlex)
  -- a subprocess driver script that maps a dir (or manifest) of question wavs to
  reply wavs as-fast-as-the-GPU. Set ``offline_script`` (+ optional
  ``offline_extra_args`` / ``inference_venv``).
* **streaming server** (realtime; for *cascaded / served* models like Unmute, and the
  legacy Moshi websocket) -- a ``server`` context manager that brings the model up
  and yields a connection *handle*, plus a ``file_client`` class
  ``FileClient(url, inp, out, *, lead_in_s, capture_s)`` whose ``.run()`` streams one
  question wav through the handle and writes the reply, and ``ws_url`` mapping a handle
  to the client url.

A backend with ``offline_script`` set runs offline (knowledge *and* FDB); otherwise it
runs the streaming server path. The benchmark jobs take these callables **directly as
arguments** (not a string key), with Moshi's as the defaults, so a backend is just
data: pass ``MOSHI_BACKEND`` / ``unmute_backend_spec()`` / ``personaplex_backend_spec()``
into ``fdb_benchmark_py`` / ``knowledge_benchmark_py`` and the exact same job code runs
it. Adding a backend means writing its driver (offline) or server+FileClient (streaming)
and one ``BackendSpec`` here -- no job changes.

``server`` takes a *uniform* keyword signature
``(*, lora_weights, lora_config, python_exe, unmute_llm)``; each backend uses only the
kwargs it needs and ignores the rest, so ``run()`` calls every backend identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .moshi_client import MoshiFileClient, _ws_url, moshi_server


@dataclass(frozen=True)
class BackendSpec:
    name: str

    # --- streaming-server modality (Moshi server, Unmute, ...) ----------------
    # (*, lora_weights, lora_config, python_exe, unmute_llm) -> ctx-mgr yielding a handle.
    server: Callable = moshi_server
    # FileClient(url, inp, out, *, lead_in_s, capture_s); ``.run()`` writes the reply wav.
    file_client: type = MoshiFileClient
    # handle (``host:port``) -> client connection url.
    ws_url: Callable[[str], str] = _ws_url

    # --- offline-batched modality (Moshi, PersonaPlex, ...) -------------------
    # Driver script filename under ``dorian_koch/`` (e.g. ``moshi_offline_inference.py``);
    # ``None`` => this backend has no offline path and uses the streaming server above.
    offline_script: str | None = None
    # Alternative to ``offline_script``: run the driver as ``python -m <module>`` (the
    # ``moshi_family`` library drivers import the top-level ``moshi_family`` package and are
    # run with PYTHONPATH set by the harness). Mutually exclusive with ``offline_script``.
    offline_module: str | None = None
    # Extra CLI args appended to the offline driver (e.g. ``("--voice", "NATM1")``).
    offline_extra_args: tuple[str, ...] = ()
    # Lazy ``() -> tk.Path`` for the interpreter the offline driver runs under; ``None``
    # => the knowledge pipeline falls back to ``moshi_venv()``. (FDB passes its own.)
    inference_venv: Callable | None = None
    # OpenAI-compatible LLM model name to serve (via common.vllm_server) as the RAG
    # *retrieval* backend (MoshiRAG). ``None`` => no retrieval server (plain offline). When
    # set, the inference job brings up vLLM and passes its url to the offline driver.
    retrieval_llm: str | None = None
    # Optional override of the inference job's ``rqmt`` (e.g. ``{"gpu": 2, ...}`` so a
    # retrieval-LLM-backed run gets a second GPU for the vLLM server). ``None`` => default.
    rqmt_override: dict | None = None

    # --- cloud realtime API modality (Gemini Live, OpenAI Realtime) -----------
    # True => the model is remote (no GPU, but needs outbound internet). The benchmark
    # jobs then run inference as a login-node mini_task (compute nodes are air-gapped).
    # Uses the streaming-server fields above with a no-op local ``server``; see
    # cloud_realtime.py. API key is read from the env at run time (gated, not hashed).
    cloud_api: bool = False

    # --- KAME oracle modality -------------------------------------------------
    # True => the offline driver needs the sampled dataset for the per-clip oracle answer/question;
    # the knowledge pipeline threads ``data`` in as ``--oracle_dataset`` (KAME).
    needs_oracle_dataset: bool = False


# The default backend. Its callables double as the jobs' default arguments, so existing
# Moshi runs keep their exact Sisyphus hash (these defaults are hash-excluded; only a
# non-Moshi backend's callables/script enter the hash).
MOSHI_BACKEND = BackendSpec(
    name="moshi",
    server=moshi_server,
    file_client=MoshiFileClient,
    ws_url=_ws_url,
    offline_script="moshi_offline_inference.py",
)


def unmute_backend_spec() -> BackendSpec:
    """Build the Unmute backend spec (imported lazily -- optional/heavy deps).

    Streaming-only (cascaded STT->LLM->TTS): no ``offline_script``.
    """
    from .unmute_backend import UnmuteFileClient, unmute_servers, unmute_ws_url

    return BackendSpec(
        name="unmute",
        server=unmute_servers,
        file_client=UnmuteFileClient,
        ws_url=unmute_ws_url,
    )


def personaplex_backend_spec() -> BackendSpec:
    """Build the PersonaPlex backend spec.

    End-to-end + causal, so it runs through the **offline** driver
    (``personaplex_offline_inference.py``) for both benchmarks -- no websocket server.
    The model download is gated on an HF token (see ``personaplex.md``); the spec itself
    builds without it. ``inference_venv`` is resolved lazily to avoid an import cycle
    with the recipe that defines ``personaplex_venv``.
    """

    def _venv():
        from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import (
            personaplex_venv,
        )

        return personaplex_venv()

    return BackendSpec(
        name="personaplex",
        server=None,  # end-to-end + causal: no websocket server, offline path only
        offline_script="personaplex_offline_inference.py",
        inference_venv=_venv,
    )


def gemini_backend_spec() -> BackendSpec:
    """Gemini Live API as a streaming backend (cloud realtime; no local server/GPU).

    Runs through the shared streaming harness like Moshi/Unmute, but the websocket is
    Google's. Needs ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) in the job env and the
    ``cloud_venv`` (websockets); see cloud_realtime.py / personaplex.md gating notes.
    """
    from .cloud_realtime import GeminiLiveFileClient, gemini_live_server, gemini_ws_url

    return BackendSpec(
        name="gemini",
        server=gemini_live_server,
        file_client=GeminiLiveFileClient,
        ws_url=gemini_ws_url,
        cloud_api=True,
    )


def openai_backend_spec() -> BackendSpec:
    """OpenAI Realtime API as a streaming backend (cloud realtime; no local server/GPU).

    Needs ``OPENAI_API_KEY`` in the job env and the ``cloud_venv`` (websockets).
    """
    from .cloud_realtime import OpenAIRealtimeFileClient, openai_realtime_server, openai_ws_url

    return BackendSpec(
        name="openai",
        server=openai_realtime_server,
        file_client=OpenAIRealtimeFileClient,
        ws_url=openai_ws_url,
        cloud_api=True,
    )


def moshi_family_backend_spec(lora_rank: int | None = None, lora_scaling: float = 2.0) -> BackendSpec:
    """Base Moshi via the local ``moshi_family`` library (latest torch, owned code).

    Runs the offline driver ``python -m moshi_family.offline_inference`` (no external moshi
    fork). Used to validate base-Moshi fidelity through the unified library before retiring
    ``moshi_venv`` (see ``moshi_family.md``). ``inference_venv`` is the one-venv
    ``moshi_family_venv`` (torch 2.12.1+cu126), resolved lazily to dodge an import cycle.

    ``lora_rank``: when evaluating a Moshi LoRA finetune (``MOSHI_LIB_ADAPTER``), pass the trained
    rank so the driver wraps the linears with ``LoraConfig(rank, scaling)`` BEFORE loading the
    ``--overlay`` adapter (else the adapter keys land on non-existent LoRALinears). Must match the
    rank the launcher trained with. ``None`` (base / partial-state overlay) wraps nothing.
    """

    def _venv():
        from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import (
            moshi_family_venv,
        )

        return moshi_family_venv()

    extra: tuple[str, ...] = ()
    if lora_rank is not None:
        extra = ("--lora_rank", str(lora_rank), "--lora_scaling", str(lora_scaling))

    return BackendSpec(
        name="moshi_family" if lora_rank is None else f"moshi_family_lora{lora_rank}",
        server=None,  # end-to-end + causal: offline driver, no websocket server
        offline_module="moshi_family.offline_inference",
        offline_extra_args=extra,
        inference_venv=_venv,
    )


def personaplex_family_backend_spec() -> BackendSpec:
    """PersonaPlex via the local ``moshi_family.personaplex`` sub-package (latest torch, owned code).

    Runs ``python -m moshi_family.personaplex.offline_inference`` through the same ``offline_module``
    seam as base-Moshi (no external personaplex fork). Used to validate PersonaPlex fidelity
    (21.2%/1.66 + FDB 4.42) through the unified library before retiring ``personaplex_venv`` (see
    ``moshi_family.md``). ``inference_venv`` is the one-venv ``moshi_family_venv`` (torch 2.12.1+cu126),
    resolved lazily to dodge an import cycle.
    """

    def _venv():
        from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import (
            moshi_family_venv,
        )

        return moshi_family_venv()

    return BackendSpec(
        name="personaplex_family",
        server=None,  # end-to-end + causal: offline driver, no websocket server
        offline_module="moshi_family.personaplex.offline_inference",
        inference_venv=_venv,
    )


def moshirag_family_backend_spec(
    retrieval_llm: str = "google/gemma-4-31B-it", lora_rank: int | None = None, lora_scaling: float = 2.0
) -> BackendSpec:
    """MoshiRAG via the local ``moshi_family.moshirag`` sub-package (latest torch, owned code).

    Runs ``python -m moshi_family.moshirag.offline_inference`` through the ``offline_module`` seam
    (no external moshi-rag fork). The driver co-launches the ARC reference-encoder server; the job
    brings up our vLLM ``retrieval_llm`` (2 GPUs). Fidelity confirmed (50.5%/2.715 vs fork
    48.0%/2.645) and the moshirag fork venv is retired (see ``moshi_family.md``). ``inference_venv``
    is the one-venv ``moshi_family_venv`` (torch 2.12.1+cu126 + xformers), resolved lazily to dodge
    an import cycle.
    """

    def _venv():
        from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import (
            moshi_family_venv,
        )

        return moshi_family_venv()

    # When evaluating a MoshiRAG LoRA finetune (MOSHIRAG_LIB_ADAPTER), pass the trained rank so the
    # driver wraps the trunk with a matching LoRA before loading the --overlay adapter (else the keys
    # land on non-existent LoRALinears). None = base released model.
    extra = ("--lora_rank", str(lora_rank), "--lora_scaling", str(lora_scaling)) if lora_rank is not None else ()
    return BackendSpec(
        name="moshirag_family" if lora_rank is None else f"moshirag_family_lora{lora_rank}",
        server=None,  # end-to-end + causal: offline driver + co-launched retrieval, no websocket
        offline_module="moshi_family.moshirag.offline_inference",
        offline_extra_args=extra,
        inference_venv=_venv,
        retrieval_llm=retrieval_llm,
        # ~14s/clip * 1000 clips ~= 4h; serial retrieval pump cannot resume, so a 3h cap (the old
        # deadlock-era guard, task #2) timed out + looped forever. The deadlock is fixed (task #3).
        rqmt_override={"gpu": 2, "cpu": 6, "mem": 48, "time": 8},
    )


def kame_family_backend_spec(hf_repo: str = "SakanaAI/kame", inject_at_s: float | None = None) -> BackendSpec:
    """KAME via the local moshi_family lib (oracle text stream; see kame_engine.py + projects/kame.md).

    Runs ``python -m moshi_family.kame_offline_inference``; single-shard B=1 (oracle is per-clip). The
    driver reads the sampled dataset (``--oracle_dataset``, threaded by the knowledge pipeline) for the
    gt answer injected into the oracle stream -- the KAME oracle ceiling, no backend LLM."""

    def _venv():
        from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import moshi_family_venv

        return moshi_family_venv()

    extra: tuple[str, ...] = ("--hf_repo", hf_repo)
    if inject_at_s is not None:
        extra += ("--inject_at_s", str(inject_at_s))
    return BackendSpec(
        name="kame_family",
        server=None,
        offline_module="moshi_family.kame_offline_inference",
        offline_extra_args=extra,
        inference_venv=_venv,
        needs_oracle_dataset=True,
        rqmt_override={"gpu": 1, "cpu": 6, "mem": 48, "time": 8},
    )


def flm_audio_backend_spec(hf_repo: str = "CofeAI/FLM-Audio") -> BackendSpec:
    """FLM-Audio (arXiv 2509.02521): a native full-duplex ~7B speech chatbot with its own FLM
    backbone over the Mimi codec, vendored at ``speech_llm.full_duplex.flmaudio`` for *inference
    only* (see ``flm_audio.md``).

    Runs ``python -m flmaudio.offline_inference`` through the ``offline_module`` seam -- a B=1
    synchronous step loop (Mimi.encode -> LMGen.step -> Mimi.decode), like base Moshi. It is a
    FOREIGN backbone (NOT a moshi_family member): its modeling pins ``transformers==4.51.3``, so
    it runs in its own isolated ``flm_audio_venv`` instead of the moshi_family venv. End-to-end +
    causal -> offline driver, no websocket server. ``inference_venv`` is resolved lazily to dodge
    an import cycle with the recipe that defines ``flm_audio_venv``.
    """

    def _venv():
        from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import flm_audio_venv

        return flm_audio_venv()

    return BackendSpec(
        name="flm_audio",
        server=None,  # end-to-end + causal: offline driver, no websocket server
        offline_module="flmaudio.offline_inference",
        offline_extra_args=("--hf_repo", hf_repo),
        inference_venv=_venv,
        rqmt_override={"gpu": 1, "cpu": 6, "mem": 48, "time": 8},
    )
