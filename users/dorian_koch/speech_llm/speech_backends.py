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


def moshirag_backend_spec(retrieval_llm: str = "google/gemma-4-31B-it") -> BackendSpec:
    """MoshiRAG (kyutai-labs/moshi-rag, arXiv 2604.12928) as an offline backend.

    A full-duplex Moshi variant that emits a ``<ret>`` token to asynchronously retrieve a
    reference document and fold it into its reply (ARC-Encoder conditioner). It runs through
    the **offline** driver (``moshirag_offline_inference.py``) like Moshi/PersonaPlex, but the
    driver co-launches a reference-encoder GPU server and the job brings up our existing vLLM
    ``retrieval_llm`` as the text-in/text-out retrieval backend (the paper's own setup). That
    second model wants its own GPU, so ``rqmt_override`` bumps the job to 2 GPUs.

    GATED: stage ``kyutai/moshika-rag-pytorch-bf16`` into the HF cache first (CC-BY; see
    ``moshirag.md``). ``inference_venv`` is resolved lazily to avoid an import cycle with the
    recipe that defines ``moshirag_venv``.
    """

    def _venv():
        from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import (
            moshirag_venv,
        )

        return moshirag_venv()

    return BackendSpec(
        name="moshirag",
        server=None,  # end-to-end + causal: offline driver, no websocket server
        offline_script="moshirag_offline_inference.py",
        inference_venv=_venv,
        retrieval_llm=retrieval_llm,
        rqmt_override={"gpu": 2, "cpu": 6, "mem": 48, "time": 12},
    )
