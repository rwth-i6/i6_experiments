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
