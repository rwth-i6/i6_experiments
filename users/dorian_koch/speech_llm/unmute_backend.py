"""Self-hosted Unmute (kyutai cascaded speech LLM) backend for the benchmarks.

Unmute is a *cascade*: Kyutai STT -> a text LLM (served by vLLM) -> Kyutai TTS,
orchestrated by a Python FastAPI backend that speaks an OpenAI-Realtime websocket.
Unlike Moshi (a single end-to-end audio model), it is four cooperating services.

This module provides the same ``(server ctx-mgr, FileClient, ws_url)`` triple Moshi
exposes in ``moshi_client.py`` (see ``speech_backends.py``), so both benchmarks can
drive Unmute through the unchanged wav -> reply-wav boundary:

* ``unmute_servers`` -- context manager that brings up STT + TTS (prebuilt Rust
  ``moshi-server`` Apptainer image) + vLLM(``unmute_llm``) + the Unmute FastAPI
  backend inside one SLURM job, wired by ``KYUTAI_*_URL`` env, and yields the
  backend's ``host:port``.
* ``unmute_ws_url`` -- turn that address into the ``/v1/realtime`` ws endpoint.
* ``UnmuteFileClient`` -- stream one wav through the backend over the OpenAI-Realtime
  protocol and write the spoken reply, mirroring ``MoshiFileClient``'s lead-in +
  silence-timeout capture so downstream Whisper/grading/FDB scoring is identical.

The Rust STT/TTS workers run from ``moshi-server.sif``; see
``projects/2026-01-speech-llm.md`` ("Unmute integration") for how the image is built.
"""

from __future__ import annotations

import base64
import json
import os
import random
import signal
import socket
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

# Unmute streams the *same* 24 kHz mono audio in 80 ms Opus frames as Moshi
# (kyutai_constants.py: SAMPLE_RATE=24000, SAMPLES_PER_FRAME=1920), so we reuse
# Moshi's framing/encoding helpers verbatim.
from .moshi_client import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    _SPEECH_THRESHOLD,
    _iter_frames,
    _load_prompt_24k,
    _pcm_to_int16,
)

# Unmute's upstream default brain; surfaced as a swappable benchmark parameter.
DEFAULT_UNMUTE_LLM = "google/gemma-3-1b-it"

# Prebuilt Rust moshi-server (STT + TTS workers) Apptainer image. One inode; baked
# from kyutai-labs/unmute's public.Dockerfile. See projects/2026-01-speech-llm.md.
MOSHI_SERVER_SIF = "/hpcwork/tt201262/unmute/moshi-server.sif"
# Configs baked into the SIF at /opt/unmute/configs (override per-instance below).
SIF_STT_CONFIG = "/opt/unmute/configs/stt.toml"
SIF_TTS_CONFIG = "/opt/unmute/configs/tts.toml"

# Shared HF cache the workers/vLLM pull model weights from (compute nodes have net).
HF_HOME = "/hpcwork/p0023999/common_hf_home"

# Unmute FastAPI backend checkout (uv-synced from its lockfile); its .venv python runs
# `uvicorn unmute.main_websocket:app`, and the dir is the backend's cwd.
UNMUTE_REPO = "/hpcwork/tt201262/unmute/unmute-repo"
UNMUTE_BACKEND_PYTHON = f"{UNMUTE_REPO}/.venv/bin/python"
# Dedicated venv with vLLM (matches unmute's pinned vllm/vllm-openai:v0.11.0).
UNMUTE_VLLM_PYTHON = "/hpcwork/tt201262/unmute/venvs/vllm/bin/python"


def unmute_ws_url(addr: str) -> str:
    """Return the OpenAI-Realtime ``/v1/realtime`` ws URL for a backend address."""
    if "://" in addr:
        scheme, _, rest = addr.partition("://")
        ws_scheme = "wss" if scheme in ("https", "wss") else "ws"
        return f"{ws_scheme}://{rest.rstrip('/')}/v1/realtime"
    return f"ws://{addr}/v1/realtime"


def _free_port(base: int) -> int:
    """Pick a free localhost TCP port near ``base`` (job-id-seeded, like moshi_server)."""
    if "SLURM_JOB_ID" in os.environ:
        port = base + (int(os.environ["SLURM_JOB_ID"]) % 1000)
    else:
        port = base + random.randint(0, 999)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        for _ in range(200):
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    return port


def _wait_port(port: int, proc: subprocess.Popen, name: str, max_wait: float = 15 * 60):
    """Block until ``port`` accepts connections, or raise if ``proc`` dies first."""
    start = time.time()
    while time.time() - start < max_wait:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} died during startup (exit {proc.returncode})")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            print(f"{name} is accepting connections on :{port}")
            return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    raise TimeoutError(f"{name} not ready on :{port} after {max_wait}s")


def _pipe_output(proc: subprocess.Popen, name: str):
    """Forward a child's merged stdout to our log with a short prefix."""

    def reader():
        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                if line:
                    print(f"[{name}] {line.rstrip()}", flush=True)

    threading.Thread(target=reader, daemon=True).start()


def _terminate(proc: subprocess.Popen | None):
    if proc is None or proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=10)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass


def _moshi_worker(config: str, port: int, env: dict) -> subprocess.Popen:
    """Launch one Rust moshi-server worker (STT or TTS) from the SIF on ``port``.

    moshi-server's ``worker`` subcommand takes ``--addr``/``--port``/``--config`` (verified
    against the pinned source), so we override the port on the CLI and point ``--config`` at
    the toml baked into the SIF at /opt/unmute/configs -- no per-instance config copy needed.
    """
    cmd = [
        "apptainer",
        "run",
        "--nv",
        "--bind",
        f"{HF_HOME}:{HF_HOME}",
        MOSHI_SERVER_SIF,
        "worker",
        "--config",
        str(config),
        "--port",
        str(port),
    ]
    print(f"Starting moshi-server worker: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        env=env,
    )


@contextmanager
def unmute_servers(
    *,
    lora_weights=None,
    lora_config=None,
    python_exe: str | None = None,
    unmute_llm: str | None = None,
    sif: str | None = None,
    llm_python: str | None = None,
    backend_python: str | None = None,
    backend_cwd: str | None = None,
    **_,
):
    """Bring up the full Unmute cascade in one job; yield the backend ``host:port``.

    Uniform backend signature (see speech_backends.py): ``lora_*`` are Moshi-only and
    ignored here. ``unmute_llm`` selects the served text LLM (default Gemma-3-1B).
    ``sif`` / ``llm_python`` / ``backend_python`` override the moshi-server image and
    the vLLM / Unmute-backend interpreters (defaults: the module SIF + ``python_exe``).
    """
    global MOSHI_SERVER_SIF
    if sif:
        MOSHI_SERVER_SIF = sif
    llm = unmute_llm or DEFAULT_UNMUTE_LLM
    llm_py = llm_python or UNMUTE_VLLM_PYTHON
    backend_py = backend_python or UNMUTE_BACKEND_PYTHON

    base_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": HF_HOME,
        "HUGGING_FACE_HUB_HOME": HF_HOME,
        # HF's Xet token endpoint rate-limits (429) when the TTS worker pulls the many
        # voice files anonymously; plain LFS download avoids that endpoint entirely.
        "HF_HUB_DISABLE_XET": "1",
    }
    # The STT/TTS models + voices are pre-staged in the shared cache, so the Rust workers
    # read cache-only (no HF network calls -> no 429 / rate-limit flakiness at boot).
    # NO_TORCH_COMPILE: the TTS "Py" module's torch.compile/Inductor CUDA codegen needs a
    # host C++ compiler and picks the absent `icx` inside the SIF; eager mode avoids it.
    worker_env = {**base_env, "HF_HUB_OFFLINE": "1", "NO_TORCH_COMPILE": "1"}

    stt_port = _free_port(8080)
    tts_port = _free_port(8180)
    llm_port = _free_port(18998)
    backend_port = _free_port(28998)

    with ExitStack() as stack:
        # 1. STT + TTS Rust workers (from the SIF), each on its own port, using the
        #    configs baked into the SIF (port overridden on the CLI).
        stt = _moshi_worker(SIF_STT_CONFIG, stt_port, worker_env)
        stack.callback(_terminate, stt)
        _pipe_output(stt, "stt")
        tts = _moshi_worker(SIF_TTS_CONFIG, tts_port, worker_env)
        stack.callback(_terminate, tts)
        _pipe_output(tts, "tts")

        # 2. vLLM serving the text LLM (OpenAI-compatible) on llm_port.
        llm_cmd = [
            llm_py,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(llm_port),
            "--model",
            llm,
            "--max-model-len",
            "1536",
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            "0.4",
        ]
        print(f"Starting vLLM: {' '.join(llm_cmd)}")
        llm_proc = subprocess.Popen(
            llm_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            env=base_env,
        )
        stack.callback(_terminate, llm_proc)
        _pipe_output(llm_proc, "llm")

        _wait_port(stt_port, stt, "moshi-server STT")
        _wait_port(tts_port, tts, "moshi-server TTS")
        _wait_port(llm_port, llm_proc, "vLLM")

        # 3. Unmute FastAPI backend, wired to the three services by env.
        backend_env = {
            **base_env,
            "KYUTAI_STT_URL": f"ws://127.0.0.1:{stt_port}",
            "KYUTAI_TTS_URL": f"ws://127.0.0.1:{tts_port}",
            "KYUTAI_LLM_URL": f"http://127.0.0.1:{llm_port}",
            "KYUTAI_LLM_MODEL": llm,
        }
        backend_cmd = [
            backend_py,
            "-m",
            "uvicorn",
            "unmute.main_websocket:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(backend_port),
            "--ws-per-message-deflate=false",
        ]
        print(f"Starting Unmute backend: {' '.join(backend_cmd)}")
        backend = subprocess.Popen(
            backend_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            env=backend_env,
            # Run from the unmute repo so the backend finds its relative resources.
            cwd=backend_cwd or UNMUTE_REPO,
        )
        stack.callback(_terminate, backend)
        _pipe_output(backend, "backend")
        _wait_port(backend_port, backend, "Unmute backend")

        print("Unmute cascade ready")
        yield f"127.0.0.1:{backend_port}"


# --- OpenAI-Realtime file client -------------------------------------------------

# Client -> server and server -> client event ``type`` strings we use, from Unmute's
# unmute/openai_realtime_api_events.py.
_EV_SESSION_UPDATE = "session.update"
_EV_AUDIO_APPEND = "input_audio_buffer.append"
_EV_AUDIO_DELTA = "response.audio.delta"
_EV_AUDIO_DONE = "response.audio.done"
_EV_TEXT_DELTA = "response.text.delta"


class UnmuteFileClient:
    """Run one wav file through the Unmute backend over its OpenAI-Realtime websocket.

    Drop-in for ``MoshiFileClient``: same ``(ws_url, inp, out, *, lead_in_s,
    capture_s, silence_timeout_s)`` constructor and ``.run()`` contract. The prompt is
    streamed as base64 Opus ``input_audio_buffer.append`` events; the spoken reply
    arrives as base64 Opus ``response.audio.delta`` events, decoded and written with
    the same lead-in + silence-timeout capture as MoshiFileClient so the output wav is
    interchangeable for downstream scoring.
    """

    def __init__(
        self,
        ws_url: str,
        inp,
        out,
        *,
        lead_in_s: float = 0.0,
        capture_s: float | None = None,
        silence_timeout_s: float = 2.0,
        voice: str | None = None,
    ):
        import sphn

        self._url = str(ws_url)
        self._inp = Path(inp)
        self._out = Path(out)
        self._voice = voice

        self._prompt = _load_prompt_24k(self._inp)
        self._prompt_len = int(len(self._prompt))
        self._reply_done = False

        # Default capture window: lead-in greeting + prompt + a generous tail so the
        # whole reply fits even when capture_s isn't given by the caller.
        self._silence_timeout = silence_timeout_s
        self._lead_in = int(round(lead_in_s * SAMPLE_RATE))
        cap_s = capture_s if capture_s is not None else (self._prompt_len / SAMPLE_RATE + 30.0)
        self._capture_max = int(round(cap_s * SAMPLE_RATE))
        self._tail = int(round((capture_s if capture_s is not None else 30.0) * SAMPLE_RATE))
        self._input = np.concatenate(
            [
                np.zeros(self._lead_in, dtype=self._prompt.dtype),
                self._prompt,
                np.zeros(self._tail, dtype=self._prompt.dtype),
            ]
        )

        self._encoder = sphn.OpusStreamWriter(SAMPLE_RATE)
        self._decoder = sphn.OpusStreamReader(SAMPLE_RATE)

    def run(self):
        import asyncio

        from websockets.exceptions import ConnectionClosed

        try:
            asyncio.run(self._converse())
        except ConnectionClosed as exc:
            print("[unmute-client] websocket closed early:", exc)

    async def _converse(self):
        import asyncio

        import websockets

        async with websockets.connect(self._url, subprotocols=["realtime"], max_size=None) as ws:
            # Configure the session (voice + don't record). allow_recording is required.
            await ws.send(
                json.dumps(
                    {
                        "type": _EV_SESSION_UPDATE,
                        "session": {"voice": self._voice, "allow_recording": False},
                    }
                )
            )
            await asyncio.gather(self._send_prompt(ws), self._save_reply(ws))
        print("[unmute-client] done:", self._inp.name)

    async def _send_prompt(self, ws):
        """Opus-encode lead-in + prompt + tail silence and stream as append events."""
        import asyncio

        for frame in _iter_frames(self._input):
            packet = self._encoder.append_pcm(frame.astype(np.float32) / 32768.0)
            if not isinstance(packet, (bytes, bytearray)):
                raise RuntimeError("Opus encoder produced no packet for a frame")
            await ws.send(
                json.dumps(
                    {
                        "type": _EV_AUDIO_APPEND,
                        "audio": base64.b64encode(bytes(packet)).decode("ascii"),
                    }
                )
            )
            # Pace the stream roughly real-time so the server's VAD behaves naturally.
            await asyncio.sleep(FRAME_SAMPLES / SAMPLE_RATE * 0.5)

        # Let the receiver decide when the turn is over (silence timeout / cap).
        while not self._reply_done:
            await asyncio.sleep(0.1)
        await ws.close()

    async def _save_reply(self, ws):
        """Decode ``response.audio.delta`` Opus and write the captured reply wav."""
        import asyncio

        import soundfile as sf

        from websockets.exceptions import ConnectionClosed

        reply_len = 0
        cap = self._capture_max
        silence_needed = int(round(self._silence_timeout * SAMPLE_RATE))
        first_audio_timeout = (self._lead_in + self._prompt_len) / SAMPLE_RATE + 30.0
        got_audio = False
        speech_started = False
        trailing_silence = 0

        with sf.SoundFile(self._out, "w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16") as sink:
            try:
                while reply_len < cap:
                    timeout = (self._silence_timeout + 5.0) if got_audio else first_audio_timeout
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    if not isinstance(raw, str):
                        continue
                    ev = json.loads(raw)
                    etype = ev.get("type")
                    if etype == _EV_TEXT_DELTA:
                        print("[unmute-text]", ev.get("delta", ""))
                        continue
                    if etype == _EV_AUDIO_DONE:
                        if speech_started:
                            break
                        continue
                    if etype != _EV_AUDIO_DELTA:
                        continue
                    opus = base64.b64decode(ev["delta"])
                    pcm = self._decoder.append_bytes(opus)
                    ended = False
                    while pcm.size:
                        remaining = cap - reply_len
                        if remaining <= 0:
                            ended = True
                            break
                        take = min(pcm.size, remaining)
                        chunk = pcm[:take]
                        sink.write(_pcm_to_int16(chunk))
                        reply_len += take
                        got_audio = True
                        if float(np.max(np.abs(chunk))) >= _SPEECH_THRESHOLD:
                            speech_started = True
                            trailing_silence = 0
                        elif speech_started:
                            trailing_silence += take
                        pcm = pcm[take:]
                    if ended or (speech_started and trailing_silence >= silence_needed):
                        break
            except (asyncio.TimeoutError, ConnectionClosed):
                pass
            finally:
                self._reply_done = True
