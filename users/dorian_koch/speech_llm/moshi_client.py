"""Standalone client for kyutai Moshi's streaming websocket API.

Everything needed to run file-based inference against a Moshi model lives here:

* ``moshi_server`` -- context manager that boots a local ``moshi.server`` on a
  free port and shuts it down afterwards.
* ``_ws_url`` -- turn a ``host[:port]`` / URL into the ``/api/chat`` endpoint.
* ``MoshiFileClient`` -- stream one wav file through the server and write the
  model's spoken reply back to disk, trimmed to the prompt length.

The websocket handling is an independent implementation of Moshi's public wire
protocol; it does not reuse any third-party client code.
"""

from __future__ import annotations

import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# Moshi streams 24 kHz mono audio split into 80 ms Opus frames.
SAMPLE_RATE = 24_000
FRAME_SAMPLES = 1_920
# The reply lags the prompt by roughly one frame; pad the start with this many
# frames of silence so the output lines up with the input.
LEAD_IN_FRAMES = 1
# Captured audio (float PCM in [-1, 1]) with peak amplitude below this is treated as silence
# when detecting the end of Moshi's answer.
_SPEECH_THRESHOLD = 0.01

# First byte of every binary websocket frame tags its payload.
_TAG_HANDSHAKE = 0x00
_TAG_AUDIO = 0x01
_TAG_TEXT = 0x02

# Port the server listens on when an address omits one.
_DEFAULT_PORT = 8998


@contextmanager
def moshi_server():
    # first: find a free port
    # take my slurm job id and modulo
    if "SLURM_JOB_ID" in os.environ:
        job_id = int(os.environ["SLURM_JOB_ID"])
        port = 8998 + (job_id % 1000)
    else:
        port = 8998 + random.randint(0, 999)
    # check if in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        for _ in range(50):
            try:
                s.bind(("localhost", port))
                break
            except OSError:
                port += 1

    print(f"Selected port {port} for Moshi server")

    cmd = [
        sys.executable,
        "-m",
        "moshi.moshi.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    print(f"Starting Moshi server: {' '.join(cmd)}")
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,  # Create process group
        cwd=os.environ.get("MOSHI_SERVER_CWD", "/home/tt201262/setups/2026-01-speech-llm/projects/moshi"),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    # Wait for server to be ready (check port 8998)
    max_wait = 15 * 60  # seconds
    start_time = time.time()
    server_ready = False

    # Also read output to look for ready signal
    def read_server_output():
        nonlocal server_ready
        if server_process.stdout:
            for line in iter(server_process.stdout.readline, ""):
                if line and "frame handled" not in line:
                    print(f"[Moshi Server] {line.rstrip()}", flush=True)
                if "Access the Web UI directly at" in line:
                    server_ready = True
        print("Moshi server output thread exiting")

    output_thread = threading.Thread(target=read_server_output, daemon=True)
    output_thread.start()

    # Wait for server to be ready
    while time.time() - start_time < max_wait:
        # Check if server process died
        if server_process.poll() is not None:
            stdout, _ = server_process.communicate()
            raise RuntimeError(f"Moshi server died during startup:\n{stdout}")

        # Try to connect to port
        try:
            sock = socket.create_connection(("localhost", port), timeout=1)
            sock.close()
            if server_ready:
                print("Moshi server is ready and accepting connections")
                break
            else:
                print("Moshi server port is open but server not ready yet")
        except (ConnectionRefusedError, socket.timeout):
            pass

        time.sleep(0.5)
    else:
        raise TimeoutError(f"Moshi server not ready after {max_wait} seconds")
    print("Moshi server started successfully")

    try:
        yield f"localhost:{port}"
    finally:
        # Stop the Moshi server if it's still running
        if server_process and server_process.poll() is None:
            print("Stopping Moshi server...")
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                else:
                    server_process.terminate()
                server_process.wait(timeout=10)
                print("Moshi server stopped gracefully")
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                print("Force killing Moshi server...")
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                else:
                    server_process.kill()
                server_process.wait()


def _ws_url(addr: str) -> str:
    """Return the ``/api/chat`` websocket URL for a Moshi server address."""
    if "://" in addr:
        scheme, _, rest = addr.partition("://")
        ws_scheme = "wss" if scheme in ("https", "wss") else "ws"
        return f"{ws_scheme}://{rest.rstrip('/')}/api/chat"
    if ":" not in addr:
        addr = f"{addr}:{_DEFAULT_PORT}"
    return f"ws://{addr}/api/chat"


def _load_prompt_24k(path) -> np.ndarray:
    """Read ``path`` as mono int16 PCM, resampled to 24 kHz."""
    import soundfile as sf

    samples, rate = sf.read(path, dtype="int16")
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.int16)
    if rate == SAMPLE_RATE:
        return samples.astype(np.int16)

    import torch
    import torchaudio.functional as taf

    print(f"[moshi-client] resampling {path} from {rate} -> {SAMPLE_RATE} Hz")
    wav = torch.from_numpy(samples.astype(np.float32) / 32768.0)[None, :]
    wav = taf.resample(wav, rate, SAMPLE_RATE)[0].numpy()
    return np.clip(np.round(wav * 32768.0), -32768, 32767).astype(np.int16)


def _iter_frames(samples: np.ndarray):
    """Yield consecutive ``FRAME_SAMPLES``-long chunks, zero-padding the tail."""
    short = len(samples) % FRAME_SAMPLES
    if short:
        samples = np.concatenate([samples, np.zeros(FRAME_SAMPLES - short, samples.dtype)])
    for start in range(0, len(samples), FRAME_SAMPLES):
        yield samples[start : start + FRAME_SAMPLES]


def _pcm_to_int16(pcm: np.ndarray) -> np.ndarray:
    return np.clip(np.round(pcm * 32768.0), -32768, 32767).astype(np.int16)


class MoshiFileClient:
    """Run a single wav file through a Moshi server over its chat websocket.

    The prompt audio is pushed as a sequence of Opus frames tagged ``0x01`` and
    the server streams its own Opus audio back the same way; text tokens arrive
    tagged ``0x02`` and are only logged. The reply is saved at the prompt
    length, with one lead-in frame of silence so it stays time-aligned.
    """

    def __init__(
        self,
        ws_url: str,
        inp,
        out,
        *,
        lead_in_s: float = 0.0,
        answer_window_s: float | None = None,
        silence_timeout_s: float = 2.0,
    ):
        import sphn

        self._url = str(ws_url)
        self._inp = Path(inp)
        self._out = Path(out)

        self._prompt = _load_prompt_24k(self._inp)
        self._prompt_len = int(len(self._prompt))
        self._reply_len = 0  # reply samples captured so far
        self._reply_done = False

        # Capture mode. When ``answer_window_s`` is None we keep the legacy behaviour: stream
        # only the prompt and capture the reply that overlaps it, trimmed to the prompt length
        # (used by fdb.py). When it is set we delay the question by ``lead_in_s`` of leading
        # silence -- so Moshi's reflexive opening greeting lands in the lead-in rather than being
        # mistaken for the answer -- then stream ``answer_window_s`` of trailing silence and
        # capture the reply only during that trailing window, which is the model's actual answer.
        self._answer_window_s = answer_window_s
        # Capture ends this many seconds after the model stops emitting audio (variable-length
        # answer). It also bounds every recv(), so a model that goes silent can never hang us.
        self._silence_timeout = silence_timeout_s
        self._lead_in = int(round(lead_in_s * SAMPLE_RATE))
        if answer_window_s is not None:
            self._answer_window = int(round(answer_window_s * SAMPLE_RATE))
            self._capture_start = self._lead_in + self._prompt_len
            self._reply_seen = 0  # total reply samples observed (including skipped)
            self._input = np.concatenate(
                [
                    np.zeros(self._lead_in, dtype=self._prompt.dtype),
                    self._prompt,
                    np.zeros(self._answer_window, dtype=self._prompt.dtype),
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
            print("[moshi-client] websocket closed early:", exc)

    async def _converse(self):
        import asyncio

        import websockets

        async with websockets.connect(self._url, max_size=None) as ws:
            primer = await self._skip_handshake(ws)
            await asyncio.gather(self._send_prompt(ws), self._save_reply(ws, primer))
        print("[moshi-client] done:", self._inp.name)

    async def _skip_handshake(self, ws):
        """Swallow the server's opening 0x00 frame; hand back anything else."""
        import asyncio

        from websockets.exceptions import ConnectionClosed

        try:
            first = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except (asyncio.TimeoutError, ConnectionClosed):
            return None
        if isinstance(first, (bytes, bytearray)) and first[:1] == bytes([_TAG_HANDSHAKE]):
            return None
        return first

    async def _send_prompt(self, ws):
        """Opus-encode the audio frame by frame and stream it to the server."""
        import asyncio

        # Legacy mode streams only the prompt; answer-window mode streams
        # lead-in silence + prompt + trailing answer-window silence.
        audio = self._prompt if self._answer_window_s is None else self._input
        for frame in _iter_frames(audio):
            packet = self._encoder.append_pcm(frame.astype(np.float32) / 32768.0)
            if not isinstance(packet, (bytes, bytearray)):
                raise RuntimeError("Opus encoder produced no packet for a frame")
            await ws.send(bytes([_TAG_AUDIO]) + packet)
            spill = self._encoder.append_pcm(np.empty(0, dtype=np.float32))
            if spill:
                raise RuntimeError("Opus encoder returned unexpected buffered data")

        # Hold the socket open until the receiver is done. In legacy mode that means a full
        # overlapping reply; in answer-window mode the receiver ends the turn on a silence
        # timeout, so we just wait for that signal.
        if self._answer_window_s is None:
            while self._reply_len < self._prompt_len and not self._reply_done:
                await asyncio.sleep(0.1)
        else:
            while not self._reply_done:
                await asyncio.sleep(0.1)
        await ws.close()

    async def _save_reply(self, ws, primer):
        """Decode the server's audio frames and write the captured reply to the output wav."""
        if self._answer_window_s is None:
            return await self._save_reply_legacy(ws, primer)
        return await self._save_reply_answer_window(ws, primer)

    async def _save_reply_legacy(self, ws, primer):
        """Capture the reply overlapping the prompt, trimmed to the prompt length (fdb.py)."""
        import soundfile as sf

        from websockets.exceptions import ConnectionClosed

        self._reply_len = 0
        padded = False

        async def messages():
            if primer is not None:
                yield primer
            async for message in ws:
                yield message

        with sf.SoundFile(self._out, "w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16") as sink:
            try:
                async for message in messages():
                    if not message:
                        continue
                    tag, payload = message[0], message[1:]
                    if tag == _TAG_TEXT:
                        print("[moshi-text]", payload.decode("utf-8", "ignore"))
                        continue
                    if tag != _TAG_AUDIO:
                        continue

                    pcm = self._decoder.append_bytes(payload)
                    while pcm.size:
                        if not padded:
                            lead = min(LEAD_IN_FRAMES * FRAME_SAMPLES, self._prompt_len)
                            sink.write(np.zeros(lead, dtype=np.int16))
                            self._reply_len += lead
                            padded = True
                        remaining = self._prompt_len - self._reply_len
                        if remaining <= 0:
                            break
                        take = min(pcm.size, remaining)
                        sink.write(_pcm_to_int16(pcm[:take]))
                        self._reply_len += take
                        pcm = self._decoder.append_bytes(b"")
            except ConnectionClosed as exc:
                print("[moshi-client] receive interrupted:", exc)
            finally:
                self._reply_done = True

            if self._reply_len < self._prompt_len:
                sink.write(np.zeros(self._prompt_len - self._reply_len, dtype=np.int16))

    async def _save_reply_answer_window(self, ws, primer):
        """Skip the reply overlapping the lead-in + question, then capture the model's answer.

        Moshi emits one output frame per input frame, so it keeps streaming (silence) frames for
        the whole trailing-silence window. Capture therefore ends on *content* silence: once Moshi
        has actually started answering and then stays quiet for ``silence_timeout`` seconds. A
        leading pause (before the answer starts) does not count, the answer window is a hard cap,
        and a time-bounded ``recv`` is a backstop so we can never hang even if frames stop coming.
        """
        import asyncio

        import soundfile as sf

        from websockets.exceptions import ConnectionClosed

        self._reply_len = 0
        target = self._answer_window
        silence_needed = int(round(self._silence_timeout * SAMPLE_RATE))
        # Wait this long for the answer to begin (Moshi works through lead-in + question first);
        # once it is talking, a tighter recv bound is just the no-frames-at-all hang backstop.
        first_audio_timeout = self._lead_in / SAMPLE_RATE + self._prompt_len / SAMPLE_RATE + 10.0
        speech_started = False
        trailing_silence = 0
        pending = primer

        with sf.SoundFile(self._out, "w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16") as sink:
            try:
                while self._reply_len < target:
                    if pending is not None:
                        message, pending = pending, None
                    else:
                        timeout = (self._silence_timeout + 3.0) if speech_started else first_audio_timeout
                        message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    if not message:
                        continue
                    tag, payload = message[0], message[1:]
                    if tag == _TAG_TEXT:
                        print("[moshi-text]", payload.decode("utf-8", "ignore"))
                        continue
                    if tag != _TAG_AUDIO:
                        continue

                    pcm = self._decoder.append_bytes(payload)
                    ended = False
                    while pcm.size:
                        if self._reply_seen < self._capture_start:
                            skip = min(pcm.size, self._capture_start - self._reply_seen)
                            self._reply_seen += skip
                            pcm = pcm[skip:]
                            continue
                        remaining = target - self._reply_len
                        if remaining <= 0:
                            ended = True
                            break
                        take = min(pcm.size, remaining)
                        chunk = pcm[:take]
                        sink.write(_pcm_to_int16(chunk))
                        self._reply_len += take
                        self._reply_seen += take
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

            if self._reply_len < target:
                sink.write(np.zeros(target - self._reply_len, dtype=np.int16))
