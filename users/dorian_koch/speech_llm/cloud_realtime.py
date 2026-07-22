"""Cloud realtime speech-to-speech backends (Gemini Live + OpenAI Realtime).

These are *streaming* backends in the exact sense ``speech_backends.py`` already
supports, except the "server" lives in the cloud: nothing is launched locally, so
the ``server`` context manager just yields the model id (the handle) and the
``file_client`` opens a websocket to the provider, streams the question audio, and
writes the spoken reply. They therefore plug into the shared streaming harness
(``inference_harness.stream_inference``) with no benchmark-job changes -- the same
access loop that drives Moshi/Unmute drives these.

Placement: the model is remote and needs **no GPU**, but it needs **outbound
internet**, which HPC compute nodes generally lack. So the inference job runs these
as a **login-node mini_task** (the login node has proxied internet); the backend
spec carries ``cloud_api=True`` to request that placement.

Gating (like the gemma-3 / PersonaPlex downloads): an API key must be present in the
job's environment -- ``OPENAI_API_KEY`` / ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
Set it as a sandbox secret / persistent env var on the cluster; the client asserts it
loudly if missing. The provider domains must also be reachable
(``api.openai.com`` / ``generativelanguage.googleapis.com``) -- both verified reachable
from the i6-rz login node.

Audio contract (matches MoshiFileClient): ``FileClient(url, inp, out, *, lead_in_s,
capture_s).run()`` reads the question wav and writes a reply wav. The reply is written
with the **input duration prepended as silence** so the output spans the full
conversation timeline (reply onset ~= end of the user's turn) -- this keeps FDB
latency measured from the correct t=0 and satisfies the harness length-check guard
(the same alignment principle as the Unmute output-alignment fix).

NOTE: the websocket message schemas below follow the providers' documented realtime
protocols; they are written defensively but have NOT been validated against the live
APIs from here (no key available at authoring time). First live run should be a tiny
smoke (a handful of clips) before a full benchmark.
"""

from __future__ import annotations

import os
import json
import base64
import contextlib

import numpy as np
import soundfile as sf

# Provider output is PCM16 mono @ 24 kHz; OpenAI also takes 24 kHz input, Gemini 16 kHz.
_OUT_SR = 24000


# ---------------------------------------------------------------------------
# audio helpers
# ---------------------------------------------------------------------------


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    try:
        import soxr  # high-quality, tiny dep

        return soxr.resample(x, sr_in, sr_out)
    except Exception:
        # polyphase fallback (scipy); avoids a hard soxr dependency
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g)


def _read_pcm16(path: str, target_sr: int) -> tuple[bytes, float]:
    """Read a wav as mono float32, resample to ``target_sr``, return (pcm16 bytes,
    duration_seconds_of_source)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr
    audio = _resample(audio, sr, target_sr)
    pcm = np.clip(audio, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes(), dur


def _write_reply(out_path: str, reply_pcm16: bytes, *, lead_silence_s: float) -> None:
    """Write the reply at 24 kHz with ``lead_silence_s`` of leading silence prepended,
    so the output spans the whole conversation timeline (reply onset = end of input)."""
    reply = np.frombuffer(reply_pcm16, dtype="<i2").astype("float32") / 32767.0
    lead = np.zeros(int(round(lead_silence_s * _OUT_SR)), dtype="float32")
    sf.write(out_path, np.concatenate([lead, reply]), _OUT_SR, subtype="PCM_16")


def _require_key(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise RuntimeError(
        f"No API key in environment (looked for {', '.join(names)}). Set it as a sandbox "
        "secret / persistent env var on the cluster before running this backend."
    )


def _chunks(b: bytes, n: int):
    for i in range(0, len(b), n):
        yield b[i : i + n]


# Default spoken-QA persona shared by both providers (kept terse: answer, don't chat).
_SYSTEM_PROMPT = (
    "You are a concise spoken question-answering assistant. The user asks a question by "
    "voice; answer it directly and briefly in speech. Do not ask follow-up questions."
)


# ---------------------------------------------------------------------------
# OpenAI Realtime API
# ---------------------------------------------------------------------------

_OPENAI_MODEL = "gpt-4o-realtime-preview"
_OPENAI_VOICE = "alloy"


@contextlib.contextmanager
def openai_realtime_server(*, lora_weights=None, lora_config=None, python_exe=None, unmute_llm=None):
    """No local process -- the model is in the cloud. Yields the model id as the handle.
    Uniform server signature; all kwargs ignored. Asserts the API key up front so a
    missing key fails the whole job immediately instead of per-clip (non-retryable)."""
    _require_key("OPENAI_API_KEY")
    yield _OPENAI_MODEL


def openai_ws_url(model_id: str) -> str:
    return f"openai-realtime://{model_id}"


class OpenAIRealtimeFileClient:
    """Stream one question wav through the OpenAI Realtime websocket; write the reply."""

    def __init__(self, url, inp, out, *, lead_in_s: float = 0.0, capture_s: float | None = None):
        self.model = str(url).split("://", 1)[-1]
        self.inp = str(inp)
        self.out = str(out)

    def run(self):
        from websockets.sync.client import connect

        key = _require_key("OPENAI_API_KEY")
        pcm, in_dur = _read_pcm16(self.inp, _OUT_SR)  # OpenAI realtime uses 24 kHz pcm16
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = {"Authorization": f"Bearer {key}", "OpenAI-Beta": "realtime=v1"}
        reply = bytearray()
        with connect(url, additional_headers=headers, max_size=None) as ws:
            ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "modalities": ["audio", "text"],
                            "voice": _OPENAI_VOICE,
                            "instructions": _SYSTEM_PROMPT,
                            "input_audio_format": "pcm16",
                            "output_audio_format": "pcm16",
                            "turn_detection": None,  # we drive the turn explicitly
                        },
                    }
                )
            )
            for chunk in _chunks(pcm, 32000):
                ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()}))
            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            ws.send(json.dumps({"type": "response.create", "response": {"modalities": ["audio"]}}))
            while True:
                msg = json.loads(ws.recv(timeout=60))
                t = msg.get("type", "")
                if t == "response.audio.delta" and msg.get("delta"):
                    reply += base64.b64decode(msg["delta"])
                elif t in ("response.done", "response.audio.done"):
                    break
                elif t == "error":
                    raise RuntimeError(f"OpenAI realtime error: {msg.get('error')}")
        _write_reply(self.out, bytes(reply), lead_silence_s=in_dur)


# ---------------------------------------------------------------------------
# Gemini Live API (BidiGenerateContent)
# ---------------------------------------------------------------------------

_GEMINI_MODEL = "gemini-2.0-flash-exp"
_GEMINI_IN_SR = 16000  # Gemini Live takes 16 kHz pcm16 input


@contextlib.contextmanager
def gemini_live_server(*, lora_weights=None, lora_config=None, python_exe=None, unmute_llm=None):
    # Assert the API key up front (non-retryable) so a missing key fails fast, not per-clip.
    _require_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    yield _GEMINI_MODEL


def gemini_ws_url(model_id: str) -> str:
    return f"gemini-live://{model_id}"


class GeminiLiveFileClient:
    """Stream one question wav through the Gemini Live websocket; write the reply."""

    def __init__(self, url, inp, out, *, lead_in_s: float = 0.0, capture_s: float | None = None):
        self.model = str(url).split("://", 1)[-1]
        self.inp = str(inp)
        self.out = str(out)

    def run(self):
        from websockets.sync.client import connect

        key = _require_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
        pcm16k, in_dur = _read_pcm16(self.inp, _GEMINI_IN_SR)
        url = (
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=" + key
        )
        reply = bytearray()
        with connect(url, max_size=None) as ws:
            ws.send(
                json.dumps(
                    {
                        "setup": {
                            "model": f"models/{self.model}",
                            "generationConfig": {"responseModalities": ["AUDIO"]},
                            "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
                        }
                    }
                )
            )
            json.loads(ws.recv(timeout=30))  # setupComplete
            for chunk in _chunks(pcm16k, 32000):
                ws.send(
                    json.dumps(
                        {
                            "realtimeInput": {
                                "mediaChunks": [
                                    {
                                        "mimeType": f"audio/pcm;rate={_GEMINI_IN_SR}",
                                        "data": base64.b64encode(chunk).decode(),
                                    }
                                ]
                            }
                        }
                    )
                )
            ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
            while True:
                msg = json.loads(ws.recv(timeout=60))
                sc = msg.get("serverContent")
                if not sc:
                    continue
                for part in sc.get("modelTurn", {}).get("parts", []):
                    inline = part.get("inlineData") or {}
                    if inline.get("data"):
                        reply += base64.b64decode(inline["data"])
                if sc.get("turnComplete"):
                    break
        _write_reply(self.out, bytes(reply), lead_silence_s=in_dur)
