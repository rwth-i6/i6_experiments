"""In-our-code MoshiRAG offline inference engine (the science layer; replaces the fork's driver).

MoshiRAG (kyutai-labs/moshi-rag, arXiv 2604.12928) is a full-duplex Moshi variant that emits a
``<ret>`` token to trigger a text-in/text-out retrieval LLM and folds the returned reference into its
reply via an ARC-Encoder conditioner (a ``[T, dim]`` streaming-sum added to the temporal embeddings).

The fork's own offline entry point (``moshi.run_inference``) runs a **batched async server**: per clip,
three coroutines (``_feed_loop`` / ``_stt_recv_loop`` / ``_output_loop``) talk to a *shared* batched
step loop through queues + a ``step_index`` condition variable. That design **deadlocks on the tail of
a batch**: the model's delay buffer needs ``max_stream_delay`` more input frames to emit its final
outputs, but ``_feed_loop`` is gated on ``step_index`` advancing, and ``step_index`` only advances when
those very outputs emerge in ``_output_loop`` -> circular wait (the 193/200 strand; see moshirag.md).

This engine eliminates that class of bug **by construction**: one synchronous lockstep pump per clip --
feed one frame, run one model step, consume one output -- with **no cross-loop gating**. Retrieval still
runs as a background asyncio task (so generation overlaps it, exactly as the released system intends),
but the main loop never *waits* on it: it keeps stepping and simply injects the reference the moment the
task finishes. There is no shared queue/``step_index`` coupling, so nothing can wedge.

It reuses the fork's components verbatim (``load_models``, ``BatchRunner``, ``LocalSpeechToText``,
``TurnManager``, ``LLMReferenceGenerator``, the remote ARC encoder) -- only the orchestration is ours,
so behaviour stays faithful to the released model. ``LocalSpeechToText.send_audio`` does all its STT
compute *inline* (VAD callback fired, word messages queued before it returns), which is what makes a
synchronous pump correct without timing races.

Driven by ``moshirag_offline_inference.py`` in two modes: knowledge (``in_dir``/``out_dir``, pad
lead-in+capture silence) and FDB (manifest pairs, processed as-is + a short tail to finish the reply).
Requires ``REFERENCE_ENCODER_URL`` (co-launched encoder) + an OpenAI-compatible retrieval LLM via
``LLM_BASE_URL``/``LLM_MODEL_NAME`` env (or ``gt_reference_text`` sidecars). ``import moshi`` here must
resolve to the moshi-rag fork in the venv site-packages, so this module lives under ``dorian_koch/``
(no ``speech_llm`` on sys.path), like ``moshirag_offline_inference.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import wave
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sphn
import torch

logger = logging.getLogger(__name__)


def _load_audio_mono_float32(path: Path, target_sr: int) -> np.ndarray:
    """Load audio as mono float32 in [-1, 1] at ``target_sr`` (mirrors the fork's loader)."""
    pcm, _ = sphn.read(str(path), sample_rate=target_sr)
    if pcm.ndim != 2:
        raise ValueError(f"expected 2D audio from sphn.read, got {pcm.shape} for {path}")
    x = np.mean(pcm.astype(np.float64), axis=0).astype(np.float32)
    np.clip(x, -1.0, 1.0, out=x)
    return x


def _write_wav_mono_float32(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 PCM in [-1, 1] as 16-bit WAV (mirrors the fork's writer)."""
    pcm = np.clip(pcm.astype(np.float32), -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())


class MoshiRagModel:
    """Loaded-once MoshiRAG inference state (moshi-rag LM + Mimi + STT + retrieval), driven serially."""

    def __init__(
        self,
        hf_repo: str,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        rag_timeout: float = 10.0,
        max_reference_tokens: int = 64,
        stt_wait_time: float = 0.5,
        vad_window_size: int = 4,
        vad_threshold: float = 0.5,
        init_active_speaker: str = "user",
        power_threshold: int = -65,
    ):
        # Reference encoder is remote (REFERENCE_ENCODER_URL set by the driver), so load_models skips
        # the in-model reference conditioner -- the ARC encoding happens in the co-launched server.
        from moshi.inference_utils import load_models
        from moshi.inference_utils.audio_processor import AudioProcessor
        from moshi.inference_utils.batch_runner import BatchRunner
        from moshi.inference_utils.utils import get_reference_encoder_url
        from moshi.inference_utils.retrieval_profiles import default_profile_id, load_retrieval_env
        from moshi.reference import LLMReferenceGenerator
        from moshi.stt import LocalSpeechToText

        self.encoder_url = get_reference_encoder_url()  # validates REFERENCE_ENCODER_URL is set
        self.device = device
        self.rag_timeout = rag_timeout
        self.max_reference_tokens = max_reference_tokens

        args = SimpleNamespace(
            hf_repo=hf_repo,
            moshi_weight=None,
            mimi_weight=None,
            tokenizer=None,
            config=None,
            cfg_coef=1.0,
            device=device,
            dtype=dtype,
            batch_size=1,
            init_active_speaker=init_active_speaker,
        )
        mimi, self.text_tokenizer, self.lm_gen = load_models(args)
        # STT gets its own Mimi copy, taken before BatchRunner mutates the main one (mirrors the fork).
        stt_mimi = deepcopy(mimi)
        mimi.set_num_codebooks(self.lm_gen.lm_model.num_codebooks - 1)
        self.runner = BatchRunner(mimi, self.lm_gen, device, batch_size=1)
        self.mimi = mimi
        self.frame_size = self.runner.frame_size
        self.sample_rate = int(mimi.sample_rate)
        self.frame_rate = float(mimi.frame_rate)
        self.rag_token_id = self.lm_gen.lm_model.rag_token_id
        self.audio_processor = AudioProcessor(power_threshold=power_threshold)

        self.stt = LocalSpeechToText(stt_mimi, device=device)
        self.stt_wait_steps = int(stt_wait_time * self.frame_rate) if stt_wait_time > 0 else 0
        self.vad_window_size = vad_window_size
        self.vad_threshold = vad_threshold
        self.init_active_speaker = init_active_speaker

        # Reference generator -- replicate ServerState's construction so behaviour matches the fork.
        retrieval_env = load_retrieval_env()
        profiles = retrieval_env.profiles
        if len(profiles) >= 2:
            self.reference_generator = LLMReferenceGenerator(retrieval_profiles=profiles)
            logger.info("[Retrieval] %d profiles, default=%r", len(profiles), default_profile_id(profiles))
        else:
            style = profiles[0].prompt_style if len(profiles) == 1 else "original"
            self.reference_generator = LLMReferenceGenerator(prompt_style=style)

        logger.info("warming up moshi-rag model + reference generator")
        self.runner.warmup()
        self.reference_generator.warmup()

    def _decode_text_token(self, token: int) -> str | None:
        if token in (0, 1, 2, 3):
            return None
        return self.text_tokenizer.id_to_piece(token).replace("▁", " ")

    def _model_step(self, frame: torch.Tensor, is_first: bool) -> tuple[int, np.ndarray | None]:
        """One batched (B=1) Mimi-encode -> LM-step -> decode; returns (text_token, model_pcm | None)."""
        from moshi.inference_utils.batch_runner import BatchInput

        filtered = self.audio_processor.filter_by_power(frame)  # [1, 1, frame_size]
        captured: dict = {"t": None, "p": None}

        def deliver(_slot, *, text_token, pcm_out):
            captured["t"] = text_token
            captured["p"] = pcm_out

        g = BatchInput(
            filtered_pcm_batch=filtered,
            lm_mask_cpu=torch.ones(1, dtype=torch.bool),
            first_mask_cpu=torch.tensor([is_first], dtype=torch.bool),
            active=[(0, None)],
        )
        self.runner.run_step(g, deliver)
        pcm = None
        if captured["p"] is not None:
            pcm = captured["p"].detach().cpu().float().numpy().reshape(-1)
        return int(captured["t"]) if captured["t"] is not None else -1, pcm

    async def _retrieve(self, context: str, gt_reference_text: str | None) -> torch.Tensor:
        """Background task: get reference text (LLM, or the gt sidecar) then encode it remotely.

        Returns the ARC streaming-sum tensor ([T, dim]) to inject, or a zero-length tensor if there
        is nothing to ground on (empty reference)."""
        from moshi.inference_utils.utils import get_conditioning_remote_async

        if gt_reference_text is not None:
            reference_text = gt_reference_text
        else:
            _, reference_text, _, _ = await self.reference_generator.generate_reference_text(
                context, [], llm_call_timeout=self.rag_timeout, max_tokens=self.max_reference_tokens
            )
        cond = await get_conditioning_remote_async(reference_text or "", self.encoder_url)
        return cond.squeeze(0)

    async def run_clip(
        self,
        in_wav: str,
        out_wav: str,
        *,
        lead_in_s: float,
        capture_s: float,
        tail_s: float,
        gt_reference_text: str | None = None,
    ) -> dict:
        """Process one clip end-to-end with a synchronous feed->step->output pump (no deadlock).

        ``lead_in_s``/``capture_s`` pad the input with silence (knowledge mode lets the full spoken
        answer land in the trailing silence); ``tail_s`` is a short extra silence so the model can
        finish its last word and the delay buffer flushes. Writes the reply WAV + a trace JSON.
        """
        from moshi.inference_utils.turn_manager import TurnManager
        from moshi.stt import STTWordMessage

        turn = TurnManager(
            window_size=self.vad_window_size,
            threshold=self.vad_threshold,
            stt_wait_steps=self.stt_wait_steps,
            init_active_speaker=self.init_active_speaker,
        )
        # Per-clip reset: fresh STT streaming state + clear any leftover reference streaming-sum so a
        # previous clip's retrieved reference can never leak into this one.
        await self.stt.start_up()
        self.stt.vad_callback = turn.update_vad
        state = self.lm_gen._streaming_state
        if state is not None:
            state.pending_streaming_sums = [None] * state.batch_size

        sr, frame = self.sample_rate, self.frame_size
        pcm = _load_audio_mono_float32(Path(in_wav), sr)
        nl, ncap, ntail = (int(round(s * sr)) for s in (lead_in_s, capture_s, tail_s))
        pcm = np.concatenate([np.zeros(nl, np.float32), pcm, np.zeros(ncap + ntail, np.float32)])
        tail = len(pcm) % frame
        if tail:
            pcm = pcm[: len(pcm) - tail]

        model_pcm: list[np.ndarray] = []
        model_text: list[str] = []
        user_text: list[str] = []
        user_ids: deque[int] = deque()
        trace = {
            "rag_trigger_step": -1,
            "retrieval_step": -1,
            "reference_text": "",
            "model_text": model_text,
            "user_text": user_text,
        }

        ret_task: asyncio.Task | None = None
        ret_countdown = -1  # steps remaining before we fire retrieval (stt_wait_steps after <ret>)
        triggered = False

        n_steps = len(pcm) // frame
        for step in range(n_steps):
            off = step * frame
            chunk_np = pcm[off : off + frame].astype(np.float32)
            await self.stt.send_audio(chunk_np)  # STT compute is inline: VAD + words ready on return
            # Drain whatever the STT produced this step into the turn manager (builds RAG context).
            while not self.stt._out_queue.empty():
                msg = self.stt._out_queue.get_nowait()
                if isinstance(msg, STTWordMessage):
                    turn.handle_spoken_text(user_text=msg.text.replace("▁", " "))
                    user_ids.append(msg.id)

            chunk = torch.from_numpy(chunk_np).to(self.device, torch.float32)[None, None]
            text_token, pcm_out = self._model_step(chunk, is_first=(step == 0))
            if pcm_out is not None:
                model_pcm.append(pcm_out)

            if text_token == self.rag_token_id and not triggered:
                triggered = True
                trace["rag_trigger_step"] = step
                ret_countdown = self.stt_wait_steps  # let STT catch up on the last user words first
                model_text.append(self.text_tokenizer.id_to_piece(text_token))
            else:
                decoded = self._decode_text_token(text_token)
                turn.handle_spoken_text(model_text=decoded)
                model_text.append("<pad>" if decoded is None else self.text_tokenizer.id_to_piece(text_token))
            user_text.append(self.stt.text_tokenizer.id_to_piece(user_ids.popleft()) if user_ids else "<pad>")

            # Fire retrieval once the post-<ret> wait elapses (background task; loop keeps stepping).
            if ret_countdown == 0 and ret_task is None:
                trace["retrieval_step"] = step
                ret_task = asyncio.create_task(self._retrieve(turn.get_context(), gt_reference_text))
            if ret_countdown >= 0:
                ret_countdown -= 1

            # Inject the reference the instant retrieval finishes -- applied from the next step on.
            if ret_task is not None and ret_task.done():
                cond = ret_task.result()
                ret_task = None
                if cond is not None and cond.shape[0] > 0:
                    self.lm_gen.update_streaming_sum_tensors([cond])
            await asyncio.sleep(0)  # let the background retrieval task make progress

        if ret_task is not None:
            try:
                await asyncio.wait_for(ret_task, timeout=self.rag_timeout + 5.0)
            except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001 - best effort at clip end
                logger.warning("retrieval task unfinished at clip end: %s", e)

        out_path = Path(out_wav)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_pcm = np.concatenate(model_pcm) if model_pcm else np.zeros(max(sr // 10, 1), np.float32)
        _write_wav_mono_float32(out_path, all_pcm, sr)
        trace_path = out_path.parent / ("output.json" if out_path.name == "output.wav" else f"{out_path.stem}.json")
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        return trace


async def run_pairs(
    model: MoshiRagModel,
    pairs: list[tuple[str, str]],
    *,
    lead_in_s: float,
    capture_s: float,
    tail_s: float,
    gt_reference_for: dict | None = None,
) -> int:
    """Run ``model`` over (in_wav, out_wav) pairs serially. Returns the number of clips written.

    Serial (batch=1) is what makes the synchronous pump trivially deadlock-free; for ~200-1000 clips
    that is fine offline (retrieval overlaps generation within each clip). Raising the throughput via
    real batching would re-introduce cross-slot coupling and is intentionally not done here.
    """
    n = len(pairs)
    for i, (in_wav, out_wav) in enumerate(pairs, 1):
        gt = (gt_reference_for or {}).get(in_wav)
        t0 = time.time()
        await model.run_clip(
            in_wav, out_wav, lead_in_s=lead_in_s, capture_s=capture_s, tail_s=tail_s, gt_reference_text=gt
        )
        print(f"[moshirag] {i}/{n} clips done ({time.time() - t0:.1f}s)", flush=True)
    return n
