"""Base Audex-2B native SPEECH-to-SPEECH cascade — true apples-to-apples benchmarking.

Question audio → Audex audio-QA (text) → Audex audiogen (text→speech codec tokens, CFG) → bundled
Audex causal speech decoder → reply wav. Emits reply wavs ``<i>.wav`` in the SAME layout as
``SpeechInference`` (knowledge mode), so the existing ``WhisperTranscription`` → ``LLMGrading`` (knowledge
benchmark) AND VoiceBench's Whisper→scorer stages consume it UNCHANGED — base Audex now pays the same
speech round-trip + ASR loss as the full-duplex models ("capturing the ASR loss is exactly the point").

All preprocessing + generation logic is Audex's OWN, transcribed verbatim from its
``inference_scripts_hf/audio_utils.py`` (audio-QA) and ``inference_scripts_vllm/audiogen_scripts/
run_audio_gen_vllm.py`` (TTS), ported to plain HF (no vLLM): CFG via transformers'
``UnbatchedClassifierFreeGuidanceLogitsProcessor``; the bundled decoder via its ``create_session`` API.
"""

import os
import re

from sisyphus import Job, Task, tk

from .audex_audioqa import AudioQA, resolve
from .clip_store import open_clips

# --- audiogen constants (verbatim from run_audio_gen_vllm.py) --------------------------------------
_CAUSAL_CHUNK_FRAMES, MIN_TTS_FRAMES, CODEC_FPS = 5, 50, 50
SYSTEM_PROMPT = "You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools."


def _tts_prompt(text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        f"<|text to speech|> Generate speech for this transcription. {text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think></think><speechgen_start>"
    )


def _tts_null_prompt() -> str:
    # CFG unconditional: same template, transcription replaced by <unk>.
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        f"<|text to speech|> Generate speech for this transcription. <unk><|im_end|>\n"
        f"<|im_start|>assistant\n<think></think><speechgen_start>"
    )


def _codec_token_maps(tokenizer) -> dict:
    speech_codec, markers = {}, {}
    names = {"<speechgen_start>": "speechgen_start", "<speechgen_end>": "speechgen_end"}
    for s, tid in tokenizer.get_vocab().items():
        if m := re.match(r"<speechcodec_(\d+)>", s):
            speech_codec[tid] = int(m.group(1))
        elif s in names:
            markers[names[s]] = tid
    return {"speech_codec": speech_codec, **markers}


def _extract_speech_ids(token_ids, maps) -> list:
    end_tid = maps.get("speechgen_end")
    sc = maps["speech_codec"]
    out = []
    for tid in token_ids:
        if tid == end_tid:
            break
        if tid in sc:
            out.append(sc[tid])
    return out


class AudexSpeechS2S(Job):
    """Base Audex-2B: question audio → its native audio-QA→audiogen→decoder cascade → reply wavs."""

    __sis_hash_exclude__ = {"max_new_tokens": 256, "max_speech_tokens": 1024, "cfg_scale": 2.0, "code_version": 1}

    def __init__(
        self,
        *,
        in_dir: tk.Path,  # question wavs <i>.wav (shared benchmark TTS prompts)
        num_shards: int = 1,
        shard: int = 0,
        max_new_tokens: int = 256,
        max_speech_tokens: int = 1024,
        cfg_scale: float = 2.0,
        code_version: int = 1,
    ):
        self.in_dir = in_dir
        self.num_shards = num_shards
        self.shard = shard
        self.max_new_tokens = max_new_tokens
        self.max_speech_tokens = max_speech_tokens
        self.cfg_scale = cfg_scale
        self.code_version = code_version
        self.out_dir = self.output_path("out", directory=True)
        # audioqa(2B)+audiogen(2B)+decoder+nv-whisper enc; wav I/O via soundfile → c25g-safe (no torchcodec).
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 48, "time": 12}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        from transformers.generation.logits_process import (
            LogitsProcessorList,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
        )

        # --- audio-QA model (question audio -> text) ---
        qa = AudioQA(max_new_tokens=self.max_new_tokens)
        qa_dir = qa.model_dir
        gen_dir = resolve("checkpoint_folder_audiogen")
        dec_dir = resolve("audex_causal_speech_decoder")

        # The audiogen checkpoint SHARES checkpoint_folder_full's weights -- the repo ships only its
        # config/index; prepare_audiogen_vllm_checkpoint.sh symlinks full's 2 shards in. Do the same
        # (idempotent) so from_pretrained finds the weights the index.json references.
        for _shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            _dst = os.path.join(gen_dir, _shard)
            if not os.path.lexists(_dst):
                os.symlink(os.path.join(qa_dir, _shard), _dst)

        # --- audiogen model (text -> speech codec tokens) + bundled decoder (tokens -> wav) ---
        gen_tok = AutoTokenizer.from_pretrained(gen_dir, trust_remote_code=True)
        gen = AutoModelForCausalLM.from_pretrained(gen_dir, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
        maps = _codec_token_maps(gen_tok)
        speechgen_end = maps.get("speechgen_end")
        decoder = AutoModel.from_pretrained(dec_dir, trust_remote_code=True).cuda().eval()
        print(f"[s2s] loaded qa+audiogen+decoder; {len(maps['speech_codec'])} speechcodec toks", flush=True)

        def answer_text(wav):
            return qa.answer(wav)

        def synth_wav(text):
            cond = gen_tok(_tts_prompt(text), return_tensors="pt", add_special_tokens=False)
            procs = LogitsProcessorList()
            if self.cfg_scale and self.cfg_scale > 1.0:
                unc = gen_tok(_tts_null_prompt(), return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                procs.append(UnbatchedClassifierFreeGuidanceLogitsProcessor(self.cfg_scale, gen, unconditional_ids=unc))
            with torch.inference_mode():
                out = gen.generate(
                    input_ids=cond.input_ids.cuda(),
                    attention_mask=cond.attention_mask.cuda(),
                    max_new_tokens=self.max_speech_tokens,
                    do_sample=True,
                    temperature=0.8,
                    logits_processor=procs,
                    eos_token_id=speechgen_end,
                    pad_token_id=gen_tok.pad_token_id or gen_tok.eos_token_id,
                )
            ids = _extract_speech_ids(out[0, cond.input_ids.shape[-1] :].tolist(), maps)
            if not ids:
                return None
            session = decoder.create_session(chunk_frames=_CAUSAL_CHUNK_FRAMES)
            chunks, rates = [], set()
            for _sr, ch in session.push([[int(c)] for c in ids]):
                chunks.append(np.asarray(ch).reshape(-1))
                rates.add(int(_sr))
            for _sr, ch in session.flush():
                chunks.append(np.asarray(ch).reshape(-1))
                rates.add(int(_sr))
            if not chunks:
                return None
            assert len(rates) == 1, f"decoder reported several sample rates: {sorted(rates)}"
            return np.concatenate(chunks).astype(np.float32), rates.pop()

        in_dir = self.in_dir.get()
        # Either clip layout; indices (not filenames) are the identity the downstream join uses.
        clips = open_clips(in_dir)
        indices = sorted(clips)[self.shard :: self.num_shards]
        out_dir = self.out_dir.get()
        os.makedirs(out_dir, exist_ok=True)
        min_seconds = MIN_TTS_FRAMES / CODEC_FPS
        seen_rates = set()
        n = len(indices)
        print(f"[s2s] shard {self.shard}/{self.num_shards}: {n} clips", flush=True)
        done = blank = 0
        blank_names = []  # written at the decoder's rate once known, so every clip of a job has one rate

        def write(name, w, sr):
            min_samples = int(min_seconds * sr)
            if w is None or len(w) == 0:
                w = np.zeros(min_samples, dtype=np.float32) + 1e-3
            elif len(w) < min_samples:
                w = np.pad(w, (0, min_samples - len(w)))
            sf.write(os.path.join(out_dir, name), w, sr)

        for k, idx in enumerate(indices):
            name = f"{idx}.wav"
            text = answer_text(clips[idx])
            res = synth_wav(text) if text.strip() else None
            # Write at the decoder's own rate: writing at a fixed 16 kHz would pitch-shift any other rate.
            if res is not None and len(res[0]):
                w, sr = res
                if sr not in seen_rates:
                    print(f"[s2s] decoder sample rate {sr} Hz", flush=True)
                seen_rates.add(sr)
                assert len(seen_rates) == 1, f"decoder sample rate changed between clips: {sorted(seen_rates)}"
                write(name, w, sr)
            else:
                blank_names.append(name)
                blank += 1
            done += 1
            if (k + 1) % 10 == 0 or k + 1 == n:
                print(f"[s2s] {k + 1}/{n} | text[:60]={text[:60]!r}", flush=True)
        for name in blank_names:  # 16 kHz only if no clip of this shard decoded
            write(name, None, next(iter(seen_rates), 16000))
        print(f"[s2s] done {done} clips ({blank} blank/empty)", flush=True)
