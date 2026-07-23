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

import glob
import math
import os
import re

from sisyphus import Job, Task, tk

_AUDEX_REPO = "nvidia/Nemotron-Labs-Audex-2B"

# --- audio-QA preprocessing (verbatim from audio_utils.py) -------------------------------------------
SOUND_PLACEHOLDER, SOUND_TOKEN = "<sound>", "<so_embedding>"
SOUND_START_TOKEN, SOUND_END_TOKEN, IM_END_TOKEN = "<so_start>", "<so_end>", "<|im_end|>"
# --- audiogen constants (verbatim from run_audio_gen_vllm.py) --------------------------------------
_CAUSAL_CHUNK_FRAMES, MIN_TTS_FRAMES, CODEC_FPS = 5, 50, 50
SYSTEM_PROMPT = "You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools."


def _audioqa_prompt(q: str) -> str:
    return f"<|im_start|>user\n<sound>\n{q}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def _expand_sound(prompt: str, n: int) -> str:
    return prompt.replace(SOUND_PLACEHOLDER, SOUND_START_TOKEN + SOUND_TOKEN * n + SOUND_END_TOKEN)


def _answer_text(dec: str) -> str:
    s = dec.split(IM_END_TOKEN, 1)[0]
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[1]
    return s.strip()


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

    def _resolve(self, subdir: str) -> str:
        # Always snapshot_download the subfolder into the roomy shared HF cache: it reuses the
        # pre-staged checkpoint_folder_full and downloads only the MISSING files (e.g. the audiogen
        # weight shards + the causal decoder, which were never fetched -> the FileNotFoundError).
        os.environ.setdefault("HF_HUB_CACHE", "/hpcwork/p0023999/common_hf_home/hub")
        from huggingface_hub import snapshot_download

        root = snapshot_download(_AUDEX_REPO, allow_patterns=[f"{subdir}/*"])
        return os.path.join(root, subdir)

    def run(self):
        import librosa
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import (
            AutoConfig,
            AutoFeatureExtractor,
            AutoModel,
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        from transformers.generation.logits_process import (
            LogitsProcessorList,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
        )

        qa_dir = self._resolve("checkpoint_folder_full")
        gen_dir = self._resolve("checkpoint_folder_audiogen")
        wh_dir = self._resolve("nv-whisper")
        dec_dir = self._resolve("audex_causal_speech_decoder")

        # The audiogen checkpoint SHARES checkpoint_folder_full's weights -- the repo ships only its
        # config/index; prepare_audiogen_vllm_checkpoint.sh symlinks full's 2 shards in. Do the same
        # (idempotent) so from_pretrained finds the weights the index.json references.
        for _shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            _dst = os.path.join(gen_dir, _shard)
            if not os.path.lexists(_dst):
                os.symlink(os.path.join(qa_dir, _shard), _dst)

        # --- audio-QA model (question audio -> text) ---
        qa_tok = AutoTokenizer.from_pretrained(qa_dir, trust_remote_code=True)
        qa_cfg = AutoConfig.from_pretrained(qa_dir, trust_remote_code=True)
        feat = AutoFeatureExtractor.from_pretrained(wh_dir)
        qa = AutoModelForCausalLM.from_pretrained(qa_dir, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
        clip_dur = float(getattr(qa_cfg, "sound_clip_duration", 30.0))
        sound_emb = int(getattr(qa_cfg, "sound_embedding_size", 750))
        sr_in = int(getattr(qa_cfg, "sound_target_rate", 16000))
        qa_eos = qa_tok.convert_tokens_to_ids(IM_END_TOKEN)

        # --- audiogen model (text -> speech codec tokens) + bundled decoder (tokens -> wav) ---
        gen_tok = AutoTokenizer.from_pretrained(gen_dir, trust_remote_code=True)
        gen = AutoModelForCausalLM.from_pretrained(gen_dir, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
        maps = _codec_token_maps(gen_tok)
        speechgen_end = maps.get("speechgen_end")
        decoder = AutoModel.from_pretrained(dec_dir, trust_remote_code=True).cuda().eval()
        print(f"[s2s] loaded qa+audiogen+decoder; {len(maps['speech_codec'])} speechcodec toks", flush=True)

        def whisper_feats(wav):
            audio, _ = librosa.load(wav, sr=sr_in, mono=True)
            audio = np.asarray(audio, dtype=np.float32)
            mx = float(np.abs(audio).max()) if audio.size else 0.0
            if mx > 1.0:
                audio = audio / mx
            cn = int(round(sr_in * clip_dur))
            if audio.size == 0:
                audio = np.zeros(1, dtype=np.float32)
            clips = []
            for j in range(max(1, math.ceil(audio.shape[0] / cn))):
                c = audio[j * cn : (j + 1) * cn]
                if c.shape[0] < cn:
                    c = np.pad(c, (0, cn - c.shape[0]))
                clips.append(c.astype(np.float32))
            return feat(clips, sampling_rate=sr_in, return_tensors="pt", padding="max_length",
                        return_attention_mask=False).input_features

        def answer_text(wav):
            feats = whisper_feats(wav)
            # question is spoken (in the audio); text prompt is a generic instruction ONLY (no question leak)
            prompt = _expand_sound(_audioqa_prompt("Answer the spoken question."), feats.shape[0] * sound_emb)
            enc = qa_tok(prompt, return_tensors="pt", add_special_tokens=False)
            with torch.inference_mode():
                out = qa.generate(input_ids=enc.input_ids.cuda(), attention_mask=enc.attention_mask.cuda(),
                                  input_features=feats.cuda().to(torch.bfloat16), max_new_tokens=self.max_new_tokens,
                                  do_sample=True, temperature=0.7, top_p=0.9, eos_token_id=qa_eos,
                                  pad_token_id=qa_tok.pad_token_id or 0)
            return _answer_text(qa_tok.decode(out[0, enc.input_ids.shape[-1]:], skip_special_tokens=False))

        def synth_wav(text):
            cond = gen_tok(_tts_prompt(text), return_tensors="pt", add_special_tokens=False)
            procs = LogitsProcessorList()
            if self.cfg_scale and self.cfg_scale > 1.0:
                unc = gen_tok(_tts_null_prompt(), return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                procs.append(UnbatchedClassifierFreeGuidanceLogitsProcessor(self.cfg_scale, gen, unconditional_ids=unc))
            with torch.inference_mode():
                out = gen.generate(input_ids=cond.input_ids.cuda(), attention_mask=cond.attention_mask.cuda(),
                                   max_new_tokens=self.max_speech_tokens, do_sample=True, temperature=0.8,
                                   logits_processor=procs, eos_token_id=speechgen_end,
                                   pad_token_id=gen_tok.pad_token_id or gen_tok.eos_token_id)
            ids = _extract_speech_ids(out[0, cond.input_ids.shape[-1]:].tolist(), maps)
            if not ids:
                return None
            session = decoder.create_session(chunk_frames=_CAUSAL_CHUNK_FRAMES)
            chunks = []
            for _sr, ch in session.push([[int(c)] for c in ids]):
                chunks.append(np.asarray(ch).reshape(-1))
            for _sr, ch in session.flush():
                chunks.append(np.asarray(ch).reshape(-1))
            return np.concatenate(chunks).astype(np.float32) if chunks else None

        in_dir = self.in_dir.get()
        wavs = sorted(glob.glob(os.path.join(in_dir, "*.wav")), key=lambda p: int(os.path.basename(p)[:-4]))
        wavs = wavs[self.shard :: self.num_shards]
        out_dir = self.out_dir.get()
        os.makedirs(out_dir, exist_ok=True)
        min_samples = int((MIN_TTS_FRAMES / CODEC_FPS) * 16000)
        n = len(wavs)
        print(f"[s2s] shard {self.shard}/{self.num_shards}: {n} clips", flush=True)
        done = blank = 0
        for k, wav in enumerate(wavs):
            name = os.path.basename(wav)
            text = answer_text(wav)
            w = synth_wav(text) if text.strip() else None
            if w is None or len(w) == 0:
                w = np.zeros(min_samples, dtype=np.float32) + 1e-3
                blank += 1
            elif len(w) < min_samples:
                w = np.pad(w, (0, min_samples - len(w)))
            sf.write(os.path.join(out_dir, name), w, 16000)
            done += 1
            if (k + 1) % 10 == 0 or k + 1 == n:
                print(f"[s2s] {k + 1}/{n} | text[:60]={text[:60]!r}", flush=True)
        print(f"[s2s] done {done} clips ({blank} blank/empty)", flush=True)
