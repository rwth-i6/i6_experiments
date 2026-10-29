"""Audex-2B's audio-QA path, shared by ``AudexSpeechQA`` (speech in, text out) and ``AudexSpeechS2S`` (speech in,
speech out): checkpoint resolution, NV-Whisper features, the prompt and the answer split.

Transcribed verbatim from the model's ``inference_scripts_hf/audio_utils.py``. One copy so both base-Audex numbers
load the same snapshot and run the same protocol.
"""

import math
import os

AUDEX_REPO = "nvidia/Nemotron-Labs-Audex-2B"
HF_HUB_CACHE = "/hpcwork/p0023999/common_hf_home/hub"

SOUND_PLACEHOLDER = "<sound>"
SOUND_TOKEN = "<so_embedding>"
SOUND_START_TOKEN = "<so_start>"
SOUND_END_TOKEN = "<so_end>"
IM_END_TOKEN = "<|im_end|>"
# The spoken question is in the AUDIO (<sound>); the text prompt is a generic instruction ONLY -- passing the
# question text here would let the model READ it (not a speech-in test; not comparable to the FD models).
QA_INSTRUCTION = "Answer the spoken question."
#: Audex's audio-QA sampling (audio_utils.py)
QA_SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.9)


def resolve(subdir: str) -> str:
    """The repo subfolder in the shared hub cache (the current snapshot; downloads only missing files)."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(AUDEX_REPO, allow_patterns=[f"{subdir}/*"], cache_dir=HF_HUB_CACHE)
    return os.path.join(root, subdir)


def qa_prompt(prompt: str = QA_INSTRUCTION) -> str:
    # reasoning=False -> answer directly, no <think> trace (concise spoken answer).
    return f"<|im_start|>user\n<sound>\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def expand_sound(prompt: str, num_embeddings: int) -> str:
    assert prompt.count(SOUND_PLACEHOLDER) == 1, prompt.count(SOUND_PLACEHOLDER)
    return prompt.replace(SOUND_PLACEHOLDER, SOUND_START_TOKEN + (SOUND_TOKEN * num_embeddings) + SOUND_END_TOKEN)


def split_answer(response: str) -> str:
    """Text after ``</think>`` (reasoning off => empty think), control tokens stripped."""
    s = response.split(IM_END_TOKEN, 1)[0]
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[1]
    return s.strip()


def whisper_features(clip, *, feat, sr: int, clip_dur: float):
    """``clip`` = ``(samples, sample_rate)`` from ``clip_store.open_clips`` -> NV-Whisper input features, one row per
    ``clip_dur`` window (resampled to ``sr``, peak-normalised only above 1.0, zero-padded)."""
    import librosa
    import numpy as np

    audio, clip_sr = clip
    audio = np.asarray(audio, dtype=np.float32)
    if clip_sr != sr:
        audio = librosa.resample(audio, orig_sr=clip_sr, target_sr=sr)
    audio = np.asarray(audio, dtype=np.float32)
    m = float(np.abs(audio).max()) if audio.size else 0.0
    if m > 1.0:
        audio = audio / m
    clip_n = int(round(sr * clip_dur))
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)
    clips = []
    for j in range(max(1, math.ceil(audio.shape[0] / clip_n))):
        c = audio[j * clip_n : (j + 1) * clip_n]
        if c.shape[0] < clip_n:
            c = np.pad(c, (0, clip_n - c.shape[0]))
        clips.append(c.astype(np.float32))
    f = feat(clips, sampling_rate=sr, return_tensors="pt", padding="max_length", return_attention_mask=False)
    return f.input_features


class AudioQA:
    """The loaded audio-QA model: ``answer(clip)`` -> Audex's text answer to the spoken question."""

    def __init__(self, *, max_new_tokens: int):
        import torch
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer

        self.model_dir = resolve("checkpoint_folder_full")
        self.whisper_dir = resolve("nv-whisper")
        self.tok = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True)
        self.feat = AutoFeatureExtractor.from_pretrained(self.whisper_dir)
        self.model = (
            AutoModelForCausalLM.from_pretrained(self.model_dir, trust_remote_code=True, dtype=torch.bfloat16)
            .eval()
            .cuda()
        )
        self.clip_dur = float(getattr(self.config, "sound_clip_duration", 30.0))
        self.sound_emb = int(getattr(self.config, "sound_embedding_size", 750))
        self.sr = int(getattr(self.config, "sound_target_rate", 16000))
        eos_id = self.tok.convert_tokens_to_ids(IM_END_TOKEN)
        if eos_id is None or eos_id == self.tok.unk_token_id:
            eos_id = getattr(self.config, "eos_token_id", None)
        self.eos_id = eos_id
        self.pad_id = self.tok.pad_token_id or getattr(self.config, "pad_token_id", 0)
        self.max_new_tokens = int(max_new_tokens)

    def answer(self, clip) -> str:
        import torch

        feats = whisper_features(clip, feat=self.feat, sr=self.sr, clip_dur=self.clip_dur)
        prompt = expand_sound(qa_prompt(), feats.shape[0] * self.sound_emb)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=enc.input_ids.cuda(),
                attention_mask=enc.attention_mask.cuda(),
                input_features=feats.cuda().to(torch.bfloat16),
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.eos_id,
                pad_token_id=self.pad_id,
                **QA_SAMPLING,
            )
        return split_answer(self.tok.decode(out[0, enc.input_ids.shape[-1] :], skip_special_tokens=False))
