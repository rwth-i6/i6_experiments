"""Base Audex-2B SPEECH-knowledge reference: answer the benchmark's TTS'd question AUDIO through
Audex's OWN native speech-in path (NV-Whisper mel -> NemotronDenseAudex audio model -> text), so we
get an apples-to-(nearly)-apples base number vs the AudexDuplex graft.

Modality: speech-IN (the exact same `tts_output/<i>.wav` question audio the other speech models get),
TEXT-out (Audex's audio-QA path answers in text). Emits a ``transcriptions.jsonl`` in the EXACT schema
``LLMGrading`` consumes (``{question, answer, aliases, category, transcription}``) with ``transcription``
= Audex's spoken-question text answer -- so it drops straight into the existing grading path, NO Whisper
(the model already answers in text). Clip index i = ``reference_data`` row i = ``<i>.wav`` (mirrors
``WhisperTranscription`` / ``ReferenceStringTranscription`` indexing exactly).

The preprocessing (NV-Whisper features, ``<sound>`` placeholder expansion, prompt template) is Audex's
own, transcribed verbatim from the model's ``inference_scripts_hf/audio_utils.py``.
"""

import glob
import json
import os

from sisyphus import Job, Task, tk

_AUDEX_REPO = "nvidia/Nemotron-Labs-Audex-2B"

# --- Audex audio-QA preprocessing (verbatim from the model's inference_scripts_hf/audio_utils.py) ----
SOUND_PLACEHOLDER = "<sound>"
SOUND_TOKEN = "<so_embedding>"
SOUND_START_TOKEN = "<so_start>"
SOUND_END_TOKEN = "<so_end>"
IM_END_TOKEN = "<|im_end|>"
# The spoken question is in the AUDIO (<sound>); the text prompt is a generic instruction ONLY --
# passing the question text here would let the model READ it (not a speech-in test; not comparable to the FD models).
_QA_INSTRUCTION = "Answer the spoken question."


def _build_prompt_template(prompt: str) -> str:
    # reasoning=False -> answer directly, no <think> trace (concise spoken answer).
    return f"<|im_start|>user\n<sound>\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def _expand_sound_placeholder(prompt: str, num_embeddings: int) -> str:
    assert prompt.count(SOUND_PLACEHOLDER) == 1, prompt.count(SOUND_PLACEHOLDER)
    replacement = SOUND_START_TOKEN + (SOUND_TOKEN * num_embeddings) + SOUND_END_TOKEN
    return prompt.replace(SOUND_PLACEHOLDER, replacement)


def _split_answer(response: str) -> str:
    """Text after ``</think>`` (reasoning off => empty think), control tokens stripped."""
    s = response.split(IM_END_TOKEN, 1)[0]
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[1]
    return s.strip()


class AudexSpeechQA(Job):
    """Base Audex-2B answers TTS'd benchmark questions in its native speech-in path -> text answers."""

    __sis_hash_exclude__ = {"max_new_tokens": 256, "batch_report": 20, "code_version": 1}

    def __init__(
        self,
        *,
        in_dir: tk.Path,  # the shared ChatterboxSingleSpeakerInference tts_output (<i>.wav question audio)
        reference_data: tk.Path,  # the subsampled dataset (question/answer/aliases/category), row i = <i>.wav
        max_new_tokens: int = 256,
        batch_report: int = 20,
        code_version: int = 1,
    ):
        self.in_dir = in_dir
        self.reference_data = reference_data
        self.max_new_tokens = max_new_tokens
        self.batch_report = batch_report
        self.code_version = code_version
        self.out_json = self.output_path("transcriptions.jsonl")
        # Lean HF (2B + NV-Whisper encoder); wav read via soundfile (no torchcodec/FFmpeg) -> c25g-safe.
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _resolve(self, subdir: str) -> str:
        from huggingface_hub import snapshot_download

        # Prefer the already-staged shared cache; else fetch just this subfolder.
        cached = glob.glob(
            f"/hpcwork/p0023999/common_hf_home/hub/models--nvidia--Nemotron-Labs-Audex-2B/snapshots/*/{subdir}"
        )
        cached = [c for c in cached if os.path.isdir(c)]
        if cached:
            return sorted(cached)[-1]
        root = snapshot_download(_AUDEX_REPO, allow_patterns=[f"{subdir}/*"])
        return os.path.join(root, subdir)

    def run(self):
        import librosa
        import numpy as np
        import torch
        from datasets import load_from_disk
        from transformers import (
            AutoConfig,
            AutoFeatureExtractor,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        model_dir = self._resolve("checkpoint_folder_full")
        whisper_dir = self._resolve("nv-whisper")
        print(f"[audex-speechqa] model={model_dir}\n[audex-speechqa] whisper={whisper_dir}", flush=True)

        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        feat = AutoFeatureExtractor.from_pretrained(whisper_dir)
        model = (
            AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
        )
        clip_dur = float(getattr(config, "sound_clip_duration", 30.0))
        sound_emb = int(getattr(config, "sound_embedding_size", 750))
        sr = int(getattr(config, "sound_target_rate", 16000))
        eos_id = tok.convert_tokens_to_ids(IM_END_TOKEN)
        if eos_id is None or eos_id == tok.unk_token_id:
            eos_id = getattr(config, "eos_token_id", None)

        ds = load_from_disk(self.reference_data.get())
        in_dir = self.in_dir.get()
        n = len(ds)
        print(f"[audex-speechqa] {n} questions", flush=True)

        def whisper_features(wav_path):
            audio, _sr = librosa.load(wav_path, sr=sr, mono=True)
            audio = np.asarray(audio, dtype=np.float32)
            m = float(np.abs(audio).max()) if audio.size else 0.0
            if m > 1.0:
                audio = audio / m
            clip_n = int(round(sr * clip_dur))
            if audio.size == 0:
                audio = np.zeros(1, dtype=np.float32)
            import math

            nclips = max(1, math.ceil(audio.shape[0] / clip_n))
            clips = []
            for j in range(nclips):
                c = audio[j * clip_n : (j + 1) * clip_n]
                if c.shape[0] < clip_n:
                    c = np.pad(c, (0, clip_n - c.shape[0]))
                clips.append(c.astype(np.float32))
            f = feat(clips, sampling_rate=sr, return_tensors="pt", padding="max_length", return_attention_mask=False)
            return f.input_features

        results = []
        missing = 0
        for i, ex in enumerate(ds):
            wav = os.path.join(in_dir, f"{i}.wav")
            if not os.path.exists(wav):
                missing += 1
                continue
            input_features = whisper_features(wav)
            num_emb = input_features.shape[0] * sound_emb
            q = ex["question"]
            prompt = _expand_sound_placeholder(_build_prompt_template(_QA_INSTRUCTION), num_emb)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
            with torch.inference_mode():
                out = model.generate(
                    input_ids=enc.input_ids.cuda(),
                    attention_mask=enc.attention_mask.cuda(),
                    input_features=input_features.cuda().to(torch.bfloat16),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=eos_id,
                    pad_token_id=tok.pad_token_id or getattr(config, "pad_token_id", 0),
                )
            dec = tok.decode(out[0, enc.input_ids.shape[-1] :], skip_special_tokens=False)
            answer = _split_answer(dec)
            results.append(
                {
                    "question": q,
                    "answer": ex["answer"],
                    "aliases": ex["aliases"],
                    "category": ex.get("category", "unknown"),
                    "transcription": answer,
                }
            )
            if (i + 1) % self.batch_report == 0 or i + 1 == n:
                print(f"[audex-speechqa] {i + 1}/{n} | last: {answer[:80]!r}", flush=True)

        n_empty = sum(1 for r in results if not r["transcription"].strip())
        assert len(results) >= max(1, (n - missing) // 2), f"too few answers: {len(results)}/{n} (missing {missing})"
        with open(self.out_json.get(), "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(
            f"[audex-speechqa] wrote {len(results)} rows ({n_empty} empty, {missing} missing wavs)",
            flush=True,
        )
