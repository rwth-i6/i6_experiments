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
own, transcribed verbatim from the model's ``inference_scripts_hf/audio_utils.py`` (``audex_audioqa``, shared with
``AudexSpeechS2S``).
"""

import json

from sisyphus import Job, Task, tk

from .audex_audioqa import AudioQA
from .clip_store import open_clips


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

    def run(self):
        from datasets import load_from_disk

        qa = AudioQA(max_new_tokens=self.max_new_tokens)
        print(f"[audex-speechqa] model={qa.model_dir}\n[audex-speechqa] whisper={qa.whisper_dir}", flush=True)

        ds = load_from_disk(self.reference_data.get())
        in_dir = self.in_dir.get()
        n = len(ds)
        print(f"[audex-speechqa] {n} questions", flush=True)

        # Either clip layout (dir of <i>.wav, or one arrow dataset under storage="hf").
        clips = open_clips(in_dir)
        keys = set(clips)  # the index keys; ``i in clips`` would decode the clip
        results = []
        missing = 0
        for i, ex in enumerate(ds):
            if i not in keys:
                missing += 1
                continue
            q = ex["question"]
            answer = qa.answer(clips[i])
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
        # Every present clip yields a row, so the bound must be on the missing clips: otherwise a TTS dir with 40%
        # of its clips gone is scored over the survivors.
        assert missing <= 0.01 * n, f"{missing}/{n} question clips missing from {in_dir}"
        with open(self.out_json.get(), "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(
            f"[audex-speechqa] wrote {len(results)} rows ({n_empty} empty, {missing} missing wavs)",
            flush=True,
        )
