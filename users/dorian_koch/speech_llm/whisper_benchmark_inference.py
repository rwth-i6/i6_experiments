"""
Whisper ASR transcription for knowledge benchmark Moshi responses.

Used by WhisperTranscription job via subprocess (runs inside the whisper venv).

Uses faster-whisper with Silero VAD to strip trailing silence before transcription.
This prevents Whisper from hallucinating phrases ("Thank you.") on the 16-37s of
pure digital silence that follows Moshi speech in the output wavs.
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import load_from_disk
from faster_whisper import WhisperModel, BatchedInferencePipeline


TARGET_SR = 16000


# NOTE: deliberately duplicated across the standalone worker scripts -- see the note in
# chatterbox_inference.py. These run under a job venv with no sisyphus/i6_experiments on the path,
# so they cannot import common.py.
def open_clips(path: str):
    """Return an int-keyed {index: (samples, sample_rate)} view of a clip store.

    Mirrors speech_llm/clip_store.py, duplicated because this worker runs under the whisper job
    venv with no sisyphus / i6_experiments importable. Accepts BOTH layouts: a dir of <i>.wav
    (original) and a single arrow dataset (storage="hf"). Column names must match clip_store.
    """
    p = str(path)
    if os.path.isfile(os.path.join(p, "dataset_info.json")) or os.path.isfile(os.path.join(p, "state.json")):
        from datasets import load_from_disk

        ds = load_from_disk(p)
        return {int(r["index"]): (np.asarray(r["audio"], dtype=np.float32), int(r["sampling_rate"])) for r in ds}
    out = {}
    for name in os.listdir(p) if os.path.isdir(p) else []:
        if name.endswith(".wav") and name[:-4].isdigit():
            out[int(name[:-4])] = os.path.join(p, name)
    return out


def _to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        # An empty reply wav (0 samples) is a legitimate model outcome (e.g. a cascaded
        # backend that stayed silent). Resample chokes on a 0-element tensor, so short-circuit.
        return np.zeros(0, dtype=np.float32)
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, TARGET_SR).numpy()
    return audio.astype(np.float32)


def load_audio(entry) -> np.ndarray:
    """Accept either a wav path (legacy layout) or a decoded (samples, sr) pair (arrow layout)."""
    if isinstance(entry, tuple):
        audio, sr = entry
    else:
        audio, sr = sf.read(entry, dtype="float32")
    return _to_mono_16k(np.asarray(audio, dtype=np.float32), int(sr))


def main():
    parser = argparse.ArgumentParser(description="Whisper transcription for benchmark responses")
    parser.add_argument("--in_dir", required=True, help="Directory of Moshi response wav files")
    parser.add_argument("--reference_data", required=True, help="Path to reference HF dataset")
    parser.add_argument("--out_json", required=True, help="Output transcriptions jsonl path")
    parser.add_argument("--model_size", default="large-v3-turbo", help="Whisper model size")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for faster-whisper internal chunking")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading WhisperModel({args.model_size!r}, device={device!r}, compute_type={compute_type!r})", flush=True)
    model = WhisperModel(args.model_size, device=device, compute_type=compute_type)
    batched = BatchedInferencePipeline(model)

    ref_ds = load_from_disk(args.reference_data)

    # Examples that actually have a Moshi response wav, in dataset order.
    clips = open_clips(args.in_dir)
    valid = [(i, ex) for i, ex in enumerate(ref_ds) if i in clips]
    print(f"Transcribing {len(valid)}/{len(ref_ds)} clips (batch_size={args.batch_size})", flush=True)

    n = 0
    with open(args.out_json, "w") as f:
        for i, example in valid:
            audio = load_audio(clips[i])

            # Sub-10ms / empty audio: a silent (empty) model reply. Score it as an empty
            # hypothesis (a benchmark miss) rather than feeding whisper a degenerate buffer.
            if audio.size < TARGET_SR // 100:
                print(f"clip {i}: empty/near-silent reply ({audio.size} samples) -> empty transcription", flush=True)
                text = ""
            else:
                segments, _ = batched.transcribe(
                    audio,
                    batch_size=args.batch_size,
                    # Silero VAD: skip non-speech regions (incl. trailing silence)
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    # Standard Whisper anti-hallucination thresholds
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                    condition_on_previous_text=False,
                )
                text = "".join(s.text for s in segments).strip()

            f.write(
                json.dumps(
                    {
                        # The CLIP index, carried explicitly. Without it the only way to join a
                        # graded row back to its reply wav / inner monologue is by row POSITION,
                        # which happens to work (this loop preserves dataset order and clip index
                        # IS the enumerate index) but is unverifiable from the file and silently
                        # wrong the moment a clip is missing -- exactly the by-position coupling
                        # the naming rule in CLAUDE.md exists to prevent. Downstream readers key
                        # on names, so an added key is backward compatible.
                        "index": i,
                        "question": example["question"],
                        "answer": example["answer"],
                        "aliases": example["aliases"],
                        "category": example.get("category", "unknown"),
                        "transcription": text,
                    }
                )
                + "\n"
            )
            n += 1
            if n % 100 == 0:
                print(f"Transcribed {n}/{len(valid)} clips", flush=True)

    print(f"Done. Wrote {n} transcriptions to {args.out_json}")


if __name__ == "__main__":
    main()
