"""
Whisper ASR transcription (batched) for knowledge benchmark Moshi responses.

Used by WhisperTranscription job via subprocess (runs inside the whisper venv).

Uses the HF transformers ASR pipeline with cross-clip batching: clips are streamed
through a generator and the pipeline batches `batch_size` clips per GPU forward pass,
which saturates the GPU far better than openai-whisper's one-file-at-a-time loop.
"""

import argparse
import json
import os

import soundfile as sf
import torch
import torchaudio
from datasets import load_from_disk
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(description="Whisper transcription for benchmark responses")
    parser.add_argument("--in_dir", required=True, help="Directory of Moshi response wav files")
    parser.add_argument("--reference_data", required=True, help="Path to reference HF dataset")
    parser.add_argument("--out_json", required=True, help="Output transcriptions jsonl path")
    parser.add_argument("--model_size", default="large-v3-turbo", help="Whisper model size")
    parser.add_argument("--batch_size", type=int, default=24, help="Cross-clip batch size")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "automatic-speech-recognition",
        model=f"openai/whisper-{args.model_size}",
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        chunk_length_s=30,
    )
    target_sr = pipe.feature_extractor.sampling_rate

    ref_ds = load_from_disk(args.reference_data)

    # Examples that actually have a Moshi response wav, in dataset order.
    valid = [(i, ex) for i, ex in enumerate(ref_ds) if os.path.exists(os.path.join(args.in_dir, f"{i}.wav"))]
    print(f"Transcribing {len(valid)}/{len(ref_ds)} clips (batch_size={args.batch_size})", flush=True)

    def clips():
        for i, _ in valid:
            audio, sr = sf.read(os.path.join(args.in_dir, f"{i}.wav"), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != target_sr:
                audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, target_sr).numpy()
            yield {"raw": audio, "sampling_rate": target_sr}

    n = 0
    with open(args.out_json, "w") as f:
        for (i, example), out in zip(valid, pipe(clips(), batch_size=args.batch_size)):
            f.write(
                json.dumps(
                    {
                        "question": example["question"],
                        "answer": example["answer"],
                        "aliases": example["aliases"],
                        "category": example.get("category", "unknown"),
                        "transcription": out["text"],
                    }
                )
                + "\n"
            )
            n += 1
            if n % 500 == 0:
                print(f"Transcribed {n}/{len(valid)} clips", flush=True)

    print(f"Done. Wrote {n} transcriptions to {args.out_json}")


if __name__ == "__main__":
    main()
