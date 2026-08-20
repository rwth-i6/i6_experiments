"""Decode one framed HF/Ogg shard with the frozen SAE §1d CTC+KenLM stack.

This is deliberately a small subprocess entry point.  The Sisyphus parent runs in the
``speech_llm`` environment (which can read Hugging Face Arrow), while this worker runs in the
isolated ``w2vu`` environment (which provides fairseq and flashlight).  The boundary is one packed
binary stream per worker, never a per-utterance audio tree.
"""

from __future__ import annotations

import argparse
import io
import json
import struct

import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils

from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.packed_audio import (
    reconstruct_pcm16,
)

from examples.speech_recognition.new.decoders.decoder import Decoder
from examples.speech_recognition.new.infer import DecodingConfig


_HEADER = struct.Struct(
    "<QQQ"
)  # global row index, uid byte count, encoded-audio byte count


def _read_exact(stream, n: int) -> bytes:
    data = stream.read(n)
    if len(data) != n:
        raise EOFError(f"truncated packed shard: wanted {n} bytes, got {len(data)}")
    return data


def iter_records(path: str):
    with open(path, "rb") as f:
        while True:
            raw = f.read(_HEADER.size)
            if not raw:
                return
            if len(raw) != _HEADER.size:
                raise EOFError("truncated packed-shard header")
            row, uid_len, audio_len = _HEADER.unpack(raw)
            uid = _read_exact(f, uid_len).decode("utf-8")
            audio = _read_exact(f, audio_len)
            yield row, uid, audio


def _decode_batch(*, records, task, models, decoder):
    waves = []
    for _row, uid, encoded in records:
        wave, sample_rate = sf.read(io.BytesIO(encoded), dtype="float32")
        if sample_rate != 16_000 or wave.ndim != 1:
            raise ValueError(
                f"{uid}: expected mono 16 kHz audio, got {wave.shape} at {sample_rate} Hz"
            )
        wave = reconstruct_pcm16(wave)
        source = torch.from_numpy(wave)
        waves.append(F.layer_norm(source, source.shape))

    max_len = max(map(len, waves))
    source = torch.zeros((len(waves), max_len), dtype=torch.float32)
    padding = torch.ones((len(waves), max_len), dtype=torch.bool)
    for i, wave in enumerate(waves):
        source[i, : len(wave)] = wave
        padding[i, : len(wave)] = False
    sample = {
        "id": torch.arange(len(waves), device="cuda"),
        "net_input": {"source": source.cuda(), "padding_mask": padding.cuda()},
    }
    with torch.inference_mode():
        hyps = task.inference_step(generator=decoder, models=models, sample=sample)
    return [
        (row, uid, " ".join(hyps[i][0]["words"]))
        for i, (row, uid, _audio) in enumerate(records)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--arpa", required=True)
    parser.add_argument("--beam", type=int, default=500)
    parser.add_argument("--lm-weight", type=float, default=2.0)
    parser.add_argument("--word-score", type=float, default=-1.0)
    parser.add_argument("--max-tokens", type=int, default=1_100_000)
    args = parser.parse_args()

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.checkpoint]
    )
    if not saved_cfg.task.normalize:
        raise ValueError(
            "packed reader is specified for the canonical normalize=True §1d checkpoint"
        )
    for model in models:
        model.cuda().eval()
        model.make_generation_fast_()
    decoder = Decoder(
        DecodingConfig(
            type="kenlm",
            lexicon=args.lexicon,
            lmpath=args.arpa,
            beam=args.beam,
            lmweight=args.lm_weight,
            wordscore=args.word_score,
        ),
        task.target_dictionary,
    )

    output = []
    batch = []
    max_len = 0
    for record in iter_records(args.packed):
        # Fairseq's max_tokens constraint is padded frames: batch size times longest waveform.
        with sf.SoundFile(io.BytesIO(record[2])) as snd:
            length = len(snd)
        proposed_max = max(max_len, length)
        if batch and (len(batch) + 1) * proposed_max > args.max_tokens:
            output.extend(
                _decode_batch(records=batch, task=task, models=models, decoder=decoder)
            )
            batch, max_len = [], 0
        batch.append(record)
        max_len = max(max_len, length)
    if batch:
        output.extend(
            _decode_batch(records=batch, task=task, models=models, decoder=decoder)
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
