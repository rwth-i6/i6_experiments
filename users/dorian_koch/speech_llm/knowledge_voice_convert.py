"""Re-voice the ASSISTANT channel of TTS'd knowledge dialogues with the TTS's own decoder (knowledge corpus v7, user
2026-09-28: "the assistant channel is always the same voice ... Just use random voices ig.").

Worker for ``persona_knowledge.KnowledgeVoiceConvert`` (chatterbox venv, one GPU). Run directly as a script, like
``chatterbox_inference.py``. The v6 corpus was made by Chatterbox-Turbo: per turn, T3 turns the text into S3 speech
tokens conditioned on the voice, then the S3Gen decoder (meanflow, 2 steps) renders the tokens in the voice of its
reference clip. v7 keeps every v6 dialogue -- words, timing, the user channel -- and changes ONE thing, the assistant
voice: per assistant turn, the turn's audio is tokenised with S3Gen's own S3 tokenizer and decoded again by the same
S3Gen with a different speaker condition. That is Chatterbox's own voice conversion (``chatterbox.vc.ChatterboxVC``:
tokenize -> ``s3gen.inference(ref_dict=...)`` -> watermark), run with the Turbo decoder and step count that made v6
and with the conditioning ``ChatterboxTurboTTS.prepare_conditionals`` builds (loudness-normalised reference, first
10 s). Re-synthesising from text instead would change the words' timing and the sampled prosody too (two changes,
and a new Whisper alignment pass), and would cost ~1 GPU-h per 1,000 dialogues against minutes here.

Per row: the voice is one of the pool's (``KnowledgeVoicePool``), assigned balanced and seeded
(``assign_voices``: a seeded permutation of the slice's rows dealt round-robin over the pool). Each assistant turn
(the TTS job's ``turns``, joined by id) is converted and written back at its own start sample, cut / zero-padded to
its original length (the decoder emits 40 ms token frames), so every turn boundary and word time of v6 stays valid;
between turns the channel stays digital silence, as the TTS wrote it. ``torch.manual_seed`` per row (the decoder
samples noise). The user channel is not touched (it is not stored here: the codes job reads it from v6).

Output: an arrow dataset, row-aligned with the annotated input: ``id``, ``audio_assistant`` (``{"bytes", "path"}``,
24 kHz 16-bit FLAC, readable exactly like the annotated corpus's column), ``voice`` (pool index), ``voice_id``,
``n_turns``, ``len_diff_ms`` (per turn: decoder output length - original, before the cut) -- and ``summary.json``
with the speed and a speaker check on the first ``--sim_rows`` rows: Chatterbox's own voice encoder (``ve``) cosine of
each converted turn to its target reference vs to the v6 voice, and of the v6 turn to the target.

Usage: python knowledge_voice_convert.py --annotated <dir> --tts <dir> --pool <dir> --name <slice> --out <dir>
           --summary <json> --orig_voice_wav <wav> [--max_rows N] [--seed 0] [--progress progress.json]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import time
import zlib

import numpy as np

SR = 24000  # S3GEN_SR, and the annotated corpus's rate
S3_SR = 16000


def write_progress(done: int, total: int, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"done": int(done), "total": int(total)}, f)
    os.replace(tmp, path)


def assign_voices(n_rows: int, n_voices: int, seed: int, name: str) -> np.ndarray:
    """``[n_rows]`` pool indices: a seeded permutation of the rows dealt round-robin, so every voice gets
    ``n_rows // n_voices`` or one more of this slice's rows. Seeded per (seed, slice name): slices are independent."""
    perm = np.random.default_rng([int(seed), zlib.crc32(name.encode())]).permutation(n_rows)
    out = np.empty(n_rows, dtype=np.int64)
    out[perm] = np.arange(n_rows) % int(n_voices)
    return out


def row_seed(seed: int, row_id: str) -> int:
    return (int(seed) * 1_000_003 + zlib.crc32(row_id.encode())) % (2**31)


def place(out: np.ndarray, start: int, length: int, wav: np.ndarray) -> int:
    """Write ``wav`` into ``out[start : start + length]`` cut / zero-padded to ``length``; returns len(wav) - length."""
    n = min(length, wav.shape[0])
    out[start : start + n] = wav[:n]
    return int(wav.shape[0] - length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", required=True, help="MoshiAnnotate dataset (id, audio_assistant, ...)")
    ap.add_argument("--tts", required=True, help="ChatterboxInference dataset of the same rows (id, turns)")
    ap.add_argument("--pool", required=True, help="KnowledgeVoicePool dataset")
    ap.add_argument("--name", required=True, help="slice name (seeds the voice assignment)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--orig_voice_wav", required=True, help="the v6 assistant reference (speaker check only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=0, help="pilot: only the first N rows (0 = all)")
    ap.add_argument("--sim_rows", type=int, default=64)
    ap.add_argument("--progress", default="progress.json")
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    import torch
    from chatterbox.models.s3tokenizer import S3_SR as TOK_SR
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from datasets import Dataset, Features, Sequence, Value, load_from_disk

    assert TOK_SR == S3_SR, TOK_SR
    t0 = time.time()
    device = "cuda"
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    assert int(model.sr) == SR, model.sr

    ann = load_from_disk(args.annotated).data.table
    tts = load_from_disk(args.tts).data.table
    turns_by_id = dict(zip(tts.column("id").to_pylist(), tts.column("turns").to_pylist()))
    ids = ann.column("id").to_pylist()
    pool = load_from_disk(args.pool).data.table
    pool_ids = pool.column("voice_id").to_pylist()
    n_voices = len(pool_ids)
    voice_of = assign_voices(len(ids), n_voices, args.seed, args.name)
    n_rows = len(ids) if args.max_rows <= 0 else min(len(ids), args.max_rows)
    print(f"[vc] {args.name}: {n_rows}/{len(ids)} rows, pool of {n_voices} voices", flush=True)

    tmpdir = tempfile.mkdtemp(prefix="kvc_")  # node-local: one reference wav at a time, deleted after use
    conds: dict = {}

    # inference mode everywhere (as chatterbox_inference.main): prepare_conditionals outside it keeps an autograd
    # graph alive in every cached conditioning -- the first pilot ran a 10 GB card out of memory within 2 minutes
    @torch.inference_mode()
    def cond(v: int):
        """The S3Gen conditioning of pool voice ``v``, exactly as ``prepare_conditionals`` builds it (cached)."""
        if v not in conds:
            p = os.path.join(tmpdir, "ref.wav")
            x, sr = sf.read(io.BytesIO(pool.column("ref_flac")[v].as_py()), dtype="float32")
            sf.write(p, x, sr)
            model.prepare_conditionals(p, norm_loudness=True)
            conds[v] = model.conds.gen
            os.unlink(p)
        return conds[v]

    @torch.inference_mode()
    def convert(seg: np.ndarray, ref_dict) -> np.ndarray:
        """``ChatterboxVC.generate``'s steps on one turn (24 kHz float32 in, 24 kHz out), Turbo decoder settings."""
        seg16 = librosa.resample(seg, orig_sr=SR, target_sr=S3_SR)
        audio_16 = torch.from_numpy(seg16).float().to(device)[None,]
        s3_tokens, _ = model.s3gen.tokenizer(audio_16)
        wav, _ = model.s3gen.inference(speech_tokens=s3_tokens, ref_dict=ref_dict, n_cfm_timesteps=2)
        wav = wav.squeeze(0).detach().cpu().numpy()
        return np.asarray(model.watermarker.apply_watermark(wav, sample_rate=SR), dtype=np.float32)

    @torch.inference_mode()
    def embed(x: np.ndarray, sr: int) -> np.ndarray:
        x16 = librosa.resample(x, orig_sr=sr, target_sr=S3_SR) if sr != S3_SR else x
        e = np.asarray(model.ve.embeds_from_wavs([x16], sample_rate=S3_SR)).mean(axis=0)
        return e / (np.linalg.norm(e) + 1e-9)

    orig_ref, orig_sr = sf.read(args.orig_voice_wav, dtype="float32", always_2d=True)
    e_orig = embed(orig_ref.mean(axis=1), orig_sr)
    e_target: dict = {}
    sims = {"conv_to_target": [], "conv_to_v6_voice": [], "v6_to_target": [], "v6_to_v6_voice": []}

    cols = {k: [] for k in ("id", "audio_assistant", "voice", "voice_id", "n_turns", "len_diff_ms")}
    n_turns_total, audio_sec, clipped = 0, 0.0, 0
    for i in range(n_rows):
        rid = ids[i]
        turns = turns_by_id.get(rid)
        if turns is None:
            raise RuntimeError(f"{args.name}: no TTS turns for {rid}")
        x, sr = sf.read(io.BytesIO(ann.column("audio_assistant")[i].as_py()["bytes"]), dtype="float32", always_2d=True)
        assert sr == SR, (rid, sr)
        x = x.mean(axis=1)
        v = int(voice_of[i])
        ref_dict = cond(v)
        torch.manual_seed(row_seed(args.seed, rid))
        out = np.zeros_like(x)
        diffs = []
        for spk, st, en in zip(turns["speaker"], turns["start_time"], turns["end_time"]):
            if spk != "assistant":
                continue
            s, e = int(round(float(st) * SR)), min(int(round(float(en) * SR)), x.shape[0])
            if e - s < int(0.1 * SR):
                continue
            wav = convert(x[s:e], ref_dict)
            diffs.append(round(1000.0 * place(out, s, e - s, wav) / SR, 1))
            if i < args.sim_rows and e - s >= SR:  # speaker check on turns >= 1 s
                if v not in e_target:
                    rx, rsr = sf.read(io.BytesIO(pool.column("ref_flac")[v].as_py()), dtype="float32")
                    e_target[v] = embed(rx, rsr)
                ec, e6 = embed(out[s:e], SR), embed(x[s:e], SR)
                sims["conv_to_target"].append(float(ec @ e_target[v]))
                sims["conv_to_v6_voice"].append(float(ec @ e_orig))
                sims["v6_to_target"].append(float(e6 @ e_target[v]))
                sims["v6_to_v6_voice"].append(float(e6 @ e_orig))
        clipped += int(np.abs(out).max(initial=0.0) > 1.0)
        np.clip(out, -1.0, 1.0, out=out)
        buf = io.BytesIO()
        sf.write(buf, out, SR, format="FLAC", subtype="PCM_16")
        cols["id"].append(rid)
        cols["audio_assistant"].append({"bytes": buf.getvalue(), "path": None})
        cols["voice"].append(v)
        cols["voice_id"].append(pool_ids[v])
        cols["n_turns"].append(len(diffs))
        cols["len_diff_ms"].append(diffs)
        n_turns_total += len(diffs)
        audio_sec += x.shape[0] / SR
        if (i + 1) % 200 == 0 or i + 1 == n_rows:
            write_progress(i + 1, n_rows, args.progress)
            el = time.time() - t0
            print(
                f"[vc] {i + 1}/{n_rows} rows, {n_turns_total} turns, {el:.0f} s, {(i + 1) / el:.2f} rows/s", flush=True
            )

    feats = Features(
        {
            "id": Value("string"),
            "audio_assistant": {"bytes": Value("binary"), "path": Value("string")},
            "voice": Value("int32"),
            "voice_id": Value("string"),
            "n_turns": Value("int32"),
            "len_diff_ms": Sequence(Value("float32")),
        }
    )
    Dataset.from_dict(cols, features=feats).save_to_disk(args.out)
    d = np.asarray([x for r in cols["len_diff_ms"] for x in r], dtype=np.float64)
    counts = np.bincount(np.asarray(cols["voice"]), minlength=n_voices)
    summary = {
        "name": args.name,
        "rows": n_rows,
        "rows_in": len(ids),
        "turns": n_turns_total,
        "audio_hours": round(audio_sec / 3600, 3),
        "clipped_rows": clipped,
        "voices_used": int((counts > 0).sum()),
        "rows_per_voice_min_max": [int(counts.min()), int(counts.max())],
        "len_diff_ms_p1_50_99": np.percentile(d, [1, 50, 99]).round(1).tolist() if d.size else None,
        "speaker_check_turns": len(sims["conv_to_target"]),
        "speaker_cos_mean": {k: round(float(np.mean(v)), 4) for k, v in sims.items() if v},
        "speaker_conv_closer_to_target": round(
            float(np.mean(np.asarray(sims["conv_to_target"]) > np.asarray(sims["conv_to_v6_voice"])))
            if sims["conv_to_target"]
            else float("nan"),
            4,
        ),
        "seconds": round(time.time() - t0, 1),
        "rows_per_s": round(n_rows / max(time.time() - t0, 1e-9), 3),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[vc] done: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
