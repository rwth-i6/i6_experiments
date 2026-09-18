"""Worker: how much does storing audio as MIMI CODES cost us if we still want augmentation?

Runs in a job venv (moshi_family_venv). Per the standing exception, it carries its own small
helpers and imports nothing from sisyphus / i6_experiments.

THE QUESTION (backlog E19). If a corpus is stored as mimi codes instead of waveforms (~12 KB vs
~11.5 MB per 60 s stereo window), a waveform augmentation -- reverb, additive noise, gain -- can no
longer be applied to the stored thing directly. The candidate fix is decode -> augment -> re-encode.
That adds a CODEC ROUND TRIP the waveform pipeline never pays, and the whole design turns on whether
that round trip is small compared to the augmentation itself.

So this measures three quantities, and only their RATIO matters:

  FLOOR    d(c0, E(D(c0)))              codec round trip alone, no augmentation.
                                        How much the codec moves the codes by itself.
  SIGNAL   d(c0, E(A(x)))               what the augmentation does at all.
                                        If this is ~0 the augmentation is not worth having.
  PENALTY  d(E(A(x)), E(A(D(c0))))      augmenting the ORIGINAL vs augmenting the DECODE.
                                        This is exactly the price of storing codes.

  Decision: PENALTY << SIGNAL  =>  storing codes does not meaningfully damage augmentation.
            PENALTY ~ SIGNAL   =>  the round trip destroys what the augmentation adds.

⚠ Reporting SIGNAL alone is the trap. "Augmentation changes 40% of codes" sounds decisive and says
nothing without the FLOOR (the codec may move 35% on its own) and the PENALTY.

Audio-domain distances use LOG-MEL L1 and SI-SDR, never raw waveform L2: a neural codec
reconstructs perceptually and is free to move phase, so waveform L2 reports a large number for an
inaudible change and would make the codec look catastrophic.
"""

import argparse
import io
import json
import os
import random
import time

import numpy as np
import soundfile as sf
import torch


# --------------------------------------------------------------------------------------------
# Augmentations (waveform domain, numpy). Self-contained on purpose: this probe must not depend on
# a noise corpus or an RIR set being staged, or it cannot be run on demand.
# --------------------------------------------------------------------------------------------
def aug_gain(x, rng, db=None):
    db = rng.uniform(-24.0, 15.0) if db is None else db
    return (x * (10.0 ** (db / 20.0))).astype(np.float32)


def aug_noise(x, rng, snr_db=20.0):
    n = rng.standard_normal(x.shape).astype(np.float32)
    px, pn = float(np.mean(x**2)) + 1e-12, float(np.mean(n**2)) + 1e-12
    n *= np.sqrt(px / (pn * (10.0 ** (snr_db / 10.0))))
    return (x + n).astype(np.float32)


def _rir(rng, sr, rt60=0.3):
    """Synthetic RIR: exponentially decaying noise. Not a measured room, but the right SHAPE -- a
    dense decaying tail is what makes reverb non-commutative with a non-linear encoder."""
    n = int(rt60 * sr)
    h = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n).astype(np.float32)
    h[0] += 1.0
    return (h / (np.linalg.norm(h) + 1e-9)).astype(np.float32)


def aug_reverb(x, rng, sr=24000, rt60=0.3):
    h = _rir(rng, sr, rt60)
    y = np.convolve(x, h, mode="full")[: x.shape[0]]
    peak = float(np.max(np.abs(y))) + 1e-9
    return (y / peak * (float(np.max(np.abs(x))) + 1e-9)).astype(np.float32)


def aug_echo(x, rng, sr=24000, delay_s=0.12, decay=0.4):
    """The Moshi-echo analogue: a delayed attenuated copy (the model hearing its own output)."""
    d = int(delay_s * sr)
    y = x.copy()
    if d < x.shape[0]:
        y[d:] += decay * x[: x.shape[0] - d]
    return y.astype(np.float32)


AUGS = {
    "gain_-12dB": lambda x, r, sr: aug_gain(x, r, db=-12.0),
    "noise_snr20": lambda x, r, sr: aug_noise(x, r, snr_db=20.0),
    "noise_snr10": lambda x, r, sr: aug_noise(x, r, snr_db=10.0),
    "reverb_rt60_0.3": lambda x, r, sr: aug_reverb(x, r, sr=sr, rt60=0.3),
    "echo_120ms": lambda x, r, sr: aug_echo(x, r, sr=sr),
}


# --------------------------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------------------------
def code_agreement(a, b):
    """Fraction of (codebook, frame) positions where two code tensors are identical.

    Returned overall AND for codebook 0 separately: in an RVQ the first book is the coarse/semantic
    one and the later books are fine residual detail, so a change confined to book 7 means something
    very different from the same change in book 0. An overall mean hides that completely.
    """
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n], b[..., :n]
    eq = (a == b).float()
    per_book = eq.mean(dim=-1).flatten().tolist()
    return float(eq.mean()), per_book


def log_mel(x, sr, n_mels=64, n_fft=1024, hop=256):
    import torch as T

    w = T.hann_window(n_fft)
    S = T.stft(T.from_numpy(x).float(), n_fft=n_fft, hop_length=hop, window=w, return_complex=True)
    mag = S.abs() ** 2
    # A plain triangular-free mel-ish pooling: group FFT bins into n_mels log-spaced bands. Enough
    # for a RELATIVE comparison, which is all this probe makes.
    edges = np.geomspace(1, mag.shape[0] - 1, n_mels + 1).astype(int)
    bands = [mag[edges[i] : max(edges[i] + 1, edges[i + 1])].mean(dim=0) for i in range(n_mels)]
    return T.log(T.stack(bands) + 1e-8)


def mel_l1(x, y, sr):
    a, b = log_mel(x, sr), log_mel(y, sr)
    n = min(a.shape[-1], b.shape[-1])
    return float((a[..., :n] - b[..., :n]).abs().mean())


def si_sdr(ref, est):
    n = min(ref.shape[0], est.shape[0])
    ref, est = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    ref = ref - ref.mean()
    est = est - est.mean()
    a = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-12))
    proj = a * ref
    noise = est - proj
    return float(10 * np.log10((np.dot(proj, proj) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


# --------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="annotated corpus dir (load_from_disk)")
    p.add_argument("--out_json", required=True)
    p.add_argument("--out_txt", required=True)
    p.add_argument("--n_rows", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration_sec", type=float, default=60.0)
    p.add_argument("--hf_repo", default="kyutai/moshiko-pytorch-bf16")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from datasets import load_from_disk
    from moshi_family.models.loaders import CheckpointInfo

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ci = CheckpointInfo.from_hf_repo(args.hf_repo)
    mimi = ci.get_mimi(device=args.device)
    sr = int(mimi.sample_rate)
    print(f"[probe] mimi loaded: sr={sr} frame_rate={mimi.frame_rate}", flush=True)

    table = load_from_disk(args.corpus).data.table
    total = table.num_rows
    # RANDOM sample with a stated seed: every corpus here is a shard merge in sorted key order, so
    # range(N) reads one template rather than the corpus.
    idx = random.Random(args.seed).sample(range(total), min(args.n_rows, total))
    print(f"[probe] sampling {len(idx)} of {total} rows at random (seed {args.seed})", flush=True)

    def load_channel(i, col):
        cell = table.column(col)[i].as_py()
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        y, sr0 = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr0 != sr:
            import sphn

            y = sphn.resample(y.astype(np.float32), sr0, sr)
        return np.asarray(y, dtype=np.float32)

    cols = table.column_names
    a_col = "audio_assistant" if "audio_assistant" in cols else None
    if a_col is None:
        raise SystemExit(f"no audio_assistant column; have {cols}")

    end = int(args.duration_sec * sr)

    def enc(x):
        w = torch.from_numpy(x[None, None, :]).float().to(args.device)
        with torch.no_grad():
            return mimi.encode(w)[0].cpu()

    def dec(c):
        with torch.no_grad():
            return mimi.decode(c[None].to(args.device))[0, 0].cpu().numpy().astype(np.float32)

    # ---- COST BENCHMARK -- done properly, separate from the distance loop ----------------------
    # The first version of this probe reported decode as 23x CHEAPER than encode (23.1 ms vs
    # 538.6 ms). That contradicts the architecture -- mimi's encoder and decoder are near mirrors,
    # both transformers run over the same 750 frames, and the only genuinely asymmetric piece is the
    # quantizer (argmin search in, embedding lookup out), which is a few GFLOP. In neural codecs the
    # decoder is usually the MORE expensive half, so a 23x decoder advantage runs backwards from
    # expectation.
    #
    # It rested on an unvalidated assumption: the decoded LENGTH was never checked. Both audio
    # metrics truncate to min(len(a), len(b)), so a short decode would be fast AND produce entirely
    # plausible numbers -- and would also inflate the code-change FLOOR by comparing a misaligned
    # signal. So: assert the length, discard warmup, and synchronise BEFORE each timer as well as
    # after (otherwise a timer absorbs whatever was still queued from the previous call).
    def _sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def _time(fn, reps=10, warmup=3):
        for _ in range(warmup):
            fn()
        _sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        _sync()
        return 1000.0 * (time.perf_counter() - t0) / reps

    bench = {}
    _x0 = None
    for _i in idx:
        try:
            _c = load_channel(_i, a_col)[:end]
        except Exception:  # noqa: BLE001
            continue
        if _c.shape[0] >= sr:
            _x0 = _c
            break
    if _x0 is None:
        raise SystemExit("no readable row long enough to benchmark")

    _w1 = torch.from_numpy(_x0[None, None, :]).float().to(args.device)
    # Batch 2 is what TRAINING actually does: encode_stereo_window passes the two channels as
    # [2, 1, T]. Reporting only batch 1 overstates the per-window cost of the real path.
    _w2 = torch.cat([_w1, _w1], dim=0)

    with torch.no_grad():
        _codes1 = mimi.encode(_w1)
        bench["encode_b1_ms"] = _time(lambda: mimi.encode(_w1))
        bench["encode_b2_ms"] = _time(lambda: mimi.encode(_w2))
        bench["decode_b1_ms"] = _time(lambda: mimi.decode(_codes1))

        # THE ASSERT. If this fails, every distance in this report is computed on misaligned
        # signals and the whole measurement is void.
        _dec = mimi.decode(_codes1)
        bench["decode_out_samples"] = int(_dec.shape[-1])
        bench["encode_in_samples"] = int(_w1.shape[-1])
        bench["decode_length_ratio"] = float(_dec.shape[-1]) / float(_w1.shape[-1])

        # Split the encode: encode_to_latent(quantize=False) is SEANet + transformer, so the
        # difference from a full encode isolates the RVQ argmin -- the one asymmetric piece.
        try:
            bench["encode_to_latent_b1_ms"] = _time(lambda: mimi.encode_to_latent(_w1, quantize=False))
            bench["quantizer_b1_ms"] = bench["encode_b1_ms"] - bench["encode_to_latent_b1_ms"]
        except Exception as e:  # noqa: BLE001
            bench["encode_to_latent_error"] = f"{type(e).__name__}: {e}"[:120]

    print("[probe] cost benchmark:", json.dumps(bench, indent=1), flush=True)
    _ratio = bench["decode_length_ratio"]
    if abs(_ratio - 1.0) > 0.02:
        raise SystemExit(
            f"decode returned {bench['decode_out_samples']} samples for "
            f"{bench['encode_in_samples']} in (ratio {_ratio:.4f}). Every distance below would be "
            "computed on misaligned signals -- refusing to report numbers. Fix the round trip first."
        )

    acc = {
        k: {
            "floor": [],
            "signal": [],
            "penalty": [],
            "floor_b0": [],
            "signal_b0": [],
            "penalty_b0": [],
            "mel_codec": [],
            "mel_aug": [],
            "mel_chain": [],
            "sdr_codec": [],
            "sdr_chain": [],
        }
        for k in AUGS
    }
    timing = {"encode_ms": [], "decode_ms": [], "aug_ms": {k: [] for k in AUGS}}
    n_used = 0

    for i in idx:
        try:
            x = load_channel(i, a_col)[:end]
        except Exception as e:  # noqa: BLE001
            print(f"[probe] row {i} unreadable ({type(e).__name__}), skipped", flush=True)
            continue
        if x.shape[0] < sr:  # under a second: nothing to measure
            continue
        n_used += 1

        t0 = time.perf_counter()
        c0 = enc(x)
        torch.cuda.synchronize() if args.device == "cuda" else None
        timing["encode_ms"].append(1000 * (time.perf_counter() - t0))

        t0 = time.perf_counter()
        x_rt = dec(c0)
        torch.cuda.synchronize() if args.device == "cuda" else None
        timing["decode_ms"].append(1000 * (time.perf_counter() - t0))

        c_rt = enc(x_rt)
        f_all, f_b = code_agreement(c0, c_rt)
        mel_c = mel_l1(x, x_rt, sr)
        sdr_c = si_sdr(x, x_rt)

        for name, fn in AUGS.items():
            t0 = time.perf_counter()
            xa = fn(x, rng, sr)
            timing["aug_ms"][name].append(1000 * (time.perf_counter() - t0))
            xa_s = fn(x_rt, rng, sr)

            ca_ideal = enc(xa)
            ca_stored = enc(xa_s)

            s_all, s_b = code_agreement(c0, ca_ideal)
            p_all, p_b = code_agreement(ca_ideal, ca_stored)

            d = acc[name]
            d["floor"].append(1.0 - f_all)
            d["signal"].append(1.0 - s_all)
            d["penalty"].append(1.0 - p_all)
            d["floor_b0"].append(1.0 - f_b[0])
            d["signal_b0"].append(1.0 - s_b[0])
            d["penalty_b0"].append(1.0 - p_b[0])
            d["mel_codec"].append(mel_c)
            d["mel_aug"].append(mel_l1(x, xa, sr))
            d["mel_chain"].append(mel_l1(xa, dec(ca_stored), sr))
            d["sdr_codec"].append(sdr_c)
            d["sdr_chain"].append(si_sdr(xa, dec(ca_stored)))

        if n_used % 8 == 0:
            print(f"[probe] {n_used}/{len(idx)} rows done", flush=True)

    def m(v):
        return float(np.mean(v)) if v else float("nan")

    enc_ms, dec_ms = m(timing["encode_ms"]), m(timing["decode_ms"])
    out = {
        "corpus": args.corpus,
        "n_rows_sampled": len(idx),
        "n_rows_used": n_used,
        "seed": args.seed,
        "duration_sec": args.duration_sec,
        "sample_rate": sr,
        "timing_ms_per_window": {
            "encode": enc_ms,
            "decode": dec_ms,
            "decode_plus_encode": dec_ms + enc_ms,
            "ratio_vs_encode_only": (dec_ms + enc_ms) / enc_ms if enc_ms else None,
            "augment": {k: m(v) for k, v in timing["aug_ms"].items()},
        },
        "augmentations": {
            k: {
                "code_change_floor_codec_only": m(v["floor"]),
                "code_change_signal_augmentation": m(v["signal"]),
                "code_change_penalty_of_storing_codes": m(v["penalty"]),
                "codebook0_floor": m(v["floor_b0"]),
                "codebook0_signal": m(v["signal_b0"]),
                "codebook0_penalty": m(v["penalty_b0"]),
                "penalty_over_signal": (m(v["penalty"]) / m(v["signal"])) if m(v["signal"]) else None,
                "mel_l1_codec_only": m(v["mel_codec"]),
                "mel_l1_augmentation": m(v["mel_aug"]),
                "mel_l1_full_chain_vs_ideal_aug": m(v["mel_chain"]),
                "si_sdr_db_codec_only": m(v["sdr_codec"]),
                "si_sdr_db_full_chain_vs_ideal_aug": m(v["sdr_chain"]),
            }
            for k, v in acc.items()
        },
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)

    L = []
    L.append(f"mimi augmentation probe -- {n_used} windows of {args.duration_sec}s, seed {args.seed}")
    L.append(f"corpus: {args.corpus}")
    L.append("")
    L.append(f"COST per {args.duration_sec}s window (1 channel):")
    L.append(f"  encode            {enc_ms:8.1f} ms")
    L.append(f"  decode            {dec_ms:8.1f} ms")
    L.append(f"  decode+encode     {dec_ms + enc_ms:8.1f} ms   = {(dec_ms + enc_ms) / enc_ms:.2f}x encode alone")
    L.append("")
    L.append("CODE CHANGE (fraction of codebook/frame positions that differ; lower = closer)")
    L.append("  FLOOR   = codec round trip alone, no augmentation")
    L.append("  SIGNAL  = what the augmentation does to the codes")
    L.append("  PENALTY = augmenting the DECODE vs augmenting the ORIGINAL  <- the price of storing codes")
    L.append("")
    L.append(
        f"  {'augmentation':<18}{'FLOOR':>8}{'SIGNAL':>8}{'PENALTY':>9}{'P/S':>7}{'melL1 aug':>11}{'SI-SDR chain':>14}"
    )
    L.append("  " + "-" * 75)
    for k, v in out["augmentations"].items():
        ps = v["penalty_over_signal"]
        L.append(
            f"  {k:<18}{v['code_change_floor_codec_only']:>8.3f}"
            f"{v['code_change_signal_augmentation']:>8.3f}"
            f"{v['code_change_penalty_of_storing_codes']:>9.3f}"
            f"{(ps if ps is not None else float('nan')):>7.2f}"
            f"{v['mel_l1_augmentation']:>11.3f}"
            f"{v['si_sdr_db_full_chain_vs_ideal_aug']:>14.1f}"
        )
    L.append("")
    L.append("READ: PENALTY/SIGNAL (P/S) well under 1 means storing codes costs little relative to")
    L.append("what the augmentation itself does. P/S near or above 1 means the extra codec round trip")
    L.append("moves the codes as much as the augmentation does, and decode->augment->encode is not a")
    L.append("faithful substitute for augmenting the waveform.")
    with open(args.out_txt, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
