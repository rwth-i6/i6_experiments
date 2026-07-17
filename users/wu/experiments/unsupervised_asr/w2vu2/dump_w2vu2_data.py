"""Worker: emit fairseq wav2vec-U 2.0 input data from frozen BEST-RQ features (conda `speech_llm`).

Two modes, both streaming per utterance (never materialising a split in RAM -- `build_units.py`
peaked at 41 GB on 100 h and would die on 960 h):

  fit-mfcc : MFCC(39-d) -> MiniBatchKMeans(64) -> centroids.npy      [CPU only]
  dump     : per split -> {split}.npy [N,512] fp16 + {split}.lengths + {split}.km   [GPU]

Layout matches fairseq's `ExtractedFeaturesDataset`: one .npy holding every frame of every utterance
concatenated, mmap'd at train time, plus a .lengths file with one frame count per line. fp16 is safe
because __getitem__ does `.float()`; it halves the mmap (~8 GB for VAD-trimmed train-clean-100).

Aux target (L_ss, the paper's largest single win: PER 15.9 -> 13.6): 64-cluster k-means over Kaldi
MFCC+delta+delta-delta, encoder-independent by design. **We emit `.km` already at the 25 Hz encoder
rate and VAD-trimmed, aligned 1:1 with the features, so the GAN must run `target_downsample_rate: 1`
(not fairseq's 2).** fairseq computes MFCC on silence-*removed* waveforms at 100 Hz and subsamples
by 2 to reach its 50 Hz feature rate, reconciling any residue by truncating to min(len) -- a hack
that can misalign by a frame. Doing the subsample+trim here instead is exact, and computing the
deltas on the *continuous* waveform avoids the artificial discontinuities that trimming first
would introduce at every excised pause.
"""

import argparse
import os
import sys

import numpy as np
import torch


def _encoder(layer, device):
    # Cross-recipe import: the frozen BEST-RQ wrapper lives in the speech-llm recipe even though it is
    # LLM-independent. Reused rather than copied so this dump and the SAE 2S units share one operator;
    # moving it would rehash live SAE 2S jobs. `.eval()` is what turns SpecAugment off (it follows the
    # wrapper's train/eval flag), matching what SAE_0a measured.
    from speech_llm.prefix_lm.model.definitions.encoders.bestrq import BestRqEncoderV1

    enc = BestRqEncoderV1(encoder_layer=layer).to(device)
    enc.eval()
    return enc


def _vad():
    from rVADfast import rVADfast

    from i6_experiments.users.wu.experiments.unsupervised_asr import vad_port

    return vad_port, rVADfast(vad_threshold=0.4)


def _mfcc_25hz(wav, downsample):
    """Kaldi MFCC 13 + delta + delta-delta = 39-d @100 Hz, subsampled to the encoder rate.

    Kaldi defaults (num_ceps=13, num_mel_bins=23, 25 ms/10 ms, use_energy=False) exactly as
    fairseq's HuBERT `dump_mfcc_feature.py`.
    """
    import torchaudio
    from torchaudio.compliance import kaldi

    x = torch.from_numpy(wav)[None, :]
    m = kaldi.mfcc(waveform=x, sample_frequency=16000, use_energy=False)  # [T100, 13]
    d1 = torchaudio.functional.compute_deltas(m.T[None])[0].T
    d2 = torchaudio.functional.compute_deltas(d1.T[None])[0].T
    return torch.cat([m, d1, d2], dim=1).numpy()[::downsample]  # [~T25, 39]


def _sil_mask(vp, vad, wav, n):
    """25 Hz rVAD silence mask reconciled to n encoder frames (truncate, or pad tail as silence)."""
    sil = vp.rvad_silence_25hz(wav, vad=vad)
    if len(sil) >= n:
        return sil[:n]
    return np.concatenate([sil, np.ones(n - len(sil), dtype=bool)])


def _iter_audio(hf_dir, split, limit=None):
    from datasets import Audio, load_from_disk

    ds = load_from_disk(hf_dir)[split].cast_column("audio", Audio(sampling_rate=16000))
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield str(ex["id"]), np.asarray(ex["audio"]["array"], dtype=np.float32)


def fit_mfcc(args):
    from sklearn.cluster import MiniBatchKMeans

    vp, vad = _vad() if args.trim_silence else (None, None)
    rng = np.random.RandomState(args.seed)
    pool, n_seen = [], 0
    for _, wav in _iter_audio(args.hf_data_dir, args.split, args.limit):
        m = _mfcc_25hz(wav, args.mfcc_downsample)
        if vad is not None:
            m = m[~_sil_mask(vp, vad, wav, len(m))]
        if len(m) == 0:
            continue
        n_seen += len(m)
        pool.append(m.astype(np.float32))
        if sum(len(p) for p in pool) > args.max_fit_vectors * 2:
            X = np.concatenate(pool)
            keep = rng.choice(len(X), args.max_fit_vectors, replace=False)
            pool = [X[keep]]
    X = np.concatenate(pool)
    if len(X) > args.max_fit_vectors:
        X = X[rng.choice(len(X), args.max_fit_vectors, replace=False)]
    print(f"[fit-mfcc] fitting k={args.num_clusters} on {len(X)}/{n_seen} frames, dim={X.shape[1]}", flush=True)

    km = MiniBatchKMeans(
        n_clusters=args.num_clusters, init="k-means++", max_iter=100, batch_size=10000,
        tol=0.0, max_no_improvement=100, n_init=20, reassignment_ratio=0.0,
        random_state=args.seed, compute_labels=False, init_size=None, verbose=0,
    ).fit(X)
    np.save(args.out_centroids, km.cluster_centers_.astype(np.float32))
    used = len(np.unique(km.predict(X)))
    with open(args.out_stats, "w") as f:
        f.write(f"k={args.num_clusters}\nfit_frames={len(X)}\nseen_frames={n_seen}\n"
                f"dim={X.shape[1]}\nnonempty_clusters={used}\ninertia={km.inertia_:.4f}\n"
                f"mfcc_downsample={args.mfcc_downsample}\ntrim_silence={args.trim_silence}\nseed={args.seed}\n")
    print(f"[fit-mfcc] done: {used}/{args.num_clusters} clusters used", flush=True)


def dump(args):
    from npy_append_array import NpyAppendArray

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = _encoder(args.encoder_layer, device)
    vp, vad = _vad() if args.trim_silence else (None, None)
    cent = np.load(args.mfcc_centroids).astype(np.float32)  # [64, 39]
    cent_sq = (cent ** 2).sum(1)

    for p in (args.out_npy,):
        if os.path.exists(p):
            os.remove(p)
    lengths, ids, n_raw_tot, n_kept_tot, n_short = [], [], 0, 0, 0

    with NpyAppendArray(args.out_npy) as npy, open(args.out_km, "w") as kmf:
        for n_utt, (tag, wav) in enumerate(_iter_audio(args.hf_data_dir, args.split, args.limit)):
            audio = torch.from_numpy(wav)[None, :].to(device)
            lens = torch.tensor([wav.shape[0]], dtype=torch.long, device=device)
            with torch.no_grad():
                out = enc.forward(audio, lens)
            t_enc = int(out[-2][0])
            feats = out[0][0, :t_enc].float().cpu().numpy()

            mf = _mfcc_25hz(wav, args.mfcc_downsample)
            t = min(len(feats), len(mf))
            feats, mf = feats[:t], mf[:t]

            if vad is not None:
                keep = ~_sil_mask(vp, vad, wav, t)
                feats, mf = feats[keep], mf[keep]
            n_raw_tot += t
            if len(feats) < args.min_length:  # fairseq's ExtractedFeaturesDataset drops these anyway
                n_short += 1
                continue

            # squared euclidean assignment, same geometry MiniBatchKMeans fit on (no normalization)
            d = (mf ** 2).sum(1, keepdims=True) - 2.0 * mf @ cent.T + cent_sq[None, :]
            km_ids = d.argmin(1)

            npy.append(np.ascontiguousarray(feats, dtype=np.float16))
            kmf.write(" ".join(map(str, km_ids.tolist())) + "\n")
            lengths.append(len(feats))
            ids.append(tag)
            n_kept_tot += len(feats)
            if (n_utt + 1) % 2000 == 0:
                print(f"[dump] {n_utt+1} utts, {n_kept_tot} frames", flush=True)

    with open(args.out_lengths, "w") as f:
        f.write("".join(f"{n}\n" for n in lengths))
    with open(args.out_ids, "w") as f:
        f.write("".join(f"{t}\n" for t in ids))
    drop = 1.0 - n_kept_tot / max(n_raw_tot, 1)
    with open(args.out_stats, "w") as f:
        f.write(f"split={args.split}\nutts={len(lengths)}\nutts_dropped_short={n_short}\n"
                f"frames_kept={n_kept_tot}\nframes_raw={n_raw_tot}\nvad_dropped_frac={drop:.4f}\n"
                f"dim={512}\ndtype=float16\nencoder_layer={args.encoder_layer}\n"
                f"trim_silence={args.trim_silence}\nmfcc_downsample={args.mfcc_downsample}\n"
                f"km_rate=encoder_frame_rate_25hz_aligned_1to1 (=> fairseq target_downsample_rate must be 1)\n")
    print(f"[dump] {args.split}: {len(lengths)} utts, {n_kept_tot} frames, vad dropped {drop:.1%}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["fit-mfcc", "dump"], required=True)
    ap.add_argument("--hf-data-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--encoder-layer", type=int, default=5)
    ap.add_argument("--trim-silence", action="store_true")
    ap.add_argument("--mfcc-downsample", type=int, default=4, help="100 Hz MFCC -> 25 Hz encoder rate")
    ap.add_argument("--min-length", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # fit-mfcc
    ap.add_argument("--num-clusters", type=int, default=64)
    ap.add_argument("--max-fit-vectors", type=int, default=2_000_000)
    ap.add_argument("--out-centroids")
    # dump
    ap.add_argument("--mfcc-centroids")
    ap.add_argument("--out-npy")
    ap.add_argument("--out-lengths")
    ap.add_argument("--out-km")
    ap.add_argument("--out-ids")
    ap.add_argument("--out-stats", required=True)
    args = ap.parse_args()
    (fit_mfcc if args.mode == "fit-mfcc" else dump)(args)


if __name__ == "__main__":
    main()
