"""Worker: emit fairseq wav2vec-U 2.0 input data from a frozen speech encoder (conda `speech_llm`).

Two encoders, selected by `--encoder-type`:
  bestrq    : the SAE frozen BEST-RQ wrapper, 512-d @ 25 Hz (default; SAE_1c BEST-RQ arm).
  wav2vec2  : an HF `Wav2Vec2Model` (the paper's wav2vec2-Large LV-60, SSL-only), 1024-d @ 50 Hz.

Two modes, both streaming per utterance (never materialising a split in RAM -- `build_units.py`
peaked at 41 GB on 100 h and would die on 960 h):

  fit-mfcc : MFCC(39-d) -> MiniBatchKMeans(64) -> centroids.npy      [CPU only]
  dump     : per split -> {split}.npy [N,D] fp16 + {split}.lengths + {split}.km   [GPU]

Layout matches fairseq's `ExtractedFeaturesDataset`: one .npy holding every frame of every utterance
concatenated, mmap'd at train time, plus a .lengths file with one frame count per line. fp16 is safe
because __getitem__ does `.float()`; it halves the mmap.

Aux target (L_ss, the paper's largest single win: PER 15.9 -> 13.6): 64-cluster k-means over Kaldi
MFCC+delta+delta-delta, encoder-independent by design. **We emit `.km` already at the encoder frame
rate and VAD-trimmed, aligned 1:1 with the features, so the GAN must run `target_downsample_rate: 1`
(not fairseq's 2).** fairseq computes MFCC on silence-*removed* waveforms at 100 Hz and subsamples to
its 50 Hz feature rate, reconciling any residue by truncating to min(len) -- a hack that can misalign
by a frame. Doing the subsample+trim here instead is exact, and computing the deltas on the
*continuous* waveform avoids the artificial discontinuities that trimming first would introduce at
every excised pause. `--mfcc-downsample` and `--vad-subframes` both follow from the encoder frame
rate (100 Hz MFCC / 100 Hz rVAD -> 4 for 25 Hz BEST-RQ, 2 for 50 Hz wav2vec2).
"""

import argparse
import os
import sys

import numpy as np
import torch


def _load_encoder(args, device):
    """Return (kind, obj): the frozen encoder plus what `_encode` needs to run it."""
    if args.encoder_type == "bestrq":
        # Cross-recipe import: the frozen BEST-RQ wrapper lives in the speech-llm recipe even though it
        # is LLM-independent. Reused rather than copied so this dump and the SAE 2S units share one
        # operator. `.eval()` is what turns SpecAugment off, matching what SAE_0a measured.
        from speech_llm.prefix_lm.model.definitions.encoders.bestrq import BestRqEncoderV1

        return "bestrq", BestRqEncoderV1(encoder_layer=args.encoder_layer).to(device).eval()

    if args.encoder_type == "wav2vec2":
        # HF port of the paper's wav2vec2-Large LV-60 (SSL, no finetuning -> no phone-label leakage).
        # `hidden_states[layer]` is the residual-stream output of transformer block (layer-1): with the
        # stable-layer-norm encoder, hidden_states[0] is the pre-transformer state, so layer=15 is the
        # output of block 14 = fairseq `layer=14` = the paper's "15th layer", without the final norm.
        # do_normalize=True applies the per-utterance waveform standardisation fairseq does via layer_norm.
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        model = Wav2Vec2Model.from_pretrained(args.hf_model_dir).to(device).eval()
        fe = Wav2Vec2FeatureExtractor.from_pretrained(args.hf_model_dir)
        return "wav2vec2", (model, fe, args.encoder_layer)

    raise ValueError(f"unknown encoder-type {args.encoder_type!r}")


def _encode(bundle, wav, device):
    """wav [T_samples] float32 -> feats [T_frames, D] float32 at the encoder frame rate."""
    kind, obj = bundle
    if kind == "bestrq":
        audio = torch.from_numpy(wav)[None, :].to(device)
        lens = torch.tensor([wav.shape[0]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = obj.forward(audio, lens)
        t_enc = int(out[-2][0])
        return out[0][0, :t_enc].float().cpu().numpy()

    model, fe, layer = obj
    inp = fe(wav, sampling_rate=16000, return_tensors="pt")  # do_normalize per utterance
    iv = inp.input_values.to(device)
    am = inp.get("attention_mask")
    with torch.no_grad():
        out = model(iv, attention_mask=am.to(device) if am is not None else None,
                    output_hidden_states=True)
    return out.hidden_states[layer][0].float().cpu().numpy()


def _vad():
    from rVADfast import rVADfast

    from i6_experiments.users.wu.experiments.unsupervised_asr import vad_port

    return vad_port, rVADfast(vad_threshold=0.4)


def _mfcc(wav, downsample):
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
    return torch.cat([m, d1, d2], dim=1).numpy()[::downsample]  # [~T_enc, 39]


def _sil_mask(vp, vad, wav, n, subframes):
    """Encoder-rate rVAD silence mask reconciled to n frames (truncate, or pad tail as silence)."""
    sil = vp.rvad_silence(wav, vad=vad, subframes=subframes)
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
        m = _mfcc(wav, args.mfcc_downsample)
        if vad is not None:
            m = m[~_sil_mask(vp, vad, wav, len(m), args.vad_subframes)]
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

    # Fail fast if the node's CUDA is dead (the transient "CUDA unknown error"): the job requested a
    # GPU, so silently falling back to CPU only means the encoder runs ~100x slower and the job dies on
    # the wall-clock limit hours later. Raising instead lets sisyphus resubmit on a healthy node.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable in the dump worker although a GPU was requested -- likely a transient "
            "node CUDA-init failure. Failing fast so this resubmits elsewhere instead of running the "
            "encoder on CPU and timing out."
        )
    device = "cuda"
    bundle = _load_encoder(args, device)
    vp, vad = _vad() if args.trim_silence else (None, None)
    cent = np.load(args.mfcc_centroids).astype(np.float32)  # [64, 39]
    cent_sq = (cent ** 2).sum(1)

    for p in (args.out_npy,):
        if os.path.exists(p):
            os.remove(p)
    lengths, ids, n_raw_tot, n_kept_tot, n_short = [], [], 0, 0, 0
    feat_dim = None

    with NpyAppendArray(args.out_npy) as npy, open(args.out_km, "w") as kmf:
        for n_utt, (tag, wav) in enumerate(_iter_audio(args.hf_data_dir, args.split, args.limit)):
            feats = _encode(bundle, wav, device)
            feat_dim = feats.shape[1]

            mf = _mfcc(wav, args.mfcc_downsample)
            t = min(len(feats), len(mf))
            feats, mf = feats[:t], mf[:t]

            if vad is not None:
                keep = ~_sil_mask(vp, vad, wav, t, args.vad_subframes)
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
                f"dim={feat_dim}\ndtype=float16\nencoder_type={args.encoder_type}\n"
                f"encoder_layer={args.encoder_layer}\ntrim_silence={args.trim_silence}\n"
                f"mfcc_downsample={args.mfcc_downsample}\nvad_subframes={args.vad_subframes}\n"
                f"km_rate=encoder_frame_rate_aligned_1to1 (=> fairseq target_downsample_rate must be 1)\n")
    print(f"[dump] {args.split}: {len(lengths)} utts, {n_kept_tot} frames, vad dropped {drop:.1%}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["fit-mfcc", "dump"], required=True)
    ap.add_argument("--hf-data-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--encoder-type", choices=["bestrq", "wav2vec2"], default="bestrq")
    ap.add_argument("--encoder-layer", type=int, default=5, help="bestrq: 0-idx layer; wav2vec2: hidden_states idx")
    ap.add_argument("--hf-model-dir", default=None, help="local dir for --encoder-type wav2vec2 (from_pretrained)")
    ap.add_argument("--trim-silence", action="store_true")
    ap.add_argument("--mfcc-downsample", type=int, default=4, help="100 Hz MFCC -> encoder rate (25 Hz:4, 50 Hz:2)")
    ap.add_argument("--vad-subframes", type=int, default=4, help="100 Hz rVAD -> encoder rate (25 Hz:4, 50 Hz:2)")
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
    if args.encoder_type == "wav2vec2" and args.mode == "dump" and not args.hf_model_dir:
        ap.error("--encoder-type wav2vec2 requires --hf-model-dir")
    (fit_mfcc if args.mode == "fit-mfcc" else dump)(args)


if __name__ == "__main__":
    main()
