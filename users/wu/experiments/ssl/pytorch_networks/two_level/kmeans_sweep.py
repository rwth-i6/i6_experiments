"""
Analysis: sweep the high-level k-means codebook size K on REAL layer-9 features, to ground the choice
of codebook size for the two-level model (is 128/256 a good starting point?).

For a subset of utterances: extract frozen layer-9 features, FIXED-WINDOW mean-pool at window=2 (80 ms)
and window=3 (120 ms), L2-normalize (the train-time geometry), then fit MiniBatchKMeans for each K and
report the adequacy/health metrics: #tokens, tokens/cluster, dead-cluster %, normalized usage entropy,
inertia (distortion) and its elbow ratio. NOT a Sisyphus job -- a quick offline probe.

Run (conda python):
  PYTHONPATH=recipe:recipe/i6_models:recipe/returnn <py> kmeans_sweep.py \
    --ckpt <epoch.pt> --n-utts 300 --ks 64 128 256 512 1024 --windows 2 3 --seed 42
"""

import argparse

import numpy as np
import torch


def build_cfg():
    from i6_experiments.users.wu.experiments.ssl.pytorch_networks.two_level.two_level_v1_cfg import TwoLevelConfig
    from i6_experiments.users.wu.experiments.ssl.pytorch_networks.common.conformer import (
        default_encoder_config, default_high_encoder_config,
    )
    from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
    from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.ls_logmel_stats import (
        LOGMEL_MEAN, LOGMEL_STD,
    )
    return TwoLevelConfig(
        feature_extraction_config=LogMelFeatureExtractionV1Config(
            sample_rate=16000, win_size=0.025, hop_size=0.01, f_min=60, f_max=7600,
            min_amp=1e-10, num_filters=80, center=False,
        ),
        encoder_config=default_encoder_config(num_layers=9),
        high_encoder_config=default_high_encoder_config(num_layers=9, rel_pos_clip=96),
        global_mean=list(LOGMEL_MEAN), global_std=list(LOGMEL_STD),
        lower_layer_index=8, cif_alpha_kernel_size=5, target_rate_hz=12.5, frame_rate_hz=25.0,
        lambda_qty=1.0, num_clusters=128, mask_prob=0.2, mask_length=3, min_masks=1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-utts", type=int, default=300)
    ap.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--windows", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from dataclasses import asdict
    from datasets import load_dataset, Audio
    from sklearn.cluster import MiniBatchKMeans
    from i6_experiments.users.wu.experiments.ssl.pytorch_networks.two_level.two_level_v1 import Model
    from i6_experiments.users.wu.experiments.ssl.data import datasets as ds

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(asdict(build_cfg()), codebook=None)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, _ = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if k.startswith(("encoder.", "feature_extraction."))], "lower stack not loaded"
    model.eval().to(device)
    print(f"device={device}  ckpt loaded", flush=True)

    files = ds.parquet_files(ds.DEV_CLEAN)
    data = load_dataset("parquet", data_files={"data": files}, split="data").cast_column(
        "audio", Audio(sampling_rate=16000)
    )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(data))[: args.n_utts]

    pooled = {w: [] for w in args.windows}
    for c, idx in enumerate(order):
        wav = torch.tensor(np.asarray(data[int(idx)]["audio"]["array"]), dtype=torch.float32, device=device)[None]
        h, flen = model.extract_layer9(wav, torch.tensor([wav.shape[1]], device=device))
        L = int(flen[0])
        for w in args.windows:
            nwin = L // w
            if nwin < 1:
                continue
            v = h[0, : nwin * w].reshape(nwin, w, h.shape[-1]).mean(dim=1)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            pooled[w].append(v.cpu().numpy())
        if c % 50 == 0:
            print(f"  {c}/{len(order)} utts", flush=True)

    for w in args.windows:
        X = np.concatenate(pooled[w], axis=0)
        rate = 25.0 / w
        print(f"\n==== window={w} frames/token ({rate:.2f} Hz, {1000/rate:.0f} ms)  n_tokens={X.shape[0]} ====")
        print(f"  {'K':>5} {'tok/clu':>8} {'dead%':>6} {'norm_ent':>9} {'inertia':>9} {'elbowΔ':>7}")
        prev = None
        for K in args.ks:
            km = MiniBatchKMeans(n_clusters=K, random_state=args.seed, batch_size=4096, max_iter=200, n_init=3)
            a = km.fit_predict(X)
            counts = np.bincount(a, minlength=K).astype(np.float64)
            p = counts / counts.sum()
            ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(K)
            dead = 100.0 * (counts == 0).mean()
            inertia = km.inertia_ / X.shape[0]  # mean per-token distortion
            elbow = "" if prev is None else f"{(prev - inertia) / prev * 100:5.1f}%"
            print(f"  {K:>5} {X.shape[0] / K:>8.0f} {dead:>6.1f} {ent:>9.3f} {inertia:>9.4f} {elbow:>7}")
            prev = inertia


if __name__ == "__main__":
    main()
