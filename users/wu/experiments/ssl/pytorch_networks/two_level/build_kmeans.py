"""
Offline build of the FROZEN high-level k-means target codebook (worker script, run by the conda python).

Procedure (matches the train-time assignment so there is NO offline/online operator drift):
  * load the FROZEN lower stack of the two-level model and the pretrained BEST-RQ checkpoint;
  * extract layer-9 features at 25 Hz on a subset of utterances;
  * FIXED-WINDOW mean-pool every ``round(25 / target_rate)`` frames; v1 uses ``target_rate=25`` => window=1
    => FRAME-LEVEL (no pooling, the content manifold itself). (window 2 @80 ms / 3 @120 ms remain available
    as the fixed-window fallback codebook; the CIF segmenter is not involved in the fit either way);
  * L2-normalize the pooled vectors (cosine geometry == the train-time NN assignment in two_level_v1);
  * fit MiniBatchKMeans; L2-normalize the centroids; save ``centroids.npy`` [num_clusters, 512].

CLI:
  build_kmeans.py --ckpt <epoch.pt> --config <config.json> --out <centroids.npy>
    --target-rate 12.5 --num-clusters 128 --max-utts 4000 --max-vectors 500000 --seed 42
    --parquet a.parquet b.parquet ...
"""

import argparse
import json
import sys

import numpy as np
import torch


def _load_model(config_path, ckpt_path):
    from i6_experiments.users.wu.experiments.ssl.pytorch_networks.two_level.two_level_v1 import Model

    with open(config_path) as f:
        model_config_dict = json.load(f)
    model = Model(model_config_dict, codebook=None)  # codebook unused for extraction
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Sanity: the frozen lower stack MUST have loaded (else extraction is from random weights).
    enc_missing = [k for k in missing if k.startswith("encoder.") or k.startswith("feature_extraction.")]
    assert not enc_missing, f"lower-stack keys NOT loaded from ckpt: {enc_missing[:5]} ..."
    print(f"loaded ckpt: {len(sd)} tensors; missing(fresh)={len(missing)} unexpected(ignored)={len(unexpected)}",
          flush=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-rate", type=float, required=True)
    ap.add_argument("--num-clusters", type=int, required=True)
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--max-utts", type=int, default=4000)
    ap.add_argument("--max-vectors", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset, Audio
    from sklearn.cluster import MiniBatchKMeans

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(args.config, args.ckpt).to(device)
    window = int(round(25.0 / args.target_rate))
    assert window >= 1
    print(f"device={device}  window={window} frames/token  target_rate={args.target_rate}", flush=True)

    ds = load_dataset("parquet", data_files={"data": list(args.parquet)}, split="data")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ds))[: args.max_utts]

    pooled = []
    n = 0
    for count, idx in enumerate(order):
        audio = ds[int(idx)]["audio"]["array"]
        wav = torch.tensor(np.asarray(audio), dtype=torch.float32, device=device)[None, :]  # [1, T]
        length = torch.tensor([wav.shape[1]], device=device)
        h, flen = model.extract_layer9(wav, length)  # [1, T, 512], [1]
        L = int(flen[0])
        nwin = L // window
        if nwin < 1:
            continue
        v = h[0, : nwin * window].reshape(nwin, window, h.shape[-1]).mean(dim=1)  # [nwin, 512]
        v = torch.nn.functional.normalize(v.float(), dim=-1)
        pooled.append(v.cpu().numpy())
        n += nwin
        if n >= args.max_vectors:
            break
        if count % 500 == 0:
            print(f"  {count}/{len(order)} utts, {n} pooled vectors", flush=True)

    X = np.concatenate(pooled, axis=0)
    if X.shape[0] > args.max_vectors:
        X = X[rng.permutation(X.shape[0])[: args.max_vectors]]
    print(f"fitting MiniBatchKMeans: {X.shape} -> {args.num_clusters} clusters", flush=True)
    km = MiniBatchKMeans(n_clusters=args.num_clusters, random_state=args.seed,
                         batch_size=4096, max_iter=200, n_init=3, verbose=0)
    km.fit(X)
    centroids = km.cluster_centers_.astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=-1, keepdims=True) + 1e-8  # match train-time cosine NN
    # quick health: assignment usage entropy on the fit set
    assign = km.predict(X)
    counts = np.bincount(assign, minlength=args.num_clusters).astype(np.float64)
    p = counts / counts.sum()
    ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(args.num_clusters)
    print(f"codebook built: {centroids.shape}  dead={int((counts==0).sum())}  norm_entropy={ent:.3f}",
          flush=True)
    np.save(args.out, centroids)


if __name__ == "__main__":
    sys.exit(main())
