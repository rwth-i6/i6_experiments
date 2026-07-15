"""SAE §0a REAL representation audit on the frozen BEST-RQ checkpoint (CPU, no GPU / no sisyphus job).

Entry point for the §0a-REAL findings in SAE_0.md. Extracts frozen encoder features at 25 Hz from
real dev audio, then answers two questions on the *same* features:

  audit  hard-kmeans-unit + oracle-map PER (the §1a(i) ceiling) across K, with S/I/D breakdown and
         optional wav2vec-U preprocessing (rVAD silence removal + adjacent mean-pool).
  probe  the wav2vec-U-2.0 alternative: a supervised *continuous* generator (linear / MLP) straight
         on the features -- the topline an adversarial continuous generator could reach.

The contrast (hard-unit oracle ~0.6 PER, insertion-dominant and ~flat in K; continuous linear ~0.15)
is the evidence that the *hard discretization*, not the encoder, is the ceiling.

Run (from workspace root, conda `speech_llm` python):
    python recipe/i6_experiments/users/wu/experiments/ssl/analysis/real_repr_probe.py --stage both
Flags: --layer (0-indexed encoder layer) --n_utts --split --K 500,1000,2000,4000 --vad --pool W.

Metric primitives come from repr_audit.py (same dir). The vad_port silence-removal preprocessor moved
to the unsupervised_asr recipe (it is wav2vec-U-style preprocessing, not representation-quality
measurement) and is loaded lazily by file path only when (re)building the feature cache. Both are
loaded by file path because at run time the frozen model must come from the *sibling* workspace whose
recipe shadows this i6_experiments package on sys.path (see SIB below)."""
import argparse
import os
import pickle
import sys
import time
from importlib.machinery import SourceFileLoader

import numpy as np
import torch

# Frozen BEST-RQ lives only in the sibling workspace (ckpt is a frozen external asset; D1 pending).
SIB = "/e/project1/spell/wu24/2026-06-17_ssl"
J = f"{SIB}/work/i6_core/returnn/training/ReturnnTrainingJob.iDPxBJeb35l8/output"
_HERE = os.path.dirname(os.path.abspath(__file__))
_UASR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "unsupervised_asr"))  # vad_port lives here now
RA = SourceFileLoader("ra_mod", os.path.join(_HERE, "repr_audit.py")).load_module()
SIL = RA.SIL_ID
CACHE_DIR = os.environ.get("SAE_CACHE", "/tmp/sae_repr_cache")


def load_model():
    for p in (f"{SIB}/recipe", f"{SIB}/recipe/i6_models", f"{SIB}/recipe/returnn"):
        sys.path.insert(0, p)  # sibling recipe (matches the ckpt) shadows this workspace's package
    rc = SourceFileLoader("rc", f"{J}/returnn.config").load_module()
    model = rc.get_model()
    model.eval()
    sd = torch.load(f"{J}/models/epoch.100.pt", map_location="cpu")
    model.load_state_dict(sd.get("model", sd), strict=False)  # exact load: 0 missing / 0 unexpected
    return model


def extract_features(layer, n_utts, split, do_vad, pool_w):
    """[(feat[T,512], gold[T], sil[T] bool)] for the split; pickle-cached by (layer,n_utts,split)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"utts_L{layer}_{n_utts}_{split}.pkl")
    if os.path.exists(cache):
        utts = pickle.load(open(cache, "rb"))
        print(f"loaded cached features: {len(utts)} utts", flush=True)
        return _preprocess(utts, do_vad, pool_w)
    model = load_model()
    from i6_experiments.users.wu.experiments.ssl.pytorch_networks.common.conformer import sequence_mask
    from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.parts.input_norm import (
        apply_global_norm,
    )
    from rVADfast import rVADfast

    VP = SourceFileLoader("vp_mod", os.path.join(_UASR, "vad_port.py")).load_module()
    vad = rVADfast(vad_threshold=0.4)
    recs = VP._load_gilkeyio(split, limit=n_utts)
    utts, t1 = [], time.time()
    for wav, phon in recs:
        wl = torch.tensor([len(wav)])
        with torch.no_grad():
            f, fl = model.feature_extraction(torch.tensor(wav, dtype=torch.float32)[None], wl)
            normed = apply_global_norm(f, fl, model.global_mean, model.global_std)
            enc, _ = model.encoder(normed, sequence_mask(normed, fl), return_layers=[layer])
        e = enc[0][0].numpy()  # [T,512]
        g = RA.frame_phone_labels(phon, e.shape[0])
        sil = VP.rvad_silence_25hz(wav, vad=vad)
        sil = sil[:e.shape[0]] if len(sil) >= e.shape[0] else np.concatenate(
            [sil, np.ones(e.shape[0] - len(sil), bool)])
        utts.append((e, g, sil))
    pickle.dump(utts, open(cache, "wb"))
    print(f"extracted {len(utts)} utts ({time.time()-t1:.0f}s), cached -> {cache}", flush=True)
    return _preprocess(utts, do_vad, pool_w)


def _pool_adjacent(e, g, sil, w):
    if w <= 1:
        return e, g, sil
    T = (e.shape[0] // w) * w
    if T == 0:
        return e[:0], g[:0], sil[:0]
    ep = e[:T].reshape(-1, w, e.shape[1]).mean(1)
    gp = np.array([np.bincount(g[i:i + w]).argmax() for i in range(0, T, w)])
    sp = (sil[:T].reshape(-1, w).mean(1) >= 0.5)
    return ep, gp, sp


def _preprocess(utts, do_vad, pool_w):
    if not do_vad and pool_w <= 1:
        return utts
    out = []
    for e, g, sil in utts:
        e, g, sil = _pool_adjacent(e, g, sil, pool_w)
        if do_vad:
            keep = ~sil
            e, g, sil = e[keep], g[keep], sil[keep]
        if e.shape[0] >= 2:
            out.append((e, g, sil))
    return out


def _sid(ref, hyp):
    """Align gold ref -> hyp; return (subs, ins, dels). ins = extra symbol in hyp (over-segmentation)."""
    ref, hyp = list(ref), list(hyp)
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i, j] = min(D[i - 1, j - 1] + c, D[i - 1, j] + 1, D[i, j - 1] + 1)
    i, j, S, I, Dl = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            S += ref[i - 1] != hyp[j - 1]
            i -= 1
            j -= 1
        elif j > 0 and D[i, j] == D[i, j - 1] + 1:
            I += 1
            j -= 1
        else:
            Dl += 1
            i -= 1
    return S, I, Dl


def _decode_stats(pred_by_utt, gold_by_utt):
    tS = tI = tD = tL = 0
    for k in pred_by_utt:
        pred = RA.run_length_dedup(np.asarray(pred_by_utt[k]))
        pred = pred[pred != SIL]
        ref = RA.gold_phone_tokens(np.asarray(gold_by_utt[k]), drop_sil=True)
        S, I, Dl = _sid(ref, pred)
        tS += S
        tI += I
        tD += Dl
        tL += len(ref)
    return dict(PER=(tS + tI + tD) / tL, sub=tS / tL, ins=tI / tL, dele=tD / tL)


def _split(n_utts, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_utts)
    nt = n_utts // 5
    return idx[nt:], idx[:nt]  # train, test (held-out 20%)


def audit(utts, Ks):
    """Hard-kmeans-unit + oracle-map PER vs K, held-out. Map built on train, applied to test."""
    from sklearn.cluster import MiniBatchKMeans

    train_i, test_i = _split(len(utts))
    Xtr = np.concatenate([utts[i][0] for i in train_i]).astype(np.float32)
    ytr = np.concatenate([utts[i][1] for i in train_i]).astype(np.int64)
    print(f"\n[audit] train frames={len(ytr)} test utts={len(test_i)}", flush=True)
    print(f"{'K':>6} {'test_frame_acc':>14} {'oracle_PER':>11} {'sub':>6} {'ins':>6} {'del':>6}", flush=True)
    for K in Ks:
        km = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=4096, n_init=3, max_iter=100).fit(Xtr)
        N = np.zeros((K, RA.NUM_PHONES))
        np.add.at(N, (km.labels_, ytr), 1.0)
        A = RA.oracle_map(N)
        pred, gold, corr, tot = {}, {}, 0, 0
        for i in test_i:
            c = km.predict(utts[i][0].astype(np.float32))
            g = utts[i][1]
            pred[i], gold[i] = A[c], g
            n = min(len(c), len(g))
            corr += int((A[c][:n] == g[:n]).sum())
            tot += n
        st = _decode_stats(pred, gold)
        print(f"{K:>6} {corr/tot:>14.3f} {st['PER']:>11.3f} {st['sub']:>6.3f} {st['ins']:>6.3f} "
              f"{st['dele']:>6.3f}", flush=True)


def probe(utts, hiddens=(0, 256)):
    """Supervised continuous generator (linear / MLP) on the same features, held-out."""
    train_i, test_i = _split(len(utts))
    Xtr = np.concatenate([utts[i][0] for i in train_i]).astype(np.float32)
    ytr = np.concatenate([utts[i][1] for i in train_i]).astype(np.int64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    Xtr = (Xtr - mu) / sd
    torch.manual_seed(0)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    print(f"\n[probe] train frames={len(ytr)} test utts={len(test_i)}", flush=True)
    for hidden in hiddens:
        layers, d = [], 512
        if hidden:
            layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU()]
            d = hidden
        layers += [torch.nn.Linear(d, RA.NUM_PHONES)]
        net = torch.nn.Sequential(*layers)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = torch.nn.CrossEntropyLoss()
        for _ in range(25):
            perm = torch.randperm(len(yt))
            for b in range(0, len(yt), 4096):
                j = perm[b:b + 4096]
                opt.zero_grad()
                lossf(net(Xt[j]), yt[j]).backward()
                opt.step()
        net.eval()
        pred, gold, corr, tot = {}, {}, 0, 0
        with torch.no_grad():
            for i in test_i:
                p = net(torch.tensor((utts[i][0].astype(np.float32) - mu) / sd)).argmax(1).numpy()
                g = utts[i][1]
                pred[i], gold[i] = p, g
                n = min(len(p), len(g))
                corr += int((p[:n] == g[:n]).sum())
                tot += n
        st = _decode_stats(pred, gold)
        tag = "linear" if not hidden else f"MLP-{hidden}"
        print(f"  {tag:10s} frame_acc={corr/tot:.3f}  PER={st['PER']:.3f}  "
              f"sub={st['sub']:.3f} ins={st['ins']:.3f} del={st['dele']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "probe", "both"], default="both")
    ap.add_argument("--layer", type=int, default=5)          # 0-indexed -> conformer layer 6
    ap.add_argument("--n_utts", type=int, default=500)
    ap.add_argument("--split", default="validation.clean")
    ap.add_argument("--K", default="500,1000,2000,4000")
    ap.add_argument("--vad", action="store_true")            # rVAD silence removal
    ap.add_argument("--pool", type=int, default=1)           # adjacent mean-pool window
    args = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.time()
    utts = extract_features(args.layer, args.n_utts, args.split, args.vad, args.pool)
    print(f"layer {args.layer+1}, {len(utts)} utts, vad={args.vad} pool={args.pool} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if args.stage in ("audit", "both"):
        audit(utts, [int(x) for x in args.K.split(",")])
    if args.stage in ("probe", "both"):
        probe(utts)
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
