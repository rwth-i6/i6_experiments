"""SAE §1.0 unsupervised validation metric (Baevski et al. / wav2vec-U) + a ground-truth calibration.

The metric selects checkpoints/hyperparameters with NO gold PER. Given decoded phone output P on dev,
the 4-gram phoneme LM, and vocabulary usage U(P) = |phones used| / |inventory|:
  1. anchor  P̂ = argmin_P  NLL_LM(P) − log U(P)          (per-token NLL; −log U rewards coverage)
  2. filter  keep P with NLL_LM(P) < NLL_LM(P̂) + log(U(P)/U(P̂)) + log 1.2
  3. winner  among survivors, argmax Σ_j Σ_t log p_LM      (non-length-normalised → punishes short decodes)

Because the (deferred) GAN and every unsupervised phase pick models with THIS and never gold, it must
rank runs the way gold PER does. `--calibrate` checks exactly that on a spread of real dev-clean
decodings with known gold PER (linear probe 0.145 → corrupted → hard-unit 0.632 → random 1.0 →
degenerate-constant), reusing the frozen-feature cache. All log10 (ArpaLM convention). CPU only."""
import argparse
import os
import time
from importlib.machinery import SourceFileLoader

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SAE = os.path.abspath(os.path.join(_HERE, "..", "experiments", "sae"))
RA = SourceFileLoader("ra_mod", os.path.join(_HERE, "repr_audit.py")).load_module()
DEC = SourceFileLoader("dec", os.path.join(_SAE, "decipher.py")).load_module()
ARPA = "output/sae/1a/phoneme_lm_o4.arpa.gz"
ID2P = RA.ARPABET_39
NSP = len(ID2P)  # 39 spoken phones (SIL excluded from P)


def seq_stats(id_seqs, lm):
    """Per-model LM stats over decoded phone-id sequences (SIL already dropped).
    Returns nll_per_tok (−mean log10 p), total_logp (Σ log10 p), U (vocab usage over 39)."""
    used, total_lp, total_tok = set(), 0.0, 0
    for s in id_seqs:
        toks = [ID2P[i] for i in s if i < NSP]
        used.update(toks)
        total_lp += lm.score_seq(toks)   # log10, <s> unscored / </s> scored
        total_tok += len(toks) + 1
    nll = -total_lp / max(total_tok, 1)
    return dict(nll_per_tok=nll, total_logp=total_lp, U=len(used) / NSP)


def select_1_0(models):
    """Apply anchor→filter→winner. models: list of dicts with nll_per_tok, U, total_logp. Returns idx."""
    obj = [m["nll_per_tok"] - np.log10(max(m["U"], 1e-9)) for m in models]
    anchor = int(np.argmin(obj))
    thr = obj[anchor] + np.log10(1.2)
    survivors = [i for i, m in enumerate(models) if m["nll_per_tok"] - np.log10(max(m["U"], 1e-9)) < thr]
    winner = max(survivors, key=lambda i: models[i]["total_logp"])
    return dict(anchor=anchor, survivors=survivors, winner=winner)


# ------------------------------------------------------------------------------------ calibration
def _decode(frame_ids_by_utt):
    """frame ids -> deduped, SIL-dropped id sequence per utt."""
    out = []
    for f in frame_ids_by_utt:
        d = RA.run_length_dedup(np.asarray(f))
        out.append(d[d != RA.SIL_ID].tolist())
    return out


def _per(dec, gold):
    e = l = 0
    for p, g in zip(dec, gold):
        ref = RA.gold_phone_tokens(np.asarray(g), drop_sil=True)
        e += RA.levenshtein(p, ref); l += len(ref)
    return e / l if l else float("nan")


def calibrate(layer=5, n_utts=500, pca=0):
    import torch, torch.nn.functional as F
    from sklearn.cluster import MiniBatchKMeans
    RRP = SourceFileLoader("rrp", os.path.join(_HERE, "real_repr_probe.py")).load_module()
    utts = RRP.extract_features(layer, n_utts, "validation.clean", do_vad=False, pool_w=1)
    rng = np.random.RandomState(0); idx = rng.permutation(len(utts)); nt = len(utts) // 5
    test_i, train_i = idx[:nt], idx[nt:]
    Xtr = np.concatenate([utts[i][0] for i in train_i]).astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    ytr = np.concatenate([utts[i][1] for i in train_i])
    gte = [utts[i][1] for i in test_i]
    lm = DEC.ArpaLM(ARPA)

    # a linear probe -> its per-frame argmax on test
    torch.manual_seed(0)
    net = torch.nn.Linear(512, RA.NUM_PHONES)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    Xt = torch.tensor((Xtr - mu) / sd); yt = torch.tensor(ytr)
    for _ in range(25):
        perm = torch.randperm(len(yt))
        for b in range(0, len(yt), 4096):
            j = perm[b:b + 4096]
            opt.zero_grad(); F.cross_entropy(net(Xt[j]), yt[j]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        probe = [net(torch.tensor((utts[i][0].astype(np.float32) - mu) / sd)).argmax(1).numpy()
                 for i in test_i]

    # hard-kmeans K=500 oracle-map argmax on test
    km = MiniBatchKMeans(500, random_state=0, batch_size=4096, n_init=3).fit(Xtr)
    N = np.zeros((500, RA.NUM_PHONES)); np.add.at(N, (km.labels_, ytr), 1.0)
    A = RA.oracle_map(N)
    kmdec = [A[km.predict(utts[i][0].astype(np.float32))] for i in test_i]

    most_freq = int(np.bincount(ytr[ytr != RA.SIL_ID]).argmax())
    rs = np.random.RandomState(1)

    def corrupt(eps):
        out = []
        for f in probe:
            r = rs.random(len(f)) < eps
            g = f.copy(); g[r] = rs.randint(0, RA.NUM_PHONES, r.sum())
            out.append(g)
        return out

    models = {
        "probe":            _decode(probe),
        "probe+corrupt.15": _decode(corrupt(0.15)),
        "probe+corrupt.30": _decode(corrupt(0.30)),
        "probe+corrupt.50": _decode(corrupt(0.50)),
        "kmeans-oracle":    _decode(kmdec),
        "random":           _decode([rs.randint(0, RA.NUM_PHONES, len(f)) for f in probe]),
        "degenerate-const": _decode([np.full(len(f), most_freq) for f in probe]),
    }
    rows = []
    for name, dec in models.items():
        st = seq_stats(dec, lm)
        rows.append((name, _per(dec, gte), st))
    order = sorted(range(len(rows)), key=lambda i: rows[i][1])  # by gold PER
    stats_list = [rows[i][2] for i in range(len(rows))]
    sel = select_1_0(stats_list)

    print(f"\n{'model':18s} {'goldPER':>8} {'NLL/tok':>8} {'U':>5} {'totLogP':>10}  role", flush=True)
    for i in order:
        name, per, st = rows[i]
        role = []
        if i == sel["anchor"]: role.append("ANCHOR")
        if i == sel["winner"]: role.append("WINNER")
        if i in sel["survivors"]: role.append("surv")
        print(f"{name:18s} {per:8.3f} {st['nll_per_tok']:8.3f} {st['U']:5.2f} "
              f"{st['total_logp']:10.0f}  {'+'.join(role)}", flush=True)
    from scipy.stats import spearmanr
    pers = [rows[i][1] for i in range(len(rows))]
    nlls = [rows[i][2]["nll_per_tok"] for i in range(len(rows))]
    rho = spearmanr(pers, nlls).correlation
    best_per_idx = min(range(len(rows)), key=lambda i: rows[i][1])
    print(f"\nSpearman(NLL/tok, goldPER) = {rho:.3f}  (want ~+1)", flush=True)
    print(f"§1.0 winner = '{rows[sel['winner']][0]}' (goldPER {rows[sel['winner']][1]:.3f}); "
          f"true best = '{rows[best_per_idx][0]}' (goldPER {rows[best_per_idx][1]:.3f})  "
          f"-> {'MATCH' if sel['winner']==best_per_idx else 'MISMATCH'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.calibrate:
        calibrate()
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
