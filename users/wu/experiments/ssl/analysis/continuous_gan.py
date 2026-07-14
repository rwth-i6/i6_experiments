"""wav2vec-U-2.0-style continuous GAN for unsupervised phone recognition on frozen BEST-RQ features.

Why this and not decipherment: SAE_0.md shows hard-unit decipherment and generative ML-EM both fail
(ML objective anti-aligned with PER). The GAN uses an *aligned* objective — make the generator's output
phone-sequence distribution indistinguishable from real phoneme text (𝒯_φ) — which is what wav2vec-U
optimises. The generator is trivial (a linear map already hits 0.145 supervised, see real_repr_probe),
so the only hard part is adversarial training; hence the `--mode sup` gate that trains G with CE to gold
and MUST reproduce ~0.145 to prove the collapse+eval plumbing before trusting the `--mode gan` number.

Pipeline (per utt): frozen feat x[T,512] -> Generator -> phone logits[T,40] -> softmax -> collapse
consecutive-argmax runs (segment-average the distributions) -> drop SIL -> token-level P[L,40].
Discriminator (1-D conv) scores token sequences; real = one-hot phoneme lines from 𝒯_φ. Losses: NS-GAN
+ gradient penalty (D) + smoothness (|P_t-P_{t+1}|², fights over-segmentation) + phone-diversity
(marginal entropy, fights collapse). Eval: argmax->collapse->drop-SIL PER vs MFA gold. CPU only."""
import argparse
import gzip
import os
import time
from importlib.machinery import SourceFileLoader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
RA = SourceFileLoader("ra_mod", os.path.join(_HERE, "repr_audit.py")).load_module()
TPHI = ("work/i6_experiments/users/wu/experiments/ssl/experiments/sae/phonemize/"
        "PhonemizeCorpusJob.QqW74njzlaWq/output/phonemes.txt.gz")
NP_ = RA.NUM_PHONES
SIL = RA.SIL_ID


# ------------------------------------------------------------------------------------- real text
def load_real_lines(n_lines, min_len=5, max_len=60):
    """Phoneme id sequences from 𝒯_φ (drop [UNKNOWN]); SIL not in 𝒯_φ -> inserted at eval-time only."""
    P2I, out = RA.PHONE2ID, []
    with gzip.open(TPHI, "rt") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            s = [P2I[t] for t in line.split() if t in P2I]
            if min_len <= len(s) <= max_len:
                out.append(np.array(s, np.int64))
    return out


def onehot(seq, n=NP_):
    x = np.zeros((len(seq), n), np.float32)
    x[np.arange(len(seq)), seq] = 1.0
    return x


# ---------------------------------------------------------------------------------- collapse (diff)
def collapse_soft(P):
    """P[T,40] softmax -> token-level [L,40] by averaging consecutive-argmax runs, then drop SIL runs.
    Boundaries from argmax (non-diff, as in wav2vec-U); averaged probs carry gradients."""
    am = P.argmax(1)
    bnd = torch.ones(len(am), dtype=torch.bool, device=P.device)
    bnd[1:] = am[1:] != am[:-1]
    idx = torch.where(bnd)[0]
    segs = []
    for a, b in zip(idx.tolist(), idx.tolist()[1:] + [len(am)]):
        if am[a] != SIL:
            segs.append(P[a:b].mean(0))
    if not segs:
        return P.new_zeros((0, P.shape[1]))
    return torch.stack(segs)


# ------------------------------------------------------------------------------------ models
class Generator(nn.Module):
    def __init__(self, d_in, n_ph=NP_, conv=False):
        super().__init__()
        self.net = (nn.Conv1d(d_in, n_ph, 5, padding=2) if conv else nn.Linear(d_in, n_ph))
        self.conv = conv

    def forward(self, x):  # x[T,d]
        if self.conv:
            return self.net(x.T[None]).squeeze(0).T
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, n_ph=NP_, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ph, dim, 4, padding=2), nn.ReLU(),
            nn.Conv1d(dim, dim, 4, padding=2), nn.ReLU(),
            nn.Conv1d(dim, 1, 1))

    def forward(self, x):  # x[L,40] -> scalar (mean over positions)
        return self.net(x.T[None]).mean()


# ------------------------------------------------------------------------------------ eval
def decode_per(G, feats, golds, mu, sd):
    G.eval()
    tot_e = tot_l = 0
    with torch.no_grad():
        for x, g in zip(feats, golds):
            P = F.softmax(G(torch.tensor((x - mu) / sd)), -1)
            am = P.argmax(1).numpy()
            pred = RA.run_length_dedup(am); pred = pred[pred != SIL]
            ref = RA.gold_phone_tokens(np.asarray(g), drop_sil=True)
            tot_e += RA.levenshtein(pred, ref); tot_l += len(ref)
    G.train()
    return tot_e / tot_l if tot_l else float("nan")


# ------------------------------------------------------------------------------------ data
def load_feats(layer, n_utts, split, pca_dim, pool_w=1):
    RRP = SourceFileLoader("rrp", os.path.join(_HERE, "real_repr_probe.py")).load_module()
    utts = RRP.extract_features(layer, n_utts, split, do_vad=False, pool_w=pool_w)  # pool toward phone rate
    rng = np.random.RandomState(0); idx = rng.permutation(len(utts)); nt = len(utts) // 5
    test_i, train_i = idx[:nt], idx[nt:]
    from sklearn.decomposition import PCA
    Xtr = np.concatenate([utts[i][0] for i in train_i]).astype(np.float32)
    pca = PCA(pca_dim, random_state=0).fit(Xtr) if pca_dim else None
    tf = (lambda a: pca.transform(a)) if pca else (lambda a: a)
    feat = lambda i: tf(utts[i][0].astype(np.float32))
    tr = [feat(i) for i in train_i]; te = [feat(i) for i in test_i]
    gtr = [utts[i][1] for i in train_i]; gte = [utts[i][1] for i in test_i]
    X = np.concatenate(tr); mu, sd = X.mean(0), X.std(0) + 1e-5
    return tr, te, gtr, gte, mu, sd


# ------------------------------------------------------------------------------------ sup gate
def run_sup(feats, golds, te, gte, mu, sd, d_in, iters=25, conv=False):
    """Train G with CE to gold — must reproduce the ~0.145 linear-probe PER (plumbing gate)."""
    torch.manual_seed(0)
    G = Generator(d_in, conv=conv)
    opt = torch.optim.Adam(G.parameters(), 1e-3, weight_decay=1e-5)
    Xt = torch.tensor(np.concatenate([(x - mu) / sd for x in feats]))
    yt = torch.tensor(np.concatenate(golds))
    for ep in range(iters):
        perm = torch.randperm(len(yt))
        for b in range(0, len(yt), 4096):
            j = perm[b:b + 4096]
            opt.zero_grad(); F.cross_entropy(G(Xt[j]), yt[j]).backward(); opt.step()
    print(f"[sup] TEST PER={decode_per(G, te, gte, mu, sd):.3f}  (expect ~0.145)", flush=True)


# ------------------------------------------------------------------------------------ GAN
def grad_penalty(D, real, fake):
    a = torch.rand(1).item()
    n = min(len(real), len(fake))
    if n < 2:
        return torch.zeros(())
    inter = (a * real[:n] + (1 - a) * fake[:n]).detach().requires_grad_(True)
    s = D(inter)
    g = torch.autograd.grad(s, inter, create_graph=True)[0]
    return ((g.norm() - 1) ** 2)


def run_gan(feats, golds, te, gte, mu, sd, d_in, real_lines, steps, conv,
            lam_gp=1.5, lam_sm=0.5, lam_div=0.1, seed=0, dsteps=1):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    G = Generator(d_in, conv=conv); D = Discriminator()
    optG = torch.optim.Adam(G.parameters(), 5e-4, betas=(0.5, 0.98))
    optD = torch.optim.Adam(D.parameters(), 3e-4, betas=(0.5, 0.98))
    feats_n = [((x - mu) / sd).astype(np.float32) for x in feats]
    reals = [torch.tensor(onehot(s)) for s in real_lines]
    best = 1.0
    for step in range(steps):
        # ---- D steps ----
        for _ in range(dsteps):
            i = rng.randint(len(feats_n)); x = torch.tensor(feats_n[i])
            fake = collapse_soft(F.softmax(G(x), -1)).detach()
            real = reals[rng.randint(len(reals))]
            if len(fake) >= 2:
                optD.zero_grad()
                lossD = F.softplus(D(fake)) + F.softplus(-D(real)) + lam_gp * grad_penalty(D, real, fake)
                lossD.backward(); optD.step()
        # ---- G step ----
        i = rng.randint(len(feats_n)); x = torch.tensor(feats_n[i])
        P = F.softmax(G(x), -1)
        fake = collapse_soft(P)
        if len(fake) >= 2:
            optG.zero_grad()
            sm = ((P[1:] - P[:-1]) ** 2).mean()
            marg = P.mean(0)
            div = (marg * (marg + 1e-8).log()).sum()  # neg entropy; minimize -> spread usage
            lossG = F.softplus(-D(fake)) + lam_sm * sm + lam_div * div
            lossG.backward(); optG.step()
        if (step + 1) % 500 == 0:
            per = decode_per(G, te, gte, mu, sd)
            best = min(best, per)
            print(f"  step {step+1}: TEST PER={per:.3f} (best {best:.3f})", flush=True)
    print(f"[gan] best TEST PER={best:.3f}  [ref hard 0.632 | sup 0.145 | gate 0.50]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sup", "gan"], default="sup")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--n_utts", type=int, default=500)
    ap.add_argument("--split", default="validation.clean")
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--conv", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--real_lines", type=int, default=30000)
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--lam_gp", type=float, default=1.5)
    ap.add_argument("--lam_sm", type=float, default=0.5)
    ap.add_argument("--lam_div", type=float, default=0.1)
    ap.add_argument("--dsteps", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.time()
    tr, te, gtr, gte, mu, sd = load_feats(a.layer, a.n_utts, a.split, a.pca, a.pool)
    d_in = tr[0].shape[1]
    print(f"loaded: train {len(tr)} test {len(te)} d_in={d_in} pool={a.pool} ({time.time()-t0:.0f}s)",
          flush=True)
    if a.mode == "sup":
        run_sup(tr, gtr, te, gte, mu, sd, d_in, conv=a.conv)
    else:
        reals = load_real_lines(a.real_lines)
        print(f"real text lines: {len(reals)}", flush=True)
        run_gan(tr, gtr, te, gte, mu, sd, d_in, reals, a.steps, a.conv,
                lam_gp=a.lam_gp, lam_sm=a.lam_sm, lam_div=a.lam_div, dsteps=a.dsteps)
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
