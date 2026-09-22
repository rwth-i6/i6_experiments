"""Bootstrap the German 1 h phone table FROM ZERO: no pretrained aligner, no other audio.

The table itself is the acoustic model: one diagonal Gaussian per phone in the ASR's own log-mel space.
  1. flat start: every utterance's frames split uniformly over its phone sequence
  2. repeat: Viterbi forced alignment with the current Gaussians -> re-estimate means/variances
Each phone is expanded into MIN_DUR chained states that share one Gaussian, so a phone cannot be shorter
than MIN_DUR frames (MFA's 10 ms floor collapse is exactly a 1-frame phone). Optional [space] between
words and at both ends. Output has the same format as de_1h_table{.npz,_durations.npz}.

usage: table_em.py <min_dur_frames> <n_iter> <out_prefix> [init_table.npz]
"""
import io
import json
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

sys.path.insert(0, "/e/home/jusers/koch13/jupiter/setups/2026-09-16-fzj-dlm/tools/returnn")
import returnn.frontend as rf
from returnn.tensor import Dim, Tensor, batch_dim

rf.select_backend_torch()
batch_dim.dyn_size_ext = rf.convert_to_tensor(torch.tensor(1, dtype=torch.int32), dims=[])

MIN_DUR, N_ITER, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
INIT = sys.argv[4] if len(sys.argv) > 4 else None
K = "/e/project1/spell/koch13/"
labels = [str(x) for x in np.load(K + "de_9h_table.npz", allow_pickle=True)["labels"]]
LI = {l: i for i, l in enumerate(labels)}
SIL, UNK, V = LI["[space]"], LI["[UNKNOWN]"], len(labels)
SIL_PEN = 5.0  # log-penalty for an inter-word silence, so silence is used only where the audio has one

lex = {}
for line in open(K + "german_ext.dict", encoding="utf8"):
    f = line.rstrip("\n").split("\t")
    lex.setdefault(f[0], [p if p != "spn" else "[UNKNOWN]" for p in f[-1].split()])

out_dim = Dim(80, name="mel")


def log_mel(a):
    p = np.max(np.abs(a))
    a = a / p if p else a
    raw = torch.tensor(a[None, :], dtype=torch.float32)
    td = Dim(int(raw.shape[1]), name="time")
    f, fd = rf.audio.log_mel_filterbank_from_raw(
        Tensor("a", dims=[batch_dim, td], dtype="float32", raw_tensor=raw),
        in_spatial_dim=td, out_dim=out_dim, sampling_rate=16000, window_len=0.025, step_len=0.01,
    )
    return f.copy_compatible_to_dims_raw([batch_dim, fd, out_dim])[0].numpy().astype(np.float64)


tbl = pq.read_table(K + "mls_de/1_hours.parquet")
utts, oov = [], 0
for a, tr in zip(tbl.column("audio").to_pylist(), tbl.column("transcript").to_pylist()):
    arr, sr = sf.read(io.BytesIO(a["bytes"] if isinstance(a, dict) else a), dtype="float32")
    assert sr == 16000
    words = []
    for w in tr.lower().split():
        ph = lex.get(w)
        if ph is None:
            oov += 1
            ph = ["[UNKNOWN]"]
        words.append([LI[p] for p in ph])
    utts.append((log_mel(arr), words))
print(f"{len(utts)} utts, {sum(len(x[0]) for x in utts)} frames, OOV words {oov}", flush=True)


def build(words):
    """State chain: [sil?] w1 [sil?] w2 ... [sil?]; returns labels, selfloop, skip-source per state, start/end sets."""
    lab, selfloop, is_sil, tok_first, tok_last = [], [], [], [], []
    toks = [("sil", SIL)]
    for i, w in enumerate(words):
        toks += [("ph", p) for p in w]
        toks.append(("sil", SIL))
    for kind, l in toks:
        n = 1 if kind == "sil" else MIN_DUR
        tok_first.append(len(lab))
        for k in range(n):
            lab.append(l)
            selfloop.append(k == n - 1)
            is_sil.append(kind == "sil")
        tok_last.append(len(lab) - 1)
    S = len(lab)
    skip = np.full(S, -1)  # state that may jump over the preceding optional silence into this one
    pen = np.zeros(S)
    for ti, (kind, _) in enumerate(toks):
        if kind == "sil":
            if 0 < ti < len(toks) - 1:
                skip[tok_first[ti + 1]] = tok_last[ti - 1]
                pen[tok_first[ti]] = SIL_PEN
    starts = [tok_first[0], tok_first[1]]
    ends = [tok_last[-1], tok_last[-2]]
    return np.array(lab), np.array(selfloop), skip, pen, starts, ends


def viterbi(E, lab, selfloop, skip, pen, starts, ends):
    T, S = E.shape[0], len(lab)
    em = E[:, lab]  # [T, S]
    NEG = -1e18
    score = np.full(S, NEG)
    score[starts] = em[0, starts]
    bp = np.zeros((T, S), dtype=np.int8)
    has_skip = skip >= 0
    for t in range(1, T):
        stay = np.where(selfloop, score, NEG)
        prev = np.concatenate([[NEG], score[:-1]]) - pen
        jump = np.where(has_skip, score[np.maximum(skip, 0)], NEG)
        best = np.maximum(np.maximum(stay, prev), jump)
        bp[t] = np.where(best == stay, 0, np.where(best == prev, 1, 2))
        score = best + em[t]
    s = max(ends, key=lambda e: score[e])
    if score[s] < NEG / 2:
        return None
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        b = bp[t, s]
        s = s if b == 0 else (s - 1 if b == 1 else skip[s])
    return lab[path], path


def estimate(assign):
    s1, s2, n = np.zeros((V, 80)), np.zeros((V, 80)), np.zeros(V)
    for (F, _), fl in zip(utts, assign):
        if fl is None:
            continue
        np.add.at(s1, fl, F)
        np.add.at(s2, fl, F * F)
        np.add.at(n, fl, 1)
    allF = np.concatenate([u[0] for u in utts])
    gm, gv = allF.mean(0), allF.var(0)
    mu = np.where(n[:, None] > 0, s1 / np.maximum(n, 1)[:, None], gm)
    var = np.where(n[:, None] > 1, s2 / np.maximum(n, 1)[:, None] - mu ** 2, gv)
    var = np.maximum(var, 0.05 * gv)  # variance floor
    return mu, var, n


def loglik(F, mu, var):
    return -0.5 * (((F[:, None, :] - mu[None]) ** 2) / var[None] + np.log(var)[None]).sum(-1)


# ---- initialisation
if INIT:
    t0 = np.load(INIT, allow_pickle=True)
    assert [str(x) for x in t0["labels"]] == labels
    mu = t0["means"].astype(np.float64)
    var = np.tile(np.concatenate([u[0] for u in utts]).var(0), (V, 1))
    print("init from", INIT)
else:  # flat start: uniform segmentation over phones (no silence)
    assign = []
    for F, words in utts:
        ph = [p for w in words for p in w]
        idx = np.minimum((np.arange(len(F)) * len(ph)) // len(F), len(ph) - 1)
        assign.append(np.array(ph)[idx])
    mu, var, n = estimate(assign)
    # silence starts as the mean of each utterance's quietest 10% frames
    q = np.concatenate([F[F.mean(1) <= np.percentile(F.mean(1), 10)] for F, _ in utts])
    mu[SIL], var[SIL] = q.mean(0), np.maximum(q.var(0), 1e-3)

graphs = [build(w) for _, w in utts]
hist = []
for it in range(N_ITER):
    assign, tot, failed = [], 0.0, 0
    for (F, _), g in zip(utts, graphs):
        r = viterbi(loglik(F, mu, var), *g)
        if r is None:
            failed += 1
            assign.append(None)
            continue
        fl, path = r
        assign.append(fl)
    mu, var, n = estimate(assign)
    # durations of the current alignment: run-lengths of the STATE path's labels, per token
    runs = {}
    for fl in assign:
        if fl is None:
            continue
        cut = np.flatnonzero(np.diff(fl)) + 1
        for seg in np.split(fl, cut):
            runs.setdefault(int(seg[0]), []).append(len(seg))
    speech = [d for l, v in runs.items() if l != SIL for d in v]
    floor = np.mean(np.array(speech) <= MIN_DUR) * 100  # share of phones at the minimum duration
    hist.append({"iter": it, "failed": failed, "floor_pct": float(floor), "median_dur": float(np.median(speech)),
                 "sil_frac": float(n[SIL] / n.sum())})
    print(hist[-1], flush=True)

med = np.array([np.median(runs[i]) if i in runs else 1.0 for i in range(V)], dtype=np.float32)
np.savez(OUT + ".npz", means=mu.astype(np.float32), labels=np.array(labels, dtype=object))
np.savez(OUT + "_durations.npz", medians=med, labels=np.array(labels, dtype=object))
json.dump({"min_dur": MIN_DUR, "iters": hist, "frame_counts": {labels[i]: int(n[i]) for i in range(V)}, "init": INIT},
          open(OUT + "_stats.json", "w"), indent=1)
print("wrote", OUT)
