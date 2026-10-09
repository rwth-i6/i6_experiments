#!/usr/bin/env python3
"""
Task E.2 -- the validated two-stage criterion on a REAL (no oracle segmentation) token stream.

E.1 (`calc_emission_ladder.py`, RESULTS.md II.E.1) moved the supervised ceiling on real audio from
71.1% PER to 41.9% (best discrete rung).  E.2 hands one of those rungs to the criterion stack that
already works on cheat-seg (II.D.3: HMM-EM from uniform -> order-4 n-gram refinement, 26.4% PER
against a 26.0% ceiling) and asks whether it transfers.

The criterion code is imported UNCHANGED from `calc_soft_map_search.py` / `calc_hmm_map_search.py`;
this script only builds the token stream and the count tables, and scores the result.  Nothing here
uses labels except the explicitly-labelled supervised reference block.

Why the segmented rung: the whole II.B / II.D stack was built and validated at `R ~ 1` (one audio
token per phoneme).  The `pooled` stream is at R = 1.33 and the collapse pushforward that would
correct for that is only derived for n <= 2 (Task C), where cold starts are at chance.  Segmenting
the stream to R ~ 1 puts the criterion back in the regime it was validated in, at the cost of some
supervised ceiling.  `--overseg 0` disables segmentation if you want the R = 1.33 stream anyway.

Modes:
  stream   build (and cache) the token stream + the paired eval/ceiling sets            [~20 min]
  ceiling  the supervised reference for THIS stream (labels; a reference, not a result) [~10 min]
  em       stage 1: HMM-EM from uniform for one seed -> npz                             [~1-2 h]
  refine   stage 2: order-4 n-gram refinement from the loss-selected EM npz -> npz       [~2 h]
  score    decode an npz map on the paired eval set: PER + shuffle control

  ./calc_e2_unsup_map.py stream  --stage pooled --overseg 1.0 --k 512
  ./calc_e2_unsup_map.py em      --seed 1
  ./calc_e2_unsup_map.py refine  --em-npz "runs/em_seed*.npz"
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

from calc_cheat_seg_identifiability import NUM_PHON, _load_hdfs, PHONEME_HDF
from calc_corpus_diagnostics import read_vocab, VOCAB
from calc_unsegmented_map import collapse, per, per_shuffled
from calc_emission_ladder import (
    FeatureStore,
    kmeans_fit,
    assign_stream,
    segment_pool,
    discrete_ceiling,
)
from calc_soft_map_search import (
    SoftMapLoss,
    unigram_bigram,
    trigram_dense,
    trigram_sparse,
    fourgram_dense,
    fourgram_sparse,
    optimize,
)
from calc_hmm_map_search import (
    HmmObjective,
    markov_from_bigram,
    smooth_chain,
    make_batches,
    run_em,
)

DEFAULT_DIR = "/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskE/e2"


def nan_acc(_assign):
    """There is no oracle map on real data -- PER is computed after the fact, never for selection."""
    return float("nan")


# ---------------------------------------------------------------- stream


def build_stream(args):
    """
    Build the rung's discrete token stream and write everything the other modes need to one npz.

    Splits are the II.C.2 / E.1 harness on the tag universe of the feature dump, with the same seed,
    so the eval set is the same 1000 utterances E.1 measured its ceilings on.
    """
    store = FeatureStore(args.stage)
    phon = _load_hdfs([PHONEME_HDF % i for i in range(10)])
    common = sorted(set(store.index) & set(phon))
    rng = np.random.default_rng(args.split_seed)
    common = [common[i] for i in rng.permutation(len(common))]
    a0 = args.num_split_audio
    t0 = a0 + args.num_split_text
    e0 = t0 + args.num_eval_utts
    audio_pool, text_pool = common[:a0], common[a0:t0]
    eval_tags, ceil_tags = common[t0:e0], common[e0 : e0 + args.num_ceiling_utts]

    audio_tags = audio_pool[: args.num_audio_utts]
    lm_tags = text_pool[: args.num_lm_utts]
    hmm_tags = text_pool[: args.num_hmm_text_utts]
    need_audio = sorted(set(audio_tags) | set(eval_tags) | set(ceil_tags))

    # phoneme axis: the symbols present in the (unpaired) text split, compacted to 0..m-1
    t1_full = np.bincount(np.concatenate([phon[t].astype(np.int64) for t in text_pool]),
                          minlength=NUM_PHON)
    keep_p = np.where(t1_full > 0)[0]
    m = len(keep_p)
    p_pos = -np.ones(NUM_PHON, dtype=np.int64)
    p_pos[keep_p] = np.arange(m)
    phon_c = {t: p_pos[phon[t].astype(np.int64)].astype(np.int16)
              for t in set(text_pool) | set(eval_tags) | set(ceil_tags)}
    assert all((phon_c[t] >= 0).all() for t in eval_tags)

    mean_s = float(np.mean([len(phon_c[t]) for t in eval_tags]))
    mean_t_feat = float(np.mean([store.length(t) for t in audio_pool[:4000]]))
    ratio = mean_t_feat / mean_s
    print("=== E.2 stream: stage %s, overseg %.2f, k %d ===" % (args.stage, args.overseg, args.k))
    print("  feature stage %.2f tokens/utt, reference %.2f phonemes/utt -> R = %.3f"
          % (mean_t_feat, mean_s, ratio))

    t0s = time.time()
    if args.overseg > 0:
        km_src = segment_pool(store, audio_pool[: args.kmeans_utts], ratio, args.overseg)
        pooled = segment_pool(store, need_audio, ratio, args.overseg)
        X = np.concatenate([km_src[t] for t in km_src])
        getter = lambda t: pooled[t]
    else:
        pooled = None
        X = store.sample_frames(audio_pool[: args.kmeans_utts], args.kmeans_frames, rng)
        getter = None
    print("  segmentation+pooling %.0fs, k-means fit on %s" % (time.time() - t0s, (X.shape,)))
    cen = kmeans_fit(X, args.k, args.split_seed)
    stream = assign_stream(store, need_audio, cen, get=getter)
    print("  stream: %.2f tokens/utt on the eval set (k = %d)"
          % (float(np.mean([len(stream[t]) for t in eval_tags])), args.k))

    out = dict(k=args.k, m=m, keep_p=keep_p, stage=args.stage, overseg=args.overseg, ratio=ratio,
               centroids=cen)
    for name, tags, src in (("audio", audio_tags, stream), ("eval_a", eval_tags, stream),
                            ("ceil_a", ceil_tags, stream), ("lm", lm_tags, phon_c),
                            ("hmm", hmm_tags, phon_c), ("eval_p", eval_tags, phon_c),
                            ("ceil_p", ceil_tags, phon_c)):
        out["%s_lens" % name] = np.array([len(src[t]) for t in tags], dtype=np.int32)
        out["%s_flat" % name] = np.concatenate([src[t] for t in tags]).astype(np.int16)
        out["%s_tags" % name] = np.array(tags)
    os.makedirs(os.path.dirname(args.stream_npz), exist_ok=True)
    np.savez_compressed(args.stream_npz, **out)
    print("  wrote %s" % args.stream_npz)


def load_stream(path):
    z = np.load(path, allow_pickle=False)
    d = {"k": int(z["k"]), "m": int(z["m"]), "keep_p": z["keep_p"], "ratio": float(z["ratio"]),
         "stage": str(z["stage"]), "overseg": float(z["overseg"])}
    for name in ("audio", "eval_a", "ceil_a", "lm", "hmm", "eval_p", "ceil_p"):
        lens = z["%s_lens" % name]
        flat = z["%s_flat" % name]
        off = np.concatenate([[0], np.cumsum(lens)])
        d[name] = [flat[off[i] : off[i + 1]] for i in range(len(lens))]
        d["%s_tags" % name] = [str(t) for t in z["%s_tags" % name]]
    return d


def _as_dict(tags, seqs):
    return dict(zip(tags, seqs))


# ---------------------------------------------------------------- scoring (labels, after the fact)


def score(assign, d, label, extra=""):
    ev_a = _as_dict(d["eval_a_tags"], d["eval_a"])
    ev_p = _as_dict(d["eval_p_tags"], d["eval_p"])
    tags = d["eval_a_tags"]
    e, lh, lr = per(assign, tags, ev_a, ev_p)
    sh = per_shuffled(assign, tags, ev_a, ev_p)
    print("  %-28s PER %6.2f%%  (hyp %.1f / ref %.1f)   shuffled %6.2f%%  gap %+.2f%s"
          % (label, e, lh, lr, sh, sh - e, extra))
    return dict(per=e, hyp=lh, ref=lr, shuffled=sh, gap=sh - e)


# ---------------------------------------------------------------- modes


def _init_from_npz(args, n, m, rng):
    """
    A hard-ish start at a given assignment, exactly `calc_soft_map_search.init_logits("assign", ...)`:
    +init_scale on the chosen phoneme, -init_scale elsewhere.

    `--init ceiling` starts from the SUPERVISED map of this stream. That uses labels, so it is never
    a result -- it is the identifiability probe the plan requires (the analogue of the cheat-seg
    oracle start): it asks whether the criterion's optimum sits near a good map, or walks away from
    it, which is what separates "the objective is wrong here" from "the search never found it".
    """
    if args.init == "ceiling":
        path = args.stream_npz.replace(".npz", ".ceiling.npz")
    else:
        cands = sorted(glob.glob(args.em_npz))
        assert cands, "no npz matched %s" % args.em_npz
        loaded = [(np.load(c), c) for c in cands]
        for z, c in loaded:
            print("  %-34s hardened %.5f  (PER %.2f%%, gap %+.2f -- NOT used for selection)"
                  % (os.path.basename(c), float(z["hard"]), float(z["per"]), float(z["gap"])))
        best = min(loaded, key=lambda zc: float(zc[0]["hard"]))
        print("  loss-selected: %s\n" % os.path.basename(best[1]))
        path = best[1]
    a = np.load(path)["assign"].astype(np.int64)
    assert a.shape == (n,), (a.shape, n)
    th = np.full((n, m), -args.init_scale)
    th[np.arange(n), a] = args.init_scale
    return torch.as_tensor(th, dtype=torch.float64)


def mode_ceiling(args):
    d = load_stream(args.stream_npz)
    ev_a, ev_p = _as_dict(d["eval_a_tags"], d["eval_a"]), _as_dict(d["eval_p_tags"], d["eval_p"])
    ce_a, ce_p = _as_dict(d["ceil_a_tags"], d["ceil_a"]), _as_dict(d["ceil_p_tags"], d["ceil_p"])
    stream = dict(ev_a); stream.update(ce_a)
    phon = dict(ev_p); phon.update(ce_p)
    print("=== supervised reference for THIS stream (LABELS USED -- a reference, not a result) ===")
    r = discrete_ceiling("E.2 stream", stream, d["eval_a_tags"], d["ceil_a_tags"], phon,
                         d["k"], d["m"], args)
    print("  ceiling %.2f%% PER   (%s, %.1f tokens/utt, I = %.3f nats)" % (r["per"], r["via"], r["tokens"], r["mi"]))
    rng = np.random.default_rng(0)
    score(rng.integers(0, d["m"], d["k"]), d, "random map")
    score(r["assign"], d, "supervised ceiling")
    np.savez(args.stream_npz.replace(".npz", ".ceiling.npz"), assign=r["assign"], per=r["per"])


def mode_em(args):
    d = load_stream(args.stream_npz)
    n, m = d["k"], d["m"]
    c1, C2 = unigram_bigram(d["audio"], n)
    pi, A = markov_from_bigram(C2 / C2.sum())
    pi, A = smooth_chain(pi, A, args.smooth)
    batches = make_batches([s.astype(np.int64) for s in d["hmm"]], args.batch_size)
    ntok = float(sum(len(s) for s in d["hmm"]))
    print("=== E.2 stage 1: HMM-EM, init %s, seed %d ===" % (args.init, args.seed))
    print("  %d audio utts -> %d x %d frozen chain; %d text utts = %.0f tokens; %d x %d map"
          % (len(d["audio"]), n, n, len(d["hmm"]), ntok, n, m))
    obj = HmmObjective(torch.as_tensor(pi), torch.as_tensor(A), batches, ntok)
    g = np.random.default_rng(args.seed)
    if args.init == "uniform":
        logits = torch.as_tensor(g.normal(0.0, 1e-3, (n, m)), dtype=torch.float64)
    else:
        logits = _init_from_npz(args, n, m, g)
    M = torch.softmax(logits, dim=1)
    M, secs = run_em(obj, M, args.em_steps, nan_acc, m, args.log_every, "em-%s-%d" % (args.init, args.seed))
    assign = M.argmax(1).numpy()
    hard = float(obj.hard(assign, m))
    print("  [em-%s-%d] hardened objective %.5f | %.0fs" % (args.init, args.seed, hard, secs))
    s = score(assign, d, "em %s seed %d" % (args.init, args.seed))
    out = args.out or os.path.join(os.path.dirname(args.stream_npz),
                                   "em_%s_seed%d.npz" % (args.init, args.seed))
    np.savez(out, assign=assign, hard=hard, seed=args.seed, **{k: v for k, v in s.items()})
    print("  wrote %s" % out)


def mode_refine(args):
    d = load_stream(args.stream_npz)
    n, m = d["k"], d["m"]
    c1, C2 = unigram_bigram(d["audio"], n)
    t1_full, t2_full = unigram_bigram(d["lm"], m)
    t1 = t1_full / t1_full.sum()
    t2 = t2_full / t2_full.sum()
    C3 = trigram_sparse(d["audio"], n)
    t3 = trigram_dense(d["lm"], m); t3 = t3 / t3.sum()
    C4 = fourgram_sparse(d["audio"], n)
    t4 = fourgram_dense(d["lm"], m); t4 = t4 / t4.sum()
    w = [float(v) for v in args.w.split(",")]
    print("=== E.2 stage 2: order-4 n-gram, init %s, seed %d (w = %s) ==="
          % (args.init, args.seed, args.w))
    print("  audio %d utts -> %d distinct triples / %d 4-grams over %d prefixes"
          % (len(d["audio"]), len(C3[2]), len(C4[4]), len(C4[0])))
    print("  text  %d utts -> trigram %.1f%% of %d cells, 4-gram %.2f%% of %d cells"
          % (len(d["lm"]), 100 * (t3 > 0).mean(), m ** 3, 100 * (t4 > 0).mean(), m ** 4))
    loss_fn = SoftMapLoss(c1, C2, t1, t2, lam=args.lam, C3=C3, t3=t3, tri_scale=1.0,
                          tri_backoff=args.tri_backoff, C4=C4, t4=t4, four_scale=1.0,
                          four_backoff=args.four_backoff, cond_weights=w)
    g = np.random.default_rng(args.seed)
    if args.init == "uniform":
        logits = torch.as_tensor(g.normal(0.0, 1e-3, (n, m)), dtype=torch.float64)
        a0 = None
    else:
        logits = _init_from_npz(args, n, m, g)
        a0 = logits.argmax(1).numpy()
    args.keep_best = True
    args.log_every = args.refine_log_every
    assign, rep = optimize(loss_fn, logits, args, nan_acc, "refine-%s-%d" % (args.init, args.seed))
    if a0 is not None:
        score(a0, d, "start (%s)" % args.init)
    s = score(assign, d, "order-4 from %s seed %d" % (args.init, args.seed))
    out = args.out or os.path.join(os.path.dirname(args.stream_npz),
                                   "refined_%s_seed%d.npz" % (args.init, args.seed))
    np.savez(out, assign=assign, hard=rep["hard"], **{k: v for k, v in s.items()})
    print("  wrote %s" % out)


def mode_score(args):
    d = load_stream(args.stream_npz)
    for f in sorted(glob.glob(args.em_npz)):
        z = np.load(f)
        score(z["assign"], d, os.path.basename(f),
              extra="   hardened %.5f" % float(z["hard"]) if "hard" in z else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["stream", "ceiling", "em", "refine", "score"])
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--stream-npz", default=None)
    # --- rung
    p.add_argument("--stage", default="pooled")
    p.add_argument("--overseg", type=float, default=1.0, help="0 = no segmentation")
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--kmeans-utts", type=int, default=2000)
    p.add_argument("--kmeans-frames", type=int, default=400000)
    # --- splits (same harness + seed as E.1, so the eval set is the one E.1 measured)
    p.add_argument("--split-seed", type=int, default=1)
    p.add_argument("--num-split-audio", type=int, default=20000)
    p.add_argument("--num-split-text", type=int, default=60000)
    p.add_argument("--num-eval-utts", type=int, default=1000)
    p.add_argument("--num-ceiling-utts", type=int, default=3000)
    p.add_argument("--num-audio-utts", type=int, default=6000, help="audio statistics (II.B envelope)")
    p.add_argument("--num-lm-utts", type=int, default=30000, help="text n-gram statistics")
    p.add_argument("--num-hmm-text-utts", type=int, default=3000, help="text for the HMM forward pass")
    # --- stage 1
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--em-steps", type=int, default=400)
    p.add_argument("--smooth", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    # --- stage 2
    p.add_argument("--em-npz", default=None)
    p.add_argument("--init", default="em", choices=["em", "uniform", "ceiling"],
                   help="em = loss-selected stage-1 map; ceiling = the SUPERVISED map (identifiability probe, uses labels); uniform = the II.B cold start")
    p.add_argument("--w", default="5,5,10,20")
    p.add_argument("--lam", type=float, default=5.0)
    p.add_argument("--tri-backoff", type=float, default=0.05)
    p.add_argument("--four-backoff", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.02)
    p.add_argument("--init-scale", type=float, default=2.0)
    p.add_argument("--entropy-scale", type=float, default=0.0)
    # --- supervised reference (labels)
    p.add_argument("--per-cd-sweeps", type=int, default=1)
    p.add_argument("--per-cd-max-k", type=int, default=600)
    p.add_argument("--per-cd-top-k", type=int, default=128)
    p.add_argument("--num-per-cd-utts", type=int, default=150)
    p.add_argument("--log-every", type=int, default=50, help="EM")
    p.add_argument("--refine-log-every", type=int, default=250,
                   help="stage 2; each log line hardens the map, which is not cheap at order 4")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.stream_npz is None:
        seg = "seg%.2f" % args.overseg if args.overseg > 0 else "noseg"
        args.stream_npz = os.path.join(args.dir, "%s_%s_k%d" % (args.stage, seg, args.k), "stream.npz")
    if args.em_npz is None:
        args.em_npz = os.path.join(os.path.dirname(args.stream_npz), "em_seed*.npz")
    t0 = time.time()
    {"stream": build_stream, "ceiling": mode_ceiling, "em": mode_em,
     "refine": mode_refine, "score": mode_score}[args.mode](args)
    print("(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
