"""SAE §4a step 4 -- the dev reverse log-likelihood of one GAN checkpoint (the activity readout).

Runs under the `w2vu` env python, GPU.  For every dev utterance it takes the generator's stride-3
logits, restricts them to the 40 phone columns, and evaluates the SAME alignment-sum reverse term
the training uses (tau = 1, beta = 0, blank-free, d_min = 2, D = 25 / D_sil = 50, band 25) twice:

  own   -- under the checkpoint's own phi (absent for a lam_rev = 0 arm; then only `cold` is written)
  cold  -- under a cold-initialized phi at `--cold-seed` (the arm's own `rev_phi_seed`, i.e. phi
           before any update)

REPORTING CONVENTION (pre-registered here, with the code that produces it):
  * the headline per split is ``logz_per_frame`` = sum_utt log Z_tau / sum_utt S, the corpus-level
    reverse LOG-LIKELIHOOD per retained frame.  HIGHER is better.
  * ``mean_utt_logz_per_frame`` is the mean over utterances of log Z_tau / S, reported beside it.
  * utterances with Z_tau = 0 are counted (``z_zero``) and excluded from both, never averaged in.
  * the generator runs in EVAL mode (batch-norm running statistics), the readout is under no_grad,
    and the logits are the pre-segmentation stride-3 ones -- the same tensor the training term uses.

The activity criterion of SAE_4A_attrib.md ("Design-review amendments", step 4) is read off these
two numbers: the arm's own phi must beat the cold phi.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _splits_of(gold_path):
    with open(gold_path) as fh:
        gold = json.load(fh)
    return {utt: split for split, d in gold.items() for utt in d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-data", required=True)
    ap.add_argument("--feats", required=True)       # valid.npy of the feature dump
    ap.add_argument("--rev-units", required=True)   # W2vu2RevUnitsJob output dir
    ap.add_argument("--rev-split", default="valid")  # which {split}.rev500 / {split}.eta.npy
    ap.add_argument("--gold", required=True)        # GoldPhonesJob json: the dev split map
    ap.add_argument("--user-dir", action="append", default=[])
    ap.add_argument("--cold-seed", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_per import _load_feats, _load_model

    model, dictionary = _load_model(args.ckpt, args.data, args.text_data,
                                    "cuda" if torch.cuda.is_available() else "cpu",
                                    user_dirs=args.user_dir)
    device = next(model.parameters()).device
    for d in args.user_dir:
        sys.path.insert(0, os.path.dirname(os.path.abspath(d)))
    from w2vu_rev import rev_term as rt

    feats, offsets, ids = _load_feats(args.feats)
    id2split = _splits_of(args.gold)
    units = [np.array(line.split(), dtype=np.int64)
             for line in open(os.path.join(args.rev_units, f"{args.rev_split}.rev500"))]
    eta = np.load(os.path.join(args.rev_units, f"{args.rev_split}.eta.npy"))
    assert len(units) == len(ids) == eta.shape[0], (len(units), len(ids), eta.shape)
    for i, u in enumerate(units):
        assert len(u) == offsets[i + 1] - offsets[i], f"row {i}: units vs features length"

    phis = {}
    own = getattr(model, "phi", None)
    if own is not None:
        phis["own"] = own
    cold_seed = args.cold_seed
    if cold_seed is None:
        cold_seed = int(getattr(getattr(model, "cfg", None), "rev_phi_seed", 0))
    phis["cold"] = rt.build_phi(seed=cold_seed).to(device)
    for phi in phis.values():
        phi.eval()

    # the arm's own term settings (tau, fp64) when it has them, else the registered defaults
    rev_cfg = getattr(model, "rev_cfg", rt.RevTermConfig())
    cols = torch.tensor(rt.phone_columns(dictionary.index, dictionary.unk()),
                        dtype=torch.long, device=device)
    acc = {k: {} for k in phis}
    order = [u for u in range(len(ids)) if ids[u] in id2split]
    if args.limit:
        order = order[: args.limit]

    with torch.no_grad():
        for start in range(0, len(order), args.batch):
            chunk = order[start: start + args.batch]
            sizes = [int(offsets[u + 1] - offsets[u]) for u in chunk]
            s_max = max(sizes)
            x = torch.zeros(len(chunk), s_max, feats.shape[1], dtype=torch.float32)
            pad = torch.ones(len(chunk), s_max, dtype=torch.bool)
            uu = torch.zeros(len(chunk), s_max, dtype=torch.long)
            for i, (u, size) in enumerate(zip(chunk, sizes)):
                x[i, :size] = torch.from_numpy(
                    np.asarray(feats[offsets[u]:offsets[u + 1]], dtype=np.float32)
                )
                pad[i, :size] = False
                uu[i, :size] = torch.from_numpy(units[u])
            x, pad, uu = x.to(device), pad.to(device), uu.to(device)
            ee = torch.from_numpy(eta[chunk]).to(device)

            gen = model.generator(x, None, pad)
            dense_x, dense_pad = gen["dense_x"], gen["dense_padding_mask"]
            feat_lens = (~dense_pad).sum(-1).long()
            unit_lens = torch.tensor(sizes, dtype=torch.long, device=device)
            assert torch.equal(feat_lens, torch.div(unit_lens + 2, 3, rounding_mode="floor"))
            log_q = rt.log_q_from_dense(dense_x, cols)

            for name, phi in phis.items():
                out = rt.reverse_term(log_q=log_q, units=uu, unit_lens=unit_lens,
                                      feat_lens=feat_lens, eta=ee.to(log_q.dtype), phi=phi,
                                      cfg=rev_cfg)
                for i, u in enumerate(chunk):
                    acc[name][ids[u]] = (float(out.per_utt[i]), sizes[i], bool(out.keep[i]))
            if start % (args.batch * 50) == 0:
                print(f"{start}/{len(order)}", flush=True)

    report = {"checkpoint": args.ckpt, "cold_seed": cold_seed, "has_own_phi": own is not None,
              "tau": rev_cfg.tau, "prior_weight": rev_cfg.prior_weight, "splits": {}}
    for name, per_utt in acc.items():
        for split in sorted(set(id2split.values())):
            rows = [(v, n, k) for t, (v, n, k) in per_utt.items() if id2split[t] == split]
            kept = [(v, n) for v, n, k in rows if k]
            frames = sum(n for _, n in kept)
            logz = -sum(v * n for v, n in kept)          # per_utt = -log Z / S
            entry = {
                "utts": len(rows),
                "kept": len(kept),
                "z_zero": len(rows) - len(kept),
                "frames": frames,
                "logz_per_frame": logz / frames if frames else None,
                "mean_utt_logz_per_frame": -sum(v for v, _ in kept) / len(kept) if kept else None,
            }
            report["splits"].setdefault(split, {})[name] = entry
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
