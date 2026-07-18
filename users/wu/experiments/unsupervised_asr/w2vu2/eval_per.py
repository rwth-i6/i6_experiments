"""SAE §1c — greedy (viterbi) PER of a trained wav2vec-U 2.0 generator, flashlight-free.

fairseq's own `w2vu_generate.py` reports this number but routes every decoder through flashlight
(`CpuViterbiPath`), which is not installed. Its VITERBI decoder uses an all-zero transition matrix
(`w2l_decoder.py`: `transitions = torch.FloatTensor(N, N).zero_()`), so it is *exactly* frame-wise
argmax followed by CTC-style collapse + silence drop -- no flashlight needed. This reproduces that
path faithfully:

  generator forward (segmentation NONE, no_softmax, matching w2vu_generate's overrides)
    -> per-frame argmax
    -> collapse consecutive-equal (itertools.groupby, = get_tokens)
    -> drop the <SIL> index
    -> edit distance vs the MFA gold phone tokens (drop_sil)

Run under the `w2vu` env. Reads the feature dump (.npy/.lengths/.ids) and the gilkeyio parquet gold,
joined by utterance id. PER = sum(edits) / sum(len(gold_tokens)), the untrimmed-reference convention
(SAE_1c.md), reported per split.
"""

from __future__ import annotations

import argparse
import itertools as it
import json
import os

# torch / editdistance are imported lazily inside the worker functions: compute_gold() runs under the
# speech_llm env (pandas), which does not carry the w2vu env's editdistance.


def _decode_utt(model, feats, sil_idx, device):
    """[T,512] float32 -> collapsed, sil-free list of phone ids (generator vocab)."""
    import torch

    x = torch.from_numpy(feats).to(device).unsqueeze(0)  # [1,T,512]
    pad = torch.zeros(x.shape[:2], dtype=torch.bool, device=device)  # nothing padded (single utt)
    with torch.no_grad():
        res = model(x, padding_mask=pad, dense_x_only=True, segment=True)
    logits = res["logits"][0]  # [T', V]; segmentation NONE => T'==T at the generator output rate
    ids = logits.argmax(-1).tolist()
    collapsed = [k for k, _ in it.groupby(ids)]           # get_tokens: consecutive-equal merge
    return [i for i in collapsed if i != sil_idx]         # drop silence (viterbi blank)


def _load_model(ckpt, data, text_data, device):
    import argparse as _ap

    import fairseq
    from fairseq import checkpoint_utils, utils

    # `unpaired_audio_text` and the `wav2vec_u` model live in fairseq's user-dir example, not the core
    # registry; training registers them via common.user_dir, so the eval must import it too.
    user_dir = os.path.join(os.path.dirname(fairseq.__file__), "examples", "wav2vec", "unsupervised")
    utils.import_user_module(_ap.Namespace(user_dir=user_dir))

    overrides = {
        "task": {"data": data, "text_data": text_data},
        # match w2vu_generate.py's generation-time model overrides exactly
        "model": {"segmentation": {"type": "NONE"}, "no_softmax": True},
    }
    models, _cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [ckpt], arg_overrides=overrides
    )
    model = models[0].to(device).eval()
    dictionary = task.target_dictionary
    return model, dictionary


def compute_gold(split, recipe_dir):
    """utt id -> sil-free gold phone-symbol list, from the gilkeyio MFA parquet.

    Runs on the speech_llm side (pandas + repr_audit), not in the w2vu worker: the parquet read needs
    pandas, which we deliberately keep out of the numpy-1.23.5-pinned w2vu env. The worker consumes
    the json this writes.
    """
    import glob
    import sys

    import pandas as pd

    sys.path.insert(0, recipe_dir)
    from i6_experiments.users.wu.experiments.ssl.analysis import repr_audit

    split_map = {"dev-clean": "dev_clean", "dev-other": "dev_other"}
    hf = os.environ.get("HF_HOME", "/e/project1/spell/common_hf_home")
    base = os.path.join(hf, "hub", "datasets--gilkeyio--librispeech-alignments", "snapshots")
    files = sorted(glob.glob(os.path.join(base, "*", "data", f"{split_map[split]}-*.parquet")))
    assert files, f"no gilkeyio parquet for {split}"
    gold = {}
    for fp in files:
        df = pd.read_parquet(fp, columns=["id", "phonemes"])
        for r in df.itertuples():
            # The PER reference is the phone sequence itself -- map each MFA phone to its stress-free
            # class and drop silence. NOT rasterized to frames (loses sub-frame phones) and NOT
            # run-length collapsed (the reference keeps real repeats; only the hypothesis collapses,
            # exactly as fairseq's get_tokens does on the decoded side).
            seq = [repr_audit.canonical_phone(p["phoneme"]) for p in r.phonemes]
            gold[r.id] = [s for s in seq if s != repr_audit.SIL]
    return gold


def _load_feats(feats_path):
    """Read the dumped dev features once: (mmap'd [N,D], per-utt offsets, ids)."""
    import numpy as np

    base = feats_path[:-4] if feats_path.endswith(".npy") else feats_path
    feats = np.load(base + ".npy", mmap_mode="r")
    lengths = [int(x) for x in open(base + ".lengths")]
    ids = [x.strip() for x in open(base + ".ids")]
    assert len(lengths) == len(ids), (len(lengths), len(ids))
    assert sum(lengths) == feats.shape[0], (sum(lengths), feats.shape[0])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    return feats, offsets, ids


def _score_model(model, dictionary, sil_idx, feats, offsets, ids, id2split, gold_all, device, limit=0):
    """Greedy PER of one loaded generator over the dumped features, per split. -> {split: {...}}."""
    import editdistance
    import numpy as np

    acc = {s: {"errs": 0, "ref": 0, "scored": 0} for s in gold_all}
    seen = {s: 0 for s in gold_all}
    missing = 0
    for u, tag in enumerate(ids):
        s = id2split.get(tag)
        if s is None:
            missing += 1
            continue
        if limit and seen[s] >= limit:
            continue
        seen[s] += 1
        f = np.asarray(feats[offsets[u]:offsets[u + 1]], dtype=np.float32)
        hyp = tuple(dictionary[i] for i in _decode_utt(model, f, sil_idx, device))
        acc[s]["errs"] += editdistance.eval(hyp, gold_all[s][tag])
        acc[s]["ref"] += len(gold_all[s][tag])
        acc[s]["scored"] += 1
    out = {}
    for s, a in acc.items():
        out[s] = {"PER": a["errs"] / max(a["ref"], 1), "errors": a["errs"],
                  "ref_phones": a["ref"], "utts": a["scored"]}
    out["missing_gold"] = missing
    return out


def _load_gold(gold_path):
    with open(gold_path) as f:
        gold_all = {s: {k: tuple(v) for k, v in d.items()} for s, d in json.load(f).items()}
    id2split = {utt: s for s, d in gold_all.items() for utt in d}
    return gold_all, id2split


def _best_num_updates(train_dir):
    """num_updates of the weighted_lm_ppl-selected checkpoint_best.pt (None if absent/unreadable).

    Tolerates a half-written checkpoint_best.pt so the curve can be run against a *live* training dir
    (early read while the GAN is still training) -- a failed read just leaves the best row unmarked.
    """
    from fairseq import checkpoint_utils
    p = os.path.join(train_dir, "checkpoint_best.pt")
    if not os.path.exists(p):
        return None
    try:
        st = checkpoint_utils.load_checkpoint_to_cpu(p)
        oh = st.get("optimizer_history")
        return oh[-1].get("num_updates") if oh else None
    except Exception as e:
        print(f"WARNING: could not read checkpoint_best.pt ({e}); leaving best unmarked", flush=True)
        return None


def run_curve(train_dir, data, text_data, feats_path, gold_path, out_path, stride, device, limit=0):
    """PER trajectory over every save_interval checkpoint, so the PER-min can be compared to the
    unsupervised (weighted_lm_ppl) checkpoint_best -- the objective-alignment check."""
    import glob
    import re

    gold_all, id2split = _load_gold(gold_path)
    feats, offsets, ids = _load_feats(feats_path)

    parsed = []
    for c in glob.glob(os.path.join(train_dir, "checkpoint_*_*.pt")):
        m = re.search(r"checkpoint_(\d+)_(\d+)\.pt$", os.path.basename(c))
        if m:
            parsed.append((int(m.group(2)), int(m.group(1)), c))  # (num_updates, epoch, path)
    parsed.sort()
    assert parsed, f"no interval checkpoints under {train_dir}"
    sub = parsed[::max(stride, 1)]
    best_upd = _best_num_updates(train_dir)
    if best_upd is not None and best_upd not in {u for u, _, _ in sub}:
        sub += [(u, e, c) for u, e, c in parsed if u == best_upd]  # never let stride drop the best
        sub.sort()

    print(f"curve: {len(sub)}/{len(parsed)} checkpoints (stride={stride}), best={best_upd}", flush=True)
    curve = []
    for upd, ep, path in sub:
        model, dictionary = _load_model(path, data, text_data, device)
        sil_idx = dictionary.index("<SIL>")
        assert sil_idx != dictionary.unk(), "<SIL> not in generator dictionary"
        res = _score_model(model, dictionary, sil_idx, feats, offsets, ids, id2split, gold_all,
                           device, limit=limit)
        row = {"updates": upd, "epoch": ep, "is_best": (upd == best_upd),
               **{s: res[s]["PER"] for s in gold_all}}
        curve.append(row)
        pers = " ".join(f"{s}={res[s]['PER']:.3f}" for s in gold_all)
        print(f"  upd={upd:>7} ep={ep:>4}{' *BEST' if row['is_best'] else '     '}  {pers}", flush=True)
        del model

    do = "dev-other" if "dev-other" in gold_all else sorted(gold_all)[0]
    best_row = next((r for r in curve if r["is_best"]), None)
    argmin = min(curve, key=lambda r: r[do])
    summary = {
        "best_updates": best_upd,
        "ppl_best": best_row,                  # what the unsupervised metric selected
        "per_min": argmin,                     # the actual PER-min checkpoint on the trajectory
        "selection_gap": (None if best_row is None else round(best_row[do] - argmin[do], 4)),
        "curve": curve,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    if best_row is not None:
        print(f"SELECTION GAP ({do}): ppl-best {best_row[do]:.3f} @{best_upd}  vs  "
              f"PER-min {argmin[do]:.3f} @{argmin['updates']}  =>  {summary['selection_gap']:+.3f}",
              flush=True)


def dump_labels(ckpt, data, text_data, feats_path, out_path, device, limit=0):
    """§1d stage-1 pseudo-labels: the GAN teacher's greedy phone transcript for every utt in a feats
    dump, written as json {id: "p1 p2 ..."} so a CTC student can be trained on them.

    Identical decode to the PER hypothesis (collapsed, sil-free) -- the pseudo-label a student sees is
    exactly the sequence §1c scored -- keyed by utt id so the audio manifest can be joined to it later.
    """
    import numpy as np

    model, dictionary = _load_model(ckpt, data, text_data, device)
    sil_idx = dictionary.index("<SIL>")
    assert sil_idx != dictionary.unk(), "<SIL> not in generator dictionary"
    feats, offsets, ids = _load_feats(feats_path)

    labels, empty = {}, 0
    for u, tag in enumerate(ids):
        if limit and u >= limit:
            break
        f = np.asarray(feats[offsets[u]:offsets[u + 1]], dtype=np.float32)
        syms = [dictionary[i] for i in _decode_utt(model, f, sil_idx, device)]
        empty += not syms
        labels[tag] = " ".join(syms)
    with open(out_path, "w") as fo:
        json.dump({"labels": labels, "utts": len(labels), "empty": empty}, fo)
    print(f"dump_labels: {len(labels)} utts, {empty} empty (sil-free collapsed)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")                         # single-checkpoint mode / dump-labels mode
    ap.add_argument("--train-dir")                    # curve mode: eval every interval checkpoint here
    ap.add_argument("--stride", type=int, default=1)  # curve mode: eval every stride-th checkpoint
    ap.add_argument("--dump-labels", action="store_true")  # write teacher pseudo-labels, no scoring
    ap.add_argument("--data", required=True)          # fairseq task.data (for dict build)
    ap.add_argument("--text-data", required=True)     # fairseq task.text_data (dict.txt)
    ap.add_argument("--feats", required=True)         # {split}.npy  (dumped features, all utts)
    ap.add_argument("--gold")                         # json {split: {id: [phones]}}; not needed to dump
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)   # cap utts scored per split (0 = all); testing
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dump_labels:
        assert args.ckpt, "dump-labels needs --ckpt (the teacher GAN checkpoint)"
        dump_labels(args.ckpt, args.data, args.text_data, args.feats, args.out, device,
                    limit=args.limit)
        return

    assert args.gold, "scoring modes need --gold"
    if args.train_dir:
        run_curve(args.train_dir, args.data, args.text_data, args.feats, args.gold, args.out,
                  args.stride, device, limit=args.limit)
        return

    assert args.ckpt, "single mode needs --ckpt (or pass --train-dir for the curve)"
    gold_all, id2split = _load_gold(args.gold)
    model, dictionary = _load_model(args.ckpt, args.data, args.text_data, device)
    sil_idx = dictionary.index("<SIL>")
    assert sil_idx != dictionary.unk(), "<SIL> not in generator dictionary"
    feats, offsets, ids = _load_feats(args.feats)
    out = _score_model(model, dictionary, sil_idx, feats, offsets, ids, id2split, gold_all,
                       device, limit=args.limit)
    for s in gold_all:
        print(s, json.dumps(out[s]), flush=True)
    if out["missing_gold"]:
        print(f"WARNING: {out['missing_gold']} utts had no gold and were skipped", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
