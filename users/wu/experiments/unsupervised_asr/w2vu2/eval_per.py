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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)          # fairseq task.data (for dict build)
    ap.add_argument("--text-data", required=True)     # fairseq task.text_data (dict.txt)
    ap.add_argument("--feats", required=True)         # {split}.npy  (dumped dev features, all utts)
    ap.add_argument("--gold", required=True)          # json {split: {id: [phones]}}, from compute_gold
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)   # cap utts scored per split (0 = all); testing
    args = ap.parse_args()

    import editdistance
    import numpy as np
    import torch

    with open(args.gold) as f:
        gold_all = {s: {k: tuple(v) for k, v in d.items()} for s, d in json.load(f).items()}
    # utt id -> its split, so scoring is robust to any dropped-short utts shifting the .ids order
    id2split = {utt: s for s, d in gold_all.items() for utt in d}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, dictionary = _load_model(args.ckpt, args.data, args.text_data, device)
    sil_idx = dictionary.index("<SIL>")
    assert sil_idx != dictionary.unk(), "<SIL> not in generator dictionary"

    base = args.feats[:-4] if args.feats.endswith(".npy") else args.feats
    feats = np.load(base + ".npy", mmap_mode="r")
    lengths = [int(x) for x in open(base + ".lengths")]
    ids = [x.strip() for x in open(base + ".ids")]
    assert len(lengths) == len(ids), (len(lengths), len(ids))
    assert sum(lengths) == feats.shape[0], (sum(lengths), feats.shape[0])

    offsets = np.concatenate([[0], np.cumsum(lengths)])
    acc = {s: {"errs": 0, "ref": 0, "scored": 0} for s in gold_all}
    seen = {s: 0 for s in gold_all}
    missing = 0
    for u, tag in enumerate(ids):
        s = id2split.get(tag)
        if s is None:
            missing += 1
            continue
        if args.limit and seen[s] >= args.limit:
            continue
        seen[s] += 1
        f = np.asarray(feats[offsets[u]:offsets[u + 1]], dtype=np.float32)
        hyp = tuple(dictionary[i] for i in _decode_utt(model, f, sil_idx, device))
        acc[s]["errs"] += editdistance.eval(hyp, gold_all[s][tag])
        acc[s]["ref"] += len(gold_all[s][tag])
        acc[s]["scored"] += 1

    out = {}
    for s, a in acc.items():
        out[s] = {
            "PER": a["errs"] / max(a["ref"], 1),
            "errors": a["errs"],
            "ref_phones": a["ref"],
            "utts": a["scored"],
        }
        print(s, json.dumps(out[s]), flush=True)
    out["missing_gold"] = missing
    if missing:
        print(f"WARNING: {missing} utts had no gold and were skipped", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
