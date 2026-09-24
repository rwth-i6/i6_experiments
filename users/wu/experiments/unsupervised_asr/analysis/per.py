"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_eval_jobs.py (_hdf_sequences :12,
_hdf_lengths :32, BlankfreeGreedyPerJob :48, epoch_reads :208) and eval_jobs.py (edit_counts :490).

Greedy phone PER of a posterior dump against the frozen MFA dev gold, and the per-checkpoint read
chain that produces it:

    ExtractSubmoduleCheckpointJob (prefix "recognizer.")      theta alone
      -> ReturnnForwardJobV2 posterior dump (.posterior)      posteriors.hdf
      -> BlankfreeGreedyPerJob                                per.json / per.txt / greedy_*.json

``BlankfreeGreedyPerJob`` reads the 40-column blank-free dump (39 ARPAbet + SIL, ``phones.PHONES``
order): per-frame argmax, collapse repeats, drop SIL, Levenshtein against the sil-free gold.
Unchanged in arithmetic and output format.  The source's 41-column ``emc_train_jobs.DecodeStatsJob``
(CTC, blank 0) is not ported: the phase-4a reference read graph never used it (its blank-free
decode statistics are ``BlankfreeGreedyPerJob``'s ``decode_stats.json``) and no port dump has 41
columns.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Optional, Sequence, Tuple

from sisyphus import Job, Task, tk

__all__ = [
    "edit_counts",
    "BlankfreeGreedyPerJob",
    "epoch_reads",
]

# =============================================================================================
# edit distance
# =============================================================================================
def edit_counts(hyp: Sequence[str], ref: Sequence[str]) -> Tuple[int, int, int]:
    """Levenshtein backtrace -> (substitutions, deletions, insertions) of ``hyp`` against ``ref``.

    Deletion = a reference symbol missing from the hypothesis, insertion = a hypothesis symbol with
    no reference counterpart (the sclite convention).  Unit costs; ties resolve substitution before
    deletion before insertion, which changes the S/D/I split of an ambiguous alignment but never
    their sum (and PER is read off the sum).
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = distance between ref[:i] and hyp[:j]
    prev = list(range(m + 1))
    ops = [[0] * (m + 1) for _ in range(n + 1)]  # 0 = match, 1 = sub, 2 = del, 3 = ins
    for j in range(1, m + 1):
        ops[0][j] = 3
    for i in range(1, n + 1):
        cur = [prev[0] + 1] + [0] * m
        ops[i][0] = 2
        for j in range(1, m + 1):
            same = ref[i - 1] == hyp[j - 1]
            c_sub = prev[j - 1] + (0 if same else 1)
            c_del = prev[j] + 1
            c_ins = cur[j - 1] + 1
            best = min(c_sub, c_del, c_ins)
            cur[j] = best
            ops[i][j] = 0 if (same and best == c_sub) else (1 if best == c_sub else (2 if best == c_del else 3))
        prev = cur
    i, j, sub, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        op = ops[i][j]
        if i > 0 and j > 0 and op in (0, 1):
            sub += op == 1
            i, j = i - 1, j - 1
        elif i > 0 and op == 2:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins


# =============================================================================================
# HDF readers
# =============================================================================================
def _hdf_sequences(paths):
    import h5py
    import numpy as np

    rows = {}
    for path in paths:
        with h5py.File(path.get_path(), "r") as fh:
            data = fh["inputs"][:]
            lengths = np.asarray(fh["seqLengths"])[:, 0].astype("int64")
            tags = [t.decode() if isinstance(t, bytes) else str(t) for t in fh["seqTags"][:]]
            offset = 0
            for tag, length in zip(tags, lengths):
                if tag in rows:
                    raise ValueError(f"duplicate tag {tag}")
                rows[tag] = data[offset:offset + length]
                offset += int(length)
            assert offset == len(data)
    return rows


def _hdf_lengths(paths):
    import h5py
    import numpy as np

    rows = {}
    for path in paths:
        with h5py.File(path.get_path(), "r") as fh:
            lengths = np.asarray(fh["seqLengths"])[:, 0].astype("int64")
            tags = [t.decode() if isinstance(t, bytes) else str(t) for t in fh["seqTags"][:]]
            for tag, length in zip(tags, lengths):
                if tag in rows:
                    raise ValueError(f"duplicate tag {tag}")
                rows[tag] = int(length)
    return rows


# =============================================================================================
# blank-free greedy PER
# =============================================================================================
class BlankfreeGreedyPerJob(Job):
    def __init__(self, *, posteriors: tk.Path, features, originals, gold: tk.Path, split: str):
        super().__init__()
        self.posteriors, self.features, self.originals = posteriors, list(features), list(originals)
        self.gold, self.split = gold, split
        self.out_per = self.output_path("per.json")
        self.out_report = self.output_path("per.txt")
        self.out_hyps = self.output_path("greedy_phones.json")
        self.out_raw_hyps = self.output_path("greedy_raw.json")
        self.out_stats = self.output_path("decode_stats.json")
        self.rqmt = {"cpu": 2, "mem": 16, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import numpy as np

        from ..phones import PHONES

        gold = json.load(open(self.gold.get_path()))[self.split]
        assert len(gold) == {"dev-clean": 2703, "dev-other": 2864}[self.split]
        post = _hdf_sequences([self.posteriors])
        features = _hdf_lengths(self.features)
        original = _hdf_sequences(self.originals)
        assert set(post) == set(gold) == set(original) == set(features)
        counts = Counter()
        lengths = Counter()
        strings = Counter()
        raw_hyps, hyps = {}, {}
        s = d = i = n = 0
        original_frames = retained_frames = output_frames = 0
        examples = []
        for tag in sorted(gold):
            q = post[tag]
            assert q.ndim == 2 and q.shape[1] == 40 and len(q) > 0
            ids = np.argmax(q, axis=1)
            ids = ids[np.r_[True, ids[1:] != ids[:-1]]]
            raw = [PHONES[int(k)] for k in ids]
            hyp = [phone for phone in raw if phone != "SIL"]
            raw_hyps[tag], hyps[tag] = raw, hyp
            lengths[len(hyp)] += 1
            strings[tuple(hyp)] += 1
            sub, delete, insert = edit_counts(hyp, gold[tag])
            s += sub; d += delete; i += insert; n += len(gold[tag])
            counts.update(raw)
            original_frames += int(np.asarray(original[tag]).reshape(-1)[0])
            output_frames += len(q)
            retained = features[tag]
            assert len(q) == (retained + 2) // 3
            retained_frames += retained
            if len(examples) < 20:
                examples.append({"tag": tag, "raw": raw, "hyp": hyp, "gold": gold[tag]})
        record = {"split": self.split, "per": (s + d + i) / n, "sub": s, "del": d,
                  "ins": i, "reference_phones": n, "utterances": len(gold),
                  "symbols": list(PHONES),
                  "output_frames": output_frames, "retained_frames": retained_frames,
                  "original_frames": original_frames,
                  "raw_emitted": sum(len(x) for x in raw_hyps.values()),
                  "phones_emitted": sum(len(x) for x in hyps.values()),
                  "phone_rate_original_hz": 50 * sum(len(x) for x in hyps.values()) / original_frames,
                  "phone_rate_retained_hz": 50 * sum(len(x) for x in hyps.values()) / retained_frames,
                  "raw_distinct_strings": len({tuple(x) for x in raw_hyps.values()}),
                  "distinct_strings": len({tuple(x) for x in hyps.values()}),
                  "phone_counts_raw": dict(counts),
                  "length_histogram": dict(sorted(lengths.items())),
                  "top_strings": [{"phones": list(x), "count": c} for x, c in strings.most_common(20)],
                  "examples": examples}
        with open(self.out_raw_hyps.get_path(), "w") as fh: json.dump(raw_hyps, fh)
        with open(self.out_hyps.get_path(), "w") as fh: json.dump(hyps, fh)
        with open(self.out_per.get_path(), "w") as fh: json.dump(record, fh, indent=2)
        with open(self.out_stats.get_path(), "w") as fh: json.dump(record, fh, indent=2)
        with open(self.out_report.get_path(), "w") as fh:
            fh.write(f"{self.split} PER={record['per']:.6f} S={s} D={d} I={i} N={n}\n")


# =============================================================================================
# per-checkpoint read chain
# =============================================================================================
def epoch_reads(
    train_job,
    epochs: Sequence[int],
    *,
    name: str,
    features,
    originals,
    split: str,
    gold: tk.Path,
    returnn_exe: Optional[tk.Path] = None,
    returnn_root: Optional[tk.Path] = None,
    net_args: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Theta slice -> posterior dump -> ``BlankfreeGreedyPerJob`` for each kept epoch of one arm.

    The source's ``epoch_reads`` wired ONE checkpoint and was called per epoch and split by the
    configs; this helper loops over ``epochs`` of a ``ReturnnTrainingJob`` (``out_checkpoints``) for
    one ``split`` and returns ``{epoch: {"theta", "post", "per"}}``.

    :param train_job: the arm's ``ReturnnTrainingJob``; ``train_job.out_checkpoints[epoch]`` is read.
    :param name: alias stem, e.g. ``"ctrl_20"``; aliases are ``.../{name}/ep{epoch}/{split}/...``.
    :param features: the split's retained-frame feature HDFs (``BlankfreeVadHdfJob`` feats).
    :param originals: the split's original-length HDFs (``BlankfreeVadHdfJob`` orig_length).
    :param gold: ``GoldPhonesJob`` json.
    :param returnn_exe, returnn_root: default ``default_tools.RETURNN_EXE`` / ``RETURNN_ROOT``.
    :param net_args: default the training config's ``NET_ARGS`` (the blank-free recognizer).
    """
    from ..training.checkpoints import theta_checkpoint
    from .posterior import posterior_dump

    if returnn_exe is None or returnn_root is None:
        from ..default_tools import RETURNN_EXE, RETURNN_ROOT

        returnn_exe = RETURNN_EXE if returnn_exe is None else returnn_exe
        returnn_root = RETURNN_ROOT if returnn_root is None else returnn_root
    if net_args is None:
        from ..training.config import NET_ARGS

        net_args = NET_ARGS

    out: Dict[int, Dict[str, Any]] = {}
    for epoch in epochs:
        read_name = f"{name}/ep{epoch}/{split}"
        theta, pt = theta_checkpoint(train_job, epoch, alias=f"sae/4a/{read_name}/theta")
        post = posterior_dump(name=read_name, checkpoint=pt,
                              feature_hdfs=features, returnn_exe=returnn_exe,
                              returnn_root=returnn_root, net_args=net_args)
        per = BlankfreeGreedyPerJob(posteriors=post.out_files["posteriors.hdf"], features=features,
                                    originals=originals, gold=gold, split=split)
        per.add_alias(f"sae/4a/blankfree/{read_name}/per")
        out[epoch] = {"theta": theta, "post": post, "per": per}
    return out
