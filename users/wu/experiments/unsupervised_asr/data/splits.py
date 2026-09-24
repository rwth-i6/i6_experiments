"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/init_jobs.py (``CvHoldoutSplitJob`` and the
``CV_HOLDOUT_*`` constants).

The disjoint train / cross-validation split of every phase-4a training: a fixed seed-0 1 % holdout
of the utterance id set, drawn from the SORTED ids, so it is a pure function of ``(id set, seed,
fraction)``.  On the 28,539 train-clean-100 ids it gives 28,254 train / 285 cv
(``CvHoldoutSplitJob.PfpCPQRCfIAk``); on the 2,849 seed ids 2,821 / 28 (``sD7U6CYs8ACM``).

The banked main-line split read its ids off the GAN pseudo-label json (``GanPseudoLabelJob``); only
the key SET of that json enters, so the port feeds the train-clean-100 id list instead
(:func:`.librispeech.get_split_ids`) and the output is identical (checked on the banked input).
"""

from __future__ import annotations

from typing import Optional

from sisyphus import Job, Task, tk

__all__ = ["CV_HOLDOUT_FRACTION", "CV_HOLDOUT_SEED", "CvHoldoutSplitJob"]

# Cross-validation holdout.  Orchestrator brief 2026-09-15: a fixed seed-0 1 % of each init's own
# utterances, held OUT of training and scored with the same pseudo-label CTC NLL.
# 0.01 * 28,539 = 285 utts (init i); 0.01 * 2,849 = 28 utts (init iii).
CV_HOLDOUT_FRACTION = 0.01
CV_HOLDOUT_SEED = 0


class CvHoldoutSplitJob(Job):
    """Split an utterance id set into a DISJOINT training / cross-validation pair.

    Orchestrator brief 2026-09-15: every init's CV set is a fixed seed-0 1 % holdout of that init's
    own utterances, removed from the training set and scored with the same pseudo-label CTC NLL.

    The two outputs are RETURNN segment lists (one seq tag per line) for
    ``seq_list_filter_file``.  They are attached to the FEATURE ``HDFDataset`` of each dataset; the
    targets HDF keeps every utterance and the ``MetaDataset`` looks the held-out tags up by tag
    (``returnn/datasets/meta.py:398-464``), so no target HDF has to be rebuilt to hold a set out.

    Determinism: the id list is SORTED before sampling, so the split is a pure function of
    ``(id set, seed, fraction)`` and does not depend on the json's key order or on a hash seed.

    :param ids_source: a json ``{uid: labels}`` map, a ``{label_key: {uid: labels}}`` wrapper, or a
        plain list of ids.
    :param label_key: the wrapper key to unwrap, or ``None`` for an already-unwrapped map/list.
    :param fraction: the held-out fraction (``CV_HOLDOUT_FRACTION``).
    :param seed: the sampling seed (``CV_HOLDOUT_SEED``).
    """

    def __init__(self, *, ids_source: tk.Path, label_key: Optional[str] = "labels",
                 fraction: float = CV_HOLDOUT_FRACTION, seed: int = CV_HOLDOUT_SEED):
        super().__init__()
        self.ids_source = ids_source
        self.label_key = label_key
        self.fraction = fraction
        self.seed = seed
        self.out_train_segments = self.output_path("train.segments")
        self.out_cv_segments = self.output_path("cv.segments")
        self.out_stats = self.output_path("split.stats.txt")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import json
        import random

        raw = json.load(open(self.ids_source.get_path()))
        if self.label_key is not None and isinstance(raw, dict) and self.label_key in raw:
            raw = raw[self.label_key]
        ids = sorted(raw) if isinstance(raw, dict) else sorted(str(x) for x in raw)
        assert ids, f"no utterance ids in {self.ids_source.get_path()}"
        assert len(set(ids)) == len(ids), "duplicate utterance ids"

        n_cv = int(round(self.fraction * len(ids)))
        assert 1 <= n_cv < len(ids), f"fraction {self.fraction} of {len(ids)} ids gives {n_cv}"
        cv = sorted(random.Random(self.seed).sample(ids, n_cv))
        cv_set = set(cv)
        train = [i for i in ids if i not in cv_set]
        assert not (set(train) & cv_set), "train and cv overlap"
        assert len(train) + len(cv) == len(ids)

        for path, part in ((self.out_train_segments, train), (self.out_cv_segments, cv)):
            with open(path.get_path(), "w") as fh:
                fh.write("\n".join(part) + "\n")

        lines = [
            f"CV holdout <- {self.ids_source.get_path()}",
            f"seed = {self.seed}   fraction = {self.fraction}",
            f"total = {len(ids)}   train = {len(train)}   cv = {len(cv)}   overlap = 0",
            f"first cv tags = {cv[:5]}",
            "the cv set is DISJOINT from training and is scored, never selected on",
        ]
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
