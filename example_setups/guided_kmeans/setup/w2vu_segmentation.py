"""
Sisyphus job for preparing the precomputed wav2vec-U segmented (wav2vecUseg)
features for training.

FilterHdfByTagsJob
    Filters RETURNN HDF shards to a subset of sequence tags, e.g. the ls-100h
    subset of the full ls-960h wav2vecUseg features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sisyphus import Job, Task, tk


class FilterHdfByTagsJob(Job):
    """Filter one or more RETURNN HDF files to a subset of sequence tags.

    Reads all input shards sequentially, keeps only sequences whose tag appears
    in the ``keep_tags`` file, and writes a single output HDF.  All standard
    RETURNN HDF datasets (``inputs``, ``seqLengths``, ``seqTags``) are copied;
    ``labels`` and ``targets`` are dropped since they are unused here.

    Typical use: extracting the ls-100h subset from schmitt's full ls-960h
    wav2vec-U mean-pooled HDF shards.

    :param feature_hdfs: list of input RETURNN HDF files (shards)
    :param keep_tags: text file with one sequence tag per line to retain;
        tags must match exactly what is stored in ``seqTags``
    """

    def __init__(
        self,
        feature_hdfs: Sequence[tk.Path],
        keep_tags: tk.Path,
        rqmt: Optional[Dict[str, Any]] = None,
    ):
        self.feature_hdfs = list(feature_hdfs)
        self.keep_tags = keep_tags

        self.out_features = self.output_path("features.hdf")

        self.rqmt = {"cpu": 2, "mem": 16, "time": 4}
        if rqmt:
            self.rqmt.update(rqmt)

    @classmethod
    def hash(cls, kwargs):
        return super().hash({k: v for k, v in kwargs.items() if k != "rqmt"})

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import h5py

        with open(self.keep_tags.get_path()) as f:
            filter_set = {l.strip() for l in f if l.strip()}
        print(
            f"[FilterHdfByTagsJob] keeping {len(filter_set)} tags",
            flush=True,
        )

        out_feats: List[np.ndarray] = []
        out_lengths: List[int] = []
        out_tags: List[str] = []
        skipped = 0

        for shard_path in self.feature_hdfs:
            fp = shard_path.get_path()
            with h5py.File(fp, "r") as f:
                raw_tags = [
                    t.decode() if isinstance(t, bytes) else t
                    for t in f["seqTags"][:]
                ]
                lengths = f["seqLengths"][:, 0]
                feature_data = f["inputs"]
                offset = 0
                for tag, length in zip(raw_tags, lengths):
                    chunk = np.array(
                        feature_data[offset : offset + length], dtype=np.float32
                    )
                    offset += length
                    if tag not in filter_set:
                        skipped += 1
                        continue
                    out_tags.append(tag)
                    out_lengths.append(int(length))
                    out_feats.append(chunk)
            print(f"[FilterHdfByTagsJob] processed {fp}", flush=True)

        print(
            f"[FilterHdfByTagsJob] kept {len(out_tags)}, skipped {skipped}",
            flush=True,
        )

        all_feats = np.concatenate(out_feats, axis=0)
        with h5py.File(self.out_features.get_path(), "w") as f_out:
            f_out.create_dataset("inputs", data=all_feats)
            lengths_arr = np.zeros((len(out_lengths), 2), dtype=np.int32)
            lengths_arr[:, 0] = out_lengths
            f_out.create_dataset("seqLengths", data=lengths_arr)
            dt = h5py.special_dtype(vlen=str)
            tags_ds = f_out.create_dataset(
                "seqTags", shape=(len(out_tags),), dtype=dt
            )
            for i, t in enumerate(out_tags):
                tags_ds[i] = t

        print(
            f"[FilterHdfByTagsJob] wrote {len(out_tags)} seqs "
            f"({all_feats.shape[0]:,} segments) "
            f"→ {self.out_features.get_path()}",
            flush=True,
        )
