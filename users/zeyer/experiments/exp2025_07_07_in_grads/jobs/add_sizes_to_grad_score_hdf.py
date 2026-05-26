"""Augment an :class:`ExtractInGradsFromModelJob` HDF with a ``sizes`` key so
that :class:`exp2025_05_05_align.CalcAlignmentMetricsJob` (which expects the
original single-chunk schema) can read it.
"""

from sisyphus import Job, Task, tk


class AddSizesToGradScoreHdfJob(Job):
    """Add a ``targets/data/sizes`` stream to a grad-score HDF.

    ``sizes[seq_idx] = [num_words_total, num_input_frames_total]`` summed across
    chunks. For short-form (one chunk per seq) this matches the original schema.
    Assumes one chunk per sequence; raises if multi-chunk.
    """

    def __init__(self, *, grad_score_hdf: tk.Path):
        super().__init__()
        self.grad_score_hdf = grad_score_hdf
        self.out_hdf = self.output_path("out.hdf")
        self.rqmt = {"cpu": 1, "mem": 4, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import shutil
        import numpy as np
        import h5py

        in_path = self.grad_score_hdf.get_path()
        out_path = self.out_hdf.get_path()
        shutil.copyfile(in_path, out_path)

        with h5py.File(out_path, "r+") as f:
            num_in = f["targets/data/num_input_frames"][:]  # [n_seqs, 1] for short-form (1 chunk)
            num_words = f["targets/data/num_words"][:]  # [n_seqs, 1]
            n_seqs = f["seqTags"].shape[0]
            assert num_in.shape == num_words.shape == (n_seqs, 1), (
                f"AddSizesToGradScoreHdfJob: assumes one chunk per seq; got "
                f"num_in.shape={num_in.shape}, n_seqs={n_seqs}"
            )
            sizes = np.concatenate([num_words, num_in], axis=-1).astype(np.int32)  # [n_seqs, 2]
            f.create_dataset("targets/data/sizes", data=sizes)
            f.create_dataset("targets/labels/sizes", data=np.array([b"classes"]))
            # Register the new stream's (dim, ndim) in the targets/size attrs group.
            # Mirrors what SimpleHDFWriter would have written for extra_type={"sizes": (2, 2, "int32")}.
            f["targets/size"].attrs["sizes"] = np.array([2, 2], dtype=np.int32)
            # Extend seqLengths by one column (one sizes row per seq).
            old_sl = f["seqLengths"][:]
            new_col = np.ones((n_seqs, 1), dtype=old_sl.dtype)
            new_sl = np.concatenate([old_sl, new_col], axis=1)
            del f["seqLengths"]
            f.create_dataset("seqLengths", data=new_sl)
