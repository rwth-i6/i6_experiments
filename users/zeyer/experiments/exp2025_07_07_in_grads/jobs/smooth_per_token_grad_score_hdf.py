"""Apply temporal smoothing to a per-token grad-score HDF (from ExtractInGradsPerTokenJob)."""

from sisyphus import Job, Task, tk


class SmoothPerTokenGradScoreHdfJob(Job):
    """Apply temporal smoothing (Gaussian or median) to a per-token grad HDF
    along the input-frame axis. Preserves all other streams.

    Counterpart to :class:`SmoothGradScoreHdfJob` for the per-token HDF
    schema written by :class:`ExtractInGradsPerTokenJob` -- the only
    difference is the per-seq reshape uses ``num_tokens`` instead of
    ``num_words``. Single-chunk only.
    """

    def __init__(self, *, grad_score_hdf: tk.Path, smooth_kind: str, smooth_size: float):
        super().__init__()
        assert smooth_kind in ("gaussian", "median"), smooth_kind
        self.grad_score_hdf = grad_score_hdf
        self.smooth_kind = smooth_kind
        self.smooth_size = smooth_size
        self.out_hdf = self.output_path("out.hdf")
        self.rqmt = {"cpu": 1, "mem": 4, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import shutil
        import numpy as np
        import h5py
        from scipy.ndimage import gaussian_filter1d, median_filter

        in_path = self.grad_score_hdf.get_path()
        out_path = self.out_hdf.get_path()
        shutil.copyfile(in_path, out_path)

        with h5py.File(out_path, "r+") as f:
            num_in = f["targets/data/num_input_frames"][:]  # [n_seqs, 1]
            num_tokens = f["targets/data/num_tokens"][:]    # [n_seqs, 1]
            n_seqs = num_in.shape[0]
            assert num_in.shape == num_tokens.shape == (n_seqs, 1), (
                f"SmoothPerTokenGradScoreHdfJob: assumes one chunk per seq; got "
                f"num_in.shape={num_in.shape}, num_tokens.shape={num_tokens.shape}"
            )
            data = f["inputs"][:].squeeze(-1)
            offset = 0
            for seq_idx in range(n_seqs):
                T = int(num_in[seq_idx, 0])
                K = int(num_tokens[seq_idx, 0])
                n = K * T
                chunk = data[offset:offset + n].reshape(K, T).astype(np.float32, copy=False)
                if self.smooth_kind == "gaussian":
                    chunk = gaussian_filter1d(chunk, sigma=float(self.smooth_size), axis=1)
                else:
                    chunk = median_filter(chunk, size=(1, int(self.smooth_size)))
                data[offset:offset + n] = chunk.reshape(-1)
                offset += n
            assert offset == len(data), f"length mismatch: offset={offset} vs len={len(data)}"
            f["inputs"][:] = data[:, None]
