"""Apply temporal smoothing to a grad-score HDF along the input-frame axis."""

from sisyphus import Job, Task, tk


class SmoothGradScoreHdfJob(Job):
    """Apply temporal smoothing (Gaussian or median filter) to a grad-score HDF
    along the input-frame axis. Preserves all other streams.

    Assumes one chunk per sequence (short-form). For multi-chunk extracts, the
    per-word reshape below would need to iterate chunks; not implemented yet.
    """

    def __init__(self, *, grad_score_hdf: tk.Path, smooth_kind: str, smooth_size: float):
        super().__init__()
        assert smooth_kind in ("gaussian", "median"), smooth_kind
        self.grad_score_hdf = grad_score_hdf
        self.smooth_kind = smooth_kind
        self.smooth_size = smooth_size  # sigma for gaussian, kernel size for median
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
            num_words = f["targets/data/num_words"][:]
            n_seqs = num_in.shape[0]
            assert num_in.shape == num_words.shape == (n_seqs, 1), (
                f"SmoothGradScoreHdfJob: assumes one chunk per seq; got "
                f"num_in.shape={num_in.shape}, n_seqs={n_seqs}"
            )
            data = f["inputs"][:].squeeze(-1)  # [total_frames]
            offset = 0
            for seq_idx in range(n_seqs):
                T = int(num_in[seq_idx, 0])
                W = int(num_words[seq_idx, 0])
                n = W * T
                chunk = data[offset:offset + n].reshape(W, T).astype(np.float32, copy=False)
                if self.smooth_kind == "gaussian":
                    chunk = gaussian_filter1d(chunk, sigma=float(self.smooth_size), axis=1)
                else:
                    chunk = median_filter(chunk, size=(1, int(self.smooth_size)))
                data[offset:offset + n] = chunk.reshape(-1)
                offset += n
            assert offset == len(data), f"length mismatch: offset={offset} vs len={len(data)}"
            f["inputs"][:] = data[:, None]
