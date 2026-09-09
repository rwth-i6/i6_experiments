__all__ = ["EncoderFeatureHdfCallback"]

import os
from typing import Optional

import numpy as np

from returnn.datasets.hdf import SimpleHDFWriter
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict


class EncoderFeatureHdfCallback(ForwardCallbackIface):
    """
    Write the encoder states marked by ``dump.encoder_features.forward_step`` into an HDF file.

    The layout is exactly the one ``DumpNumpyFeaturesToHdfJobV2`` produces for the wav2vec
    features (``SimpleHDFWriter(dim=F, ndim=2)`` + a ``seq_sizes`` extra), so the resulting file
    drops straight into the existing ``HdfDataset(files=[...])`` + ``FeatureDatastream`` path with
    no reader-side change.

    :param out_file: file name written in the job's cwd (must be listed in the forward job's
        ``output_files``).
    :param dtype: on-disk dtype. ``float16`` halves the ~85 GB a full train-960 dump costs in
        float32; RETURNN reads the dtype back from the file and hands the train step whatever is
        stored (``raw_dict_to_extern_data`` overwrites the template dtype), so consumers must cast.
    :param summary_file: if given, a small human-readable summary written next to the HDF (also
        needs to be in ``output_files``).
    """

    def __init__(
        self,
        *,
        out_file: str = "features.hdf",
        dtype: str = "float16",
        summary_file: Optional[str] = "summary.txt",
        **kwargs,
    ):
        self.out_file = out_file
        self.dtype = dtype
        self.summary_file = summary_file
        self.hdf_writer: Optional[SimpleHDFWriter] = None
        self.num_seqs = 0
        self.num_frames = 0
        self.feature_dim: Optional[int] = None
        self.min_value = float("inf")
        self.max_value = float("-inf")
        self.num_non_finite = 0

    def init(self, *args, **kwargs):
        self.hdf_writer = None
        self.num_seqs = 0
        self.num_frames = 0
        self.feature_dim = None
        self.min_value = float("inf")
        self.max_value = float("-inf")
        self.num_non_finite = 0

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, **kwargs):
        features = np.asarray(outputs["features"].raw_tensor)  # [T, F]
        assert features.ndim == 2, f"{seq_tag}: expected [T, F], got {features.shape}"
        if features.shape[0] == 0:
            return

        if self.hdf_writer is None:
            self.feature_dim = int(features.shape[1])
            self.hdf_writer = SimpleHDFWriter(filename=self.out_file, dim=self.feature_dim, ndim=2)
        assert features.shape[1] == self.feature_dim, (
            f"{seq_tag}: feature dim changed: {features.shape[1]} vs {self.feature_dim}"
        )

        # track the value range in full precision before the (possibly lossy) cast, so the summary
        # says whether float16's ~65504 max / ~6e-5 subnormal range was a problem
        finite = np.isfinite(features)
        self.num_non_finite += int((~finite).sum())
        if finite.any():
            self.min_value = min(self.min_value, float(features[finite].min()))
            self.max_value = max(self.max_value, float(features[finite].max()))

        seq_data = features.astype(self.dtype, copy=False)[None, ...]  # [1, T, F]
        seq_len = seq_data.shape[1]
        seq_lens = {0: np.array([seq_len])}
        self.hdf_writer.insert_batch(
            seq_data,
            seq_len=seq_lens,
            seq_tag=[seq_tag],
            extra={"seq_sizes": seq_lens[0][:, None]},
        )
        self.num_seqs += 1
        self.num_frames += seq_len

    def finish(self, **kwargs):
        if self.hdf_writer is not None:
            self.hdf_writer.close()
        if self.summary_file is None:
            return
        with open(self.summary_file, "w") as f:
            f.write(f"num_seqs: {self.num_seqs}\n")
            f.write(f"num_frames: {self.num_frames}\n")
            f.write(f"feature_dim: {self.feature_dim}\n")
            f.write(f"dtype: {self.dtype}\n")
            if self.num_seqs:
                f.write(f"avg_frames_per_seq: {self.num_frames / self.num_seqs:.2f}\n")
                f.write(f"min_value: {self.min_value:.6g}\n")
                f.write(f"max_value: {self.max_value:.6g}\n")
            f.write(f"num_non_finite_values: {self.num_non_finite}\n")
        assert self.num_non_finite == 0, (
            f"{self.num_non_finite} non-finite encoder states -- see {os.path.abspath(self.summary_file)}"
        )
