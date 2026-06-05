from __future__ import annotations

from sisyphus import Job, Task, tk


class ApplyPcaToFeatureHDFJob(Job):
    """
    Stream a dense RETURNN feature HDF through a stored PCA projection.

    The PCA state is expected to contain `mean` with shape [input_dim] and
    `components` with shape [pca_dim, input_dim].
    """

    def __init__(
        self,
        feature_hdf: tk.Path,
        pca_state: tk.Path,
        *,
        output_filename: str = "features_pca.hdf",
        chunk_size: int = 100_000,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.feature_hdf = feature_hdf
        self.pca_state = pca_state
        self.chunk_size = chunk_size
        self.out_hdf = self.output_path(output_filename)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 24, "time": 24})

    def run(self):
        import h5py
        import numpy as np
        import torch

        pca_state = torch.load(self.pca_state.get_path(), map_location="cpu")
        mean = pca_state["mean"].detach().cpu().numpy().astype("float32")
        components = pca_state["components"].detach().cpu().numpy().astype("float32")
        pca_dim = int(components.shape[0])

        with h5py.File(self.feature_hdf.get_path(), "r") as in_hdf, h5py.File(self.out_hdf.get_path(), "w") as out_hdf:
            inputs = in_hdf["inputs"]
            num_frames = int(inputs.shape[0])
            for key, value in in_hdf.attrs.items():
                out_hdf.attrs[key] = value
            out_hdf.attrs["inputPattSize"] = pca_dim
            out_hdf.attrs["numLabels"] = pca_dim

            out_inputs = out_hdf.create_dataset(
                "inputs",
                shape=(num_frames, pca_dim),
                dtype="float32",
                chunks=(min(self.chunk_size, max(1, num_frames)), pca_dim),
            )
            for start in range(0, num_frames, self.chunk_size):
                end = min(start + self.chunk_size, num_frames)
                chunk = inputs[start:end].astype("float32")
                out_inputs[start:end] = (chunk - mean) @ components.T

            seq_lengths = in_hdf["seqLengths"][:].astype("int32")
            out_hdf.create_dataset("seqLengths", data=seq_lengths)
            out_hdf.create_dataset("seqTags", data=in_hdf["seqTags"][:], dtype=in_hdf["seqTags"].dtype)
            if "labels" in in_hdf:
                out_hdf.create_dataset("labels", data=in_hdf["labels"][:], dtype=in_hdf["labels"].dtype)
            else:
                out_hdf.create_dataset("labels", data=np.asarray([], dtype="S5"))
