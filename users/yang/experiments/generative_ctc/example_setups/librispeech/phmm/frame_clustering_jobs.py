from __future__ import annotations

from sisyphus import Job, Task, tk

from .segmenter_jobs import _read_tag, _write_segment_starts_hdf
from .segment_clustering_jobs import AssignFaissSegmentClustersJob, TrainFaissKMeansJob, _get_path, _maybe_cache_path


class DetectClusterChangeStartFramesJob(Job):
    """
    Detect start frames from frame-level cluster labels.

    Frame 0 is always a start. Every later frame t is a start iff
    cluster_id[t] != cluster_id[t - 1]. The main output keeps the input HDF frame
    rate. Optionally, a second HDF with scaled start indices is written, e.g. for
    converting 20ms frame starts to 10ms frame starts via scale factor 2.
    """

    def __init__(
        self,
        cluster_hdfs,
        *,
        output_filename: str = "cluster_change_starts.hdf",
        output_scaled_filename: str | None = None,
        output_segment_cluster_filename: str | None = None,
        scaled_frame_factor: int = 2,
        mem_rqmt: int = 24,
        time_rqmt: int = 5,
    ):
        if scaled_frame_factor <= 0:
            raise ValueError(f"scaled_frame_factor must be positive, got {scaled_frame_factor}")
        self.cluster_hdfs = cluster_hdfs if isinstance(cluster_hdfs, (list, tuple)) else [cluster_hdfs]
        self.output_scaled_filename = output_scaled_filename
        self.output_segment_cluster_filename = output_segment_cluster_filename
        self.scaled_frame_factor = scaled_frame_factor
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_hdf = self.output_path(output_filename)
        self.out_scaled_hdf = self.output_path(output_scaled_filename) if output_scaled_filename is not None else None
        self.out_segment_cluster_hdf = (
            self.output_path(output_segment_cluster_filename)
            if output_segment_cluster_filename is not None
            else None
        )
        self.out_stats = self.output_path("cluster_change_start_stats.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

    @staticmethod
    def _cluster_change_starts(labels):
        import numpy as np

        labels = np.asarray(labels)
        if labels.ndim > 1:
            labels = labels[:, 0]
        if labels.shape[0] == 0:
            return np.asarray([0], dtype="int32")
        change_indices = np.nonzero(labels[1:] != labels[:-1])[0] + 1
        return np.concatenate([[0], change_indices]).astype("int32")

    def run(self):
        import h5py
        import numpy as np

        seq_tags = []
        start_seqs = []
        segment_cluster_seqs = []
        total_frames = 0
        total_starts = 0

        for hdf_path in self.cluster_hdfs:
            path = _get_path(hdf_path)
            with h5py.File(path, "r") as hdf:
                inputs = hdf["inputs"]
                seq_lengths = hdf["seqLengths"][:, 0]
                hdf_seq_tags = hdf["seqTags"][:]
                offsets = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])
                for seq_tag, offset, length in zip(hdf_seq_tags, offsets, seq_lengths):
                    offset = int(offset)
                    length = int(length)
                    labels = inputs[offset : offset + length]
                    if labels.ndim > 1:
                        labels = labels[:, 0]
                    starts = self._cluster_change_starts(labels)
                    seq_tags.append(_read_tag(seq_tag))
                    start_seqs.append(starts)
                    segment_cluster_seqs.append(labels[starts].astype("int32", copy=False))
                    total_frames += length
                    total_starts += int(starts.shape[0])

        _write_segment_starts_hdf(self.out_hdf.get_path(), seq_tags, start_seqs)

        if self.out_scaled_hdf is not None:
            scaled_start_seqs = [
                (starts.astype("int64") * int(self.scaled_frame_factor)).astype("int32") for starts in start_seqs
            ]
            _write_segment_starts_hdf(self.out_scaled_hdf.get_path(), seq_tags, scaled_start_seqs)

        if self.out_segment_cluster_hdf is not None:
            _write_segment_starts_hdf(self.out_segment_cluster_hdf.get_path(), seq_tags, segment_cluster_seqs)

        with open(self.out_stats.get_path(), "w") as f:
            f.write(f"num_input_hdfs: {len(self.cluster_hdfs)}\n")
            f.write(f"num_sequences: {len(seq_tags)}\n")
            f.write(f"total_frames: {total_frames}\n")
            f.write(f"total_starts: {total_starts}\n")
            f.write(f"starts_per_frame: {total_starts / total_frames if total_frames else 0.0:.10f}\n")
            f.write(f"wrote_scaled_hdf: {self.out_scaled_hdf is not None}\n")
            f.write(f"wrote_segment_cluster_hdf: {self.out_segment_cluster_hdf is not None}\n")
            f.write(f"scaled_frame_factor: {self.scaled_frame_factor}\n")


class SampleFrameFeaturesJob(Job):
    """
    Uniformly sample frame-level vectors from dense feature HDFs.

    This is intentionally separate from SampleSegmentFeaturesJob so frame-level
    clustering changes do not alter existing segment-clustering job hashes.
    """

    def __init__(
        self,
        feature_hdfs,
        *,
        num_samples: int = 4_000_000,
        random_seed: int = 1,
        chunk_size: int = 100_000,
        samples_per_chunk: int = 16_384,
        use_cache_manager: bool = True,
        output_filename: str = "frame_samples.npy",
        mem_rqmt: int = 24,
        time_rqmt: int = 48,
    ):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if samples_per_chunk <= 0:
            raise ValueError(f"samples_per_chunk must be positive, got {samples_per_chunk}")
        self.feature_hdfs = feature_hdfs if isinstance(feature_hdfs, (list, tuple)) else [feature_hdfs]
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.chunk_size = chunk_size
        self.samples_per_chunk = samples_per_chunk
        self.use_cache_manager = use_cache_manager
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_samples = self.output_path(output_filename)
        self.out_stats = self.output_path("frame_sample_stats.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

    @staticmethod
    def _read_counts(paths):
        import h5py

        counts = []
        feature_dim = None
        for path in paths:
            with h5py.File(path, "r") as hdf:
                inputs = hdf["inputs"]
                if inputs.ndim != 2:
                    raise ValueError(f"Expected dense inputs with ndim=2, got {inputs.shape} in {path}")
                counts.append(int(inputs.shape[0]))
                if feature_dim is None:
                    feature_dim = int(inputs.shape[1])
                elif feature_dim != int(inputs.shape[1]):
                    raise ValueError(f"Feature dim mismatch: expected {feature_dim}, got {inputs.shape[1]} in {path}")
        return counts, feature_dim

    def run(self):
        import h5py
        import numpy as np

        cached_paths = [_maybe_cache_path(path, use_cache_manager=self.use_cache_manager) for path in self.feature_hdfs]
        counts, feature_dim = self._read_counts(cached_paths)
        total_frames = int(sum(counts))
        if self.num_samples > total_frames:
            raise ValueError(f"Requested {self.num_samples} samples but only {total_frames} frames are available.")

        rng = np.random.default_rng(self.random_seed)
        out = np.lib.format.open_memmap(
            self.out_samples.get_path(),
            mode="w+",
            dtype="float32",
            shape=(self.num_samples, feature_dim),
        )

        chunks = []
        for file_idx, (path, file_count) in enumerate(zip(cached_paths, counts)):
            with h5py.File(path, "r") as hdf:
                inputs = hdf["inputs"]
                read_chunk_size = int(inputs.chunks[0]) if inputs.chunks is not None else self.chunk_size
            for chunk_start in range(0, file_count, read_chunk_size):
                chunk_end = min(chunk_start + read_chunk_size, file_count)
                chunks.append((file_idx, chunk_start, chunk_end))
        chunk_order = rng.permutation(len(chunks))
        sample_write_offset = 0

        open_hdfs = {}
        try:
            for num_chunks_read, chunk_idx in enumerate(chunk_order, start=1):
                file_idx, chunk_start, chunk_end = chunks[int(chunk_idx)]
                num_to_pick = min(self.samples_per_chunk, chunk_end - chunk_start, self.num_samples - sample_write_offset)
                if num_to_pick <= 0:
                    break
                if file_idx not in open_hdfs:
                    open_hdfs[file_idx] = h5py.File(cached_paths[file_idx], "r")
                inputs = open_hdfs[file_idx]["inputs"]
                chunk = inputs[chunk_start:chunk_end].astype("float32")
                local_indices = np.sort(rng.choice(chunk.shape[0], size=num_to_pick, replace=False)).astype("int64")
                picked = chunk[local_indices]
                out[sample_write_offset : sample_write_offset + num_to_pick] = picked
                sample_write_offset += num_to_pick
                if num_chunks_read % 25 == 0 or sample_write_offset == self.num_samples:
                    print(
                        f"sampled {sample_write_offset}/{self.num_samples} frames "
                        f"from {num_chunks_read}/{len(chunks)} selected chunks",
                        flush=True,
                    )
                if sample_write_offset == self.num_samples:
                    break
        finally:
            for hdf in open_hdfs.values():
                hdf.close()

        if sample_write_offset != self.num_samples:
            raise RuntimeError(
                f"Only wrote {sample_write_offset} samples out of {self.num_samples}. "
                f"Increase samples_per_chunk or allow sampling with replacement."
            )

        out.flush()
        with open(self.out_stats.get_path(), "w") as f:
            f.write(f"num_input_hdfs: {len(cached_paths)}\n")
            f.write(f"total_frames: {total_frames}\n")
            f.write(f"num_samples: {self.num_samples}\n")
            f.write(f"feature_dim: {feature_dim}\n")
            f.write(f"random_seed: {self.random_seed}\n")
            f.write(f"chunk_size: {self.chunk_size}\n")
            f.write(f"samples_per_chunk: {self.samples_per_chunk}\n")
            f.write(f"num_available_chunks: {len(chunks)}\n")
            f.write(f"use_cache_manager: {self.use_cache_manager}\n")
            f.write(f"written_samples: {sample_write_offset}\n")


def create_faiss_frame_clustering_pipeline(
    *,
    output_prefix: str,
    feature_hdfs,
    num_clusters: int = 128,
    num_samples: int = 4_000_000,
    random_seed: int = 1,
    use_cache_manager: bool = True,
    train_gpu_mem: int = 24,
    assign_gpu_mem: int = 24,
):
    feature_hdfs = feature_hdfs if isinstance(feature_hdfs, (list, tuple)) else [feature_hdfs]

    sample_job = SampleFrameFeaturesJob(
        feature_hdfs,
        num_samples=num_samples,
        random_seed=random_seed,
        use_cache_manager=use_cache_manager,
        output_filename=f"frame_samples_{num_samples}.npy",
        mem_rqmt=24,
        time_rqmt=48,
    )
    sample_job.add_alias(output_prefix + f"/sample_{num_samples}")
    tk.register_output(output_prefix + f"/frame_samples_{num_samples}.npy", sample_job.out_samples)
    tk.register_output(output_prefix + f"/frame_samples_{num_samples}.txt", sample_job.out_stats)

    train_job = TrainFaissKMeansJob(
        sample_job.out_samples,
        num_clusters=num_clusters,
        random_seed=random_seed,
        gpu=True,
        gpu_mem=train_gpu_mem,
        output_filename=f"centers_k{num_clusters}.npy",
        mem_rqmt=32,
        time_rqmt=24,
    )
    train_job.add_alias(output_prefix + f"/train_k{num_clusters}")
    tk.register_output(output_prefix + f"/centers_k{num_clusters}.npy", train_job.out_centers)
    tk.register_output(output_prefix + f"/centers_k{num_clusters}.txt", train_job.out_stats)

    assignment_hdfs = []
    for idx, feature_hdf in enumerate(feature_hdfs):
        assign_job = AssignFaissSegmentClustersJob(
            segment_hdf=feature_hdf,
            centers_npy=train_job.out_centers,
            use_cache_manager=use_cache_manager,
            gpu=True,
            gpu_mem=assign_gpu_mem,
            output_filename=f"frame_cluster_labels_k{num_clusters}.{idx:03d}.hdf",
            mem_rqmt=24,
            time_rqmt=48,
        )
        assign_job.add_alias(output_prefix + f"/assign_k{num_clusters}/part_{idx:03d}")
        tk.register_output(output_prefix + f"/frame_cluster_labels_k{num_clusters}.{idx:03d}.hdf", assign_job.out_hdf)
        assignment_hdfs.append(assign_job.out_hdf)
    return {
        "samples": sample_job.out_samples,
        "centers": train_job.out_centers,
        "assignments": assignment_hdfs,
    }
