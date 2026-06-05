from __future__ import annotations

from sisyphus import Job, Task, tk


def _get_path(path):
    return path.get_path() if hasattr(path, "get_path") else str(path)


def _read_tag(tag):
    if isinstance(tag, (bytes, bytearray)):
        return tag.decode("utf8")
    return str(tag)


def _maybe_cache_path(path, *, use_cache_manager: bool):
    import subprocess

    path = _get_path(path)
    if not use_cache_manager:
        return path
    return subprocess.check_output(["cf", path], text=True).strip()


class MakeSparseClusterHDFDatasetCompatibleJob(Job):
    """
    Copy sparse input-only cluster HDFs and add the dummy seqLengths target column
    expected by RETURNN's legacy HDFDataset when a labels dataset is present.
    """

    def __init__(self, hdf_files, *, output_suffix: str = ".returnn_compat.hdf"):
        import os

        self.hdf_files = hdf_files if isinstance(hdf_files, (list, tuple)) else [hdf_files]
        self.output_suffix = output_suffix
        self.out_hdf_files = [
            self.output_path(os.path.basename(_get_path(hdf_file)).replace(".hdf", output_suffix))
            for hdf_file in self.hdf_files
        ]

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import h5py
        import numpy as np
        import shutil

        for src, dst in zip(self.hdf_files, self.out_hdf_files):
            shutil.copyfile(_get_path(src), dst.get_path())
            with h5py.File(dst.get_path(), "r+") as hdf:
                seq_lengths = hdf["seqLengths"][...]
                if seq_lengths.ndim != 2:
                    raise ValueError(f"{_get_path(src)}: expected 2-D seqLengths, got {seq_lengths.shape}")
                if seq_lengths.shape[1] == 1:
                    fixed = np.concatenate([seq_lengths, seq_lengths], axis=1)
                    del hdf["seqLengths"]
                    hdf.create_dataset("seqLengths", data=fixed, dtype=seq_lengths.dtype)
                elif seq_lengths.shape[1] != 2:
                    raise ValueError(f"{_get_path(src)}: expected 1 or 2 seqLengths columns, got {seq_lengths.shape}")


class SampleSegmentFeaturesJob(Job):
    """
    Uniformly sample segment vectors from a list of dense segment-representation HDFs.

    The output is a float32 .npy matrix with shape [num_samples, feature_dim].
    """

    def __init__(
        self,
        segment_hdfs,
        *,
        num_samples: int = 4_000_000,
        random_seed: int = 1,
        chunk_size: int = 50_000,
        use_cache_manager: bool = True,
        output_filename: str = "segment_samples.npy",
        mem_rqmt: int = 24,
        time_rqmt: int = 24,
    ):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.segment_hdfs = segment_hdfs if isinstance(segment_hdfs, (list, tuple)) else [segment_hdfs]
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.chunk_size = chunk_size
        self.use_cache_manager = use_cache_manager
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_samples = self.output_path(output_filename)
        self.out_stats = self.output_path("sample_stats.txt")

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

        cached_paths = [_maybe_cache_path(path, use_cache_manager=self.use_cache_manager) for path in self.segment_hdfs]
        counts, feature_dim = self._read_counts(cached_paths)
        total_segments = int(sum(counts))
        if self.num_samples > total_segments:
            raise ValueError(f"Requested {self.num_samples} samples but only {total_segments} segments are available.")

        rng = np.random.default_rng(self.random_seed)
        global_indices = np.sort(rng.choice(total_segments, size=self.num_samples, replace=False))
        out = np.lib.format.open_memmap(
            self.out_samples.get_path(),
            mode="w+",
            dtype="float32",
            shape=(self.num_samples, feature_dim),
        )

        global_file_offsets = np.concatenate([[0], np.cumsum(counts[:-1])])
        sample_write_offset = 0
        for path, file_offset, file_count in zip(cached_paths, global_file_offsets, counts):
            selected_begin = int(np.searchsorted(global_indices, file_offset, side="left"))
            selected_end = int(np.searchsorted(global_indices, file_offset + file_count, side="left"))
            if selected_begin == selected_end:
                continue
            local_indices = global_indices[selected_begin:selected_end] - file_offset
            with h5py.File(path, "r") as hdf:
                inputs = hdf["inputs"]
                local_pos = 0
                while local_pos < len(local_indices):
                    chunk_start = int(local_indices[local_pos] // self.chunk_size * self.chunk_size)
                    chunk_end = min(chunk_start + self.chunk_size, file_count)
                    next_pos = int(np.searchsorted(local_indices, chunk_end, side="left"))
                    chunk = inputs[chunk_start:chunk_end].astype("float32")
                    picked = chunk[local_indices[local_pos:next_pos] - chunk_start]
                    out[sample_write_offset : sample_write_offset + len(picked)] = picked
                    sample_write_offset += len(picked)
                    local_pos = next_pos

        out.flush()
        with open(self.out_stats.get_path(), "w") as f:
            f.write(f"num_input_hdfs: {len(cached_paths)}\n")
            f.write(f"total_segments: {total_segments}\n")
            f.write(f"num_samples: {self.num_samples}\n")
            f.write(f"feature_dim: {feature_dim}\n")
            f.write(f"random_seed: {self.random_seed}\n")
            f.write(f"use_cache_manager: {self.use_cache_manager}\n")
            f.write(f"written_samples: {sample_write_offset}\n")


class TrainFaissKMeansJob(Job):
    """Train FAISS k-means centroids from sampled segment vectors."""

    def __init__(
        self,
        samples_npy: tk.Path,
        *,
        num_clusters: int = 60,
        niter: int = 25,
        nredo: int = 1,
        random_seed: int = 1,
        gpu: bool = True,
        gpu_mem: int = 24,
        output_filename: str = "centers.npy",
        mem_rqmt: int = 32,
        time_rqmt: int = 24,
    ):
        if num_clusters <= 0:
            raise ValueError(f"num_clusters must be positive, got {num_clusters}")
        self.samples_npy = samples_npy
        self.num_clusters = num_clusters
        self.niter = niter
        self.nredo = nredo
        self.random_seed = random_seed
        self.gpu = gpu
        self.gpu_mem = gpu_mem
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_centers = self.output_path(output_filename)
        self.out_stats = self.output_path("kmeans_stats.txt")

    def tasks(self):
        yield Task(
            "run",
            rqmt={
                "cpu": 4,
                "mem": self.mem_rqmt,
                "time": self.time_rqmt,
                "gpu": 1 if self.gpu else 0,
                **({"gpu_mem": self.gpu_mem} if self.gpu else {}),
            },
        )

    def run(self):
        import numpy as np
        import faiss

        samples = np.load(self.samples_npy.get_path()).astype("float32", copy=False)
        if samples.ndim != 2:
            raise ValueError(f"Expected 2D sample matrix, got {samples.shape}")
        dim = int(samples.shape[1])
        kmeans = faiss.Kmeans(
            dim,
            self.num_clusters,
            niter=self.niter,
            nredo=self.nredo,
            verbose=True,
            gpu=self.gpu,
            seed=self.random_seed,
        )
        kmeans.train(samples)
        centers = np.asarray(kmeans.centroids, dtype="float32")
        np.save(self.out_centers.get_path(), centers)

        with open(self.out_stats.get_path(), "w") as f:
            f.write(f"samples: {self.samples_npy.get_path()}\n")
            f.write(f"num_samples: {samples.shape[0]}\n")
            f.write(f"feature_dim: {dim}\n")
            f.write(f"num_clusters: {self.num_clusters}\n")
            f.write(f"niter: {self.niter}\n")
            f.write(f"nredo: {self.nredo}\n")
            f.write(f"random_seed: {self.random_seed}\n")
            f.write(f"gpu: {self.gpu}\n")
            f.write(f"objective: {kmeans.obj[-1] if len(kmeans.obj) else 'nan'}\n")


class AssignFaissSegmentClustersJob(Job):
    """Assign every segment vector in one HDF to the nearest FAISS centroid."""

    def __init__(
        self,
        segment_hdf: tk.Path,
        centers_npy: tk.Path,
        *,
        chunk_size: int = 100_000,
        use_cache_manager: bool = True,
        gpu: bool = True,
        gpu_mem: int = 24,
        output_filename: str = "cluster_labels.hdf",
        mem_rqmt: int = 24,
        time_rqmt: int = 24,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.segment_hdf = segment_hdf
        self.centers_npy = centers_npy
        self.chunk_size = chunk_size
        self.use_cache_manager = use_cache_manager
        self.gpu = gpu
        self.gpu_mem = gpu_mem
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_hdf = self.output_path(output_filename)

    def tasks(self):
        yield Task(
            "run",
            rqmt={
                "cpu": 2,
                "mem": self.mem_rqmt,
                "time": self.time_rqmt,
                "gpu": 1 if self.gpu else 0,
                **({"gpu_mem": self.gpu_mem} if self.gpu else {}),
            },
        )

    @staticmethod
    def _make_index(centers, *, gpu: bool):
        import faiss

        index = faiss.IndexFlatL2(int(centers.shape[1]))
        index.add(centers.astype("float32", copy=False))
        if gpu:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index)
        return index

    def run(self):
        import h5py
        import numpy as np

        segment_hdf_path = _maybe_cache_path(self.segment_hdf, use_cache_manager=self.use_cache_manager)
        centers = np.load(self.centers_npy.get_path()).astype("float32", copy=False)
        index = self._make_index(centers, gpu=self.gpu)
        num_clusters = int(centers.shape[0])

        with h5py.File(segment_hdf_path, "r") as in_hdf, h5py.File(self.out_hdf.get_path(), "w") as out_hdf:
            inputs = in_hdf["inputs"]
            if inputs.ndim != 2:
                raise ValueError(f"Expected dense inputs with ndim=2, got {inputs.shape}")
            num_segments = int(inputs.shape[0])
            out_inputs = out_hdf.create_dataset("inputs", shape=(num_segments,), dtype="int32")
            for start in range(0, num_segments, self.chunk_size):
                end = min(start + self.chunk_size, num_segments)
                chunk = inputs[start:end].astype("float32")
                _distances, labels = index.search(chunk, 1)
                out_inputs[start:end] = labels[:, 0].astype("int32")

            out_hdf.attrs["numTimesteps"] = num_segments
            out_hdf.attrs["inputPattSize"] = num_clusters
            out_hdf.attrs["numDims"] = 1
            out_hdf.attrs["numLabels"] = num_clusters
            out_hdf.attrs["numSeqs"] = int(in_hdf["seqLengths"].shape[0])
            out_hdf.create_dataset("seqLengths", data=in_hdf["seqLengths"][:, :1].astype("int32"))
            out_hdf.create_dataset("seqTags", data=in_hdf["seqTags"][:], dtype=in_hdf["seqTags"].dtype)
            labels = np.asarray([str(i).encode("utf8") for i in range(num_clusters)], dtype=object)
            out_hdf.create_dataset("labels", data=labels, dtype=h5py.special_dtype(vlen=bytes))


class MergeConsecutiveClusterIdsJob(Job):
    """
    Collapse consecutive duplicate cluster ids per sequence in one cluster-label HDF.
    """

    def __init__(
        self,
        cluster_hdfs,
        *,
        output_filename: str = "cluster_labels_merged.hdf",
        report_filename: str = "cluster_labels_merged_report.txt",
        mem_rqmt: int = 8,
        time_rqmt: int = 4,
    ):
        self.cluster_hdfs = cluster_hdfs if isinstance(cluster_hdfs, (list, tuple)) else [cluster_hdfs]
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_hdf = self.output_path(output_filename)
        self.out_report = self.output_path(report_filename)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

    def run(self):
        import h5py
        import numpy as np

        before_total = 0
        after_total = 0
        num_seqs = 0
        num_changed_seqs = 0
        merged_sequences = []
        merged_lengths = []

        all_seq_tags = []
        labels = None
        labels_dtype = None
        attrs = {}
        for cluster_hdf in self.cluster_hdfs:
            with h5py.File(_get_path(cluster_hdf), "r") as in_hdf:
                inputs = in_hdf["inputs"]
                if inputs.ndim != 1:
                    raise ValueError(f"Expected sparse cluster-id inputs with ndim=1, got {inputs.shape}")
                seq_lengths = in_hdf["seqLengths"][:, 0].astype("int64")
                seq_starts = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])
                seq_tags = in_hdf["seqTags"][:]
                all_seq_tags.append(seq_tags)

                if labels is None and "labels" in in_hdf:
                    labels = in_hdf["labels"][:]
                    labels_dtype = h5py.special_dtype(vlen=bytes)
                elif labels is not None and "labels" in in_hdf:
                    other_labels = in_hdf["labels"][:]
                    if len(other_labels) != len(labels) or any(a != b for a, b in zip(other_labels, labels)):
                        raise ValueError(f"Label mismatch in {_get_path(cluster_hdf)}")
                if not attrs:
                    attrs = {key: value for key, value in in_hdf.attrs.items()}

                for start, length in zip(seq_starts, seq_lengths):
                    seq = inputs[int(start) : int(start + length)].astype("int32")
                    before_total += int(length)
                    num_seqs += 1
                    if len(seq) == 0:
                        merged = seq
                    else:
                        keep = np.ones((len(seq),), dtype=bool)
                        keep[1:] = seq[1:] != seq[:-1]
                        merged = seq[keep]
                    if len(merged) != len(seq):
                        num_changed_seqs += 1
                    after_total += int(len(merged))
                    merged_lengths.append(int(len(merged)))
                    merged_sequences.append(merged)

        if merged_sequences:
            merged_inputs = np.concatenate(merged_sequences).astype("int32", copy=False)
        else:
            merged_inputs = np.zeros((0,), dtype="int32")
        merged_seq_lengths = np.asarray(merged_lengths, dtype="int32")[:, None]
        # RETURNN's legacy HDFDataset assumes a dummy "classes" target
        # length column when a labels dataset exists.
        merged_seq_lengths_with_dummy_target = np.concatenate([merged_seq_lengths, merged_seq_lengths], axis=1)
        seq_tags = np.concatenate(all_seq_tags) if all_seq_tags else np.asarray([], dtype=object)

        with h5py.File(self.out_hdf.get_path(), "w") as out_hdf:
            out_hdf.create_dataset("inputs", data=merged_inputs)
            out_hdf.create_dataset("seqLengths", data=merged_seq_lengths_with_dummy_target)
            out_hdf.create_dataset("seqTags", data=seq_tags, dtype=h5py.special_dtype(vlen=bytes))
            for key, value in attrs.items():
                out_hdf.attrs[key] = value
            out_hdf.attrs["numTimesteps"] = int(after_total)
            out_hdf.attrs["numSeqs"] = int(num_seqs)
            if labels is not None:
                out_hdf.create_dataset("labels", data=labels, dtype=labels_dtype)

        merged_total = before_total - after_total
        with open(self.out_report.get_path(), "w") as f:
            f.write(f"num_input_hdfs: {len(self.cluster_hdfs)}\n")
            for i, cluster_hdf in enumerate(self.cluster_hdfs):
                f.write(f"input_hdf_{i}: {_get_path(cluster_hdf)}\n")
            f.write(f"output_hdf: {self.out_hdf.get_path()}\n")
            f.write(f"num_sequences: {num_seqs}\n")
            f.write(f"num_changed_sequences: {num_changed_seqs}\n")
            f.write(f"total_cluster_ids_before: {before_total}\n")
            f.write(f"total_cluster_ids_after: {after_total}\n")
            f.write(f"merged_cluster_ids: {merged_total}\n")
            f.write(f"merged_ratio: {merged_total / before_total if before_total else 0.0:.8f}\n")


class AggregateClusterIdMergeReportsJob(Job):
    """Aggregate multiple MergeConsecutiveClusterIdsJob reports."""

    def __init__(
        self,
        report_files,
        *,
        output_filename: str = "cluster_labels_merged_aggregate_report.txt",
        mem_rqmt: int = 1,
        time_rqmt: int = 1,
    ):
        self.report_files = report_files if isinstance(report_files, (list, tuple)) else [report_files]
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_report = self.output_path(output_filename)

    def tasks(self):
        yield Task("run", mini_task=True, rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

    @staticmethod
    def _read_report(path):
        values = {}
        with open(_get_path(path), "r") as f:
            for line in f:
                line = line.strip()
                if not line or ": " not in line:
                    continue
                key, value = line.split(": ", 1)
                values[key] = value
        return values

    def run(self):
        totals = {
            "num_sequences": 0,
            "num_changed_sequences": 0,
            "total_cluster_ids_before": 0,
            "total_cluster_ids_after": 0,
            "merged_cluster_ids": 0,
        }
        reports = []
        for report_file in self.report_files:
            report = self._read_report(report_file)
            reports.append((_get_path(report_file), report))
            for key in totals:
                totals[key] += int(report[key])

        before = totals["total_cluster_ids_before"]
        merged = totals["merged_cluster_ids"]
        with open(self.out_report.get_path(), "w") as f:
            f.write(f"num_partitions: {len(reports)}\n")
            for key, value in totals.items():
                f.write(f"{key}: {value}\n")
            f.write(f"merged_ratio: {merged / before if before else 0.0:.8f}\n")
            f.write("\npartition_reports:\n")
            for report_path, report in reports:
                f.write(
                    f"{report_path}\tbefore={report['total_cluster_ids_before']}"
                    f"\tafter={report['total_cluster_ids_after']}"
                    f"\tmerged={report['merged_cluster_ids']}\n"
                )


class ComputeClusterToGmmPerJob(Job):
    """
    Evaluate cluster-to-phoneme mapping from an OT transport plan against GMM alignments.

    The transport plan is interpreted as joint distribution P(phoneme, cluster).
    Every cluster id is mapped to the phoneme with maximum joint probability in
    its column. Both source and reference are collapsed after phoneme
    normalization. GMM eow markers (#) are stripped and [SILENCE] is normalized
    to [SIL].
    """

    def __init__(
        self,
        *,
        alignment_hdfs,
        cluster_hdfs,
        transport_npz: tk.Path,
        phoneme_vocab: tk.Path,
        cluster_vocab: tk.Path,
        gmm_idx_to_phoneme: tk.Path,
        utterance_select: tk.Path,
        output_filename: str = "cluster_to_gmm_per.txt",
        mem_rqmt: int = 24,
        time_rqmt: int = 5,
    ):
        self.alignment_hdfs = alignment_hdfs if isinstance(alignment_hdfs, (list, tuple)) else [alignment_hdfs]
        self.cluster_hdfs = cluster_hdfs if isinstance(cluster_hdfs, (list, tuple)) else [cluster_hdfs]
        self.transport_npz = transport_npz
        self.phoneme_vocab = phoneme_vocab
        self.cluster_vocab = cluster_vocab
        self.gmm_idx_to_phoneme = gmm_idx_to_phoneme
        self.utterance_select = utterance_select
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_report = self.output_path(output_filename)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

    @staticmethod
    def _read_vocab(path):
        import ast

        with open(_get_path(path), "r") as f:
            return {str(k): int(v) for k, v in ast.literal_eval(f.read()).items()}

    @staticmethod
    def _read_idx_to_phoneme(path):
        idx_to_phoneme = {}
        with open(_get_path(path), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx, phoneme = line.split(None, 1)
                idx_to_phoneme[int(idx)] = phoneme
        return idx_to_phoneme

    @staticmethod
    def _read_select(path):
        indices = []
        tags = []
        with open(_get_path(path), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.isdigit():
                    indices.append(int(line))
                else:
                    tags.append(line)
        return indices, tags

    @staticmethod
    def _normalize_phoneme(phoneme: str) -> str:
        phoneme = phoneme.rstrip("#")
        if phoneme == "[SILENCE]":
            return "[SIL]"
        return phoneme

    @classmethod
    def _collapse_tokens(cls, tokens):
        collapsed = []
        prev = None
        for token in tokens:
            token = cls._normalize_phoneme(str(token))
            if token != prev:
                collapsed.append(token)
                prev = token
        return collapsed

    @staticmethod
    def _remove_silence(tokens):
        return [token for token in tokens if token != "[SIL]"]

    @staticmethod
    def _edit_distance(source, target):
        if len(source) < len(target):
            prev = list(range(len(source) + 1))
            for j, tgt in enumerate(target, start=1):
                curr = [j]
                for i, src in enumerate(source, start=1):
                    curr.append(
                        min(
                            prev[i] + 1,
                            curr[i - 1] + 1,
                            prev[i - 1] + (0 if src == tgt else 1),
                        )
                    )
                prev = curr
            return prev[-1]

        prev = list(range(len(target) + 1))
        for i, src in enumerate(source, start=1):
            curr = [i]
            for j, tgt in enumerate(target, start=1):
                curr.append(
                    min(
                        prev[j] + 1,
                        curr[j - 1] + 1,
                        prev[j - 1] + (0 if src == tgt else 1),
                    )
                )
            prev = curr
        return prev[-1]

    @staticmethod
    def _read_hdf_sequences(paths, *, selected_tags=None):
        import h5py
        import numpy as np

        selected_tags = set(selected_tags) if selected_tags is not None else None
        seqs = {}
        global_idx_to_tag = []
        for path in paths:
            with h5py.File(_get_path(path), "r") as hdf:
                inputs = hdf["inputs"]
                seq_lengths = hdf["seqLengths"][:, 0]
                seq_tags = hdf["seqTags"][:]
                starts = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])
                for seq_tag, start, length in zip(seq_tags, starts, seq_lengths):
                    tag = _read_tag(seq_tag)
                    global_idx_to_tag.append(tag)
                    if selected_tags is not None and tag not in selected_tags:
                        continue
                    start = int(start)
                    length = int(length)
                    labels = inputs[start : start + length]
                    if labels.ndim > 1:
                        labels = labels[:, 0]
                    seqs[tag] = labels.astype("int64", copy=False)
        return seqs, global_idx_to_tag

    @staticmethod
    def _read_global_tags(paths):
        import h5py

        global_idx_to_tag = []
        for path in paths:
            with h5py.File(_get_path(path), "r") as hdf:
                global_idx_to_tag.extend(_read_tag(seq_tag) for seq_tag in hdf["seqTags"][:])
        return global_idx_to_tag

    def _cluster_to_phoneme_map(self):
        import numpy as np

        phoneme_vocab = self._read_vocab(self.phoneme_vocab)
        cluster_vocab = self._read_vocab(self.cluster_vocab)
        transport_data = np.load(_get_path(self.transport_npz), allow_pickle=True)
        transport = transport_data["transport"]
        phoneme_tokens = [str(t) for t in transport_data["phoneme_tokens"].tolist()]
        cluster_tokens = [str(t) for t in transport_data["cluster_tokens"].tolist()]

        if transport.shape != (len(phoneme_tokens), len(cluster_tokens)):
            raise ValueError(
                f"Transport shape {transport.shape} does not match tokens "
                f"{len(phoneme_tokens)}x{len(cluster_tokens)}"
            )
        missing_phonemes = [tok for tok in phoneme_tokens if tok not in phoneme_vocab]
        missing_clusters = [tok for tok in cluster_tokens if tok not in cluster_vocab]
        if missing_phonemes or missing_clusters:
            raise ValueError(f"Tokens missing from vocab: phonemes={missing_phonemes}, clusters={missing_clusters}")

        cluster_to_phoneme = {}
        best_phoneme_indices = transport.argmax(axis=0)
        for cluster_token, phoneme_idx in zip(cluster_tokens, best_phoneme_indices):
            cluster_to_phoneme[int(cluster_token)] = phoneme_tokens[int(phoneme_idx)]
        return cluster_to_phoneme

    def run(self):
        import numpy as np

        selected_indices, selected_tags = self._read_select(self.utterance_select)
        cluster_to_phoneme = self._cluster_to_phoneme_map()
        idx_to_phoneme = self._read_idx_to_phoneme(self.gmm_idx_to_phoneme)

        if selected_indices:
            global_idx_to_tag = self._read_global_tags(self.cluster_hdfs)
            try:
                selected_tags.extend(global_idx_to_tag[idx] for idx in selected_indices)
            except IndexError as exc:
                raise IndexError(
                    f"Utterance select index out of range for {len(global_idx_to_tag)} cluster sequences"
                ) from exc
        selected_tags = list(dict.fromkeys(selected_tags))

        cluster_seqs, _ = self._read_hdf_sequences(self.cluster_hdfs, selected_tags=selected_tags)
        alignment_seqs, _ = self._read_hdf_sequences(self.alignment_hdfs, selected_tags=selected_tags)

        metrics = {
            "with_silence": {"total_edits": 0, "total_source_len": 0, "total_target_len": 0, "num_eval": 0},
            "without_silence": {"total_edits": 0, "total_source_len": 0, "total_target_len": 0, "num_eval": 0},
        }
        missing_cluster = []
        missing_alignment = []
        examples = []

        for tag in selected_tags:
            if tag not in cluster_seqs:
                missing_cluster.append(tag)
                continue
            if tag not in alignment_seqs:
                missing_alignment.append(tag)
                continue

            target_labels = alignment_seqs[tag]
            target_tokens = self._collapse_tokens(idx_to_phoneme[int(label)] for label in target_labels)
            source_labels = cluster_seqs[tag]
            source_tokens = self._collapse_tokens(cluster_to_phoneme[int(label)] for label in source_labels)
            eval_pairs = {
                "with_silence": (source_tokens, target_tokens),
                "without_silence": (self._remove_silence(source_tokens), self._remove_silence(target_tokens)),
            }
            per_by_variant = {}
            edits_by_variant = {}
            target_len_by_variant = {}
            for variant, (variant_source, variant_target) in eval_pairs.items():
                edits = self._edit_distance(variant_source, variant_target)
                target_len = len(variant_target)
                if target_len == 0:
                    continue
                metrics[variant]["total_edits"] += edits
                metrics[variant]["total_source_len"] += len(variant_source)
                metrics[variant]["total_target_len"] += target_len
                metrics[variant]["num_eval"] += 1
                per = edits / target_len
                per_by_variant[variant] = per
                edits_by_variant[variant] = edits
                target_len_by_variant[variant] = target_len
            if len(examples) < 10:
                examples.append(
                    (
                        tag,
                        edits_by_variant,
                        target_len_by_variant,
                        per_by_variant,
                        source_tokens[:80],
                        target_tokens[:80],
                        self._remove_silence(source_tokens)[:80],
                        self._remove_silence(target_tokens)[:80],
                    )
                )

        if metrics["with_silence"]["total_target_len"] == 0:
            raise ValueError("No selected utterances with non-empty target sequence were evaluated.")

        with open(self.out_report.get_path(), "w") as f:
            f.write(f"transport_npz: {_get_path(self.transport_npz)}\n")
            f.write(f"phoneme_vocab: {_get_path(self.phoneme_vocab)}\n")
            f.write(f"cluster_vocab: {_get_path(self.cluster_vocab)}\n")
            f.write(f"gmm_idx_to_phoneme: {_get_path(self.gmm_idx_to_phoneme)}\n")
            f.write(f"utterance_select: {_get_path(self.utterance_select)}\n")
            f.write(f"num_selected_tags: {len(selected_tags)}\n")
            f.write(f"num_missing_cluster: {len(missing_cluster)}\n")
            f.write(f"num_missing_alignment: {len(missing_alignment)}\n")
            for variant, values in metrics.items():
                total_edits = values["total_edits"]
                total_source_len = values["total_source_len"]
                total_target_len = values["total_target_len"]
                f.write(f"\n[{variant}]\n")
                f.write(f"num_eval: {values['num_eval']}\n")
                f.write(f"total_edits: {total_edits}\n")
                f.write(f"total_source_phonemes: {total_source_len}\n")
                f.write(f"total_target_phonemes: {total_target_len}\n")
                if total_target_len > 0:
                    f.write(f"micro_per: {total_edits / total_target_len:.12g}\n")
                    f.write(f"source_target_phoneme_ratio: {total_source_len / total_target_len:.12g}\n")
                else:
                    f.write("micro_per: nan\n")
                    f.write("source_target_phoneme_ratio: nan\n")
            f.write("\ncluster_to_phoneme:\n")
            for cluster_id in sorted(cluster_to_phoneme):
                f.write(f"{cluster_id}\t{cluster_to_phoneme[cluster_id]}\n")
            f.write("\nexamples:\n")
            for (
                tag,
                edits_by_variant,
                target_len_by_variant,
                per_by_variant,
                source_tokens,
                target_tokens,
                source_no_sil,
                target_no_sil,
            ) in examples:
                f.write(f"tag: {tag}\n")
                for variant in ("with_silence", "without_silence"):
                    if variant in per_by_variant:
                        f.write(
                            f"{variant}: edits: {edits_by_variant[variant]} "
                            f"target_len: {target_len_by_variant[variant]} per: {per_by_variant[variant]:.12g}\n"
                        )
                f.write("source_with_silence: " + " ".join(source_tokens) + "\n")
                f.write("target_with_silence: " + " ".join(target_tokens) + "\n")
                f.write("source_without_silence: " + " ".join(source_no_sil) + "\n")
                f.write("target_without_silence: " + " ".join(target_no_sil) + "\n\n")


def create_faiss_segment_clustering_pipeline(
    *,
    output_prefix: str,
    segment_hdfs,
    num_clusters: int = 60,
    num_samples: int = 4_000_000,
    random_seed: int = 1,
    use_cache_manager: bool = True,
    train_gpu_mem: int = 24,
    assign_gpu_mem: int = 11,
):
    sample_job = SampleSegmentFeaturesJob(
        segment_hdfs,
        num_samples=num_samples,
        random_seed=random_seed,
        use_cache_manager=use_cache_manager,
        output_filename=f"segment_samples_{num_samples}.npy",
        mem_rqmt=24,
        time_rqmt=48,
    )
    sample_job.add_alias(output_prefix + f"/sample_{num_samples}")
    tk.register_output(output_prefix + f"/segment_samples_{num_samples}.npy", sample_job.out_samples)
    tk.register_output(output_prefix + f"/segment_samples_{num_samples}.txt", sample_job.out_stats)

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
    for idx, segment_hdf in enumerate(segment_hdfs):
        assign_job = AssignFaissSegmentClustersJob(
            segment_hdf=segment_hdf,
            centers_npy=train_job.out_centers,
            use_cache_manager=use_cache_manager,
            gpu=True,
            gpu_mem=assign_gpu_mem,
            output_filename=f"cluster_labels_k{num_clusters}.{idx:03d}.hdf",
            mem_rqmt=24,
            time_rqmt=24,
        )
        assign_job.add_alias(output_prefix + f"/assign_k{num_clusters}/part_{idx:03d}")
        tk.register_output(output_prefix + f"/cluster_labels_k{num_clusters}.{idx:03d}.hdf", assign_job.out_hdf)
        assignment_hdfs.append(assign_job.out_hdf)
    return {
        "samples": sample_job.out_samples,
        "centers": train_job.out_centers,
        "assignments": assignment_hdfs,
    }
