from __future__ import annotations

from sisyphus import Job, Task, tk


def _get_path(path):
    return path.get_path() if hasattr(path, "get_path") else str(path)


def _read_tag(tag):
    if isinstance(tag, (bytes, bytearray)):
        return tag.decode("utf8")
    return str(tag)


class ComputeClusterPurityPnmiJob(Job):
    """
    Compute cluster purity, phone purity and PNMI against frame-level GMM labels.

    `cluster_input_type="segment"` expects segment-level cluster HDFs and expands
    each segment label to 20ms frames using the original segment-start HDF.
    `cluster_input_type="frame"` expects frame-level cluster HDFs directly.
    The 10ms GMM alignment is downsampled by taking every `alignment_downsample_rate`th
    label, default 2.
    """

    def __init__(
        self,
        *,
        alignment_hdfs,
        cluster_hdfs,
        utterance_select: tk.Path,
        idx_to_phoneme: tk.Path | None = None,
        strip_eow_from_alignment: bool = True,
        compute_frame_error_rate: bool = False,
        cluster_input_type: str = "segment",
        segment_start_hdf: tk.Path | None = None,
        feature_hdf: tk.Path | None = None,
        segment_downsample_rate: int = 2,
        alignment_downsample_rate: int = 2,
        segment_random_seed: int = 1,
        num_clusters: int | None = None,
        store_joint_probability: bool = False,
        joint_probability_filename: str = "phone_cluster_joint_probability.npz",
        output_filename: str = "cluster_purity_pnmi.txt",
        mem_rqmt: int = 24,
        time_rqmt: int = 5,
    ):
        if cluster_input_type not in {"segment", "frame"}:
            raise ValueError(f"cluster_input_type must be 'segment' or 'frame', got {cluster_input_type!r}")
        if cluster_input_type == "segment" and (segment_start_hdf is None or feature_hdf is None):
            raise ValueError("segment_start_hdf and feature_hdf are required for segment-level cluster inputs")
        if segment_downsample_rate <= 0:
            raise ValueError(f"segment_downsample_rate must be positive, got {segment_downsample_rate}")
        if alignment_downsample_rate <= 0:
            raise ValueError(f"alignment_downsample_rate must be positive, got {alignment_downsample_rate}")
        self.alignment_hdfs = alignment_hdfs if isinstance(alignment_hdfs, (list, tuple)) else [alignment_hdfs]
        self.cluster_hdfs = cluster_hdfs if isinstance(cluster_hdfs, (list, tuple)) else [cluster_hdfs]
        self.utterance_select = utterance_select
        self.idx_to_phoneme = idx_to_phoneme
        self.strip_eow_from_alignment = strip_eow_from_alignment
        self.compute_frame_error_rate = compute_frame_error_rate
        self.cluster_input_type = cluster_input_type
        self.segment_start_hdf = segment_start_hdf
        self.feature_hdf = feature_hdf
        self.segment_downsample_rate = segment_downsample_rate
        self.alignment_downsample_rate = alignment_downsample_rate
        self.segment_random_seed = segment_random_seed
        self.num_clusters = num_clusters
        self.store_joint_probability = store_joint_probability
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_report = self.output_path(output_filename)
        self.out_joint_probability = self.output_path(joint_probability_filename) if store_joint_probability else None

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

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
    def _read_global_tags(paths):
        import h5py

        tags = []
        for path in paths:
            with h5py.File(_get_path(path), "r") as hdf:
                tags.extend(_read_tag(tag) for tag in hdf["seqTags"][:])
        return tags

    @staticmethod
    def _read_seq_index(path):
        import h5py
        import numpy as np

        with h5py.File(_get_path(path), "r") as hdf:
            seq_lengths = hdf["seqLengths"][:, 0]
            seq_tags = hdf["seqTags"][:]
            offsets = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])
            return {
                _read_tag(tag): (int(offset), int(length))
                for tag, offset, length in zip(seq_tags, offsets, seq_lengths)
            }

    @staticmethod
    def _read_seq_lengths(path):
        import h5py

        with h5py.File(_get_path(path), "r") as hdf:
            return {
                _read_tag(tag): int(length)
                for tag, length in zip(hdf["seqTags"][:], hdf["seqLengths"][:, 0])
            }

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
    def _normalize_phoneme(phoneme: str, *, strip_eow: bool):
        phoneme = str(phoneme)
        if strip_eow:
            phoneme = phoneme.rstrip("#")
        if phoneme == "[SILENCE]":
            phoneme = "[SIL]"
        return phoneme

    @classmethod
    def _build_alignment_label_map(cls, idx_to_phoneme, *, strip_eow: bool):
        phoneme_to_label = {}
        max_idx = max(idx_to_phoneme) if idx_to_phoneme else -1
        label_map = [-1] * (max_idx + 1)
        normalized_phonemes = []
        for idx in sorted(idx_to_phoneme):
            phoneme = cls._normalize_phoneme(idx_to_phoneme[idx], strip_eow=strip_eow)
            if phoneme not in phoneme_to_label:
                phoneme_to_label[phoneme] = len(phoneme_to_label)
                normalized_phonemes.append(phoneme)
            label_map[idx] = phoneme_to_label[phoneme]
        return label_map, normalized_phonemes

    @staticmethod
    def _read_alignment_labels(paths, selected_tags, *, downsample_rate: int, label_map=None):
        import h5py
        import numpy as np

        selected_tags = set(selected_tags)
        labels_by_tag = {}
        max_label = -1
        for path in paths:
            with h5py.File(_get_path(path), "r") as hdf:
                inputs = hdf["inputs"]
                seq_lengths = hdf["seqLengths"][:, 0]
                seq_tags = hdf["seqTags"][:]
                offsets = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])
                for seq_tag, offset, length in zip(seq_tags, offsets, seq_lengths):
                    tag = _read_tag(seq_tag)
                    if tag not in selected_tags:
                        continue
                    labels = inputs[int(offset) : int(offset) + int(length)]
                    if labels.ndim > 1:
                        labels = labels[:, 0]
                    labels = labels[::downsample_rate].astype("int64", copy=False)
                    if label_map is not None:
                        mapped = np.full(labels.shape, -1, dtype="int64")
                        for old_idx, new_idx in enumerate(label_map):
                            if new_idx >= 0:
                                mapped[labels == old_idx] = int(new_idx)
                        labels = mapped
                    labels_by_tag[tag] = labels
                    if labels.size:
                        max_label = max(max_label, int(labels.max()))
        return labels_by_tag, max_label + 1

    @staticmethod
    def _read_start_sequences(path):
        import h5py
        import numpy as np

        starts_by_tag = {}
        with h5py.File(_get_path(path), "r") as hdf:
            inputs = hdf["inputs"]
            seq_lengths = hdf["seqLengths"][:, 0]
            seq_tags = hdf["seqTags"][:]
            offsets = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])
            for seq_tag, offset, length in zip(seq_tags, offsets, seq_lengths):
                starts = inputs[int(offset) : int(offset) + int(length)]
                if starts.ndim > 1:
                    starts = starts[:, 0]
                starts_by_tag[_read_tag(seq_tag)] = starts.astype("int64", copy=False)
        return starts_by_tag

    @staticmethod
    def _downsample_starts(starts, *, downsample_rate: int, feature_len: int, rng):
        import numpy as np

        starts = np.asarray(starts, dtype="int64")
        starts = starts[starts >= 0]
        if starts.size == 0:
            starts = np.asarray([0], dtype="int64")

        base = starts // downsample_rate
        remainder = starts % downsample_rate
        random_offsets = rng.integers(0, downsample_rate, size=starts.shape[0])
        downsampled = base + (random_offsets < remainder).astype("int64")

        downsampled = downsampled[(downsampled >= 0) & (downsampled < feature_len)]
        if feature_len > 0:
            downsampled = np.concatenate([[0], downsampled])
        if downsampled.size == 0:
            return np.asarray([], dtype="int64")
        return np.unique(downsampled).astype("int64")

    @staticmethod
    def _expand_segment_clusters(segment_labels, starts, *, feature_len: int):
        import numpy as np

        segment_labels = np.asarray(segment_labels, dtype="int64")
        starts = np.asarray(starts, dtype="int64")
        starts = starts[(starts >= 0) & (starts < feature_len)]
        starts = np.unique(starts)
        if feature_len == 0 or starts.size == 0 or segment_labels.size == 0:
            return np.zeros((0,), dtype="int64")
        num_segments = min(int(segment_labels.size), int(starts.size))
        starts = starts[:num_segments]
        segment_labels = segment_labels[:num_segments]
        ends = np.concatenate([starts[1:], np.asarray([feature_len], dtype="int64")])
        valid = ends > starts
        starts = starts[valid]
        ends = ends[valid]
        segment_labels = segment_labels[valid]
        frame_labels = np.empty((feature_len,), dtype="int64")
        frame_labels[:] = -1
        for label, start, end in zip(segment_labels, starts, ends):
            frame_labels[int(start) : int(end)] = int(label)
        return frame_labels[frame_labels >= 0]

    @staticmethod
    def _update_contingency(contingency, phone_labels, cluster_labels):
        import numpy as np

        valid = (phone_labels >= 0) & (cluster_labels >= 0)
        if not np.any(valid):
            return 0
        phone_labels = phone_labels[valid]
        cluster_labels = cluster_labels[valid]
        np.add.at(contingency, (phone_labels, cluster_labels), 1)
        return int(phone_labels.size)

    @staticmethod
    def _metrics(contingency, *, compute_frame_error_rate: bool):
        import numpy as np

        contingency = contingency.astype("float64", copy=False)
        total = float(contingency.sum())
        if total <= 0.0:
            return {
                "num_frames": 0,
                "cluster_purity": float("nan"),
                "phone_purity": float("nan"),
                "pnmi": float("nan"),
                "mutual_information": float("nan"),
                "phone_entropy": float("nan"),
            }
        cluster_purity = float(contingency.max(axis=0).sum() / total)
        phone_purity = float(contingency.max(axis=1).sum() / total)

        joint = contingency / total
        phone_marginal = joint.sum(axis=1)
        cluster_marginal = joint.sum(axis=0)
        nonzero = joint > 0.0
        denom = phone_marginal[:, None] * cluster_marginal[None, :]
        mutual_information = float((joint[nonzero] * np.log(joint[nonzero] / denom[nonzero])).sum())
        phone_nonzero = phone_marginal > 0.0
        phone_entropy = float(-(phone_marginal[phone_nonzero] * np.log(phone_marginal[phone_nonzero])).sum())
        pnmi = float(mutual_information / phone_entropy) if phone_entropy > 0.0 else float("nan")
        metrics = {
            "num_frames": int(total),
            "cluster_purity": cluster_purity,
            "phone_purity": phone_purity,
            "pnmi": pnmi,
            "mutual_information": mutual_information,
            "phone_entropy": phone_entropy,
        }
        if compute_frame_error_rate:
            majority_correct = float(contingency.max(axis=0).sum())
            metrics["majority_mapping_frame_error_rate"] = float((total - majority_correct) / total)
            metrics["majority_mapping_frame_errors"] = int(round(total - majority_correct))
        return metrics

    def _selected_tags(self):
        selected_indices, selected_tags = self._read_select(self.utterance_select)
        if selected_indices:
            global_tags = self._read_global_tags(self.alignment_hdfs)
            try:
                selected_tags.extend(global_tags[idx] for idx in selected_indices)
            except IndexError as exc:
                raise IndexError(f"Utterance select index out of range for {len(global_tags)} alignment sequences") from exc
        return list(dict.fromkeys(selected_tags))

    def _infer_num_clusters(self):
        import h5py

        if self.num_clusters is not None:
            return int(self.num_clusters)
        max_cluster = -1
        attr_num_clusters = None
        for path in self.cluster_hdfs:
            with h5py.File(_get_path(path), "r") as hdf:
                if "numLabels" in hdf.attrs:
                    attr_num_clusters = max(int(attr_num_clusters or 0), int(hdf.attrs["numLabels"]))
                inputs = hdf["inputs"]
                if inputs.shape[0] > 0:
                    max_cluster = max(max_cluster, int(inputs[:].max()))
        return max(int(attr_num_clusters or 0), max_cluster + 1)

    def run(self):
        import h5py
        import numpy as np

        selected_tags = self._selected_tags()
        selected_tag_set = set(selected_tags)
        alignment_label_map = None
        normalized_phonemes = None
        if self.idx_to_phoneme is not None:
            idx_to_phoneme = self._read_idx_to_phoneme(self.idx_to_phoneme)
            alignment_label_map, normalized_phonemes = self._build_alignment_label_map(
                idx_to_phoneme,
                strip_eow=self.strip_eow_from_alignment,
            )
        alignment_by_tag, num_phones = self._read_alignment_labels(
            self.alignment_hdfs,
            selected_tags,
            downsample_rate=self.alignment_downsample_rate,
            label_map=alignment_label_map,
        )
        if normalized_phonemes is not None:
            num_phones = len(normalized_phonemes)
        num_clusters = self._infer_num_clusters()
        contingency = np.zeros((num_phones, num_clusters), dtype="int64")

        starts_by_tag = None
        feature_lengths = None
        if self.cluster_input_type == "segment":
            starts_by_tag = self._read_start_sequences(self.segment_start_hdf)
            feature_lengths = self._read_seq_lengths(self.feature_hdf)

        num_eval = 0
        num_missing_alignment = 0
        num_missing_aux = 0
        num_length_mismatch = 0
        examples = []

        for part_idx, cluster_hdf in enumerate(self.cluster_hdfs):
            rng = np.random.default_rng(self.segment_random_seed + part_idx)
            with h5py.File(_get_path(cluster_hdf), "r") as hdf:
                inputs = hdf["inputs"]
                seq_lengths = hdf["seqLengths"][:, 0]
                seq_tags = hdf["seqTags"][:]
                offsets = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])
                for seq_tag, offset, length in zip(seq_tags, offsets, seq_lengths):
                    tag = _read_tag(seq_tag)
                    labels = inputs[int(offset) : int(offset) + int(length)]
                    if labels.ndim > 1:
                        labels = labels[:, 0]
                    labels = labels.astype("int64", copy=False)

                    if self.cluster_input_type == "segment":
                        if tag not in starts_by_tag or tag not in feature_lengths:
                            if tag in selected_tag_set:
                                num_missing_aux += 1
                            continue
                        starts_20ms = self._downsample_starts(
                            starts_by_tag[tag],
                            downsample_rate=self.segment_downsample_rate,
                            feature_len=feature_lengths[tag],
                            rng=rng,
                        )
                        cluster_frame_labels = self._expand_segment_clusters(
                            labels,
                            starts_20ms,
                            feature_len=feature_lengths[tag],
                        )
                    else:
                        cluster_frame_labels = labels

                    if tag not in selected_tag_set:
                        continue
                    if tag not in alignment_by_tag:
                        num_missing_alignment += 1
                        continue

                    phone_labels = alignment_by_tag[tag]
                    valid_len = min(int(phone_labels.shape[0]), int(cluster_frame_labels.shape[0]))
                    if valid_len <= 0:
                        continue
                    if int(phone_labels.shape[0]) != int(cluster_frame_labels.shape[0]):
                        num_length_mismatch += 1
                    used = self._update_contingency(contingency, phone_labels[:valid_len], cluster_frame_labels[:valid_len])
                    if used > 0:
                        num_eval += 1
                    if len(examples) < 5:
                        examples.append((tag, int(phone_labels.shape[0]), int(cluster_frame_labels.shape[0]), valid_len))

        metrics = self._metrics(contingency, compute_frame_error_rate=self.compute_frame_error_rate)
        with open(self.out_report.get_path(), "w") as f:
            f.write(f"cluster_input_type: {self.cluster_input_type}\n")
            f.write(f"alignment_downsample_rate: {self.alignment_downsample_rate}\n")
            f.write(f"idx_to_phoneme: {_get_path(self.idx_to_phoneme) if self.idx_to_phoneme is not None else None}\n")
            f.write(f"strip_eow_from_alignment: {self.strip_eow_from_alignment}\n")
            f.write(f"compute_frame_error_rate: {self.compute_frame_error_rate}\n")
            f.write(f"store_joint_probability: {self.store_joint_probability}\n")
            if self.out_joint_probability is not None:
                f.write(f"joint_probability_npz: {self.out_joint_probability.get_path()}\n")
            f.write(f"segment_downsample_rate: {self.segment_downsample_rate}\n")
            f.write(f"segment_random_seed: {self.segment_random_seed}\n")
            f.write(f"utterance_select: {_get_path(self.utterance_select)}\n")
            f.write(f"num_selected_tags: {len(selected_tags)}\n")
            f.write(f"num_eval_sequences: {num_eval}\n")
            f.write(f"num_missing_alignment: {num_missing_alignment}\n")
            f.write(f"num_missing_segment_or_feature: {num_missing_aux}\n")
            f.write(f"num_length_mismatch: {num_length_mismatch}\n")
            f.write(f"num_phones: {num_phones}\n")
            f.write(f"num_clusters: {num_clusters}\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.12g}\n" if isinstance(value, float) else f"{key}: {value}\n")
            f.write("\nexamples tag phone_frames cluster_frames used_frames:\n")
            for tag, phone_len, cluster_len, valid_len in examples:
                f.write(f"{tag}\t{phone_len}\t{cluster_len}\t{valid_len}\n")
            if normalized_phonemes is not None:
                f.write("\nphone_label_map:\n")
                for idx, phoneme in enumerate(normalized_phonemes):
                    f.write(f"{idx}\t{phoneme}\n")
            f.write("\ncluster_majority_phone_index:\n")
            for cluster_idx in range(num_clusters):
                if contingency[:, cluster_idx].sum() == 0:
                    f.write(f"{cluster_idx}\t-1\t0\n")
                else:
                    phone_idx = int(contingency[:, cluster_idx].argmax())
                    phone_name = normalized_phonemes[phone_idx] if normalized_phonemes is not None else str(phone_idx)
                    f.write(f"{cluster_idx}\t{phone_idx}\t{phone_name}\t{int(contingency[phone_idx, cluster_idx])}\n")

        if self.out_joint_probability is not None:
            total = contingency.sum()
            if total <= 0:
                raise ValueError("Cannot store joint probability for empty contingency table.")
            phoneme_tokens = (
                normalized_phonemes
                if normalized_phonemes is not None
                else [str(phone_idx) for phone_idx in range(contingency.shape[0])]
            )
            cluster_tokens = [str(cluster_idx) for cluster_idx in range(contingency.shape[1])]
            np.savez(
                self.out_joint_probability.get_path(),
                transport=contingency.astype("float64") / float(total),
                counts=contingency,
                phoneme_tokens=np.asarray(phoneme_tokens, dtype=object),
                cluster_tokens=np.asarray(cluster_tokens, dtype=object),
            )
