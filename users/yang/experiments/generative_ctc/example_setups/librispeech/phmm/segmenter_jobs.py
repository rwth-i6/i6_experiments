from __future__ import annotations

from sisyphus import Job, Task, tk


def _read_tag(tag):
    if isinstance(tag, (bytes, bytearray)):
        return tag.decode("utf8")
    return str(tag)


def _write_segment_starts_hdf(path, seq_tags, segment_start_seqs):
    import h5py
    import numpy as np

    out_lengths = [len(seq) for seq in segment_start_seqs]
    total_len = sum(out_lengths)
    out_inputs = np.zeros((total_len, 1), dtype="int32")
    offset = 0
    for segment_starts in segment_start_seqs:
        next_offset = offset + len(segment_starts)
        out_inputs[offset:next_offset, 0] = segment_starts
        offset = next_offset

    with h5py.File(path, "w") as out_hdf:
        out_hdf.create_dataset("inputs", data=out_inputs)
        out_hdf.create_dataset("seqLengths", data=np.asarray(out_lengths, dtype="int32")[:, None])
        dt = h5py.special_dtype(vlen=bytes)
        out_hdf.create_dataset(
            "seqTags",
            data=np.asarray([_read_tag(tag).encode("utf8") for tag in seq_tags], dtype=object),
            dtype=dt,
        )
        out_hdf.create_dataset("labels", data=np.asarray([], dtype="S5"))


class DetectBoundariesFromScoreHDFJob(Job):
    """
    Detect segment starts from per-frame boundary scores.

    For each utterance, frame 0 is always a segment start. Additional starts are
    the frames after the top-scoring boundary positions. The per-utterance target
    ratio is ratio * (1 + u), where u is uniformly sampled from [-perturb, perturb].
    """

    def __init__(
        self,
        score_hdf: tk.Path,
        *,
        ratio: float = 0.12,
        perturb: float = 0.1,
        min_distance: int = 1,
        random_seed: int = 1,
        output_filename: str = "segment_starts.hdf",
    ):
        if ratio <= 0.0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if perturb < 0.0:
            raise ValueError(f"perturb must be non-negative, got {perturb}")
        if min_distance < 0:
            raise ValueError(f"min_distance must be non-negative, got {min_distance}")
        self.score_hdf = score_hdf
        self.ratio = ratio
        self.perturb = perturb
        self.min_distance = min_distance
        self.random_seed = random_seed
        self.out_hdf = self.output_path(output_filename)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 24, "time": 5})

    @staticmethod
    def _merge_close_starts(starts, *, min_distance: int):
        import numpy as np

        if len(starts) <= 1 or min_distance < 0:
            return starts
        merged = []
        for start in starts:
            start = int(start)
            if merged and start - merged[-1] <= min_distance:
                merged[-1] = start
            else:
                merged.append(start)
        return np.asarray(merged, dtype="int32")

    @classmethod
    def _select_segment_starts(cls, scores, *, ratio: float, min_distance: int):
        import numpy as np

        num_frames = len(scores) + 1
        num_starts = int(round(num_frames * ratio))
        num_starts = max(1, min(num_frames, num_starts))
        num_extra_starts = num_starts - 1
        if len(scores) == 0 or num_extra_starts == 0:
            return np.asarray([0], dtype="int32")

        top_score_indices = np.argpartition(-scores, num_extra_starts - 1)[:num_extra_starts]
        starts = np.sort(np.concatenate([[0], top_score_indices + 1])).astype("int32")
        return cls._merge_close_starts(starts, min_distance=min_distance)

    def run(self):
        import h5py
        import numpy as np

        rng = np.random.default_rng(self.random_seed)

        with h5py.File(self.score_hdf.get_path(), "r") as in_hdf:
            inputs = in_hdf["inputs"][:, 0]
            seq_lengths = in_hdf["seqLengths"][:, 0]
            seq_tags = in_hdf["seqTags"][:]
            starts = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])

            segment_start_seqs = []
            for start, length in zip(starts, seq_lengths):
                start = int(start)
                length = int(length)
                ratio = self.ratio * (1.0 + rng.uniform(-self.perturb, self.perturb))
                segment_starts = self._select_segment_starts(
                    inputs[start : start + length],
                    ratio=ratio,
                    min_distance=self.min_distance,
                )
                segment_start_seqs.append(segment_starts)

            _write_segment_starts_hdf(self.out_hdf.get_path(), seq_tags, segment_start_seqs)


class DetectPeaksFromScoreHDFJob(Job):
    """
    Detect segment starts from local peaks in per-frame boundary scores.

    This follows the inference style of the UnsupSeg code: score peaks with
    sufficient prominence are treated as segment ends; the following frames are
    segment starts. Frame 0 is always included.
    """

    def __init__(
        self,
        score_hdf: tk.Path,
        *,
        prominence: float = 0.05,
        min_distance: int = 1,
        output_filename: str = "peak_segment_starts.hdf",
    ):
        if prominence < 0.0:
            raise ValueError(f"prominence must be non-negative, got {prominence}")
        if min_distance < 0:
            raise ValueError(f"min_distance must be non-negative, got {min_distance}")
        self.score_hdf = score_hdf
        self.prominence = prominence
        self.min_distance = min_distance
        self.out_hdf = self.output_path(output_filename)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 24, "time": 5})

    @staticmethod
    def _merge_close_starts(starts, *, min_distance: int):
        return DetectBoundariesFromScoreHDFJob._merge_close_starts(starts, min_distance=min_distance)

    @classmethod
    def _select_segment_starts(cls, scores, *, prominence: float, min_distance: int):
        import numpy as np

        if len(scores) == 0:
            return np.asarray([0], dtype="int32")

        try:
            from scipy.signal import find_peaks
        except ImportError as exc:
            raise ImportError("DetectPeaksFromScoreHDFJob requires scipy.signal.find_peaks") from exc

        peaks, _properties = find_peaks(scores, prominence=prominence)
        starts = np.sort(np.concatenate([[0], peaks + 1])).astype("int32")
        return cls._merge_close_starts(starts, min_distance=min_distance)

    def run(self):
        import h5py
        import numpy as np

        with h5py.File(self.score_hdf.get_path(), "r") as in_hdf:
            inputs = in_hdf["inputs"][:, 0]
            seq_lengths = in_hdf["seqLengths"][:, 0]
            seq_tags = in_hdf["seqTags"][:]
            starts = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])

            segment_start_seqs = []
            for start, length in zip(starts, seq_lengths):
                start = int(start)
                length = int(length)
                segment_starts = self._select_segment_starts(
                    inputs[start : start + length],
                    prominence=self.prominence,
                    min_distance=self.min_distance,
                )
                segment_start_seqs.append(segment_starts)

            _write_segment_starts_hdf(self.out_hdf.get_path(), seq_tags, segment_start_seqs)


class DumpGmmAlignmentSegmentStartsJob(Job):
    """
    Convert frame-level GMM alignment labels to segment-start frame indices.

    Frame 0 is always a start. Every later frame t is a start iff label[t] != label[t - 1].
    """

    def __init__(
        self,
        alignment_hdfs,
        *,
        output_filename: str = "gmm_segment_starts.hdf",
    ):
        self.alignment_hdfs = alignment_hdfs if isinstance(alignment_hdfs, (list, tuple)) else [alignment_hdfs]
        self.out_hdf = self.output_path(output_filename)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 24, "time": 5})

    @staticmethod
    def _alignment_starts(labels):
        import numpy as np

        if len(labels) == 0:
            return np.asarray([0], dtype="int32")
        change_indices = np.nonzero(labels[1:] != labels[:-1])[0] + 1
        return np.concatenate([[0], change_indices]).astype("int32")

    def run(self):
        import h5py
        import numpy as np

        seq_tags = []
        segment_start_seqs = []
        for hdf_path in self.alignment_hdfs:
            with h5py.File(hdf_path.get_path(), "r") as in_hdf:
                inputs = in_hdf["inputs"]
                seq_lengths = in_hdf["seqLengths"][:, 0]
                hdf_seq_tags = in_hdf["seqTags"][:]
                starts = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])
                for seq_tag, start, length in zip(hdf_seq_tags, starts, seq_lengths):
                    start = int(start)
                    length = int(length)
                    labels = inputs[start : start + length]
                    if labels.ndim > 1:
                        labels = labels[:, 0]
                    seq_tags.append(seq_tag)
                    segment_start_seqs.append(self._alignment_starts(labels))

        _write_segment_starts_hdf(self.out_hdf.get_path(), seq_tags, segment_start_seqs)


class ComputeSegmentRepresentationsFromHDFJob(Job):
    """
    Average frame-level features into segment-level representations.

    Segment starts are read from an HDF where `inputs` stores start-frame indices.
    Starts can be in a higher frame rate than the feature HDF, e.g. 10ms starts
    and 20ms wav2vec2 features. In that case, starts are downsampled with
    randomized unbiased rounding:

        floor(t / a) + 1{randint(0, a - 1) < (t mod a)}

    Empty segments after downsampling are ignored.
    """

    def __init__(
        self,
        start_frame_hdf: tk.Path,
        feature_hdf: tk.Path,
        *,
        downsample_rate: int = 2,
        random_seed: int = 1,
        num_partitions: int = 1,
        partition_index: int = 0,
        mem_rqmt: int = 24,
        time_rqmt: int = 24,
        output_filename: str = "segment_representations.hdf",
    ):
        if downsample_rate <= 0:
            raise ValueError(f"downsample_rate must be positive, got {downsample_rate}")
        if num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive, got {num_partitions}")
        if not 0 <= partition_index < num_partitions:
            raise ValueError(
                f"partition_index must be in [0, num_partitions), got {partition_index} for {num_partitions}"
            )
        self.start_frame_hdf = start_frame_hdf
        self.feature_hdf = feature_hdf
        self.downsample_rate = downsample_rate
        self.random_seed = random_seed
        self.num_partitions = num_partitions
        self.partition_index = partition_index
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_hdf = self.output_path(output_filename)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

    @staticmethod
    def _read_seq_index(hdf):
        import numpy as np

        seq_lengths = hdf["seqLengths"][:, 0]
        seq_tags = hdf["seqTags"][:]
        seq_offsets = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])
        return {
            _read_tag(tag): (int(offset), int(length))
            for tag, offset, length in zip(seq_tags, seq_offsets, seq_lengths)
        }

    @staticmethod
    def _copy_hdf_labels(in_hdf, out_hdf):
        import numpy as np

        if "labels" in in_hdf:
            out_hdf.create_dataset("labels", data=in_hdf["labels"][:], dtype=in_hdf["labels"].dtype)
        else:
            out_hdf.create_dataset("labels", data=np.asarray([], dtype="S5"))

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
    def _compute_segment_means(features, starts):
        import numpy as np

        feature_len = int(features.shape[0])
        if feature_len == 0 or starts.size == 0:
            return np.zeros((0, int(features.shape[1])), dtype="float32")

        starts = starts[(starts >= 0) & (starts < feature_len)]
        if starts.size == 0:
            return np.zeros((0, int(features.shape[1])), dtype="float32")
        starts = np.unique(starts.astype("int64"))
        ends = np.concatenate([starts[1:], np.asarray([feature_len], dtype="int64")])
        valid = ends > starts
        starts = starts[valid]
        ends = ends[valid]
        if starts.size == 0:
            return np.zeros((0, int(features.shape[1])), dtype="float32")

        cumsum = np.concatenate(
            [np.zeros((1, features.shape[1]), dtype="float64"), np.cumsum(features.astype("float64"), axis=0)],
            axis=0,
        )
        sums = cumsum[ends] - cumsum[starts]
        lengths = (ends - starts).astype("float64")[:, None]
        return (sums / lengths).astype("float32")

    def run(self):
        import h5py
        import numpy as np

        rng = np.random.default_rng(self.random_seed)

        with h5py.File(self.start_frame_hdf.get_path(), "r") as start_hdf, h5py.File(
            self.feature_hdf.get_path(), "r"
        ) as feature_hdf, h5py.File(self.out_hdf.get_path(), "w") as out_hdf:
            feature_inputs = feature_hdf["inputs"]
            start_inputs = start_hdf["inputs"]
            if feature_inputs.ndim != 2:
                raise ValueError(f"Expected dense feature inputs with ndim=2, got shape {feature_inputs.shape}")
            feature_dim = int(feature_inputs.shape[1])

            feature_index = self._read_seq_index(feature_hdf)
            start_lengths = start_hdf["seqLengths"][:, 0]
            start_tags = start_hdf["seqTags"][:]
            start_offsets = np.concatenate([[0], np.cumsum(start_lengths[:-1])])
            partition_seq_indices = np.array_split(
                np.arange(len(start_lengths), dtype="int64"), self.num_partitions
            )[self.partition_index]

            for key, value in feature_hdf.attrs.items():
                out_hdf.attrs[key] = value
            out_hdf.attrs["inputPattSize"] = feature_dim
            out_hdf.attrs["numLabels"] = feature_dim
            out_hdf.attrs["numTimesteps"] = 0
            out_hdf.attrs["numSeqs"] = 0

            out_inputs = out_hdf.create_dataset(
                "inputs",
                shape=(0, feature_dim),
                maxshape=(None, feature_dim),
                dtype="float32",
                chunks=(1024, feature_dim),
            )
            out_seq_lengths = out_hdf.create_dataset("seqLengths", shape=(0, 2), maxshape=(None, 2), dtype="int32")
            dt = h5py.special_dtype(vlen=str)
            out_seq_tags = out_hdf.create_dataset("seqTags", shape=(0,), maxshape=(None,), dtype=dt)
            self._copy_hdf_labels(feature_hdf, out_hdf)

            total_segments = 0
            num_output_seqs = 0
            missing_feature_tags = []
            for seq_idx in partition_seq_indices:
                seq_tag = start_tags[seq_idx]
                start_offset = start_offsets[seq_idx]
                start_length = start_lengths[seq_idx]
                tag = _read_tag(seq_tag)
                if tag not in feature_index:
                    missing_feature_tags.append(tag)
                    continue

                feature_offset, feature_len = feature_index[tag]
                starts = start_inputs[int(start_offset) : int(start_offset) + int(start_length)]
                if starts.ndim > 1:
                    starts = starts[:, 0]
                downsampled_starts = self._downsample_starts(
                    starts,
                    downsample_rate=self.downsample_rate,
                    feature_len=feature_len,
                    rng=rng,
                )
                features = feature_inputs[feature_offset : feature_offset + feature_len]
                segment_representations = self._compute_segment_means(features, downsampled_starts)
                num_segments = int(segment_representations.shape[0])

                out_inputs.resize(total_segments + num_segments, axis=0)
                if num_segments > 0:
                    out_inputs[total_segments : total_segments + num_segments] = segment_representations
                out_seq_lengths.resize(num_output_seqs + 1, axis=0)
                out_seq_lengths[num_output_seqs, 0] = num_segments
                out_seq_lengths[num_output_seqs, 1] = 0
                out_seq_tags.resize(num_output_seqs + 1, axis=0)
                out_seq_tags[num_output_seqs] = tag

                total_segments += num_segments
                num_output_seqs += 1

            if missing_feature_tags:
                examples = ", ".join(missing_feature_tags[:5])
                raise KeyError(
                    f"{len(missing_feature_tags)} sequence tags from start_frame_hdf are missing in feature_hdf. "
                    f"Examples: {examples}"
                )

            out_hdf.attrs["numTimesteps"] = total_segments
            out_hdf.attrs["numSeqs"] = num_output_seqs


def create_partitioned_segment_representation_jobs(
    *,
    output_prefix: str,
    start_frame_hdf: tk.Path,
    feature_hdf: tk.Path,
    num_partitions: int = 10,
    downsample_rate: int = 2,
    random_seed: int = 1,
    mem_rqmt: int = 24,
    time_rqmt: int = 24,
    output_filename_prefix: str = "segment_representations",
):
    """
    Create independent CPU jobs for segment-representation extraction.

    Returns the list of partition HDF paths. Each partition output is also
    registered under `output_prefix`.
    """
    outputs = []
    for partition_index in range(num_partitions):
        output_filename = f"{output_filename_prefix}.{partition_index:03d}.hdf"
        job = ComputeSegmentRepresentationsFromHDFJob(
            start_frame_hdf=start_frame_hdf,
            feature_hdf=feature_hdf,
            downsample_rate=downsample_rate,
            random_seed=random_seed + partition_index,
            num_partitions=num_partitions,
            partition_index=partition_index,
            mem_rqmt=mem_rqmt,
            time_rqmt=time_rqmt,
            output_filename=output_filename,
        )
        job.add_alias(f"{output_prefix}/part_{partition_index:03d}")
        tk.register_output(f"{output_prefix}/{output_filename}", job.out_hdf)
        outputs.append(job.out_hdf)
    return outputs


class CompareSegmentStartHDFJob(Job):
    """
    Compare predicted/source segment starts against target/reference segment starts.

    Only utterances present in the source HDF are evaluated. Boundaries are matched
    one-to-one within +/- tolerance frames using a two-pointer scan on sorted starts.
    """

    def __init__(
        self,
        source_hdf: tk.Path,
        target_hdf: tk.Path,
        *,
        tolerance: int = 2,
        ignore_zero: bool = True,
        source_score_hdf: tk.Path | None = None,
        target_alignment_hdfs=None,
        output_filename: str = "boundary_comparison.txt",
    ):
        if tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")
        self.source_hdf = source_hdf
        self.target_hdf = target_hdf
        self.tolerance = tolerance
        self.ignore_zero = ignore_zero
        self.source_score_hdf = source_score_hdf
        self.target_alignment_hdfs = (
            target_alignment_hdfs if isinstance(target_alignment_hdfs, (list, tuple)) else [target_alignment_hdfs]
        ) if target_alignment_hdfs is not None else None
        self.out_report = self.output_path(output_filename)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 24, "time": 5})

    @staticmethod
    def _read_start_hdf(path):
        import h5py
        import numpy as np

        data = {}
        with h5py.File(path, "r") as hdf:
            inputs = hdf["inputs"][:, 0]
            seq_lengths = hdf["seqLengths"][:, 0]
            seq_tags = hdf["seqTags"][:]
            starts = np.concatenate([[0], np.cumsum(seq_lengths[:-1])])
            for seq_tag, start, length in zip(seq_tags, starts, seq_lengths):
                start = int(start)
                length = int(length)
                tag = _read_tag(seq_tag)
                data[tag] = np.asarray(inputs[start : start + length], dtype="int64")
        return data

    @staticmethod
    def _match_boundaries(source, target, *, tolerance: int):
        source_idx = 0
        target_idx = 0
        matched = 0
        abs_error_sum = 0
        while source_idx < len(source) and target_idx < len(target):
            source_boundary = int(source[source_idx])
            target_boundary = int(target[target_idx])
            diff = source_boundary - target_boundary
            if abs(diff) <= tolerance:
                matched += 1
                abs_error_sum += abs(diff)
                source_idx += 1
                target_idx += 1
            elif source_boundary < target_boundary:
                source_idx += 1
            else:
                target_idx += 1
        return matched, abs_error_sum

    @staticmethod
    def _safe_div(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    @staticmethod
    def _read_score_frame_counts(path):
        import h5py

        with h5py.File(path, "r") as hdf:
            seq_lengths = hdf["seqLengths"][:, 0]
            seq_tags = hdf["seqTags"][:]
            return {_read_tag(tag): int(length) + 1 for tag, length in zip(seq_tags, seq_lengths)}

    @staticmethod
    def _read_alignment_frame_counts(paths):
        import h5py

        frame_counts = {}
        for path in paths:
            with h5py.File(path.get_path(), "r") as hdf:
                seq_lengths = hdf["seqLengths"][:, 0]
                seq_tags = hdf["seqTags"][:]
                for tag, length in zip(seq_tags, seq_lengths):
                    frame_counts[_read_tag(tag)] = int(length)
        return frame_counts

    def run(self):
        import numpy as np

        source_data = self._read_start_hdf(self.source_hdf.get_path())
        target_data = self._read_start_hdf(self.target_hdf.get_path())
        source_frame_counts = (
            self._read_score_frame_counts(self.source_score_hdf.get_path()) if self.source_score_hdf is not None else {}
        )
        target_frame_counts = (
            self._read_alignment_frame_counts(self.target_alignment_hdfs)
            if self.target_alignment_hdfs is not None
            else {}
        )

        num_utts = 0
        missing_targets = 0
        total_source = 0
        total_target = 0
        total_matched = 0
        total_abs_error = 0
        total_source_frames = 0
        total_target_frames = 0
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        per_utt_lines = []

        for tag in sorted(source_data):
            source = np.sort(source_data[tag])
            target = target_data.get(tag)
            if target is None:
                missing_targets += 1
                continue
            target = np.sort(target)
            if self.ignore_zero:
                source = source[source != 0]
                target = target[target != 0]

            matched, abs_error = self._match_boundaries(source, target, tolerance=self.tolerance)
            precision = self._safe_div(matched, len(source))
            recall = self._safe_div(matched, len(target))
            f1 = self._safe_div(2 * precision * recall, precision + recall)

            num_utts += 1
            total_source += len(source)
            total_target += len(target)
            total_matched += matched
            total_abs_error += abs_error
            total_source_frames += source_frame_counts.get(tag, 0)
            total_target_frames += target_frame_counts.get(tag, 0)
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1
            per_utt_lines.append(
                f"{tag}\tsource={len(source)}\ttarget={len(target)}\tmatched={matched}"
                f"\tprecision={precision:.6f}\trecall={recall:.6f}\tf1={f1:.6f}"
            )

        precision = self._safe_div(total_matched, total_source)
        recall = self._safe_div(total_matched, total_target)
        f1 = self._safe_div(2 * precision * recall, precision + recall)
        mean_abs_error = self._safe_div(total_abs_error, total_matched)
        pred_ref_ratio = self._safe_div(total_source, total_target)
        source_boundary_frame_ratio = self._safe_div(total_source, total_source_frames)
        target_boundary_frame_ratio = self._safe_div(total_target, total_target_frames)
        macro_precision = self._safe_div(macro_precision, num_utts)
        macro_recall = self._safe_div(macro_recall, num_utts)
        macro_f1 = self._safe_div(macro_f1, num_utts)

        with open(self.out_report.get_path(), "w") as f:
            f.write(f"source_hdf: {self.source_hdf.get_path()}\n")
            f.write(f"target_hdf: {self.target_hdf.get_path()}\n")
            f.write(f"tolerance_frames: {self.tolerance}\n")
            f.write(f"ignore_zero: {self.ignore_zero}\n")
            f.write(f"num_source_utts: {len(source_data)}\n")
            f.write(f"num_eval_utts: {num_utts}\n")
            f.write(f"missing_target_utts: {missing_targets}\n")
            f.write(f"num_source_boundaries: {total_source}\n")
            f.write(f"num_target_boundaries: {total_target}\n")
            f.write(f"num_matched_boundaries: {total_matched}\n")
            f.write(f"precision: {precision:.10f}\n")
            f.write(f"recall: {recall:.10f}\n")
            f.write(f"f1: {f1:.10f}\n")
            f.write(f"macro_precision: {macro_precision:.10f}\n")
            f.write(f"macro_recall: {macro_recall:.10f}\n")
            f.write(f"macro_f1: {macro_f1:.10f}\n")
            f.write(f"mean_abs_error_frames: {mean_abs_error:.10f}\n")
            f.write(f"source_target_boundary_ratio: {pred_ref_ratio:.10f}\n")
            f.write(f"num_source_frames: {total_source_frames}\n")
            f.write(f"num_target_frames: {total_target_frames}\n")
            f.write(f"source_boundaries_per_frame: {source_boundary_frame_ratio:.10f}\n")
            f.write(f"target_boundaries_per_frame: {target_boundary_frame_ratio:.10f}\n")
            f.write("\nper_utterance:\n")
            for line in per_utt_lines:
                f.write(line + "\n")
