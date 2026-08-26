"""Per-subword mean log-mel arrays and durations, from a CTC forced alignment.

The MFA table needs a phoneme lexicon and an external aligner.
A CTC alignment from our own ASR model gives the same statistics
on the subwords the model already predicts, so the pseudo-speech encoder needs neither.

A phoneme maps to roughly one encoder frame, so the MFA table holds one log-mel frame per phone.
A subword spans a range of frames, so this table holds a short array per subword,
and applying it means resampling that array to the sampled duration rather than repeating a frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any, Dict, List, Tuple

if TYPE_CHECKING:
    import numpy

from sisyphus import Job, Task, tk


class ComputeCtcSubwordStatsJob(Job):
    """
    Per-subword mean log-mel array and mean duration over real audio, from CTC align HDFs.

    CTC alignments are peaky, so each subword's boundaries are widened by ``widen_frames``
    on each side, capped at half a gap shared with the next subword so spans never overlap.
    Blank keeps whatever the widening leaves and stays a unit of its own.

    Every instance of a subword is resampled to that subword's own mean length and averaged,
    so a stored array sits at its natural length and generation only has to correct it.
    The stored length is capped at ``max_len``; ``out_duration_table`` is not capped,
    so a clipped array is still stretched to the real duration at generation time.

    Two streaming passes, since the mean length has to be known before the arrays can be filled.
    The first reads the alignment alone, so the audio is decoded only in the second,
    where a ``MetaDataset`` pairs it with the alignment by seq tag.

    The front-end must match the ASR ``asr_logmel`` setting,
    i.e. the ``rf.audio.log_mel_filterbank_from_raw`` defaults in the AED model.

    Outputs: ``out_mean_table`` npz with ``means`` [vocab, table_len, num_filters],
    ``lengths`` [vocab] and ``labels``;
    ``out_duration_table`` npz with ``durations`` [vocab] in log-mel frames and ``labels``;
    ``out_stats`` json with per-label counts and a config echo.

    :param align_hdfs: alignment shards, loaded as one HDFDataset.
        The HDF ships its own ``labels``, so no separate vocab input.
    :param dataset_dict: raw-audio dataset, paired with the alignment by seq tag
    :param audio_data_key: key of the audio in ``dataset_dict``
    :param subsample_factor: log-mel frames per alignment frame, checked per sequence
    :param widen_frames: frames added on each side of a subword, to undo some of the peakiness
    :param max_len: cap on a stored array's length, so one long mean cannot set the table width
    :param max_seqs: stop after this many sequences, for a partial pass
    """

    def __init__(
        self,
        *,
        align_hdfs: List[tk.Path],
        dataset_dict: Dict[str, Any],
        returnn_root: tk.Path,
        audio_data_key: str = "data",
        subsample_factor: int = 6,
        sample_rate: int = 16_000,
        window_len: float = 0.025,
        step_len: float = 0.010,
        num_filters: int = 80,
        peak_normalization: bool = True,
        widen_frames: int = 2,
        max_len: int = 32,
        max_seqs: Optional[int] = None,
    ):
        super().__init__()
        self.align_hdfs = align_hdfs
        self.dataset_dict = dataset_dict
        self.returnn_root = returnn_root
        self.audio_data_key = audio_data_key
        self.subsample_factor = subsample_factor
        self.sample_rate = sample_rate
        self.window_len = window_len
        self.step_len = step_len
        self.num_filters = num_filters
        self.peak_normalization = peak_normalization
        self.widen_frames = widen_frames
        self.max_len = max_len
        self.max_seqs = max_seqs

        self.rqmt = {"cpu": 4, "mem": 16, "time": 24}

        self.out_mean_table = self.output_path("mean_logmel.npz")
        self.out_duration_table = self.output_path("durations.npz")
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        """sis tasks"""
        yield Task("run", resume="run", rqmt=self.rqmt)

    def _align_dataset_dict(self) -> Dict[str, Any]:
        return {
            "class": "HDFDataset",
            "files": [p.get_path() for p in self.align_hdfs],
            "use_cache_manager": True,
        }

    def _labels(self) -> List[str]:
        """:return: the vocab, read from the first shard without touching its data"""
        import h5py

        with h5py.File(self.align_hdfs[0].get_path(), "r") as f:
            return [s.decode() if isinstance(s, bytes) else str(s) for s in f["labels"][:]]

    def run(self):
        """collect the tables"""
        import sys

        sys.path.insert(0, self.returnn_root.get_path())

        import json
        import numpy
        import torch
        from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy
        import returnn.frontend as rf
        from returnn.tensor import Tensor, Dim
        from returnn.datasets import init_dataset

        rf.select_backend_torch()

        labels = self._labels()
        num_labels = len(labels)
        blank_idx = num_labels - 1
        dim_f = self.num_filters
        batch_dim = Dim(1, name="batch")
        out_dim = Dim(dim_f, name="mel")

        # Pass one reads the alignment alone, so the audio is decoded once, in pass two.
        counts = numpy.zeros((num_labels,), dtype=numpy.int64)
        span_frames = numpy.zeros((num_labels,), dtype=numpy.int64)
        align_ds = init_dataset(self._align_dataset_dict())
        align_ds.init_seq_order(epoch=1)
        seq_idx = 0
        while align_ds.is_less_than_num_seqs(seq_idx):
            align_ds.load_seqs(seq_idx, seq_idx + 1)
            for idx, start, end in self._spans(align_ds.get_data(seq_idx, "data"), blank_idx):
                counts[idx] += 1
                span_frames[idx] += end - start
            seq_idx += 1
            if self.max_seqs is not None and seq_idx >= self.max_seqs:
                break
        del align_ds

        seen = counts > 0
        mean_mel = numpy.where(seen, span_frames / numpy.maximum(counts, 1), 0.0) * self.subsample_factor
        lengths = numpy.clip(numpy.rint(mean_mel), 1, self.max_len).astype(numpy.int64)
        lengths[~seen] = 0
        table_len = int(max(lengths.max(), 1))

        sums = numpy.zeros((num_labels, table_len, dim_f), dtype=numpy.float64)
        used = numpy.zeros((num_labels,), dtype=numpy.int64)

        def _log_mel(audio_np: numpy.ndarray) -> numpy.ndarray:
            if self.peak_normalization:
                peak = numpy.max(numpy.abs(audio_np))
                if peak != 0.0:
                    audio_np = audio_np / peak
            raw = torch.tensor(audio_np[None, :], dtype=torch.float32)
            time_dim = Dim(int(raw.shape[1]), name="time")
            src = Tensor("audio", dims=[batch_dim, time_dim], dtype="float32", raw_tensor=raw)
            feats, feats_dim = rf.audio.log_mel_filterbank_from_raw(
                src,
                in_spatial_dim=time_dim,
                out_dim=out_dim,
                sampling_rate=self.sample_rate,
                window_len=self.window_len,
                step_len=self.step_len,
            )
            return feats.copy_compatible_to_dims_raw([batch_dim, feats_dim, out_dim])[0].numpy()

        # MetaDataset pairs the two by seq tag, so neither side is held in memory.
        meta = {
            "class": "MetaDataset",
            "datasets": {"audio": instanciate_delayed_copy(self.dataset_dict), "align": self._align_dataset_dict()},
            "data_map": {"data": ("audio", self.audio_data_key), "alignment": ("align", "data")},
            "seq_order_control_dataset": "audio",
        }
        dataset = init_dataset(meta)
        dataset.init_seq_order(epoch=1)
        n_seqs = 0
        n_rate_mismatch = 0
        seq_idx = 0
        while dataset.is_less_than_num_seqs(seq_idx):
            dataset.load_seqs(seq_idx, seq_idx + 1)
            frames = dataset.get_data(seq_idx, "alignment")
            audio_np = numpy.asarray(dataset.get_data(seq_idx, "data"), dtype=numpy.float32)
            seq_idx += 1
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]
            feats = _log_mel(audio_np)  # [T, F]
            # The frame rate is the assumption everything downstream rests on,
            # so check it per sequence instead of trusting the configured factor.
            if abs(len(frames) * self.subsample_factor - feats.shape[0]) > self.subsample_factor:
                n_rate_mismatch += 1
                assert n_rate_mismatch < 100, (
                    f"{len(frames)} align frames x {self.subsample_factor} vs {feats.shape[0]} log-mel frames"
                )
                continue
            for idx, start, end in self._spans(frames, blank_idx):
                lo = start * self.subsample_factor
                hi = min(end * self.subsample_factor, feats.shape[0])
                if hi <= lo or lengths[idx] == 0:
                    continue
                block = feats[lo:hi].astype(numpy.float64)
                sums[idx, : lengths[idx]] += self._resample(block, int(lengths[idx]))
                used[idx] += 1
            n_seqs += 1
            if self.max_seqs is not None and n_seqs >= self.max_seqs:
                break

        means = numpy.zeros((num_labels, table_len, dim_f), dtype=numpy.float32)
        # An unseen label still has to be looked up at training time, so fall back to the global mean.
        global_mean = sums.sum(axis=(0, 1)) / max(int((used * lengths).sum()), 1)
        for i in range(num_labels):
            if used[i] > 0:
                means[i, : lengths[i]] = sums[i, : lengths[i]] / used[i]
            else:
                means[i, :] = global_mean
                lengths[i] = table_len
        durations = mean_mel.astype(numpy.float32)
        durations[~seen] = float(mean_mel[seen].mean()) if seen.any() else 0.0
        labels_np = numpy.array(labels, dtype=object)
        numpy.savez(self.out_mean_table.get_path(), means=means, lengths=lengths, labels=labels_np)
        numpy.savez(self.out_duration_table.get_path(), durations=durations, labels=labels_np)

        stats = {
            "n_seqs": n_seqs,
            "n_rate_mismatch": n_rate_mismatch,
            "n_labels_seen": int(seen.sum()),
            "n_labels": num_labels,
            "table_len": table_len,
            "n_labels_at_max_len": int((lengths >= self.max_len).sum()),
            "blank_duration_logmel_frames": float(durations[blank_idx]),
            "occurrences": {labels[i]: int(counts[i]) for i in range(num_labels) if counts[i] > 0},
            "config": {
                "audio_data_key": self.audio_data_key,
                "subsample_factor": self.subsample_factor,
                "sample_rate": self.sample_rate,
                "window_len": self.window_len,
                "step_len": self.step_len,
                "num_filters": self.num_filters,
                "peak_normalization": self.peak_normalization,
                "widen_frames": self.widen_frames,
                "max_len": self.max_len,
                "max_seqs": self.max_seqs,
            },
        }
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(stats, f, indent=2)
        print("done:", n_seqs, "seqs; table_len", table_len, ";", int(seen.sum()), "of", num_labels, "labels seen")

    @staticmethod
    def _resample(block: numpy.ndarray, out_len: int) -> numpy.ndarray:
        """Linear resample along time.

        :param block: [n, num_filters]
        :return: [out_len, num_filters]
        """
        import numpy

        n = block.shape[0]
        if n == out_len:
            return block
        pos = numpy.linspace(0.0, n - 1, out_len)
        lo = numpy.floor(pos).astype(numpy.int64)
        hi = numpy.minimum(lo + 1, n - 1)
        frac = (pos - lo)[:, None]
        return block[lo] * (1.0 - frac) + block[hi] * frac

    def _spans(self, frames: numpy.ndarray, blank_idx: int) -> List[Tuple[int, int, int]]:
        """Unit spans in alignment frames, after widening each subword into the surrounding blank.

        A gap between two subwords is shared, so each side takes at most half of it;
        at a sequence edge the whole gap is available.
        The spans tile the sequence: contiguous, no gap, no overlap.

        :return: list of (label index, start frame, end frame), silence carried as the blank index
        """
        n = len(frames)
        runs = []  # [label, start, end]
        i = 0
        while i < n:
            j = i
            while j < n and frames[j] == frames[i]:
                j += 1
            runs.append([int(frames[i]), i, j])
            i = j
        for k, run in enumerate(runs):
            if run[0] == blank_idx:
                continue
            for side in (-1, 1):
                m = k + side
                if not 0 <= m < len(runs) or runs[m][0] != blank_idx:
                    continue
                gap = runs[m][2] - runs[m][1]
                far = m + side
                shared = 0 <= far < len(runs) and runs[far][0] != blank_idx
                take = min(self.widen_frames, gap // 2 if shared else gap)
                if not take:
                    continue
                if side < 0:
                    run[1] -= take
                    runs[m][2] -= take
                else:
                    run[2] += take
                    runs[m][1] += take
        return [(label, start, end) for label, start, end in runs if end > start]
