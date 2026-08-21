"""
LibriSpeech MFA alignments (HF ``gilkeyio/librispeech-alignments``)
and a per-phone mean log-mel table computed from them.

The HF dataset ships the LibriSpeech audio together with MFA word/phone alignments
(``english_us_arpa``: stressed ARPAbet labels like ``IH1``, per-phone ``start``/``end`` in seconds).
Explicit silence tokens were removed in the HF formatting,
but the per-phone timestamps are kept,
so silence is recoverable as the complement (gaps between phone intervals).

The mean table is the "table-lookup TTS" for the pseudo-speech-encoder experiments:
per phone of the GlowTTS phoneme vocab (stress-stripped ARPAbet, exact 1:1),
the mean 100Hz log-mel frame over all aligned frames of real LibriSpeech audio,
with the identical front-end the ASR uses in the ``asr_logmel`` setting
(peak-normalized waveform -> ``rf.audio.log_mel_filterbank_from_raw``).
Silence (gaps) -> ``[space]`` (also used as the blank row); no frames -> global mean.
"""

from __future__ import annotations

from typing import Sequence
from functools import cache

from sisyphus import Job, Task, tk

from i6_core.datasets.huggingface import DownloadAndPrepareHuggingFaceDatasetJob


__all__ = [
    "get_librispeech_mfa_alignments_dir",
    "get_mfa_phone_mean_logmel_table",
    "ComputeMfaPhoneMeanLogMelJob",
    "get_mfa_phone_duration_table",
    "ComputeMfaPhoneDurationStatsJob",
]


# Shared HF cache on project storage. The home dir has a small quota (~20GB) and
# ~/.cache/huggingface is deliberately a broken symlink so accidental use fails loudly;
# every HF download must go through HF_HOME below (DEFAULT_ENVIRONMENT_SET in settings.py
# sets the same path, but local-engine tasks have been observed not to get it, so the
# download job here sets it explicitly).
_HF_HOME = "/e/project1/spell/common_hf_home"


class _DownloadAndPrepareHuggingFaceDatasetWithHfHomeJob(DownloadAndPrepareHuggingFaceDatasetJob):
    """Like the base job, but with HF_HOME explicitly set (see _HF_HOME comment above)."""

    def run(self):
        import os

        os.makedirs(_HF_HOME, exist_ok=True)
        os.environ["HF_HOME"] = _HF_HOME
        super().run()


@cache
def get_librispeech_mfa_alignments_dir() -> tk.Path:
    """:return: prepared HF dataset dir (all splits, incl. audio; ~70GB)"""
    job = _DownloadAndPrepareHuggingFaceDatasetWithHfHomeJob(
        "gilkeyio/librispeech-alignments",
        revision="0daa1eb43dda38ee6ce752e785555380e5628f5c",
        time_rqmt=12,
        mem_rqmt=8,
        cpu_rqmt=2,
        mini_task=False,
    )
    job.add_alias("datasets/LibriSpeech/mfa_alignments_hf")
    return job.out_dir


@cache
def get_mfa_phone_mean_logmel_table() -> ComputeMfaPhoneMeanLogMelJob:
    """:return: job computing the per-phone mean log-mel table for the GlowTTS phoneme vocab"""
    from i6_experiments.users.zeyer.external_models.glow_tts import get_glow_tts_phoneme_vocab
    from i6_experiments.users.zeyer import tools_paths

    job = ComputeMfaPhoneMeanLogMelJob(
        dataset_dir=get_librispeech_mfa_alignments_dir(),
        returnn_root=tools_paths.get_returnn_root(),
        phoneme_vocab=get_glow_tts_phoneme_vocab(),
    )
    job.add_alias("datasets/LibriSpeech/mfa_phone_mean_logmel")
    tk.register_output("datasets/LibriSpeech/mfa_phone_mean_logmel.npz", job.out_mean_table)
    tk.register_output("datasets/LibriSpeech/mfa_phone_mean_logmel_stats.json", job.out_stats)
    return job


class ComputeMfaPhoneMeanLogMelJob(Job):
    """
    Computes the per-phone mean 100Hz log-mel frame over real LibriSpeech audio,
    using the MFA alignments (see module docstring).

    Front-end (must match the ASR ``asr_logmel`` setting, i.e. the
    ``rf.audio.log_mel_filterbank_from_raw`` defaults in the AED model):
    peak-normalized waveform, 16kHz, window 25ms, step 10ms, 80 mel filters, log10.

    Mapping: MFA stressed ARPAbet -> stress-stripped (``IH1`` -> ``IH``), exact 1:1 to the
    39 phones of the GlowTTS phoneme vocab; ``spn`` -> ``[UNKNOWN]``;
    gaps between phone intervals (= silence; MFA ``sil``/``sp`` were dropped in the HF formatting)
    -> ``[space]``. Vocab labels without any frames get the global mean frame.

    Output ``out_mean_table``: npz with ``means`` [vocab_size, 80] (float32, vocab order)
    and ``labels`` [vocab_size] (the vocab labels, for verification);
    ``out_stats``: json with per-label frame counts and config echo.
    """

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        returnn_root: tk.Path,
        phoneme_vocab: tk.Path,
        splits: Sequence[str] = ("train_clean_100", "train_clean_360", "train_other_500"),
        sample_rate: int = 16_000,
        window_len: float = 0.025,
        step_len: float = 0.010,
        num_filters: int = 80,
        peak_normalization: bool = True,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.returnn_root = returnn_root
        self.phoneme_vocab = phoneme_vocab
        self.splits = tuple(splits)
        self.sample_rate = sample_rate
        self.window_len = window_len
        self.step_len = step_len
        self.num_filters = num_filters
        self.peak_normalization = peak_normalization

        self.rqmt = {"cpu": 4, "mem": 16, "time": 24}

        self.out_mean_table = self.output_path("mean_logmel.npz")
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        import sys

        sys.path.insert(0, self.returnn_root.get_path())

        import json
        import re
        import numpy
        import torch
        from datasets import load_from_disk
        import returnn.frontend as rf
        from returnn.tensor import Tensor, Dim, batch_dim
        from returnn.datasets.util.vocabulary import Vocabulary

        rf.select_backend_torch()
        # standalone usage (no engine): the global batch dim needs an explicit size for the rf ops
        batch_dim.dyn_size_ext = rf.convert_to_tensor(torch.tensor(1, dtype=torch.int32), dims=[])

        vocab = Vocabulary(self.phoneme_vocab.get_path(), unknown_label="[UNKNOWN]")
        labels = vocab.labels
        label_to_idx = {l: i for i, l in enumerate(labels)}
        silence_idx = label_to_idx["[space]"]
        unknown_idx = label_to_idx["[UNKNOWN]"]
        num_labels = len(labels)
        dim_f = self.num_filters

        sums = numpy.zeros((num_labels, dim_f), dtype=numpy.float64)
        counts = numpy.zeros((num_labels,), dtype=numpy.int64)
        n_seqs = 0
        unmapped = {}

        out_dim = Dim(dim_f, name="mel")

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

        ds_all = load_from_disk(self.dataset_dir.get_path())
        n_len_mismatch = 0
        for split in self.splits:
            ds = ds_all[split]
            for ex in ds:
                audio = ex["audio"]["array"]
                assert ex["audio"]["sampling_rate"] == self.sample_rate
                audio_np = numpy.asarray(audio, dtype=numpy.float32)
                audio_dur = len(audio_np) / self.sample_rate
                # Consistency check: the alignment must fit into the audio
                # (catches audio/alignment pairing or time-unit errors).
                last_end = max((ph["end"] for ph in ex["phonemes"]), default=0.0)
                if last_end > audio_dur + 0.05:
                    n_len_mismatch += 1
                    assert n_len_mismatch < 100, (
                        f"too many alignment/audio length mismatches, e.g. {ex['id']}: "
                        f"alignment end {last_end} vs audio duration {audio_dur}"
                    )
                    continue
                feats = _log_mel(audio_np)  # [T, F]
                # Front-end consistency: number of feature frames must match the audio duration.
                assert abs(feats.shape[0] * self.step_len - audio_dur) <= self.window_len + 2 * self.step_len, (
                    f"{ex['id']}: {feats.shape[0]} feature frames vs audio duration {audio_dur}s"
                )
                # frame t covers [t*step, t*step+window); assign by frame center time
                t_centers = (numpy.arange(feats.shape[0]) * self.step_len) + self.window_len / 2
                frame_label = numpy.full((feats.shape[0],), silence_idx, dtype=numpy.int64)
                for ph in ex["phonemes"]:
                    base = re.sub(r"\d+$", "", ph["phoneme"])
                    if base in label_to_idx:
                        idx = label_to_idx[base]
                    elif base == "spn":
                        idx = unknown_idx
                    else:
                        unmapped[base] = unmapped.get(base, 0) + 1
                        continue
                    sel = (t_centers >= ph["start"]) & (t_centers < ph["end"])
                    frame_label[sel] = idx
                numpy.add.at(sums, frame_label, feats.astype(numpy.float64))
                numpy.add.at(counts, frame_label, 1)
                n_seqs += 1

        global_mean = sums.sum(axis=0) / max(int(counts.sum()), 1)
        means = numpy.zeros((num_labels, dim_f), dtype=numpy.float32)
        for i in range(num_labels):
            means[i] = (sums[i] / counts[i]) if counts[i] > 0 else global_mean
        numpy.savez(
            self.out_mean_table.get_path(),
            means=means,
            labels=numpy.array(labels, dtype=object),
        )
        stats = {
            "n_seqs": n_seqs,
            "n_len_mismatch": n_len_mismatch,
            "frame_counts": {labels[i]: int(counts[i]) for i in range(num_labels)},
            "unmapped": unmapped,
            "splits": list(self.splits),
            "config": {
                "sample_rate": self.sample_rate,
                "window_len": self.window_len,
                "step_len": self.step_len,
                "num_filters": self.num_filters,
                "peak_normalization": self.peak_normalization,
            },
        }
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(stats, f, indent=2)
        print("done:", n_seqs, "seqs; unmapped:", unmapped)


@cache
def get_mfa_phone_duration_table() -> ComputeMfaPhoneDurationStatsJob:
    """:return: job computing the per-phone duration table (in 10ms frames) for the GlowTTS phoneme vocab"""
    from i6_experiments.users.zeyer.external_models.glow_tts import get_glow_tts_phoneme_vocab
    from i6_experiments.users.zeyer import tools_paths

    job = ComputeMfaPhoneDurationStatsJob(
        dataset_dir=get_librispeech_mfa_alignments_dir(),
        returnn_root=tools_paths.get_returnn_root(),
        phoneme_vocab=get_glow_tts_phoneme_vocab(),
    )
    job.add_alias("datasets/LibriSpeech/mfa_phone_durations")
    tk.register_output("datasets/LibriSpeech/mfa_phone_durations.npz", job.out_duration_table)
    tk.register_output("datasets/LibriSpeech/mfa_phone_durations_stats.json", job.out_stats)
    return job


class ComputeMfaPhoneDurationStatsJob(Job):
    """
    Per-phone duration statistics (in 10ms frames) from the MFA alignments,
    for the pseudo-speech encoder's duration sampling.
    Real durations are right-skewed and vary ~3.4x across phones, which a uniform range cannot represent;
    sample ``median[phone] * exp(sigma * N(0,1))`` instead, with a shared sigma ~0.45.

    Drops the audio column before iterating, so it is cheap
    and avoids torchcodec, whose ffmpeg libs do not load on this cluster.

    Mapping matches :class:`ComputeMfaPhoneMeanLogMelJob`:
    stress-stripped ARPAbet (``IH1`` -> ``IH``), and ``spn`` -> ``[UNKNOWN]``.
    Silence is not a phone label here, so the gaps between phone intervals go to the ``[space]`` row.

    Output ``out_duration_table``: npz with ``medians``/``means``/``counts``/``labels``, all [vocab_size].
    ``out_stats``: json with the same per label, plus a config echo.
    """

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        returnn_root: tk.Path,
        phoneme_vocab: tk.Path,
        splits: Sequence[str] = ("train_clean_100", "train_clean_360", "train_other_500"),
        step_len: float = 0.010,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.returnn_root = returnn_root
        self.phoneme_vocab = phoneme_vocab
        self.splits = tuple(splits)
        self.step_len = step_len

        self.rqmt = {"cpu": 2, "mem": 8, "time": 4}

        self.out_duration_table = self.output_path("phone_durations.npz")
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        import sys

        sys.path.insert(0, self.returnn_root.get_path())

        import json
        import re
        import numpy
        from datasets import load_from_disk
        from returnn.datasets.util.vocabulary import Vocabulary

        vocab = Vocabulary(self.phoneme_vocab.get_path(), unknown_label="[UNKNOWN]")
        labels = vocab.labels
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        silence_idx = label_to_idx["[space]"]
        unknown_idx = label_to_idx["[UNKNOWN]"]
        num_labels = len(labels)

        durs = [[] for _ in range(num_labels)]
        unmapped = {}
        n_seqs = 0

        ds_all = load_from_disk(self.dataset_dir.get_path())
        for split in self.splits:
            # Drop everything but the alignment: decoding audio would pull in torchcodec.
            ds = ds_all[split].select_columns(["phonemes"])
            for ex in ds:
                prev_end = None
                for ph in ex["phonemes"]:
                    if prev_end is not None and ph["start"] - prev_end > 1e-6:
                        # gap between phones = silence -> the [space] row
                        durs[silence_idx].append((ph["start"] - prev_end) / self.step_len)
                    prev_end = ph["end"]
                    base = re.sub(r"\d+$", "", ph["phoneme"])
                    if base in label_to_idx:
                        idx = label_to_idx[base]
                    elif base == "spn":
                        idx = unknown_idx
                    else:
                        unmapped[base] = unmapped.get(base, 0) + 1
                        continue
                    durs[idx].append((ph["end"] - ph["start"]) / self.step_len)
                n_seqs += 1

        counts = numpy.array([len(d) for d in durs], dtype="int64")
        # Global median over the speech phones only,
        # as the fallback for labels never seen: [start]/[end]/[blank], and any phone absent.
        speech = numpy.concatenate(
            [numpy.asarray(d) for i, d in enumerate(durs) if d and i not in (silence_idx, unknown_idx)]
        )
        global_median = float(numpy.median(speech))
        global_mean = float(speech.mean())

        medians = numpy.full((num_labels,), global_median, dtype="float32")
        means = numpy.full((num_labels,), global_mean, dtype="float32")
        for i, d in enumerate(durs):
            if d:
                a = numpy.asarray(d)
                medians[i] = numpy.median(a)
                means[i] = a.mean()

        numpy.savez(
            self.out_duration_table.get_path(),
            medians=medians,
            means=means,
            counts=counts,
            labels=numpy.array(labels),
        )
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(
                {
                    "n_seqs": n_seqs,
                    "global_median": global_median,
                    "global_mean": global_mean,
                    "per_label": {
                        labels[i]: {"median": float(medians[i]), "mean": float(means[i]), "count": int(counts[i])}
                        for i in range(num_labels)
                    },
                    "unmapped": unmapped,
                    "splits": list(self.splits),
                    "config": {"step_len": self.step_len},
                },
                f,
                indent=2,
            )
