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
        for split in self.splits:
            ds = ds_all[split]
            for ex in ds:
                audio = ex["audio"]["array"]
                assert ex["audio"]["sampling_rate"] == self.sample_rate
                feats = _log_mel(numpy.asarray(audio, dtype=numpy.float32))  # [T, F]
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
