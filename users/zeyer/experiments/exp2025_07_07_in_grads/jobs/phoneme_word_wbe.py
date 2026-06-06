"""Word-level WBE for the phoneme grad-align.

The phoneme :class:`ExtractInGradsPerTokenJob` (target_source="phonetic_detail")
emits one grad row per TIMIT phone; :class:`WordAlignFromPerTokenGradsJob`
(boundary_source="phonetic_detail") scores PHONE-boundary WBE. This job reuses
the same grad HDF + the same DP aligner, but COLLAPSES the per-phone predicted
boundaries to per-word boundaries (via the TIMIT phone<->word time overlap) and
reports WORD-boundary WBE -- directly comparable to the cross-model word table.

Self-contained (does not subclass the word align job) so adding it never
re-hashes any existing align. Replicates only the alignment path the phoneme
aligns use (plain saliency, optional energy weighting + energy-silence blank).
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from sisyphus import Job, Task, tk

from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class PhonemeAlignWordWbeJob(Job):
    """Collapse the phoneme grad-align to per-word boundaries -> word WBE."""

    __sis_hash_exclude__ = {"audio_energy_pow": 0.0, "blank_silence_energy_scale": 0.0}

    def __init__(
        self,
        *,
        grad_score_hdf: tk.Path,
        grad_score_key: str,
        dataset_dir: tk.Path,
        dataset_key: str,
        dataset_offset_factors: int,
        align_opts: Dict[str, Any],
        audio_energy_pow: float = 0.0,
        blank_silence_energy_scale: float = 0.0,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.grad_score_hdf = grad_score_hdf
        self.grad_score_key = grad_score_key
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.dataset_offset_factors = dataset_offset_factors
        self.align_opts = align_opts
        self.audio_energy_pow = float(audio_energy_pow)
        self.blank_silence_energy_scale = float(blank_silence_energy_scale)
        self.returnn_root = returnn_root
        self.out_word_wbe = self.output_var("word_wbe.txt")
        self.out_word_metrics = self.output_var("word_metrics.txt")
        self.out_phone_wbe = self.output_var("phone_wbe.txt")  # cross-check vs the align job

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 10, "time": 5})

    def run(self):
        import os
        import sys
        import numpy as np

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.hdf import HDFDataset
        from i6_experiments.users.zeyer.experiments.exp2025_05_05_align import Aligner
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_metrics import (
            per_utt_boundary_errors,
            aggregate_corpus,
            collapse_phones_to_words,
        )
        from datasets import load_dataset

        grad_ds = HDFDataset([self.grad_score_hdf.get_path()])
        grad_ds.initialize()
        grad_ds.init_seq_order(epoch=1)

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print("Using key:", self.dataset_key, "num seqs:", len(ds[self.dataset_key]))

        aligner = Aligner(**self.align_opts)
        phone_errs, word_errs = [], []

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            grad_ds.load_seqs(seq_idx, seq_idx + 1)
            samplerate = data["audio"]["sampling_rate"]
            num_audio_samples = len(data["audio"]["array"])
            audio_len_secs = num_audio_samples / samplerate

            num_input_frames = int(grad_ds.get_data(seq_idx, "num_input_frames")[0, 0])
            num_tokens = int(grad_ds.get_data(seq_idx, "num_tokens")[0, 0])
            tokens_per_word = grad_ds.get_data(seq_idx, "num_tokens_per_word").reshape(-1).astype(int)
            assert tokens_per_word.sum() == num_tokens, f"{tokens_per_word.sum()=} {num_tokens=}"
            # phoneme extract => one token per phone (each phone is its own "word" unit).
            assert (tokens_per_word == 1).all(), "expected 1 token/phone for the phoneme grad HDF"

            grad_mat = grad_ds.get_data(seq_idx, self.grad_score_key).reshape(num_tokens, num_input_frames)
            secs_per_tf = audio_len_secs / num_input_frames

            e = None
            if self.audio_energy_pow or self.blank_silence_energy_scale:
                audio = np.asarray(data["audio"]["array"], dtype=np.float64)
                win = max(int(0.025 * samplerate), 1)
                hann = np.hanning(win)
                hann = hann / hann.sum()
                env = np.sqrt(np.convolve(audio * audio, hann, mode="same") + 1e-9)
                centers = ((np.arange(num_input_frames) + 0.5) / num_input_frames * (len(env) - 1)).astype(int)
                e = env[centers]
                e = e / (e.max() + 1e-9)

            blank_override = None
            if self.blank_silence_energy_scale:
                _s = np.log(grad_mat.astype(np.float64) + 1e-12)
                _s = _s - max(float(_s.max()), 0.0)
                _m = _s.max(axis=1, keepdims=True)
                _s = _s - _m - np.log(np.sum(np.exp(_s - _m), axis=1, keepdims=True))
                _mean_t, _std_t = _s.mean(0), _s.std(0)
                _ze = (e - e.mean()) / (e.std() + 1e-9)
                blank_override = _mean_t - self.blank_silence_energy_scale * _ze * _std_t

            if self.audio_energy_pow:
                grad_mat = grad_mat * (e[None, :] ** self.audio_energy_pow)

            token_se = aligner.align(grad_mat, blank_override=blank_override)
            assert len(token_se) == num_tokens
            pred_phone_se = [(s * secs_per_tf, e_ * secs_per_tf) for s, e_ in token_se]

            ph = data["phonetic_detail"]
            wd = data["word_detail"]
            scale = self.dataset_offset_factors / samplerate
            # phone-boundary WBE (cross-check vs the align job's number).
            ref_phone_se = [(s * scale, e_ * scale) for s, e_ in zip(ph["start"], ph["stop"])]
            phone_errs.append(per_utt_boundary_errors(pred_phone_se, ref_phone_se))
            # word-boundary WBE (the comparable metric).
            pred_word_se, ref_word_se = collapse_phones_to_words(
                pred_phone_se, ph["start"], ph["stop"], wd["start"], wd["stop"], scale
            )
            word_errs.append(per_utt_boundary_errors(pred_word_se, ref_word_se))
            if seq_idx % 200 == 0:
                print(f"seq {seq_idx}: {len(ref_word_se)} words, {num_tokens} phones", flush=True)

        word_metrics = aggregate_corpus(word_errs)
        phone_metrics = aggregate_corpus(phone_errs)
        print("WORD METRICS:", word_metrics)
        print("PHONE METRICS:", phone_metrics)
        self.out_word_wbe.set(word_metrics["wbe"])
        self.out_word_metrics.set(word_metrics)
        self.out_phone_wbe.set(phone_metrics["wbe"])
