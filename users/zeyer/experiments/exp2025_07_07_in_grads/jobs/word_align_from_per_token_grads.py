"""Word alignment from per-token grad scores."""

from typing import Optional, Any, Dict, List
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class WordAlignFromPerTokenGradsJob(Job):
    """Align per-token grad scores then collapse to word boundaries; compute WBE.

    Counterpart to :class:`exp2025_05_05_align.CalcAlignmentMetricsJob` for the
    per-token HDF schema written by :class:`ExtractInGradsPerTokenJob`. Differs
    in:

    1. The grad data is shaped ``[num_tokens, num_timeframes]`` per chunk (not
       ``[num_words, num_timeframes]``).
    2. After running the Aligner at the token level, token boundaries are
       collapsed to word boundaries using the ``num_tokens_per_word`` stream:
       ``word_w_start = first_token_in_word.start``,
       ``word_w_end = last_token_in_word.end``.
    3. WBE is then computed against the dataset's reference word boundaries
       (same metric as :class:`CalcAlignmentMetricsJob`).

    Single-chunk only for now (TIMIT short-form). Buckeye / long-form chunked
    alignment would need a chunked variant analogous to
    :class:`CalcChunkedAlignmentMetricsJob`.
    """

    # audio_energy_pow=0.0 (no filter) excluded so existing aligns keep their hash.
    __sis_hash_exclude__ = {
        "audio_energy_pow": 0.0,
        "blank_grad_zscore_kappa": 0.0,
        "blank_silence_energy_scale": 0.0,
        "local_norm_window_s": None,
        "boundary_source": "word_detail",
    }

    # v2: emit the richer metric set (acc@collar, edge/interior, start/end MAE) via align_metrics.
    __sis_version__ = 2

    def __init__(
        self,
        *,
        returnn_root: Optional[tk.Path] = None,
        grad_score_hdf: tk.Path,
        grad_score_key: str,
        dataset_dir: tk.Path,
        dataset_key: str,
        dataset_offset_factors: int,
        align_opts: Dict[str, Any],
        audio_energy_pow: float = 0.0,
        blank_grad_zscore_kappa: float = 0.0,
        blank_silence_energy_scale: float = 0.0,
        local_norm_window_s: Optional[float] = None,
        boundary_source: str = "word_detail",
    ):
        """
        :param grad_score_hdf: from :class:`ExtractInGradsPerTokenJob`. Must
            have streams ``num_input_frames``, ``num_tokens``,
            ``num_tokens_per_word`` (single chunk per seq).
        :param grad_score_key: usually ``"data"``.
        :param align_opts: see :class:`exp2025_05_05_align.Aligner`.
        """
        super().__init__()
        self.returnn_root = returnn_root
        self.grad_score_hdf = grad_score_hdf
        self.grad_score_key = grad_score_key
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.dataset_offset_factors = dataset_offset_factors
        self.align_opts = align_opts
        self.audio_energy_pow = float(audio_energy_pow)
        self.blank_grad_zscore_kappa = float(blank_grad_zscore_kappa)
        self.blank_silence_energy_scale = float(blank_silence_energy_scale)
        self.local_norm_window_s = local_norm_window_s
        self.boundary_source = boundary_source
        assert boundary_source in ("word_detail", "phonetic_detail"), boundary_source
        assert not (self.blank_grad_zscore_kappa and self.blank_silence_energy_scale), (
            "blank_grad_zscore_kappa and blank_silence_energy_scale are mutually exclusive"
        )

        self.out_wbe = self.output_var("wbe.txt")
        # Richer metrics (see align_metrics): tolerance-collar accuracy,
        # edge-vs-interior WBE, start/end MAE. out_metrics holds the full dict.
        self.out_metrics = self.output_var("metrics.txt")
        self.out_acc50 = self.output_var("acc50.txt")
        self.out_interior_wbe = self.output_var("interior_wbe.txt")
        self.out_edge_wbe = self.output_var("edge_wbe.txt")

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
        )

        grad_score_hdf_ds = HDFDataset([self.grad_score_hdf.get_path()])
        grad_score_hdf_ds.initialize()
        grad_score_hdf_ds.init_seq_order(epoch=1)

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset: {ds}")
        print("Dataset keys:", ds.keys())
        print("Using key:", self.dataset_key)
        print("Num seqs:", len(ds[self.dataset_key]))

        aligner = Aligner(**self.align_opts)

        utt_errs = []

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            grad_score_hdf_ds.load_seqs(seq_idx, seq_idx + 1)

            samplerate = data["audio"]["sampling_rate"]
            num_audio_samples = len(data["audio"]["array"])
            audio_len_secs = num_audio_samples / samplerate

            num_input_frames = grad_score_hdf_ds.get_data(seq_idx, "num_input_frames")
            num_tokens = grad_score_hdf_ds.get_data(seq_idx, "num_tokens")
            num_tokens_per_word = grad_score_hdf_ds.get_data(seq_idx, "num_tokens_per_word")
            assert num_input_frames.shape == (1, 1), f"single-chunk only, got {num_input_frames.shape=}"
            assert num_tokens.shape == (1, 1), f"single-chunk only, got {num_tokens.shape=}"
            chunk_num_timeframes = int(num_input_frames[0, 0])
            chunk_num_tokens = int(num_tokens[0, 0])
            tokens_per_word = num_tokens_per_word.reshape(-1).astype(int)
            assert tokens_per_word.sum() == chunk_num_tokens, f"{tokens_per_word.sum()=} != {chunk_num_tokens=}"
            num_words = len(tokens_per_word)

            grad_mat = grad_score_hdf_ds.get_data(seq_idx, self.grad_score_key)
            assert grad_mat.shape == (chunk_num_tokens * chunk_num_timeframes, 1), (
                f"{grad_mat.shape=} {chunk_num_tokens=} {chunk_num_timeframes=}"
            )
            grad_mat = grad_mat.reshape(chunk_num_tokens, chunk_num_timeframes)
            secs_per_timeframe = audio_len_secs / chunk_num_timeframes

            # Smoothed audio-RMS-energy envelope on the grad frame grid, in [0,1].
            # Used both to energy-weight tokens (audio_energy_pow)
            # and to drive an explicit energy/VAD-derived silence emission for the blank row.
            e = None
            if self.audio_energy_pow or self.blank_silence_energy_scale:
                audio = np.asarray(data["audio"]["array"], dtype=np.float64)
                win = max(int(0.025 * samplerate), 1)
                hann = np.hanning(win)
                hann = hann / hann.sum()
                env = np.sqrt(np.convolve(audio * audio, hann, mode="same") + 1e-9)
                centers = ((np.arange(chunk_num_timeframes) + 0.5) / chunk_num_timeframes * (len(env) - 1)).astype(int)
                e = env[centers]
                e = e / (e.max() + 1e-9)

            blank_override = None
            if self.blank_grad_zscore_kappa or self.blank_silence_energy_scale:
                # Per-frame token score distribution on the Aligner's transform
                # (log-softmax-over-time, per token),
                # computed from the ORIGINAL (pre-energy) grads
                # so the blank still calibrates in silence.
                _s = np.log(grad_mat.astype(np.float64) + 1e-12)
                _s = _s - max(float(_s.max()), 0.0)
                _m = _s.max(axis=1, keepdims=True)
                _s = _s - _m - np.log(np.sum(np.exp(_s - _m), axis=1, keepdims=True))
                _mean_t, _std_t = _s.mean(0), _s.std(0)
                if self.blank_grad_zscore_kappa:
                    # Self-calibrating constant-margin blank: beta_t = mean_t + kappa*std_t.
                    # A token wins iff its within-frame z-score over tokens exceeds kappa
                    # (decoupled from the energy weighting).
                    blank_override = _mean_t + self.blank_grad_zscore_kappa * _std_t
                else:
                    # Explicit silence label (whisper-timestamped-style VAD analog):
                    # an energy-driven blank emission, beta_t = mean_t - scale*z(E_t)*std_t,
                    # with z(E) the per-frame energy z-score over time.
                    # In low-energy (silence) frames z(E)<0,
                    # so the blank is raised above the token mean and owns the frame;
                    # in speech z(E)>0, so the blank drops and tokens win.
                    # Self-calibrating via the token spread std_t.
                    _ze = (e - e.mean()) / (e.std() + 1e-9)
                    blank_override = _mean_t - self.blank_silence_energy_scale * _ze * _std_t

            if self.audio_energy_pow:
                # Multiply each token's saliency by the energy envelope^pow
                # (a silence-suppressing filter; conceptually grad*input at the raw-waveform level).
                # Works at any frame rate.
                grad_mat = grad_mat * (e[None, :] ** self.audio_energy_pow)

            if self.local_norm_window_s:
                # Local emission normalization: divide each token row by a sliding-window mean, with
                # the window given in SECONDS (-> frames per seq, so it is frame-rate-comparable across
                # models). A ~1.5 s window gently de-trends without the noise blow-up of a tiny window.
                from scipy.ndimage import uniform_filter1d

                _w = max(3, int(round(self.local_norm_window_s / secs_per_timeframe)))
                grad_mat = grad_mat / (uniform_filter1d(grad_mat, _w, axis=1, mode="nearest") + 1e-9)

            align_token_start_ends = aligner.align(grad_mat, blank_override=blank_override)
            assert len(align_token_start_ends) == chunk_num_tokens

            # Collapse to per-word boundaries.
            align_word_start_ends = []
            cursor = 0
            for w, k in enumerate(tokens_per_word):
                first = align_token_start_ends[cursor]
                last = align_token_start_ends[cursor + k - 1]
                align_word_start_ends.append((first[0] * secs_per_timeframe, last[1] * secs_per_timeframe))
                cursor += k
            assert cursor == chunk_num_tokens

            print(
                f"** seq {seq_idx}, {num_words=} {chunk_num_tokens=}"
                f" {audio_len_secs=} {num_audio_samples=} {secs_per_timeframe=}"
            )

            _bsrc = getattr(self, "boundary_source", "word_detail")
            words: List[str] = data[_bsrc]["utterance"]
            ref_word_starts: List[float] = data[_bsrc]["start"]
            ref_word_ends: List[float] = data[_bsrc]["stop"]
            assert num_words == len(words) == len(ref_word_starts) == len(ref_word_ends), (
                f"{num_words=} {len(words)=} {len(ref_word_starts)=}"
            )
            ref_word_start_ends = [
                tuple(t * self.dataset_offset_factors / samplerate for t in ts)
                for ts in zip(ref_word_starts, ref_word_ends)
            ]

            utt_err = per_utt_boundary_errors(align_word_start_ends, ref_word_start_ends)
            print("  WBE:", float(np.mean(utt_err["wbe"])))
            utt_errs.append(utt_err)

        metrics = aggregate_corpus(utt_errs)
        print("CORPUS METRICS:", metrics)
        self.out_wbe.set(metrics["wbe"])
        self.out_metrics.set(metrics)
        self.out_acc50.set(metrics["acc_50ms"])
        self.out_interior_wbe.set(metrics["interior_wbe"])
        self.out_edge_wbe.set(metrics["edge_wbe"])
