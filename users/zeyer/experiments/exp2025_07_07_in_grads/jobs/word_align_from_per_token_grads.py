"""Word alignment from per-token grad scores."""

from typing import Optional, Any, Dict, List
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


def _openai_dtw_path(cost):
    """openai-whisper's DTW (whisper/timing.py dtw_cpu), copied. cost: [N,M]; returns the min-cost
    monotonic path as a list of (text_idx, time_idx). Transitions: diagonal / up (text advances,
    no time) / left (time advances, same text). Allows zero-duration tokens, unlike our Viterbi."""
    import numpy as np

    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    tr = np.zeros((n + 1, m + 1), dtype=np.int8)
    d[0, 0] = 0.0
    tr[0, :] = 2  # came from left
    tr[:, 0] = 1  # came from up
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c0, c1, c2 = d[i - 1, j - 1], d[i - 1, j], d[i, j - 1]
            if c0 <= c1 and c0 <= c2:
                d[i, j] = cost[i - 1, j - 1] + c0
                tr[i, j] = 0
            elif c1 <= c2:
                d[i, j] = cost[i - 1, j - 1] + c1
                tr[i, j] = 1
            else:
                d[i, j] = cost[i - 1, j - 1] + c2
                tr[i, j] = 2
    i, j = n, m
    path = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        t = tr[i, j]
        if t == 0:
            i, j = i - 1, j - 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def _dtw_spans_from_cost(cost):
    """Run the whisper dtw_cpu recursion on `cost` [S,T]
    and collapse the path to per-token [start, end) frame spans
    (a zero-duration token gets a zero-width span at the running cursor)."""
    import numpy as np

    s = cost.shape[0]
    path = _openai_dtw_path(cost)
    text = np.array([p[0] for p in path])
    time = np.array([p[1] for p in path])
    spans = []
    for i in range(s):
        fr = time[text == i]
        if len(fr) == 0:
            prev = spans[-1][1] if spans else 0
            spans.append((prev, prev))
        else:
            spans.append((int(fr.min()), int(fr.max()) + 1))
    return spans


def _openai_dtw_align(score_matrix):
    """Token start/end frames via openai's DTW on the score matrix. Same non-blank score transform
    (log-softmax over time) as our Aligner, so only the DP differs."""
    import numpy as np
    from scipy.special import logsumexp

    logp = score_matrix - logsumexp(score_matrix, axis=1, keepdims=True)  # [S,T] log-softmax over time
    return _dtw_spans_from_cost(-logp.astype(np.float64))  # cost = -logprob (path maximizes prob)


def _true_dtw_align(score_matrix, apply_log: bool):
    """True DTW recursion (whisper dtw_cpu) with a configurable cost.
    apply_log=True -> cost = -log-softmax-over-time
    (our log-prob variant: every extra path cell costs more,
    so the DP never takes a vertical step -> equivalent to our monotonic alignment).
    apply_log=False -> cost = -score_matrix directly
    (whisper's dtw(-matrix); only non-degenerate when the matrix is signed, e.g. after token z-norm
    -- otherwise every vertical step improves it)."""
    import numpy as np
    from scipy.special import logsumexp

    m = score_matrix.astype(np.float64)
    if apply_log:
        m = m - logsumexp(m, axis=1, keepdims=True)
    return _dtw_spans_from_cost(-m)


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
        "with_ref_metrics": True,
        "with_confidence": False,
        "confidence_version": 1,
        "word_topology": False,
        "dtw_no_blank": False,
        "whisper_dtw": False,
        "attn_token_zscore": False,
        "median_filter_width": 0,
        "true_dtw": False,
    }

    # v2: emit the richer metric set (acc@collar, edge/interior, start/end MAE) via align_metrics.
    __sis_version__ = 5  # boundary off-by-one fix (end frame t, was t-1 -> inverted 1-frame tokens)

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
        with_ref_metrics: bool = True,
        with_confidence: bool = False,
        confidence_version: int = 1,
        word_topology: bool = False,
        dtw_no_blank: bool = False,
        whisper_dtw: bool = False,
        attn_token_zscore: bool = False,
        median_filter_width: int = 0,
        true_dtw: bool = False,
    ):
        """
        :param grad_score_hdf: from :class:`ExtractInGradsPerTokenJob`. Must
            have streams ``num_input_frames``, ``num_tokens``,
            ``num_tokens_per_word`` (single chunk per seq).
        :param grad_score_key: usually ``"data"``.
        :param align_opts: see :class:`exp2025_05_05_align.Aligner`.
        :param with_ref_metrics: compare to the dataset's reference boundaries
            (the default). Set False when the dataset carries hypothesis
            transcripts without meaningful start/stop times;
            then only ``out_word_boundaries_hdf`` is produced
            and the metric outputs are set to None.
        :param with_confidence: additionally run the forward-backward DP over the
            alignment lattice (see ``Aligner.align(collect_posteriors=...)``) and report
            per-word posterior-occupancy confidence and its correlation with the per-word
            WBE in ``out_conf_corr`` (requires ``with_ref_metrics``).
        :param confidence_version: 1 = span-mean occupancy (saturates; near-zero corr on segA);
            2 = negative posterior-mass leak outside the Viterbi span (boundary-localized).
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
        self.with_ref_metrics = with_ref_metrics
        self.with_confidence = with_confidence
        self.confidence_version = confidence_version
        # Word-aware DP topology: blank/silence only between words, not within (opt-in).
        self.word_topology = bool(word_topology)
        # DTW-style: no blank/silence states at all (monotonic token->frame).
        self.dtw_no_blank = bool(dtw_no_blank)
        # Faithful openai-whisper DTW (their actual algorithm on the score matrix).
        self.whisper_dtw = bool(whisper_dtw)
        # Whisper find_alignment difference-ablation knobs (all default off):
        # (D) z-norm the matrix over the token axis per frame
        # (whisper standardizes the softmax-over-time weights over tokens -> signed, ~half negative).
        self.attn_token_zscore = bool(attn_token_zscore)
        # (E) median filter over time (whisper uses width 7).
        self.median_filter_width = int(median_filter_width)
        # (G) true DTW recursion (diag/up/left, frame-sharing + zero-duration tokens),
        # whisper's dtw_cpu, vs our Viterbi-with-blank.
        # apply_log (align_opts) selects the log-prob cost (our variant)
        # vs the raw-matrix cost (whisper's dtw(-matrix)).
        self.true_dtw = bool(true_dtw)
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
        # Aligned per-word (start, end) in SECONDS (float), [num_words, 2] per seq --
        # same format as the forced-align baseline jobs.
        # (Jobs finished before this output existed lack the file.)
        self.out_word_boundaries_hdf = self.output_path("word_boundaries.hdf")
        # Confidence-vs-error analysis (with_confidence): correlations + quartile WBEs.
        self.out_conf_corr = self.output_var("conf_corr.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 8, "time": 5, "engine": "short"})

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
        # The extract writes exactly one entry per dataset seq, in order (a 0-token entry for an
        # empty recog -- it never skips), so this HDF is index-aligned with the dataset and we
        # iterate it in lockstep (load_seqs expects in-order, gap-free access). Map tag -> position
        # purely as a sanity check: a missing tag (lost seq) raises below, and `assert seq_idx == _gi`
        # makes any reordering/drift fail loudly. get_all_tags reads tags only (no data preload).
        _g_tag2idx = {t: i for i, t in enumerate(grad_score_hdf_ds.get_all_tags())}

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset: {ds}")
        print("Dataset keys:", ds.keys())
        print("Using key:", self.dataset_key)
        print("Num seqs:", len(ds[self.dataset_key]))

        aligner = Aligner(**self.align_opts)

        from returnn.datasets.hdf import SimpleHDFWriter

        boundaries_writer = SimpleHDFWriter(self.out_word_boundaries_hdf.get_path(), dim=2, ndim=2)
        utt_errs = []
        conf_wbe_pairs = []  # (per-word confidence, per-word wbe), with_confidence only

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            _gi = _g_tag2idx.get(f"seq-{seq_idx}")
            if _gi is None:
                # The extract writes an entry for EVERY seq (a 0-token entry for an empty recog),
                # so a missing entry here is never "just empty" -- it means a seq was lost in the
                # extract, which is a bug. Fail loudly instead of silently backfilling.
                raise KeyError(
                    f"seq-{seq_idx} has NO entry in {self.grad_score_hdf.get_path()} --"
                    " every seq must have an entry (an empty recog still gets a 0-token entry);"
                    " a missing entry means a seq was lost in the extract."
                )
            # In order, no holes: the extract never skips, so the entry must sit at seq_idx.
            # (We then load/read in lockstep with seq_idx, not _gi -- HDFDataset wants ordered access.)
            assert seq_idx == _gi, f"extract HDF not in dataset order: seq-{seq_idx} at position {_gi}"
            grad_score_hdf_ds.load_seqs(seq_idx, seq_idx + 1)

            samplerate = data["audio"]["sampling_rate"]
            num_audio_samples = len(data["audio"]["array"])
            audio_len_secs = num_audio_samples / samplerate

            num_input_frames = grad_score_hdf_ds.get_data(seq_idx, "num_input_frames")
            num_tokens = grad_score_hdf_ds.get_data(seq_idx, "num_tokens")
            if num_tokens.size == 0:
                # 0-token entry: an empty recog (the extract writes a 0-row entry, never skips).
                # No tokens to align -> emit a 0-row boundary so the seq stays present, then skip.
                # Must precede the single-chunk asserts and the secs_per_timeframe division below,
                # which both assume >=1 chunk/token/frame. (Empties only occur in hyp-mode.)
                boundaries_writer.insert_batch(np.zeros((1, 0, 2), dtype="float32"), [0], [f"seq-{seq_idx}"])
                continue
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

            # More tokens than time frames (char-level targets on a coarse ~80 ms attention
            # grid) -> the monotonic DP (needs S<=T) can't run. Upsample the time axis by
            # repeating each frame k times so S<=k*T. Boundaries then resolve to
            # secs_per_timeframe/k, but the underlying signal is unchanged (the coarse grid
            # is the real resolution limit; this just lets >1 token share a frame). No-op for
            # all grad rows (tokens<=frames), so finished jobs are unaffected (logic-only,
            # no hash change).
            if chunk_num_tokens > chunk_num_timeframes:
                _up_k = -(-chunk_num_tokens // chunk_num_timeframes)  # ceil
                grad_mat = np.repeat(grad_mat, _up_k, axis=1)
                chunk_num_timeframes = chunk_num_timeframes * _up_k
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

            # Whisper find_alignment difference-ablation transforms on the score matrix (default off).
            if getattr(self, "attn_token_zscore", False):
                # (D) z-norm over tokens per frame, like whisper's std_mean over the token axis.
                _mu = grad_mat.mean(axis=0, keepdims=True)
                _sd = grad_mat.std(axis=0, keepdims=True)
                grad_mat = (grad_mat - _mu) / (_sd + 1e-9)
            if getattr(self, "median_filter_width", 0) and self.median_filter_width > 1:
                # (E) median filter over time.
                from scipy.ndimage import median_filter as _median_filter

                grad_mat = _median_filter(grad_mat, size=(1, self.median_filter_width), mode="nearest")

            _conf = {} if getattr(self, "with_confidence", False) else None
            _blank_state_mask = None
            if getattr(self, "dtw_no_blank", False):
                # DTW analog: no INTERIOR silence (keep only the leading/trailing blank so a path
                # exists even for short token sequences); inter-word pauses absorbed into words.
                _blank_state_mask = np.array(
                    [(i == 0 or i == chunk_num_tokens) for i in range(chunk_num_tokens + 1)], dtype=bool
                )
            elif getattr(self, "word_topology", False):
                # Keep blank only at word boundaries (+ start/end); forbid intra-word blanks.
                _word_starts = set(int(x) for x in np.cumsum(tokens_per_word)[:-1])
                _blank_state_mask = np.array(
                    [(i == 0 or i == chunk_num_tokens or i in _word_starts) for i in range(chunk_num_tokens + 1)],
                    dtype=bool,
                )
            if getattr(self, "true_dtw", False):
                align_token_start_ends = _true_dtw_align(
                    grad_mat, apply_log=bool(self.align_opts.get("apply_log", True))
                )
            elif getattr(self, "whisper_dtw", False):
                align_token_start_ends = _openai_dtw_align(grad_mat)
            else:
                align_token_start_ends = aligner.align(
                    grad_mat,
                    blank_override=blank_override,
                    blank_state_mask=_blank_state_mask,
                    collect_posteriors=_conf,
                )
            assert len(align_token_start_ends) == chunk_num_tokens

            # Collapse to per-word boundaries (and token confidences to per-word).
            align_word_start_ends = []
            word_confs = []
            cursor = 0
            for w, k in enumerate(tokens_per_word):
                first = align_token_start_ends[cursor]
                last = align_token_start_ends[cursor + k - 1]
                align_word_start_ends.append((first[0] * secs_per_timeframe, last[1] * secs_per_timeframe))
                if _conf is not None:
                    if getattr(self, "confidence_version", 1) >= 2:
                        word_confs.append(-float(np.sum(_conf["leak"][cursor : cursor + k])))
                    else:
                        word_confs.append(float(np.mean(_conf["mean_occ"][cursor : cursor + k])))
                cursor += k
            assert cursor == chunk_num_tokens

            print(
                f"** seq {seq_idx}, {num_words=} {chunk_num_tokens=}"
                f" {audio_len_secs=} {num_audio_samples=} {secs_per_timeframe=}"
            )

            _bsrc = getattr(self, "boundary_source", "word_detail")
            words: List[str] = data[_bsrc]["utterance"]
            assert num_words == len(words), f"{num_words=} {len(words)=}"

            boundaries_writer.insert_batch(
                np.array([align_word_start_ends], dtype="float32"),  # [1, num_words, 2], seconds
                [num_words],
                [f"seq-{seq_idx}"],
            )

            if not getattr(self, "with_ref_metrics", True):
                continue
            ref_word_starts: List[float] = data[_bsrc]["start"]
            ref_word_ends: List[float] = data[_bsrc]["stop"]
            assert num_words == len(ref_word_starts) == len(ref_word_ends), f"{num_words=} {len(ref_word_starts)=}"
            ref_word_start_ends = [
                tuple(t * self.dataset_offset_factors / samplerate for t in ts)
                for ts in zip(ref_word_starts, ref_word_ends)
            ]

            utt_err = per_utt_boundary_errors(align_word_start_ends, ref_word_start_ends)
            print("  WBE:", float(np.mean(utt_err["wbe"])))
            utt_errs.append(utt_err)
            if getattr(self, "with_confidence", False):
                conf_wbe_pairs.extend(zip(word_confs, utt_err["wbe"]))

        boundaries_writer.close()
        if not getattr(self, "with_ref_metrics", True):
            for out in (
                self.out_wbe,
                self.out_metrics,
                self.out_acc50,
                self.out_interior_wbe,
                self.out_edge_wbe,
                self.out_conf_corr,
            ):
                out.set(None)
            return
        metrics = aggregate_corpus(utt_errs)
        print("CORPUS METRICS:", metrics)
        self.out_wbe.set(metrics["wbe"])
        self.out_metrics.set(metrics)
        self.out_acc50.set(metrics["acc_50ms"])
        self.out_interior_wbe.set(metrics["interior_wbe"])
        self.out_edge_wbe.set(metrics["edge_wbe"])

        if not getattr(self, "with_confidence", False):
            self.out_conf_corr.set(None)
            return
        # Does the FB-DP posterior confidence predict the boundary error?
        conf = np.array([c for c, _ in conf_wbe_pairs])
        wbe = np.array([w for _, w in conf_wbe_pairs])
        _finite = np.isfinite(conf) & np.isfinite(wbe)
        n_dropped_nan = int(len(conf) - _finite.sum())
        conf, wbe = conf[_finite], wbe[_finite]

        def _pearson(x, y):
            x, y = x - x.mean(), y - y.mean()
            d = float(np.sqrt((x**2).sum() * (y**2).sum()))
            return float((x * y).sum() / d) if d else float("nan")

        def _ranks(x):
            return np.argsort(np.argsort(x)).astype(np.float64)

        order = np.argsort(-conf)  # most-confident first
        quartile_wbe = [float(wbe[idx].mean()) for idx in np.array_split(order, 4)]
        conf_corr = {
            "pearson": _pearson(conf, wbe),
            "spearman": _pearson(_ranks(conf), _ranks(wbe)),
            "n_words": int(len(conf)),
            "n_dropped_nan": n_dropped_nan,
            "confidence_version": getattr(self, "confidence_version", 1),
            "mean_conf": float(conf.mean()),
            "quartile_wbe_by_conf_desc": quartile_wbe,
        }
        print("CONF-vs-WBE:", conf_corr)
        self.out_conf_corr.set(conf_corr)
