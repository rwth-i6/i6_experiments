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

        self.out_wbe = self.output_var("wbe.txt")

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

        wbe_utts = []

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
            assert tokens_per_word.sum() == chunk_num_tokens, (
                f"{tokens_per_word.sum()=} != {chunk_num_tokens=}"
            )
            num_words = len(tokens_per_word)

            grad_mat = grad_score_hdf_ds.get_data(seq_idx, self.grad_score_key)
            assert grad_mat.shape == (chunk_num_tokens * chunk_num_timeframes, 1), (
                f"{grad_mat.shape=} {chunk_num_tokens=} {chunk_num_timeframes=}"
            )
            grad_mat = grad_mat.reshape(chunk_num_tokens, chunk_num_timeframes)
            secs_per_timeframe = audio_len_secs / chunk_num_timeframes

            align_token_start_ends = aligner.align(grad_mat)
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

            words: List[str] = data["word_detail"]["utterance"]
            ref_word_starts: List[float] = data["word_detail"]["start"]
            ref_word_ends: List[float] = data["word_detail"]["stop"]
            assert num_words == len(words) == len(ref_word_starts) == len(ref_word_ends), (
                f"{num_words=} {len(words)=} {len(ref_word_starts)=}"
            )
            ref_word_start_ends = [
                tuple(t * self.dataset_offset_factors / samplerate for t in ts)
                for ts in zip(ref_word_starts, ref_word_ends)
            ]

            wbe_utt = np.mean(
                [
                    0.5
                    * (
                        abs(ref_word_start_ends[w][0] - align_word_start_ends[w][0])
                        + abs(ref_word_start_ends[w][1] - align_word_start_ends[w][1])
                    )
                    for w in range(num_words)
                ]
            )
            print("  WBE:", float(wbe_utt))
            wbe_utts.append(wbe_utt)

        wbe = float(np.mean(wbe_utts))
        self.out_wbe.set(wbe)
