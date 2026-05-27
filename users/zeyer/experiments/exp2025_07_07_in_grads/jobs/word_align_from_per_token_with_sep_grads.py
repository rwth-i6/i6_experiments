"""Word alignment from a per-token-with-separator-grads HDF."""

from typing import Optional, Any, Dict, List
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class WordAlignFromPerTokenWithSepGradsJob(Job):
    """Run the Aligner on a per-subword (chars + separators) grad matrix, then
    collapse to per-word boundaries from char rows only.

    Counterpart to :class:`WordAlignFromPerTokenGradsJob` for the
    :class:`ExtractInGradsPerTokenWithSepGradsJob` HDF schema. Differs in:

    - Rows are interleaved: word_0_chars, inter_0_1_seps, word_1_chars, ...
    - The Aligner sees all rows in order (including separator rows acting as
      explicit silence anchors).
    - Word boundary collapse uses only char-row positions:
      ``word_w_start = first_char_in_word.start``,
      ``word_w_end = last_char_in_word.end``.

    Single-chunk only for now.
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

        ds_hdf = HDFDataset([self.grad_score_hdf.get_path()])
        ds_hdf.initialize()
        ds_hdf.init_seq_order(epoch=1)

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset key={self.dataset_key}, num_seqs={len(ds[self.dataset_key])}")

        aligner = Aligner(**self.align_opts)
        wbe_utts = []

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            ds_hdf.load_seqs(seq_idx, seq_idx + 1)

            samplerate = data["audio"]["sampling_rate"]
            num_audio_samples = len(data["audio"]["array"])
            audio_len_secs = num_audio_samples / samplerate

            num_input_frames = ds_hdf.get_data(seq_idx, "num_input_frames")
            num_subwords = ds_hdf.get_data(seq_idx, "num_subwords")
            num_tokens_per_word = ds_hdf.get_data(seq_idx, "num_tokens_per_word").reshape(-1).astype(int)
            num_inter_word_tokens = ds_hdf.get_data(seq_idx, "num_inter_word_tokens").reshape(-1).astype(int)
            assert num_input_frames.shape == (1, 1) and num_subwords.shape == (1, 1)
            chunk_T = int(num_input_frames[0, 0])
            chunk_S = int(num_subwords[0, 0])
            assert len(num_inter_word_tokens) == len(num_tokens_per_word) - 1, (
                f"{len(num_inter_word_tokens)=} vs {len(num_tokens_per_word)=}"
            )
            assert num_tokens_per_word.sum() + num_inter_word_tokens.sum() == chunk_S, (
                f"sum mismatch: chars={num_tokens_per_word.sum()} + seps={num_inter_word_tokens.sum()} != {chunk_S=}"
            )

            grad = ds_hdf.get_data(seq_idx, self.grad_score_key)
            assert grad.shape == (chunk_S * chunk_T, 1), f"{grad.shape=}"
            grad_mat = grad.reshape(chunk_S, chunk_T)
            secs_per_tf = audio_len_secs / chunk_T

            token_se = aligner.align(grad_mat)
            assert len(token_se) == chunk_S

            # Walk row layout: word0 chars, inter01 seps, word1 chars, inter12, ..., word_{N-1} chars.
            align_word_start_ends = []
            cursor = 0
            num_words = len(num_tokens_per_word)
            for w in range(num_words):
                k = num_tokens_per_word[w]
                first = token_se[cursor]
                last = token_se[cursor + k - 1]
                align_word_start_ends.append(
                    (first[0] * secs_per_tf, last[1] * secs_per_tf)
                )
                cursor += k
                if w < num_words - 1:
                    cursor += num_inter_word_tokens[w]
            assert cursor == chunk_S

            words: List[str] = data["word_detail"]["utterance"]
            ref_word_starts: List[float] = data["word_detail"]["start"]
            ref_word_ends: List[float] = data["word_detail"]["stop"]
            assert num_words == len(words) == len(ref_word_starts) == len(ref_word_ends)
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
            print(f"** seq {seq_idx} WBE={float(wbe_utt):.4f}")
            wbe_utts.append(wbe_utt)

        wbe = float(np.mean(wbe_utts))
        self.out_wbe.set(wbe)
