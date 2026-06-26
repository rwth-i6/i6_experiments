"""Per-ref-token CTC posterior dump for the alignment-saliency figure (and "our DP on the posteriors").

For each utterance, dumps ``data[i, t] = P(class(ref_token_i) | t)`` from the model's own CTC emission,
in the SAME HDF format as :class:`ExtractInGradsPerTokenJob` (the streams ``num_input_frames`` /
``num_tokens`` / ``num_tokens_per_word`` plus the flat per-token-over-time ``data``). So the same
:class:`WriteFigureDataJob` renders it as a heatmap and the same
:class:`WordAlignFromPerTokenGradsJob` aligns it -- the CTC posteriors run through our DP instead of a
single Viterbi forced-align path. Batch 1. CTC head only (model must expose ``ctc_posteriors_per_token``).
"""

from __future__ import annotations

from typing import Optional

from sisyphus import Job, Task, tk

from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class ExtractCtcPosteriorsPerTokenJob(Job):
    """Dump per-ref-token CTC posteriors ``[n_tok, T_enc]`` per seq, in the per-token-score HDF format."""

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        model_config: dict,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.model_config = model_config
        self.returnn_root = returnn_root
        self.rqmt = {"time": 4, "cpu": 2, "gpu": 1, "mem": 16}
        self.out_hdf = self.output_path("posteriors.hdf")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        import numpy as np
        import torch
        from datasets import load_dataset
        from returnn.datasets.hdf import SimpleHDFWriter
        from .models import make_model

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = make_model(**instanciate_delayed_copy(self.model_config), device=dev)

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print("Using key:", self.dataset_key, "num seqs:", len(ds[self.dataset_key]), flush=True)

        writer = SimpleHDFWriter(
            self.out_hdf.get_path(),
            dim=1,
            ndim=2,
            extra_type={
                "num_input_frames": (1, 2, "int32"),
                "num_tokens": (1, 2, "int32"),
                "num_tokens_per_word": (1, 2, "int32"),
            },
        )
        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = torch.tensor(np.asarray(data["audio"]["array"], dtype=np.float32))
            sr = int(data["audio"]["sampling_rate"])
            words = list(data["word_detail"]["utterance"])
            if not words:  # empty hyp -> emit an empty seq (WordAlign/WriteFigureData skip it)
                writer.insert_batch(
                    np.zeros((1, 0, 1), dtype="float32"),
                    [0],
                    [f"seq-{seq_idx}"],
                    extra={
                        "num_input_frames": np.zeros((1, 0, 1), dtype="int32"),
                        "num_tokens": np.zeros((1, 0, 1), dtype="int32"),
                        "num_tokens_per_word": np.zeros((1, 0, 1), dtype="int32"),
                    },
                )
                continue
            post, tokens_per_word, t_enc = model.ctc_posteriors_per_token(audio=audio, sample_rate=sr, words=words)
            n_tok = int(post.shape[0])
            # data flattened [n_tok * T_enc] (row-major: token-major), matching the grad HDF layout.
            writer.insert_batch(
                post.reshape(1, n_tok * t_enc, 1).astype("float32"),
                [n_tok * t_enc],
                [f"seq-{seq_idx}"],
                extra={
                    "num_input_frames": np.array([[[t_enc]]], dtype="int32"),
                    "num_tokens": np.array([[[n_tok]]], dtype="int32"),
                    "num_tokens_per_word": np.array(tokens_per_word, dtype="int32")[None, :, None],
                },
            )
            if seq_idx % 200 == 0:
                print(f"seq {seq_idx}: {len(words)} words, {n_tok} tokens, T_enc={t_enc}", flush=True)
        writer.close()
