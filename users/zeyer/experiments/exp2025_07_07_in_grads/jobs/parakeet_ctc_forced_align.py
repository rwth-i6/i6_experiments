"""Plain CTC forced-alignment baseline (torchaudio Viterbi) for the Parakeet-CTC model.

The 'posteriors' counterpart to grad-align on ``nvidia/parakeet-ctc-1.1b``: force-align the reference
words on the model's own CTC emission and report word-boundary WBE. Mirrors
:class:`ForcedAlignPhonemeBaselineJob` (g2p word mode) but on the NeMo FastConformer-CTC via the
:class:`ParakeetCtc` wrapper (BPE subwords grouped to words). Brackets the grad-align WBE from above
(the model's own preferred alignment of the SAME emissions).
"""

from __future__ import annotations
from typing import Optional
from sisyphus import Job, Task, tk

from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class ParakeetCtcForcedAlignJob(Job):
    """torchaudio CTC forced-alignment of reference words on nvidia/parakeet-ctc-1.1b."""

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        model_dir: tk.Path,
        overlay_path: str,
        dataset_offset_factors: int,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.model_dir = model_dir
        self.overlay_path = overlay_path
        self.dataset_offset_factors = dataset_offset_factors
        self.returnn_root = returnn_root
        self.rqmt = {"time": 4, "cpu": 2, "gpu": 1, "mem": 16}
        self.out_word_wbe = self.output_var("word_wbe.txt")
        self.out_word_metrics = self.output_var("word_metrics.txt")

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
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.parakeet_ctc import ParakeetCtc
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_metrics import (
            per_utt_boundary_errors,
            aggregate_corpus,
        )

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ParakeetCtc(device=dev, model_dir=self.model_dir.get_path(), overlay_path=self.overlay_path)
        scale = self.dataset_offset_factors / model.target_sr

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print("Using key:", self.dataset_key, "num seqs:", len(ds[self.dataset_key]), flush=True)

        word_errs = []
        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = torch.tensor(np.asarray(data["audio"]["array"], dtype=np.float32))
            sr = int(data["audio"]["sampling_rate"])
            wd = data["word_detail"]
            words = list(wd["utterance"])
            pred_word_se = model.forced_align_words(audio=audio, sample_rate=sr, words=words)
            ref_word_se = [(s * scale, e * scale) for s, e in zip(wd["start"], wd["stop"])]
            word_errs.append(per_utt_boundary_errors(pred_word_se, ref_word_se))
            if seq_idx % 200 == 0:
                print(f"seq {seq_idx}: {len(words)} words", flush=True)

        word_metrics = aggregate_corpus(word_errs)
        print("WORD METRICS:", word_metrics)
        self.out_word_wbe.set(word_metrics["wbe"])
        self.out_word_metrics.set(word_metrics)
