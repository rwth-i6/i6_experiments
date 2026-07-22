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
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class ParakeetCtcForcedAlignJob(Job):
    """torchaudio CTC forced-alignment of reference words on nvidia/parakeet-ctc-1.1b."""

    # model_config (opt-in, hash-excluded when None) generalizes this to any CTC model
    # with a forced_align_words() method (e.g. streaming FastConformer);
    # None keeps the ParakeetCtc path.
    __sis_hash_exclude__ = {"model_config": None}
    __sis_version__ = (
        3  # always dump the word-boundary HDF; WBE/stats now via CalcAlignmentMetricsFromWordBoundariesJob
    )

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        model_dir: Optional[tk.Path] = None,
        overlay_path: Optional[str] = None,
        dataset_offset_factors: int,
        returnn_root: Optional[tk.Path] = None,
        model_config: Optional[dict] = None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.model_dir = model_dir
        self.overlay_path = overlay_path
        self.dataset_offset_factors = dataset_offset_factors
        self.returnn_root = returnn_root
        self.model_config = model_config
        self.rqmt = {"time": 4, "cpu": 2, "gpu": 1, "mem": 16}
        # Always dump the predicted per-word boundaries (seconds); metrics (WBE etc.) are computed
        # uniformly downstream by CalcAlignmentMetricsFromWordBoundariesJob, same as every other aligner.
        self.out_word_boundaries_hdf = self.output_path("word_boundaries.hdf")

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

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model_config is not None:
            from .models import make_model

            model = make_model(**instanciate_delayed_copy(self.model_config), device=dev)
        else:
            model = ParakeetCtc(device=dev, model_dir=self.model_dir.get_path(), overlay_path=self.overlay_path)
        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print("Using key:", self.dataset_key, "num seqs:", len(ds[self.dataset_key]), flush=True)

        from returnn.datasets.hdf import SimpleHDFWriter

        boundaries_writer = SimpleHDFWriter(self.out_word_boundaries_hdf.get_path(), dim=2, ndim=2)
        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = torch.tensor(np.asarray(data["audio"]["array"], dtype=np.float32))
            sr = int(data["audio"]["sampling_rate"])
            words = list(data["word_detail"]["utterance"])
            # empty hypothesis (model recognised nothing) has no words to align and would crash
            # torchaudio.forced_align (max() over an empty target); emit empty boundaries for that seq.
            pred_word_se = model.forced_align_words(audio=audio, sample_rate=sr, words=words) if words else []
            # predicted boundaries are already seconds; CalcAlignmentMetricsFromWordBoundariesJob scales the ref.
            arr = np.asarray(pred_word_se, dtype="float32").reshape(1, len(pred_word_se), 2)
            boundaries_writer.insert_batch(arr, [len(pred_word_se)], [f"seq-{seq_idx}"])
            if seq_idx % 200 == 0:
                print(f"seq {seq_idx}: {len(words)} words", flush=True)
        boundaries_writer.close()
