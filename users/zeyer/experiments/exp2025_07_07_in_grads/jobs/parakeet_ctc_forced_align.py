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

    # model_config (opt-in, hash-excluded when None) generalizes this to any CTC model with a
    # forced_align_words() method (e.g. streaming FastConformer); None keeps the ParakeetCtc path.
    __sis_hash_exclude__ = {"model_config": None, "emit_boundaries_hdf": False}
    __sis_version__ = 2  # center_offset / width_signed_err / center_abs (align_metrics)

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
        emit_boundaries_hdf: bool = False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.model_dir = model_dir
        self.overlay_path = overlay_path
        self.dataset_offset_factors = dataset_offset_factors
        self.returnn_root = returnn_root
        self.model_config = model_config
        self.emit_boundaries_hdf = emit_boundaries_hdf
        self.rqmt = {"time": 4, "cpu": 2, "gpu": 1, "mem": 16}
        # hyp-mode: emit only the predicted word-boundary HDF (the hyp dataset has no gold times);
        # forced-mode (default): the in-job WBE against the dataset's gold word_detail.
        if emit_boundaries_hdf:
            self.out_word_boundaries_hdf = self.output_path("word_boundaries.hdf")
        else:
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
        if self.model_config is not None:
            from .models import make_model

            model = make_model(**instanciate_delayed_copy(self.model_config), device=dev)
        else:
            model = ParakeetCtc(device=dev, model_dir=self.model_dir.get_path(), overlay_path=self.overlay_path)
        scale = self.dataset_offset_factors / model.target_sr

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print("Using key:", self.dataset_key, "num seqs:", len(ds[self.dataset_key]), flush=True)

        boundaries_writer = None
        if self.emit_boundaries_hdf:
            from returnn.datasets.hdf import SimpleHDFWriter

            boundaries_writer = SimpleHDFWriter(self.out_word_boundaries_hdf.get_path(), dim=2, ndim=2)

        word_errs = []
        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = torch.tensor(np.asarray(data["audio"]["array"], dtype=np.float32))
            sr = int(data["audio"]["sampling_rate"])
            wd = data["word_detail"]
            words = list(wd["utterance"])
            # empty hypothesis (the model recognised nothing) has no words to align and would crash
            # torchaudio.forced_align (max() over an empty target); emit empty boundaries for that seq.
            pred_word_se = model.forced_align_words(audio=audio, sample_rate=sr, words=words) if words else []
            if boundaries_writer is not None:
                # predicted boundaries are already seconds -> CalcHypAlignMetricsJob scales the ref.
                arr = np.asarray(pred_word_se, dtype="float32").reshape(1, len(pred_word_se), 2)
                boundaries_writer.insert_batch(arr, [len(pred_word_se)], [f"seq-{seq_idx}"])
            else:
                ref_word_se = [(s * scale, e * scale) for s, e in zip(wd["start"], wd["stop"])]
                word_errs.append(per_utt_boundary_errors(pred_word_se, ref_word_se))
            if seq_idx % 200 == 0:
                print(f"seq {seq_idx}: {len(words)} words", flush=True)

        if boundaries_writer is not None:
            boundaries_writer.close()
            return
        word_metrics = aggregate_corpus(word_errs)
        print("WORD METRICS:", word_metrics)
        self.out_word_wbe.set(word_metrics["wbe"])
        self.out_word_metrics.set(word_metrics)
