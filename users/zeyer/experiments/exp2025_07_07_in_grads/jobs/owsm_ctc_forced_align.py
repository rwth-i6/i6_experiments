"""CTC forced-alignment baseline (torchaudio Viterbi) for OWSM-CTC, per inter-CTC emit block.

The 'posteriors' counterpart to grad-align on ``espnet/owsm_ctc_v4_1B``: force-align the reference
words on the model's own CTC emission (the selected ``layer``) and report word-boundary WBE. The
emission forced-aligns well at every block even though grad-align does not -- this brackets the
grad-align WBE from above. Mirrors :class:`ParakeetCtcForcedAlignJob` but on the ESPnet wrapper.
"""

from __future__ import annotations
from typing import Optional
from sisyphus import Job, Task, tk

from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class OwsmCtcForcedAlignJob(Job):
    """torchaudio CTC forced-alignment of reference words on espnet/owsm_ctc_v4_1B (per emit block)."""

    __sis_version__ = 2  # center_offset / width_signed_err / center_abs (align_metrics)

    @classmethod
    def hash(cls, parsed_args):
        # layer=None (final block) is the default -> drop it so a per-block table that adds the
        # final block via layer=None keeps a hash independent of this kwarg's introduction.
        parsed_args = dict(parsed_args)
        if parsed_args.get("layer") is None:
            parsed_args.pop("layer", None)
        if not parsed_args.get("emit_boundaries_hdf"):
            parsed_args.pop("emit_boundaries_hdf", None)
        return super().hash(parsed_args)

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        model_dir: tk.Path,
        dataset_offset_factors: int,
        layer: Optional[int] = None,
        version: int = 2,
        returnn_root: Optional[tk.Path] = None,
        emit_boundaries_hdf: bool = False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.model_dir = model_dir
        self.dataset_offset_factors = dataset_offset_factors
        self.layer = layer
        self.version = version
        self.returnn_root = returnn_root
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
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.owsm_ctc import OwsmCtc
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_metrics import (
            per_utt_boundary_errors,
            aggregate_corpus,
        )

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OwsmCtc(device=dev, model_dir=self.model_dir.get_path(), version=self.version, layer=self.layer)
        target_sr = 16000
        scale = self.dataset_offset_factors / target_sr

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print("Using key:", self.dataset_key, "num seqs:", len(ds[self.dataset_key]), "layer:", self.layer, flush=True)

        boundaries_writer = None
        if self.emit_boundaries_hdf:
            from returnn.datasets.hdf import SimpleHDFWriter

            boundaries_writer = SimpleHDFWriter(self.out_word_boundaries_hdf.get_path(), dim=2, ndim=2)

        word_errs = []
        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = torch.tensor(np.asarray(data["audio"]["array"], dtype=np.float32))
            sr = int(data["audio"]["sampling_rate"])
            assert sr == target_sr, f"OWSM-CTC expects 16 kHz, got {sr}"
            wd = data["word_detail"]
            words = list(wd["utterance"])
            pred_word_se = model.forced_align_words(audio=audio, sample_rate=sr, words=words)
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
