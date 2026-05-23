"""
Grad-based alignment for ASR

Continuation/cleanup of :mod:`exp2025_05_05_align` using the generic,
model-agnostic job classes from :mod:`exp2025_07_07_in_grads`.

Conventions:

- Grad/score extraction:
  :class:`exp2025_07_07_in_grads.jobs.extract_in_grad_scores.ExtractInGradsFromModelJob`
  Parameterized by `(mult_grad_by_inputs: bool, attr_reduction: str)`.
  Output HDF has a single score key "data".

  Mapping to the old `grad_type` strings (see :mod:`exp2025_05_05_align`):
      L01_e_grad  ↔  mult_grad_by_inputs=True,  attr_reduction="L0.1"
      L1_e_grad   ↔  mult_grad_by_inputs=True,  attr_reduction="L1"
      L2_e_grad   ↔  mult_grad_by_inputs=True,  attr_reduction="L2"
      L01_grad    ↔  mult_grad_by_inputs=False, attr_reduction="L0.1"
      L1_grad     ↔  mult_grad_by_inputs=False, attr_reduction="L1"
      L2_grad     ↔  mult_grad_by_inputs=False, attr_reduction="L2"
      dot_e_grad  ↔  (not yet supported by generic job — TODO)

- Chunked long-form segmentation:
  :class:`exp2025_07_07_in_grads.jobs.chunk_segmentation.ChunkSegmentationFromModelJob`

- Alignment metric (WBE) computation: reuse
  :class:`exp2025_05_05_align.CalcAlignmentMetricsJob` and
  :class:`exp2025_05_05_align.CalcChunkedAlignmentMetricsJob` (already
  model-agnostic — just take a grad-score HDF + dataset).

Next steps (paper-blocking, in rough priority):

1. Port the prompt-sensitivity sweep (lines ~172-203 of old recipe).
2. Port the long-form Buckeye block (lines ~86-170 of old recipe), using
   `ChunkSegmentationFromModelJob` + `ExtractInGradsFromModelJob` chained.
3. Add an external-aligner baseline (MFA or WhisperX phoneme-align) computing
   WBE on the same TIMIT/Buckeye references.
4. Optional stretch: Qwen2-Audio model_config — extend
   `exp2025_07_07_in_grads.jobs.models` with a `Qwen2Audio` interface and
   register via the `type` dispatcher.
"""

from __future__ import annotations

from typing import Any, Dict
from sisyphus import tk

from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJobV2,
)
from i6_experiments.users.zeyer.external_models.phi4multimodal import (
    download_phi4multimodal_model,
)

from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.extract_in_grad_scores import (
    ExtractInGradsFromModelJob,
)

# Alignment-metric jobs and the dict-naming helper live in the old recipe and
# are already model-agnostic. Reusing them avoids a copy-paste.
from i6_experiments.users.zeyer.experiments.exp2025_05_05_align import (
    CalcAlignmentMetricsJob,
    _name_for_dict,
)


# (mult_grad_by_inputs, attr_reduction, old_grad_type_name_for_output_alias)
_GRAD_VARIANTS = [
    (True, "L0.1", "L01_e_grad"),
    (True, "L1", "L1_e_grad"),
    (True, "L2", "L2_e_grad"),
    (False, "L0.1", "L01_grad"),
    (False, "L1", "L1_grad"),
    (False, "L2", "L2_grad"),
]


# Align-opts grid (taken from the old recipe, lines ~57-72). Trim later for
# the paper headline; for now keep both so we can re-confirm numbers.
_ALIGN_OPTS_GRID = [
    {"apply_softmax_over_time": True, "blank_score": -6},
    {
        "apply_softmax_over_time": True,
        "blank_score": "calc",
        "blank_score_est": "flipped_after_softmax_over_time",
        "non_blank_score_reduce": "log_mean_exp",
        "blank_score_flipped_percentile": 60,
        "apply_softmax_over_labels": True,
    },
]


_DATASET_OFFSET_FACTORS = {"timit": 1, "buckeye": 1000}


def _phi4mm_model_config(
    model_dir: tk.Path, *, speech_prompt: str = "Transcribe the audio clip into text."
) -> Dict[str, Any]:
    """
    model_config dict consumed by the generic `make_model(type=..., **opts)`
    registry in :mod:`exp2025_07_07_in_grads.jobs.models`.
    """
    return {
        "type": "Phi4MM",
        "model_dir": model_dir,
        "speech_prompt": speech_prompt,
    }


def py():
    """Sisyphus entry point."""
    dl_phi4mi_dir = download_phi4multimodal_model()

    dl_ds_timit = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/timit", repo_type="dataset")
    tk.register_output("timit-dataset", dl_ds_timit.out_hub_cache_dir)

    dl_ds_buckeye = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/buckeye", repo_type="dataset")
    tk.register_output("buckeye-dataset", dl_ds_buckeye.out_hub_cache_dir)

    # --- Short-form: TIMIT val/test, Phi4-multimodal -----------------------
    # Mirrors exp2025_05_05_align lines ~44-84 but via the generic Job.

    phi4mm_cfg = _phi4mm_model_config(dl_phi4mi_dir)

    for ds_name, ds_dl, keys in [
        ("timit", dl_ds_timit, ["val", "test"]),
        # ("buckeye", dl_ds_buckeye, ["test", "train"]),  # enable when ready
    ]:
        for key in keys:
            for mult_grad_by_inputs, attr_reduction, grad_type_alias in _GRAD_VARIANTS:
                gen = ExtractInGradsFromModelJob(
                    dataset_dir=ds_dl.out_hub_cache_dir,
                    dataset_key=key,
                    model_config=phi4mm_cfg,
                    mult_grad_by_inputs=mult_grad_by_inputs,
                    attr_reduction=attr_reduction,
                )
                gen.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                name = f"phi4mm-{ds_name}-{key}-{grad_type_alias}"
                gen.add_alias(name)
                tk.register_output(f"{name}.hdf", gen.out_hdf)

                for align_opts in _ALIGN_OPTS_GRID:
                    align_name = f"align/{name}-{_name_for_dict(align_opts)}"
                    align = CalcAlignmentMetricsJob(
                        grad_score_hdf=gen.out_hdf,
                        grad_score_key="data",
                        dataset_dir=ds_dl.out_hub_cache_dir,
                        dataset_key=key,
                        dataset_offset_factors=_DATASET_OFFSET_FACTORS[ds_name],
                        align_opts=align_opts,
                    )
                    align.add_alias(align_name)
                    tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # --- TODO: prompt sensitivity sweep ------------------------------------
    # See exp2025_05_05_align lines ~172-203.

    # --- TODO: long-form Buckeye (chunked segmentation + grad-align) -------
    # See exp2025_05_05_align lines ~86-170. Use:
    #   from exp2025_07_07_in_grads.jobs.chunk_segmentation import ChunkSegmentationFromModelJob
    #   from exp2025_05_05_align import CalcChunkedAlignmentMetricsJob

    # --- TODO: external-aligner baselines (MFA / WhisperX phoneme align) ---
    # Needed for the SLT paper headline table.
