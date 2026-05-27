"""
Grad-based alignment for ASR.

Continuation / cleanup of :mod:`exp2025_05_05_align` using the generic,
model-agnostic Job classes from :mod:`exp2025_07_07_in_grads`.

Conventions:

- Grad/score extraction:
  :class:`exp2025_07_07_in_grads.jobs.extract_in_grad_scores.ExtractInGradsFromModelJob`
  Parameterized by `(mult_grad_by_inputs: bool, attr_reduction: str)`.
  Output HDF has a single score key "data".

  Mapping to the old `grad_type` strings (see :mod:`exp2025_05_05_align`):
      L01_e_grad  <-> mult_grad_by_inputs=True,  attr_reduction="L0.1"
      L05_e_grad  <-> mult_grad_by_inputs=True,  attr_reduction="L0.5"
      L1_e_grad   <-> mult_grad_by_inputs=True,  attr_reduction="L1"
      L2_e_grad   <-> mult_grad_by_inputs=True,  attr_reduction="L2"
      L01_grad    <-> mult_grad_by_inputs=False, attr_reduction="L0.1"
      L05_grad    <-> mult_grad_by_inputs=False, attr_reduction="L0.5"
      L1_grad     <-> mult_grad_by_inputs=False, attr_reduction="L1"
      L2_grad     <-> mult_grad_by_inputs=False, attr_reduction="L2"
      dot_e_grad  <-> mult_grad_by_inputs=True,  attr_reduction="sum"

- Chunked long-form segmentation:
  :class:`exp2025_07_07_in_grads.jobs.chunk_segmentation.ChunkSegmentationFromModelJob`

- Alignment metric (WBE) computation: reuse
  :class:`exp2025_05_05_align.CalcAlignmentMetricsJob` and
  :class:`exp2025_05_05_align.CalcChunkedAlignmentMetricsJob` (already
  model-agnostic -- just take a grad-score HDF + dataset).

The Phi4MM model variant flags (margin_grad, char_level, ...) were folded
into the base :class:`Phi4MM` class. This recipe just passes them via
``rf.build_dict(Phi4MM, ...)``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

import returnn.frontend as rf
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
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.extract_per_token_grads import (
    ExtractInGradsPerTokenJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.word_align_from_per_token_grads import (
    WordAlignFromPerTokenGradsJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.add_sizes_to_grad_score_hdf import (
    AddSizesToGradScoreHdfJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.smooth_grad_score_hdf import (
    SmoothGradScoreHdfJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.smooth_per_token_grad_score_hdf import (
    SmoothPerTokenGradScoreHdfJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.phi4mm import Phi4MM

# Alignment-metric jobs and the dict-naming helper live in the old recipe and
# are already model-agnostic. Reusing them avoids a copy-paste.
from i6_experiments.users.zeyer.experiments.exp2025_05_05_align import (
    CalcAlignmentMetricsJob,
    _name_for_dict,
)


# (mult_grad_by_inputs, attr_reduction, old_grad_type_name_for_output_alias)
_GRAD_VARIANTS = [
    (True, "sum", "dot_e_grad"),
    (True, "L0.1", "L01_e_grad"),
    (True, "L0.5", "L05_e_grad"),
    (True, "L1", "L1_e_grad"),
    (True, "L2", "L2_e_grad"),
    (False, "L0.1", "L01_grad"),
    (False, "L0.5", "L05_grad"),
    (False, "L1", "L1_grad"),
    (False, "L2", "L2_grad"),
]


# Align-opts grid (taken from the old recipe). Trim later; for now keep both
# so we can re-confirm numbers.
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
    model_dir: tk.Path,
    *,
    speech_prompt: str = "Transcribe the audio clip into text.",
    fake_loss_grad: bool = False,
    margin_grad: bool = False,
    margin_grad_k: int = 1,
    eos_margin: bool = False,
    first_subword_only: bool = False,
    char_level: bool = False,
    char_level_sep: Optional[str] = None,
    char_level_brackets: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the model_config dict consumed by ``make_model`` in
    :mod:`exp2025_07_07_in_grads.jobs.models`. Uses ``rf.build_dict`` so the
    class is a code symbol (find-usages works).

    Each variant flag, when at its default, is omitted from the resulting dict
    so the hash matches across configs that don't touch it. The torch-2.7
    compat flags (``unwrap_checkpoint_wrappers``, ``target_start_end_to_device``)
    are always-on for this recipe -- they fix the audio-encoder vs. autograd
    incompatibility and the CPU/CUDA target_start_end mismatch respectively.
    """
    extra: Dict[str, Any] = {}
    if fake_loss_grad:
        extra["fake_loss_grad"] = True
    if margin_grad:
        extra["margin_grad"] = True
    if margin_grad_k != 1:
        extra["margin_grad_k"] = margin_grad_k
    if eos_margin:
        extra["eos_margin"] = True
    if first_subword_only:
        extra["first_subword_only"] = True
    if char_level:
        extra["char_level"] = True
    if char_level_sep is not None:
        extra["char_level_sep"] = char_level_sep
    if char_level_brackets is not None:
        extra["char_level_brackets"] = char_level_brackets
    return rf.build_dict(
        Phi4MM,
        model_dir=model_dir,
        speech_prompt=speech_prompt,
        unwrap_checkpoint_wrappers=True,
        target_start_end_to_device=True,
        **extra,
    )


def py():
    """Sisyphus entry point."""
    dl_phi4mi_dir = download_phi4multimodal_model()

    dl_ds_timit = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/timit", repo_type="dataset")
    tk.register_output("timit-dataset", dl_ds_timit.out_hub_cache_dir)

    dl_ds_buckeye = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/buckeye", repo_type="dataset")
    tk.register_output("buckeye-dataset", dl_ds_buckeye.out_hub_cache_dir)

    # --- Short-form: TIMIT val/test, Phi4-multimodal -----------------------
    # Mirrors exp2025_05_05_align lines ~44-84 but via the generic Job.
    # Several model_config variants of the per-word gradient signal.

    for variant_suffix, variant_kwargs in [
        ("", {}),  # default: actual d log p / d inputs
        ("-fakegrad", {"fake_loss_grad": True}),  # OLD-style fake-loss-grad
        ("-margin", {"margin_grad": True}),  # log p(t) - log p(top non-t)
        ("-firstsub", {"first_subword_only": True}),  # only first subword per word
        ("-charlev", {"char_level": True}),  # tokenize ref char-by-char, no boundary marker
        # char-level boundary markers + margin top-K + flag combos.
        ("-charlev-dot", {"char_level": True, "char_level_sep": "·"}),  # ·
        ("-charlev-spc", {"char_level": True, "char_level_sep": " "}),
        ("-margin-firstsub", {"margin_grad": True, "first_subword_only": True}),
        ("-margin-k3", {"margin_grad": True, "margin_grad_k": 3}),
    ]:
        phi4mm_cfg = _phi4mm_model_config(dl_phi4mi_dir, **variant_kwargs)
        _build_timit_phi4mm(
            phi4mm_cfg=phi4mm_cfg,
            variant_suffix=variant_suffix,
            dl_ds_timit=dl_ds_timit,
        )

    # Boundary-marker scouting variants: only L2_e_grad + val to keep the cost
    # low. Wider sweep can be added later if any of these clearly wins.
    for variant_suffix, variant_kwargs in [
        ("-charlev-brk", {"char_level": True, "char_level_brackets": "char"}),
        ("-charlev-brk-spc", {"char_level": True, "char_level_brackets": "char", "char_level_sep": " "}),
        ("-charlev-brkw", {"char_level": True, "char_level_brackets": "word"}),
        ("-charlev-brkw-spc", {"char_level": True, "char_level_brackets": "word", "char_level_sep": " "}),
        # Scout EOS margin (log p(target) - log p(end-of-assistant)).
        ("-eos-margin", {"eos_margin": True}),
        # charlev-spc combos. Best single-flag variant so far is charlev-spc;
        # see whether stacking margin / first-subword on top buys anything.
        ("-charlev-spc-margin", {"char_level": True, "char_level_sep": " ", "margin_grad": True}),
        ("-charlev-spc-firstsub", {"char_level": True, "char_level_sep": " ", "first_subword_only": True}),
        ("-charlev-spc-margin-k3", {"char_level": True, "char_level_sep": " ", "margin_grad": True, "margin_grad_k": 3}),
    ]:
        phi4mm_cfg = _phi4mm_model_config(dl_phi4mi_dir, **variant_kwargs)
        _build_timit_phi4mm(
            phi4mm_cfg=phi4mm_cfg,
            variant_suffix=variant_suffix,
            dl_ds_timit=dl_ds_timit,
            grad_variants_subset=[(True, "L2", "L2_e_grad")],
            splits_subset=["val"],
        )

    # Temporal smoothing scout. Re-uses the existing default and charlev-spc
    # Extract HDFs (hash-identical to those already computed -- no extra GPU
    # work, only new Smooth + Calc).
    for variant_suffix, variant_kwargs in [
        ("", {}),
        ("-charlev-spc", {"char_level": True, "char_level_sep": " "}),
    ]:
        phi4mm_cfg = _phi4mm_model_config(dl_phi4mi_dir, **variant_kwargs)
        gen = ExtractInGradsFromModelJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=phi4mm_cfg,
            mult_grad_by_inputs=True,
            attr_reduction="L2",
        )
        gen.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        base_name = f"phi4mm-timit-val-L2_e_grad{variant_suffix}"
        hdf_with_sizes = AddSizesToGradScoreHdfJob(grad_score_hdf=gen.out_hdf)
        for smooth_kind, smooth_size in [
            ("gaussian", 1.0),
            ("gaussian", 2.0),
            ("gaussian", 3.0),
            ("median", 3),
            ("median", 5),
        ]:
            tag = f"{smooth_kind}{smooth_size}".replace(".", "p")
            smoother = SmoothGradScoreHdfJob(
                grad_score_hdf=hdf_with_sizes.out_hdf,
                smooth_kind=smooth_kind,
                smooth_size=smooth_size,
            )
            smoother.add_alias(f"{base_name}-smooth-{tag}")
            for align_opts in _ALIGN_OPTS_GRID:
                align_name = f"align/{base_name}-smooth-{tag}-{_name_for_dict(align_opts)}"
                align = CalcAlignmentMetricsJob(
                    grad_score_hdf=smoother.out_hdf,
                    grad_score_key="data",
                    dataset_dir=dl_ds_timit.out_hub_cache_dir,
                    dataset_key="val",
                    dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                    align_opts=align_opts,
                )
                align.add_alias(align_name)
                tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)


    # Per-token block. ExtractInGradsPerTokenJob does K backwards per word
    # (one per subword/token) instead of 1, so we get per-token grad maps and
    # can align at token level, then collapse to word boundaries via
    # num_tokens_per_word. Cost is ~K x base extract.
    #
    # On TIMIT val L2_e_grad, pertoken-charlev-spc gave WBE 0.087 vs 0.140
    # for the per-word charlev-spc baseline (~40% rel improvement). This block
    # scouts whether the gain transfers to other char_level configs, other
    # grad reductions, the test split, and smoothing.

    def _wire_pertoken(*, dl_ds, ds_name, split, phi4mm_cfg, attr, mgi, grad_alias, suffix):
        gen = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds.out_hub_cache_dir,
            dataset_key=split,
            model_config=phi4mm_cfg,
            mult_grad_by_inputs=mgi,
            attr_reduction=attr,
        )
        gen.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        name = f"phi4mm-{ds_name}-{split}-{grad_alias}-pertoken{suffix}"
        gen.add_alias(name)
        tk.register_output(f"{name}.hdf", gen.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=gen.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds.out_hub_cache_dir,
                dataset_key=split,
                dataset_offset_factors=_DATASET_OFFSET_FACTORS[ds_name],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)
        return gen

    # (a) Model-variant sweep, val + L2_e_grad. Pertoken on each char_level
    # mode, including the BOW/EOW bracket variants we never tested per-token.
    for variant_suffix, variant_kwargs in [
        ("", {}),
        ("-charlev", {"char_level": True}),
        ("-charlev-spc", {"char_level": True, "char_level_sep": " "}),
        ("-charlev-dot", {"char_level": True, "char_level_sep": "\u00b7"}),
        ("-charlev-brk", {"char_level": True, "char_level_brackets": "char"}),
        ("-charlev-brk-spc", {"char_level": True, "char_level_brackets": "char", "char_level_sep": " "}),
        ("-charlev-brkw", {"char_level": True, "char_level_brackets": "word"}),
        ("-charlev-brkw-spc", {"char_level": True, "char_level_brackets": "word", "char_level_sep": " "}),
    ]:
        phi4mm_cfg = _phi4mm_model_config(dl_phi4mi_dir, **variant_kwargs)
        _wire_pertoken(
            dl_ds=dl_ds_timit, ds_name="timit", split="val",
            phi4mm_cfg=phi4mm_cfg, attr="L2", mgi=True,
            grad_alias="L2_e_grad", suffix=variant_suffix,
        )

    # (b) pertoken-charlev-spc with other grad reductions, val.
    pt_csp_cfg = _phi4mm_model_config(dl_phi4mi_dir, char_level=True, char_level_sep=" ")
    for mgi, attr, grad_alias in [
        (True, "L1", "L1_e_grad"),
        (True, "L0.5", "L05_e_grad"),
        (True, "sum", "dot_e_grad"),
    ]:
        _wire_pertoken(
            dl_ds=dl_ds_timit, ds_name="timit", split="val",
            phi4mm_cfg=pt_csp_cfg, attr=attr, mgi=mgi,
            grad_alias=grad_alias, suffix="-charlev-spc",
        )

    # (c) pertoken-charlev-spc on test split, L2_e_grad.
    _wire_pertoken(
        dl_ds=dl_ds_timit, ds_name="timit", split="test",
        phi4mm_cfg=pt_csp_cfg, attr="L2", mgi=True,
        grad_alias="L2_e_grad", suffix="-charlev-spc",
    )

    # (d) Smoothing the pertoken-charlev-spc HDF. Reuses the val L2_e_grad
    # extract (hash-identical to the one in (a)) -- only adds Smooth + Calc.
    pt_csp_gen = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=pt_csp_cfg,
        mult_grad_by_inputs=True,
        attr_reduction="L2",
    )
    pt_csp_gen.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    smooth_base = "phi4mm-timit-val-L2_e_grad-pertoken-charlev-spc"
    for smooth_kind, smooth_size in [
        ("gaussian", 1.0),
        ("gaussian", 2.0),
        ("gaussian", 3.0),
        ("median", 3),
        ("median", 5),
    ]:
        tag = f"{smooth_kind}{smooth_size}".replace(".", "p")
        smoother = SmoothPerTokenGradScoreHdfJob(
            grad_score_hdf=pt_csp_gen.out_hdf,
            smooth_kind=smooth_kind,
            smooth_size=smooth_size,
        )
        smoother.add_alias(f"{smooth_base}-smooth-{tag}")
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{smooth_base}-smooth-{tag}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=smoother.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)


def _build_timit_phi4mm(
    *,
    phi4mm_cfg: Dict[str, Any],
    variant_suffix: str,
    dl_ds_timit,
    grad_variants_subset: Optional[List] = None,
    splits_subset: Optional[List] = None,
) -> None:
    grad_variants = grad_variants_subset if grad_variants_subset is not None else _GRAD_VARIANTS
    splits = splits_subset if splits_subset is not None else ["val", "test"]
    for ds_name, ds_dl, keys in [
        ("timit", dl_ds_timit, splits),
    ]:
        for key in keys:
            for mult_grad_by_inputs, attr_reduction, grad_type_alias in grad_variants:
                gen = ExtractInGradsFromModelJob(
                    dataset_dir=ds_dl.out_hub_cache_dir,
                    dataset_key=key,
                    model_config=phi4mm_cfg,
                    mult_grad_by_inputs=mult_grad_by_inputs,
                    attr_reduction=attr_reduction,
                )
                gen.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                name = f"phi4mm-{ds_name}-{key}-{grad_type_alias}{variant_suffix}"
                gen.add_alias(name)
                tk.register_output(f"{name}.hdf", gen.out_hdf)

                # Add `sizes` key to the HDF (CalcAlignmentMetricsJob expects it).
                hdf_with_sizes = AddSizesToGradScoreHdfJob(grad_score_hdf=gen.out_hdf)
                hdf_with_sizes.add_alias(f"{name}-sizes")

                for align_opts in _ALIGN_OPTS_GRID:
                    align_name = f"align/{name}-{_name_for_dict(align_opts)}"
                    align = CalcAlignmentMetricsJob(
                        grad_score_hdf=hdf_with_sizes.out_hdf,
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
    # For comparison against the grad-align word-boundary numbers.
