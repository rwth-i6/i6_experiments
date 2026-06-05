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
from i6_experiments.users.zeyer.external_models.voxtral import (
    download_voxtral_mini_3b_model,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.voxtral import Voxtral
from i6_experiments.users.zeyer.external_models.canary_qwen import (
    download_canary_qwen_2_5b_model,
    download_qwen3_1_7b_model,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.canary_qwen import CanaryQwen
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.wav2vec2_ctc import Wav2Vec2Ctc
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.emformer_rnnt import EmformerRnnt
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.parakeet_rnnt import ParakeetRnnt
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.whisper import Whisper

from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.extract_in_grad_scores import (
    ExtractInGradsFromModelJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.extract_per_token_grads import (
    ExtractInGradsPerTokenJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.forced_align_baseline import (
    ForcedAlignBaselineJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.whisper_crossattn_align import (
    WhisperCrossAttnForcedAlignJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.buckeye_fine_dataset import (
    BuildBuckeyeFineDatasetJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.word_align_from_per_token_grads import (
    WordAlignFromPerTokenGradsJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.apptainer import (
    PullApptainerImageJob,
    ApptainerExeWrapperJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.extract_per_token_with_sep_grads import (
    ExtractInGradsPerTokenWithSepGradsJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.word_align_from_per_token_with_sep_grads import (
    WordAlignFromPerTokenWithSepGradsJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.word_align_from_per_token_grads_ext_metrics import (
    WordAlignFromPerTokenGradsExtMetricsJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.recog_from_model import (
    RecogFromModelJob,
)
from i6_experiments.users.zeyer.datasets.huggingface.extract_text import (
    ExtractWordDetailFromHuggingFaceDatasetJob,
)
from i6_experiments.users.zeyer.datasets.utils.sclite_generic_score import (
    sclite_score_hyps_to_ref,
)
from i6_experiments.users.zeyer.datasets.huggingface.open_asr_leaderboard import (
    text_dict_normalize_file,
    download_esb_datasets_test_only_sorted,
)
from i6_experiments.users.zeyer.datasets.huggingface.extract_text import (
    ExtractTextFromHuggingFaceDatasetJob,
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
    CalcAlignmentMetricsFromWordBoundariesJob,
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

# Signed reductions (sum / "dot") have large +/- outliers that wreck a plain
# softmax-over-time (the DP locks onto one outlier frame).
# Sweep the outlier-taming transforms the old exp2025_05_05_align recipe used:
# clip away negatives, normalize (std / abs-mean), and/or log-sigmoid squash.
_SIGNED_ALIGN_OPTS = [
    {"clip_scores": (1e-5, None), "apply_softmax_over_time": True, "blank_score": -6},
    {"norm_scores": "absmeanS", "clip_scores": (1e-5, None), "apply_softmax_over_time": True, "blank_score": -6},
    {"norm_scores": "stdmeanS", "apply_log_sigmoid": True, "apply_softmax_over_time": True, "blank_score": -6},
    {"norm_scores": "std0S", "apply_log_sigmoid": True, "apply_softmax_over_time": True, "blank_score": -6},
    {"norm_scores": "absmeanS", "apply_log_sigmoid": True, "apply_softmax_over_time": True, "blank_score": -6},
]


_DATASET_OFFSET_FACTORS = {"timit": 1, "buckeye": 1000}

# Isolated env overlays (NeMo / openai-whisper) that some adapters activate on sys.path.
# Passed into the model/job from here so the adapter defs stay free of infra-specific paths.
_NEMO_OVERLAY = "/home/az668407/work/canary-qwen-overlay"
_WHISPER_TS_OVERLAY = "/home/az668407/work/whisper-ts-overlay"


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
    char_level_skip_chars: Optional[List[str]] = None,
    grad_wrt: str = "speech_embeddings",
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
    if char_level_skip_chars is not None:
        extra["char_level_skip_chars"] = list(char_level_skip_chars)
    if grad_wrt != "speech_embeddings":
        extra["grad_wrt"] = grad_wrt
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
    # Fine-resolution Buckeye (float-ms GT, pre-segmented at silence into 2-25 s utterances, audio bundled);
    # supersedes the 62.5 ms-quantized nh0znoisung copy for long-form WBE eval.
    dl_ds_buckeye_fine = DownloadHuggingFaceRepoJobV2(repo_id="alexwengg/buckeye", repo_type="dataset")
    # The xet protocol's read-token endpoint gets HTTP-429 rate-limited on the shared
    # cluster IP; disable it so the download falls back to the standard LFS fetch.
    dl_ds_buckeye_fine.set_env("HF_HUB_DISABLE_XET", "1")
    tk.register_output("buckeye-fine-raw", dl_ds_buckeye_fine.out_hub_cache_dir)
    tk.register_output("buckeye-dataset", dl_ds_buckeye.out_hub_cache_dir)

    dl_whisper = DownloadHuggingFaceRepoJobV2(repo_id="openai/whisper-base", repo_type="model")
    dl_parakeet_rnnt = DownloadHuggingFaceRepoJobV2(repo_id="nvidia/parakeet-rnnt-1.1b", repo_type="model")
    dl_parakeet_tdt = DownloadHuggingFaceRepoJobV2(repo_id="nvidia/parakeet-tdt-0.6b-v2", repo_type="model")
    tk.register_output("whisper-base-model", dl_whisper.out_hub_cache_dir)

    # --- External baseline: MMS_FA neural-CTC forced alignment (WhisperX-style) ---
    # Forced-align the reference transcript -> per-word boundaries -> WBE via the
    # existing metric job. Runs in the active env (torchaudio + cached MMS_FA model).
    for _split in ("val", "test"):
        fa_base = ForcedAlignBaselineJob(dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_key=_split)
        fa_base.add_alias(f"baseline-mms_fa-timit-{_split}")
        fa_metric = CalcAlignmentMetricsFromWordBoundariesJob(
            word_boundaries_hdf=fa_base.out_hdf,
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key=_split,
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
        )
        tk.register_output(f"baseline-mms_fa-timit-{_split}-wbe.txt", fa_metric.out_wbe)

    # MMS_FA forced-align with per-seq time-stretch:
    # a finer emission grid for the SAME forced aligner
    # (does the 50 Hz frame quantization limit the 36.5 ms baseline?).
    # Boundaries map back to the original timeline inside the job.
    # method: vocoder (librosa phase vocoder, pitch-preserving, has artifacts) vs
    # resample (clean interpolation, pitch-shifted). Does the resolution help once
    # the vocoder artifacts are removed?
    for _fa_ts in (1.5, 2.0, 3.0):
        for _fa_m in ("vocoder", "resample"):
            _fa_msfx = "" if _fa_m == "vocoder" else f"-{_fa_m}"
            fa_ts = ForcedAlignBaselineJob(
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                audio_time_stretch=_fa_ts,
                time_stretch_method=_fa_m,
            )
            fa_ts.add_alias(f"baseline-mms_fa-timit-val-ts{_fa_ts}{_fa_msfx}")
            fa_ts_metric = CalcAlignmentMetricsFromWordBoundariesJob(
                word_boundaries_hdf=fa_ts.out_hdf,
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            )
            tk.register_output(f"baseline-mms_fa-timit-val-ts{_fa_ts}{_fa_msfx}-wbe.txt", fa_ts_metric.out_wbe)

    # Whisper-native forced alignment (cross-attention DTW) --
    # the same-model baseline for our grad-align ON Whisper (vs the model's own alignment method).
    whisper_fa = WhisperCrossAttnForcedAlignJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_key="val", overlay=_WHISPER_TS_OVERLAY
    )
    whisper_fa.add_alias("baseline-whisper-crossattn-timit-val")
    whisper_fa_metric = CalcAlignmentMetricsFromWordBoundariesJob(
        word_boundaries_hdf=whisper_fa.out_hdf,
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
    )
    tk.register_output("baseline-whisper-crossattn-timit-val-wbe.txt", whisper_fa_metric.out_wbe)

    # --- Wav2Vec2-CTC grad-align: OUR method on the SAME model the MMS_FA
    # baseline above uses (wav2vec2 + char-level CTC). Aligning one model two
    # ways disentangles "different model" from "different alignment method".
    # Grad target: the feature-extractor output (~50 Hz), the model's own
    # emission rate; per-token score is the CTC partial score (incl. blank).
    w2v_ctc_cfg = rf.build_dict(Wav2Vec2Ctc)
    for _w_mgi, _w_attr, _w_ga in [(False, "L2", "L2_grad"), (True, "L2", "L2_e_grad"), (False, "L1", "L1_grad")]:
        w2v_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=w2v_ctc_cfg,
            mult_grad_by_inputs=_w_mgi,
            attr_reduction=_w_attr,
        )
        w2v_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        w2v_name = f"wav2vec2ctc-featext-timit-val-{_w_ga}-pertoken"
        w2v_extract.add_alias(w2v_name)
        tk.register_output(f"{w2v_name}.hdf", w2v_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{w2v_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=w2v_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # Port the in-house CTC grad-align winning combo (blankStopGrad +
    # inclBlankState + low-p norm) to Wav2Vec2-CTC. inclBlankState is the
    # adapter default; here we add stop_grad_blank and sweep the reduction p
    # to isolate which lever closes the gap to CTC-forced-align (0.0365).
    # (In the in-house ablation each lever barely helped alone, but the three
    # together dropped boundary error ~11%; grad*input hurt, so mgi stays off.)
    w2v_ctc_blankstop_cfg = rf.build_dict(Wav2Vec2Ctc, stop_grad_blank=True)
    w2v_variants = [
        (w2v_ctc_blankstop_cfg, "blankstop-L0.1", "L0.1"),
        (w2v_ctc_blankstop_cfg, "blankstop-L0.5", "L0.5"),
        (w2v_ctc_blankstop_cfg, "blankstop-L1", "L1"),
        (w2v_ctc_blankstop_cfg, "blankstop-L2", "L2"),
        (w2v_ctc_cfg, "L0.1", "L0.1"),
        (w2v_ctc_cfg, "L0.5", "L0.5"),
    ]
    for _cfg, _tag, _attr in w2v_variants:
        w2vs_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=_cfg,
            mult_grad_by_inputs=False,
            attr_reduction=_attr,
        )
        w2vs_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        w2vs_name = f"wav2vec2ctc-featext-timit-val-{_tag}_grad-pertoken"
        w2vs_extract.add_alias(w2vs_name)
        tk.register_output(f"{w2vs_name}.hdf", w2vs_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{w2vs_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=w2vs_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # CTC prefix-score modes x blank_score DP sweep on Wav2Vec2-CTC. The
    # reduction was inert and stop_grad_blank hurt, so the live levers are
    # (a) the partial-score telescoping (include_next_blank), which changes the
    # grad scores, and (b) the DP blank_score, the most sensitive align knob in
    # the in-house setup (best ~-6 there, but grad-target-specific). Reduction
    # fixed to L2 (inert); mult off (grad*input hurt). -6 is already in the grid.
    _w2v_bs_sweep = [{"apply_softmax_over_time": True, "blank_score": _bs} for _bs in (-2, -3, -4, -5, -7, -8)]
    w2v_prefix_cfgs = [
        ("inbTrue", w2v_ctc_cfg),
        ("inbFalse", rf.build_dict(Wav2Vec2Ctc, include_next_blank=False)),
        ("inbBoth", rf.build_dict(Wav2Vec2Ctc, include_next_blank="both")),
        ("inbBothPrev", rf.build_dict(Wav2Vec2Ctc, include_next_blank="both_prev")),
    ]
    for _ptag, _pcfg in w2v_prefix_cfgs:
        w2vp_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=_pcfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
        )
        w2vp_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        w2vp_name = f"wav2vec2ctc-featext-timit-val-{_ptag}-L2_grad-pertoken"
        w2vp_extract.add_alias(w2vp_name)
        tk.register_output(f"{w2vp_name}.hdf", w2vp_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID + _w2v_bs_sweep:
            align_name = f"align/{w2vp_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=w2vp_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # grad_wrt = raw waveform: backprop through the whole conv feature-extractor
    # to the 16 kHz input (leaf reshaped to [n_frames, 320] for per-frame
    # pooling). The finest input-level grad target -- tests whether moving off
    # the feature-extractor output (which leaks saliency into silence) toward
    # the raw input sharpens the boundaries.
    w2v_raw_cfg = rf.build_dict(Wav2Vec2Ctc, grad_wrt="raw_waveform")
    w2v_raw_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=w2v_raw_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
    )
    w2v_raw_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    w2v_raw_name = "wav2vec2ctc-rawwav-timit-val-L2_grad-pertoken"
    w2v_raw_extract.add_alias(w2v_raw_name)
    tk.register_output(f"{w2v_raw_name}.hdf", w2v_raw_extract.out_hdf)
    for align_opts in _ALIGN_OPTS_GRID + [{"apply_softmax_over_time": True, "blank_score": _b} for _b in (-4, -5)]:
        align_name = f"align/{w2v_raw_name}-{_name_for_dict(align_opts)}"
        align = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=w2v_raw_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=align_opts,
        )
        align.add_alias(align_name)
        tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # grad_wrt sweep: conv blocks 0-5 (conv6 == feat_extract_out, already done at
    # ~0.073), the feature_projection LayerNorm (512) + Linear (1024, transformer
    # input), and raw-waveform at several resolutions (pool=1 -> full 16 kHz
    # sample-level alignment, pool=80/16 -> finer-than-50 Hz with RMS smoothing).
    # Tests which attribution surface localizes word boundaries best.
    _w2v_gradwrt_sweep = [(f"conv{_i}", rf.build_dict(Wav2Vec2Ctc, grad_wrt=f"conv{_i}")) for _i in range(6)] + [
        ("fproj_ln", rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_layernorm")),
        ("fproj_out", rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out")),
        ("rawwav-pool80", rf.build_dict(Wav2Vec2Ctc, grad_wrt="raw_waveform", raw_pool=80)),
        ("rawwav-pool16", rf.build_dict(Wav2Vec2Ctc, grad_wrt="raw_waveform", raw_pool=16)),
        ("rawwav-pool1", rf.build_dict(Wav2Vec2Ctc, grad_wrt="raw_waveform", raw_pool=1)),
    ]
    _w2v_gw_aligns = _ALIGN_OPTS_GRID + [{"apply_softmax_over_time": True, "blank_score": -5}]
    for _gwtag, _gwcfg in _w2v_gradwrt_sweep:
        gw_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=_gwcfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
        )
        gw_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        gw_name = f"wav2vec2ctc-{_gwtag}-timit-val-L2_grad-pertoken"
        gw_extract.add_alias(gw_name)
        tk.register_output(f"{gw_name}.hdf", gw_extract.out_hdf)
        for align_opts in _w2v_gw_aligns:
            align_name = f"align/{gw_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=gw_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # === audio-energy filter, grad*input on conv/raw, attention-disabled grads ===
    def _w2v_aligns(extract_job, base_name, energy_pows=(0.0,)):
        for _ep in energy_pows:
            _ao = {"apply_softmax_over_time": True, "blank_score": -5}
            _sfx = _name_for_dict(_ao) + (f"-en{_ep}" if _ep else "")
            _al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=extract_job.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_ao,
                audio_energy_pow=_ep,
            )
            _al.add_alias(f"align/{base_name}-{_sfx}")
            tk.register_output(f"align/{base_name}-{_sfx}-wbe.txt", _al.out_wbe)

    # Audio-energy filter (grad x energy^pow) on the best existing surfaces.
    for _en_cfg, _en_name in [
        (rf.build_dict(Wav2Vec2Ctc), "wav2vec2ctc-featext-timit-val-L2_grad-pertoken"),
        (rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out"), "wav2vec2ctc-fproj_out-timit-val-L2_grad-pertoken"),
    ]:
        _en_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=_en_cfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
        )
        _en_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _w2v_aligns(_en_ex, _en_name, energy_pows=(0.3, 0.5, 1.0))

    # grad*input (mult) on conv/raw surfaces, and self-attention-disabled grads.
    # On raw, grad*input = grad x audio (the energy weighting); each variant also
    # gets an explicit energy-filter align (en0.5).
    _w2v_more = [
        ("conv1-gi", rf.build_dict(Wav2Vec2Ctc, grad_wrt="conv1"), True),
        ("conv2-gi", rf.build_dict(Wav2Vec2Ctc, grad_wrt="conv2"), True),
        ("conv3-gi", rf.build_dict(Wav2Vec2Ctc, grad_wrt="conv3"), True),
        ("rawwav-gi", rf.build_dict(Wav2Vec2Ctc, grad_wrt="raw_waveform"), True),
        ("rawwav-pool1-gi", rf.build_dict(Wav2Vec2Ctc, grad_wrt="raw_waveform", raw_pool=1), True),
        ("noattn_value", rf.build_dict(Wav2Vec2Ctc, disable_self_attention="value"), False),
        ("noattn_zero", rf.build_dict(Wav2Vec2Ctc, disable_self_attention="zero"), False),
        (
            "fproj_out-noattn_value",
            rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", disable_self_attention="value"),
            False,
        ),
    ]
    for _mtag, _mcfg, _mgi in _w2v_more:
        _mex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=_mcfg,
            mult_grad_by_inputs=_mgi,
            attr_reduction="L2",
        )
        _mex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _mname = f"wav2vec2ctc-{_mtag}-timit-val-{'L2_e' if _mgi else 'L2'}_grad-pertoken"
        _mex.add_alias(_mname)
        tk.register_output(f"{_mname}.hdf", _mex.out_hdf)
        _w2v_aligns(_mex, _mname, energy_pows=(0.0, 0.5))

    # Decoupled silence combo (best so far, full-val ~58.4): energy-weighted
    # tokens (edges) + std-margin self-calibrating blank from the ORIGINAL
    # tokens (interior). zsk = blank_grad_zscore_kappa. On the two best surfaces.
    for _dc_cfg, _dc_name in [
        (rf.build_dict(Wav2Vec2Ctc), "wav2vec2ctc-featext-timit-val-L2_grad-pertoken"),
        (rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out"), "wav2vec2ctc-fproj_out-timit-val-L2_grad-pertoken"),
    ]:
        _dc_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=_dc_cfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
        )
        _dc_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        for _ep, _kap in [(0.0, 2.0), (0.3, 1.0), (0.5, 1.0)]:
            _ao = {"apply_softmax_over_time": True, "blank_score": -5}
            _sfx = _name_for_dict(_ao) + (f"-en{_ep}" if _ep else "") + f"-zsk{_kap}"
            _al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_dc_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_ao,
                audio_energy_pow=_ep,
                blank_grad_zscore_kappa=_kap,
            )
            _al.add_alias(f"align/{_dc_name}-{_sfx}")
            tk.register_output(f"align/{_dc_name}-{_sfx}-wbe.txt", _al.out_wbe)
        # Explicit silence label (whisper-timestamped VAD analog):
        # an energy-driven blank emission beta_t = mean_t - scale*z(E_t)*std_t,
        # optionally with energy-weighted tokens too.
        # sil = blank_silence_energy_scale.
        for _ep, _sc in [(0.0, 0.5), (0.0, 1.0), (0.0, 2.0), (0.5, 1.0), (0.5, 2.0), (0.3, 1.0)]:
            _ao = {"apply_softmax_over_time": True, "blank_score": -5}
            _sfx = _name_for_dict(_ao) + (f"-en{_ep}" if _ep else "") + f"-sil{_sc}"
            _al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_dc_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_ao,
                audio_energy_pow=_ep,
                blank_silence_energy_scale=_sc,
            )
            _al.add_alias(f"align/{_dc_name}-{_sfx}")
            tk.register_output(f"align/{_dc_name}-{_sfx}-wbe.txt", _al.out_wbe)

    # Wav2Vec2 grad-align with per-seq time-stretch: a finer (sub-20ms) saliency grid for the SAME model.
    # Mirror of the MMS_FA-forced-align time-stretch,
    # so we can compare whether the resolution gain helps grad-align and forced-align the same way.
    # Best surface (feat_proj_out) + best align variants.
    # New GPU extract per stretch factor (~Sx longer sequences).
    # method: vocoder (phase vocoder, has artifacts) vs resample (clean interpolation,
    # pitch-shifted). time_stretch_method is opt-in via **extra so the finished vocoder
    # extracts keep their hash (build_dict only includes passed kwargs).
    for _ts in (1.5, 2.0, 3.0):
        for _ts_m in ("vocoder", "resample"):
            _ts_extra = {} if _ts_m == "vocoder" else {"time_stretch_method": "resample"}
            _ts_msfx = "" if _ts_m == "vocoder" else f"-{_ts_m}"
            _ts_cfg = rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", audio_time_stretch=_ts, **_ts_extra)
            _ts_ex = ExtractInGradsPerTokenJob(
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                model_config=_ts_cfg,
                mult_grad_by_inputs=False,
                attr_reduction="L2",
            )
            _ts_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            _ts_name = f"wav2vec2ctc-fproj_out-timit-val-L2_grad-pertoken-ts{_ts}{_ts_msfx}"
            _ts_ex.add_alias(_ts_name)
            tk.register_output(f"{_ts_name}.hdf", _ts_ex.out_hdf)
            _w2v_aligns(_ts_ex, _ts_name, energy_pows=(0.0, 0.5))
            for _ts_ep, _ts_sc in [(0.5, 1.0), (0.0, 2.0)]:
                _ts_ao = {"apply_softmax_over_time": True, "blank_score": -5}
                _ts_sfx = _name_for_dict(_ts_ao) + (f"-en{_ts_ep}" if _ts_ep else "") + f"-sil{_ts_sc}"
                _ts_al = WordAlignFromPerTokenGradsJob(
                    grad_score_hdf=_ts_ex.out_hdf,
                    grad_score_key="data",
                    dataset_dir=dl_ds_timit.out_hub_cache_dir,
                    dataset_key="val",
                    dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                    align_opts=_ts_ao,
                    audio_energy_pow=_ts_ep,
                    blank_silence_energy_scale=_ts_sc,
                )
                _ts_al.add_alias(f"align/{_ts_name}-{_ts_sfx}")
                tk.register_output(f"align/{_ts_name}-{_ts_sfx}-wbe.txt", _ts_al.out_wbe)

    # --- Transducer (Emformer RNN-T): the 4th architecture family (CTC + AED/LLM
    # + RNN-T). torchaudio EMFORMER_RNNT_BASE_LIBRISPEECH; subword-level, log-mel
    # 100 Hz grad target; per-token score = logsumexp_t of the teacher-forced
    # joiner emission. Apply the (universal) energy filter + the decoupled combo.
    rnnt_cfg = rf.build_dict(EmformerRnnt)
    rnnt_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=rnnt_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
    )
    rnnt_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rnnt_name = "emformer-rnnt-logmel-timit-val-L2_grad-pertoken"
    rnnt_extract.add_alias(rnnt_name)
    tk.register_output(f"{rnnt_name}.hdf", rnnt_extract.out_hdf)
    _w2v_aligns(rnnt_extract, rnnt_name, energy_pows=(0.0, 0.5))
    for _re_ep, _re_kap in [(0.5, 1.0), (0.3, 1.0)]:
        _re_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _re_sfx = _name_for_dict(_re_ao) + f"-en{_re_ep}-zsk{_re_kap}"
        _re_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=rnnt_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_re_ao,
            audio_energy_pow=_re_ep,
            blank_grad_zscore_kappa=_re_kap,
        )
        _re_al.add_alias(f"align/{rnnt_name}-{_re_sfx}")
        tk.register_output(f"align/{rnnt_name}-{_re_sfx}-wbe.txt", _re_al.out_wbe)
    for _re_ep, _re_sc in [(0.5, 1.0), (0.3, 1.0)]:
        _re_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _re_sfx = _name_for_dict(_re_ao) + f"-en{_re_ep}-sil{_re_sc}"
        _re_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=rnnt_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_re_ao,
            audio_energy_pow=_re_ep,
            blank_silence_energy_scale=_re_sc,
        )
        _re_al.add_alias(f"align/{rnnt_name}-{_re_sfx}")
        tk.register_output(f"align/{rnnt_name}-{_re_sfx}-wbe.txt", _re_al.out_wbe)

    # RNN-T with the PROPER transducer prefix-score (RNN-T forward; analog of the CTC partial score)
    # instead of the crude logsumexp-emission --
    # should localize better than the emission baseline (158 ms).
    # + energy/decoupled.
    rnnt_px_cfg = rf.build_dict(EmformerRnnt, per_token_score="prefix")
    rnnt_px_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=rnnt_px_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
    )
    rnnt_px_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rnnt_px_name = "emformer-rnnt-prefix-logmel-timit-val-L2_grad-pertoken"
    rnnt_px_extract.add_alias(rnnt_px_name)
    tk.register_output(f"{rnnt_px_name}.hdf", rnnt_px_extract.out_hdf)
    _w2v_aligns(rnnt_px_extract, rnnt_px_name, energy_pows=(0.0, 0.5))
    for _rpx_ep, _rpx_kap in [(0.5, 1.0), (0.3, 1.0)]:
        _rpx_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _rpx_sfx = _name_for_dict(_rpx_ao) + f"-en{_rpx_ep}-zsk{_rpx_kap}"
        _rpx_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=rnnt_px_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_rpx_ao,
            audio_energy_pow=_rpx_ep,
            blank_grad_zscore_kappa=_rpx_kap,
        )
        _rpx_al.add_alias(f"align/{rnnt_px_name}-{_rpx_sfx}")
        tk.register_output(f"align/{rnnt_px_name}-{_rpx_sfx}-wbe.txt", _rpx_al.out_wbe)
    # Energy-driven silence (sil) vs the grad-background z-score (zsk) above:
    # the Parakeet transducers LIKE sil, so test it apples-to-apples on the Emformer.
    for _rpx_ep, _rpx_sc in [(0.5, 1.0), (0.3, 1.0)]:
        _rpx_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _rpx_sfx = _name_for_dict(_rpx_ao) + f"-en{_rpx_ep}-sil{_rpx_sc}"
        _rpx_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=rnnt_px_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_rpx_ao,
            audio_energy_pow=_rpx_ep,
            blank_silence_energy_scale=_rpx_sc,
        )
        _rpx_al.add_alias(f"align/{rnnt_px_name}-{_rpx_sfx}")
        tk.register_output(f"align/{rnnt_px_name}-{_rpx_sfx}-wbe.txt", _rpx_al.out_wbe)

    # Parakeet RNN-T 1.1B (NeMo FastConformer-Transducer): a STRONG leaderboard-grade
    # transducer, vs the tiny Emformer base above (139 ms).
    # Same proper RNN-T prefix-score; log-mel grad target (~100 Hz). + energy/silence.
    pk_cfg = rf.build_dict(
        ParakeetRnnt, model_dir=dl_parakeet_rnnt.out_hub_cache_dir, per_token_score="prefix", overlay_path=_NEMO_OVERLAY
    )
    pk_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=pk_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
    )
    pk_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pk_name = "parakeet-rnnt-1.1b-logmel-timit-val-L2_grad-pertoken"
    pk_extract.add_alias(pk_name)
    tk.register_output(f"{pk_name}.hdf", pk_extract.out_hdf)
    _w2v_aligns(pk_extract, pk_name, energy_pows=(0.0, 0.5))
    for _pk_ep, _pk_sc in [(0.5, 1.0), (0.3, 1.0)]:
        _pk_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _pk_sfx = _name_for_dict(_pk_ao) + f"-en{_pk_ep}-sil{_pk_sc}"
        _pk_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=pk_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_pk_ao,
            audio_energy_pow=_pk_ep,
            blank_silence_energy_scale=_pk_sc,
        )
        _pk_al.add_alias(f"align/{pk_name}-{_pk_sfx}")
        tk.register_output(f"align/{pk_name}-{_pk_sfx}-wbe.txt", _pk_al.out_wbe)

    # Parakeet TDT 0.6B v2 (NeMo Token-and-Duration Transducer): the top transducer
    # on the Open ASR Leaderboard. Same adapter; the joint appends duration logits,
    # so we score the TOKEN emission (per_token_score='emission'); a duration-aware
    # TDT-forward prefix-score is a follow-up. + energy/silence.
    tdt_cfg = rf.build_dict(
        ParakeetRnnt,
        model_dir=dl_parakeet_tdt.out_hub_cache_dir,
        per_token_score="emission",
        overlay_path=_NEMO_OVERLAY,
    )
    tdt_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=tdt_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
    )
    tdt_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    tdt_name = "parakeet-tdt-0.6b-v2-logmel-timit-val-L2_grad-pertoken"
    tdt_extract.add_alias(tdt_name)
    tk.register_output(f"{tdt_name}.hdf", tdt_extract.out_hdf)
    _w2v_aligns(tdt_extract, tdt_name, energy_pows=(0.0, 0.5))
    for _tdt_ep, _tdt_sc in [(0.5, 1.0), (0.3, 1.0)]:
        _tdt_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _tdt_sfx = _name_for_dict(_tdt_ao) + f"-en{_tdt_ep}-sil{_tdt_sc}"
        _tdt_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=tdt_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_tdt_ao,
            audio_energy_pow=_tdt_ep,
            blank_silence_energy_scale=_tdt_sc,
        )
        _tdt_al.add_alias(f"align/{tdt_name}-{_tdt_sfx}")
        tk.register_output(f"align/{tdt_name}-{_tdt_sfx}-wbe.txt", _tdt_al.out_wbe)

    # --- Whisper AED (classic encoder-decoder, distinct from the LLM-decoders;
    # also the basis of the WhisperX baseline). Autoregressive -> per-token score
    # = direct log p(y_i|y_<i, audio). Log-mel 100 Hz grad target. + energy/decoupled.
    whisper_cfg = rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir)
    whisper_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=whisper_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
    )
    whisper_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    whisper_name = "whisper-base-logmel-timit-val-L2_grad-pertoken"
    whisper_extract.add_alias(whisper_name)
    tk.register_output(f"{whisper_name}.hdf", whisper_extract.out_hdf)
    _w2v_aligns(whisper_extract, whisper_name, energy_pows=(0.0, 0.5))
    for _wh_ep, _wh_kap in [(0.5, 1.0), (0.3, 1.0)]:
        _wh_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _wh_sfx = _name_for_dict(_wh_ao) + f"-en{_wh_ep}-zsk{_wh_kap}"
        _wh_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=whisper_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_wh_ao,
            audio_energy_pow=_wh_ep,
            blank_grad_zscore_kappa=_wh_kap,
        )
        _wh_al.add_alias(f"align/{whisper_name}-{_wh_sfx}")
        tk.register_output(f"align/{whisper_name}-{_wh_sfx}-wbe.txt", _wh_al.out_wbe)
    # Explicit silence label for Whisper (the same-model analog of what whisper-timestamped does with VAD):
    # an energy-driven blank emission beta_t = mean_t - scale*z(E_t)*std_t,
    # +/- energy-weighted tokens.
    for _wh_ep, _wh_sc in [(0.0, 0.5), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0), (0.5, 1.0), (0.5, 2.0)]:
        _wh_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _wh_sfx = _name_for_dict(_wh_ao) + (f"-en{_wh_ep}" if _wh_ep else "") + f"-sil{_wh_sc}"
        _wh_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=whisper_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_wh_ao,
            audio_energy_pow=_wh_ep,
            blank_silence_energy_scale=_wh_sc,
        )
        _wh_al.add_alias(f"align/{whisper_name}-{_wh_sfx}")
        tk.register_output(f"align/{whisper_name}-{_wh_sfx}-wbe.txt", _wh_al.out_wbe)

    # Whisper char-level (per-char teacher-forced scoring;
    # the big win for the LLM decoders, e.g. Phi4-MM 0.190->0.087).
    # Single-token-per-char lookup so BPE doesn't re-merge;
    # space (id 220) separates words as context.
    # Same log-mel grad target. + energy + explicit-silence blank.
    whisper_char_cfg = rf.build_dict(
        Whisper, model_dir=dl_whisper.out_hub_cache_dir, char_level=True, char_level_sep=" "
    )
    whisper_char_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=whisper_char_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
    )
    whisper_char_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    whisper_char_name = "whisper-base-logmel-timit-val-L2_grad-pertoken-charlev-spc"
    whisper_char_extract.add_alias(whisper_char_name)
    tk.register_output(f"{whisper_char_name}.hdf", whisper_char_extract.out_hdf)
    _w2v_aligns(whisper_char_extract, whisper_char_name, energy_pows=(0.0, 0.5))

    # SmoothGrad ablation (attribution axis A) on the headline char-level Whisper surface: average the
    # per-token gradient over noise_n_samples noisy forward+backward passes (Gaussian noise on the raw
    # waveform). Tests whether saliency denoising sharpens boundaries vs the single-pass 47 ms baseline.
    for _sg_std, _sg_n in [(0.01, 8), (0.03, 8)]:
        _sg_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=whisper_char_cfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
            noise_std=_sg_std,
            noise_n_samples=_sg_n,
        )
        _sg_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _sg_name = f"{whisper_char_name}-smoothgrad-std{_sg_std}-n{_sg_n}"
        _sg_ex.add_alias(_sg_name)
        tk.register_output(f"{_sg_name}.hdf", _sg_ex.out_hdf)
        _sg_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _sg_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_sg_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_sg_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
        )
        _sg_nm = f"align/{_sg_name}-{_name_for_dict(_sg_ao)}-en0.5-sil1.0"
        _sg_al.add_alias(_sg_nm)
        tk.register_output(f"{_sg_nm}-wbe.txt", _sg_al.out_wbe)
    for _whc_ep, _whc_kap in [(0.5, 1.0), (0.3, 1.0)]:
        _whc_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _whc_sfx = _name_for_dict(_whc_ao) + f"-en{_whc_ep}-zsk{_whc_kap}"
        _whc_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=whisper_char_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_whc_ao,
            audio_energy_pow=_whc_ep,
            blank_grad_zscore_kappa=_whc_kap,
        )
        _whc_al.add_alias(f"align/{whisper_char_name}-{_whc_sfx}")
        tk.register_output(f"align/{whisper_char_name}-{_whc_sfx}-wbe.txt", _whc_al.out_wbe)
    for _whc_ep, _whc_sc in [(0.0, 1.0), (0.5, 1.0)]:
        _whc_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _whc_sfx = _name_for_dict(_whc_ao) + (f"-en{_whc_ep}" if _whc_ep else "") + f"-sil{_whc_sc}"
        _whc_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=whisper_char_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_whc_ao,
            audio_energy_pow=_whc_ep,
            blank_silence_energy_scale=_whc_sc,
        )
        _whc_al.add_alias(f"align/{whisper_char_name}-{_whc_sfx}")
        tk.register_output(f"align/{whisper_char_name}-{_whc_sfx}-wbe.txt", _whc_al.out_wbe)

    # --- Phi4 encoder-output (~12.5 Hz) grad target, for completeness vs the default
    # log-mel (100 Hz) target. Word-level only (encoder-out is too coarse for char-level).
    phi4_enc_cfg = _phi4mm_model_config(dl_phi4mi_dir, grad_wrt="encoder_out")
    for _p4e_mgi, _p4e_attr, _p4e_ga in [(True, "L2", "L2_e_grad"), (False, "L2", "L2_grad")]:
        p4e_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=phi4_enc_cfg,
            mult_grad_by_inputs=_p4e_mgi,
            attr_reduction=_p4e_attr,
        )
        p4e_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        p4e_name = f"phi4mm-encoderout-timit-val-{_p4e_ga}-pertoken"
        p4e_extract.add_alias(p4e_name)
        tk.register_output(f"{p4e_name}.hdf", p4e_extract.out_hdf)
        # encoder-out (~12.5 Hz) is a confirmed dead-end (WBE 0.44); restrict to
        # the bs-6 align -- the bscalc backtrace crashes (s=-1) on its degenerate,
        # near-flat score matrix (a latent Aligner edge case, not worth chasing here).
        for align_opts in _ALIGN_OPTS_GRID[:1]:
            align_name = f"align/{p4e_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=p4e_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # --- Phi4-MM open recognition (WER sanity gate for hyp-mode alignment).
    # Prerequisite for evaluating against open-recognition baselines
    # (WhisperX, whisper-timestamped) on their native metric. If WER here is
    # high, hyp-mode alignment numbers won't be meaningful. Reuses i6_core's
    # ScliteJob via the zeyer sclite_score_hyps_to_ref helper.
    phi4mm_recog_cfg = _phi4mm_model_config(dl_phi4mi_dir)
    for split in ("val", "test"):
        recog = RecogFromModelJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key=split,
            model_config=phi4mm_recog_cfg,
        )
        recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        recog.add_alias(f"phi4mm-timit-{split}-recog")
        tk.register_output(f"phi4mm-timit-{split}-recog.hyps.txt.gz", recog.out_hyps_txt)
        refs = ExtractWordDetailFromHuggingFaceDatasetJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_split=split
        )
        refs.add_alias(f"phi4mm-timit-{split}-refs")
        # Normalize both sides with the same text normalizer before scoring,
        # so the WER reflects acoustic-side errors, not punctuation /
        # case / contraction-expansion differences. Swap the normalizer
        # function if a different convention is desired.
        hyps_norm = text_dict_normalize_file(recog.out_hyps_txt)
        refs_norm = text_dict_normalize_file(refs.out_text)
        score = sclite_score_hyps_to_ref(
            hyps_text_dict=hyps_norm,
            ref_text_dict=refs_norm,
            corpus_name=f"phi4mm-timit-{split}",
        )
        tk.register_output(f"phi4mm-timit-{split}-wer.txt", score.main_measure_value)
        tk.register_output(f"phi4mm-timit-{split}-wer-report", score.report)

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
        (
            "-charlev-spc-margin-k3",
            {"char_level": True, "char_level_sep": " ", "margin_grad": True, "margin_grad_k": 3},
        ),
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
            dl_ds=dl_ds_timit,
            ds_name="timit",
            split="val",
            phi4mm_cfg=phi4mm_cfg,
            attr="L2",
            mgi=True,
            grad_alias="L2_e_grad",
            suffix=variant_suffix,
        )

    # (b) pertoken-charlev-spc with other grad reductions, val.
    pt_csp_cfg = _phi4mm_model_config(dl_phi4mi_dir, char_level=True, char_level_sep=" ")
    for mgi, attr, grad_alias in [
        (True, "L1", "L1_e_grad"),
        (True, "L0.5", "L05_e_grad"),
        (True, "sum", "dot_e_grad"),
        # Plain grad (mult off) is best at 100 Hz;
        # Phi4's grad target is already ~100 Hz log-mel (input_audio_embeds),
        # so plain L2/L1 should beat the L2_e champion (0.087).
        (False, "L2", "L2_grad"),
        (False, "L1", "L1_grad"),
    ]:
        _wire_pertoken(
            dl_ds=dl_ds_timit,
            ds_name="timit",
            split="val",
            phi4mm_cfg=pt_csp_cfg,
            attr=attr,
            mgi=mgi,
            grad_alias=grad_alias,
            suffix="-charlev-spc",
        )

    # (c) pertoken-charlev-spc on test split, L2_e_grad.
    _wire_pertoken(
        dl_ds=dl_ds_timit,
        ds_name="timit",
        split="test",
        phi4mm_cfg=pt_csp_cfg,
        attr="L2",
        mgi=True,
        grad_alias="L2_e_grad",
        suffix="-charlev-spc",
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

    # The audio-energy silence filter transfers across models (local check: Phi4
    # charlev-spc 87->79 / 95->80). Wire it full-val on Phi4, both grad*input
    # (L2_e, = pt_csp_gen) and plain grad (L2). The std-margin blank is CTC-specific
    # so we only carry the energy filter here.
    _w2v_aligns(pt_csp_gen, "phi4mm-timit-val-L2_e_grad-pertoken-charlev-spc", energy_pows=(0.3, 0.5))
    _p4_l2_ex = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=pt_csp_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
    )
    _p4_l2_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _w2v_aligns(_p4_l2_ex, "phi4mm-timit-val-L2_grad-pertoken-charlev-spc", energy_pows=(0.3, 0.5))

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

    # Extended-metrics scoring for the champion pertoken-charlev-spc variant
    # (val + test). Adds mean/median WBE in ms, % within {10, 25, 50, 100} ms,
    # F1 @ 20 ms collar. Matches the table format used in Rousso et al. 2024
    # and Huang et al. 2024 -- direct comparability with MFA, WhisperX, MMS,
    # Charsiu, less-peaky-CTC, etc.
    for split in ("val", "test"):
        ext_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key=split,
            model_config=pt_csp_cfg,
            mult_grad_by_inputs=True,
            attr_reduction="L2",
        )
        ext_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        for align_opts in _ALIGN_OPTS_GRID:
            ext_name = f"phi4mm-timit-{split}-L2_e_grad-pertoken-charlev-spc-ext-{_name_for_dict(align_opts)}"
            ext = WordAlignFromPerTokenGradsExtMetricsJob(
                grad_score_hdf=ext_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key=split,
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            ext.add_alias(f"align/{ext_name}")
            tk.register_output(f"align/{ext_name}-wbe_ms.txt", ext.out_wbe_ms)
            tk.register_output(f"align/{ext_name}-median_wbe_ms.txt", ext.out_median_wbe_ms)
            tk.register_output(f"align/{ext_name}-pct_within_10ms.txt", ext.out_pct_within_10ms)
            tk.register_output(f"align/{ext_name}-pct_within_25ms.txt", ext.out_pct_within_25ms)
            tk.register_output(f"align/{ext_name}-pct_within_50ms.txt", ext.out_pct_within_50ms)
            tk.register_output(f"align/{ext_name}-pct_within_100ms.txt", ext.out_pct_within_100ms)
            tk.register_output(f"align/{ext_name}-f1_20ms.txt", ext.out_f1_20ms)
            tk.register_output(f"align/{ext_name}-report.txt", ext.out_report)
            tk.register_output(f"align/{ext_name}-word_boundaries.py.gz", ext.out_word_boundaries)

    # (e) Two follow-up ideas on top of pertoken-charlev-spc:
    #
    # 1) no-apos: drop the apostrophe char from each word (e.g. 'don't' -> 'dont')
    #    before char-level expansion. Apostrophe rows have weak/noisy grads in
    #    the per-token matrix (no audio for the punctuation char itself), and
    #    those rows force the Aligner to place tokens for non-acoustic events.
    # 2) with-sep: also compute per-token grads for the separator subwords
    #    sitting *between* words (the ' ' char in charlev-spc), and let the
    #    Aligner see them as explicit silence anchors.
    # 3) Combined: drop apostrophe AND use the separator grads.
    #
    # All three are val + L2_e_grad only for now.

    # 1) no-apos
    pt_csp_noapos_cfg = _phi4mm_model_config(
        dl_phi4mi_dir, char_level=True, char_level_sep=" ", char_level_skip_chars=["'"]
    )
    _wire_pertoken(
        dl_ds=dl_ds_timit,
        ds_name="timit",
        split="val",
        phi4mm_cfg=pt_csp_noapos_cfg,
        attr="L2",
        mgi=True,
        grad_alias="L2_e_grad",
        suffix="-charlev-spc-no-apos",
    )

    # 2) with-sep (and 3) no-apos + with-sep): use the new ExtractInGradsPerTokenWithSepGradsJob
    # + WordAlignFromPerTokenWithSepGradsJob.
    def _wire_pertoken_with_sep(*, dl_ds, ds_name, split, phi4mm_cfg, attr, mgi, grad_alias, suffix):
        gen = ExtractInGradsPerTokenWithSepGradsJob(
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
            align = WordAlignFromPerTokenWithSepGradsJob(
                grad_score_hdf=gen.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds.out_hub_cache_dir,
                dataset_key=split,
                dataset_offset_factors=_DATASET_OFFSET_FACTORS[ds_name],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    _wire_pertoken_with_sep(
        dl_ds=dl_ds_timit,
        ds_name="timit",
        split="val",
        phi4mm_cfg=pt_csp_cfg,
        attr="L2",
        mgi=True,
        grad_alias="L2_e_grad",
        suffix="-charlev-spc-with-sep",
    )
    _wire_pertoken_with_sep(
        dl_ds=dl_ds_timit,
        ds_name="timit",
        split="val",
        phi4mm_cfg=pt_csp_noapos_cfg,
        attr="L2",
        mgi=True,
        grad_alias="L2_e_grad",
        suffix="-charlev-spc-no-apos-with-sep",
    )

    # --- Voxtral Mini 3B smoke test ---------------------------------
    # First-pass integration: one extract + one recog on TIMIT val to
    # validate the adapter (audio-embed splicing, prompt construction,
    # log_probs path). No char_level yet -- per-token here means per-Mistral
    # subword, K=1 for most words.
    dl_voxtral = download_voxtral_mini_3b_model()
    voxtral_cfg = rf.build_dict(Voxtral, model_dir=dl_voxtral)

    # Verbatim-prompt variant. Voxtral default prompt produced ~6.9%
    # insertions on TIMIT val (LLM paraphrasing instead of transcribing).
    # "Verbatim" hint may suppress that. Separate hash from the default
    # variant so both numbers are kept side-by-side.
    voxtral_verbatim_cfg = rf.build_dict(
        Voxtral,
        model_dir=dl_voxtral,
        speech_prompt="Please transcribe the audio verbatim, exactly as spoken.",
    )
    for split in ("val",):
        vx_v_recog = RecogFromModelJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key=split,
            model_config=voxtral_verbatim_cfg,
        )
        vx_v_recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        vx_v_recog.add_alias(f"voxtral-verbatim-timit-{split}-recog")
        tk.register_output(f"voxtral-verbatim-timit-{split}-recog.hyps.txt.gz", vx_v_recog.out_hyps_txt)
        vx_v_refs = ExtractWordDetailFromHuggingFaceDatasetJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_split=split
        )
        vx_v_hyps_norm = text_dict_normalize_file(vx_v_recog.out_hyps_txt)
        vx_v_refs_norm = text_dict_normalize_file(vx_v_refs.out_text)
        vx_v_score = sclite_score_hyps_to_ref(
            hyps_text_dict=vx_v_hyps_norm,
            ref_text_dict=vx_v_refs_norm,
            corpus_name=f"voxtral-verbatim-timit-{split}",
        )
        tk.register_output(f"voxtral-verbatim-timit-{split}-wer.txt", vx_v_score.main_measure_value)
        tk.register_output(f"voxtral-verbatim-timit-{split}-wer-report", vx_v_score.report)

    vx_recog = RecogFromModelJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=voxtral_cfg,
    )
    vx_recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    vx_recog.add_alias("voxtral-timit-val-recog")
    tk.register_output("voxtral-timit-val-recog.hyps.txt.gz", vx_recog.out_hyps_txt)
    vx_refs = ExtractWordDetailFromHuggingFaceDatasetJob(dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_split="val")
    vx_refs.add_alias("voxtral-timit-val-refs")
    vx_hyps_norm = text_dict_normalize_file(vx_recog.out_hyps_txt)
    vx_refs_norm = text_dict_normalize_file(vx_refs.out_text)
    vx_score = sclite_score_hyps_to_ref(
        hyps_text_dict=vx_hyps_norm,
        ref_text_dict=vx_refs_norm,
        corpus_name="voxtral-timit-val",
    )
    tk.register_output("voxtral-timit-val-wer.txt", vx_score.main_measure_value)
    tk.register_output("voxtral-timit-val-wer-report", vx_score.report)

    # OpenASRLeaderboard consistency check: run Voxtral recog on a couple
    # of ESB test sets and compare WER to the leaderboard's reported numbers.
    # If our pipeline is consistent, we should land within ~0.5% of the
    # leaderboard's published Voxtral WER on the same data. Large gap =>
    # something's off (prompt, normalization, audio preproc).
    dl_esb = download_esb_datasets_test_only_sorted()
    # hf-audio/esb-datasets-test-only-sorted only contains:
    # ami, common_voice, earnings22, gigaspeech, librispeech, spgispeech,
    # voxpopuli (no tedlium). Use librispeech-test.clean for consistency check.
    for esb_name, esb_split in (("librispeech", "test.clean"),):
        vx_esb_recog = RecogFromModelJob(
            dataset_dir=dl_esb,
            dataset_name=esb_name,
            dataset_key=esb_split,
            model_config=voxtral_cfg,
            max_new_tokens=512,
        )
        vx_esb_recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        vx_esb_recog.add_alias(f"voxtral-esb-{esb_name}-{esb_split}-recog")
        tk.register_output(f"voxtral-esb-{esb_name}-{esb_split}-recog.hyps.txt.gz", vx_esb_recog.out_hyps_txt)
        vx_esb_refs = ExtractTextFromHuggingFaceDatasetJob(
            dataset_dir=dl_esb,
            dataset_name=esb_name,
            dataset_split=esb_split,
        )
        vx_esb_hyps_norm = text_dict_normalize_file(vx_esb_recog.out_hyps_txt)
        vx_esb_refs_norm = text_dict_normalize_file(vx_esb_refs.out_text)
        vx_esb_score = sclite_score_hyps_to_ref(
            hyps_text_dict=vx_esb_hyps_norm,
            ref_text_dict=vx_esb_refs_norm,
            corpus_name=f"voxtral-esb-{esb_name}-{esb_split}",
        )
        tk.register_output(f"voxtral-esb-{esb_name}-{esb_split}-wer.txt", vx_esb_score.main_measure_value)
        tk.register_output(f"voxtral-esb-{esb_name}-{esb_split}-wer-report", vx_esb_score.report)

    # --- Voxtral transcription-mode recogs (the leaderboard's mode) -------
    # The chat-mode recogs above use apply_chat_template -> instruct behavior
    # that paraphrases/reorders (inflated WER: 4.18% on LS-clean, mostly
    # insertions). The OpenASR leaderboard uses apply_transcription_request;
    # this config switches recog to that path for an apples-to-apples number.
    voxtral_transcribe_cfg = rf.build_dict(Voxtral, model_dir=dl_voxtral, recog_mode="transcription")
    vx_t_recog = RecogFromModelJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=voxtral_transcribe_cfg,
    )
    vx_t_recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    vx_t_recog.add_alias("voxtral-transcribe-timit-val-recog")
    tk.register_output("voxtral-transcribe-timit-val-recog.hyps.txt.gz", vx_t_recog.out_hyps_txt)
    vx_t_hyps_norm = text_dict_normalize_file(vx_t_recog.out_hyps_txt)
    vx_t_score = sclite_score_hyps_to_ref(
        hyps_text_dict=vx_t_hyps_norm,
        ref_text_dict=vx_refs_norm,
        corpus_name="voxtral-transcribe-timit-val",
    )
    tk.register_output("voxtral-transcribe-timit-val-wer.txt", vx_t_score.main_measure_value)
    tk.register_output("voxtral-transcribe-timit-val-wer-report", vx_t_score.report)

    vx_t_esb_recog = RecogFromModelJob(
        dataset_dir=dl_esb,
        dataset_name="librispeech",
        dataset_key="test.clean",
        model_config=voxtral_transcribe_cfg,
        max_new_tokens=512,
    )
    vx_t_esb_recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    vx_t_esb_recog.add_alias("voxtral-transcribe-esb-librispeech-test.clean-recog")
    tk.register_output("voxtral-transcribe-esb-librispeech-test.clean-recog.hyps.txt.gz", vx_t_esb_recog.out_hyps_txt)
    vx_t_esb_refs = ExtractTextFromHuggingFaceDatasetJob(
        dataset_dir=dl_esb, dataset_name="librispeech", dataset_split="test.clean"
    )
    vx_t_esb_hyps_norm = text_dict_normalize_file(vx_t_esb_recog.out_hyps_txt)
    vx_t_esb_refs_norm = text_dict_normalize_file(vx_t_esb_refs.out_text)
    vx_t_esb_score = sclite_score_hyps_to_ref(
        hyps_text_dict=vx_t_esb_hyps_norm,
        ref_text_dict=vx_t_esb_refs_norm,
        corpus_name="voxtral-transcribe-esb-librispeech-test.clean",
    )
    tk.register_output("voxtral-transcribe-esb-librispeech-test.clean-wer.txt", vx_t_esb_score.main_measure_value)
    tk.register_output("voxtral-transcribe-esb-librispeech-test.clean-wer-report", vx_t_esb_score.report)

    # --- Canary-Qwen 2.5B (NeMo SALM) recog ----------------------------
    # First pass: recog-only. Forward path (forced alignment) requires a
    # grad-able hook into SALM internals -- deferred.
    dl_canary = download_canary_qwen_2_5b_model()
    dl_qwen3 = download_qwen3_1_7b_model()
    canary_cfg = rf.build_dict(CanaryQwen, model_dir=dl_canary, llm_model_dir=dl_qwen3, version=3)

    # Canary log-mel grad target: 100 Hz input-feature grad via the new perception split.
    # Word-level plain-L2 first to validate the NeMo split; char-level follows once confirmed.
    canary_logmel_cfg = rf.build_dict(
        CanaryQwen, model_dir=dl_canary, llm_model_dir=dl_qwen3, grad_wrt="log_mel", version=5
    )
    cq_lm_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=canary_logmel_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
    )
    cq_lm_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cq_lm_name = "canary-qwen-logmel-timit-val-L2_grad-pertoken"
    cq_lm_extract.add_alias(cq_lm_name)
    tk.register_output(f"{cq_lm_name}.hdf", cq_lm_extract.out_hdf)
    for align_opts in _ALIGN_OPTS_GRID:
        align_name = f"align/{cq_lm_name}-{_name_for_dict(align_opts)}"
        align = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=cq_lm_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=align_opts,
        )
        align.add_alias(align_name)
        tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # Canary char-level + log-mel (100 Hz): the headline cross-model target.
    # ensure_audio_long_enough handles char-count > 18 Hz encoder frames in the splice;
    # the grad leaf is the 100 Hz mel, so the resolution should finally pay off here
    # (log-mel lost badly to the 18 Hz encoder-out at word level: 0.270 vs 0.187).
    canary_charlev_logmel_cfg = rf.build_dict(
        CanaryQwen,
        model_dir=dl_canary,
        llm_model_dir=dl_qwen3,
        grad_wrt="log_mel",
        char_level=True,
        char_level_sep=" ",
        ensure_audio_long_enough=True,
        version=5,
    )
    for cl_mgi, cl_attr, cl_alias in [(False, "L1", "L1_grad"), (False, "L2", "L2_grad")]:
        cq_cl_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=canary_charlev_logmel_cfg,
            mult_grad_by_inputs=cl_mgi,
            attr_reduction=cl_attr,
        )
        cq_cl_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        cq_cl_name = f"canary-qwen-charlev-spc-logmel-timit-val-{cl_alias}-pertoken"
        cq_cl_extract.add_alias(cq_cl_name)
        tk.register_output(f"{cq_cl_name}.hdf", cq_cl_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{cq_cl_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=cq_cl_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)
        # Audio-energy silence filter (transfers across models; Canary 133->106).
        _w2v_aligns(cq_cl_extract, cq_cl_name, energy_pows=(0.5, 1.0))

    # Consistency: Canary char + log-mel + stretch (same lever sweep as Voxtral).
    canary_charlev_logmel_st15_cfg = rf.build_dict(
        CanaryQwen,
        model_dir=dl_canary,
        llm_model_dir=dl_qwen3,
        grad_wrt="log_mel",
        char_level=True,
        char_level_sep=" ",
        ensure_audio_long_enough=True,
        audio_time_stretch=1.5,
        version=5,
    )
    cq_cls_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=canary_charlev_logmel_st15_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L1",
    )
    cq_cls_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cq_cls_name = "canary-qwen-charlev-spc-logmel-st15-timit-val-L1_grad-pertoken"
    cq_cls_extract.add_alias(cq_cls_name)
    tk.register_output(f"{cq_cls_name}.hdf", cq_cls_extract.out_hdf)
    for align_opts in _ALIGN_OPTS_GRID:
        align_name = f"align/{cq_cls_name}-{_name_for_dict(align_opts)}"
        align = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=cq_cls_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=align_opts,
        )
        align.add_alias(align_name)
        tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    cq_recog = RecogFromModelJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=canary_cfg,
    )
    cq_recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cq_recog.add_alias("canary-qwen-timit-val-recog")
    tk.register_output("canary-qwen-timit-val-recog.hyps.txt.gz", cq_recog.out_hyps_txt)
    cq_refs = ExtractWordDetailFromHuggingFaceDatasetJob(dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_split="val")
    cq_hyps_norm = text_dict_normalize_file(cq_recog.out_hyps_txt)
    cq_refs_norm = text_dict_normalize_file(cq_refs.out_text)
    cq_score = sclite_score_hyps_to_ref(
        hyps_text_dict=cq_hyps_norm,
        ref_text_dict=cq_refs_norm,
        corpus_name="canary-qwen-timit-val",
    )
    tk.register_output("canary-qwen-timit-val-wer.txt", cq_score.main_measure_value)
    tk.register_output("canary-qwen-timit-val-wer-report", cq_score.report)

    # Dedicated version=3 config (fixed splice/n_audio_real) for the chat-mode
    # grad extract. voxtral_cfg stays at the default version so the already-finished voxtral recog jobs
    # keep their hashes and are not re-run.
    voxtral_grad_cfg = rf.build_dict(Voxtral, model_dir=dl_voxtral, version=3)
    vx_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=voxtral_grad_cfg,
        mult_grad_by_inputs=True,
        attr_reduction="L2",
    )
    vx_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    vx_extract_name = "voxtral-timit-val-L2_e_grad-pertoken"
    vx_extract.add_alias(vx_extract_name)
    tk.register_output(f"{vx_extract_name}.hdf", vx_extract.out_hdf)
    for align_opts in _ALIGN_OPTS_GRID:
        align_name = f"align/{vx_extract_name}-{_name_for_dict(align_opts)}"
        align = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=vx_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=align_opts,
        )
        align.add_alias(align_name)
        tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # --- Voxtral transcription-mode forced-align grad extract -------------
    # The chat-mode forward above gave poor WBE (0.47-0.57): instruct mode
    # diffuses audio attention. Redo the forced alignment in Voxtral's native
    # transcription mode (forward_mode="transcription"), which should localize
    # the per-token audio gradients -- same fix that dropped recog WER 10.2->2.6.
    # version=2: fixed _splice_audio_into_embeds to use get_audio_features'
    # projected output (pooler_output, 375 tokens) instead of pre-projection
    # encoder states (1500 rows). Old version misaligned audio in the splice.
    voxtral_transcribe_fwd_cfg = rf.build_dict(Voxtral, model_dir=dl_voxtral, forward_mode="transcription", version=3)
    for mgi, attr, grad_alias in (
        (True, "L2", "L2_e_grad"),
        (True, "L1", "L1_e_grad"),
        (True, "L0.5", "L05_e_grad"),
        (False, "L2", "L2_grad"),
        (False, "L1", "L1_grad"),
    ):
        vxt_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=voxtral_transcribe_fwd_cfg,
            mult_grad_by_inputs=mgi,
            attr_reduction=attr,
        )
        vxt_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        vxt_extract_name = f"voxtral-transcribe-timit-val-{grad_alias}-pertoken"
        vxt_extract.add_alias(vxt_extract_name)
        tk.register_output(f"{vxt_extract_name}.hdf", vxt_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{vxt_extract_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=vxt_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # --- Voxtral log-mel grad target (100 Hz) --------------------------------
    # Same model, but differentiate w.r.t. the log-mel input features instead
    # of the projected speech embeddings. This gives 100 Hz time resolution
    # (vs ~12.5 Hz for projected) and unblocks char-level alignment, which is
    # the testbed for the variants documented in projects/2026-05-23-grad-align.md.
    #
    # Initial L2_e_grad run gave WBE 0.179/0.191 vs 0.130/0.157 for projected --
    # log-mel was WORSE. Likely suspects: (a) `mult_grad_by_inputs` is questionable
    # for log-mel (signed/varying magnitudes), (b) L2-over-mel-bins may wash out
    # localization, (c) grad through full encoder is noisier. Sweep variants.
    voxtral_logmel_cfg = rf.build_dict(
        Voxtral,
        model_dir=dl_voxtral,
        forward_mode="transcription",
        grad_wrt="log_mel",
        version=4,
    )
    # Variants: cover both with/without input-mult and several reduction norms.
    # "e_grad" suffix = mult_grad_by_inputs=True (input * grad).
    # "_grad" suffix = plain grad.
    _voxtral_logmel_variants = [
        (True, "L2", "L2_e_grad"),  # initial run, WBE 0.179
        (False, "L2", "L2_grad"),  # plain grad, L2 over mel
        (False, "L1", "L1_grad"),  # plain grad, L1 over mel
        (False, "sum", "dot_grad"),  # plain grad, signed sum over mel
        (True, "L1", "L1_e_grad"),  # input * grad, L1
        (True, "sum", "dot_e_grad"),  # input * grad, signed sum
        (False, "L0.5", "L05_grad"),  # plain grad, L0.5 (sharper)
    ]
    for mgi, attr, grad_alias in _voxtral_logmel_variants:
        vxt_lm_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=voxtral_logmel_cfg,
            mult_grad_by_inputs=mgi,
            attr_reduction=attr,
        )
        vxt_lm_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        vxt_lm_extract_name = f"voxtral-transcribe-logmel-timit-val-{grad_alias}-pertoken"
        vxt_lm_extract.add_alias(vxt_lm_extract_name)
        tk.register_output(f"{vxt_lm_extract_name}.hdf", vxt_lm_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{vxt_lm_extract_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=vxt_lm_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)
        # Signed sum reductions ("dot") additionally get the signed-score align;
        # the magnitude grid above mishandles arbitrarily-signed scores.
        if attr == "sum":
            for align_opts in _SIGNED_ALIGN_OPTS:
                align_name = f"align/{vxt_lm_extract_name}-{_name_for_dict(align_opts)}"
                align = WordAlignFromPerTokenGradsJob(
                    grad_score_hdf=vxt_lm_extract.out_hdf,
                    grad_score_key="data",
                    dataset_dir=dl_ds_timit.out_hub_cache_dir,
                    dataset_key="val",
                    dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                    align_opts=align_opts,
                )
                align.add_alias(align_name)
                tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # --- Voxtral fixed audio time-stretch (variant 6) ------------------------
    # Stretch every audio by a fixed factor (>1 = slower) via librosa
    # phase-vocoder (pitch-preserving). Gives encoder more frames per syllable.
    # Stays on the projected-speech-embeddings grad target since that gave the
    # best WBE (0.130) in the previous sweep. WBE is reported on the ORIGINAL
    # audio timeline (the model internally maps stretched encoder frames back
    # via original n_samples).
    for stretch in (1.2, 1.5, 1.75, 2.0):
        suf = f"stretch{stretch:.1f}".replace(".", "")
        voxtral_stretch_cfg = rf.build_dict(
            Voxtral,
            model_dir=dl_voxtral,
            forward_mode="transcription",
            audio_time_stretch=stretch,
            version=5,
        )
        vxt_st_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=voxtral_stretch_cfg,
            mult_grad_by_inputs=True,
            attr_reduction="L2",
        )
        vxt_st_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        vxt_st_extract_name = f"voxtral-transcribe-{suf}-timit-val-L2_e_grad-pertoken"
        vxt_st_extract.add_alias(vxt_st_extract_name)
        tk.register_output(f"{vxt_st_extract_name}.hdf", vxt_st_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{vxt_st_extract_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=vxt_st_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # --- Voxtral conv1 grad target (100 Hz, d_model space) -------------------
    # Variant 4 in the project notes: grad w.r.t. Whisper encoder's conv1
    # output (captured via forward hook). Same 100 Hz time resolution as
    # log_mel but in the model's learned feature space (d_model=1280),
    # hopefully cleaner than raw log-mel grads.
    voxtral_conv1_cfg = rf.build_dict(
        Voxtral,
        model_dir=dl_voxtral,
        forward_mode="transcription",
        grad_wrt="encoder_conv1_out",
        version=7,
    )
    for mgi, attr, grad_alias in [
        (True, "L2", "L2_e_grad"),
        (False, "L2", "L2_grad"),
        (False, "L1", "L1_grad"),
    ]:
        vxt_c1_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=voxtral_conv1_cfg,
            mult_grad_by_inputs=mgi,
            attr_reduction=attr,
        )
        vxt_c1_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        vxt_c1_extract_name = f"voxtral-transcribe-conv1-timit-val-{grad_alias}-pertoken"
        vxt_c1_extract.add_alias(vxt_c1_extract_name)
        tk.register_output(f"{vxt_c1_extract_name}.hdf", vxt_c1_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{vxt_c1_extract_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=vxt_c1_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # --- Voxtral char-level + per-seq just-enough resample -------------------
    # Variant 12 (char-level vocab via per-char lookup, bypasses BPE merging)
    # + variant 5 (per-seq just-enough time-stretch when char-count exceeds
    # default encoder-frame count). Char-level alone might fail with S>T on
    # short fast-speech seqs, so the two go together. Stays on the projected
    # grad target (best WBE so far) for the initial test.
    voxtral_charlev_cfg = rf.build_dict(
        Voxtral,
        model_dir=dl_voxtral,
        forward_mode="transcription",
        char_level=True,
        char_level_sep=" ",
        ensure_audio_long_enough=True,
        version=8,
    )
    # Default L2_e plus plain-grad L2/L1 (plain grad is best at finer resolution).
    for mgi, attr, grad_alias in [(True, "L2", "L2_e_grad"), (False, "L2", "L2_grad"), (False, "L1", "L1_grad")]:
        vxt_cl_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=voxtral_charlev_cfg,
            mult_grad_by_inputs=mgi,
            attr_reduction=attr,
        )
        vxt_cl_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        vxt_cl_extract_name = f"voxtral-transcribe-charlev-spc-timit-val-{grad_alias}-pertoken"
        vxt_cl_extract.add_alias(vxt_cl_extract_name)
        tk.register_output(f"{vxt_cl_extract_name}.hdf", vxt_cl_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{vxt_cl_extract_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=vxt_cl_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # --- Voxtral char-level on 100 Hz grad targets (log_mel / conv1) ----------
    # Char-level wants fine time resolution.
    # Word-level 100 Hz (plain grad) already lands within ~0.02 of projected,
    # and char-level on projected is 0.108 --
    # so this tests whether char-level + 100 Hz finally beats char-level projected.
    # No ensure_audio_long_enough needed: at 100 Hz, frames >> chars always.
    # version=7 satisfies the char_level (>=7) and non-speech-embeddings (>=4) asserts;
    # it is an existing value, not a new bump -- the option combo is already hash-distinct.
    voxtral_charlev_logmel_cfg = rf.build_dict(
        Voxtral,
        model_dir=dl_voxtral,
        forward_mode="transcription",
        grad_wrt="log_mel",
        char_level=True,
        char_level_sep=" ",
        version=7,
    )
    voxtral_charlev_conv1_cfg = rf.build_dict(
        Voxtral,
        model_dir=dl_voxtral,
        forward_mode="transcription",
        grad_wrt="encoder_conv1_out",
        char_level=True,
        char_level_sep=" ",
        version=7,
    )
    # Stack the winning levers: char-level + 100 Hz + audio stretch.
    # Stretch helped at projected resolution, and plain-L1 is the best reduction so far;
    # sweep a couple of stretch factors on both 100 Hz targets.
    _charlev_stretch = []
    for _tgt, _gw in (("logmel", "log_mel"), ("conv1", "encoder_conv1_out")):
        for _st, _stk in ((1.5, "15"), (2.0, "20")):
            _st_cfg = rf.build_dict(
                Voxtral,
                model_dir=dl_voxtral,
                forward_mode="transcription",
                grad_wrt=_gw,
                char_level=True,
                char_level_sep=" ",
                audio_time_stretch=_st,
                version=7,
            )
            _charlev_stretch.append((_st_cfg, f"{_tgt}-st{_stk}", [(False, "L1", "L1_grad")]))
    # Plain grad (mult_grad_by_inputs=False) is best at 100 Hz;
    # keep one mult variant per target to confirm the mult-hurts pattern holds at char level.
    _charlev_100hz = [
        (
            voxtral_charlev_logmel_cfg,
            "logmel",
            [(False, "L2", "L2_grad"), (False, "L1", "L1_grad"), (True, "L2", "L2_e_grad")],
        ),
        (
            voxtral_charlev_conv1_cfg,
            "conv1",
            [(False, "L1", "L1_grad"), (False, "L2", "L2_grad"), (True, "L2", "L2_e_grad")],
        ),
    ] + _charlev_stretch
    for cl_cfg, cl_tname, cl_variants in _charlev_100hz:
        for mgi, attr, grad_alias in cl_variants:
            vxt_cl2_extract = ExtractInGradsPerTokenJob(
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                model_config=cl_cfg,
                mult_grad_by_inputs=mgi,
                attr_reduction=attr,
            )
            vxt_cl2_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            vxt_cl2_extract_name = f"voxtral-transcribe-charlev-spc-{cl_tname}-timit-val-{grad_alias}-pertoken"
            vxt_cl2_extract.add_alias(vxt_cl2_extract_name)
            tk.register_output(f"{vxt_cl2_extract_name}.hdf", vxt_cl2_extract.out_hdf)
            for align_opts in _ALIGN_OPTS_GRID:
                align_name = f"align/{vxt_cl2_extract_name}-{_name_for_dict(align_opts)}"
                align = WordAlignFromPerTokenGradsJob(
                    grad_score_hdf=vxt_cl2_extract.out_hdf,
                    grad_score_key="data",
                    dataset_dir=dl_ds_timit.out_hub_cache_dir,
                    dataset_key="val",
                    dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                    align_opts=align_opts,
                )
                align.add_alias(align_name)
                tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # --- SmoothGrad on Voxtral (variant 7) -----------------------------------
    # N=5 forward+backward passes per seq with Gaussian noise on raw audio;
    # averaged grad maps. Tested on the best projected-grad config (L2_e).
    # noise_std=0.01 is ~1% of typical waveform amplitude (signals in [-1,1]).
    for sg_n, sg_std, sg_alias in [
        (5, 0.01, "sg5-std001"),
        (5, 0.05, "sg5-std005"),
    ]:
        vxt_sg_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=voxtral_transcribe_fwd_cfg,
            mult_grad_by_inputs=True,
            attr_reduction="L2",
            noise_n_samples=sg_n,
            noise_std=sg_std,
        )
        vxt_sg_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        vxt_sg_name = f"voxtral-transcribe-timit-val-L2_e_grad-pertoken-{sg_alias}"
        vxt_sg_extract.add_alias(vxt_sg_name)
        tk.register_output(f"{vxt_sg_name}.hdf", vxt_sg_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{vxt_sg_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=vxt_sg_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # --- Canary-Qwen 2.5B grad-variant sweep on TIMIT val -----------------
    # A sensible subset (not the full 9x grid): the grad x input ("e_grad")
    # family at a few norms, plus one plain-grad contrast. Each extract runs
    # the full per-word backward sweep, so keep the count modest.
    _canary_grad_variants = [
        (True, "L2", "L2_e_grad"),
        (True, "L1", "L1_e_grad"),
        (True, "sum", "dot_e_grad"),
        (True, "L0.5", "L05_e_grad"),
        (False, "L2", "L2_grad"),
    ]
    for mgi, attr, grad_alias in _canary_grad_variants:
        cq_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=canary_cfg,
            mult_grad_by_inputs=mgi,
            attr_reduction=attr,
        )
        cq_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        cq_extract_name = f"canary-qwen-timit-val-{grad_alias}-pertoken"
        cq_extract.add_alias(cq_extract_name)
        tk.register_output(f"{cq_extract_name}.hdf", cq_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{cq_extract_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=cq_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # Prompt variant: ask for accurate verbatim transcription, with the
    # best-ish grad variant (L2_e) so the prompt effect is isolated.
    canary_acc_cfg = rf.build_dict(
        CanaryQwen,
        model_dir=dl_canary,
        llm_model_dir=dl_qwen3,
        speech_prompt="Transcribe the following exactly and accurately, word for word:",
        version=3,
    )
    cq_acc_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=canary_acc_cfg,
        mult_grad_by_inputs=True,
        attr_reduction="L2",
    )
    cq_acc_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cq_acc_name = "canary-qwen-acc-timit-val-L2_e_grad-pertoken"
    cq_acc_extract.add_alias(cq_acc_name)
    tk.register_output(f"{cq_acc_name}.hdf", cq_acc_extract.out_hdf)
    for align_opts in _ALIGN_OPTS_GRID:
        align_name = f"align/{cq_acc_name}-{_name_for_dict(align_opts)}"
        align = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=cq_acc_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=align_opts,
        )
        align.add_alias(align_name)
        tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    cq_acc_recog = RecogFromModelJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=canary_acc_cfg,
    )
    cq_acc_recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Canary char-level variants (mirroring Phi4-MM; the best Phi4-MM variant
    # was charlev-spc at WBE 0.087 vs 0.190 for word-level Canary)
    # Qwen char-level: use per-char vocab lookup (not string tokenization)
    # so BPE merging is bypassed. All sep-based variants should now work.
    for char_suf, char_extra in [
        ("-charlev-spc", {"char_level": True, "char_level_sep": " "}),
        ("-charlev-spc-upper", {"char_level": True, "char_level_sep": " ", "char_level_case": "upper"}),
        ("-charlev-spc-title", {"char_level": True, "char_level_sep": " ", "char_level_case": "title"}),
    ]:
        cq_char_cfg = rf.build_dict(
            CanaryQwen,
            model_dir=dl_canary,
            llm_model_dir=dl_qwen3,
            ensure_audio_long_enough=True,
            version=5,
            **char_extra,
        )
        cq_char_extract = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=cq_char_cfg,
            mult_grad_by_inputs=True,
            attr_reduction="L2",
        )
        cq_char_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        cq_char_name = f"canary-qwen-timit-val-L2_e_grad-pertoken{char_suf}"
        cq_char_extract.add_alias(cq_char_name)
        tk.register_output(f"{cq_char_name}.hdf", cq_char_extract.out_hdf)
        for align_opts in _ALIGN_OPTS_GRID:
            align_name = f"align/{cq_char_name}-{_name_for_dict(align_opts)}"
            align = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=cq_char_extract.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=align_opts,
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    cq_acc_recog.add_alias("canary-qwen-acc-timit-val-recog")
    tk.register_output("canary-qwen-acc-timit-val-recog.hyps.txt.gz", cq_acc_recog.out_hyps_txt)
    cq_acc_hyps_norm = text_dict_normalize_file(cq_acc_recog.out_hyps_txt)
    cq_acc_score = sclite_score_hyps_to_ref(
        hyps_text_dict=cq_acc_hyps_norm,
        ref_text_dict=cq_refs_norm,
        corpus_name="canary-qwen-acc-timit-val",
    )
    tk.register_output("canary-qwen-acc-timit-val-wer.txt", cq_acc_score.main_measure_value)
    tk.register_output("canary-qwen-acc-timit-val-wer-report", cq_acc_score.report)

    # Speech-LLM generic-setting check: does the CTC/AED energy+silence recipe
    # transfer to the LLM-decoders (Phi4 / Voxtral / Canary)?
    # They were originally run on the old grid only;
    # add the generic en0.5 / en0.5-sil1.0 / en0.5-zsk1.0 on each model's best extract.
    # (Hint: Canary char-level en0.5 was WORSE than no-energy, so do not assume -- measure.)
    # Voxtral: re-create the char-level log-mel L1 extract by config (same hash -> reuses
    # the finished one), only to get its out_hdf for the new aligns.
    vxt_best_ex = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=voxtral_charlev_logmel_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L1",
    )
    vxt_best_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for _sll_ex, _sll_name, _sll_en05 in [
        (pt_csp_gen, "phi4mm-timit-val-L2_e_grad-pertoken-charlev-spc", False),
        (cq_cls_extract, cq_cls_name, True),
        (vxt_best_ex, "voxtral-charlevlogmel-L1-timit-val-pertoken", True),
    ]:
        _sll_variants = []
        if _sll_en05:
            _sll_variants.append(("-en0.5", {"audio_energy_pow": 0.5}))
        _sll_variants += [
            ("-en0.5-sil1.0", {"audio_energy_pow": 0.5, "blank_silence_energy_scale": 1.0}),
            ("-en0.5-zsk1.0", {"audio_energy_pow": 0.5, "blank_grad_zscore_kappa": 1.0}),
        ]
        for _sll_sfx, _sll_kw in _sll_variants:
            _sll_ao = {"apply_softmax_over_time": True, "blank_score": -5}
            _sll_al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_sll_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_sll_ao,
                **_sll_kw,
            )
            _sll_nm = f"align/{_sll_name}-{_name_for_dict(_sll_ao)}{_sll_sfx}"
            _sll_al.add_alias(_sll_nm)
            tk.register_output(f"{_sll_nm}-wbe.txt", _sll_al.out_wbe)

    # --- TIMIT TEST split (headline models) -- generalization beyond the val-tuned config.
    # The two best grad-align surfaces with the generic en0.5-sil1.0 setting,
    # plus the Whisper-native cross-attn baseline on test (MMS_FA test already exists).
    _test_specs = [
        (
            rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir, char_level=True, char_level_sep=" "),
            "whisper-base-logmel-timit-test-L2_grad-pertoken-charlev-spc",
        ),
        (
            rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out"),
            "wav2vec2ctc-fproj_out-timit-test-L2_grad-pertoken",
        ),
    ]
    for _t_cfg, _t_name in _test_specs:
        _t_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="test",
            model_config=_t_cfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
        )
        _t_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _t_ex.add_alias(_t_name)
        tk.register_output(f"{_t_name}.hdf", _t_ex.out_hdf)
        _t_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _t_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_t_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="test",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_t_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
        )
        _t_nm = f"align/{_t_name}-{_name_for_dict(_t_ao)}-en0.5-sil1.0"
        _t_al.add_alias(_t_nm)
        tk.register_output(f"{_t_nm}-wbe.txt", _t_al.out_wbe)
    whisper_fa_test = WhisperCrossAttnForcedAlignJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_key="test", overlay=_WHISPER_TS_OVERLAY
    )
    whisper_fa_test.add_alias("baseline-whisper-crossattn-timit-test")
    whisper_fa_test_metric = CalcAlignmentMetricsFromWordBoundariesJob(
        word_boundaries_hdf=whisper_fa_test.out_hdf,
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="test",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
    )
    tk.register_output("baseline-whisper-crossattn-timit-test-wbe.txt", whisper_fa_test_metric.out_wbe)

    # --- MFA (Montreal Forced Aligner) baseline infra ---------------------------
    # Pull the MFA Docker image into a tracked .sif, and wrap `mfa` as a plain executable so the
    # (to-be-added) MfaForcedAlignJob can take a configurable mfa_exe (native binary OR this
    # container wrapper). Image/wrapper registered as outputs so they build now; the align job
    # is wired once the in-container MFA CLI is verified.
    _mfa_image = PullApptainerImageJob("docker://mmcauliffe/montreal-forced-aligner:latest")
    tk.register_output("mfa/image.sif", _mfa_image.out_image)
    _mfa_exe = ApptainerExeWrapperJob(_mfa_image.out_image, command="mfa")
    tk.register_output("mfa/mfa-run.sh", _mfa_exe.out_exe)

    # --- Buckeye long-form (fine-resolution, literature-comparable) ------------
    # Convert the alexwengg manifest+wav benchmark to our schema, then run the headline
    # grad-align (en0.5-sil1.0) + MMS_FA forced-align on it.
    # vs Huang et al. 2024: Buckeye WBE 43 ms (label-prior CTC) / 58 ms (plain CTC).
    # Two segmentation variants on the SAME <=20 s source segments:
    #  - "buckeye": as alexwengg ships them. ~half the segments group a backchannel at each end
    #    around 15+ s of the OTHER speaker's silence -> no aligner can place words in that silence
    #    -> mean WBE blows up (report acc@collar / dense subset for these).
    #  - "buckeye-reseg": split at internal silences > 1 s, drop <2-word fragments (Rousso 2024 /
    #    Less-Peaky >1 s non-speech protocol). ~463 segs / 7.1k words (98% retained), all <=18.5 s
    #    -> every word alignable, mean WBE meaningful, and ~4x faster per-token backward (shorter
    #    forward graphs). This is the clean primary; "buckeye" is kept as a robustness reference.
    _bk_off = _DATASET_OFFSET_FACTORS["timit"]  # start/stop in samples (factor 1), like TIMIT
    for _bk_ds_job, _bk_tag in [
        (BuildBuckeyeFineDatasetJob(raw_dir=dl_ds_buckeye_fine.out_hub_cache_dir, max_duration_s=20.0), "buckeye"),
        (
            BuildBuckeyeFineDatasetJob(
                raw_dir=dl_ds_buckeye_fine.out_hub_cache_dir,
                max_duration_s=20.0,
                resegment_gap_s=1.0,
                min_words=2,
            ),
            "buckeye-reseg",
        ),
    ]:
        tk.register_output(f"{_bk_tag}-fine-dataset", _bk_ds_job.out_hub_cache_dir)
        _bk_dir = _bk_ds_job.out_hub_cache_dir

        bk_fa = ForcedAlignBaselineJob(dataset_dir=_bk_dir, dataset_key="test")
        bk_fa.add_alias(f"baseline-mms_fa-{_bk_tag}")
        bk_fa_metric = CalcAlignmentMetricsFromWordBoundariesJob(
            word_boundaries_hdf=bk_fa.out_hdf,
            dataset_dir=_bk_dir,
            dataset_key="test",
            dataset_offset_factors=_bk_off,
        )
        tk.register_output(f"baseline-mms_fa-{_bk_tag}-wbe.txt", bk_fa_metric.out_wbe)

        # Headline grad-align surfaces (char-level Whisper / Wav2Vec2-CTC) on every variant; on the
        # clean reseg primary ALSO a speech-LLM (Voxtral char, best LLM on TIMIT) and a transducer
        # (Parakeet RNN-T, best transducer on TIMIT) to test whether the cross-model TIMIT ranking
        # generalizes to spontaneous speech -- in particular whether transducers collapse like CTC.
        _whc_cfg = rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir, char_level=True, char_level_sep=" ")
        _bk_models = [
            (_whc_cfg, f"whisper-base-logmel-{_bk_tag}-L2_grad-pertoken-charlev-spc", "L2", False),
            (
                rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out"),
                f"wav2vec2ctc-fproj_out-{_bk_tag}-L2_grad-pertoken",
                "L2",
                False,
            ),
        ]
        if _bk_tag == "buckeye-reseg":
            _bk_models += [
                (voxtral_charlev_logmel_cfg, f"voxtral-charlevlogmel-{_bk_tag}-L1_grad-pertoken", "L1", False),
                (pk_cfg, f"parakeet-rnnt-1.1b-logmel-{_bk_tag}-L2_grad-pertoken", "L2", False),
                # Remaining speech-LLMs on Buckeye (Voxtral already above): Phi4-MM (char L2_e_grad) and
                # Canary-Qwen (char L1, st1.5) -- complete the cross-model Buckeye column.
                (pt_csp_cfg, f"phi4mm-{_bk_tag}-L2_e_grad-pertoken-charlev-spc", "L2", True),
                (
                    canary_charlev_logmel_st15_cfg,
                    f"canary-qwen-charlev-spc-logmel-st15-{_bk_tag}-L1_grad-pertoken",
                    "L1",
                    False,
                ),
                # Re-tune the model/grad-score axes on Buckeye (vs the TIMIT-tuned char-L2_grad surface):
                # subword tokenization (char vs SPM), grad*input (L2_e_grad), and L1 reduction.
                (
                    rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir),
                    f"whisper-base-logmel-{_bk_tag}-L2_grad-pertoken",
                    "L2",
                    False,
                ),
                (_whc_cfg, f"whisper-base-logmel-{_bk_tag}-L2_e_grad-pertoken-charlev-spc", "L2", True),
                (_whc_cfg, f"whisper-base-logmel-{_bk_tag}-L1_grad-pertoken-charlev-spc", "L1", False),
            ]
        for _bk_cfg, _bk_name, _bk_attr, _bk_mgi in _bk_models:
            _bk_ex = ExtractInGradsPerTokenJob(
                dataset_dir=_bk_dir,
                dataset_key="test",
                model_config=_bk_cfg,
                mult_grad_by_inputs=_bk_mgi,
                attr_reduction=_bk_attr,
            )
            _bk_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            # Long-form segments -> more walltime than the default; keep the default GPU/partition.
            _bk_ex.rqmt = {**_bk_ex.rqmt, "time": 12}
            _bk_ex.add_alias(_bk_name)
            tk.register_output(f"{_bk_name}.hdf", _bk_ex.out_hdf)
            _bk_ao = {"apply_softmax_over_time": True, "blank_score": -5}
            _bk_al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_bk_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=_bk_dir,
                dataset_key="test",
                dataset_offset_factors=_bk_off,
                align_opts=_bk_ao,
                audio_energy_pow=0.5,
                blank_silence_energy_scale=1.0,
            )
            _bk_nm = f"align/{_bk_name}-{_name_for_dict(_bk_ao)}-en0.5-sil1.0"
            _bk_al.add_alias(_bk_nm)
            tk.register_output(f"{_bk_nm}-wbe.txt", _bk_al.out_wbe)


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
