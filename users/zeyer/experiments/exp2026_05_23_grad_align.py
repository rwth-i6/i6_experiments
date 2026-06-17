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
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.wav2vec2_phoneme_ctc import (
    Wav2Vec2PhonemeCtc,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.owls import Owls
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.owsm_ctc import OwsmCtc
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.emformer_rnnt import EmformerRnnt
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.fastconformer_streaming import (
    FastConformerStreaming,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.parakeet_rnnt import ParakeetRnnt
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.parakeet_ctc import ParakeetCtc
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
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.hyp_align import (
    BuildDatasetWithHypTranscriptsJob,
    CalcHypAlignMetricsJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.native_transducer_align import (
    NativeTransducerAlignJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.extract_self_attn import (
    SelectSelfAttnAlignHeadsJob,
    ExtractSelfAttnPerTokenJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.crisper_whisper_align import (
    CrisperWhisperOfficialAlignJob,
)
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.word_align_from_per_token_grads import (
    WordAlignFromPerTokenGradsJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.phoneme_word_wbe import (
    PhonemeAlignWordWbeJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.phoneme_forced_align_baseline import (
    ForcedAlignPhonemeBaselineJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.parakeet_ctc_forced_align import (
    ParakeetCtcForcedAlignJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.owsm_ctc_forced_align import (
    OwsmCtcForcedAlignJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_cost_benchmark import (
    AlignCostBenchmarkJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.batch_equiv_probe import (
    BatchForwardEquivalenceProbeJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.wer_noise_sweep import WerNoiseSweepJob
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.apptainer import (
    PullApptainerImageJob,
    ApptainerExeWrapperJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.mfa_forced_align import (
    MfaDownloadModelJob,
    MfaForcedAlignJob,
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
    attn_implementation: Optional[str] = None,
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
    if attn_implementation is not None:
        extra["attn_implementation"] = attn_implementation
    return rf.build_dict(
        Phi4MM,
        model_dir=model_dir,
        speech_prompt=speech_prompt,
        unwrap_checkpoint_wrappers=True,
        target_start_end_to_device=True,
        **extra,
    )


# Capture every registered output's Variable as it is created (during py()), keyed by output name,
# so the auto-generated result tables (build_tables) reference only outputs THIS recipe actually
# produces -- never stale on-disk aliases. (sis_graph isn't introspectable mid-py(): targets is
# empty, jobs() only partial -- so we collect at registration time. Module-level so the helper
# functions that also register outputs share it.)
_table_results: Dict[str, Any] = {}


def reg(name, value, **kwargs):
    tk.register_output(name, value, **kwargs)
    _table_results[name] = value
    return value


def py():
    """Sisyphus entry point."""
    dl_phi4mi_dir = download_phi4multimodal_model()

    dl_ds_timit = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/timit", repo_type="dataset")
    reg("timit-dataset", dl_ds_timit.out_hub_cache_dir)

    dl_ds_buckeye = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/buckeye", repo_type="dataset")
    # Fine-resolution Buckeye (float-ms GT, pre-segmented at silence into 2-25 s utterances, audio bundled);
    # supersedes the 62.5 ms-quantized nh0znoisung copy for long-form WBE eval.
    dl_ds_buckeye_fine = DownloadHuggingFaceRepoJobV2(repo_id="alexwengg/buckeye", repo_type="dataset")
    # The xet protocol's read-token endpoint gets HTTP-429 rate-limited on the shared
    # cluster IP; disable it so the download falls back to the standard LFS fetch.
    dl_ds_buckeye_fine.set_env("HF_HUB_DISABLE_XET", "1")
    reg("buckeye-fine-raw", dl_ds_buckeye_fine.out_hub_cache_dir)
    reg("buckeye-dataset", dl_ds_buckeye.out_hub_cache_dir)

    # POWSM: phonetic Whisper/OWSM-style foundation model (espnet/powsm). One repo holds TWO checkpoints --
    # the AED (exp/s2t_train_s2t_ebf...) and the encoder-only POWSM-CTC
    # (textnorm_retrained/exp/s2t_train_ctc3...). Loaded via ESPnet S2T, same path as the OWLS adapter.
    dl_powsm = DownloadHuggingFaceRepoJobV2(repo_id="espnet/powsm", repo_type="model")
    dl_powsm.set_env("HF_HUB_DISABLE_XET", "1")
    reg("powsm-model", dl_powsm.out_hub_cache_dir)
    # OWSM-CTC v4 1B: a GENERAL graphemic (BPE-50k) transcription CTC -- the representative general CTC
    # (vs MMS_FA = an aligner, vitouphy / POWSM = phonetic). Loaded via ESPnet S2TCTCTask (CTC head).
    dl_owsm_ctc = DownloadHuggingFaceRepoJobV2(repo_id="espnet/owsm_ctc_v4_1B", repo_type="model")
    dl_owsm_ctc.set_env("HF_HUB_DISABLE_XET", "1")
    reg("owsm-ctc-v4-1b-model", dl_owsm_ctc.out_hub_cache_dir)
    owsm_ctc_cfg = rf.build_dict(OwsmCtc, model_dir=dl_owsm_ctc.out_hub_cache_dir, version=2)
    # Per-layer OWSM-CTC grad-align (side table): emit from each inter-CTC self-cond block (6/12/15/21).
    # The final block (27) is the default `owsm_ctc_cfg` above (layer=None). See PLAN: OWSM per-layer table.
    owsm_ctc_cfg_by_layer = {
        _ol: rf.build_dict(OwsmCtc, model_dir=dl_owsm_ctc.out_hub_cache_dir, version=2, layer=_ol)
        for _ol in [6, 12, 15, 21]
    }

    dl_whisper = DownloadHuggingFaceRepoJobV2(repo_id="openai/whisper-base", repo_type="model")
    dl_whisper_l3 = DownloadHuggingFaceRepoJobV2(repo_id="openai/whisper-large-v3", repo_type="model")
    dl_crisper = DownloadHuggingFaceRepoJobV2(repo_id="nyrahealth/CrisperWhisper", repo_type="model")
    dl_parakeet_rnnt = DownloadHuggingFaceRepoJobV2(repo_id="nvidia/parakeet-rnnt-1.1b", repo_type="model")
    dl_parakeet_tdt = DownloadHuggingFaceRepoJobV2(repo_id="nvidia/parakeet-tdt-0.6b-v2", repo_type="model")
    dl_parakeet_ctc = DownloadHuggingFaceRepoJobV2(repo_id="nvidia/parakeet-ctc-1.1b", repo_type="model")
    # NeMo cache-aware streaming FastConformer (hybrid CTC + RNN-T on ONE streaming encoder): the
    # streaming representative -- both a streaming CTC and a streaming transducer from one checkpoint.
    # Limited-context attention (att_context_size=[left,right]) + cache-aware; here a single masked
    # forward at the chosen look-ahead. Same FastConformer family as offline Parakeet -> controlled
    # offline-vs-streaming comparison. Ref: Noroozi et al. arXiv:2312.17279.
    dl_fc_stream = DownloadHuggingFaceRepoJobV2(
        repo_id="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi", repo_type="model"
    )
    reg("fastconformer-streaming-model", dl_fc_stream.out_hub_cache_dir)
    reg("whisper-base-model", dl_whisper.out_hub_cache_dir)

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
        reg(f"baseline-mms_fa-timit-{_split}-wbe.txt", fa_metric.out_wbe)

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
            reg(f"baseline-mms_fa-timit-val-ts{_fa_ts}{_fa_msfx}-wbe.txt", fa_ts_metric.out_wbe)

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
    reg("baseline-whisper-crossattn-timit-val-wbe.txt", whisper_fa_metric.out_wbe)

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
        reg(f"{w2v_name}.hdf", w2v_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{w2vs_name}.hdf", w2vs_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{w2vp_name}.hdf", w2vp_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
    reg(f"{w2v_raw_name}.hdf", w2v_raw_extract.out_hdf)
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
        reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{gw_name}.hdf", gw_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
            reg(f"align/{base_name}-{_sfx}-wbe.txt", _al.out_wbe)

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
        reg(f"{_mname}.hdf", _mex.out_hdf)
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
            reg(f"align/{_dc_name}-{_sfx}-wbe.txt", _al.out_wbe)
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
            reg(f"align/{_dc_name}-{_sfx}-wbe.txt", _al.out_wbe)

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
            reg(f"{_ts_name}.hdf", _ts_ex.out_hdf)
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
                reg(f"align/{_ts_name}-{_ts_sfx}-wbe.txt", _ts_al.out_wbe)

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
    reg(f"{rnnt_name}.hdf", rnnt_extract.out_hdf)
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
        reg(f"align/{rnnt_name}-{_re_sfx}-wbe.txt", _re_al.out_wbe)
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
        reg(f"align/{rnnt_name}-{_re_sfx}-wbe.txt", _re_al.out_wbe)

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
    reg(f"{rnnt_px_name}.hdf", rnnt_px_extract.out_hdf)
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
        reg(f"align/{rnnt_px_name}-{_rpx_sfx}-wbe.txt", _rpx_al.out_wbe)
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
        reg(f"align/{rnnt_px_name}-{_rpx_sfx}-wbe.txt", _rpx_al.out_wbe)

    # Native RNN-T Viterbi forced-align for the streaming Emformer: the same-model forced-vs-grad
    # counterpart, on TIMIT-val (where the Emformer grad rows live). Uses the collect_lattice hook.
    _em_nt = NativeTransducerAlignJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_key="val", model_config=rnnt_cfg
    )
    _em_nt.add_alias("baseline-emformer-rnnt-native-viterbi-timit-val")
    _em_nt_m = CalcAlignmentMetricsFromWordBoundariesJob(
        word_boundaries_hdf=_em_nt.out_word_boundaries_hdf,
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
    )
    reg("baseline-emformer-rnnt-native-viterbi-timit-val-wbe.txt", _em_nt_m.out_wbe)

    # Parakeet RNN-T 1.1B (NeMo FastConformer-Transducer): a STRONG leaderboard-grade
    # transducer, vs the tiny Emformer base above (139 ms).
    # Same proper RNN-T prefix-score; log-mel grad target (~100 Hz). + energy/silence.
    pk_cfg = rf.build_dict(
        ParakeetRnnt, model_dir=dl_parakeet_rnnt.out_hub_cache_dir, per_token_score="prefix", overlay_path=_NEMO_OVERLAY
    )
    # nvidia/parakeet-ctc-1.1b: a plain FastConformer CTC -- the general graphemic CTC representative.
    parakeet_ctc_cfg = rf.build_dict(
        ParakeetCtc, model_dir=dl_parakeet_ctc.out_hub_cache_dir, overlay_path=_NEMO_OVERLAY
    )
    # prefix_diff per-token score = the AED-consistent CTC prefix-score difference (vs the raw acc[2i+1]
    # the parakeet/owsm wrappers used). Fixes grad-align for these conformer/branchformer CTCs
    # (parakeet TIMIT 237 -> ~149 ms on a 100-seq probe). See jobs.models.ctc_partial.
    parakeet_ctc_prefdiff_cfg = rf.build_dict(
        ParakeetCtc,
        model_dir=dl_parakeet_ctc.out_hub_cache_dir,
        overlay_path=_NEMO_OVERLAY,
        per_token_score="prefix_diff",
    )
    owsm_ctc_prefdiff_cfg = rf.build_dict(
        OwsmCtc, model_dir=dl_owsm_ctc.out_hub_cache_dir, version=2, per_token_score="prefix_diff"
    )
    # inb_true = the blank-anchored (robust, universal) prefix-score diff. On the subword CTCs we expect
    # prefix_diff to win, but quantify the gap full-corpus to choose uniform-inb_true vs per-model best.
    parakeet_ctc_inbtrue_cfg = rf.build_dict(
        ParakeetCtc,
        model_dir=dl_parakeet_ctc.out_hub_cache_dir,
        overlay_path=_NEMO_OVERLAY,
        per_token_score="inb_true",
    )
    owsm_ctc_inbtrue_cfg = rf.build_dict(
        OwsmCtc, model_dir=dl_owsm_ctc.out_hub_cache_dir, version=2, per_token_score="inb_true"
    )
    # prefix_fwd = the REAL CTC prefix score (Graves/ESPnet, as in DLM-sum label-sync) = exact
    # log p(y_i|y_<i,x). Blank-aware, robust to repeats. Best on the parakeet probe (98ms). The
    # candidate for a single uniform per-token score across all CTCs.
    parakeet_ctc_prefixfwd_cfg = rf.build_dict(
        ParakeetCtc,
        model_dir=dl_parakeet_ctc.out_hub_cache_dir,
        overlay_path=_NEMO_OVERLAY,
        per_token_score="prefix_fwd",
    )
    owsm_ctc_prefixfwd_cfg = rf.build_dict(
        OwsmCtc, model_dir=dl_owsm_ctc.out_hub_cache_dir, version=2, per_token_score="prefix_fwd"
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
    reg(f"{pk_name}.hdf", pk_extract.out_hdf)
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
        reg(f"align/{pk_name}-{_pk_sfx}-wbe.txt", _pk_al.out_wbe)

    # Parakeet TDT 0.6B v2 (NeMo Token-and-Duration Transducer): the top transducer on the Open ASR
    # Leaderboard. tdt_cfg (emission) is the hash-stable cfg for the score-agnostic uses (native-viterbi
    # forced-align, recog); the grad extracts + cost use tdt_grad_cfg with the proper duration-aware
    # TDT-forward prefix score (per_token_score='prefix' auto-dispatches to the TDT lattice; the TDT
    # analog of the CTC prefix_fwd / RNN-T prefix, verified == reference loop == brute-force).
    tdt_cfg = rf.build_dict(
        ParakeetRnnt,
        model_dir=dl_parakeet_tdt.out_hub_cache_dir,
        per_token_score="emission",
        overlay_path=_NEMO_OVERLAY,
    )
    tdt_grad_cfg = rf.build_dict(
        ParakeetRnnt,
        model_dir=dl_parakeet_tdt.out_hub_cache_dir,
        per_token_score="prefix",
        overlay_path=_NEMO_OVERLAY,
    )
    tdt_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=tdt_grad_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
    )
    tdt_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    tdt_name = "parakeet-tdt-0.6b-v2-logmel-timit-val-L2_grad-pertoken"
    tdt_extract.add_alias(tdt_name)
    reg(f"{tdt_name}.hdf", tdt_extract.out_hdf)
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
        reg(f"align/{tdt_name}-{_tdt_sfx}-wbe.txt", _tdt_al.out_wbe)

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
    reg(f"{whisper_name}.hdf", whisper_extract.out_hdf)
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
        reg(f"align/{whisper_name}-{_wh_sfx}-wbe.txt", _wh_al.out_wbe)
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
        reg(f"align/{whisper_name}-{_wh_sfx}-wbe.txt", _wh_al.out_wbe)

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
    reg(f"{whisper_char_name}.hdf", whisper_char_extract.out_hdf)
    _w2v_aligns(whisper_char_extract, whisper_char_name, energy_pows=(0.0, 0.5))

    # --- Whisper SCALE sweep (does a bigger/stronger AED align better?) -------------
    # The headline AED is whisper-BASE (74M), which already BEATS the giant speech-LLMs
    # (47 vs 82-86 ms) -> alignment quality may NOT track ASR quality/size. Sweep
    # base->small->medium->large-v3 with the SAME char-level grad-align at the headline
    # config (en0.5-sil1.0). base = the existing whisper-char headline (reused).
    for _ws in ["small", "medium", "large-v3"]:
        _ws_dl = DownloadHuggingFaceRepoJobV2(repo_id=f"openai/whisper-{_ws}", repo_type="model")
        reg(f"whisper-{_ws}-model", _ws_dl.out_hub_cache_dir)
        _ws_cfg = rf.build_dict(Whisper, model_dir=_ws_dl.out_hub_cache_dir, char_level=True, char_level_sep=" ")
        _ws_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=_ws_cfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
        )
        _ws_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _ws_ex.rqmt = {**_ws_ex.rqmt, "time": 24}  # large-v3 ~1.5B params -> slower than base
        _ws_name = f"whisper-{_ws}-logmel-timit-val-L2_grad-pertoken-charlev-spc"
        _ws_ex.add_alias(_ws_name)
        reg(f"{_ws_name}.hdf", _ws_ex.out_hdf)
        _ws_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _ws_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_ws_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_ws_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
        )
        _ws_nm = f"align/{_ws_name}-{_name_for_dict(_ws_ao)}-en0.5-sil1.0"
        _ws_al.add_alias(_ws_nm)
        reg(f"{_ws_nm}-wbe.txt", _ws_al.out_wbe)

    # --- Transducer scale points (matched-variant 0.6b<->1.1b pairs; same adapter) -----
    # Fixes the headline mismatch (RNN-T 1.1b vs TDT 0.6b-v2 were picked as best-per-type,
    # not a size pair): add RNN-T 0.6b and TDT 1.1b so each variant has a 0.6b<->1.1b pair.
    # en0.5-sil1.0 (headline) + en0.5 no-sil (transducers double-count silence).
    for _pk_repo, _pk_pts, _pk_sname in [
        ("nvidia/parakeet-rnnt-0.6b", "prefix", "parakeet-rnnt-0.6b-logmel-timit-val-L2_grad-pertoken"),
        ("nvidia/parakeet-tdt-1.1b", "emission", "parakeet-tdt-1.1b-logmel-timit-val-L2_grad-pertoken"),
    ]:
        _pk_sdl = DownloadHuggingFaceRepoJobV2(repo_id=_pk_repo, repo_type="model")
        reg(f"{_pk_repo.split('/')[-1]}-model", _pk_sdl.out_hub_cache_dir)
        _pk_scfg = rf.build_dict(
            ParakeetRnnt, model_dir=_pk_sdl.out_hub_cache_dir, per_token_score=_pk_pts, overlay_path=_NEMO_OVERLAY
        )
        _pk_sex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=_pk_scfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
        )
        _pk_sex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _pk_sex.rqmt = {**_pk_sex.rqmt, "time": 24}
        _pk_sex.add_alias(_pk_sname)
        reg(f"{_pk_sname}.hdf", _pk_sex.out_hdf)
        for _pk_ep, _pk_sc in [(0.5, 1.0), (0.5, 0.0)]:
            _pk_sao = {"apply_softmax_over_time": True, "blank_score": -5}
            _pk_sal = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_pk_sex.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_pk_sao,
                audio_energy_pow=_pk_ep,
                blank_silence_energy_scale=_pk_sc,
            )
            _pk_snm = f"align/{_pk_sname}-{_name_for_dict(_pk_sao)}-en{_pk_ep}-sil{_pk_sc}"
            _pk_sal.add_alias(_pk_snm)
            reg(f"{_pk_snm}-wbe.txt", _pk_sal.out_wbe)

    # --- OWLS controlled scaling suite (ESPnet Whisper-style AED) -------------------
    # The methodologically clean scale axis: ONE recipe/data, scaled. Two axes here
    # (model size @180K hrs; data scale @1B params); training-intermediates axis is
    # separate (DeepSpeed checkpoints, needs consolidation). Subword grad-align,
    # log-mel @100Hz leaf, en0.5-sil1.0 (AED headline config). TIMIT val.
    _owls_models = [
        # model-size ladder (all 180K hrs)
        ("espnet/owls_025B_180K", "owls-025B-180K"),
        ("espnet/owls_05B_180K", "owls-05B-180K"),
        ("espnet/owls_1B_180K", "owls-1B-180K"),
        ("espnet/owls_2B_180K", "owls-2B-180K"),
        # data-scale axis (1B params, varied hours; 180K point = owls-1B-180K above)
        ("espnet/owls_1b_11k", "owls-1b-11k"),
        ("espnet/owls_1b_22K", "owls-1b-22K"),
        ("espnet/owls_1b_45k", "owls-1b-45k"),
        ("espnet/owls_1b_90k", "owls-1b-90k"),
    ]
    for _ow_repo, _ow_tag in _owls_models:
        _ow_dl = DownloadHuggingFaceRepoJobV2(repo_id=_ow_repo, repo_type="model")
        reg(f"{_ow_tag}-model", _ow_dl.out_hub_cache_dir)
        _ow_cfg = rf.build_dict(Owls, model_dir=_ow_dl.out_hub_cache_dir, char_level=True)
        _ow_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=_ow_cfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
        )
        _ow_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _ow_ex.rqmt = {**_ow_ex.rqmt, "time": 24}
        _ow_name = f"{_ow_tag}-charlev-logmel-timit-val-L2_grad-pertoken"
        _ow_ex.add_alias(_ow_name)
        reg(f"{_ow_name}.hdf", _ow_ex.out_hdf)
        _ow_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _ow_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_ow_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_ow_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
        )
        _ow_nm = f"align/{_ow_name}-{_name_for_dict(_ow_ao)}-en0.5-sil1.0"
        _ow_al.add_alias(_ow_nm)
        reg(f"{_ow_nm}-wbe.txt", _ow_al.out_wbe)

    # OWLS training-INTERMEDIATES axis (does a checkpoint align better AS it trains?).
    # The *_intermediates repos are DeepSpeed ckpts -> download only <step>/mp_rank_00_model_states.pt for a
    # log-spaced subset of the 45 saved steps; build from the final 1B config + load the intermediate
    # ['module'] (keys verified identical). 1B SUBWORD (OWLS char underperformed). TIMIT val, en0.5-sil1.0.
    dl_owls_1b_full = DownloadHuggingFaceRepoJobV2(repo_id="espnet/owls_1B_180K", repo_type="model")
    for _ow_step in [1, 2, 4, 8, 16, 32, 45]:
        _ow_sdl = DownloadHuggingFaceRepoJobV2(
            repo_id="espnet/owls_1B_180K_intermediates",
            repo_type="model",
            include=[f"{_ow_step}/mp_rank_00_model_states.pt"],
        )
        # both granularities: subword (opt-in absent -> existing jobs reused) + char (best variant, user ask)
        for _ow_cl, _ow_clsfx in [(False, ""), (True, "-charlev")]:
            _ow_scfg = rf.build_dict(
                Owls,
                model_dir=dl_owls_1b_full.out_hub_cache_dir,
                state_override_dir=_ow_sdl.out_hub_cache_dir,
                **({"char_level": True} if _ow_cl else {}),
            )
            _ow_sex = ExtractInGradsPerTokenJob(
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                model_config=_ow_scfg,
                mult_grad_by_inputs=False,
                attr_reduction="L2",
            )
            _ow_sex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            _ow_sex.rqmt = {**_ow_sex.rqmt, "time": 24}
            _ow_sname = f"owls-1B-180K-step{_ow_step:02d}{_ow_clsfx}-logmel-timit-val-L2_grad-pertoken"
            _ow_sex.add_alias(_ow_sname)
            reg(f"{_ow_sname}.hdf", _ow_sex.out_hdf)
            _ow_sao = {"apply_softmax_over_time": True, "blank_score": -5}
            _ow_sal = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_ow_sex.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_ow_sao,
                audio_energy_pow=0.5,
                blank_silence_energy_scale=1.0,
            )
            _ow_snm = f"align/{_ow_sname}-{_name_for_dict(_ow_sao)}-en0.5-sil1.0"
            _ow_sal.add_alias(_ow_snm)
            reg(f"{_ow_snm}-wbe.txt", _ow_sal.out_wbe)

    # --- Phoneme-CTC grad-align (axis: native phoneme model, WhisperX-style) ---
    # vitouphy/wav2vec2-xls-r-300m-timit-phoneme: wav2vec2 + 39-IPA CTC head, TIMIT-trained.
    # Targets = TIMIT's own phone labels (phonetic_detail), folded 61->39 IPA inside the adapter;
    # each phone segment = one CTC unit. The align job measures PHONE-boundary error vs the dataset's
    # phone boundaries -- a strictly finer, on-distribution alignment than word/char. CTC => may front-load.
    dl_w2v_phoneme = DownloadHuggingFaceRepoJobV2(
        repo_id="vitouphy/wav2vec2-xls-r-300m-timit-phoneme", repo_type="model"
    )
    reg("w2v-phoneme-model", dl_w2v_phoneme.out_hub_cache_dir)
    w2v_phoneme_cfg = rf.build_dict(Wav2Vec2PhonemeCtc, model_dir=dl_w2v_phoneme.out_hub_cache_dir)
    w2v_ph_extract = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=w2v_phoneme_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
        target_source="phonetic_detail",
    )
    w2v_ph_extract.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    w2v_ph_name = "w2v-phoneme-timit-val-L2_grad-perphone"
    w2v_ph_extract.add_alias(w2v_ph_name)
    reg(f"{w2v_ph_name}.hdf", w2v_ph_extract.out_hdf)
    for _ph_ep, _ph_sc in [(0.0, 0.0), (0.5, 1.0)]:
        _ph_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _ph_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=w2v_ph_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_ph_ao,
            audio_energy_pow=_ph_ep,
            blank_silence_energy_scale=_ph_sc,
            boundary_source="phonetic_detail",
        )
        _ph_nm = f"align/{w2v_ph_name}-en{_ph_ep}-sil{_ph_sc}"
        _ph_al.add_alias(_ph_nm)
        reg(f"{_ph_nm}-wbe.txt", _ph_al.out_wbe)
        # Collapse the per-phone grad-align to per-WORD boundaries -> word-WBE
        # (comparable to the cross-model word table). Reuses the same grad HDF + DP.
        _ph_ww = PhonemeAlignWordWbeJob(
            grad_score_hdf=w2v_ph_extract.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_ph_ao,
            audio_energy_pow=_ph_ep,
            blank_silence_energy_scale=_ph_sc,
        )
        _ph_ww.add_alias(f"{_ph_nm}-wordcollapse")
        reg(f"{_ph_nm}-word-wbe.txt", _ph_ww.out_word_wbe)

    # Standard CTC forced-alignment baseline on the SAME phoneme model (torchaudio
    # Viterbi): the "WhisperX on its own phoneme model" point. Brackets our grad-align
    # phone-WBE from above; reports both phone-WBE and word-WBE.
    _ph_fa = ForcedAlignPhonemeBaselineJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_dir=dl_w2v_phoneme.out_hub_cache_dir,
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
    )
    _ph_fa.add_alias("align/w2v-phoneme-timit-val-ctc-forced-align")
    reg("align/w2v-phoneme-timit-val-ctc-forced-align-phone-wbe.txt", _ph_fa.out_phone_wbe)
    reg("align/w2v-phoneme-timit-val-ctc-forced-align-word-wbe.txt", _ph_fa.out_word_wbe)

    # Same phoneme-CTC model on TIMIT TEST (generalization column for the per-model table;
    # only the headline en0.5-sil1.0 grad word-collapse + the forced-align baseline, both word-WBE).
    w2v_ph_extract_te = ExtractInGradsPerTokenJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="test",
        model_config=w2v_phoneme_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
        target_source="phonetic_detail",
    )
    w2v_ph_extract_te.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    w2v_ph_name_te = "w2v-phoneme-timit-test-L2_grad-perphone"
    w2v_ph_extract_te.add_alias(w2v_ph_name_te)
    reg(f"{w2v_ph_name_te}.hdf", w2v_ph_extract_te.out_hdf)
    _ph_ww_te = PhonemeAlignWordWbeJob(
        grad_score_hdf=w2v_ph_extract_te.out_hdf,
        grad_score_key="data",
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="test",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
        align_opts={"apply_softmax_over_time": True, "blank_score": -5},
        audio_energy_pow=0.5,
        blank_silence_energy_scale=1.0,
    )
    _ph_nm_te = f"align/{w2v_ph_name_te}-en0.5-sil1.0"
    _ph_ww_te.add_alias(f"{_ph_nm_te}-wordcollapse")
    reg(f"{_ph_nm_te}-word-wbe.txt", _ph_ww_te.out_word_wbe)
    _ph_fa_te = ForcedAlignPhonemeBaselineJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="test",
        model_dir=dl_w2v_phoneme.out_hub_cache_dir,
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
    )
    _ph_fa_te.add_alias("align/w2v-phoneme-timit-test-ctc-forced-align")
    reg("align/w2v-phoneme-timit-test-ctc-forced-align-phone-wbe.txt", _ph_fa_te.out_phone_wbe)
    reg("align/w2v-phoneme-timit-test-ctc-forced-align-word-wbe.txt", _ph_fa_te.out_word_wbe)

    # --- CONTROLLED PARAM-NOISE robustness (user idea): degrade the model with Gaussian param noise,
    # compare how grad-align vs the model's NATIVE align (Whisper cross-attn-DTW / CTC forced-align) decay.
    # sigma=0 = the existing baselines (whisper-char 47 / crossattn 86.8 / phone-grad 39.4 / phone-forced 31.1).
    # Each method's headline align config. Opt-in param-pass so the sigma=0 jobs keep their hash.
    _pn_seed = 42
    for _pn_std in [0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.2]:
        # (1a) Whisper char grad-align
        _pn_wg_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=rf.build_dict(
                Whisper,
                model_dir=dl_whisper.out_hub_cache_dir,
                char_level=True,
                char_level_sep=" ",
                param_noise_std=_pn_std,
                param_noise_seed=_pn_seed,
            ),
            mult_grad_by_inputs=False,
            attr_reduction="L2",
        )
        _pn_wg_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _pn_wg_ex.add_alias(f"paramnoise/whisper-char-grad-std{_pn_std}")
        _pn_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _pn_wg_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_pn_wg_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_pn_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
        )
        reg(f"paramnoise/whisper-char-grad-std{_pn_std}-wbe.txt", _pn_wg_al.out_wbe)

        # (1b) Whisper cross-attention DTW (the whisper-timestamped-style native method)
        _pn_wca = WhisperCrossAttnForcedAlignJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            overlay=_WHISPER_TS_OVERLAY,
            param_noise_std=_pn_std,
            param_noise_seed=_pn_seed,
        )
        _pn_wca_m = CalcAlignmentMetricsFromWordBoundariesJob(
            word_boundaries_hdf=_pn_wca.out_hdf,
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
        )
        reg(f"paramnoise/whisper-crossattn-std{_pn_std}-wbe.txt", _pn_wca_m.out_wbe)

        # (2a) phoneme-CTC grad-align (en0/sil0 = the phone headline)
        _pn_pg_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=rf.build_dict(
                Wav2Vec2PhonemeCtc,
                model_dir=dl_w2v_phoneme.out_hub_cache_dir,
                param_noise_std=_pn_std,
                param_noise_seed=_pn_seed,
            ),
            mult_grad_by_inputs=False,
            attr_reduction="L2",
            target_source="phonetic_detail",
        )
        _pn_pg_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _pn_pg_ex.add_alias(f"paramnoise/phoneme-grad-std{_pn_std}")
        _pn_pg_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_pn_pg_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_pn_ao,
            audio_energy_pow=0.0,
            blank_silence_energy_scale=0.0,
            boundary_source="phonetic_detail",
        )
        reg(f"paramnoise/phoneme-grad-std{_pn_std}-wbe.txt", _pn_pg_al.out_wbe)

        # (2b) phoneme-CTC forced-align (torchaudio Viterbi) -- SAME perturbed model as (2a)
        _pn_pf = ForcedAlignPhonemeBaselineJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_dir=dl_w2v_phoneme.out_hub_cache_dir,
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            param_noise_std=_pn_std,
            param_noise_seed=_pn_seed,
        )
        reg(f"paramnoise/phoneme-forced-std{_pn_std}-phone-wbe.txt", _pn_pf.out_phone_wbe)

    # WER/PER vs the SAME param-noise, to overlay on the robustness plot (does recognition collapse where
    # alignment does?). Whisper: WER on a 200-utt subset (autoregressive decode is slow). CTC: PER, full val.
    _wer_sigmas = [0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.2]
    _wer_wh = WerNoiseSweepJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_kind="whisper",
        model_dir=dl_whisper.out_hub_cache_dir,
        sigmas=_wer_sigmas,
        max_seqs=200,
    )
    _wer_wh.add_alias("paramnoise/wer-whisper")
    reg("paramnoise/wer-whisper.txt", _wer_wh.out_wer)
    _wer_ph = WerNoiseSweepJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_kind="wav2vec2ctc",
        model_dir=dl_w2v_phoneme.out_hub_cache_dir,
        sigmas=_wer_sigmas,
    )
    _wer_ph.add_alias("paramnoise/per-phoneme")
    reg("paramnoise/per-phoneme.txt", _wer_ph.out_wer)

    # --- CONTROLLED NON-PARAM degradation (user idea): input noise / activation dropout, expected to
    # give a SMOOTH ramp vs the param-noise cliff. Same 4 methods + recognition; opt-in perturb kwargs so
    # the level=0 jobs keep the existing baseline hashes.
    _nz_ao = {"apply_softmax_over_time": True, "blank_score": -5}
    for _nz_key, _nz_levels, _nz_rkind in [
        ("input_noise_std", [0.05, 0.1, 0.2, 0.4, 0.8, 1.6], "input"),
        ("act_dropout", [0.02, 0.05, 0.1, 0.2, 0.3, 0.5], "act_dropout"),
    ]:
        for _nz in _nz_levels:
            _pk = {_nz_key: _nz, "perturb_seed": _pn_seed}
            _ntag = f"{_nz_key}{_nz}"
            # (1a) Whisper char grad-align
            _nz_wg_ex = ExtractInGradsPerTokenJob(
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                model_config=rf.build_dict(
                    Whisper,
                    model_dir=dl_whisper.out_hub_cache_dir,
                    char_level=True,
                    char_level_sep=" ",
                    **_pk,
                ),
                mult_grad_by_inputs=False,
                attr_reduction="L2",
            )
            _nz_wg_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            _nz_wg_ex.add_alias(f"noise/whisper-char-grad-{_ntag}")
            _nz_wg_al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_nz_wg_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_nz_ao,
                audio_energy_pow=0.5,
                blank_silence_energy_scale=1.0,
            )
            reg(f"noise/whisper-char-grad-{_ntag}-wbe.txt", _nz_wg_al.out_wbe)

            # (1b) Whisper cross-attention DTW
            _nz_wca = WhisperCrossAttnForcedAlignJob(
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                overlay=_WHISPER_TS_OVERLAY,
                **_pk,
            )
            _nz_wca_m = CalcAlignmentMetricsFromWordBoundariesJob(
                word_boundaries_hdf=_nz_wca.out_hdf,
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            )
            reg(f"noise/whisper-crossattn-{_ntag}-wbe.txt", _nz_wca_m.out_wbe)

            # (2a) phoneme-CTC grad-align
            _nz_pg_ex = ExtractInGradsPerTokenJob(
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                model_config=rf.build_dict(
                    Wav2Vec2PhonemeCtc,
                    model_dir=dl_w2v_phoneme.out_hub_cache_dir,
                    **_pk,
                ),
                mult_grad_by_inputs=False,
                attr_reduction="L2",
                target_source="phonetic_detail",
            )
            _nz_pg_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            _nz_pg_ex.add_alias(f"noise/phoneme-grad-{_ntag}")
            _nz_pg_al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_nz_pg_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_nz_ao,
                audio_energy_pow=0.0,
                blank_silence_energy_scale=0.0,
                boundary_source="phonetic_detail",
            )
            reg(f"noise/phoneme-grad-{_ntag}-wbe.txt", _nz_pg_al.out_wbe)

            # (2b) phoneme-CTC forced-align -- SAME perturbed model as (2a)
            _nz_pf = ForcedAlignPhonemeBaselineJob(
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                model_dir=dl_w2v_phoneme.out_hub_cache_dir,
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                **_pk,
            )
            reg(f"noise/phoneme-forced-{_ntag}-phone-wbe.txt", _nz_pf.out_phone_wbe)

        # recognition (WER/PER) under the SAME degradation axis (one job, all levels)
        _nz_wer_wh = WerNoiseSweepJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_kind="whisper",
            model_dir=dl_whisper.out_hub_cache_dir,
            sigmas=_nz_levels,
            max_seqs=200,
            perturb_kind=_nz_rkind,
        )
        _nz_wer_wh.add_alias(f"noise/wer-whisper-{_nz_key}")
        reg(f"noise/wer-whisper-{_nz_key}.txt", _nz_wer_wh.out_wer)
        _nz_wer_ph = WerNoiseSweepJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_kind="wav2vec2ctc",
            model_dir=dl_w2v_phoneme.out_hub_cache_dir,
            sigmas=_nz_levels,
            perturb_kind=_nz_rkind,
        )
        _nz_wer_ph.add_alias(f"noise/per-phoneme-{_nz_key}")
        reg(f"noise/per-phoneme-{_nz_key}.txt", _nz_wer_ph.out_wer)

    # SmoothGrad ablation (attribution axis A) on the headline char-level Whisper surface: average the
    # per-token gradient over noise_n_samples noisy forward+backward passes (Gaussian noise on the raw
    # waveform). Tests whether saliency denoising sharpens boundaries vs the single-pass 47 ms baseline.
    for _sg_std, _sg_n in [(0.005, 8), (0.01, 8), (0.03, 8)]:
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
        reg(f"{_sg_name}.hdf", _sg_ex.out_hdf)
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
        reg(f"{_sg_nm}-wbe.txt", _sg_al.out_wbe)
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
        reg(f"align/{whisper_char_name}-{_whc_sfx}-wbe.txt", _whc_al.out_wbe)
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
        reg(f"align/{whisper_char_name}-{_whc_sfx}-wbe.txt", _whc_al.out_wbe)

    # VarGrad ablation (attribution axis A): per-frame STD of the saliency across SmoothGrad noise
    # samples (vs the SmoothGrad mean). Headline char Whisper.
    for _vg_std in [0.01, 0.02, 0.04]:  # std SWEEP (best-of for the ablation)
        _vg_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=whisper_char_cfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
            noise_std=_vg_std,
            noise_n_samples=16,
            vargrad=True,
        )
        _vg_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _vg_name = f"whisper-base-logmel-timit-val-L2-vargrad-pertoken-charlev-spc-std{_vg_std}-n16"
        _vg_ex.add_alias(_vg_name)
        reg(f"{_vg_name}.hdf", _vg_ex.out_hdf)
        _vg_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _vg_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_vg_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_vg_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
        )
        _vg_nm = f"align/{_vg_name}-{_name_for_dict(_vg_ao)}-en0.5-sil1.0"
        _vg_al.add_alias(_vg_nm)
        reg(f"{_vg_nm}-wbe.txt", _vg_al.out_wbe)

    # Integrated-Gradients ablation (attribution axis A) on the headline char Whisper: integrate the
    # per-token gradient along the audio-amplitude path (ig_steps Riemann steps), vs single-pass 47 ms.
    for _ig_n in [8, 16, 32]:  # IG step-count SWEEP (convergence to grad)
        _ig_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=whisper_char_cfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
            ig_steps=_ig_n,
        )
        _ig_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _ig_name = f"whisper-base-logmel-timit-val-L2-integrated-grad-pertoken-charlev-spc-ig{_ig_n}"
        _ig_ex.add_alias(_ig_name)
        reg(f"{_ig_name}.hdf", _ig_ex.out_hdf)
        _ig_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _ig_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_ig_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_ig_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
        )
        _ig_nm = f"align/{_ig_name}-{_name_for_dict(_ig_ao)}-en0.5-sil1.0"
        _ig_al.add_alias(_ig_nm)
        reg(f"{_ig_nm}-wbe.txt", _ig_al.out_wbe)

    # Expected-Gradients ablation (axis A): IG with RANDOM noise baselines x' (baseline-std SWEEP) +
    # random alpha per step. Headline char Whisper.
    for _eg_std in [0.01, 0.05, 0.1]:
        _eg_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=whisper_char_cfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
            eg_steps=16,
            eg_baseline_std=_eg_std,
        )
        _eg_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _eg_name = f"whisper-base-logmel-timit-val-L2-expected-grad-pertoken-charlev-spc-eg16-bstd{_eg_std}"
        _eg_ex.add_alias(_eg_name)
        reg(f"{_eg_name}.hdf", _eg_ex.out_hdf)
        _eg_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        _eg_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_eg_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_eg_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
        )
        _eg_nm = f"align/{_eg_name}-{_name_for_dict(_eg_ao)}-en0.5-sil1.0"
        _eg_al.add_alias(_eg_nm)
        reg(f"{_eg_nm}-wbe.txt", _eg_al.out_wbe)

    # Local-emission-normalization ablation (seconds-based window) on the headline char Whisper, on
    # BOTH TIMIT val + Buckeye-reseg, to validate the small AED collar win found offline (the reseg
    # extract is reconstructed by config -> same hash -> reuses the finished hdf).
    _ln_bk_ds = BuildBuckeyeFineDatasetJob(
        raw_dir=dl_ds_buckeye_fine.out_hub_cache_dir, max_duration_s=20.0, resegment_gap_s=1.0, min_words=2
    )
    _ln_bk_ex = ExtractInGradsPerTokenJob(
        dataset_dir=_ln_bk_ds.out_hub_cache_dir,
        dataset_key="test",
        model_config=whisper_char_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
    )
    for _ln_ex, _ln_name, _ln_dir, _ln_key in [
        (whisper_char_extract, whisper_char_name, dl_ds_timit.out_hub_cache_dir, "val"),
        (
            _ln_bk_ex,
            "whisper-base-logmel-buckeye-reseg-L2_grad-pertoken-charlev-spc",
            _ln_bk_ds.out_hub_cache_dir,
            "test",
        ),
    ]:
        for _ln_w in [1.0, 1.5]:
            _ln_ao = {"apply_softmax_over_time": True, "blank_score": -5}
            _ln_al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_ln_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=_ln_dir,
                dataset_key=_ln_key,
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_ln_ao,
                audio_energy_pow=0.5,
                blank_silence_energy_scale=1.0,
                local_norm_window_s=_ln_w,
            )
            _ln_nm = f"align/{_ln_name}-{_name_for_dict(_ln_ao)}-en0.5-sil1.0-ln{_ln_w}s"
            _ln_al.add_alias(_ln_nm)
            reg(f"{_ln_nm}-wbe.txt", _ln_al.out_wbe)

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
        reg(f"{p4e_name}.hdf", p4e_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"phi4mm-timit-{split}-recog.hyps.txt.gz", recog.out_hyps_txt)
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
        reg(f"phi4mm-timit-{split}-wer.txt", score.main_measure_value)
        reg(f"phi4mm-timit-{split}-wer-report", score.report)

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
                reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{name}.hdf", gen.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
            reg(f"align/{ext_name}-wbe_ms.txt", ext.out_wbe_ms)
            reg(f"align/{ext_name}-median_wbe_ms.txt", ext.out_median_wbe_ms)
            reg(f"align/{ext_name}-pct_within_10ms.txt", ext.out_pct_within_10ms)
            reg(f"align/{ext_name}-pct_within_25ms.txt", ext.out_pct_within_25ms)
            reg(f"align/{ext_name}-pct_within_50ms.txt", ext.out_pct_within_50ms)
            reg(f"align/{ext_name}-pct_within_100ms.txt", ext.out_pct_within_100ms)
            reg(f"align/{ext_name}-f1_20ms.txt", ext.out_f1_20ms)
            reg(f"align/{ext_name}-report.txt", ext.out_report)
            reg(f"align/{ext_name}-word_boundaries.py.gz", ext.out_word_boundaries)

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
        reg(f"{name}.hdf", gen.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"voxtral-verbatim-timit-{split}-recog.hyps.txt.gz", vx_v_recog.out_hyps_txt)
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
        reg(f"voxtral-verbatim-timit-{split}-wer.txt", vx_v_score.main_measure_value)
        reg(f"voxtral-verbatim-timit-{split}-wer-report", vx_v_score.report)

    vx_recog = RecogFromModelJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=voxtral_cfg,
    )
    vx_recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    vx_recog.add_alias("voxtral-timit-val-recog")
    reg("voxtral-timit-val-recog.hyps.txt.gz", vx_recog.out_hyps_txt)
    vx_refs = ExtractWordDetailFromHuggingFaceDatasetJob(dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_split="val")
    vx_refs.add_alias("voxtral-timit-val-refs")
    vx_hyps_norm = text_dict_normalize_file(vx_recog.out_hyps_txt)
    vx_refs_norm = text_dict_normalize_file(vx_refs.out_text)
    vx_score = sclite_score_hyps_to_ref(
        hyps_text_dict=vx_hyps_norm,
        ref_text_dict=vx_refs_norm,
        corpus_name="voxtral-timit-val",
    )
    reg("voxtral-timit-val-wer.txt", vx_score.main_measure_value)
    reg("voxtral-timit-val-wer-report", vx_score.report)

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
        reg(f"voxtral-esb-{esb_name}-{esb_split}-recog.hyps.txt.gz", vx_esb_recog.out_hyps_txt)
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
        reg(f"voxtral-esb-{esb_name}-{esb_split}-wer.txt", vx_esb_score.main_measure_value)
        reg(f"voxtral-esb-{esb_name}-{esb_split}-wer-report", vx_esb_score.report)

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
    reg("voxtral-transcribe-timit-val-recog.hyps.txt.gz", vx_t_recog.out_hyps_txt)
    vx_t_hyps_norm = text_dict_normalize_file(vx_t_recog.out_hyps_txt)
    vx_t_score = sclite_score_hyps_to_ref(
        hyps_text_dict=vx_t_hyps_norm,
        ref_text_dict=vx_refs_norm,
        corpus_name="voxtral-transcribe-timit-val",
    )
    reg("voxtral-transcribe-timit-val-wer.txt", vx_t_score.main_measure_value)
    reg("voxtral-transcribe-timit-val-wer-report", vx_t_score.report)

    vx_t_esb_recog = RecogFromModelJob(
        dataset_dir=dl_esb,
        dataset_name="librispeech",
        dataset_key="test.clean",
        model_config=voxtral_transcribe_cfg,
        max_new_tokens=512,
    )
    vx_t_esb_recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    vx_t_esb_recog.add_alias("voxtral-transcribe-esb-librispeech-test.clean-recog")
    reg("voxtral-transcribe-esb-librispeech-test.clean-recog.hyps.txt.gz", vx_t_esb_recog.out_hyps_txt)
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
    reg("voxtral-transcribe-esb-librispeech-test.clean-wer.txt", vx_t_esb_score.main_measure_value)
    reg("voxtral-transcribe-esb-librispeech-test.clean-wer-report", vx_t_esb_score.report)

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
    reg(f"{cq_lm_name}.hdf", cq_lm_extract.out_hdf)
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
        reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{cq_cl_name}.hdf", cq_cl_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)
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
    reg(f"{cq_cls_name}.hdf", cq_cls_extract.out_hdf)
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
        reg(f"{align_name}-wbe.txt", align.out_wbe)

    cq_recog = RecogFromModelJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=canary_cfg,
    )
    cq_recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cq_recog.add_alias("canary-qwen-timit-val-recog")
    reg("canary-qwen-timit-val-recog.hyps.txt.gz", cq_recog.out_hyps_txt)
    cq_refs = ExtractWordDetailFromHuggingFaceDatasetJob(dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_split="val")
    cq_hyps_norm = text_dict_normalize_file(cq_recog.out_hyps_txt)
    cq_refs_norm = text_dict_normalize_file(cq_refs.out_text)
    cq_score = sclite_score_hyps_to_ref(
        hyps_text_dict=cq_hyps_norm,
        ref_text_dict=cq_refs_norm,
        corpus_name="canary-qwen-timit-val",
    )
    reg("canary-qwen-timit-val-wer.txt", cq_score.main_measure_value)
    reg("canary-qwen-timit-val-wer-report", cq_score.report)

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
    reg(f"{vx_extract_name}.hdf", vx_extract.out_hdf)
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
        reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{vxt_extract_name}.hdf", vxt_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{vxt_lm_extract_name}.hdf", vxt_lm_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)
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
                reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{vxt_st_extract_name}.hdf", vxt_st_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{vxt_c1_extract_name}.hdf", vxt_c1_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{vxt_cl_extract_name}.hdf", vxt_cl_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
            reg(f"{vxt_cl2_extract_name}.hdf", vxt_cl2_extract.out_hdf)
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
                reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{vxt_sg_name}.hdf", vxt_sg_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{cq_extract_name}.hdf", cq_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

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
    reg(f"{cq_acc_name}.hdf", cq_acc_extract.out_hdf)
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
        reg(f"{align_name}-wbe.txt", align.out_wbe)

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
        reg(f"{cq_char_name}.hdf", cq_char_extract.out_hdf)
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
            reg(f"{align_name}-wbe.txt", align.out_wbe)

    cq_acc_recog.add_alias("canary-qwen-acc-timit-val-recog")
    reg("canary-qwen-acc-timit-val-recog.hyps.txt.gz", cq_acc_recog.out_hyps_txt)
    cq_acc_hyps_norm = text_dict_normalize_file(cq_acc_recog.out_hyps_txt)
    cq_acc_score = sclite_score_hyps_to_ref(
        hyps_text_dict=cq_acc_hyps_norm,
        ref_text_dict=cq_refs_norm,
        corpus_name="canary-qwen-acc-timit-val",
    )
    reg("canary-qwen-acc-timit-val-wer.txt", cq_acc_score.main_measure_value)
    reg("canary-qwen-acc-timit-val-wer-report", cq_acc_score.report)

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
            reg(f"{_sll_nm}-wbe.txt", _sll_al.out_wbe)

    # --- TIMIT TEST split (headline models) -- generalization beyond the val-tuned config.
    # The two best grad-align surfaces with the generic en0.5-sil1.0 setting,
    # plus the Whisper-native cross-attn baseline on test (MMS_FA test already exists).
    # Full headline cross-model set on the held-out TEST split, each with its generic
    # en0.5-sil1.0 align (transducers ALSO get plain en0.5 -- silence double-counts their
    # internal blank). (cfg, name, attr_reduction, mult_grad_by_inputs, [(energy_pow, sil_scale), ...])
    _whc_cfg_test = rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir, char_level=True, char_level_sep=" ")
    _test_specs = [
        (_whc_cfg_test, "whisper-base-logmel-timit-test-L2_grad-pertoken-charlev-spc", "L2", False, [(0.5, 1.0)]),
        (
            rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir),
            "whisper-base-logmel-timit-test-L2_grad-pertoken-subword",
            "L2",
            False,
            [(0.5, 1.0)],
        ),  # fairness-2x2: grad-subword on TIMIT-test (char twin above)
        (
            rf.build_dict(Whisper, model_dir=dl_whisper_l3.out_hub_cache_dir, char_level=True, char_level_sep=" "),
            "whisper-large-v3-logmel-timit-test-L2_grad-pertoken-charlev-spc",
            "L2",
            False,
            [(0.5, 1.0)],
        ),
        (
            rf.build_dict(Whisper, model_dir=dl_crisper.out_hub_cache_dir, char_level=True, char_level_sep=" "),
            "crisperwhisper-logmel-timit-test-L2_grad-pertoken-charlev-spc",
            "L2",
            False,
            [(0.5, 1.0)],
        ),
        (
            rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out"),
            "wav2vec2ctc-fproj_out-timit-test-L2_grad-pertoken",
            "L2",
            False,
            [(0.5, 1.0)],
        ),
        # prefix_diff on the CTCs that already grad-align well (wav2vec2 uses inb_true at 59ms; phoneme
        # likewise): does prefix_diff also beat inb_true, or is inb_true model-specifically better here?
        (
            rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", per_token_score="prefix_diff"),
            "wav2vec2ctc-fproj_out-prefdiff-timit-test-L2_grad-pertoken",
            "L2",
            False,
            [(0.5, 1.0)],
        ),
        (
            rf.build_dict(
                Wav2Vec2PhonemeCtc,
                model_dir=dl_w2v_phoneme.out_hub_cache_dir,
                g2p_word_targets=True,
                per_token_score="prefix_diff",
            ),
            "phoneme-vitouphy-prefdiff-timit-test-L2_grad-pertoken-g2pword",
            "L2",
            False,
            [(0.5, 1.0)],
        ),
        (
            voxtral_charlev_logmel_cfg,
            "voxtral-charlevlogmel-timit-test-L1_grad-pertoken",
            "L1",
            False,
            [(0.5, 1.0)],
        ),
        (pt_csp_cfg, "phi4mm-timit-test-L2_e_grad-pertoken-charlev-spc", "L2", True, [(0.5, 1.0)]),
        (
            canary_charlev_logmel_st15_cfg,
            "canary-qwen-charlev-spc-logmel-st15-timit-test-L1_grad-pertoken",
            "L1",
            False,
            [(0.5, 1.0)],
        ),
        (
            pk_cfg,
            "parakeet-rnnt-1.1b-logmel-timit-test-L2_grad-pertoken",
            "L2",
            False,
            [(0.5, 1.0), (0.5, 0.0)],
        ),
        # Emformer RNN-T: the STREAMING transducer (prefix score); same splits as the offline ones.
        (
            rnnt_px_cfg,
            "emformer-rnnt-prefix-logmel-timit-test-L2_grad-pertoken",
            "L2",
            False,
            [(0.5, 1.0), (0.5, 0.0)],
        ),
        (
            tdt_grad_cfg,
            "parakeet-tdt-0.6b-v2-logmel-timit-test-L2_grad-pertoken",
            "L2",
            False,
            [(0.5, 1.0), (0.5, 0.0)],
        ),
        # OWSM-CTC: a general graphemic CTC (BPE subword units, ~12.5 Hz emission).
        (owsm_ctc_cfg, "owsm-ctc-v4-1b-timit-test-L2_grad-pertoken", "L2", False, [(0.5, 1.0)]),
        # Nvidia CTC (parakeet-ctc-1.1b): the general graphemic CTC representative (plain FastConformer).
        (parakeet_ctc_cfg, "parakeet-ctc-1.1b-timit-test-L2_grad-pertoken", "L2", False, [(0.5, 1.0)]),
        # prefix_diff (AED-consistent) score -- the fix for the conformer/branchformer CTCs.
        (
            parakeet_ctc_prefdiff_cfg,
            "parakeet-ctc-1.1b-prefdiff-timit-test-L2_grad-pertoken",
            "L2",
            False,
            [(0.5, 1.0)],
        ),
        (owsm_ctc_prefdiff_cfg, "owsm-ctc-v4-1b-prefdiff-timit-test-L2_grad-pertoken", "L2", False, [(0.5, 1.0)]),
        # inb_true (robust universal) on the same subword CTCs, to compare vs prefix_diff.
        (parakeet_ctc_inbtrue_cfg, "parakeet-ctc-1.1b-inbtrue-timit-test-L2_grad-pertoken", "L2", False, [(0.5, 1.0)]),
        (owsm_ctc_inbtrue_cfg, "owsm-ctc-v4-1b-inbtrue-timit-test-L2_grad-pertoken", "L2", False, [(0.5, 1.0)]),
        # prefix_fwd (the real CTC prefix score) on all four CTCs -- the uniform-winner candidate.
        (
            parakeet_ctc_prefixfwd_cfg,
            "parakeet-ctc-1.1b-prefixfwd-timit-test-L2_grad-pertoken",
            "L2",
            False,
            [(0.5, 1.0)],
        ),
        (owsm_ctc_prefixfwd_cfg, "owsm-ctc-v4-1b-prefixfwd-timit-test-L2_grad-pertoken", "L2", False, [(0.5, 1.0)]),
        (
            rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", per_token_score="prefix_fwd"),
            "wav2vec2ctc-fproj_out-prefixfwd-timit-test-L2_grad-pertoken",
            "L2",
            False,
            [(0.5, 1.0)],
        ),
        (
            rf.build_dict(
                Wav2Vec2PhonemeCtc,
                model_dir=dl_w2v_phoneme.out_hub_cache_dir,
                g2p_word_targets=True,
                per_token_score="prefix_fwd",
            ),
            "phoneme-vitouphy-prefixfwd-timit-test-L2_grad-pertoken-g2pword",
            "L2",
            False,
            [(0.5, 1.0)],
        ),
        # OWSM-CTC per inter-CTC layer (side table): grad-align WBE from blocks 6/12/15/21 (27=above).
        *[
            (_ocfg, f"owsm-ctc-v4-1b-lyr{_ol}-timit-test-L2_grad-pertoken", "L2", False, [(0.5, 1.0)])
            for _ol, _ocfg in owsm_ctc_cfg_by_layer.items()
        ],
    ]
    for _t_cfg, _t_name, _t_attr, _t_mgi, _t_aligns in _test_specs:
        _t_ex = ExtractInGradsPerTokenJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="test",
            model_config=_t_cfg,
            mult_grad_by_inputs=_t_mgi,
            attr_reduction=_t_attr,
            # batched_backward (vmap per-token VJP) for the prefix_fwd CTC extracts: ~tokens-per-word
            # faster, mathematically identical. Opt-in (hash-stable) so finished bb=False jobs untouched.
            **({"batched_backward": True} if "prefixfwd" in _t_name else {}),
        )
        _t_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _t_ex.rqmt = {**_t_ex.rqmt, "time": 24}
        _t_ex.add_alias(_t_name)
        reg(f"{_t_name}.hdf", _t_ex.out_hdf)
        for _t_ep, _t_sc in _t_aligns:
            _t_ao = {"apply_softmax_over_time": True, "blank_score": -5}
            _t_al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_t_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="test",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_t_ao,
                audio_energy_pow=_t_ep,
                blank_silence_energy_scale=_t_sc,
            )
            _t_nm = f"align/{_t_name}-{_name_for_dict(_t_ao)}-en{_t_ep}-sil{_t_sc}"
            _t_al.add_alias(_t_nm)
            reg(f"{_t_nm}-wbe.txt", _t_al.out_wbe)
    # Nvidia CTC posteriors baseline: torchaudio CTC forced-align on parakeet-ctc's own emission.
    _pc_fa_test = ParakeetCtcForcedAlignJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="test",
        model_dir=dl_parakeet_ctc.out_hub_cache_dir,
        overlay_path=_NEMO_OVERLAY,
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
    )
    _pc_fa_test.add_alias("baseline-parakeet-ctc-1.1b-timit-test")
    reg("baseline-parakeet-ctc-1.1b-timit-test-wbe.txt", _pc_fa_test.out_word_wbe)
    # OWSM-CTC posteriors baseline per inter-CTC emit block (forced-align of the block's emission).
    for _ol in [None, 6, 12, 15, 21]:
        _otag = "" if _ol is None else f"lyr{_ol}-"
        _ofa = OwsmCtcForcedAlignJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="test",
            model_dir=dl_owsm_ctc.out_hub_cache_dir,
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            layer=_ol,
        )
        _ofa.add_alias(f"baseline-owsm-ctc-v4-1b-{_otag}timit-test")
        reg(f"baseline-owsm-ctc-v4-1b-{_otag}timit-test-wbe.txt", _ofa.out_word_wbe)

    # === Speech-LLM prompt-splice sensitivity (Phi-4-MM, side table) ===================
    # Forced GT transcription on TIMIT-test; vary ONLY the instruction spliced around the audio.
    # char + subword targets; one prompt instructs char-level (user idea). Batched grad extract.
    _PP_PROMPTS = [
        ("default", "Transcribe the audio clip into text."),
        ("verbatim", "Please transcribe the audio verbatim, exactly as spoken."),
        ("word", "Transcribe the following exactly and accurately, word for word:"),
        ("char", "Transcribe the audio at the character level, spelling out each word letter by letter."),
    ]
    for _pp_tag, _pp_prompt in _PP_PROMPTS:
        for _pp_level in ["subword", "char"]:
            if _pp_tag == "char" and _pp_level == "subword":
                continue  # the char-level instruction only pairs with char-level targets
            # eager attention so batched_backward (vmap-VJP) works -- FlashAttention's backward has no
            # vmap rule; eager's forward is a bit slower but the single batched backward more than pays it back.
            if _pp_level == "char":
                _pp_cfg = _phi4mm_model_config(
                    dl_phi4mi_dir,
                    char_level=True,
                    char_level_sep=" ",
                    speech_prompt=_pp_prompt,
                    attn_implementation="eager",
                )
            else:
                _pp_cfg = _phi4mm_model_config(dl_phi4mi_dir, speech_prompt=_pp_prompt, attn_implementation="eager")
            _pp_name = f"phi4mm-prompt-{_pp_tag}-{_pp_level}-timit-test-L2_e_grad-pertoken"
            _pp_ex = ExtractInGradsPerTokenJob(
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="test",
                model_config=_pp_cfg,
                mult_grad_by_inputs=True,
                attr_reduction="L2",
                batched_backward=True,  # OK with eager attn (FlashAttention's backward has no vmap rule)
            )
            _pp_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            _pp_ex.rqmt = {**_pp_ex.rqmt, "time": 24}
            _pp_ex.add_alias(_pp_name)
            reg(f"{_pp_name}.hdf", _pp_ex.out_hdf)
            _pp_ao = {"apply_softmax_over_time": True, "blank_score": -5}
            _pp_al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_pp_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="test",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_pp_ao,
                audio_energy_pow=0.5,
                blank_silence_energy_scale=1.0,
            )
            _pp_nm = f"align/{_pp_name}-{_name_for_dict(_pp_ao)}-en0.5-sil1.0"
            _pp_al.add_alias(_pp_nm)
            reg(f"{_pp_nm}-wbe.txt", _pp_al.out_wbe)

    # === Cost benchmark: forward vs batched grad-backward, per model, same GPU =========
    # Speech LLMs need eager attention for the batched backward (FlashAttention's backward has no vmap
    # rule); the CTC/transducer/Whisper models batch on their default attention.
    _cost_models = {
        "Wav2Vec2-CTC": rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", per_token_score="prefix_fwd"),
        "Nvidia CTC": parakeet_ctc_prefixfwd_cfg,
        "OWSM-CTC": owsm_ctc_prefixfwd_cfg,
        "Whisper-base": rf.build_dict(
            Whisper, model_dir=dl_whisper.out_hub_cache_dir, char_level=True, char_level_sep=" "
        ),
        "Parakeet RNN-T": pk_cfg,
        "Parakeet TDT": tdt_grad_cfg,
        "Voxtral": voxtral_charlev_logmel_cfg,
        "Phi-4-MM": _phi4mm_model_config(
            dl_phi4mi_dir, char_level=True, char_level_sep=" ", attn_implementation="eager"
        ),
        "Canary-Qwen": canary_charlev_logmel_st15_cfg,
    }
    _cost_bench = AlignCostBenchmarkJob(
        model_configs=_cost_models,
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="test",
        num_seqs=40,
    )
    _cost_bench.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _cost_bench.add_alias("cost-benchmark")
    reg("cost-benchmark-metrics.txt", _cost_bench.out_metrics)
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
    reg("baseline-whisper-crossattn-timit-test-wbe.txt", whisper_fa_test_metric.out_wbe)
    # fairness-2x2: cross-attn CHAR on TIMIT-test (char twin of the subword baseline above; segA has it).
    whisper_fa_test_char = WhisperCrossAttnForcedAlignJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_key="test", overlay=_WHISPER_TS_OVERLAY, char_level=True
    )
    whisper_fa_test_char.add_alias("baseline-whisper-crossattn-charlev-timit-test")
    whisper_fa_test_char_metric = CalcAlignmentMetricsFromWordBoundariesJob(
        word_boundaries_hdf=whisper_fa_test_char.out_hdf,
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="test",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
    )
    reg("baseline-whisper-crossattn-charlev-timit-test-wbe.txt", whisper_fa_test_char_metric.out_wbe)

    # --- MFA (Montreal Forced Aligner) baseline infra ---------------------------
    # Pull the MFA Docker image into a tracked .sif, and wrap `mfa` as a plain executable so the
    # (to-be-added) MfaForcedAlignJob can take a configurable mfa_exe (native binary OR this
    # container wrapper). Image/wrapper registered as outputs so they build now; the align job
    # is wired once the in-container MFA CLI is verified.
    _mfa_image = PullApptainerImageJob("docker://mmcauliffe/montreal-forced-aligner:latest")
    reg("mfa/image.sif", _mfa_image.out_image)
    # `work` symlinks to the hpcwork scratch FS, which is a per-PROJECT nested mount
    # (/rwthfs/rz/cluster/hpcwork/p0023999) -- binding the hpcwork parent misses it. Bind home (the
    # symlink + recipe paths) AND the project hpcwork mount (the symlink targets / actual job files).
    _mfa_exe = ApptainerExeWrapperJob(
        _mfa_image.out_image,
        command="mfa",
        bind=["/rwthfs/rz/cluster/home", "/rwthfs/rz/cluster/hpcwork/p0023999"],
    )
    reg("mfa/mfa-run.sh", _mfa_exe.out_exe)
    _mfa_models = MfaDownloadModelJob(
        mfa_exe=_mfa_exe.out_exe,
        models=[("acoustic", "english_us_arpa"), ("dictionary", "english_us_arpa"), ("g2p", "english_us_arpa")],
    )
    # MFA forced-align baseline (vs GOLD boundaries) on TIMIT val + Buckeye-reseg, like MMS_FA.
    _mfa_bk = BuildBuckeyeFineDatasetJob(
        raw_dir=dl_ds_buckeye_fine.out_hub_cache_dir, max_duration_s=20.0, resegment_gap_s=1.0, min_words=2
    )
    for _mfa_dir, _mfa_key, _mfa_tag in [
        (dl_ds_timit.out_hub_cache_dir, "val", "timit-val"),
        (dl_ds_timit.out_hub_cache_dir, "test", "timit-test"),
        (_mfa_bk.out_hub_cache_dir, "test", "buckeye-reseg"),
    ]:
        _mfa_al = MfaForcedAlignJob(
            dataset_dir=_mfa_dir,
            dataset_key=_mfa_key,
            mfa_exe=_mfa_exe.out_exe,
            model_root=_mfa_models.out_model_root,
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
        )
        _mfa_al.add_alias(f"baseline-mfa-{_mfa_tag}")
        # WBE computed in-job (robust to partial MFA coverage); no separate metric job needed.
        reg(f"baseline-mfa-{_mfa_tag}-wbe.txt", _mfa_al.out_wbe)
        reg(f"baseline-mfa-{_mfa_tag}-metrics.txt", _mfa_al.out_metrics)
        reg(f"baseline-mfa-{_mfa_tag}-coverage.txt", _mfa_al.out_coverage)

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
        reg(f"{_bk_tag}-fine-dataset", _bk_ds_job.out_hub_cache_dir)
        _bk_dir = _bk_ds_job.out_hub_cache_dir

        bk_fa = ForcedAlignBaselineJob(dataset_dir=_bk_dir, dataset_key="test")
        bk_fa.add_alias(f"baseline-mms_fa-{_bk_tag}")
        bk_fa_metric = CalcAlignmentMetricsFromWordBoundariesJob(
            word_boundaries_hdf=bk_fa.out_hdf,
            dataset_dir=_bk_dir,
            dataset_key="test",
            dataset_offset_factors=_bk_off,
        )
        reg(f"baseline-mms_fa-{_bk_tag}-wbe.txt", bk_fa_metric.out_wbe)

        # Whisper cross-attn DTW = the AED-native baseline (the "grad beats Whisper's own attention"
        # comparison), on Buckeye too, matching TIMIT val/test.
        bk_wca = WhisperCrossAttnForcedAlignJob(dataset_dir=_bk_dir, dataset_key="test", overlay=_WHISPER_TS_OVERLAY)
        bk_wca.add_alias(f"baseline-whisper-crossattn-{_bk_tag}")
        bk_wca_metric = CalcAlignmentMetricsFromWordBoundariesJob(
            word_boundaries_hdf=bk_wca.out_hdf,
            dataset_dir=_bk_dir,
            dataset_key="test",
            dataset_offset_factors=_bk_off,
        )
        reg(f"baseline-whisper-crossattn-{_bk_tag}-wbe.txt", bk_wca_metric.out_wbe)

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
                # Parakeet-TDT (2nd transducer) to complete the transducer pair on Buckeye, matching TIMIT.
                (tdt_grad_cfg, f"parakeet-tdt-0.6b-v2-logmel-{_bk_tag}-L2_grad-pertoken", "L2", False),
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
            _bk_ex.rqmt = {**_bk_ex.rqmt, "time": 24}
            _bk_ex.add_alias(_bk_name)
            reg(f"{_bk_name}.hdf", _bk_ex.out_hdf)
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
            reg(f"{_bk_nm}-wbe.txt", _bk_al.out_wbe)

    # --- Buckeye SEGMENTATION-PROTOCOL ablation: variants A/B/C, ~5 h speaker-stratified each -----------
    # A = split>1s + recursive split-to-<=20s (floor .5); B = size-only split-to-<=20s; C = split>1s + drop>20s.
    # All forced <=20s so EVERY method (incl char-level) runs on the SAME data. Few methods only, to pick the
    # headline Buckeye segmentation later -- do NOT yet run on all models.
    _seg_ao = {"apply_softmax_over_time": True, "blank_score": -5}
    _seg_whc_cfg = rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir, char_level=True, char_level_sep=" ")
    _seg_grad_methods = [
        (_seg_whc_cfg, "whisper-base-logmel-{tag}-L2_grad-pertoken-charlev-spc", "L2", False),
        (
            rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out"),
            "wav2vec2ctc-fproj_out-{tag}-L2_grad-pertoken",
            "L2",
            False,
        ),
    ]
    for _sv, _sv_kw in [
        ("A", dict(resegment_gap_s=1.0, split_up_to_max_seq_len_s=18.0)),
        ("B", dict(split_up_to_max_seq_len_s=18.0)),
        ("C", dict(resegment_gap_s=1.0, drop_above_max_seq_len_s=18.0)),
    ]:
        _sv_tag = f"buckeye-seg{_sv}-5h"
        _sv_ds = BuildBuckeyeFineDatasetJob(
            raw_dir=dl_ds_buckeye_fine.out_hub_cache_dir,
            min_words=2,
            skip_misaligned_wavs=True,
            subsample_target_h=5.0,
            subsample_seed=42,
            **_sv_kw,
        )
        reg(f"{_sv_tag}-dataset", _sv_ds.out_hub_cache_dir)
        _sv_dir = _sv_ds.out_hub_cache_dir

        # MMS_FA (CTC forced-align) baseline
        _sv_fa = ForcedAlignBaselineJob(dataset_dir=_sv_dir, dataset_key="test")
        _sv_fa.add_alias(f"baseline-mms_fa-{_sv_tag}")
        _sv_fa_m = CalcAlignmentMetricsFromWordBoundariesJob(
            word_boundaries_hdf=_sv_fa.out_hdf, dataset_dir=_sv_dir, dataset_key="test", dataset_offset_factors=_bk_off
        )
        reg(f"baseline-mms_fa-{_sv_tag}-wbe.txt", _sv_fa_m.out_wbe)

        # Whisper cross-attn DTW baseline
        _sv_wca = WhisperCrossAttnForcedAlignJob(dataset_dir=_sv_dir, dataset_key="test", overlay=_WHISPER_TS_OVERLAY)
        _sv_wca.add_alias(f"baseline-whisper-crossattn-{_sv_tag}")
        _sv_wca_m = CalcAlignmentMetricsFromWordBoundariesJob(
            word_boundaries_hdf=_sv_wca.out_hdf, dataset_dir=_sv_dir, dataset_key="test", dataset_offset_factors=_bk_off
        )
        reg(f"baseline-whisper-crossattn-{_sv_tag}-wbe.txt", _sv_wca_m.out_wbe)

        # MFA (GMM-HMM) baseline -- the ceiling
        _sv_mfa = MfaForcedAlignJob(
            dataset_dir=_sv_dir,
            dataset_key="test",
            mfa_exe=_mfa_exe.out_exe,
            model_root=_mfa_models.out_model_root,
            dataset_offset_factors=_bk_off,
        )
        _sv_mfa.add_alias(f"baseline-mfa-{_sv_tag}")
        reg(f"baseline-mfa-{_sv_tag}-wbe.txt", _sv_mfa.out_wbe)

        # grad-align surfaces (en0.5-sil1.0): AED-Whisper char + CTC-Wav2Vec2
        for _cfg, _nmt, _attr, _mgi in _seg_grad_methods:
            _nm = _nmt.format(tag=_sv_tag)
            _ex = ExtractInGradsPerTokenJob(
                dataset_dir=_sv_dir,
                dataset_key="test",
                model_config=_cfg,
                mult_grad_by_inputs=_mgi,
                attr_reduction=_attr,
            )
            _ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            _ex.rqmt = {**_ex.rqmt, "time": 24}
            _ex.add_alias(_nm)
            reg(f"{_nm}.hdf", _ex.out_hdf)
            _al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=_sv_dir,
                dataset_key="test",
                dataset_offset_factors=_bk_off,
                align_opts=_seg_ao,
                audio_energy_pow=0.5,
                blank_silence_energy_scale=1.0,
            )
            _alnm = f"align/{_nm}-{_name_for_dict(_seg_ao)}-en0.5-sil1.0"
            _al.add_alias(_alnm)
            reg(f"{_alnm}-wbe.txt", _al.out_wbe)
            # zsk (audio-free, grad-stats silence): compare vs energy-driven on the noisier segments
            # (segB), where low energy != silence (background noise) may fool the energy blank.
            _al_zsk = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=_sv_dir,
                dataset_key="test",
                dataset_offset_factors=_bk_off,
                align_opts=_seg_ao,
                audio_energy_pow=0.5,
                blank_grad_zscore_kappa=1.0,
            )
            _zsknm = f"align/{_nm}-{_name_for_dict(_seg_ao)}-en0.5-zsk1.0"
            _al_zsk.add_alias(_zsknm)
            reg(f"{_zsknm}-wbe.txt", _al_zsk.out_wbe)
            # word-topology twin on the segments too.
            _al_wt = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=_sv_dir,
                dataset_key="test",
                dataset_offset_factors=_bk_off,
                align_opts=_seg_ao,
                audio_energy_pow=0.5,
                blank_silence_energy_scale=1.0,
                word_topology=True,
            )
            _wtnm = f"align/{_nm}-{_name_for_dict(_seg_ao)}-en0.5-sil1.0-wordtopo"
            _al_wt.add_alias(_wtnm)
            reg(f"{_wtnm}-wbe.txt", _al_wt.out_wbe)

        if _sv == "A":
            # Variant A is the chosen headline Buckeye -> run the remaining cross-model set on it
            # (whisper-char + wav2vec2 grad + MMS_FA / cross-attn / MFA already wired on segA above).
            # parakeet-rnnt: batched_backward (validated exact-equivalent, CPU+GPU test node) --
            # the sequential variant times out even at 24h. All finished extracts keep the default
            # (False, hash-excluded) so they don't re-hash.
            for _hA_cfg, _hA_name, _hA_attr, _hA_mgi, _hA_bb in [
                (voxtral_charlev_logmel_cfg, f"voxtral-charlevlogmel-{_sv_tag}-L1_grad-pertoken", "L1", False, False),
                (pk_cfg, f"parakeet-rnnt-1.1b-logmel-{_sv_tag}-L2_grad-pertoken", "L2", False, True),
                (tdt_grad_cfg, f"parakeet-tdt-0.6b-v2-logmel-{_sv_tag}-L2_grad-pertoken", "L2", False, False),
                (pt_csp_cfg, f"phi4mm-{_sv_tag}-L2_e_grad-pertoken-charlev-spc", "L2", True, False),
                (
                    canary_charlev_logmel_st15_cfg,
                    f"canary-qwen-charlev-spc-logmel-st15-{_sv_tag}-L1_grad-pertoken",
                    "L1",
                    False,
                    False,
                ),
            ]:
                _hA_ex = ExtractInGradsPerTokenJob(
                    dataset_dir=_sv_dir,
                    dataset_key="test",
                    model_config=_hA_cfg,
                    mult_grad_by_inputs=_hA_mgi,
                    attr_reduction=_hA_attr,
                    batched_backward=_hA_bb,
                )
                _hA_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                _hA_ex.rqmt = {**_hA_ex.rqmt, "time": 24}
                _hA_ex.add_alias(_hA_name)
                reg(f"{_hA_name}.hdf", _hA_ex.out_hdf)
                _hA_al = WordAlignFromPerTokenGradsJob(
                    grad_score_hdf=_hA_ex.out_hdf,
                    grad_score_key="data",
                    dataset_dir=_sv_dir,
                    dataset_key="test",
                    dataset_offset_factors=_bk_off,
                    align_opts=_seg_ao,
                    audio_energy_pow=0.5,
                    blank_silence_energy_scale=1.0,
                )
                _hA_alnm = f"align/{_hA_name}-{_name_for_dict(_seg_ao)}-en0.5-sil1.0"
                _hA_al.add_alias(_hA_alnm)
                reg(f"{_hA_alnm}-wbe.txt", _hA_al.out_wbe)

    # === Buckeye variant A (headline) extra experiments: fairness 2x2 (grad/cross-attn x char/subword),
    # phoneme model via espeak-ng G2P, transducer no-sil, and grad-score x energy/silence ablations
    # (cheap on the small whisper-base). The segA dataset job is shared (same params -> same hash). ===
    _xa_ds = BuildBuckeyeFineDatasetJob(
        raw_dir=dl_ds_buckeye_fine.out_hub_cache_dir,
        resegment_gap_s=1.0,
        split_up_to_max_seq_len_s=18.0,
        min_words=2,
        skip_misaligned_wavs=True,
        subsample_target_h=5.0,
        subsample_seed=42,
    )
    _xa_dir = _xa_ds.out_hub_cache_dir
    _xa_tag = "buckeye-segA-5h"
    _xa_off = _DATASET_OFFSET_FACTORS["timit"]
    _xa_ao = {"apply_softmax_over_time": True, "blank_score": -5}

    def _xa_extract(cfg, name, attr, mgi, time=24, bb=False):
        ex = ExtractInGradsPerTokenJob(
            dataset_dir=_xa_dir,
            dataset_key="test",
            model_config=cfg,
            mult_grad_by_inputs=mgi,
            attr_reduction=attr,
            batched_backward=bb,
        )
        ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        ex.rqmt = {**ex.rqmt, "time": time}
        if name is not None:  # None -> reuse a job already aliased/registered elsewhere
            ex.add_alias(name)
            reg(f"{name}.hdf", ex.out_hdf)
        return ex

    def _xa_align(
        ex,
        name,
        sfx,
        energy=0.5,
        sil=1.0,
        zsk=None,
        word_topology=False,
        dtw_no_blank=False,
        whisper_dtw=False,
        apply_log=True,
        _twin=True,
    ):
        kw = {"blank_grad_zscore_kappa": zsk} if zsk is not None else {"blank_silence_energy_scale": sil}
        if word_topology:
            kw["word_topology"] = True
        if dtw_no_blank:
            kw["dtw_no_blank"] = True
        if whisper_dtw:
            kw["whisper_dtw"] = True
        # apply_log=True reuses _xa_ao exactly (hash-stable); False = the no-log variant.
        _ao = _xa_ao if apply_log else {**_xa_ao, "apply_log": False}
        al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=ex.out_hdf,
            grad_score_key="data",
            dataset_dir=_xa_dir,
            dataset_key="test",
            dataset_offset_factors=_xa_off,
            align_opts=_ao,
            audio_energy_pow=energy,
            **kw,
        )
        nm = f"align/{name}-{_name_for_dict(_ao)}-{sfx}"
        al.add_alias(nm)
        reg(f"{nm}-wbe.txt", al.out_wbe)
        # Word-aware-topology twin of every headline (en0.5-sil1.0) align: blank only between words,
        # not within. Broad degradation sweep across all segA models before making it the default.
        if _twin and not word_topology and not dtw_no_blank and not whisper_dtw and sfx == "en0.5-sil1.0":
            _xa_align(ex, name, sfx + "-wordtopo", energy=energy, sil=sil, zsk=zsk, word_topology=True, _twin=False)
            _xa_align(ex, name, sfx + "-dtw", energy=energy, sil=sil, zsk=zsk, dtw_no_blank=True, _twin=False)
            _xa_align(ex, name, sfx + "-wdtw", energy=energy, sil=sil, zsk=zsk, whisper_dtw=True, _twin=False)
        return al

    # (1) Fairness: grad SUBWORD whisper (compare to the existing crossattn-subword baseline).
    _xa_sub_ex = _xa_extract(
        rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir),
        f"whisper-base-logmel-{_xa_tag}-L2_grad-pertoken-subword",
        "L2",
        False,
    )
    _xa_align(_xa_sub_ex, f"whisper-base-logmel-{_xa_tag}-L2_grad-pertoken-subword", "en0.5-sil1.0")

    # (2) Fairness: cross-attn CHAR (compare to grad-char) -- identical char tokenization, only the signal differs.
    _xa_cac = WhisperCrossAttnForcedAlignJob(
        dataset_dir=_xa_dir, dataset_key="test", overlay=_WHISPER_TS_OVERLAY, char_level=True
    )
    _xa_cac.add_alias(f"baseline-whisper-crossattn-charlev-{_xa_tag}")
    _xa_cac_m = CalcAlignmentMetricsFromWordBoundariesJob(
        word_boundaries_hdf=_xa_cac.out_hdf,
        dataset_dir=_xa_dir,
        dataset_key="test",
        dataset_offset_factors=_xa_off,
    )
    reg(f"baseline-whisper-crossattn-charlev-{_xa_tag}-wbe.txt", _xa_cac_m.out_wbe)

    # (3) Phoneme model (vitouphy) via espeak-ng G2P word targets: grad-align + CTC forced-align baseline.
    _xa_ph_ex = _xa_extract(
        rf.build_dict(Wav2Vec2PhonemeCtc, model_dir=dl_w2v_phoneme.out_hub_cache_dir, g2p_word_targets=True),
        f"phoneme-vitouphy-{_xa_tag}-L2_grad-pertoken-g2pword",
        "L2",
        False,
    )
    _xa_align(_xa_ph_ex, f"phoneme-vitouphy-{_xa_tag}-L2_grad-pertoken-g2pword", "en0.5-sil1.0")
    _xa_phfa = ForcedAlignPhonemeBaselineJob(
        dataset_dir=_xa_dir,
        dataset_key="test",
        model_dir=dl_w2v_phoneme.out_hub_cache_dir,
        dataset_offset_factors=_xa_off,
        g2p_word_targets=True,
    )
    _xa_phfa.add_alias(f"baseline-phoneme-fa-{_xa_tag}")
    reg(f"baseline-phoneme-fa-{_xa_tag}-word-wbe.txt", _xa_phfa.out_word_wbe)

    # (4) Transducer no-sil (TIMIT showed transducers prefer sil0.0): re-align the shared extracts
    # (parakeet-rnnt: batched_backward, matching the headline-A job).
    for _xa_t_cfg, _xa_t_name, _xa_t_bb in [
        (pk_cfg, f"parakeet-rnnt-1.1b-logmel-{_xa_tag}-L2_grad-pertoken", True),
        (tdt_grad_cfg, f"parakeet-tdt-0.6b-v2-logmel-{_xa_tag}-L2_grad-pertoken", False),
    ]:
        _xa_align(_xa_extract(_xa_t_cfg, None, "L2", False, bb=_xa_t_bb), _xa_t_name, "en0.5-sil0.0", sil=0.0)

    # (4b) TDT duration-aware prefix score (the emission-scored TDT rows above are the cruder
    # time-marginal): exact TDT forward incl. per-emission durations, vectorized + verified
    # like the RNN-T lattice. Both silence settings (TDT preferred sil1.0 on segA).
    _xa_tdtp_name = f"parakeet-tdt-0.6b-v2-logmel-prefix-{_xa_tag}-L2_grad-pertoken"
    _xa_tdtp_ex = _xa_extract(
        rf.build_dict(
            ParakeetRnnt,
            model_dir=dl_parakeet_tdt.out_hub_cache_dir,
            per_token_score="prefix",
            overlay_path=_NEMO_OVERLAY,
        ),
        _xa_tdtp_name,
        "L2",
        False,
    )
    _xa_align(_xa_tdtp_ex, _xa_tdtp_name, "en0.5-sil1.0")
    _xa_align(_xa_tdtp_ex, _xa_tdtp_name, "en0.5-sil0.0", sil=0.0)

    # (4c) Whisper-large-v3 (largest released Whisper): scale-up of the headline AED row,
    # char-level. batched_backward as verified for whisper-base (same architecture/kernels;
    # re-verified on the test node before the job runs).
    _xa_wl3_name = f"whisper-large-v3-logmel-{_xa_tag}-L2_grad-pertoken-charlev-spc"
    _xa_wl3_ex = _xa_extract(
        rf.build_dict(Whisper, model_dir=dl_whisper_l3.out_hub_cache_dir, char_level=True, char_level_sep=" "),
        _xa_wl3_name,
        "L2",
        False,
        bb=True,
    )
    _xa_align(_xa_wl3_ex, _xa_wl3_name, "en0.5-sil1.0")

    # (4e) Native transducer Viterbi alignment (the transducer's OWN alignment path,
    # quantized to its 80 ms encoder grid): the same-model counterpart of the parakeet grad rows.
    for _nt_cfg, _nt_name in [
        (pk_cfg, "parakeet-rnnt-1.1b"),
        (tdt_cfg, "parakeet-tdt-0.6b-v2"),
        (rnnt_cfg, "emformer-rnnt"),  # streaming transducer
    ]:
        for _nt_dir, _nt_key, _nt_off, _nt_tag in [
            (_xa_dir, "test", _xa_off, _xa_tag),
            (dl_ds_timit.out_hub_cache_dir, "test", _DATASET_OFFSET_FACTORS["timit"], "timit-test"),
        ]:
            _nt = NativeTransducerAlignJob(dataset_dir=_nt_dir, dataset_key=_nt_key, model_config=_nt_cfg)
            _nt.add_alias(f"baseline-{_nt_name}-native-viterbi-{_nt_tag}")
            _nt_m = CalcAlignmentMetricsFromWordBoundariesJob(
                word_boundaries_hdf=_nt.out_word_boundaries_hdf,
                dataset_dir=_nt_dir,
                dataset_key=_nt_key,
                dataset_offset_factors=_nt_off,
            )
            reg(f"baseline-{_nt_name}-native-viterbi-{_nt_tag}-wbe.txt", _nt_m.out_wbe)

    # --- Streaming FastConformer (NeMo hybrid CTC + RNN-T, cache-aware streaming): one checkpoint
    # gives a streaming CTC and a streaming transducer. att_context_size [70,6] ~= 480 ms look-ahead.
    # Both heads GPU-verified. grad (CTC+RNN-T) + CTC forced-align + RNN-T native-viterbi, test+segA.
    _fc_att = [70, 6]
    fc_ctc_cfg = rf.build_dict(
        FastConformerStreaming,
        model_dir=dl_fc_stream.out_hub_cache_dir,
        overlay_path=_NEMO_OVERLAY,
        head="ctc",
        att_context_size=_fc_att,
    )
    fc_rnnt_cfg = rf.build_dict(
        FastConformerStreaming,
        model_dir=dl_fc_stream.out_hub_cache_dir,
        overlay_path=_NEMO_OVERLAY,
        head="rnnt",
        att_context_size=_fc_att,
    )
    for _fc_dir, _fc_key, _fc_off, _fc_tag in [
        (dl_ds_timit.out_hub_cache_dir, "test", _DATASET_OFFSET_FACTORS["timit"], "timit-test"),
        (_xa_dir, "test", _xa_off, _xa_tag),
    ]:
        _fc_ao = {"apply_softmax_over_time": True, "blank_score": -5}
        for _fc_cfg, _fc_h in [(fc_ctc_cfg, "ctc"), (fc_rnnt_cfg, "rnnt")]:
            _fc_name = f"fastconformer-stream-{_fc_h}-{_fc_tag}-L2_grad-pertoken"
            _fc_ex = ExtractInGradsPerTokenJob(
                dataset_dir=_fc_dir,
                dataset_key=_fc_key,
                model_config=_fc_cfg,
                mult_grad_by_inputs=False,
                attr_reduction="L2",
            )
            _fc_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            _fc_ex.add_alias(_fc_name)
            reg(f"{_fc_name}.hdf", _fc_ex.out_hdf)
            _fc_al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_fc_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=_fc_dir,
                dataset_key=_fc_key,
                dataset_offset_factors=_fc_off,
                align_opts=_fc_ao,
                audio_energy_pow=0.5,
                blank_silence_energy_scale=1.0,
            )
            _fc_alnm = f"align/{_fc_name}-{_name_for_dict(_fc_ao)}-en0.5-sil1.0"
            _fc_al.add_alias(_fc_alnm)
            reg(f"{_fc_alnm}-wbe.txt", _fc_al.out_wbe)
        # CTC forced-align (torchaudio Viterbi on the streaming CTC emission)
        _fc_cfa = ParakeetCtcForcedAlignJob(
            dataset_dir=_fc_dir,
            dataset_key=_fc_key,
            model_config=fc_ctc_cfg,
            dataset_offset_factors=_fc_off,
        )
        _fc_cfa.add_alias(f"baseline-fastconformer-stream-ctc-{_fc_tag}")
        reg(f"baseline-fastconformer-stream-ctc-{_fc_tag}-wbe.txt", _fc_cfa.out_word_wbe)
        # RNN-T native-viterbi forced-align
        _fc_nt = NativeTransducerAlignJob(dataset_dir=_fc_dir, dataset_key=_fc_key, model_config=fc_rnnt_cfg)
        _fc_nt.add_alias(f"baseline-fastconformer-stream-rnnt-native-viterbi-{_fc_tag}")
        _fc_nt_m = CalcAlignmentMetricsFromWordBoundariesJob(
            word_boundaries_hdf=_fc_nt.out_word_boundaries_hdf,
            dataset_dir=_fc_dir,
            dataset_key=_fc_key,
            dataset_offset_factors=_fc_off,
        )
        reg(f"baseline-fastconformer-stream-rnnt-native-viterbi-{_fc_tag}-wbe.txt", _fc_nt_m.out_wbe)

    # (4d) Whisper-large-v3 cross-attn DTW (whisper-timestamped heads for large-v3):
    # the same-model attention counterpart of the large-v3 grad row, on segA and TIMIT-test.
    for _ca3_dir, _ca3_key, _ca3_off, _ca3_tag in [
        (_xa_dir, "test", _xa_off, _xa_tag),
        (dl_ds_timit.out_hub_cache_dir, "test", _DATASET_OFFSET_FACTORS["timit"], "timit-test"),
    ]:
        _ca3 = WhisperCrossAttnForcedAlignJob(
            dataset_dir=_ca3_dir, dataset_key=_ca3_key, overlay=_WHISPER_TS_OVERLAY, whisper_model="large-v3"
        )
        _ca3.add_alias(f"baseline-whisper-large-v3-crossattn-{_ca3_tag}")
        _ca3_m = CalcAlignmentMetricsFromWordBoundariesJob(
            word_boundaries_hdf=_ca3.out_hdf,
            dataset_dir=_ca3_dir,
            dataset_key=_ca3_key,
            dataset_offset_factors=_ca3_off,
        )
        reg(f"baseline-whisper-large-v3-crossattn-{_ca3_tag}-wbe.txt", _ca3_m.out_wbe)

    # (4g) CrisperWhisper (Whisper finetuned for verbatim transcription + silence-aware
    # timestamps): grad-align on their checkpoint -- does training-free grad-align already
    # give what they finetuned for? Their OFFICIAL pipeline is decoding-time -> hyp-mode
    # row via a separate wrapper job (TODO, see readme spec). Auto-head cross-attn rows
    # for their checkpoint AND whisper-base are wired in (4h) below.
    _cw_name = f"crisperwhisper-logmel-{_xa_tag}-L2_grad-pertoken-charlev-spc"
    _cw_ex = _xa_extract(
        rf.build_dict(Whisper, model_dir=dl_crisper.out_hub_cache_dir, char_level=True, char_level_sep=" "),
        _cw_name,
        "L2",
        False,
        bb=True,
    )
    _xa_align(_cw_ex, _cw_name, "en0.5-sil1.0")

    # grad-align with CrisperWhisper's OWN (retokenized verbatim) targets: subword mode
    # uses the checkpoint's native tokenizer (the special vocab w/ uh/um filler tokens,
    # standalone spaces), same targets their official pipeline aligns. This is the fair
    # "same model + SAME targets, grad vs their cross-attn DTW / vs generic auto-crossattn"
    # comparison (the charlev row above uses different, generic char targets).
    _cw_sub_name = f"crisperwhisper-logmel-{_xa_tag}-L2_grad-pertoken-subword"
    _cw_sub_ex = _xa_extract(
        rf.build_dict(Whisper, model_dir=dl_crisper.out_hub_cache_dir),
        _cw_sub_name,
        "L2",
        False,
        bb=True,
    )
    _xa_align(_cw_sub_ex, _cw_sub_name, "en0.5-sil1.0")

    # (4h) Auto-head cross-attn (generic attention method, heads selected on TIMIT val):
    # for whisper-base (contrast with the official hand-picked heads in the crossattn baseline)
    # and CrisperWhisper (their model + generic method; the official pipeline is separate).
    for _ah_dl, _ah_model in [(dl_whisper, "whisper-base"), (dl_crisper, "crisperwhisper")]:
        _ah_cfg = rf.build_dict(Whisper, model_dir=_ah_dl.out_hub_cache_dir, attn_implementation="eager")
        _ah_sel = SelectSelfAttnAlignHeadsJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir, dataset_key="val", model_config=_ah_cfg
        )
        _ah_sel.add_alias(f"selfattn/{_ah_model}-head-selection")
        reg(f"selfattn/{_ah_model}-heads.txt", _ah_sel.out_heads)
        reg(f"selfattn/{_ah_model}-heads-report.txt", _ah_sel.out_report)
        for _ah_dir, _ah_key, _ah_off, _ah_tag in [
            (_xa_dir, "test", _xa_off, _xa_tag),
            (dl_ds_timit.out_hub_cache_dir, "test", _DATASET_OFFSET_FACTORS["timit"], "timit-test"),
        ]:
            _ah_ex = ExtractSelfAttnPerTokenJob(
                dataset_dir=_ah_dir, dataset_key=_ah_key, model_config=_ah_cfg, heads=_ah_sel.out_heads
            )
            _ah_name = f"baseline-{_ah_model}-crossattn-auto-{_ah_tag}"
            _ah_ex.add_alias(f"{_ah_name}-extract")
            reg(f"{_ah_name}.hdf", _ah_ex.out_hdf)
            for _ah_ep, _ah_sc, _ah_sfx in [(0.5, 1.0, "-en0.5-sil1.0"), (0.0, 0.0, "")]:
                _ah_al = WordAlignFromPerTokenGradsJob(
                    grad_score_hdf=_ah_ex.out_hdf,
                    grad_score_key="data",
                    dataset_dir=_ah_dir,
                    dataset_key=_ah_key,
                    dataset_offset_factors=_ah_off,
                    align_opts=_xa_ao,
                    audio_energy_pow=_ah_ep,
                    blank_silence_energy_scale=_ah_sc,
                )
                _ah_al.add_alias(f"align/{_ah_name}-{_name_for_dict(_xa_ao)}{_ah_sfx}")
                reg(f"align/{_ah_name}-{_name_for_dict(_xa_ao)}{_ah_sfx}-wbe.txt", _ah_al.out_wbe)
                # FORCED (ground-truth) cross-attn: our Aligner word-topology, no-blank DTW, and the
                # faithful openai DTW (= forced Whisper-DTW / CrisperWhisper-DTW, no hyp-mode).
                if _ah_sfx == "-en0.5-sil1.0":
                    for _ah_kw, _ah_tsfx in [
                        ({"word_topology": True}, "-wordtopo"),
                        ({"dtw_no_blank": True}, "-dtw"),
                        ({"whisper_dtw": True}, "-wdtw"),
                    ]:
                        _ah_t = WordAlignFromPerTokenGradsJob(
                            grad_score_hdf=_ah_ex.out_hdf,
                            grad_score_key="data",
                            dataset_dir=_ah_dir,
                            dataset_key=_ah_key,
                            dataset_offset_factors=_ah_off,
                            align_opts=_xa_ao,
                            audio_energy_pow=_ah_ep,
                            blank_silence_energy_scale=_ah_sc,
                            **_ah_kw,
                        )
                        _ah_tnm = f"align/{_ah_name}-{_name_for_dict(_xa_ao)}{_ah_sfx}{_ah_tsfx}"
                        _ah_t.add_alias(_ah_tnm)
                        reg(f"{_ah_tnm}-wbe.txt", _ah_t.out_wbe)

    # (4f) Speech-LLM SELF-attention DTW (voxtral + canary-qwen): the attention analog
    # of whisper's cross-attn timestamps. No published alignment-head masks exist for
    # these models, so heads are auto-selected on TIMIT val (gold) and frozen for the
    # eval rows. Same DP as grad-align -> "same DP, different signal"; also the direct
    # 80 ms-attention-grid vs 10 ms-grad-grid resolution comparison on one model.
    # Tokenization axis: "subword" (model-native) AND "char" (forced char-level,
    # same as the grad rows), so the grad-vs-self-attn comparison is same-target.
    # The subword cfg/aliases are byte-identical to the original (empty char kwargs),
    # so the already-finished subword jobs reuse their hashes; char rows are new.
    # Explicit per-tokenization configs (char needs a higher model version than
    # subword for voxtral: char_level=True asserts version>=7; canary char is fine
    # at version 3). subword cfgs are byte-identical to the originals -> finished
    # subword jobs reuse their hashes; char rows are new.
    for _sa_model, _sa_cfgs in [
        (
            "voxtral",
            {
                "subword": rf.build_dict(
                    Voxtral, model_dir=dl_voxtral, forward_mode="transcription", attn_implementation="eager", version=3
                ),
                "char": rf.build_dict(
                    Voxtral,
                    model_dir=dl_voxtral,
                    forward_mode="transcription",
                    attn_implementation="eager",
                    char_level=True,
                    char_level_sep=" ",
                    version=7,
                ),
            },
        ),
        (
            "canary-qwen",
            {
                "subword": rf.build_dict(
                    CanaryQwen, model_dir=dl_canary, llm_model_dir=dl_qwen3, attn_implementation="eager", version=3
                ),
                "char": rf.build_dict(
                    CanaryQwen,
                    model_dir=dl_canary,
                    llm_model_dir=dl_qwen3,
                    attn_implementation="eager",
                    char_level=True,
                    char_level_sep=" ",
                    version=3,
                ),
            },
        ),
        (
            "phi4mm",
            {
                # Phi-4-MM is a decoder-LLM like Voxtral/Canary; its native attention aligner is
                # decoder self-attn over the audio-token block (eager attn enables output_attentions).
                "subword": _phi4mm_model_config(dl_phi4mi_dir, attn_implementation="eager"),
                "char": _phi4mm_model_config(
                    dl_phi4mi_dir, attn_implementation="eager", char_level=True, char_level_sep=" "
                ),
            },
        ),
    ]:
        for _sa_tokn, _sa_cfg in _sa_cfgs.items():
            # Keep the bare model name for subword (stable aliases); suffix char rows.
            _sa_mt = _sa_model if _sa_tokn == "subword" else f"{_sa_model}-char"
            _sa_sel = SelectSelfAttnAlignHeadsJob(
                dataset_dir=dl_ds_timit.out_hub_cache_dir,
                dataset_key="val",
                model_config=_sa_cfg,
                # char tokens exceed the 80 ms audio-token grid (S>T) -> upsample so head
                # selection can DP-align (else every head is degenerate). Bumps only the
                # char head-sel hash, cascading a fresh char extract+align; subword unchanged.
                time_upsample_when_short=(_sa_tokn == "char"),
            )
            _sa_sel.add_alias(f"selfattn/{_sa_mt}-head-selection")
            reg(f"selfattn/{_sa_mt}-heads.txt", _sa_sel.out_heads)
            reg(f"selfattn/{_sa_mt}-heads-report.txt", _sa_sel.out_report)
            for _sa_dir, _sa_key, _sa_off, _sa_tag in [
                (_xa_dir, "test", _xa_off, _xa_tag),
                (dl_ds_timit.out_hub_cache_dir, "test", _DATASET_OFFSET_FACTORS["timit"], "timit-test"),
            ]:
                _sa_ex = ExtractSelfAttnPerTokenJob(
                    dataset_dir=_sa_dir, dataset_key=_sa_key, model_config=_sa_cfg, heads=_sa_sel.out_heads
                )
                _sa_name = f"baseline-{_sa_mt}-selfattn-{_sa_tag}"
                _sa_ex.add_alias(f"{_sa_name}-extract")
                reg(f"{_sa_name}.hdf", _sa_ex.out_hdf)
                for _sa_ep, _sa_sc, _sa_sfx in [(0.5, 1.0, "-en0.5-sil1.0"), (0.0, 0.0, "")]:
                    _sa_al = WordAlignFromPerTokenGradsJob(
                        grad_score_hdf=_sa_ex.out_hdf,
                        grad_score_key="data",
                        dataset_dir=_sa_dir,
                        dataset_key=_sa_key,
                        dataset_offset_factors=_sa_off,
                        align_opts=_xa_ao,
                        audio_energy_pow=_sa_ep,
                        blank_silence_energy_scale=_sa_sc,
                    )
                    _sa_al.add_alias(f"align/{_sa_name}-{_name_for_dict(_xa_ao)}{_sa_sfx}")
                    reg(f"align/{_sa_name}-{_name_for_dict(_xa_ao)}{_sa_sfx}-wbe.txt", _sa_al.out_wbe)
                    # word-topology + DTW (no-blank) twins of the self-attn headline align.
                    if _sa_sfx == "-en0.5-sil1.0":
                        for _sa_kw, _sa_tsfx in [
                            ({"word_topology": True}, "-wordtopo"),
                            ({"dtw_no_blank": True}, "-dtw"),
                            ({"whisper_dtw": True}, "-wdtw"),
                        ]:
                            _sa_t = WordAlignFromPerTokenGradsJob(
                                grad_score_hdf=_sa_ex.out_hdf,
                                grad_score_key="data",
                                dataset_dir=_sa_dir,
                                dataset_key=_sa_key,
                                dataset_offset_factors=_sa_off,
                                align_opts=_xa_ao,
                                audio_energy_pow=_sa_ep,
                                blank_silence_energy_scale=_sa_sc,
                                **_sa_kw,
                            )
                            _sa_tnm = f"align/{_sa_name}-{_name_for_dict(_xa_ao)}{_sa_sfx}{_sa_tsfx}"
                            _sa_t.add_alias(_sa_tnm)
                            reg(f"{_sa_tnm}-wbe.txt", _sa_t.out_wbe)

    # (5) grad-score x energy/silence ablation on the headline whisper-char extract (shared; free aligns).
    _xa_whc = _xa_extract(whisper_char_cfg, None, "L2", False)
    _xa_whc_name = f"whisper-base-logmel-{_xa_tag}-L2_grad-pertoken-charlev-spc"
    _xa_align(_xa_whc, _xa_whc_name, "en0.5", sil=0.0)
    _xa_align(_xa_whc, _xa_whc_name, "en0.5-sil2.0", sil=2.0)
    _xa_align(_xa_whc, _xa_whc_name, "en0.5-zsk1.0", zsk=1.0)
    # space-as-silence (CrisperWhisper-style): align the separator tokens as the silence anchors
    # (no DP blank); word boundaries from char rows only. CrisperWhisper (standalone space IS trained
    # -> should work) vs whisper-base (untrained -> contrast). Buckeye-segA.
    for _ws_dl, _ws_tag in [(dl_crisper, "crisperwhisper"), (dl_whisper, "whisper-base")]:
        _ws_cfg = rf.build_dict(Whisper, model_dir=_ws_dl.out_hub_cache_dir, char_level=True, char_level_sep=" ")
        _ws_ex = ExtractInGradsPerTokenWithSepGradsJob(
            dataset_dir=_xa_dir,
            dataset_key="test",
            model_config=_ws_cfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
        )
        _ws_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _ws_ex.rqmt = {**_ws_ex.rqmt, "time": 24}
        _ws_exname = f"{_ws_tag}-logmel-{_xa_tag}-L2_grad-pertoken-charlev-spc-with-sep"
        _ws_ex.add_alias(_ws_exname)
        reg(f"{_ws_exname}.hdf", _ws_ex.out_hdf)
        for _ws_ao in _ALIGN_OPTS_GRID:
            _ws_al = WordAlignFromPerTokenWithSepGradsJob(
                grad_score_hdf=_ws_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=_xa_dir,
                dataset_key="test",
                dataset_offset_factors=_xa_off,
                align_opts=_ws_ao,
            )
            _ws_alnm = f"align/{_ws_exname}-{_name_for_dict(_ws_ao)}"
            _ws_al.add_alias(_ws_alnm)
            reg(f"{_ws_alnm}-wbe.txt", _ws_al.out_wbe)
    # apply_log on/off ablation (small): grad-CTC (wav2vec2 prefix_fwd) + grad-AED (whisper-char),
    # x silence-on (en0.5-sil1.0) and silence-off (dtw no-blank). apply_log=True rows exist already;
    # add the apply_log=False rows. (apply_log=False + sil-off == the openai-DTW config.)
    _ab_w2v = _xa_extract(
        rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", per_token_score="prefix_fwd"),
        f"wav2vec2ctc-fproj_out-prefixfwd-{_xa_tag}-L2_grad-pertoken",
        "L2",
        False,
        bb=True,
    )
    for _ab_ex, _ab_name in [
        (_ab_w2v, f"wav2vec2ctc-fproj_out-prefixfwd-{_xa_tag}-L2_grad-pertoken"),
        (_xa_whc, _xa_whc_name),
    ]:
        _xa_align(_ab_ex, _ab_name, "en0.5-sil1.0", apply_log=False, _twin=False)
        _xa_align(_ab_ex, _ab_name, "en0.5-sil1.0-dtw", apply_log=False, dtw_no_blank=True, _twin=False)
    for _xa_mgi, _xa_attr, _xa_ga in [(True, "L2", "L2_e_grad"), (False, "L1", "L1_grad")]:
        _xa_av_name = f"whisper-base-logmel-{_xa_tag}-{_xa_ga}-pertoken-charlev-spc"
        _xa_av = _xa_extract(
            rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir, char_level=True, char_level_sep=" "),
            _xa_av_name,
            _xa_attr,
            _xa_mgi,
        )
        _xa_align(_xa_av, _xa_av_name, "en0.5-sil1.0")

    # === Cross-model grad-SIGNAL ablation (Table A): grad-score reduction across all five families,
    # plus attribution method on the two fast models (multi-pass IG/EG is prohibitive on the
    # transducer / streaming / speech-LLM). Buckeye-segA, fixed word-topology + en0.5-sil1.0 so the
    # signal choice is not confounded by the DP opts (those are ablated separately). ===
    _abl_ao = {"apply_softmax_over_time": True, "blank_score": -5}

    def _abl_align(ex, ename):
        al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=ex.out_hdf,
            grad_score_key="data",
            dataset_dir=_xa_dir,
            dataset_key="test",
            dataset_offset_factors=_xa_off,
            align_opts=_abl_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
            word_topology=True,
        )
        nm = f"align/{ename}-{_name_for_dict(_abl_ao)}-en0.5-sil1.0-wordtopo"
        al.add_alias(nm)
        reg(f"{nm}-wbe.txt", al.out_wbe)

    # (model_config, name_prefix, batched_backward, run_attribution)
    _ABL_MODELS = [
        (
            rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", per_token_score="prefix_fwd"),
            "wav2vec2ctc-fproj_out-prefixfwd",
            True,
            True,
        ),
        (whisper_char_cfg, "whisper-base-logmel-charlev-spc", False, True),
        (pk_cfg, "parakeet-rnnt-1.1b-logmel", True, False),
        (rnnt_px_cfg, "emformer-rnnt-prefix-logmel", False, False),
        (voxtral_charlev_logmel_cfg, "voxtral-charlevlogmel", False, False),
    ]
    for _ac, _apre, _abb, _ado_attr in _ABL_MODELS:
        # grad-score axis: L0.5 / L1 / L2 / L2_e (L2_e = L2 norm on grad*input).
        for _ar, _amgi, _arn in [
            ("L0.5", False, "L0.5_grad"),
            ("L1", False, "L1_grad"),
            ("L2", False, "L2_grad"),
            ("L2", True, "L2_e_grad"),
        ]:
            _aen = f"{_apre}-abl-{_xa_tag}-{_arn}-pertoken"
            _abl_align(_xa_extract(_ac, _aen, _ar, _amgi, bb=_abb), _aen)
        # attribution axis (fast models only): SmoothGrad / VarGrad / IG / EG, on L2.
        if _ado_attr:
            for _akw, _asfx in [
                ({"noise_std": 0.01, "noise_n_samples": 8}, "smoothgrad-std0.01-n8"),
                ({"noise_std": 0.02, "noise_n_samples": 16, "vargrad": True}, "vargrad-std0.02-n16"),
                ({"ig_steps": 16}, "integrated-grad-ig16"),
                ({"eg_steps": 16, "eg_baseline_std": 0.05}, "expected-grad-eg16-bstd0.05"),
            ]:
                _aen = f"{_apre}-abl-{_xa_tag}-L2-{_asfx}-pertoken"
                _aex = ExtractInGradsPerTokenJob(
                    dataset_dir=_xa_dir,
                    dataset_key="test",
                    model_config=_ac,
                    mult_grad_by_inputs=False,
                    attr_reduction="L2",
                    **_akw,
                )
                _aex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                _aex.rqmt = {**_aex.rqmt, "time": 24}
                _aex.add_alias(_aen)
                reg(f"{_aen}.hdf", _aex.out_hdf)
                _abl_align(_aex, _aen)

    # (6) Complete the char-vs-subword grid for the speech LLMs: grad SUBWORD extracts (the
    # char-level counterparts are already wired in the headline-A block). Each model uses its own
    # best grad attribution (voxtral/canary L1, phi4 L2_e). These are heavy models -> 24h walltime.
    for _xa_s_cfg, _xa_s_name, _xa_s_attr, _xa_s_mgi in [
        (
            rf.build_dict(Voxtral, model_dir=dl_voxtral, forward_mode="transcription", grad_wrt="log_mel", version=7),
            f"voxtral-logmel-{_xa_tag}-L1_grad-pertoken-subword",
            "L1",
            False,
        ),
        (
            phi4mm_recog_cfg,
            f"phi4mm-{_xa_tag}-L2_e_grad-pertoken-subword",
            "L2",
            True,
        ),
        (
            rf.build_dict(
                CanaryQwen,
                model_dir=dl_canary,
                llm_model_dir=dl_qwen3,
                grad_wrt="log_mel",
                ensure_audio_long_enough=True,
                audio_time_stretch=1.5,
                version=5,
            ),
            f"canary-qwen-logmel-st15-{_xa_tag}-L1_grad-pertoken-subword",
            "L1",
            False,
        ),
        # OWSM-CTC: a general graphemic CTC (BPE subword units, ~12.5 Hz emission).
        (owsm_ctc_cfg, f"owsm-ctc-v4-1b-{_xa_tag}-L2_grad-pertoken", "L2", False),
        # Nvidia CTC (parakeet-ctc-1.1b): the general graphemic CTC representative (plain FastConformer).
        (parakeet_ctc_cfg, f"parakeet-ctc-1.1b-{_xa_tag}-L2_grad-pertoken", "L2", False),
        # prefix_diff (AED-consistent) score -- the fix for the conformer/branchformer CTCs.
        (parakeet_ctc_prefdiff_cfg, f"parakeet-ctc-1.1b-prefdiff-{_xa_tag}-L2_grad-pertoken", "L2", False),
        (owsm_ctc_prefdiff_cfg, f"owsm-ctc-v4-1b-prefdiff-{_xa_tag}-L2_grad-pertoken", "L2", False),
        # inb_true (robust universal) on the same subword CTCs, to compare vs prefix_diff.
        (parakeet_ctc_inbtrue_cfg, f"parakeet-ctc-1.1b-inbtrue-{_xa_tag}-L2_grad-pertoken", "L2", False),
        (owsm_ctc_inbtrue_cfg, f"owsm-ctc-v4-1b-inbtrue-{_xa_tag}-L2_grad-pertoken", "L2", False),
        # prefix_fwd (the real CTC prefix score) on all four CTCs -- the uniform-winner candidate.
        (parakeet_ctc_prefixfwd_cfg, f"parakeet-ctc-1.1b-prefixfwd-{_xa_tag}-L2_grad-pertoken", "L2", False),
        (owsm_ctc_prefixfwd_cfg, f"owsm-ctc-v4-1b-prefixfwd-{_xa_tag}-L2_grad-pertoken", "L2", False),
        # Emformer RNN-T: the STREAMING transducer (prefix score). Re-enabled 2026-06-16 with the
        # VECTORIZED RNN-T wavefront (~3x bwd, exact) -- the scalar T*U version was the launch-bound bottleneck.
        (rnnt_px_cfg, f"emformer-rnnt-prefix-logmel-{_xa_tag}-L2_grad-pertoken", "L2", False),
        (
            rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", per_token_score="prefix_fwd"),
            f"wav2vec2ctc-fproj_out-prefixfwd-{_xa_tag}-L2_grad-pertoken",
            "L2",
            False,
        ),
        # Normalized-gradient (ICLR 2024) applied at EXTRACTION (not training): reweight the
        # per-class emission gradient by clamped inverse occupancy to de-emphasize blank's
        # diffuse gradient (probe on the documented Wav2Vec2-CTC front-loading; same MMS_FA
        # forced-align reference). C05_11 = the paper's clamp (0.5, 1.1).
        (
            rf.build_dict(
                Wav2Vec2Ctc,
                grad_wrt="feat_proj_out",
                per_token_score="prefix_fwd",
                normed_grad_opts={"clamp_min": 0.5, "clamp_max": 1.1, "prior_exp": 1.0},
            ),
            f"wav2vec2ctc-fproj_out-prefixfwd-normedgradC05_11-{_xa_tag}-L2_grad-pertoken",
            "L2",
            False,
        ),
        (
            rf.build_dict(
                Wav2Vec2PhonemeCtc,
                model_dir=dl_w2v_phoneme.out_hub_cache_dir,
                g2p_word_targets=True,
                per_token_score="prefix_fwd",
            ),
            f"phoneme-vitouphy-prefixfwd-{_xa_tag}-L2_grad-pertoken-g2pword",
            "L2",
            False,
        ),
        # prefix_diff on the already-good CTCs (wav2vec2 / phoneme), to compare vs their inb_true.
        (
            rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", per_token_score="prefix_diff"),
            f"wav2vec2ctc-fproj_out-prefdiff-{_xa_tag}-L2_grad-pertoken",
            "L2",
            False,
        ),
        (
            rf.build_dict(
                Wav2Vec2PhonemeCtc,
                model_dir=dl_w2v_phoneme.out_hub_cache_dir,
                g2p_word_targets=True,
                per_token_score="prefix_diff",
            ),
            f"phoneme-vitouphy-prefdiff-{_xa_tag}-L2_grad-pertoken-g2pword",
            "L2",
            False,
        ),
        # OWSM-CTC per inter-CTC layer (side table): grad-align WBE from blocks 6/12/15/21 (27=above).
        *[
            (_ocfg, f"owsm-ctc-v4-1b-lyr{_ol}-{_xa_tag}-L2_grad-pertoken", "L2", False)
            for _ol, _ocfg in owsm_ctc_cfg_by_layer.items()
        ],
    ]:
        # batched_backward for the prefix_fwd CTC extracts (vmap per-token VJP, ~tokens-per-word
        # faster, mathematically identical). bb=False for the rest keeps their hash unchanged.
        _xa_s_ex = _xa_extract(_xa_s_cfg, _xa_s_name, _xa_s_attr, _xa_s_mgi, time=24, bb="prefix" in _xa_s_name)
        _xa_align(_xa_s_ex, _xa_s_name, "en0.5-sil1.0")
    # Word-aware DP topology probe (blank only between words, not within): the documented Wav2Vec2-CTC
    # front-loading case + an AED model (whisper-char). Same DP, so it also covers the self-attn path.
    _wt_w2v_ex = _xa_extract(
        rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", per_token_score="prefix_fwd"),
        f"wav2vec2ctc-fproj_out-prefixfwd-{_xa_tag}-L2_grad-pertoken",
        "L2",
        False,
        bb=True,
    )
    _xa_align(
        _wt_w2v_ex,
        f"wav2vec2ctc-fproj_out-prefixfwd-{_xa_tag}-L2_grad-pertoken",
        "en0.5-sil1.0-wordtopo",
        word_topology=True,
    )
    _xa_align(_xa_whc, _xa_whc_name, "en0.5-sil1.0-wordtopo", word_topology=True)
    _xa_align(_xa_whc, _xa_whc_name, "en0.5-sil1.0-dtw", dtw_no_blank=True, _twin=False)
    _xa_align(_xa_whc, _xa_whc_name, "en0.5-sil1.0-wdtw", whisper_dtw=True, _twin=False)
    # Nvidia CTC posteriors baseline on segA: torchaudio CTC forced-align on parakeet-ctc's own emission.
    _xa_pc_fa = ParakeetCtcForcedAlignJob(
        dataset_dir=_xa_dir,
        dataset_key="test",
        model_dir=dl_parakeet_ctc.out_hub_cache_dir,
        overlay_path=_NEMO_OVERLAY,
        dataset_offset_factors=_xa_off,
    )
    _xa_pc_fa.add_alias(f"baseline-parakeet-ctc-1.1b-{_xa_tag}")
    reg(f"baseline-parakeet-ctc-1.1b-{_xa_tag}-wbe.txt", _xa_pc_fa.out_word_wbe)
    # OWSM-CTC posteriors baseline per inter-CTC emit block on segA.
    for _ol in [None, 6, 12, 15, 21]:
        _otag = "" if _ol is None else f"lyr{_ol}-"
        _ofa = OwsmCtcForcedAlignJob(
            dataset_dir=_xa_dir,
            dataset_key="test",
            model_dir=dl_owsm_ctc.out_hub_cache_dir,
            dataset_offset_factors=_xa_off,
            layer=_ol,
        )
        _ofa.add_alias(f"baseline-owsm-ctc-v4-1b-{_otag}{_xa_tag}")
        reg(f"baseline-owsm-ctc-v4-1b-{_otag}{_xa_tag}-wbe.txt", _ofa.out_word_wbe)

    # (7) grad-score x energy/silence x align-opts ablation on a speech LLM (Voxtral). The char-level
    # L1 extract is the headline; we add grad-score variants (L2, L2_e), re-align the headline extract
    # across energy/silence settings, and sweep the full align-opts grid -- mirroring the whisper-char
    # ablation in (5) so the LLM gets the same ablation coverage.
    _xa_vx = _xa_extract(voxtral_charlev_logmel_cfg, None, "L1", False, time=24)
    _xa_vx_name = f"voxtral-charlevlogmel-{_xa_tag}-L1_grad-pertoken"
    _xa_align(_xa_vx, _xa_vx_name, "en0.5", sil=0.0)
    _xa_align(_xa_vx, _xa_vx_name, "en0.5-sil2.0", sil=2.0)
    _xa_align(_xa_vx, _xa_vx_name, "en0.5-zsk1.0", zsk=1.0)
    for _xa_v_ao in _ALIGN_OPTS_GRID:
        _xa_v_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_xa_vx.out_hdf,
            grad_score_key="data",
            dataset_dir=_xa_dir,
            dataset_key="test",
            dataset_offset_factors=_xa_off,
            align_opts=_xa_v_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
        )
        _xa_v_alnm = f"align/{_xa_vx_name}-{_name_for_dict(_xa_v_ao)}-en0.5-sil1.0"
        _xa_v_al.add_alias(_xa_v_alnm)
        reg(f"{_xa_v_alnm}-wbe.txt", _xa_v_al.out_wbe)
    for _xa_v_mgi, _xa_v_attr, _xa_v_ga in [(True, "L2", "L2_e_grad"), (False, "L2", "L2_grad")]:
        _xa_v_name = f"voxtral-charlevlogmel-{_xa_tag}-{_xa_v_ga}-pertoken"
        _xa_v_ex = _xa_extract(voxtral_charlev_logmel_cfg, _xa_v_name, _xa_v_attr, _xa_v_mgi, time=24)
        _xa_align(_xa_v_ex, _xa_v_name, "en0.5-sil1.0")

    # === FB-DP confidence: re-align headline extracts with with_confidence=True -> per-word
    # posterior-occupancy confidence + conf-vs-WBE correlation (cheap CPU; extracts shared). ===
    for _cf_ex, _cf_name in [
        (_xa_whc, f"whisper-base-logmel-{_xa_tag}-L2_grad-pertoken-charlev-spc"),
        (_xa_vx, f"voxtral-charlevlogmel-{_xa_tag}-L1_grad-pertoken"),
    ]:
        _cf_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_cf_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=_xa_dir,
            dataset_key="test",
            dataset_offset_factors=_xa_off,
            align_opts=_xa_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
            with_confidence=True,
            confidence_version=2,
        )
        _cf_al.add_alias(f"conf/{_cf_name}-v2")
        reg(f"conf/{_cf_name}-v2-conf-corr.txt", _cf_al.out_conf_corr)

    # === Speed: batched per-token backward (autograd.grad is_grads_batched=True). Exact same per-token
    # math as the K sequential backwards -- one vmapped traversal of the shared graph, no logic change.
    # Validate numerical equivalence (and surface any vmap-incompatible backward kernels, e.g. on the
    # transducer / flash-attn LLMs) on a tiny subset before enabling on the slow models. The bb0 extract
    # shares the existing (unbatched) hash; bb1 is a distinct job, so we can diff their HDFs. ===
    _sb_ds = BuildBuckeyeFineDatasetJob(
        raw_dir=dl_ds_buckeye_fine.out_hub_cache_dir,
        resegment_gap_s=1.0,
        split_up_to_max_seq_len_s=18.0,
        min_words=2,
        skip_misaligned_wavs=True,
        subsample_target_h=0.25,
        subsample_seed=42,
    )
    _sb_dir = _sb_ds.out_hub_cache_dir
    for _sb_cfg, _sb_name, _sb_attr, _sb_mgi in [
        (
            rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir, char_level=True, char_level_sep=" "),
            "whisper-base-charlev",
            "L2",
            False,
        ),
        (pk_cfg, "parakeet-rnnt", "L2", False),
        (voxtral_charlev_logmel_cfg, "voxtral-charlevlogmel", "L1", False),
        (pt_csp_cfg, "phi4mm-charlev-spc", "L2", True),
        (rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out"), "wav2vec2ctc-fproj_out", "L2", False),
        (canary_charlev_logmel_st15_cfg, "canary-qwen-charlev-spc-logmel-st15", "L1", False),
    ]:
        for _sb_bb in [False, True]:
            if _sb_bb and _sb_name == "phi4mm-charlev-spc":
                # flash-attn backward has no vmap batching rule (verified crash): sequential only.
                continue
            _sb_ex = ExtractInGradsPerTokenJob(
                dataset_dir=_sb_dir,
                dataset_key="test",
                model_config=_sb_cfg,
                mult_grad_by_inputs=_sb_mgi,
                attr_reduction=_sb_attr,
                batched_backward=_sb_bb,
            )
            _sb_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            _sb_ex.rqmt = {**_sb_ex.rqmt, "time": 4}
            _sb_alias = f"speedcmp/{_sb_name}-bb{int(_sb_bb)}"
            _sb_ex.add_alias(_sb_alias)
            reg(f"{_sb_alias}.hdf", _sb_ex.out_hdf)
            # WBE A/B: same fixed align_opts on batched vs unbatched grads -> verify identical WBE.
            _sb_al = WordAlignFromPerTokenGradsJob(
                grad_score_hdf=_sb_ex.out_hdf,
                grad_score_key="data",
                dataset_dir=_sb_dir,
                dataset_key="test",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
                align_opts=_ALIGN_OPTS_GRID[0],
            )
            _sb_al.add_alias(f"{_sb_alias}-wbe")
            reg(f"{_sb_alias}-wbe.txt", _sb_al.out_wbe)

    # AMP A/B: run fp32 models (AED / transducer / CTC) with bf16 activations (weights stay f32)
    # via amp_dtype, on top of batched_backward=True (the standard). Compare WBE to the f32
    # reference (-bb1-wbe) -> WBE-close gate; job runtime -> the 'actually faster' gate.
    for _amp_cfg, _amp_name, _amp_attr, _amp_mgi in [
        (
            rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir, char_level=True, char_level_sep=" "),
            "whisper-base-charlev",
            "L2",
            False,
        ),
        (pk_cfg, "parakeet-rnnt", "L2", False),
        (rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out"), "wav2vec2ctc-fproj_out", "L2", False),
    ]:
        _amp_ex = ExtractInGradsPerTokenJob(
            dataset_dir=_sb_dir,
            dataset_key="test",
            model_config=_amp_cfg,
            mult_grad_by_inputs=_amp_mgi,
            attr_reduction=_amp_attr,
            batched_backward=True,
            amp_dtype="bfloat16",
        )
        _amp_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _amp_ex.rqmt = {**_amp_ex.rqmt, "time": 4}
        _amp_alias = f"speedcmp/{_amp_name}-ampbf16"
        _amp_ex.add_alias(_amp_alias)
        reg(f"{_amp_alias}.hdf", _amp_ex.out_hdf)
        _amp_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_amp_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=_sb_dir,
            dataset_key="test",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_ALIGN_OPTS_GRID[0],
        )
        _amp_al.add_alias(f"{_amp_alias}-wbe")
        reg(f"{_amp_alias}-wbe.txt", _amp_al.out_wbe)

    # SDPA-f32 probe: amp-bf16 but force the attention core (Q*K, softmax, attn*V) to f32 -> does
    # the +5.3ms recover while keeping bf16 speed? (parakeet-rnnt conformer self-attn = suspect.)
    _amp_af_ex = ExtractInGradsPerTokenJob(
        dataset_dir=_sb_dir,
        dataset_key="test",
        model_config=pk_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
        batched_backward=True,
        amp_dtype="bfloat16",
        amp_attn_fp32=True,
    )
    _amp_af_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _amp_af_ex.rqmt = {**_amp_af_ex.rqmt, "time": 4}
    _amp_af_alias = "speedcmp/parakeet-rnnt-ampbf16-attnf32"
    _amp_af_ex.add_alias(_amp_af_alias)
    reg(f"{_amp_af_alias}.hdf", _amp_af_ex.out_hdf)
    _amp_af_al = WordAlignFromPerTokenGradsJob(
        grad_score_hdf=_amp_af_ex.out_hdf,
        grad_score_key="data",
        dataset_dir=_sb_dir,
        dataset_key="test",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
        align_opts=_ALIGN_OPTS_GRID[0],
    )
    _amp_af_al.add_alias(f"{_amp_af_alias}-wbe")
    reg(f"{_amp_af_alias}-wbe.txt", _amp_af_al.out_wbe)

    # amp-bf16 + attention MODULES forced to f32 (catches NeMo conformer's explicit-matmul attn).
    _amp_am_ex = ExtractInGradsPerTokenJob(
        dataset_dir=_sb_dir,
        dataset_key="test",
        model_config=pk_cfg,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
        batched_backward=True,
        amp_dtype="bfloat16",
        amp_attn_fp32_module=True,
    )
    _amp_am_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _amp_am_ex.rqmt = {**_amp_am_ex.rqmt, "time": 4}
    _amp_am_alias = "speedcmp/parakeet-rnnt-ampbf16-attnmodf32"
    _amp_am_ex.add_alias(_amp_am_alias)
    reg(f"{_amp_am_alias}.hdf", _amp_am_ex.out_hdf)
    _amp_am_al = WordAlignFromPerTokenGradsJob(
        grad_score_hdf=_amp_am_ex.out_hdf,
        grad_score_key="data",
        dataset_dir=_sb_dir,
        dataset_key="test",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
        align_opts=_ALIGN_OPTS_GRID[0],
    )
    _amp_am_al.add_alias(f"{_amp_am_alias}-wbe")
    reg(f"{_amp_am_alias}-wbe.txt", _amp_am_al.out_wbe)

    # Emformer (torchaudio) amp speed/WBE check: f32 ref vs amp-bf16 + attn-module-f32. The real
    # target -- Buckeye-Emformer grad is ~50h single-seq; amp ~6x would finish before the deadline.
    for _em_amp, _em_sfx in [({}, "f32"), ({"amp_dtype": "bfloat16", "amp_attn_fp32_module": True}, "ampbf16amf32")]:
        _em_ex = ExtractInGradsPerTokenJob(
            dataset_dir=_sb_dir,
            dataset_key="test",
            model_config=rnnt_px_cfg,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
            batched_backward=True,
            **_em_amp,
        )
        _em_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _em_ex.rqmt = {**_em_ex.rqmt, "time": 6}
        _em_alias = f"speedcmp/emformer-rnnt-{_em_sfx}"
        _em_ex.add_alias(_em_alias)
        reg(f"{_em_alias}.hdf", _em_ex.out_hdf)
        _em_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_em_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=_sb_dir,
            dataset_key="test",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_ALIGN_OPTS_GRID[0],
        )
        _em_al.add_alias(f"{_em_alias}-wbe")
        reg(f"{_em_alias}-wbe.txt", _em_al.out_wbe)

    # Track 2 -- single-vs-batched forward equivalence + leak localization: which leaf submodule
    # first diverges under B>1 right-padding. wav2vec2 (conv feature extractor + GroupNorm over the
    # time axis) is the canonical case and already supports B>1 forward.
    _be_probe = BatchForwardEquivalenceProbeJob(
        model_configs={"wav2vec2ctc-fproj_out": rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out")},
        dataset_dir=_sb_dir,
        dataset_key="test",
        num_seqs=10,
    )
    _be_probe.add_alias("speedcmp/batch-equiv-probe-wav2vec2")
    reg("speedcmp/batch-equiv-probe-wav2vec2.txt", _be_probe.out_report)

    # Phi4 eager-attention equality probe: the checkpoint defaults to flash-attn-2
    # (no output_attentions, no vmap); eager would unblock self-attn alignment and
    # batched backwards. Must reproduce the FA2-produced phi4-bb0 reference HDF.
    _p4e_ex = ExtractInGradsPerTokenJob(
        dataset_dir=_sb_dir,
        dataset_key="test",
        model_config=_phi4mm_model_config(
            dl_phi4mi_dir, char_level=True, char_level_sep=" ", attn_implementation="eager"
        ),
        mult_grad_by_inputs=True,
        attr_reduction="L2",
    )
    _p4e_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _p4e_ex.rqmt = {**_p4e_ex.rqmt, "time": 4}
    _p4e_ex.add_alias("speedcmp/phi4mm-charlev-spc-eager-bb0")
    reg("speedcmp/phi4mm-charlev-spc-eager-bb0.hdf", _p4e_ex.out_hdf)

    # Seq-level batching (B>1 per forward; wav2vec2 adapter only so far):
    # sb4 = seq_batch_size=4 with sequential backwards, sb4-bb1 = combined with the vmapped backward.
    # Both must reproduce the wav2vec2 bb0 reference HDF within fp noise.
    for _sb4_bb, _sb4_sfx in [(False, "sb4"), (True, "sb4-bb1")]:
        _sb4_ex = ExtractInGradsPerTokenJob(
            dataset_dir=_sb_dir,
            dataset_key="test",
            model_config=rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out"),
            mult_grad_by_inputs=False,
            attr_reduction="L2",
            batched_backward=_sb4_bb,
            seq_batch_size=4,
        )
        _sb4_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _sb4_ex.rqmt = {**_sb4_ex.rqmt, "time": 4}
        _sb4_alias = f"speedcmp/wav2vec2ctc-fproj_out-{_sb4_sfx}"
        _sb4_ex.add_alias(_sb4_alias)
        reg(f"{_sb4_alias}.hdf", _sb4_ex.out_hdf)
        _sb4_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_sb4_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=_sb_dir,
            dataset_key="test",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["timit"],
            align_opts=_ALIGN_OPTS_GRID[0],
        )
        _sb4_al.add_alias(f"{_sb4_alias}-wbe")
        reg(f"{_sb4_alias}-wbe.txt", _sb4_al.out_wbe)

    # === Hypothesis-mode alignment on Buckeye variant A: recog -> swap the hyps in as the dataset
    # transcript -> align with each model's best setting -> identity-gated F1@collar + matched WBE
    # vs the reference (standard 1:1 WBE is undefined when hyp != ref). The align runs with
    # with_ref_metrics=False (hyp word_detail carries no times). On the whisper hyps additionally
    # the method grid (cross-attn DTW, MMS_FA) for a same-hyps method comparison. ===
    def _hy_metrics(name, boundaries_hdf, hyp_dir):
        m = CalcHypAlignMetricsJob(
            word_boundaries_hdf=boundaries_hdf,
            hyp_dataset_dir=hyp_dir,
            ref_dataset_dir=_xa_dir,
            dataset_key="test",
            dataset_offset_factors=_xa_off,
        )
        m.add_alias(f"hyp-align/{name}")
        reg(f"hyp-align/{name}-metrics.txt", m.out_metrics)
        reg(f"hyp-align/{name}-f1-100ms.txt", m.out_f1_100)

    for _hy_name, _hy_recog_cfg, _hy_ex_cfg, _hy_attr, _hy_mgi, _hy_sil, _hy_bb in [
        (
            "whisper-base-charlev",
            rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir),
            rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir, char_level=True, char_level_sep=" "),
            "L2",
            False,
            1.0,
            True,
        ),
        (
            "whisper-large-v3-charlev",
            rf.build_dict(Whisper, model_dir=dl_whisper_l3.out_hub_cache_dir),
            rf.build_dict(Whisper, model_dir=dl_whisper_l3.out_hub_cache_dir, char_level=True, char_level_sep=" "),
            "L2",
            False,
            1.0,
            True,
        ),
        (
            "voxtral-charlevlogmel",
            rf.build_dict(Voxtral, model_dir=dl_voxtral, recog_mode="transcription", version=3),
            voxtral_charlev_logmel_cfg,
            "L1",
            False,
            1.0,
            False,
        ),
        ("phi4mm-charlev-spc", phi4mm_recog_cfg, pt_csp_cfg, "L2", True, 1.0, False),
        ("canary-qwen-charlev-spc-logmel-st15", canary_cfg, canary_charlev_logmel_st15_cfg, "L1", False, 1.0, False),
        # Transducers: best align is sil0.0 (TIMIT + segA); parakeet-rnnt needs the batched backward.
        ("parakeet-rnnt-1.1b-logmel", pk_cfg, pk_cfg, "L2", False, 0.0, True),
        ("parakeet-tdt-0.6b-v2-logmel", tdt_cfg, tdt_grad_cfg, "L2", False, 0.0, False),
        # CTC + streaming models (added 2026-06-17) to fill the per-model-merged hyp cells:
        # they recognise words, so hyp-mode is real (not structurally n/a).
        # Emformer is omitted -- its model class has no recog() yet.
        ("parakeet-ctc-1.1b", parakeet_ctc_cfg, parakeet_ctc_prefixfwd_cfg, "L2", False, 1.0, False),
        ("owsm-ctc-v4-1b", owsm_ctc_cfg, owsm_ctc_prefixfwd_cfg, "L2", False, 1.0, False),
        ("fastconformer-stream-ctc", fc_ctc_cfg, fc_ctc_cfg, "L2", False, 1.0, False),
        ("fastconformer-stream-rnnt", fc_rnnt_cfg, fc_rnnt_cfg, "L2", False, 1.0, False),
    ]:
        _hy_recog = RecogFromModelJob(
            dataset_dir=_xa_dir,
            dataset_key="test",
            model_config=_hy_recog_cfg,
            max_new_tokens=256,
            # Cap generation length vs audio duration to kill hallucination loops
            # (e.g. whisper's "i am sorry" x64) that overflow char-level downstream.
            # Opt-in per model (hash-excluded at None) so only enabled models re-run.
            # Enabled on whisper (the one that looped); ~12 subword-tokens/s of audio
            # never truncates real speech (~7 words/s).
            max_tokens_per_sec=12.0 if _hy_name.startswith("whisper") else None,
        )
        _hy_recog.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _hy_recog.add_alias(f"hyp-align/{_hy_name}-{_xa_tag}-recog")
        reg(f"hyp-align/{_hy_name}-{_xa_tag}-recog.hyps.txt.gz", _hy_recog.out_hyps_txt)
        _hy_ds = BuildDatasetWithHypTranscriptsJob(
            dataset_dir=_xa_dir,
            dataset_key="test",
            hyps_txt=text_dict_normalize_file(_hy_recog.out_hyps_txt),
        )
        _hy_dir = _hy_ds.out_hub_cache_dir
        _hy_ex = ExtractInGradsPerTokenJob(
            dataset_dir=_hy_dir,
            dataset_key="test",
            model_config=_hy_ex_cfg,
            mult_grad_by_inputs=_hy_mgi,
            attr_reduction=_hy_attr,
            batched_backward=_hy_bb,
        )
        _hy_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _hy_ex.rqmt = {**_hy_ex.rqmt, "time": 24}
        _hy_ex.add_alias(f"hyp-align/{_hy_name}-{_xa_tag}-extract")
        _hy_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_hy_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=_hy_dir,
            dataset_key="test",
            dataset_offset_factors=_xa_off,
            align_opts=_xa_ao,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=_hy_sil,
            with_ref_metrics=False,
        )
        _hy_al.add_alias(f"hyp-align/{_hy_name}-{_xa_tag}-align")
        _hy_metrics(f"{_hy_name}-{_xa_tag}-grad", _hy_al.out_word_boundaries_hdf, _hy_dir)
        if _hy_name == "whisper-base-charlev":
            _hy_wca = WhisperCrossAttnForcedAlignJob(
                dataset_dir=_hy_dir, dataset_key="test", overlay=_WHISPER_TS_OVERLAY
            )
            _hy_wca.add_alias(f"hyp-align/whisper-crossattn-{_xa_tag}-align")
            _hy_metrics(f"whisper-crossattn-{_xa_tag}", _hy_wca.out_hdf, _hy_dir)
            # NOTE: MMS_FA (= wav2vec2 + CTC forced-align) and MFA used to force-align WHISPER's hyp
            # here. That was inconsistent: every method should align its OWN model's recognition.
            # MMS_FA now aligns wav2vec2's own CTC hyp (wired below); MFA stays the forced-mode
            # ceiling only (its tool exposes no recognizer).

    # CrisperWhisper OFFICIAL pipeline row (hyp-mode): their decode-time word timestamps
    # (verbatim retokenized vocab + heads finetuned with attention loss + cross-attn DTW
    # + pause splitting, capped 160 ms). Needs their transformers fork. Their heads were
    # trained on timestamped TIMIT, so ONLY the Buckeye row is clean -- no TIMIT row here.
    # Contrast rows on the same checkpoint: grad-align (4g) and auto-head cross-attn (4h).
    _cw_fork = CloneGitRepositoryJob(
        url="https://github.com/nyrahealth/transformers",
        branch="crisper_whisper",
        checkout_folder_name="transformers",
    )
    _cw_repo = CloneGitRepositoryJob(
        url="https://github.com/nyrahealth/CrisperWhisper",
        checkout_folder_name="CrisperWhisper",
    )
    _cw_off = CrisperWhisperOfficialAlignJob(
        model_dir=dl_crisper.out_hub_cache_dir,
        transformers_fork_dir=_cw_fork.out_repository,
        crisper_repo_dir=_cw_repo.out_repository,
        dataset_dir=_xa_dir,
        dataset_key="test",
    )
    _cw_off.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _cw_off.add_alias(f"hyp-align/crisperwhisper-official-{_xa_tag}-decode")
    reg(f"hyp-align/crisperwhisper-official-{_xa_tag}-hyps.txt.gz", _cw_off.out_hyps_txt)
    _cw_off_ds = BuildDatasetWithHypTranscriptsJob(
        dataset_dir=_xa_dir, dataset_key="test", hyps_txt=_cw_off.out_hyps_txt
    )
    _hy_metrics(f"crisperwhisper-official-{_xa_tag}", _cw_off.out_word_boundaries_hdf, _cw_off_ds.out_hub_cache_dir)

    # --- Auto-generated LaTeX result tables (in-graph; reference only this recipe's outputs) ---
    from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.grad_align_tables import build_tables

    build_tables(_table_results)

    # --- Auto-generated grad-score figures (in-graph; same captured-outputs dict) ---
    from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.grad_align_plots import build_plots

    build_plots(_table_results)


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
                reg(f"{name}.hdf", gen.out_hdf)

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
                    reg(f"{align_name}-wbe.txt", align.out_wbe)

    # --- prompt sensitivity sweep: DONE.
    # Implemented (and richer) via the _PP_PROMPTS block above:
    # 4 instructions x subword/char targets on TIMIT-test -> tab:phi4-prompt.
    # Finding: grad-align is robust to prompt phrasing (WBE varies only ~6 ms).
    # Supersedes the old 3-prompt sweep (exp2025_05_05_align ~172-203).

    # --- TODO: long-form Buckeye (chunked segmentation + grad-align) -------
    # See exp2025_05_05_align lines ~86-170. Use:
    #   from exp2025_07_07_in_grads.jobs.chunk_segmentation import ChunkSegmentationFromModelJob
    #   from exp2025_05_05_align import CalcChunkedAlignmentMetricsJob

    # --- TODO: external-aligner baselines (MFA / WhisperX phoneme align) ---
    # For comparison against the grad-align word-boundary numbers.
