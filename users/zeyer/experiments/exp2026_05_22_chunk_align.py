"""
DP-based chunked long-form decoding / segmentation.

Shares infrastructure with :mod:`exp2026_05_23_grad_align`:

- Datasets (TIMIT, Buckeye) via the same ``DownloadHuggingFaceRepoJobV2`` calls
  (same hash -> the finished download can be reused across setups).
  Buckeye uses ``nh0znoisung/buckeye`` directly --
  its unsegmented tracks (``track_id`` like "s0101a") ARE the raw long-form recordings;
  its word timestamps are quantized to a ~62.5 ms grid,
  which is negligible at the 10-30s chunk granularity this sweep cares about.
  We do NOT synthetically concatenate the ``alexwengg/buckeye`` fine segments here:
  that dataset explicitly cuts out inter-segment silence,
  so concatenation wouldn't reconstruct anything real.
- Model backbones from :mod:`exp2025_07_07_in_grads.jobs.models`
  (Phi4MM first, more later).
- ``ConcatenateForLongFormJob`` (:mod:`exp2025_07_07_in_grads.jobs.concat_longform_dataset`)
  is generic infra kept for a later TIMIT long-form variant
  (TIMIT utterances are independently prompted, not slices of one recording,
  so any concatenation there is synthetic by construction --
  unlike Buckeye, there's no real long-form recording being reconstructed).

Chunk assignment itself:
:class:`exp2025_07_07_in_grads.jobs.chunk_segmentation.ChunkSegmentationFromModelJob`
(DP over an (S+1)*C grid,
model's own next-word + exit log-probs only,
no VAD/external aligner) --
it only dumps the chunk assignment
(word-index range + chunk sample range) into an HDF,
nothing else.

Metrics are a separate job,
:class:`exp2025_07_07_in_grads.jobs.chunk_segmentation.CalcChunkAssignmentMetricsJob`:
word-to-chunk accuracy + chunk-index MAE,
not WBE
(WBE would need a within-chunk word position the DP never decides;
see that job's docstring).
"""

from __future__ import annotations

from typing import Any, Dict

import returnn.frontend as rf
from sisyphus import tk

from i6_experiments.users.zeyer.external_models.huggingface import DownloadHuggingFaceRepoJobV2
from i6_experiments.users.zeyer.external_models.phi4multimodal import download_phi4multimodal_model
from i6_experiments.users.zeyer.external_models.voxtral import download_voxtral_mini_3b_model
from i6_experiments.users.zeyer.external_models.canary_qwen import (
    download_canary_qwen_2_5b_model,
    download_qwen3_1_7b_model,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.phi4mm import Phi4MM
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.wav2vec2_ctc import Wav2Vec2Ctc
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.wav2vec2_phoneme_ctc import (
    Wav2Vec2PhonemeCtc,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.parakeet_ctc import ParakeetCtc
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.parakeet_rnnt import ParakeetRnnt
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.owsm_ctc import OwsmCtc
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.fastconformer_streaming import (
    FastConformerStreaming,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.emformer_rnnt import EmformerRnnt
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.whisper import Whisper
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.owls import Owls
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.voxtral import Voxtral
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.canary_qwen import CanaryQwen
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.canary_flash import CanaryFlash
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.granite_speech import GraniteSpeech
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.extract_self_attn import (
    SelectSelfAttnAlignHeadsJob,
    ExtractSelfAttnPerTokenJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.word_align_from_per_token_grads import (
    WordAlignFromPerTokenGradsJob,
)
from i6_experiments.users.zeyer.experiments.exp2026_05_23_grad_align import _phi4mm_model_config
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.chunk_segmentation import (
    ChunkSegmentationFromModelJob,
    ChunkSegmentationFromModelBatchedJob,
    ChunkBoundaryReverifyJob,
    DriftSpanRepairJob,
    CalcChunkAssignmentMetricsJob,
    ProportionalChunkAssignmentJob,
    GreedyExitChunkSegmentationJob,
    FreeDecodeLcsChunkSegmentationJob,
    ChunkAssignmentFromWordBoundariesJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.buckeye_fine_dataset import (
    MapBuckeyeFineTimestampsToLongFormJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.forced_align_baseline import (
    ForcedAlignBaselineJob,
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
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.native_transducer_align import (
    NativeTransducerAlignJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.apptainer import (
    PullApptainerImageJob,
    ApptainerExeWrapperJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.mfa_forced_align import (
    MfaDownloadModelJob,
    MfaForcedAlignJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_05_05_align import (
    CalcAlignmentMetricsFromWordBoundariesJob,
)

# Same convention as exp2025_05_05_align.py / exp2026_05_23_grad_align.py.
_DATASET_OFFSET_FACTORS = {"timit": 1, "buckeye": 1000}

# Same NeMo overlay as exp2026_05_23_grad_align (import patches for Canary/Parakeet/FastConformer).
_NEMO_OVERLAY = "/home/az668407/work/canary-qwen-overlay"

_table_results: Dict[str, tk.Variable] = {}


def reg(name, value, **kwargs):
    tk.register_output(name, value, **kwargs)
    _table_results[name] = value
    return value


def py():
    """Sisyphus entry point."""
    dl_phi4mm_dir = download_phi4multimodal_model()

    # Same repo_ids/args as exp2026_05_23_grad_align.py -> identical job hash,
    # import the finished download from that setup rather than re-downloading.
    dl_ds_timit = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/timit", repo_type="dataset")
    reg("timit-dataset", dl_ds_timit.out_hub_cache_dir)
    dl_ds_buckeye = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/buckeye", repo_type="dataset")
    reg("buckeye-dataset", dl_ds_buckeye.out_hub_cache_dir)

    dl_ds_buckeye_fine = DownloadHuggingFaceRepoJobV2(repo_id="alexwengg/buckeye", repo_type="dataset")
    dl_ds_buckeye_fine.set_env("HF_HUB_DISABLE_XET", "1")
    reg("buckeye-fine-raw", dl_ds_buckeye_fine.out_hub_cache_dir)

    # Same tracks/order/audio/transcript as nh0znoisung val,
    # word timestamps upgraded from the ~62.5 ms grid to the fine float-ms annotation
    # where the alexwengg segments cover them (word_fine marks which).
    # Chunk-seg HDFs computed on the nh0znoisung dataset stay valid here;
    # metric jobs read this dataset with dataset_offset_factors=1 (sample indices).
    buckeye_val_fine_ts = MapBuckeyeFineTimestampsToLongFormJob(
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        fine_raw_dir=dl_ds_buckeye_fine.out_hub_cache_dir,
    )
    buckeye_val_fine_ts.add_alias("buckeye-val-fine-ts-dataset")
    reg("buckeye-val-fine-ts-dataset", buckeye_val_fine_ts.out_hub_cache_dir)

    phi4mm_cfg = rf.build_dict(Phi4MM, model_dir=dl_phi4mm_dir)

    # TIMIT deferred: utterances (~3s) are shorter than every chunk_size_secs tried below,
    # so the DP degenerates to a single chunk there -- not informative as is.
    # Revisit once we've discussed concatenating TIMIT seqs
    # into longer pseudo-utterances to get real multi-chunk cases.
    _datasets = {
        "buckeye": (dl_ds_buckeye.out_hub_cache_dir, "val", _DATASET_OFFSET_FACTORS["buckeye"]),
    }
    _backbones: Dict[str, Dict[str, Any]] = {"phi4mm": phi4mm_cfg}

    for model_name, model_cfg in _backbones.items():
        for ds_name, (ds_dir, ds_key, ds_offset) in _datasets.items():
            # (chunk_size_secs, chunk_overlap_secs) variants. The name encodes the overlap so the
            # three schemes coexist: original fixed 5s (10/20/30), non-overlapping 0s (5/3/2), and
            # half-chunk 50% overlap across all sizes (incl. cs1).
            for chunk_size_secs, chunk_overlap_secs in [
                # fixed 5s overlap -- the original 10/20/30 (already finished; reused)
                (30.0, 5.0),
                (20.0, 5.0),
                (10.0, 5.0),
                # non-overlapping -- the 5/3/2 already running (reused)
                (30.0, 0.0),
                (20.0, 0.0),
                (10.0, 0.0),
                (5.0, 0.0),
                (3.0, 0.0),
                (2.0, 0.0),
                (1.0, 0.0),
                (0.5, 0.0),
                # half-chunk overlap (50%); (10.0, 5.0) is already listed above
                (30.0, 15.0),
                (20.0, 10.0),
                (10.0, 5.0),
                (5.0, 2.5),
                (3.0, 1.5),
                (2.0, 1.0),
                (1.0, 0.5),
                # overlap sweep
                (10.0, 5.0),
                (10.0, 2.5),
                (10.0, 1.0),
                (10.0, 0.0),
            ]:
                seg_name = (
                    f"chunk-align/{model_name}-{ds_name}-{ds_key}-cs{chunk_size_secs:.0f}-ov{chunk_overlap_secs:g}"
                )
                # Segmentation via the batched fast-path DP only (verified quality-neutral vs the
                # single-seq DP, ~3x faster): grad_wrt=None (no grads needed), bf16.
                batched_cfg = {**model_cfg, "grad_wrt": None, "model_dtype": "bfloat16"}
                seg = ChunkSegmentationFromModelBatchedJob(
                    dataset_dir=ds_dir,
                    dataset_key=ds_key,
                    model_config=batched_cfg,
                    chunk_size_secs=chunk_size_secs,
                    chunk_overlap_secs=chunk_overlap_secs,
                    max_batch_size=8,
                )
                seg.add_alias(seg_name)
                reg(f"{seg_name}.hdf", seg.out_hdf)

                # Metric vs the nh0znoisung word timestamps (~62.5 ms grid; fine enough at
                # second-scale chunking). We do NOT use buckeye_val_fine_ts: the alexwengg fine
                # annotation covers only ~half the val speakers (23/46 tracks have zero fine words),
                # so it would be a coarse/fine mix, not a clean gold.
                metric = CalcChunkAssignmentMetricsJob(
                    chunk_seg_hdf=seg.out_hdf,
                    dataset_dir=ds_dir,
                    dataset_key=ds_key,
                    dataset_offset_factors=ds_offset,
                )
                metric.add_alias(f"{seg_name}-metric")
                reg(f"{seg_name}-accuracy.txt", metric.out_accuracy)
                reg(f"{seg_name}-chunk_idx_mae.txt", metric.out_chunk_idx_mae)
                reg(f"{seg_name}-error-median-sec.txt", metric.out_error_median_sec)
                reg(f"{seg_name}-error-p95-sec.txt", metric.out_error_p95_sec)
                reg(f"{seg_name}-frac-gt-1s.txt", metric.out_frac_gt_1s)

    # Hyper-param sweep at the best 10s setting (cs10/ov2.5): empty_exit_penalty x word_start_heuristic.
    # empty_exit_penalty only applies to exiting a chunk with zero words assigned; overlap makes such
    # empty chunks more likely, so this is where it should matter. word_start_heuristic=False turns the
    # pruning off (every chunk forwards the transcript from word 0 instead of from the prev chunk's best
    # exit) -- the exact reference for what the heuristic costs, at ~2x the compute.
    # (eep=-5, wsh=True) are the defaults, i.e. the same job as the plain cs10-ov2.5 above (reused).
    _cfg_hp = rf.build_dict(Phi4MM, model_dir=dl_phi4mm_dir, grad_wrt=None, model_dtype="bfloat16")
    for _wsh in [True, False]:
        for _eep in [0.0, -2.0, -5.0, -10.0, -20.0]:
            _seg_hp = ChunkSegmentationFromModelBatchedJob(
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                model_config=_cfg_hp,
                chunk_size_secs=10.0,
                chunk_overlap_secs=2.5,
                empty_exit_penalty=_eep,
                word_start_heuristic=_wsh,
                max_batch_size=8,
            )
            _hp_name = f"chunk-align/phi4mm-buckeye-val-cs10-ov2.5-eep{_eep:g}-wsh{int(_wsh)}"
            _seg_hp.add_alias(_hp_name)
            reg(f"{_hp_name}.hdf", _seg_hp.out_hdf)

            _metric_hp = CalcChunkAssignmentMetricsJob(
                chunk_seg_hdf=_seg_hp.out_hdf,
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
            )
            _metric_hp.add_alias(f"{_hp_name}-metric")
            reg(f"{_hp_name}-accuracy.txt", _metric_hp.out_accuracy)
            reg(f"{_hp_name}-chunk_idx_mae.txt", _metric_hp.out_chunk_idx_mae)

    # empty_exit_penalty across chunk configs (word_start_heuristic=True throughout: the cs10/ov2.5
    # ablation showed the heuristic is ~2x cheaper and no worse). Not a full cs x ov grid on purpose:
    # eep only fires on a chunk that would exit with ZERO words, so what decides whether it acts at
    # all is the empty-chunk pressure -- chunks (~duration/(cs-ov)) vs words (~3/s). cs and ov act
    # through that same step, so we sweep along the pressure axis and isolate overlap once at fixed cs.
    # eep=-5 is the default -> those cells reuse the plain cs*-ov* jobs above (free).
    for _cs, _ov, _eeps in [
        # low pressure control (step 25s, ~75 words/chunk): expect eep to do nothing
        (30.0, 5.0, [0.0, -5.0, -20.0]),
        # isolate OVERLAP at fixed cs=10 (cs10/ov2.5 is already swept above): step 5s vs 7.5s
        (10.0, 5.0, [0.0, -2.0, -5.0, -10.0, -20.0]),
        # isolate CHUNK SIZE at zero overlap (step 2s)
        (2.0, 0.0, [0.0, -2.0, -5.0, -10.0, -20.0]),
        # collapse probe (step 0.5s -> ~2 chunks/s vs ~3 words/s, so most chunks MUST be empty and
        # the -5 default fights that): does removing the penalty recover acc 0.07 / 0.02?
        (1.0, 0.5, [0.0, -5.0]),
        (0.5, 0.0, [0.0, -5.0]),
        # optimal overlap at cs1 once eep is set right (eep=0): ov0.5 gave 0.5530, so map the rest.
        # eep=-5 for cs1-ov0 is the plain cs1-ov0 job above (0.6318).
        (1.0, 0.0, [0.0]),
        (1.0, 0.25, [0.0]),
        (1.0, 0.75, [0.0]),
        # eep=0 for the remaining overlapped cells of the main cs/ov sweep (paper stride table):
        # that sweep ran everything at eep=-5, the wrong sign for overlapped configs,
        # plus cs10-ov0/ov1 so the stride table is uniform in eep.
        (30.0, 15.0, [0.0]),
        (20.0, 10.0, [0.0]),
        (10.0, 1.0, [0.0]),
        (10.0, 0.0, [0.0]),
        (5.0, 2.5, [0.0]),
        (3.0, 1.5, [0.0]),
        (2.0, 1.0, [0.0]),
        # eep=0 no-overlap anchors, so the stride table has a stride = L row per chunk size
        (30.0, 0.0, [0.0]),
        (20.0, 0.0, [0.0]),
        (5.0, 0.0, [0.0]),
        (3.0, 0.0, [0.0]),
    ]:
        for _eep in _eeps:
            _seg_e = ChunkSegmentationFromModelBatchedJob(
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                model_config=_cfg_hp,
                chunk_size_secs=_cs,
                chunk_overlap_secs=_ov,
                empty_exit_penalty=_eep,
                max_batch_size=8,
            )
            _e_name = f"chunk-align/phi4mm-buckeye-val-cs{_cs:.0f}-ov{_ov:g}-eep{_eep:g}"
            _seg_e.add_alias(_e_name)
            reg(f"{_e_name}.hdf", _seg_e.out_hdf)

            _metric_e = CalcChunkAssignmentMetricsJob(
                chunk_seg_hdf=_seg_e.out_hdf,
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
            )
            _metric_e.add_alias(f"{_e_name}-metric")
            reg(f"{_e_name}-accuracy.txt", _metric_e.out_accuracy)
            reg(f"{_e_name}-chunk_idx_mae.txt", _metric_e.out_chunk_idx_mae)

    # word_start_heuristic=False (exact, unpruned) at smaller chunk sizes. At cs10/ov2.5 the exact DP
    # was FLAT across eep (0.9847-0.9849) while the heuristic swung (0.976-0.986), i.e. eep acts only
    # through the heuristic's argmax, not through the DP itself. If that generalizes, the small-chunk
    # degradation is a heuristic failure and eep merely modulates it. Not run at the step-0.5s configs:
    # unpruned there means ~1200 chunks x full-transcript forwards per seq, which would blow the 12h cap.
    for _cs, _ov, _eeps in [
        (2.0, 0.0, [0.0, -5.0, -20.0]),
        (1.0, 0.0, [0.0, -5.0]),
    ]:
        for _eep in _eeps:
            _seg_x = ChunkSegmentationFromModelBatchedJob(
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                model_config=_cfg_hp,
                chunk_size_secs=_cs,
                chunk_overlap_secs=_ov,
                empty_exit_penalty=_eep,
                word_start_heuristic=False,
                max_batch_size=8,
            )
            _x_name = f"chunk-align/phi4mm-buckeye-val-cs{_cs:.0f}-ov{_ov:g}-eep{_eep:g}-wsh0"
            _seg_x.add_alias(_x_name)
            reg(f"{_x_name}.hdf", _seg_x.out_hdf)

            _metric_x = CalcChunkAssignmentMetricsJob(
                chunk_seg_hdf=_seg_x.out_hdf,
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
            )
            _metric_x.add_alias(f"{_x_name}-metric")
            reg(f"{_x_name}-accuracy.txt", _metric_x.out_accuracy)
            reg(f"{_x_name}-chunk_idx_mae.txt", _metric_x.out_chunk_idx_mae)

    # --- algorithm variants (batched job only) ---

    # 1) word_start_beam: one knob interpolating the argmax heuristic (beam=None, the old default)
    # and the exact DP (word_start_heuristic=False).
    # cs1-ov0 is where the two endpoints differ most (0.6318 argmax vs 0.2473 exact), so map the curve there:
    # wider beam = less pruning = closer to exact, which we expect to get WORSE
    # (search-error effect, cf. the NMT beam-search curse).
    for _beam in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]:
        _seg_bm = ChunkSegmentationFromModelBatchedJob(
            dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
            dataset_key="val",
            model_config=_cfg_hp,
            chunk_size_secs=1.0,
            chunk_overlap_secs=0.0,
            max_batch_size=8,
            word_start_beam=_beam,
        )
        _bm_name = f"chunk-align/phi4mm-buckeye-val-cs1-ov0-beam{_beam:g}"
        _seg_bm.add_alias(_bm_name)
        reg(f"{_bm_name}.hdf", _seg_bm.out_hdf)
        _m_bm = CalcChunkAssignmentMetricsJob(
            chunk_seg_hdf=_seg_bm.out_hdf,
            dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
        )
        _m_bm.add_alias(f"{_bm_name}-metric")
        reg(f"{_bm_name}-accuracy.txt", _m_bm.out_accuracy)
        reg(f"{_bm_name}-chunk_idx_mae.txt", _m_bm.out_chunk_idx_mae)

    # 2) exit_bias: a global words-per-chunk knob (added to EVERY exit),
    # vs empty_exit_penalty which only hits chunks exiting with zero words,
    # and which we showed is a pure heuristic artifact.
    # Swept with eep=0 so the bias is the only exit knob.
    # bias=0 is the existing cs2-ov0-eep0 (0.8535).
    for _bias in [-2.0, -1.0, 1.0, 2.0]:
        _seg_eb = ChunkSegmentationFromModelBatchedJob(
            dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
            dataset_key="val",
            model_config=_cfg_hp,
            chunk_size_secs=2.0,
            chunk_overlap_secs=0.0,
            empty_exit_penalty=0.0,
            max_batch_size=8,
            exit_bias=_bias,
        )
        _eb_name = f"chunk-align/phi4mm-buckeye-val-cs2-ov0-eep0-bias{_bias:g}"
        _seg_eb.add_alias(_eb_name)
        reg(f"{_eb_name}.hdf", _seg_eb.out_hdf)
        _m_eb = CalcChunkAssignmentMetricsJob(
            chunk_seg_hdf=_seg_eb.out_hdf,
            dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
        )
        _m_eb.add_alias(f"{_eb_name}-metric")
        reg(f"{_eb_name}-accuracy.txt", _m_eb.out_accuracy)
        reg(f"{_eb_name}-chunk_idx_mae.txt", _m_eb.out_chunk_idx_mae)

    # 3) length_norm: score a word by its per-token MEAN log-prob instead of the sum,
    # so a multi-token word is not systematically dearer to emit than the single exit token.
    # Compare against the plain cs*-ov* jobs (cs2-ov0 0.8747, cs10-ov2.5 0.9856, cs1-ov0 0.6318).
    for _cs, _ov in [(2.0, 0.0), (10.0, 2.5), (1.0, 0.0)]:
        _seg_ln = ChunkSegmentationFromModelBatchedJob(
            dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
            dataset_key="val",
            model_config=_cfg_hp,
            chunk_size_secs=_cs,
            chunk_overlap_secs=_ov,
            max_batch_size=8,
            length_norm=True,
        )
        _ln_name = f"chunk-align/phi4mm-buckeye-val-cs{_cs:.0f}-ov{_ov:g}-lnorm"
        _seg_ln.add_alias(_ln_name)
        reg(f"{_ln_name}.hdf", _seg_ln.out_hdf)
        _m_ln = CalcChunkAssignmentMetricsJob(
            chunk_seg_hdf=_seg_ln.out_hdf,
            dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
        )
        _m_ln.add_alias(f"{_ln_name}-metric")
        reg(f"{_ln_name}-accuracy.txt", _m_ln.out_accuracy)
        reg(f"{_ln_name}-chunk_idx_mae.txt", _m_ln.out_chunk_idx_mae)

    # cs30-ov0 with per-word score dump:
    # the raw material for confidence flagging / drift detection
    # (does a low word score predict the >5s-misplaced words?).
    _seg_sc = ChunkSegmentationFromModelBatchedJob(
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        model_config=_cfg_hp,
        chunk_size_secs=30.0,
        chunk_overlap_secs=0.0,
        max_batch_size=8,
        dump_word_scores=True,
    )
    _seg_sc.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-scores")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-scores.hdf", _seg_sc.out_hdf)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-word-scores.hdf", _seg_sc.out_word_scores_hdf)

    # Second, staggered chunking (grid offset by L/2 = 15s): every grid-A boundary falls
    # mid-chunk of grid B, so B is robust exactly where A is fragile.
    # Combination ideas: A-vs-B disagreement as a detector, and the A/B intersection cell
    # as a double-resolution localization with full 30s context per score.
    _seg_off = ChunkSegmentationFromModelBatchedJob(
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        model_config=_cfg_hp,
        chunk_size_secs=30.0,
        chunk_overlap_secs=0.0,
        max_batch_size=8,
        dump_word_scores=True,
        chunk_offset_secs=15.0,
    )
    _seg_off.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-offset15")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-offset15.hdf", _seg_off.out_hdf)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-offset15-word-scores.hdf", _seg_off.out_word_scores_hdf)
    _m_off = CalcChunkAssignmentMetricsJob(
        chunk_seg_hdf=_seg_off.out_hdf,
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
    )
    _m_off.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-offset15-metric")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-offset15-accuracy.txt", _m_off.out_accuracy)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-offset15-error-p95-sec.txt", _m_off.out_error_p95_sec)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-offset15-frac-gt-1s.txt", _m_off.out_frac_gt_1s)

    # Char-level scoring probe: score the transcript exploded to chars instead of canonical BPE.
    # Hypothesis: char-by-char spelling weakens the pure-LM shortcut behind too-early emission,
    # so the systematic tail (drift runs) should shrink if that shortcut is the cause.
    _seg_char = ChunkSegmentationFromModelBatchedJob(
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        model_config={**_cfg_hp, "char_level": True},
        chunk_size_secs=30.0,
        chunk_overlap_secs=0.0,
        max_batch_size=8,
        dump_word_scores=True,
    )
    _seg_char.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-charlevel")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-charlevel.hdf", _seg_char.out_hdf)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-charlevel-word-scores.hdf", _seg_char.out_word_scores_hdf)
    _m_char = CalcChunkAssignmentMetricsJob(
        chunk_seg_hdf=_seg_char.out_hdf,
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
    )
    _m_char.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-charlevel-metric")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-charlevel-accuracy.txt", _m_char.out_accuracy)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-charlevel-error-p95-sec.txt", _m_char.out_error_p95_sec)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-charlevel-frac-gt-1s.txt", _m_char.out_frac_gt_1s)

    # Boundary re-verification (local repair) on that cs30-ov0 assignment:
    # per-word acoustic comparison in the two adjacent chunks, +-10 words per boundary.
    # Metric on the refined assignment tells whether the local 1-5-word runs get fixed
    # (compare vs the plain cs30-ov0 metric).
    _rv = ChunkBoundaryReverifyJob(
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        model_config=_cfg_hp,
        chunk_seg_hdf=_seg_sc.out_hdf,
    )
    _rv.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify.hdf", _rv.out_hdf)
    _m_rv = CalcChunkAssignmentMetricsJob(
        chunk_seg_hdf=_rv.out_hdf,
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
    )
    _m_rv.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-metric")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-accuracy.txt", _m_rv.out_accuracy)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-error-p95-sec.txt", _m_rv.out_error_p95_sec)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-frac-gt-1s.txt", _m_rv.out_frac_gt_1s)

    # Gated variant: skip boundaries in low-windowed-confidence (drifted) regions,
    # where the ungated pass made the worst case WORSE (max 45s -> 77s).
    _rvg = ChunkBoundaryReverifyJob(
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        model_config=_cfg_hp,
        chunk_seg_hdf=_seg_sc.out_hdf,
        word_scores_hdf=_seg_sc.out_word_scores_hdf,
    )
    _rvg.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-gated")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-gated.hdf", _rvg.out_hdf)
    _m_rvg = CalcChunkAssignmentMetricsJob(
        chunk_seg_hdf=_rvg.out_hdf,
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
    )
    _m_rvg.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-gated-metric")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-gated-accuracy.txt", _m_rvg.out_accuracy)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-gated-error-p95-sec.txt", _m_rvg.out_error_p95_sec)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-gated-frac-gt-1s.txt", _m_rvg.out_frac_gt_1s)

    # Margin variant (with the orphan-bugfix): moves must be decisive
    # (>= 2 log-prob units per moved word), protecting correct but poorly-scored words.
    _rvm = ChunkBoundaryReverifyJob(
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        model_config=_cfg_hp,
        chunk_seg_hdf=_seg_sc.out_hdf,
        word_scores_hdf=_seg_sc.out_word_scores_hdf,
        min_move_margin=2.0,
    )
    _rvm.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-m2")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-m2.hdf", _rvm.out_hdf)
    _m_rvm = CalcChunkAssignmentMetricsJob(
        chunk_seg_hdf=_rvm.out_hdf,
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
    )
    _m_rvm.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-m2-metric")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-m2-accuracy.txt", _m_rvm.out_accuracy)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-m2-error-p95-sec.txt", _m_rvm.out_error_p95_sec)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-m2-frac-gt-1s.txt", _m_rvm.out_frac_gt_1s)

    # Drift-span repair: the reverify layer fixes local 1-5-word runs but is gated OFF in
    # drifted regions; this targets exactly those (windowed-conf flagged spans, e.g. the
    # 97-word run of seq 16) with an anchored acoustic re-assignment DP over whole spans.
    _dr = DriftSpanRepairJob(
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        model_config=_cfg_hp,
        chunk_seg_hdf=_seg_sc.out_hdf,
        word_scores_hdf=_seg_sc.out_word_scores_hdf,
    )
    _dr.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-driftrepair")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-driftrepair.hdf", _dr.out_hdf)
    _m_dr = CalcChunkAssignmentMetricsJob(
        chunk_seg_hdf=_dr.out_hdf,
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
    )
    _m_dr.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov0-driftrepair-metric")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-driftrepair-accuracy.txt", _m_dr.out_accuracy)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-driftrepair-error-p95-sec.txt", _m_dr.out_error_p95_sec)
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov0-driftrepair-frac-gt-1s.txt", _m_dr.out_frac_gt_1s)

    # fp32 batched (default fast path) at cs30, to check the fast path (esp. batched_logprobs) is
    # bit-exact vs the fp32 single-seq reference below. The bf16 sweep diverges more for small
    # chunks, so this isolates real logic differences from bf16 numerical noise.
    _cfg_fp32b = rf.build_dict(Phi4MM, model_dir=dl_phi4mm_dir, grad_wrt=None, model_dtype="float32")
    _segb = ChunkSegmentationFromModelBatchedJob(
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        model_config=_cfg_fp32b,
        chunk_size_secs=30.0,
        max_batch_size=4,
    )
    _segb.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov5-batched-float32")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov5-batched-float32.hdf", _segb.out_hdf)

    # fp32 single-seq reference: the proper same-precision baseline for the fp32 batched job
    # (the default single cs30-ov5 is bf16). fp32 batched vs fp32 single should be 0% word diff.
    _cfg_fp32s = rf.build_dict(Phi4MM, model_dir=dl_phi4mm_dir, model_dtype="float32")
    _seg_fp32s = ChunkSegmentationFromModelJob(
        dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
        dataset_key="val",
        model_config=_cfg_fp32s,
        chunk_size_secs=30.0,
    )
    _seg_fp32s.add_alias("chunk-align/phi4mm-buckeye-val-cs30-ov5-single-float32")
    reg("chunk-align/phi4mm-buckeye-val-cs30-ov5-single-float32.hdf", _seg_fp32s.out_hdf)

    # === Model zoo: all grad-align paper models through the chunk DP (single-seq job). ===
    # Download jobs re-created with the exact grad-align args (identical hash);
    # the finished download dirs are imported (symlinked) from that setup, NEVER rerun.
    # Word scores = the grad-align labelwise prefix scores (prefix_fwd CTC / prefix transducer);
    # exit scores from the same lattices (verified; TDT falls back to exit=0).
    # Whisper family uses the native <|startofprev|> prev context (pass_omitted_prev_words).
    dl_owsm_ctc = DownloadHuggingFaceRepoJobV2(repo_id="espnet/owsm_ctc_v4_1B", repo_type="model")
    dl_owsm_ctc.set_env("HF_HUB_DISABLE_XET", "1")
    dl_whisper = DownloadHuggingFaceRepoJobV2(repo_id="openai/whisper-base", repo_type="model")
    dl_whisper_l3 = DownloadHuggingFaceRepoJobV2(repo_id="openai/whisper-large-v3", repo_type="model")
    dl_crisper = DownloadHuggingFaceRepoJobV2(repo_id="nyrahealth/CrisperWhisper", repo_type="model")
    dl_parakeet_rnnt = DownloadHuggingFaceRepoJobV2(repo_id="nvidia/parakeet-rnnt-1.1b", repo_type="model")
    dl_parakeet_tdt = DownloadHuggingFaceRepoJobV2(repo_id="nvidia/parakeet-tdt-0.6b-v2", repo_type="model")
    dl_parakeet_ctc = DownloadHuggingFaceRepoJobV2(repo_id="nvidia/parakeet-ctc-1.1b", repo_type="model")
    dl_fc_stream = DownloadHuggingFaceRepoJobV2(
        repo_id="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi", repo_type="model"
    )
    dl_owls_1b = DownloadHuggingFaceRepoJobV2(repo_id="espnet/owls_1B_180K", repo_type="model")
    dl_w2v_phoneme = DownloadHuggingFaceRepoJobV2(
        repo_id="vitouphy/wav2vec2-xls-r-300m-timit-phoneme", repo_type="model"
    )
    dl_voxtral = download_voxtral_mini_3b_model()
    dl_canary = download_canary_qwen_2_5b_model()
    dl_qwen3 = download_qwen3_1_7b_model()
    # Open ASR Leaderboard additions (2026-08-30): a top-10 AED with a hard decoder-length cap
    # and a top-3 LLM decoder without one (the long-form contrast pair).
    dl_canary_flash = DownloadHuggingFaceRepoJobV2(repo_id="nvidia/canary-1b-flash", repo_type="model")
    dl_granite = DownloadHuggingFaceRepoJobV2(repo_id="ibm-granite/granite-speech-3.3-8b", repo_type="model")

    _fc_att = [70, 6]
    _zoo = [  # (name, model_config, pass_omitted_prev_words)
        ("mms-fa", rf.build_dict(Wav2Vec2Ctc, per_token_score="prefix_fwd"), False),
        (
            "w2v-phoneme",
            rf.build_dict(
                Wav2Vec2PhonemeCtc,
                model_dir=dl_w2v_phoneme.out_hub_cache_dir,
                g2p_word_targets=True,
                per_token_score="prefix_fwd",
            ),
            False,
        ),
        (
            "parakeet-ctc-1.1b",
            rf.build_dict(
                ParakeetCtc,
                model_dir=dl_parakeet_ctc.out_hub_cache_dir,
                overlay_path=_NEMO_OVERLAY,
                per_token_score="prefix_fwd",
            ),
            False,
        ),
        (
            "owsm-ctc-v4-1b",
            rf.build_dict(OwsmCtc, model_dir=dl_owsm_ctc.out_hub_cache_dir, version=2, per_token_score="prefix_fwd"),
            False,
        ),
        (
            "fastconformer-stream-ctc",
            rf.build_dict(
                FastConformerStreaming,
                model_dir=dl_fc_stream.out_hub_cache_dir,
                overlay_path=_NEMO_OVERLAY,
                head="ctc",
                att_context_size=_fc_att,
            ),
            False,
        ),
        (
            "fastconformer-stream-rnnt",
            rf.build_dict(
                FastConformerStreaming,
                model_dir=dl_fc_stream.out_hub_cache_dir,
                overlay_path=_NEMO_OVERLAY,
                head="rnnt",
                att_context_size=_fc_att,
            ),
            False,
        ),
        (
            "parakeet-rnnt-1.1b",
            rf.build_dict(
                ParakeetRnnt,
                model_dir=dl_parakeet_rnnt.out_hub_cache_dir,
                per_token_score="prefix",
                overlay_path=_NEMO_OVERLAY,
            ),
            False,
        ),
        (
            "parakeet-tdt-0.6b-v2",
            rf.build_dict(
                ParakeetRnnt,
                model_dir=dl_parakeet_tdt.out_hub_cache_dir,
                per_token_score="prefix",
                overlay_path=_NEMO_OVERLAY,
            ),
            False,
        ),
        ("emformer-rnnt", rf.build_dict(EmformerRnnt, per_token_score="prefix"), False),
        # exit_score="timestamps": Whisper's native close-the-segment signal (timestamp-token
        # mass) as the chunk exit; mid-transcript EOT is degenerate (round-1/2/3 finding).
        # FINAL whisper config = the round-2 best (EOT exit + startofprev prev context
        # + completion norm True): 0.78 / 0.82 / 0.79. Variants tried and worse:
        # consumed-norm (0.62-0.66), timestamp exit (0.75-0.78), no prev context (0.68-0.75).
        # Whisper's teacher-forced token scores are position-insensitive (plateau ~0.8);
        # its localization lives in cross-attention (cf. grad-align), not token log-probs.
        ("whisper-base", rf.build_dict(Whisper, model_dir=dl_whisper.out_hub_cache_dir), True),
        ("whisper-large-v3", rf.build_dict(Whisper, model_dir=dl_whisper_l3.out_hub_cache_dir), True),
        ("crisperwhisper", rf.build_dict(Whisper, model_dir=dl_crisper.out_hub_cache_dir), True),
        ("owls-1b-180k", rf.build_dict(Owls, model_dir=dl_owls_1b.out_hub_cache_dir), False),
        (
            "voxtral",
            rf.build_dict(Voxtral, model_dir=dl_voxtral, forward_mode="transcription", version=3),
            False,
        ),
        (
            "canary-qwen",
            rf.build_dict(CanaryQwen, model_dir=dl_canary, llm_model_dir=dl_qwen3, version=3),
            False,
        ),
        (
            "canary-1b-flash",
            rf.build_dict(CanaryFlash, model_dir=dl_canary_flash.out_hub_cache_dir, overlay_path=_NEMO_OVERLAY),
            False,
        ),
        ("granite-speech-8b", rf.build_dict(GraniteSpeech, model_dir=dl_granite.out_hub_cache_dir), False),
    ]
    # Cap candidate words per chunk for limited-context models
    # (Whisper: 448 decoder positions incl. the <|startofprev|> prompt;
    # Emformer: the T x (U+1) x V joint lattice OOMs on full-transcript U).
    # A 30s chunk holds at most ~100 words, so the caps are semantically free.
    _zoo_max_words = {
        "whisper-base": 120,
        "whisper-large-v3": 120,
        "whisper-large-v3-noprev": 120,
        "crisperwhisper": 120,
        "emformer-rnnt": 200,
    }
    # Length-fair word-start selection for the weak-score models (acc < 90% in the first round;
    # plain argmax exit degenerates to extreme lag / erratic starts there, see the job docstring).
    # Per-model best after rounds 1-3 (see readme): mms-fa / w2v-phoneme best WITHOUT norm
    # (both norm variants collapsed them); emformer best with "consumed";
    # whisper best with the naive True norm (paired with the timestamp exit above).
    _zoo_start_norm = {
        "whisper-base": True,
        "whisper-large-v3": True,
        "whisper-large-v3-noprev": True,
        "crisperwhisper": True,
        "emformer-rnnt": "consumed",
    }
    # No-context-marker ablation ("feed nothing as history"): the LLM wrappers prepend a
    # "... " continuation marker for chunks after the first; is it needed at all?
    # (Whisper's analog -- no startofprev -- was round 5: worse. CTC feeds nothing anyway.)
    _zoo = _zoo + [
        # NOT {**_cfg_hp, ...}: that carries grad_wrt=None (batched-job convention),
        # which the single-seq Phi4MM forward asserts against; default grad_wrt is fine
        # (the job disables grads globally).
        (
            "phi4mm-noctx",
            rf.build_dict(Phi4MM, model_dir=dl_phi4mm_dir, model_dtype="bfloat16", omitted_ctx_marker=False),
            False,
        ),
        (
            "voxtral-noctx",
            rf.build_dict(
                Voxtral, model_dir=dl_voxtral, forward_mode="transcription", version=3, omitted_ctx_marker=False
            ),
            False,
        ),
        # Transducer prev-label-context ablation: condition the predictor on the TRUE
        # previous labels (last 32 words, no emission required -- cur-position joint states
        # only), vs the plain fresh-predictor-per-chunk approximation.
        (
            "parakeet-rnnt-1.1b-prevctx",
            rf.build_dict(
                ParakeetRnnt,
                model_dir=dl_parakeet_rnnt.out_hub_cache_dir,
                per_token_score="prefix",
                overlay_path=_NEMO_OVERLAY,
            ),
            True,
        ),
        (
            "parakeet-tdt-0.6b-v2-prevctx",
            rf.build_dict(
                ParakeetRnnt,
                model_dir=dl_parakeet_tdt.out_hub_cache_dir,
                per_token_score="prefix",
                overlay_path=_NEMO_OVERLAY,
            ),
            True,
        ),
        # AED context ablation: Whisper-large-v3 WITHOUT the native <|startofprev|> prev text.
        # Same final config otherwise (EOT exit, completion norm, cap 120); the old round-5
        # no-prev runs are confounded by the timestamp exit, so this is the clean pair.
        ("whisper-large-v3-noprev", rf.build_dict(Whisper, model_dir=dl_whisper_l3.out_hub_cache_dir), False),
    ]
    for _zname, _zcfg, _zprev in _zoo:
        _zseg = ChunkSegmentationFromModelJob(
            dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
            dataset_key="val",
            model_config=_zcfg,
            chunk_size_secs=30.0,
            chunk_overlap_secs=0.0,
            pass_omitted_prev_words=_zprev,
            max_words_per_chunk=_zoo_max_words.get(_zname),
            word_start_completion_norm=_zoo_start_norm.get(_zname, False),
        )
        _zseg.add_alias(f"chunk-align/zoo/{_zname}-buckeye-val-cs30-ov0")
        reg(f"chunk-align/zoo/{_zname}-buckeye-val-cs30-ov0.hdf", _zseg.out_hdf)
        _zm = CalcChunkAssignmentMetricsJob(
            chunk_seg_hdf=_zseg.out_hdf,
            dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
            dataset_key="val",
            dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
        )
        _zm.add_alias(f"chunk-align/zoo/{_zname}-buckeye-val-cs30-ov0-metric")
        reg(f"chunk-align/zoo/{_zname}-buckeye-val-cs30-ov0-accuracy.txt", _zm.out_accuracy)
        reg(f"chunk-align/zoo/{_zname}-buckeye-val-cs30-ov0-error-p95-sec.txt", _zm.out_error_p95_sec)
        reg(f"chunk-align/zoo/{_zname}-buckeye-val-cs30-ov0-frac-gt-1s.txt", _zm.out_frac_gt_1s)

    # exit_scale sweep: how much of the exit score does the DP need at all?
    # Crossed with the word-start heuristic: its boundary choice reads accum_exit,
    # so at exit_scale 0 it degenerates toward "consume little" (the lag mechanism);
    # the exact DP (heuristic off) is the clean no-exit test.
    # (1.0, True) = the existing sweep cells; (1.0, False) = the exact-DP baseline.
    for _es_cs, _es_ov in [(30.0, 0.0), (10.0, 2.5)]:
        for _es_scale in [1.0, 0.5, 0.0]:
            for _es_wsh in [True, False]:
                _es_seg = ChunkSegmentationFromModelBatchedJob(
                    dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                    dataset_key="val",
                    model_config=_cfg_hp,
                    chunk_size_secs=_es_cs,
                    chunk_overlap_secs=_es_ov,
                    word_start_heuristic=_es_wsh,
                    max_batch_size=8,
                    exit_scale=_es_scale,
                )
                _es_name = (
                    f"chunk-align/exit-scale/phi4mm-buckeye-val-cs{_es_cs:g}-ov{_es_ov:g}"
                    f"-es{_es_scale:g}-wsh{int(_es_wsh)}"
                )
                _es_seg.add_alias(_es_name)
                reg(f"{_es_name}.hdf", _es_seg.out_hdf)
                _es_m = CalcChunkAssignmentMetricsJob(
                    chunk_seg_hdf=_es_seg.out_hdf,
                    dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                    dataset_key="val",
                    dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
                )
                _es_m.add_alias(f"{_es_name}-metric")
                reg(f"{_es_name}-accuracy.txt", _es_m.out_accuracy)
                reg(f"{_es_name}-error-p95-sec.txt", _es_m.out_error_p95_sec)
                reg(f"{_es_name}-frac-gt-1s.txt", _es_m.out_frac_gt_1s)

    # === Native alignment baselines directly on LONG-FORM Buckeye (no chunking). ===
    # Each model's own aligner (job classes shared with the grad-align recipe; that recipe ran
    # them on the RESEGMENTED Buckeye to work around long-sequence issues, here we deliberately
    # feed the raw unsegmented val tracks). Several are expected to break -- bounded audio
    # context (Whisper's 30s window), lattice memory (transducers), training-length mismatch --
    # each such case is a reported result, not an infrastructure failure.
    # MFA (GMM-HMM) as the external reference aligner.
    _lf_dir = dl_ds_buckeye.out_hub_cache_dir
    _lf_off = _DATASET_OFFSET_FACTORS["buckeye"]
    # openai-whisper timestamp overlay, same checkout as the grad-align recipe uses
    _lf_whisper_overlay = "/home/az668407/work/whisper-ts-overlay"

    def _lf_metric(_wb_hdf, _lf_name):
        _lf_m = CalcAlignmentMetricsFromWordBoundariesJob(
            word_boundaries_hdf=_wb_hdf,
            dataset_dir=_lf_dir,
            dataset_key="val",
            dataset_offset_factors=_lf_off,
        )
        _lf_m.add_alias(f"{_lf_name}-metric")
        reg(f"{_lf_name}-wbe.txt", _lf_m.out_wbe)
        reg(f"{_lf_name}-acc50.txt", _lf_m.out_acc50)
        reg(f"{_lf_name}-metrics.txt", _lf_m.out_metrics)
        # also the chunk-assignment metrics: bucket the boundaries into the same cs30
        # non-overlapping grid as the DP experiments, then the shared chunk metric job
        _lf_ca = ChunkAssignmentFromWordBoundariesJob(
            dataset_dir=_lf_dir,
            dataset_key="val",
            word_boundaries_hdf=_wb_hdf,
            chunk_size_secs=30.0,
            chunk_overlap_secs=0.0,
        )
        _lf_ca.add_alias(f"{_lf_name}-chunkassign")
        _lf_cm = CalcChunkAssignmentMetricsJob(
            chunk_seg_hdf=_lf_ca.out_hdf,
            dataset_dir=_lf_dir,
            dataset_key="val",
            dataset_offset_factors=_lf_off,
        )
        _lf_cm.add_alias(f"{_lf_name}-chunkassign-metric")
        reg(f"{_lf_name}-chunk-accuracy.txt", _lf_cm.out_accuracy)
        reg(f"{_lf_name}-chunk-error-p95-sec.txt", _lf_cm.out_error_p95_sec)
        reg(f"{_lf_name}-chunk-frac-gt-1s.txt", _lf_cm.out_frac_gt_1s)

    # MMS-FA (torchaudio CTC forced alignment)
    _lf_fa = ForcedAlignBaselineJob(dataset_dir=_lf_dir, dataset_key="val")
    _lf_fa.add_alias("chunk-align/native-longform/mms-fa")
    _lf_metric(_lf_fa.out_hdf, "chunk-align/native-longform/mms-fa")

    # XLS-R phoneme-CTC forced alignment
    _lf_ph = ForcedAlignPhonemeBaselineJob(
        dataset_dir=_lf_dir,
        dataset_key="val",
        model_dir=dl_w2v_phoneme.out_hub_cache_dir,
        dataset_offset_factors=_lf_off,
        # G2P word targets, as in the grad-align Buckeye phoneme baseline
        # (the dataset phonetic_detail carries Buckeye labels beyond TIMIT61)
        g2p_word_targets=True,
        dump_word_boundaries=True,
    )
    _lf_ph.add_alias("chunk-align/native-longform/w2v-phoneme")
    _lf_metric(_lf_ph.out_word_boundaries_hdf, "chunk-align/native-longform/w2v-phoneme")

    # CTC posteriors forced alignment (torchaudio Viterbi on the model's own emission)
    _lf_pc = ParakeetCtcForcedAlignJob(
        dataset_dir=_lf_dir,
        dataset_key="val",
        model_dir=dl_parakeet_ctc.out_hub_cache_dir,
        overlay_path=_NEMO_OVERLAY,
        dataset_offset_factors=_lf_off,
    )
    _lf_pc.add_alias("chunk-align/native-longform/parakeet-ctc-1.1b")
    _lf_metric(_lf_pc.out_word_boundaries_hdf, "chunk-align/native-longform/parakeet-ctc-1.1b")

    _lf_ow = OwsmCtcForcedAlignJob(
        dataset_dir=_lf_dir,
        dataset_key="val",
        model_dir=dl_owsm_ctc.out_hub_cache_dir,
        dataset_offset_factors=_lf_off,
    )
    _lf_ow.add_alias("chunk-align/native-longform/owsm-ctc-v4-1b")
    _lf_metric(_lf_ow.out_word_boundaries_hdf, "chunk-align/native-longform/owsm-ctc-v4-1b")

    _lf_fcc = ParakeetCtcForcedAlignJob(
        dataset_dir=_lf_dir,
        dataset_key="val",
        model_config=rf.build_dict(
            FastConformerStreaming,
            model_dir=dl_fc_stream.out_hub_cache_dir,
            overlay_path=_NEMO_OVERLAY,
            head="ctc",
            att_context_size=_fc_att,
        ),
        dataset_offset_factors=_lf_off,
    )
    _lf_fcc.add_alias("chunk-align/native-longform/fastconformer-stream-ctc")
    _lf_metric(_lf_fcc.out_word_boundaries_hdf, "chunk-align/native-longform/fastconformer-stream-ctc")

    # Transducer native Viterbi forced alignment (model configs as in the grad-align recipe;
    # the T x U joint lattice on a 10-min track is the expected memory breaker)
    for _lf_nt_cfg, _lf_nt_name in [
        (
            rf.build_dict(
                ParakeetRnnt,
                model_dir=dl_parakeet_rnnt.out_hub_cache_dir,
                per_token_score="prefix",
                overlay_path=_NEMO_OVERLAY,
            ),
            "parakeet-rnnt-1.1b",
        ),
        (
            rf.build_dict(
                ParakeetRnnt,
                model_dir=dl_parakeet_tdt.out_hub_cache_dir,
                per_token_score="emission",
                overlay_path=_NEMO_OVERLAY,
            ),
            "parakeet-tdt-0.6b-v2",
        ),
        (
            rf.build_dict(
                FastConformerStreaming,
                model_dir=dl_fc_stream.out_hub_cache_dir,
                overlay_path=_NEMO_OVERLAY,
                head="rnnt",
                att_context_size=_fc_att,
            ),
            "fastconformer-stream-rnnt",
        ),
        (rf.build_dict(EmformerRnnt), "emformer-rnnt"),
    ]:
        _lf_nt = NativeTransducerAlignJob(dataset_dir=_lf_dir, dataset_key="val", model_config=_lf_nt_cfg)
        _lf_nt.add_alias(f"chunk-align/native-longform/{_lf_nt_name}-native-viterbi")
        _lf_metric(_lf_nt.out_word_boundaries_hdf, f"chunk-align/native-longform/{_lf_nt_name}-native-viterbi")

    # No Whisper cross-attention entry: long-form input is structurally impossible for
    # Whisper (learned decoder pos embeddings, 448 positions; encoder asserts 30s);
    # the table carries an authored note instead (verified crash: 1102 vs 448).

    # MFA (GMM-HMM) external reference; infra identical to the grad-align setup
    # (same hashes -> the image/wrapper/model downloads are imported, never rebuilt)
    _mfa_image = PullApptainerImageJob("docker://mmcauliffe/montreal-forced-aligner:latest")
    _mfa_exe = ApptainerExeWrapperJob(
        _mfa_image.out_image,
        command="mfa",
        bind=["/rwthfs/rz/cluster/home", "/rwthfs/rz/cluster/hpcwork/p0023999"],
    )
    _mfa_models = MfaDownloadModelJob(
        mfa_exe=_mfa_exe.out_exe,
        models=[("acoustic", "english_us_arpa"), ("dictionary", "english_us_arpa"), ("g2p", "english_us_arpa")],
    )
    _lf_mfa = MfaForcedAlignJob(
        dataset_dir=_lf_dir,
        dataset_key="val",
        mfa_exe=_mfa_exe.out_exe,
        model_root=_mfa_models.out_model_root,
        dataset_offset_factors=_lf_off,
        # long-form: MFA gave up on 1/46 tracks (u000016, the mumbled seq) despite retry_beam;
        # tolerate it (word-uniform fallback rows, true coverage reported) instead of hard-failing
        allow_failed_seqs=True,
    )
    _lf_mfa.add_alias("chunk-align/native-longform/mfa")
    _lf_metric(_lf_mfa.out_word_boundaries_hdf, "chunk-align/native-longform/mfa")
    reg("chunk-align/native-longform/mfa-coverage.txt", _lf_mfa.out_coverage)

    # MFA option probes on the interviewer-turn failure (seq5 etc., see project notes):
    # a wide initial beam tests the pruning hypothesis
    # (the beam-10 first pass may discard the stay-in-silence path),
    # boost_silence tests the model hypothesis
    # (the silence GMM loses against transcript-words-on-interviewer-speech).
    # the stay-in-silence path accumulates its deficit over the whole untranscribed prefix
    # (~82 s on seq5) before paying off, so the needed beam can be in the thousands
    for _lf_mfa_name, _lf_mfa_kw in [
        ("mfa-beam100", dict(beam=100, retry_beam=1000)),
        ("mfa-beam1000", dict(beam=1000, retry_beam=4000)),
        ("mfa-boost4", dict(boost_silence=4.0)),
        ("mfa-beam100-boost4", dict(beam=100, retry_beam=1000, boost_silence=4.0)),
    ]:
        _lf_mfa_v = MfaForcedAlignJob(
            dataset_dir=_lf_dir,
            dataset_key="val",
            mfa_exe=_mfa_exe.out_exe,
            model_root=_mfa_models.out_model_root,
            dataset_offset_factors=_lf_off,
            allow_failed_seqs=True,
            **_lf_mfa_kw,
        )
        # wide beams on 10-min utterances are slow (Kaldi CPU decode)
        _lf_mfa_v.rqmt = {**_lf_mfa_v.rqmt, "time": 24}
        _lf_mfa_v.add_alias(f"chunk-align/native-longform/{_lf_mfa_name}")
        _lf_metric(_lf_mfa_v.out_word_boundaries_hdf, f"chunk-align/native-longform/{_lf_mfa_name}")
        reg(f"chunk-align/native-longform/{_lf_mfa_name}-coverage.txt", _lf_mfa_v.out_coverage)

    # Speech-LLM decoder self-attention DTW + OWLS cross-attention DTW,
    # directly on the long-form tracks.
    # Head selection (TIMIT val, gold) is byte-identical to the grad-align recipe,
    # so the finished jobs are reused; extract + DP align are new.
    for _lf_sa_name, _lf_sa_kind, _lf_sa_cfg, _lf_sa_ups in [
        (
            "owls-1b-180k",
            "crossattn",
            rf.build_dict(Owls, model_dir=dl_owls_1b.out_hub_cache_dir, char_level=True),
            True,
        ),
        (
            "voxtral",
            "selfattn",
            rf.build_dict(
                Voxtral, model_dir=dl_voxtral, forward_mode="transcription", attn_implementation="eager", version=3
            ),
            False,
        ),
        (
            "canary-qwen",
            "selfattn",
            rf.build_dict(
                CanaryQwen, model_dir=dl_canary, llm_model_dir=dl_qwen3, attn_implementation="eager", version=3
            ),
            False,
        ),
        ("phi4mm", "selfattn", _phi4mm_model_config(dl_phi4mm_dir, attn_implementation="eager"), False),
        (
            "granite-speech-8b",
            "selfattn",
            rf.build_dict(GraniteSpeech, model_dir=dl_granite.out_hub_cache_dir, attn_implementation="eager"),
            False,
        ),
    ]:
        _lf_sa_sel = SelectSelfAttnAlignHeadsJob(
            dataset_dir=dl_ds_timit.out_hub_cache_dir,
            dataset_key="val",
            model_config=_lf_sa_cfg,
            time_upsample_when_short=_lf_sa_ups,
        )
        _lf_sa_ex = ExtractSelfAttnPerTokenJob(
            dataset_dir=_lf_dir,
            dataset_key="val",
            # collect_attn_heads: the wrapper captures only the selected heads' matrices
            # (plain output_attentions retains every layer's [H, L, L] and OOMs on 10-min tracks)
            model_config={**_lf_sa_cfg, "collect_attn_heads": _lf_sa_sel.out_heads},
            heads=_lf_sa_sel.out_heads,
        )
        _lf_sa_nm = f"chunk-align/native-longform/{_lf_sa_name}-{_lf_sa_kind}"
        _lf_sa_ex.add_alias(f"{_lf_sa_nm}-extract")
        _lf_sa_al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=_lf_sa_ex.out_hdf,
            grad_score_key="data",
            dataset_dir=_lf_dir,
            dataset_key="val",
            dataset_offset_factors=_lf_off,
            # the grad-align headline align opts (softmax over time, blank -5, en0.5, sil1.0)
            align_opts={"apply_softmax_over_time": True, "blank_score": -5},
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
        )
        _lf_sa_al.add_alias(_lf_sa_nm)
        _lf_metric(_lf_sa_al.out_word_boundaries_hdf, _lf_sa_nm)

    # === Baselines (same model, no DP): the DP ablation set. ===
    # proportional split (trivial floor), greedy exit threshold (no global objective),
    # free-decode + LCS stitching (the HF chunked-pipeline analogue).
    # Non-overlapping configs: LCS needs them (overlap duplicates hyp words),
    # and the plain cs30-ov0 / cs10-ov0 DP rows are the references.
    # NOT _cfg_hp: that carries grad_wrt=None (batched convention),
    # the single-seq-style forward asserts a grad target (cf. phi4mm-noctx).
    _cfg_bl = rf.build_dict(Phi4MM, model_dir=dl_phi4mm_dir, model_dtype="bfloat16")
    for _bl_cs, _bl_mnt in [(30.0, 400), (10.0, 200)]:
        _bl_jobs = {
            "proportional": ProportionalChunkAssignmentJob(
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                chunk_size_secs=_bl_cs,
                chunk_overlap_secs=0.0,
            ),
            "greedy-tau0.5": GreedyExitChunkSegmentationJob(
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                model_config=_cfg_bl,
                chunk_size_secs=_bl_cs,
                chunk_overlap_secs=0.0,
                exit_prob_threshold=0.5,
            ),
            "greedy-tau0.9": GreedyExitChunkSegmentationJob(
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                model_config=_cfg_bl,
                chunk_size_secs=_bl_cs,
                chunk_overlap_secs=0.0,
                exit_prob_threshold=0.9,
            ),
            "lcs": FreeDecodeLcsChunkSegmentationJob(
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                model_config=_cfg_bl,
                chunk_size_secs=_bl_cs,
                chunk_overlap_secs=0.0,
                max_new_tokens=_bl_mnt,
            ),
        }
        for _bl_name, _bl_job in _bl_jobs.items():
            _bl_full = f"chunk-align/baselines/phi4mm-buckeye-val-cs{_bl_cs:g}-ov0-{_bl_name}"
            _bl_job.add_alias(_bl_full)
            reg(f"{_bl_full}.hdf", _bl_job.out_hdf)
            _bl_m = CalcChunkAssignmentMetricsJob(
                chunk_seg_hdf=_bl_job.out_hdf,
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors=_DATASET_OFFSET_FACTORS["buckeye"],
            )
            _bl_m.add_alias(f"{_bl_full}-metric")
            reg(f"{_bl_full}-accuracy.txt", _bl_m.out_accuracy)
            reg(f"{_bl_full}-error-p95-sec.txt", _bl_m.out_error_p95_sec)
            reg(f"{_bl_full}-frac-gt-1s.txt", _bl_m.out_frac_gt_1s)

    # === Tables (data): resolve registered outputs into tables-data/. ===
    # Presentation (headers/units/captions) lives in separate repo
    # (tables-spec/ + scripts/render_tables.py); scripts/sync_tables.sh rsyncs the JSONs.
    from i6_experiments.users.zeyer.utils.table_data import WriteTableDataJob, write_preview_manifest

    def _table(name: str, columns, rows):
        _tj = WriteTableDataJob(columns=list(columns), rows=rows)
        reg(f"tables-data/{name}.data.json", _tj.out_json)
        reg(f"tables-data/{name}.tsv", _tj.out_tsv)
        # live-preview manifest: `table_data.py --refresh-preview` re-resolves it from disk
        # any time (pending cells -> a placeholder glyph, a finished table job wins),
        # see the paper repo's scripts/sync_tables.sh
        write_preview_manifest(name, list(columns), rows, "output/tables-data-preview")

    def _m3(base: str):
        # the three headline metrics of a metric job, by registered-output name
        # (.get: a missing registration becomes a null cell, not a config error)
        return {
            "acc": _table_results.get(f"{base}-accuracy.txt"),
            "err_p95_sec": _table_results.get(f"{base}-error-p95-sec.txt"),
            "frac_gt_1s": _table_results.get(f"{base}-frac-gt-1s.txt"),
        }

    # family + display name per zoo model, matching the grad-align paper's per-model table;
    # third entry = the model's native long-form aligner:
    # a short name under chunk-align/native-longform/, or ("note", text) = authored literal
    _whisper_lf_note = "input too long for the model \\\\ (learned decoder pos. embeddings)"
    _zoo_family = {
        "mms-fa": ("CTC", "MMS-FA", "mms-fa"),
        "w2v-phoneme": ("CTC", "XLS-R (Phoneme)", "w2v-phoneme"),
        "parakeet-ctc-1.1b": ("CTC", "Parakeet CTC", "parakeet-ctc-1.1b"),
        "owsm-ctc-v4-1b": ("CTC", "OWSM-CTC", "owsm-ctc-v4-1b"),
        "fastconformer-stream-ctc": ("CTC", "FastConformer (streaming)", "fastconformer-stream-ctc"),
        # parakeet first, so the two identical FastConformer cells are not adjacent
        # (auto_merge would otherwise fuse them across the CTC/Transd. boundary)
        "parakeet-rnnt-1.1b": ("Transd.", "Parakeet RNN-T", "parakeet-rnnt-1.1b-native-viterbi"),
        "parakeet-tdt-0.6b-v2": ("Transd.", "Parakeet TDT", "parakeet-tdt-0.6b-v2-native-viterbi"),
        "fastconformer-stream-rnnt": (
            "Transd.",
            "FastConformer (streaming)",
            "fastconformer-stream-rnnt-native-viterbi",
        ),
        "emformer-rnnt": ("Transd.", "Emformer (streaming)", "emformer-rnnt-native-viterbi"),
        "whisper-base": ("AED", "Whisper-base", ("note", _whisper_lf_note)),
        "whisper-large-v3": ("AED", "Whisper-large-v3", ("note", _whisper_lf_note)),
        "crisperwhisper": ("AED", "CrisperWhisper", ("note", _whisper_lf_note)),
        "owls-1b-180k": ("AED", "OWLS-1B", "owls-1b-180k-crossattn"),
        # verified: decoder budget 1024 learned positions, a 10-min transcript needs ~4000
        "canary-1b-flash": (
            "AED",
            "Canary-1B-Flash",
            ("note", "input too long for the model \\\\ (learned decoder pos., 1024 tokens)"),
        ),
        "voxtral": ("Speech LLM", "Voxtral", "voxtral-selfattn"),
        "canary-qwen": ("Speech LLM", "Canary-Qwen", "canary-qwen-selfattn"),
        "granite-speech-8b": ("Speech LLM", "Granite-Speech-8B", "granite-speech-8b-selfattn"),
    }

    _nat_keys = ["nat_wbe", "nat_acc50", "nat_acc", "nat_err_p95", "nat_frac_gt_1s"]

    def _nat4(_t_n):
        """Native long-form metric cells, see the ``_zoo_family`` third-entry forms."""
        if _t_n is None:
            return dict.fromkeys(_nat_keys)
        if isinstance(_t_n, tuple):
            _t_kind, _t_txt = _t_n
            assert _t_kind == "note", _t_n
            return dict.fromkeys(_nat_keys, _t_txt)
        return {
            "nat_wbe": _table_results.get(f"chunk-align/native-longform/{_t_n}-wbe.txt"),
            "nat_acc50": _table_results.get(f"chunk-align/native-longform/{_t_n}-acc50.txt"),
            "nat_acc": _table_results.get(f"chunk-align/native-longform/{_t_n}-chunk-accuracy.txt"),
            "nat_err_p95": _table_results.get(f"chunk-align/native-longform/{_t_n}-chunk-error-p95-sec.txt"),
            "nat_frac_gt_1s": _table_results.get(f"chunk-align/native-longform/{_t_n}-chunk-frac-gt-1s.txt"),
        }

    _zoo_rows = (
        [
            {
                "family": "Speech LLM",
                "model": "Phi-4-MM",
                **_m3("chunk-align/phi4mm-buckeye-val-cs30-ov0"),
                **_nat4("phi4mm-selfattn"),
            }
        ]
        + [
            {"family": _zf, "model": _zd, **_m3(f"chunk-align/zoo/{_zn}-buckeye-val-cs30-ov0"), **_nat4(_zln)}
            for _zn, (_zf, _zd, _zln) in _zoo_family.items()
        ]
        + [
            # MFA aligns the long-form tracks directly but is no chunk-scoring model,
            # so the chunk-align side stays empty
            # best-effort MFA (beam 1000/4000, full coverage); default beams derail on long-form,
            # see the mfa vs mfa-beam* probe outputs and the paper prose
            {
                "family": "GM-HMM",
                "model": "MFA",
                "acc": "--",
                "err_p95_sec": "--",
                "frac_gt_1s": "--",
                **_nat4("mfa-beam1000"),
            }
        ]
    )
    # contiguous family blocks (stable within-family order; phi4mm groups with the speech LLMs)
    _zoo_fam_order = ["CTC", "Transd.", "AED", "Speech LLM", "GM-HMM"]
    _zoo_rows.sort(key=lambda _r: _zoo_fam_order.index(_r["family"]))
    _table(
        "zoo",
        ["family", "model", "acc", "err_p95_sec", "nat_wbe", "nat_acc50", "nat_acc", "nat_err_p95"],
        _zoo_rows,
    )
    # one context-ablation pair per family with a label context to ablate
    # (CTC has no label state at all -> stated in the caption, no rows)
    _ctx_rows = [
        ("Transd.", "Parakeet RNN-T", "none", "chunk-align/zoo/parakeet-rnnt-1.1b-buckeye-val-cs30-ov0"),
        (
            "Transd.",
            "Parakeet RNN-T",
            "prev labels",
            "chunk-align/zoo/parakeet-rnnt-1.1b-prevctx-buckeye-val-cs30-ov0",
        ),
        ("Transd.", "Parakeet TDT", "none", "chunk-align/zoo/parakeet-tdt-0.6b-v2-buckeye-val-cs30-ov0"),
        (
            "Transd.",
            "Parakeet TDT",
            "prev labels",
            "chunk-align/zoo/parakeet-tdt-0.6b-v2-prevctx-buckeye-val-cs30-ov0",
        ),
        ("AED", "Whisper-large-v3", "prev text", "chunk-align/zoo/whisper-large-v3-buckeye-val-cs30-ov0"),
        ("AED", "Whisper-large-v3", "none", "chunk-align/zoo/whisper-large-v3-noprev-buckeye-val-cs30-ov0"),
        ("Speech LLM", "Phi-4-MM", "marker", "chunk-align/phi4mm-buckeye-val-cs30-ov0"),
        ("Speech LLM", "Phi-4-MM", "none", "chunk-align/zoo/phi4mm-noctx-buckeye-val-cs30-ov0"),
        ("Speech LLM", "Voxtral", "marker", "chunk-align/zoo/voxtral-buckeye-val-cs30-ov0"),
        ("Speech LLM", "Voxtral", "none", "chunk-align/zoo/voxtral-noctx-buckeye-val-cs30-ov0"),
    ]
    _table(
        "context-ablation",
        ["family", "model", "context", "acc", "frac_gt_1s"],
        [{"family": _t_f, "model": _t_d, "context": _t_c, **_m3(_t_b)} for _t_f, _t_d, _t_c, _t_b in _ctx_rows],
    )
    _table(
        "exit-scale",
        ["chunk_size", "chunk_stride", "exit_scale", "word_start_heuristic", "acc", "frac_gt_1s"],
        [
            {
                "chunk_size": _t_cs,
                "chunk_stride": _t_cs - _t_ov,
                "exit_scale": _t_es,
                "word_start_heuristic": int(_t_wsh),
                **_m3(
                    f"chunk-align/phi4mm-buckeye-val-cs{_t_cs:g}-ov{_t_ov:g}"
                    if (_t_es == 1.0 and _t_wsh)
                    else f"chunk-align/exit-scale/phi4mm-buckeye-val-cs{_t_cs:g}-ov{_t_ov:g}-es{_t_es:g}-wsh{int(_t_wsh)}"
                ),
            }
            for _t_cs, _t_ov in [(30.0, 0.0), (10.0, 2.5)]
            for _t_es in [1.0, 0.5, 0.0]
            for _t_wsh in [True, False]
        ],
    )
    # chunk-size sweep (non-overlapping, eep=-5 default throughout;
    # the cs0.5 registered name is "cs0" via the plain loop's {:.0f} formatting)
    _table(
        "chunk-size",
        ["chunk_size", "acc", "err_p95_sec", "frac_gt_1s"],
        [
            {"chunk_size": _t_cs, **_m3(f"chunk-align/phi4mm-buckeye-val-cs{_t_cs:.0f}-ov0")}
            for _t_cs in [30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5]
        ],
    )
    # chunk-stride sweep across chunk sizes, uniform eep=0 (the overlap-correct setting;
    # only accuracy is registered for these cells).
    # cells = every eep=0 (cs, ov) combination above; cs10-ov2.5 comes from the hp sweep
    # (its name carries -wsh1); cs0.5's registered name is "cs0" via the {:.0f} formatting
    _stride_cells = [
        (30.0, [(0.0, ""), (5.0, ""), (15.0, "")]),
        (20.0, [(0.0, ""), (10.0, "")]),
        (10.0, [(0.0, ""), (1.0, ""), (2.5, "-wsh1"), (5.0, "")]),
        (5.0, [(0.0, ""), (2.5, "")]),
        (3.0, [(0.0, ""), (1.5, "")]),
        (2.0, [(0.0, ""), (1.0, "")]),
        (1.0, [(0.0, ""), (0.25, ""), (0.5, ""), (0.75, "")]),
        (0.5, [(0.0, "")]),
    ]
    _table(
        "chunk-stride",
        ["chunk_size", "chunk_stride", "acc"],
        [
            {
                "chunk_size": _t_cs,
                "chunk_stride": _t_cs - _t_ov,
                "acc": _table_results.get(
                    f"chunk-align/phi4mm-buckeye-val-cs{_t_cs:.0f}-ov{_t_ov:g}-eep0{_t_sfx}-accuracy.txt"
                ),
            }
            for _t_cs, _t_ovs in _stride_cells
            for _t_ov, _t_sfx in _t_ovs
        ],
    )
    # empty-exit penalty sweep cells (registered with accuracy only)
    _eep_cells = [
        (30.0, 5.0, [0.0, -5.0, -20.0], ""),
        (10.0, 5.0, [0.0, -2.0, -5.0, -10.0, -20.0], ""),
        (10.0, 2.5, [0.0, -2.0, -5.0, -10.0, -20.0], "-wsh1"),
        (2.0, 0.0, [0.0, -2.0, -5.0, -10.0, -20.0], ""),
        (1.0, 0.5, [0.0, -5.0], ""),
        (0.5, 0.0, [0.0, -5.0], ""),
    ]
    _table(
        "empty-exit-penalty",
        ["chunk_size", "chunk_stride", "eep", "acc"],
        [
            {
                "chunk_size": _t_cs,
                "chunk_stride": _t_cs - _t_ov,
                "eep": _t_e,
                "acc": _table_results.get(
                    f"chunk-align/phi4mm-buckeye-val-cs{_t_cs:.0f}-ov{_t_ov:g}-eep{_t_e:g}{_t_sfx}-accuracy.txt"
                ),
            }
            for _t_cs, _t_ov, _t_eeps, _t_sfx in _eep_cells
            for _t_e in _t_eeps
        ],
    )
    # word-start pruning (heuristic) vs the exact DP; the cs30-ov0 exact cell
    # comes from the exit-scale sweep (es1-wsh0)
    _prune_cells = [
        (
            30.0,
            0.0,
            "chunk-align/phi4mm-buckeye-val-cs30-ov0",
            "chunk-align/exit-scale/phi4mm-buckeye-val-cs30-ov0-es1-wsh0",
        ),
        (
            10.0,
            2.5,
            "chunk-align/phi4mm-buckeye-val-cs10-ov2.5",
            "chunk-align/phi4mm-buckeye-val-cs10-ov2.5-eep-5-wsh0",
        ),
        (
            2.0,
            0.0,
            "chunk-align/phi4mm-buckeye-val-cs2-ov0",
            "chunk-align/phi4mm-buckeye-val-cs2-ov0-eep-5-wsh0",
        ),
        (
            1.0,
            0.0,
            "chunk-align/phi4mm-buckeye-val-cs1-ov0",
            "chunk-align/phi4mm-buckeye-val-cs1-ov0-eep-5-wsh0",
        ),
    ]
    _table(
        "pruning",
        ["chunk_size", "chunk_stride", "search", "acc"],
        [
            {
                "chunk_size": _t_cs,
                "chunk_stride": _t_cs - _t_ov,
                "search": _t_lbl,
                "acc": _table_results.get(_t_b + "-accuracy.txt"),
            }
            for _t_cs, _t_ov, _t_bp, _t_bx in _prune_cells
            for _t_lbl, _t_b in [("pruned", _t_bp), ("exact", _t_bx)]
        ],
    )
    # tail-repair layers on the cs30-ov0 reference assignment
    _table(
        "repair",
        ["variant", "acc", "err_p95_sec", "frac_gt_1s"],
        [
            {"variant": _t_lbl, **_m3(_t_b)}
            for _t_lbl, _t_b in [
                ("none", "chunk-align/phi4mm-buckeye-val-cs30-ov0"),
                ("boundary reverify", "chunk-align/phi4mm-buckeye-val-cs30-ov0-reverify-m2"),
                ("drift-span repair", "chunk-align/phi4mm-buckeye-val-cs30-ov0-driftrepair"),
                ("staggered grid", "chunk-align/phi4mm-buckeye-val-cs30-ov0-offset15"),
                ("char-level scoring", "chunk-align/phi4mm-buckeye-val-cs30-ov0-charlevel"),
            ]
        ],
    )
    # Native-aligner degradation, utterance-level (segmented Buckeye val, segA) -> raw long-form.
    # Segmented WBE = the finished grad-align setup outputs, referenced by absolute path
    # (WriteTableDataJob reads Path cells; re-instantiating those job graphs here is not worth it).
    _ga_out = "/home/az668407/setups/2026-05-23-grad-align/output"
    _sa_sfx = "selfattn-buckeye-segA-5h-asotTrue-bs-5-en0.5-sil1.0-wbe.txt"  # under output/align/
    _deg_rows = [
        ("CTC", "MMS-FA", "baseline-mms_fa-buckeye-segA-5h-wbe.txt", "mms-fa"),
        ("CTC", "XLS-R (Phoneme)", "baseline-phoneme-fa-buckeye-segA-5h-word-wbe.txt", "w2v-phoneme"),
        ("CTC", "Parakeet CTC", "baseline-parakeet-ctc-1.1b-buckeye-segA-5h-wbe.txt", "parakeet-ctc-1.1b"),
        ("CTC", "OWSM-CTC", "baseline-owsm-ctc-v4-1b-buckeye-segA-5h-wbe.txt", "owsm-ctc-v4-1b"),
        (
            "CTC",
            "FastConformer (streaming)",
            "baseline-fastconformer-stream-ctc-buckeye-segA-5h-wbe.txt",
            "fastconformer-stream-ctc",
        ),
        (
            "Transd.",
            "Parakeet RNN-T",
            "baseline-parakeet-rnnt-1.1b-native-viterbi-buckeye-segA-5h-wbe.txt",
            "parakeet-rnnt-1.1b-native-viterbi",
        ),
        (
            "Transd.",
            "Parakeet TDT",
            "baseline-parakeet-tdt-0.6b-v2-native-viterbi-buckeye-segA-5h-wbe.txt",
            "parakeet-tdt-0.6b-v2-native-viterbi",
        ),
        (
            "Transd.",
            "FastConformer (streaming)",
            "baseline-fastconformer-stream-rnnt-native-viterbi-buckeye-segA-5h-wbe.txt",
            "fastconformer-stream-rnnt-native-viterbi",
        ),
        (
            "Transd.",
            "Emformer (streaming)",
            "baseline-emformer-rnnt-native-viterbi-buckeye-segA-5h-wbe.txt",
            "emformer-rnnt-native-viterbi",
        ),
        (
            "AED",
            "OWLS-1B",
            "align/baseline-owls-1B-180K-crossattn-auto-buckeye-segA-5h-asotTrue-bs-5-en0.5-sil1.0-wbe.txt",
            "owls-1b-180k-crossattn",
        ),
        ("Speech LLM", "Phi-4-MM", f"align/baseline-phi4mm-{_sa_sfx}", "phi4mm-selfattn"),
        ("Speech LLM", "Voxtral", f"align/baseline-voxtral-{_sa_sfx}", "voxtral-selfattn"),
        ("Speech LLM", "Canary-Qwen", f"align/baseline-canary-qwen-{_sa_sfx}", "canary-qwen-selfattn"),
        ("GM-HMM", "MFA", "baseline-mfa-buckeye-segA-5h-wbe.txt", "mfa-beam1000"),
    ]
    _table(
        "longform-degradation",
        ["family", "model", "wbe_seg", "wbe_lf"],
        [
            {
                "family": _t_f,
                "model": _t_d,
                "wbe_seg": tk.Path(f"{_ga_out}/{_t_seg}"),
                "wbe_lf": _table_results.get(f"chunk-align/native-longform/{_t_n}-wbe.txt"),
            }
            for _t_f, _t_d, _t_seg, _t_n in _deg_rows
        ],
    )

    # DP vs the no-DP baselines (same model, same metric)
    _table(
        "baselines",
        ["chunk_size", "method", "acc", "err_p95_sec", "frac_gt_1s"],
        [
            {"chunk_size": _t_cs, "method": _t_lbl, **_m3(_t_b)}
            for _t_cs in [30.0, 10.0]
            for _t_lbl, _t_b in [
                ("DP (ours)", f"chunk-align/phi4mm-buckeye-val-cs{_t_cs:.0f}-ov0"),
                ("proportional split", f"chunk-align/baselines/phi4mm-buckeye-val-cs{_t_cs:g}-ov0-proportional"),
                ("greedy exit (tau 0.5)", f"chunk-align/baselines/phi4mm-buckeye-val-cs{_t_cs:g}-ov0-greedy-tau0.5"),
                ("greedy exit (tau 0.9)", f"chunk-align/baselines/phi4mm-buckeye-val-cs{_t_cs:g}-ov0-greedy-tau0.9"),
                ("free-decode + LCS", f"chunk-align/baselines/phi4mm-buckeye-val-cs{_t_cs:g}-ov0-lcs"),
            ]
        ],
    )
