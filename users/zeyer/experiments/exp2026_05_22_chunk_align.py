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
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.phi4mm import Phi4MM
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.chunk_segmentation import (
    ChunkSegmentationFromModelJob,
    ChunkSegmentationFromModelBatchedJob,
    ChunkBoundaryReverifyJob,
    CalcChunkAssignmentMetricsJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.buckeye_fine_dataset import (
    MapBuckeyeFineTimestampsToLongFormJob,
)

# Same convention as exp2025_05_05_align.py / exp2026_05_23_grad_align.py.
_DATASET_OFFSET_FACTORS = {"timit": 1, "buckeye": 1000}

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
