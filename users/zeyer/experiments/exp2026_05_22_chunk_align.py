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
