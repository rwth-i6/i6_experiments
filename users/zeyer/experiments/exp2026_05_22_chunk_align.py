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
    CalcChunkAssignmentMetricsJob,
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
            for chunk_size_secs in [10.0, 20.0, 30.0]:
                seg_name = f"chunk-align/{model_name}-{ds_name}-{ds_key}-cs{chunk_size_secs:.0f}"
                seg = ChunkSegmentationFromModelJob(
                    dataset_dir=ds_dir,
                    dataset_key=ds_key,
                    model_config=model_cfg,
                    chunk_size_secs=chunk_size_secs,
                )
                seg.add_alias(seg_name)
                reg(f"{seg_name}.hdf", seg.out_hdf)

                metric = CalcChunkAssignmentMetricsJob(
                    chunk_seg_hdf=seg.out_hdf,
                    dataset_dir=ds_dir,
                    dataset_key=ds_key,
                    dataset_offset_factors=ds_offset,
                )
                metric.add_alias(f"{seg_name}-metric")
                reg(f"{seg_name}-accuracy.txt", metric.out_accuracy)
                reg(f"{seg_name}-chunk_idx_mae.txt", metric.out_chunk_idx_mae)
