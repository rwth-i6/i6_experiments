"""
Two-level BEST-RQ + CIF SSL pretraining on LibriSpeech 960h.

Builds on a FROZEN pretrained BEST-RQ encoder (lower encoder = its first 9 conformer layers, layer-9
output), adds a trainable CIF segmenter/pooler, a 9-layer high/global Conformer over the CIF tokens, and
a masked-prediction head against a FROZEN offline k-means codebook over frame-level layer-9 features
(ONE codebook shared by both rate arms; see [[cif-length-control-and-finetune]] for why frame-level).

Stage-2 only (the BEST-RQ pretraining is stage 1, reused as-is): we train the CIF predictor, the high
encoder, the mask embedding and the prediction head; the lower encoder + log-mel are frozen.
See memory [[cif-length-control-and-finetune]] / [[cif-segmenter-eval-metrics]] for the full design.

Launch (once a base BEST-RQ checkpoint exists):
    source /e/project1/spell/wu24/env/sis_env/bin/activate
    cd /e/project1/spell/wu24/2026-06-17_ssl
    PYTHONPATH=tools/sisyphus:recipe:recipe/i6_models:recipe/returnn \
      tools/sisyphus/sis --config config/ssl_two_level.py manager -r
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

import numpy as np

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ...config import get_training_config
from ...pipeline import training
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
from ...data import datasets as ds
from ...pytorch_networks.two_level.two_level_v1_cfg import TwoLevelConfig
from ...pytorch_networks.common.conformer import default_encoder_config, default_high_encoder_config
from ...pytorch_networks.best_rq.ls_logmel_stats import LOGMEL_MEAN, LOGMEL_STD
from ..pretrain_bestrq.baseline import _oclr, bestrq_ls960_base
from .kmeans import build_kmeans_codebook

NETWORK_MODULE = "two_level.two_level_v1"

# Derived rate constants (25 Hz lower encoder): token rate by token length.
RATE_80MS = 12.5   # 80 ms tokens  -> 2 frames/token
RATE_120MS = 25.0 / 3.0  # 120 ms tokens -> 3 frames/token (~8.333 Hz)
FRAME_RATE_HZ = 25.0  # codebook fit at the frame rate => window = round(25/25) = 1 => frame-level


def build_model_config(
    *,
    target_rate_hz: float = RATE_80MS,
    num_clusters: int = 128,
    mask_prob: float = 0.08,   # span-START prob; with mask_length=3 -> ~1-(1-.08)^3 ~= 22% coverage
    mask_length: int = 3,
    lambda_qty: float = 1.0,
    cif_kernel: int = 5,
    lower_layers: int = 9,
    high_layers: int = 9,
    rel_pos_clip: int = 96,
    high_dropout: float = 0.1,
) -> TwoLevelConfig:
    feature_extraction_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000, win_size=0.025, hop_size=0.01,
        f_min=60, f_max=7600, min_amp=1e-10, num_filters=80, center=False,
    )
    # Lower encoder: 9 layers so its state_dict keys (encoder.module_list.0..8) are a subset of the
    # 12-layer BEST-RQ ckpt -> preload 1:1; frozen + eval at runtime so its dropout value is irrelevant.
    encoder_config = default_encoder_config(num_layers=lower_layers)
    high_encoder_config = default_high_encoder_config(
        num_layers=high_layers, rel_pos_clip=rel_pos_clip, dropout=high_dropout
    )
    return TwoLevelConfig(
        feature_extraction_config=feature_extraction_config,
        encoder_config=encoder_config,
        high_encoder_config=high_encoder_config,
        global_mean=list(LOGMEL_MEAN),
        global_std=list(LOGMEL_STD),
        lower_layer_index=lower_layers - 1,  # 0-based: output AFTER the 9th layer
        cif_alpha_kernel_size=cif_kernel,
        target_rate_hz=target_rate_hz,
        frame_rate_hz=25.0,
        lambda_qty=lambda_qty,
        num_clusters=num_clusters,
        mask_prob=mask_prob,
        mask_length=mask_length,
        min_masks=1,
    )


def _train_config(*, peak_lr: float, num_epochs: int, batch_size_sec: int, grad_clip: float = 5.0) -> dict:
    """Audio-only SSL training config (mirrors the BEST-RQ recipe: bf16 no-scaler, per-step DDP grad sync)."""
    return {
        "behavior_version": 21,
        "extern_data": ds.extern_data_audio(),
        "optimizer": {"class": "adamw", "weight_decay": 0.01, "epsilon": 1e-8, "betas": (0.9, 0.98)},
        "learning_rates": _oclr(peak_lr, num_epochs),
        "torch_distributed": {"reduce_type": "grad"},
        "accum_grad_multiple_step": 1,
        "gradient_clip_global_norm": grad_clip,
        "torch_amp": {"dtype": "bfloat16", "grad_scaler": None},
        "batch_size": batch_size_sec * 16000,
        "max_seq_length": {"audio": 35 * 16000},
        "max_seqs": 200,
        "torch_dataloader_opts": {"num_workers": 2},
    }


def frame_kmeans_codebook(*, base_checkpoint, num_clusters: int, split=ds.TRAIN_CLEAN_100):
    """Build (once) the FROZEN frame-level k-means target codebook, shared across every rate arm.

    We cannot fit on the true CIF-pooled statistic (CIF is untrained when the codebook is built), so we
    fit on the frozen layer-9 FRAMES -- the content manifold itself, window-agnostic -- and assign each
    CIF-pooled token to its nearest frame centroid at train time. The fit-on-frames/assign-on-means
    utilization gap (the "are frames within a segment close enough" assumption) is monitored online via
    the ``code_ent`` diagnostic in ``train_step``. Identical (base ckpt, num_clusters) across the 80 ms and
    120 ms arms => identical job hash => Sisyphus builds it ONCE and both arms depend on it."""
    canon_config = build_model_config(target_rate_hz=FRAME_RATE_HZ, num_clusters=num_clusters)
    return build_kmeans_codebook(
        prefix="ssl/pretrain_two_level/kmeans_frame",
        base_checkpoint=base_checkpoint,
        model_config_dict=asdict(canon_config),
        target_rate_hz=FRAME_RATE_HZ,  # 25 Hz => window=1 => frame-level (arm-independent)
        num_clusters=num_clusters,
        split=split,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
    )


def two_level_assets(
    *,
    target_rate_hz: float = RATE_80MS,
    num_clusters: int = 128,
    base_epoch: int = 100,
    kmeans_split=ds.TRAIN_CLEAN_100,
):
    """The deduped ``(model_config, codebook)`` for one two-level arm -- the SINGLE source of truth shared
    by the pretraining job AND any downstream probe (e.g. ``analysis/seg_diag``), so their hashed net_args
    match exactly and Sisyphus reuses the same codebook job instead of rebuilding it."""
    model_config = build_model_config(target_rate_hz=target_rate_hz, num_clusters=num_clusters)
    base_ckpt = bestrq_ls960_base().out_checkpoints[base_epoch]
    # frozen FRAME-LEVEL target codebook from the same frozen lower stack -- ONE codebook for both rate
    # arms (it does not depend on target_rate_hz). Shared by job-hash dedup.
    codebook = frame_kmeans_codebook(base_checkpoint=base_ckpt, num_clusters=num_clusters, split=kmeans_split)
    return model_config, codebook


def two_level_pretrain(
    prefix: str,
    *,
    target_rate_hz: float = RATE_80MS,
    num_clusters: int = 128,
    base_epoch: int = 100,
    kmeans_split=ds.TRAIN_CLEAN_100,
    num_epochs: int = 50,
    peak_lr: float = 5e-4,
    batch_size_sec: int = 600,
):
    """Wire one two-level pretraining run on top of the frozen BEST-RQ base.

    1. base BEST-RQ pretraining (reused; we depend on its ``base_epoch`` checkpoint -- default the final);
    2. offline frozen k-means codebook over fixed-window-pooled layer-9 features (the target);
    3. two-level SSL training: encoder preloaded + frozen, train CIF + high encoder + head.
    """
    # (1)+(2) model config + frozen FRAME-LEVEL target codebook via the shared asset helper, so any
    # downstream probe (analysis/seg_diag) reconstructs byte-identical hashed args -> same codebook job.
    model_config, codebook = two_level_assets(
        target_rate_hz=target_rate_hz, num_clusters=num_clusters, base_epoch=base_epoch, kmeans_split=kmeans_split
    )
    base_ckpt = bestrq_ls960_base().out_checkpoints[base_epoch]  # frozen lower-stack preload source

    # (3) two-level training. Preload ONLY the shared frozen lower stack (encoder + feature_extraction +
    # global-norm buffers) from the BEST-RQ ckpt; CIF/high-encoder/head/codebook are fresh. ignore_missing
    # skips the ckpt's quantizer/heads/blocks-9..11 and our fresh params (the proven CTC-finetune preload).
    config = _train_config(peak_lr=peak_lr, num_epochs=num_epochs, batch_size_sec=batch_size_sec)
    config["preload_from_files"] = {
        "ssl_lower": {"filename": base_ckpt, "init_for_train": True, "ignore_missing": True}
    }
    returnn_config = get_training_config(
        train_dataset=ds.audio_hf_dataset(ds.TRAIN_960H, seq_ordering="random", ddp_seq_shard=True),
        dev_dataset=ds.audio_hf_dataset(ds.DEV_ALL, seq_ordering="default"),
        network_module=NETWORK_MODULE,
        net_args={"model_config_dict": asdict(model_config), "codebook": codebook},
        train_step_args={},
        config=config,
    )
    train_job = training(
        prefix, returnn_config, num_epochs=num_epochs,
        returnn_exe=RETURNN_EXE, returnn_root=RETURNN_ROOT,
        num_processes=4, gpu_mem=96, time_rqmt=24, mem_rqmt=100, cpu_rqmt=16,
    )
    # No summary.md registered here: like the BEST-RQ base, the per-experiment summary.md is filled
    # DOWNSTREAM by the CTC finetune (finetune_two_level -> register_multi_wer_summary on this prefix),
    # so the pretraining experiment surfaces its downstream WER as the headline -- NOT the SSL losses.
    # The per-epoch ce/qty/acc/fps_tok/code_ent stay in train_job.out_learning_rates for ad-hoc inspection.
    return train_job


def two_level_80ms_c128():
    """v1: 80 ms (12.5 Hz) CIF tokens, 128-cluster codebook, learned-CIF arm on the LS960 base.

    80 ms = 2 frames/token: a thin pooling between phoneme and sub-phoneme (the conservative point)."""
    return two_level_pretrain(
        "ssl/pretrain_two_level/ls960_cif80ms_k128",
        target_rate_hz=RATE_80MS, num_clusters=128,
    )


def two_level_120ms_c128():
    """v1: 120 ms (8.333 Hz) CIF tokens, 128-cluster codebook -- a coarser, genuinely higher-level unit
    than 80 ms (3 frames/token, ~phoneme/syllable scale). Own k-means codebook (fixed-window=3 frames)
    and own training job (distinct prefix => distinct hashes); runs alongside the 80 ms arm."""
    return two_level_pretrain(
        "ssl/pretrain_two_level/ls960_cif120ms_k128",
        target_rate_hz=RATE_120MS, num_clusters=128,
    )
