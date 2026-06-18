"""
BEST-RQ self-supervised pretraining on LibriSpeech.

* ``bestrq_smoke()`` -- tiny end-to-end validation run (dev-clean as train), to confirm the
  pipeline trains and the loss/codebook metrics show up in the log before scaling.
* ``bestrq_ls960_base()`` -- the real 12x512 pretraining on LS960 (~150k updates target).

Cluster: 4x GH200, per-step gradient sync (``torch_distributed={"reduce_type":"grad"}``), no grad
accumulation, bf16. Data read offline from the local HF parquet (no cache manager, no node disk).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List

import numpy as np

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ...config import get_training_config
from ...pipeline import training, register_train_summary
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
from ...data import datasets as ds
from ...pytorch_networks.best_rq.best_rq_v1_cfg import BestRQConfig
from ...pytorch_networks.common.conformer import default_encoder_config
from ...pytorch_networks.best_rq.ls_logmel_stats import LOGMEL_MEAN, LOGMEL_STD

NETWORK_MODULE = "best_rq.best_rq_v1"


def build_model_config(
    *,
    num_layers: int = 12,
    conformer_size: int = 512,
    num_heads: int = 8,
    ff_dim: int = 2048,
    num_codebooks: int = 4,
    codebook_dim: int = 16,
    vocab_size: int = 8192,
    mask_prob: float = 0.04,
    mask_length: int = 10,
) -> BestRQConfig:
    feature_extraction_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=False,
    )
    encoder_config = default_encoder_config(
        conformer_size=conformer_size,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
    )
    return BestRQConfig(
        feature_extraction_config=feature_extraction_config,
        encoder_config=encoder_config,
        global_mean=list(LOGMEL_MEAN),
        global_std=list(LOGMEL_STD),
        stack_size=4,
        codebook_dim=codebook_dim,
        vocab_size=vocab_size,
        num_codebooks=num_codebooks,
        quantizer_seed=42,
        mask_prob=mask_prob,
        mask_length=mask_length,
        min_masks=1,
        noise_std=0.1,
    )


def _oclr(
    peak: float,
    num_epochs: int,
    warmup_frac: float = 0.1,
    init: float = 1e-5,
    final: float = 1e-7,
) -> List[float]:
    n_warmup = max(1, int(num_epochs * warmup_frac))
    n_decay = num_epochs - n_warmup
    lrs = list(np.linspace(init, peak, n_warmup))
    if n_decay > 0:
        lrs += list(np.linspace(peak, final, n_decay))
    return [float(x) for x in lrs[:num_epochs]] or [peak]


def _train_config(
    *,
    peak_lr: float,
    num_epochs: int,
    batch_size_sec: int,
    num_processes: int,
    grad_clip: float = 5.0,
) -> dict:
    return {
        "behavior_version": 21,
        "extern_data": ds.extern_data_audio(),
        "optimizer": {
            "class": "adamw",
            "weight_decay": 0.01,
            "epsilon": 1e-8,
            "betas": (0.9, 0.98),
        },
        "learning_rates": _oclr(peak_lr, num_epochs),
        # per-step DDP gradient sync (NOT param averaging), no gradient accumulation:
        "torch_distributed": {"reduce_type": "grad"},
        "accum_grad_multiple_step": 1,
        # loose global-norm clip: a safety net against rare spikes, not a throttle (1.0 was too
        # conservative for a 94M conformer). Refine from the logged `grad_norm:p2` (see config.py).
        "gradient_clip_global_norm": grad_clip,
        # bf16 has fp32's exponent range -> no loss scaling needed. grad_scaler MUST be None: if
        # omitted, RETURNN auto-creates a GradScaler whenever autocast is on (engine.py), which is
        # pointless for bf16 and can skip steps on spurious inf/nan checks.
        "torch_amp": {"dtype": "bfloat16", "grad_scaler": None},
        "batch_size": batch_size_sec * 16000,
        "max_seq_length": {"audio": 35 * 16000},
        "max_seqs": 200,
        "torch_dataloader_opts": {"num_workers": 2},
    }


def bestrq_smoke():
    """Validation + sizing run: real 12x512 model on 4 GPUs, train on dev-clean / eval dev-other.

    Runs at the real candidate batch (300 s/GPU) so the logged peak GPU memory and ``grad_norm:p2``
    transfer to the real LS960 run (used to set batch_size and the grad clip there)."""
    prefix = "ssl/pretrain_bestrq/smoke"
    num_epochs = 3
    model_config = build_model_config()
    train_set = ds.audio_hf_dataset(ds.DEV_CLEAN, seq_ordering="random", partition_epoch=1)
    dev_set = ds.audio_hf_dataset(ds.DEV_OTHER, seq_ordering="default")
    returnn_config = get_training_config(
        train_dataset=train_set,
        dev_dataset=dev_set,
        network_module=NETWORK_MODULE,
        net_args={"model_config_dict": asdict(model_config)},
        train_step_args={},
        config=_train_config(peak_lr=1e-4, num_epochs=num_epochs, batch_size_sec=300, num_processes=4),
    )
    return training(
        prefix,
        returnn_config,
        num_epochs=num_epochs,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
        num_processes=4,
        gpu_mem=96,
        time_rqmt=4,
        mem_rqmt=100,
        cpu_rqmt=16,
    )


def _bestrq_ls960(prefix: str, *, mask_prob: float, title: str):
    """Real BEST-RQ pretraining on LS960 (parameterized by masking `mask_prob`).

    Budget = 100 passes over LS960. partition_epoch=1 => one sub-epoch == one full pass (~1976
    steps/pass at the padding-limited batch, ~12 min), giving frequent checkpoints for the 11.5h
    Slurm cap + auto-resume. ~1976 steps/pass x 100 ~= 198k optimizer updates. LR warmup 10%
    (10 passes 1e-5->1e-3), then linear anneal over 90 passes -> 1e-7.

    Why 100 (not 200): the SSL loss plateaus hard (96% of the dev-CE drop by ~ep50); academic BEST-RQ
    at this exact 12x512/LS960 regime uses ~46 (Open-BEST-RQ, "sufficient") to ~104 (Optimized-BEST-RQ)
    passes. 200 passes was ~2x past the knee for a 100h-finetune target. Note budgets compare on PASSES
    (= audio = compute), NOT updates: 100 passes here = 96,000h of audio. See [[bestrq-pretrain-budget]].

    `mask_prob` is the per-frame span-START probability; with mask_length=10 the expected masked
    fraction is 1-(1-p)^10 (spans overlap). 0.04 -> ~33.5% coverage; 0.065 -> ~49% (the validated
    wav2vec2/HuBERT operating point). Only this knob varies across the A/B; everything else identical.
    """
    num_epochs = 100  # per-rank seq sharding => 1 epoch == 1 disjoint pass over LS960 (see note)
    model_config = build_model_config(mask_prob=mask_prob)
    # Per-rank SEQUENCE-level DDP sharding (ddp_seq_shard): the 4 GPUs iterate disjoint quarters of the
    # shuffled order, so one RETURNN epoch is ONE 960h pass (not 4, as an unsharded HF dataset gives).
    # 100 epochs == 100 passes; at batch 1200 s/GPU the effective batch is padding-limited to ~440 s
    # (random ordering, no duration column) -> ~1976 steps/pass -> ~198k updates.
    train_set = ds.audio_hf_dataset(ds.TRAIN_960H, seq_ordering="random", ddp_seq_shard=True)
    dev_set = ds.audio_hf_dataset(ds.DEV_ALL, seq_ordering="default")  # eval stays unsharded
    # Sizing from the smoke: ~30 GB/GPU at 1200 s (random-order padding keeps batches ~35 seqs), well
    # under 96 GB. grad clip 5.0 is a safety net (smoke grad_norm:p2 maxed at 3.79). peak_lr 1e-3.
    returnn_config = get_training_config(
        train_dataset=train_set,
        dev_dataset=dev_set,
        network_module=NETWORK_MODULE,
        net_args={"model_config_dict": asdict(model_config)},
        train_step_args={},
        config=_train_config(peak_lr=1e-3, num_epochs=num_epochs, batch_size_sec=1200, num_processes=4),
    )
    train_job = training(
        prefix,
        returnn_config,
        num_epochs=num_epochs,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
        num_processes=4,
        gpu_mem=96,
        time_rqmt=24,
        mem_rqmt=100,
        cpu_rqmt=16,
    )
    # BEST-RQ has no labels -> no WER recog; the per-experiment summary reports the pretraining scores
    # (bestrq_ce / masked_acc / code_ppl per epoch from out_learning_rates). Downstream-only -> no re-run.
    register_train_summary(prefix, train_job, title=title)
    return train_job


def bestrq_ls960_base():
    """Baseline pretraining: mask_prob=0.04 (~33.5% masked-frame coverage)."""
    return _bestrq_ls960(
        "ssl/pretrain_bestrq/ls960_12x512_n4",
        mask_prob=0.04,
        title="BEST-RQ pretraining",
    )


def bestrq_ls960_mask49():
    """A/B variant: mask_prob=0.065 (~49% coverage, the wav2vec2/HuBERT operating point).

    Identical to ``bestrq_ls960_base`` except for the higher masking probability, so the comparison
    isolates 'does moving from ~33.5% to ~49% masked-frame coverage improve the SSL encoder'. Runs
    alongside the baseline (distinct prefix -> distinct job hash), with its own summary report.
    """
    return _bestrq_ls960(
        "ssl/pretrain_bestrq/ls960_12x512_n4_mask49",
        mask_prob=0.065,
        title="BEST-RQ pretraining (mask~49%)",
    )
