"""
CTC finetuning / from-scratch baseline on LibriSpeech train-clean-100 (SPM-5120, torch ctc_loss).

* ``ctc_ls100_scratch()`` -- from-scratch CTC baseline (no pretraining), the comparison point.
* ``ctc_ls100_finetune(ssl_checkpoint)`` -- same model, encoder initialized from a BEST-RQ
  checkpoint via ``preload_from_files`` (only the shared encoder + feature_extraction + global-norm
  buffers are loaded; the CTC head is fresh).

Shares the 12x512 rel-pos conformer + log-mel + global input-norm with the SSL model. 4 GPU, per-step
grad sync, bf16 (no scaler), no accum. WER via greedy decode + sclite is wired separately.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ...config import get_training_config
from ...pipeline import training, ctc_recog_all_epochs, ctc_recog_best_checkpoint, fraction_epochs
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
from ...data import datasets as ds
from ...data.spm import get_ls960_spm, SPM_VOCAB_SIZE
from ...pytorch_networks.ctc.conformer_ctc_v1_cfg import CTCConfig, SpecaugConfig
from ...pytorch_networks.common.conformer import default_encoder_config
from ...pytorch_networks.best_rq.ls_logmel_stats import LOGMEL_MEAN, LOGMEL_STD
from .. import pretrain_bestrq  # noqa: F401  (ensure package import)
from ..pretrain_bestrq.baseline import _oclr

NETWORK_MODULE = "ctc.conformer_ctc_v1"
DECODER_MODULE = "ctc.conformer_ctc_v1"  # SPM_VOCAB_SIZE imported from data.spm


def build_ctc_config(specaug_start_step: int = 5000, dropout: float = 0.1) -> CTCConfig:
    return CTCConfig(
        feature_extraction_config=LogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            f_min=60,
            f_max=7600,
            min_amp=1e-10,
            num_filters=80,
            center=False,
        ),
        encoder_config=default_encoder_config(dropout=dropout),
        global_mean=list(LOGMEL_MEAN),
        global_std=list(LOGMEL_STD),
        specaug_config=SpecaugConfig(repeat_per_n_frames=25, max_dim_time=20, num_repeat_feat=5, max_dim_feat=16),
        # step-based gate (partition_epoch=1 => 1 epoch == 1 pass, so an epoch gate means wildly different real
        # schedules across ls100/ls960; speech_llm gates on global step ~5000). See CTC-revisit study.
        specaug_start_step=specaug_start_step,
        # InterCTC: aux heads at layers 4 & 8 (scale 0.3) + the final layer-12 head (out_linear, scale 1.0)
        # == the canonical i6 12x512 config [3,7,11]@[0.3,0.3,1.0] (0-based). Prevents from-scratch blank collapse.
        aux_ctc_layers=[4, 8],
        aux_ctc_scales=[0.3, 0.3],
    )


def _ctc_train_config(
    *, peak_lr: float, num_epochs: int, batch_size_sec: int, vocab_size: int, grad_clip: float = 5.0
) -> dict:
    # adamw eps 1e-16 = the i6 conformer-CTC standard (CTC-revisit study). grad clip 5.0, betas (0.9,0.98) and
    # the OCLR warmup-init are kept at the existing values per the user's selection.
    return {
        "behavior_version": 21,
        "extern_data": ds.extern_data_audio_text(vocab_size),
        "optimizer": {"class": "adamw", "weight_decay": 0.01, "epsilon": 1e-16, "betas": (0.9, 0.98)},
        "learning_rates": _oclr(peak_lr, num_epochs),
        "torch_distributed": {"reduce_type": "grad"},
        "accum_grad_multiple_step": 1,
        "gradient_clip_global_norm": grad_clip,
        "torch_amp": {"dtype": "bfloat16", "grad_scaler": None},
        "batch_size": batch_size_sec * 16000,
        "max_seq_length": {"audio": 35 * 16000},
        "max_seqs": 300,
        "torch_dataloader_opts": {"num_workers": 2},
    }


def _run(
    prefix: str,
    *,
    train_splits,
    peak_lr: float,
    num_epochs: int,
    batch_size_sec: int = 480,
    dropout: float = 0.1,
    specaug_start_step: int = 5000,
    preload: Optional[dict] = None,
):
    _spm_ds, spm_model = get_ls960_spm()  # shared SPM-5120 label space across all CTC runs
    model_config = build_ctc_config(specaug_start_step=specaug_start_step, dropout=dropout)
    # per-rank SEQUENCE-level DDP sharding (1 epoch == 1 disjoint pass); dev stays unsharded.
    train_set = ds.labeled_hf_dataset(train_splits, spm_model=spm_model, seq_ordering="random", ddp_seq_shard=True)
    dev_set = ds.labeled_hf_dataset(ds.DEV_ALL, spm_model=spm_model, seq_ordering="default")
    config = _ctc_train_config(
        peak_lr=peak_lr, num_epochs=num_epochs, batch_size_sec=batch_size_sec, vocab_size=SPM_VOCAB_SIZE
    )
    if preload is not None:
        config["preload_from_files"] = preload
    # keep the fractional-epoch checkpoints (10/30/.../100%) alive for multi-checkpoint recognition
    keep_epochs = fraction_epochs(num_epochs)
    returnn_config = get_training_config(
        train_dataset=train_set,
        dev_dataset=dev_set,
        network_module=NETWORK_MODULE,
        net_args={"model_config_dict": asdict(model_config), "vocab_size": SPM_VOCAB_SIZE},
        train_step_args={},
        config=config,
        keep_epochs=keep_epochs,
    )
    train_job = training(
        prefix,
        returnn_config,
        num_epochs=num_epochs,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
        num_processes=4,
        gpu_mem=96,
        time_rqmt=12,
        mem_rqmt=100,
        cpu_rqmt=16,
    )
    # greedy decode + sclite WER on dev/test clean+other at 10/30/50/70/90/100% checkpoints + summary report
    ctc_recog_all_epochs(
        prefix,
        train_job=train_job,
        num_epochs=num_epochs,
        model_config_dict=asdict(model_config),
        vocab_size=SPM_VOCAB_SIZE,
        spm_model=spm_model,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
    )
    # also decode the dev-loss-best checkpoint (keep_best_n keeps it) -> recog/epbest; the fixed
    # fraction checkpoints can miss the dev optimum (LS100 overfits, dev CTC bottoms early).
    ctc_recog_best_checkpoint(
        prefix,
        train_job=train_job,
        model_config_dict=asdict(model_config),
        vocab_size=SPM_VOCAB_SIZE,
        spm_model=spm_model,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
    )
    return train_job


def ctc_ls100_scratch():
    """From-scratch CTC baseline on train-clean-100 (no SSL pretraining) -- the low-resource point.

    Overfit fix (2026-06-18, diagnosed: dev CTC bottoms ~ep24 then drifts up): cut from 200 real
    passes to 100 (the dominant lever; ~i6 LS100 ~83-pass budget), raise all conformer dropouts
    0.1->0.2 (i6 LS100 standard; was half), and start SpecAugment at step 2000 instead of 5000 (it
    was off until ~ep13-26, the memorization window). SpecAug mask strength is already strong (time
    ~40%, freq ~35%) so it is NOT increased. weight_decay 0.01 / peak_lr 1e-3 kept per earlier choice."""
    return _run(
        "ssl/ctc_ls100/scratch_12x512_spm5k",
        train_splits=ds.TRAIN_CLEAN_100,
        peak_lr=1e-3,
        num_epochs=100,
        dropout=0.2,
        specaug_start_step=2000,
    )


def ctc_ls960_scratch():
    """From-scratch supervised CTC on the full labeled 960h -- the supervised topline for comparison.

    ~960h / (480 s/GPU x 4 = 1920 s/step) ~= 1800 steps/pass; 60 passes ~= 108k updates.
    Regularization deliberately kept LIGHT (dropout 0.1, specaug start step 5000, 60 passes): 960h is
    ~9.6x the LS100 data, so the LS100 overfit fix does NOT apply here -- over-regularizing would
    underfit the topline. Unchanged from the original config (hash-stable)."""
    return _run("ssl/ctc_ls960/scratch_12x512_spm5k", train_splits=ds.TRAIN_960H, peak_lr=1e-3, num_epochs=60)


def ctc_ls100_finetune(ssl_checkpoint):
    """CTC finetuning with the encoder initialized from a BEST-RQ checkpoint (PtCheckpoint or path)."""
    preload = {
        "ssl_encoder": {
            "filename": ssl_checkpoint,
            "init_for_train": True,
            "ignore_missing": True,  # CTC head is fresh; SSL heads/quantizer are extra
        }
    }
    # Finetune on LS100 (same small data as scratch) but from a pretrained encoder -> moderate reg:
    # dropout 0.15 (between scratch 0.2 and SSL 0.1) and 100 passes (200 was too long for 100h), with
    # the earlier specaug start; peak_lr 5e-4 kept. The pretrained encoder reduces (not removes) the
    # 100h overfit risk. See [[ctc100-overfit-diagnosis]].
    return _run(
        "ssl/ctc_ls100/finetune_bestrq_12x512_spm5k",
        train_splits=ds.TRAIN_CLEAN_100,
        peak_lr=5e-4,
        num_epochs=100,
        dropout=0.15,
        specaug_start_step=2000,
        preload=preload,
    )
