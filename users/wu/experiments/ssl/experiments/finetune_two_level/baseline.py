"""
Two-level BEST-RQ + CIF -> CTC finetuning on LibriSpeech train-clean-100 (FT-1, SPM-5120).

Takes a two-level PRETRAINED checkpoint (frozen lower encoder + CIF segmenter + high encoder), swaps the
masked-prediction head for a CTC head over the CIF token sequence (network ``two_level.two_level_ctc_v1``),
and finetunes on LS100. Per pretrained rate arm we run BOTH regimes:
  * ft1_ctc_frozenseg  -- CIF segmenter FROZEN (probe of the pretrained segmentation), and
  * ft1_ctc_trainseg   -- CIF segmenter TRAINABLE (best WER / ablation).
WER = greedy CTC decode + sclite at the fraction-epoch checkpoints + the dev-loss-best checkpoint (the
base-CTC pipeline). Each pretraining experiment's ``summary.md`` is filled with these finetune WERs (both
variants) instead of the SSL losses -- the base BEST-RQ pattern. See [[cif-length-control-and-finetune]]
and [[ctc-finetune-blueprint]].

NOTE: FT-2 (scaled-CIF + NAR per-token CE, Paraformer-style) is the OTHER planned finetune head. It is NOT
an imitation of the base CTC finetune (different head + scaled-alpha + retargeted quantity loss), so it is
deliberately left to a separate module -- still pending.
"""

from __future__ import annotations

from dataclasses import asdict

from ...config import get_training_config
from ...pipeline import (
    training,
    ctc_recog_all_epochs,
    fraction_epochs,
    register_multi_wer_summary,
)
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
from ...data import datasets as ds
from ...data.spm import get_ls960_spm, SPM_VOCAB_SIZE
from ..ctc_ls100.baseline import (  # shared CTC train config (adamw eps 1e-16, OCLR, bf16) + two-rate LR helper
    _ctc_train_config,
    _apply_head_lr_multiplier,
    TWO_RATE_PROLOG,
    HEAD_LR_MULT,
)
from ..pretrain_two_level.baseline import (
    build_model_config,
    two_level_80ms_c128,
    two_level_120ms_c128,
    meanpool_80ms_c128,
    RATE_80MS,
    RATE_120MS,
)

NETWORK_MODULE = "two_level.two_level_ctc_v1"
DECODER_MODULE = "two_level.two_level_ctc_v1"
# FT-2: Paraformer-style scaled-CIF + NAR per-token CE (model == decoder; argmax-per-token, no collapse).
CE_NETWORK_MODULE = "two_level.two_level_ce_v1"
# mean-pool CONTROL CTC finetune (fixed 2x pool, no learned segmenter -> single regime).
MEANPOOL_NETWORK_MODULE = "two_level.two_level_meanpool_ctc_v1"

# two_level_pretrain default num_epochs == 50, so the final pretrained checkpoint is epoch 50.
PRETRAIN_BASE_EPOCH = 50


def _ft_run(
    prefix: str,
    *,
    pretrain_job,
    model_config,
    spm_model,
    freeze_segmenter: bool,
    num_epochs: int = 100,
    peak_lr: float = 1e-4,  # FT peak LR (was 5e-4); lowered to match the base BEST-RQ FT and the SSL-paper
    batch_size_sec: int = 480,  # FT/pretrain-LR convention (backbone finetuned well below its pretrain peak).
    head_lr_multiplier: float = HEAD_LR_MULT,  # fresh heads at Nx the backbone LR (BEST-RQ-style two-rate)
    base_epoch: int = PRETRAIN_BASE_EPOCH,
    network_module: str = NETWORK_MODULE,  # default = CIF CTC FT; mean-pool / CE FT pass their own module
    decoder_module: str = DECODER_MODULE,
    quantity_loss_scale=None,  # CIF rate anchor; set ONLY for the trainable-segmenter FT-1 (see _ft_arm)
):
    """One finetune run on top of a two-level pretrained checkpoint. Returns (train_job, wers).

    The default modules are the FT-1 CIF CTC head; ``network_module``/``decoder_module`` select the
    mean-pool CTC FT (``two_level.two_level_meanpool_ctc_v1``) or the FT-2 NAR CE head
    (``two_level.two_level_ce_v1``) instead. ``out_linear``/``aux_linears`` are the fresh heads in all
    three, so the two-rate head LR multiplier applies uniformly; the NAR-CE recog (argmax per token) is
    driven by the CE module's own ForwardCallback, reusing the same greedy-recog harness."""
    # Preload the FROZEN lower stack + CIF + high encoder from the two-level pretrained ckpt; the CTC head
    # (out_linear) is fresh, and the pretrained head/mask_emb/codebook buffer are extra -> ignore_missing.
    preload = {
        "two_level": {
            "filename": pretrain_job.out_checkpoints[base_epoch],
            "init_for_train": True,
            "ignore_missing": True,
        }
    }
    config = _ctc_train_config(
        peak_lr=peak_lr, num_epochs=num_epochs, batch_size_sec=batch_size_sec, vocab_size=SPM_VOCAB_SIZE
    )
    config["preload_from_files"] = preload
    # two-rate finetune (BEST-RQ-style): fresh CTC heads (out_linear + aux_linears) at head_lr_multiplier x the
    # scheduled LR, pretrained backbone (high encoder + CIF segmenter) at 1x. Same mechanism as the base BEST-RQ FT.
    _apply_head_lr_multiplier(config, head_lr_multiplier)
    train_set = ds.labeled_hf_dataset(
        ds.TRAIN_CLEAN_100, spm_model=spm_model, seq_ordering="random", ddp_seq_shard=True
    )
    dev_set = ds.labeled_hf_dataset(ds.DEV_ALL, spm_model=spm_model, seq_ordering="default")
    keep_epochs = fraction_epochs(num_epochs)
    # net_args built as a dict so the rate-anchor key is added ONLY when set: omitting it for the
    # frozen-seg / mean-pool / FT-2 arms keeps their job hash (import-path + net_args) byte-identical so
    # they are NOT re-run; the trainseg arm gets the extra key -> a fresh hash -> a clean re-run.
    net_args = {
        "model_config_dict": asdict(model_config),
        "vocab_size": SPM_VOCAB_SIZE,
        "freeze_segmenter": freeze_segmenter,
        "specaug_start_step": 2000,  # activate SpecAugment (same strength/step as the base CTC FT)
    }
    if quantity_loss_scale is not None:
        net_args["quantity_loss_scale"] = quantity_loss_scale
    returnn_config = get_training_config(
        train_dataset=train_set,
        dev_dataset=dev_set,
        network_module=network_module,
        net_args=net_args,
        train_step_args={},
        config=config,
        keep_epochs=keep_epochs,
        python_prolog=[TWO_RATE_PROLOG],
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
    wers = ctc_recog_all_epochs(
        prefix,
        train_job=train_job,
        num_epochs=num_epochs,
        model_config_dict=asdict(model_config),
        vocab_size=SPM_VOCAB_SIZE,
        spm_model=spm_model,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
        decoder_module=decoder_module,
    )
    # NOTE: no automatic best-checkpoint selection -- the dev optimum is read directly off the per-epoch WER
    # tables in the summary (register_multi_wer_summary); dev CTC loss bottoms early on LS100 while WER gains.
    return train_job, wers


def _ft_arm(arm: str, pretrain_job, model_config, *, num_epochs: int = 100):
    """Wire all finetune heads for one CIF pretrained arm and mirror their WER into the pretrain summary.md.

    Three downstream variants, one summary table each: FT-1 CTC (frozen segmenter), FT-1 CTC (trainable
    segmenter), and FT-2 CE (Paraformer-style scaled-CIF + NAR per-token CE). All preload the same frozen
    lower stack + CIF + high encoder from the pretrained ckpt.

    :param arm: the pretraining arm name, e.g. ``ls960_cif80ms_k128`` (also the pretrain prefix tail).
    """
    _spm_ds, spm_model = get_ls960_spm()
    named = {}  # ordered: label -> (wers, num_epochs) for the combined pretrain summary
    for freeze, tag, label in [
        (True, "ft1_ctc_frozenseg", "FT-1 CTC · frozen segmenter"),
        (False, "ft1_ctc_trainseg", "FT-1 CTC · trainable segmenter"),
    ]:
        # TRAINABLE segmenter needs the CIF rate anchor (quantity loss at the pretrained lambda_qty) or it
        # collapses K<N under zero_infinity CTC (see [[ft-collapse-diagnosis]]); the FROZEN arm has no
        # learnable alpha so it needs none (and stays hash-stable / not re-run).
        _job, wers = _ft_run(
            f"ssl/finetune_two_level/{arm}/{tag}",
            pretrain_job=pretrain_job,
            model_config=model_config,
            spm_model=spm_model,
            freeze_segmenter=freeze,
            num_epochs=num_epochs,
            quantity_loss_scale=None if freeze else model_config.lambda_qty,
        )
        named[label] = (wers, num_epochs)
    # FT-2 CE: scaled-CIF + per-token CE (Paraformer-style NAR). TRAINABLE segmenter only -- the CIF must
    # retune from the pretrained ~12.5/8.33 Hz rate down to the SPM label rate, so a frozen segmenter would
    # fire ~12.5 Hz >> N at inference (NAR garbage). The CE module is its own NAR decoder (argmax/token).
    _job, wers = _ft_run(
        f"ssl/finetune_two_level/{arm}/ft2_ce_trainseg",
        pretrain_job=pretrain_job,
        model_config=model_config,
        spm_model=spm_model,
        freeze_segmenter=False,
        num_epochs=num_epochs,
        network_module=CE_NETWORK_MODULE,
        decoder_module=CE_NETWORK_MODULE,
    )
    named["FT-2 CE · scaled-CIF (Paraformer)"] = (wers, num_epochs)
    # surface ALL downstream variants' WER as the pretraining experiment's headline (not the SSL losses)
    register_multi_wer_summary(
        f"ssl/pretrain_two_level/{arm}", named, title="two-level downstream WER (FT-1 CTC + FT-2 CE)"
    )
    return named


def two_level_ft_80ms():
    """FT-1 CTC (frozen + trainable seg) + FT-2 CE on the 80 ms / 128-cluster pretrained arm."""
    pretrain_job = two_level_80ms_c128()
    # high_dropout 0.15 for FT (the pretrain default is 0.1) -> matches the base BEST-RQ CTC FT; dropout is a
    # runtime config (not a weight), so it does not affect the preload of the pretrained high-encoder weights.
    model_config = build_model_config(target_rate_hz=RATE_80MS, num_clusters=128, high_dropout=0.15)
    return _ft_arm("ls960_cif80ms_k128", pretrain_job, model_config)


def two_level_ft_120ms():
    """FT-1 CTC (frozen + trainable seg) + FT-2 CE on the 120 ms / 128-cluster pretrained arm."""
    pretrain_job = two_level_120ms_c128()
    # high_dropout 0.15 for FT (see two_level_ft_80ms): matches the base BEST-RQ CTC FT; preload unaffected.
    model_config = build_model_config(target_rate_hz=RATE_120MS, num_clusters=128, high_dropout=0.15)
    return _ft_arm("ls960_cif120ms_k128", pretrain_job, model_config)


def meanpool_ft_80ms():
    """CTC finetune of the mean-pool CONTROL arm (single regime: no learned segmenter to freeze).

    Mirrors the FT-1 CTC recipe detail-for-detail (peak 1e-4, two-rate head 3x, dropout 0.15, SpecAug,
    InterCTC aux [4,8]@0.3) on the mean-pool pretrained ckpt, so the only difference vs the 80 ms CIF
    arm's ``ft1_ctc_*`` WER is the segmenter (fixed 2x pool vs learned CIF). One WER table in summary.md."""
    pretrain_job = meanpool_80ms_c128()
    model_config = build_model_config(target_rate_hz=RATE_80MS, num_clusters=128, high_dropout=0.15)
    _spm_ds, spm_model = get_ls960_spm()
    _job, wers = _ft_run(
        "ssl/finetune_two_level/ls960_meanpool80ms_k128/ft1_ctc",
        pretrain_job=pretrain_job,
        model_config=model_config,
        spm_model=spm_model,
        freeze_segmenter=False,  # inert for the mean-pool model (swallowed); kept for the uniform _ft_run API
        num_epochs=100,
        network_module=MEANPOOL_NETWORK_MODULE,
        decoder_module=MEANPOOL_NETWORK_MODULE,
    )
    register_multi_wer_summary(
        "ssl/pretrain_two_level/ls960_meanpool80ms_k128",
        {"FT-1 CTC · fixed mean-pool 2x": (wers, 100)},
        title="mean-pool control downstream CTC-WER",
    )
    return wers
