"""
Attention-based encoder-decoder (AED) experiments.

The default baseline is based on exp2023_04_25_rf/aed.py and uses a Conformer encoder and a Transformer decoder.

Changes from that baseline:
- new train_v3 function
- new Librispeech corpus (see get_librispeech_task_bpe10k_raw_v2)
- no log_base=math.exp(2.3026) in log_mel_filterbank_from_raw
"""

from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence
import numpy
import torch
import torch.nn as nn
import tree



from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, TrainDef
from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.yang.torch.loss.ctc_forward_backward import ctc_forward
from i6_experiments.users.yang.torch.utils.masking import get_seq_mask

from .configs import *
from .configs import _get_cfg_lrlin_oclr_by_bs_nep, _batch_size_factor, _fine_tune_get_cfg_lrlin_oclr_by_bs_nep
from .aed_new import Model

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, ModelWithCheckpoint
    from i6_experiments.users.zeyer.datasets.task import Task


# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def py():
    # train_exp(  # 5.32, but should give 5.11?
    #     "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2",
    #     config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
    #         "optimizer.weight_decay": 1e-2,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #     },
    # )

    # for max_seq_len in [
    #     60,  # 5.39
    #     74,  # now EOS is not counted, so this is same as before
    #     75,  # 5.32?
    #     None,  # 5.17
    # ]:
    #     train_exp(
    #         f"v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-maxSeqLen{max_seq_len}-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2",
    #         config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
    #             "optimizer.weight_decay": 1e-2,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #             "max_seq_length_default_target": max_seq_len,
    #         },
    #     )

    # Comparing vocabs.
    # for vocab in [
    #     "spm20k",  # 5.14 (but test-other is 6.18!)
    #     "bpe10k",  # 5.32
    #     "spm10k",  # 5.16
    #     "spm_bpe10k",  # 5.21
    #     "spm4k",
    #     "spm1k",
    # ]:
    #     train_exp(  # 5.16
    #         f"v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-{vocab}",
    #         config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
    #             "optimizer.weight_decay": 1e-2,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         },
    #         vocab=vocab,
    #     )

    # Comparing vocabs with better settings: feature norm, sampling, no max seq len.
    # maxSeqLenNone might not be actually better though...
    base_checkpoint_path = "/work/asr4/zyang/torch_checkpoints/aed/albert_bpe10k_sampling001/epoch.487.pt"
    # baseline wer: aed "best_scores": {"dev-clean": 2.23, "dev-other": 5.23, "test-clean": 2.42, "test-other": 5.47}, "best_epoch": 487}
    # for vocab, alpha in [
    #     # ("spm20k", 0.7),
    #     ("bpe10k", 0.01),  # 5.23
    #     # ("spm10k", 0.7),  # 5.12, slightly worse than before...
    #     # ("spm_bpe10k", ...),  # unclear what sampling scheme...
    #     # ("spm4k", 0.7),
    #     # ("spm1k", 0.7),
    #     # ("spm_bpe1k", ...)
    # ]:
    #     config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4)
    #     #config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_gpu1_wd1e_4)
    #     config_finetune_ctc12.update(aux_loss_layers=[4,8,12])
    #     train_exp(  # 5.16
    #         f"v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-"
    #         f"-speedpertV2-{vocab}"
    #         f"-{'spmSample' if vocab.startswith('spm') else 'bpeSample'}{str(alpha).replace('.', '')}-finetune-ctc12-scale1.0-others0.1",
    #         config_finetune_ctc12,
    #         model_config={"feature_batch_norm": True},
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 100, peak_lr=1e-3),
    #             "optimizer.weight_decay": 1e-2,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #             "max_seq_length_default_target": None,
    #             "preload_from_files": {
    #             "base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
    #             "aux_loss_scales": [0.1,0.1,1],
    #             "aed_loss_scale": 0.1,
    #
    #         },
    #         vocab=vocab,
    #         train_vocab_opts={
    #             "other_opts": (
    #                 {"enable_sampling": True, "alpha": alpha}
    #                 if vocab.startswith("spm")
    #                 else {"class": "SamplingBytePairEncoding", "breadth_prob": alpha}
    #             )
    #         },
    #     )

    # baseline {"best_scores": {"dev-clean": 2.21, "dev-other": 5.15, "test-clean": 2.49, "test-other": 5.48}, "best_epoch": 482}
    base_checkpoint_path = "/work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.V18BvuJ52QAA/output/models/epoch.482.pt"

    # check conformer_ctc_recog for aed results
    bszs = [(15_000, 100)]
    aed_logit_scales = [0.2, 0.4,0.6, 1.0]
    for aed_logit_scale in aed_logit_scales:
        for bsz in bszs:
            for vocab, alpha in [
                # ("spm20k", 0.7),
                ("bpe10k", 0.01),  # 5.23
                #("bpe10k", 0.001),
                # ("spm10k", 0.7),  # 5.12, slightly worse than before...
                # ("spm_bpe10k", ...),  # unclear what sampling scheme...
                # ("spm4k", 0.7),
                # ("spm1k", 0.7),
                # ("spm_bpe1k", ...)
            ]:
                #config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4)
                config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_gpu1_wd1e_4)
                config_finetune_ctc12.update(aux_loss_layers=[4, 8, 12])

                aed_ctc_kd_loss = True
                ctc_kd_layer = 12
                kd_warmup_steps = 20000
                kd_scale = 1.0
                top_k = 10

                train_exp(
                    f"debug-v6-bhv20-11gb-f32-bs15k-accgrad1-singlemgpu"
                    f"-speedpertV2-{vocab}"
                    f"kd_layer{ctc_kd_layer}-wm{kd_warmup_steps}-aed_scale{aed_logit_scale}-topk{top_k}-bsz{bsz[0]}-ep{bsz[1]}",
                    config_finetune_ctc12,
                    model_config={"feature_batch_norm": True},
                    config_updates={
                        **_fine_tune_get_cfg_lrlin_oclr_by_bs_nep(bsz[0], bsz[1], peak_lr=1e-5),
                        "optimizer.weight_decay": 1e-2,
                        "__train_audio_preprocess": speed_pert_librosa_config,
                        "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                        "max_seq_length_default_target": None,
                        "preload_from_files": {
                        "base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
                        "aed_ctc_kd_loss": aed_ctc_kd_loss,
                        "ctc_kd_layer": ctc_kd_layer,
                        "kd_warmup_steps": kd_warmup_steps,
                        "kd_scale": kd_scale,
                        "top_k": top_k,
                        "aed_logit_scale": aed_logit_scale,
                    },
                    vocab=vocab,
                    mem_rqmt=20,
                )

    bszs = [(15_000, 100)]
    aed_logit_scales = [0.2, 0.4,0.6, 1.0]
    for aed_logit_scale in aed_logit_scales:
        for bsz in bszs:
            for vocab, alpha in [
                # ("spm20k", 0.7),
                ("bpe10k", 0.01),  # 5.23
                #("bpe10k", 0.001),
                # ("spm10k", 0.7),  # 5.12, slightly worse than before...
                # ("spm_bpe10k", ...),  # unclear what sampling scheme...
                # ("spm4k", 0.7),
                # ("spm1k", 0.7),
                # ("spm_bpe1k", ...)
            ]:
                #config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4)
                config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_gpu1_wd1e_4)
                config_finetune_ctc12.update(aux_loss_layers=[4, 8, 12])

                aed_ctc_kd_loss = True
                ctc_kd_layer = 12
                kd_warmup_steps = 20000
                kd_scale = 1.0
                top_k = 10

                train_exp(
                    f"debug-v6-bhv20-11gb-f32-bs15k-accgrad1-singlemgpu"
                    f"-speedpertV2-{vocab}"
                    f"kd_layer{ctc_kd_layer}-wm{kd_warmup_steps}-aed_scale{aed_logit_scale}-topk{top_k}-bsz{bsz[0]}-ep{bsz[1]}-sampling{alpha}",
                    config_finetune_ctc12,
                    model_config={"feature_batch_norm": True},
                    config_updates={
                        **_fine_tune_get_cfg_lrlin_oclr_by_bs_nep(bsz[0], bsz[1], peak_lr=1e-5),
                        "optimizer.weight_decay": 1e-2,
                        "__train_audio_preprocess": speed_pert_librosa_config,
                        "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                        "max_seq_length_default_target": None,
                        "preload_from_files": {
                        "base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
                        "aed_ctc_kd_loss": aed_ctc_kd_loss,
                        "ctc_kd_layer": ctc_kd_layer,
                        "kd_warmup_steps": kd_warmup_steps,
                        "kd_scale": kd_scale,
                        "top_k": top_k,
                        "aed_logit_scale": aed_logit_scale,
                    },
                    vocab=vocab,
                    mem_rqmt=20,
                    train_vocab_opts={
                                "other_opts": (
                                    {"enable_sampling": True, "alpha": alpha}
                                    if vocab.startswith("spm")
                                    else {"class": "SamplingBytePairEncoding", "breadth_prob": alpha}
                                )
                            },
                )
    bszs = [(15_000, 100)]
    aed_logit_scales = [1.0,0.2,0.4,0.6]
    for aed_logit_scale in aed_logit_scales:
        for bsz in bszs:
            for vocab, alpha in [
                # ("spm20k", 0.7),
                ("bpe10k", 0.01),  # 5.23
                #("bpe10k", 0.001),
                # ("spm10k", 0.7),  # 5.12, slightly worse than before...
                # ("spm_bpe10k", ...),  # unclear what sampling scheme...
                # ("spm4k", 0.7),
                # ("spm1k", 0.7),
                # ("spm_bpe1k", ...)
            ]:
                #config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4)
                config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_gpu1_wd1e_4)
                config_finetune_ctc12.update(aux_loss_layers=[4, 8, 12])

                aed_ctc_kd_loss = True
                ctc_kd_layer = 12
                kd_warmup_steps = 20000
                kd_scale = 1.0
                top_k = 10

                train_exp(
                    f"debug-v6-bhv20-11gb-f32-bs15k-accgrad1-singlemgpu"
                    f"-speedpertV2-{vocab}"
                    f"fw_only-kd_layer{ctc_kd_layer}-wm{kd_warmup_steps}-aed_scale{aed_logit_scale}-topk{top_k}-bsz{bsz[0]}-ep{bsz[1]}-sampling{alpha}",
                    config_finetune_ctc12,
                    model_config={"feature_batch_norm": True},
                    config_updates={
                        **_fine_tune_get_cfg_lrlin_oclr_by_bs_nep(bsz[0], bsz[1], peak_lr=1e-5),
                        "optimizer.weight_decay": 1e-2,
                        "__train_audio_preprocess": speed_pert_librosa_config,
                        "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                        "max_seq_length_default_target": None,
                        "preload_from_files": {
                        "base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
                        "aed_ctc_kd_loss": aed_ctc_kd_loss,
                        "ctc_kd_layer": ctc_kd_layer,
                        "kd_warmup_steps": kd_warmup_steps,
                        "kd_scale": kd_scale,
                        "top_k": top_k,
                        "aed_logit_scale": aed_logit_scale,
                        "kd_w_bw": False,
                    },
                    vocab=vocab,
                    mem_rqmt=20,
                    train_vocab_opts={
                                "other_opts": (
                                    {"enable_sampling": True, "alpha": alpha}
                                    if vocab.startswith("spm")
                                    else {"class": "SamplingBytePairEncoding", "breadth_prob": alpha}
                                )
                            },
                )

    bszs = [(15_000, 100)]
    aed_logit_scales = [1.0,0.2,0.4,0.6]
    for kd_w_bw in [True, False]:
        for aed_logit_scale in aed_logit_scales:
            for bsz in bszs:
                for vocab, alpha in [
                    # ("spm20k", 0.7),
                    ("bpe10k", 0.01),  # 5.23
                    #("bpe10k", 0.001),
                    # ("spm10k", 0.7),  # 5.12, slightly worse than before...
                    # ("spm_bpe10k", ...),  # unclear what sampling scheme...
                    # ("spm4k", 0.7),
                    # ("spm1k", 0.7),
                    # ("spm_bpe1k", ...)
                ]:
                    #config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4)
                    config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_gpu1_wd1e_4)
                    config_finetune_ctc12.update(aux_loss_layers=[4, 8, 12])

                    aed_ctc_kd_loss = True
                    ctc_kd_layer = 12
                    kd_warmup_steps = 20000
                    kd_scale = 1.0
                    top_k = 10
                    if kd_w_bw:
                        subname = ''
                    else:
                        subname = 'fw_only'

                    train_exp(
                        f"debug-v6-bhv20-11gb-f32-bs15k-accgrad1-singlemgpu"
                        f"-speedpertV2-{vocab}"
                        f"-{subname}-target_detach-kd_layer-{ctc_kd_layer}-wm{kd_warmup_steps}-aed_scale{aed_logit_scale}-topk{top_k}-bsz{bsz[0]}-ep{bsz[1]}-sampling{alpha}",
                        config_finetune_ctc12,
                        model_config={"feature_batch_norm": True},
                        config_updates={
                            **_fine_tune_get_cfg_lrlin_oclr_by_bs_nep(bsz[0], bsz[1], peak_lr=1e-5),
                            "optimizer.weight_decay": 1e-2,
                            "__train_audio_preprocess": speed_pert_librosa_config,
                            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                            "max_seq_length_default_target": None,
                            "preload_from_files": {
                            "base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
                            "aed_ctc_kd_loss": aed_ctc_kd_loss,
                            "ctc_kd_layer": ctc_kd_layer,
                            "kd_warmup_steps": kd_warmup_steps,
                            "kd_scale": kd_scale,
                            "top_k": top_k,
                            "aed_logit_scale": aed_logit_scale,
                            "kd_w_bw": kd_w_bw,
                            "kd_target_detach": True,
                        },
                        vocab=vocab,
                        mem_rqmt=20,
                        train_vocab_opts={
                                    "other_opts": (
                                        {"enable_sampling": True, "alpha": alpha}
                                        if vocab.startswith("spm")
                                        else {"class": "SamplingBytePairEncoding", "breadth_prob": alpha}
                                    )
                                },
                    )
#   constant learning rate, longer training
#   different aed loss scales
    bszs = [(15_000, 100)]
    aed_logit_scales = [1.0, 0.6, 0.8]
    epoch = 200
    for kd_scale in [0.2,0.6,1.0]:
        for learning_rate in [1e-6,5e-7]:
            for kd_w_bw in [False]:
                for aed_logit_scale in aed_logit_scales:
                    for bsz in bszs:
                        for vocab, alpha in [
                            # ("spm20k", 0.7),
                            ("bpe10k", 0.01),  # 5.23
                            #("bpe10k", 0.001),
                            # ("spm10k", 0.7),  # 5.12, slightly worse than before...
                            # ("spm_bpe10k", ...),  # unclear what sampling scheme...
                            # ("spm4k", 0.7),
                            # ("spm1k", 0.7),
                            # ("spm_bpe1k", ...)
                        ]:
                            #config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4)
                            config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_gpu1_wd1e_4)
                            config_finetune_ctc12.update(aux_loss_layers=[4, 8, 12])

                            aed_ctc_kd_loss = True
                            ctc_kd_layer = 12
                            kd_warmup_steps = 20000
                            #kd_scale = 1.0
                            top_k = 10
                            if kd_w_bw:
                                subname = 'fw_bw'
                            else:
                                subname = 'fw_only'

                            train_exp(
                                f"fine-tune-accgrad1-singlemgpu"
                                f"-speedpertV2-{vocab}"
                                f"-{subname}-target_detach-kd_layer-{ctc_kd_layer}-wm{kd_warmup_steps}-kd_scale{kd_scale}-aed_logit_scale{aed_logit_scale}-topk{top_k}-bsz{bsz[0]}-ep{epoch}-constlr-{learning_rate}-sampling{alpha}",
                                config_finetune_ctc12,
                                model_config={"feature_batch_norm": True},
                                config_updates={
                                    #**_fine_tune_get_cfg_lrlin_oclr_by_bs_nep(bsz[0], bsz[1], peak_lr=1e-5),
                                    "batch_size": bsz[0] * 160,
                                    "__num_epochs": 200,
                                    "learning_rate": learning_rate,
                                    "optimizer.weight_decay": 1e-2,
                                    "__train_audio_preprocess": speed_pert_librosa_config,
                                    "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                                    "max_seq_length_default_target": None,
                                    "preload_from_files": {
                                    "base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
                                    "aed_ctc_kd_loss": aed_ctc_kd_loss,
                                    "ctc_kd_layer": ctc_kd_layer,
                                    "kd_warmup_steps": kd_warmup_steps,
                                    "kd_scale": kd_scale,
                                    "top_k": top_k,
                                    "aed_logit_scale": aed_logit_scale,
                                    "kd_w_bw": kd_w_bw,
                                    "kd_target_detach": True,
                                },
                                vocab=vocab,
                                mem_rqmt=20,
                                train_vocab_opts={
                                            "other_opts": (
                                                {"enable_sampling": True, "alpha": alpha}
                                                if vocab.startswith("spm")
                                                else {"class": "SamplingBytePairEncoding", "breadth_prob": alpha}
                                            )
                                        },
                            )



    # bszs = [(10_000, 100)]
    # aed_logit_scales = [1.0]
    # for aed_logit_scale in aed_logit_scales:
    #     for bsz in bszs:
    #         for vocab, alpha in [
    #             # ("spm20k", 0.7),
    #             ("bpe10k", 0.01),  # 5.23
    #             #("bpe10k", 0.001),
    #             # ("spm10k", 0.7),  # 5.12, slightly worse than before...
    #             # ("spm_bpe10k", ...),  # unclear what sampling scheme...
    #             # ("spm4k", 0.7),
    #             # ("spm1k", 0.7),
    #             # ("spm_bpe1k", ...)
    #         ]:
    #             config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4)
    #             #config_finetune_ctc12 = copy.deepcopy(config_11gb_v6_f32_accgrad1_gpu1_wd1e_4)
    #             config_finetune_ctc12.update(aux_loss_layers=[4, 8, 12])
    #
    #             aed_ctc_kd_loss = True
    #             ctc_kd_layer = 12
    #             kd_warmup_steps = 50000
    #             kd_scale = 1.0
    #             top_k = 10
    #
    #             train_exp(
    #                 f"debug-v6-bhv20-11gb-f32-bs15k-accgrad1-gpu4"
    #                 f"-speedpertV2-{vocab}"
    #                 f"kd_layer{ctc_kd_layer}-wm{kd_warmup_steps}-aed_scale{aed_logit_scale}-topk{top_k}-bsz{bsz[0]}-ep{bsz[1]}",
    #                 config_finetune_ctc12,
    #                 model_config={"feature_batch_norm": True},
    #                 config_updates={
    #                     **_fine_tune_get_cfg_lrlin_oclr_by_bs_nep(bsz[0], bsz[1], peak_lr=1e-5),
    #                     "optimizer.weight_decay": 1e-2,
    #                     "__train_audio_preprocess": speed_pert_librosa_config,
    #                     "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #                     "max_seq_length_default_target": None,
    #                     "preload_from_files": {
    #                     "base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
    #                     "aed_ctc_kd_loss": aed_ctc_kd_loss,
    #                     "ctc_kd_layer": ctc_kd_layer,
    #                     "kd_warmup_steps": kd_warmup_steps,
    #                     "kd_scale": kd_scale,
    #                     "top_k": top_k,
    #                     "aed_logit_scale": aed_logit_scale,
    #                 },
    #                 vocab=vocab,
    #                 #mem_rqmt=15,
    #                 #reserve_code="hlt_11",
    #             )




# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    *,
    model_def: Optional[Union[ModelDefWithCfg, ModelDef[Model]]] = None,
    vocab: str = "bpe10k",
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_def: Optional[TrainDef[Model]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    time_rqmt: Optional[int] = None,  # set this to 1 or below to get the fast test queue
    mem_rqmt: Optional[int] = None,
    reserve_code: Optional[str] = None
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from i6_experiments.users.yang.torch.albert_exp2024_04_23_baselines.train_v3 import train
    from i6_experiments.users.zeyer.recog import recog_training_exp
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    task = get_librispeech_task_raw_v2(vocab=vocab, train_vocab_opts=train_vocab_opts)
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    if "__num_epochs" in config:
        num_epochs = config.pop("__num_epochs")
    if "__gpu_mem" in config:
        gpu_mem = config.pop("__gpu_mem")
    if "__num_processes" in config:
        num_processes = config.pop("__num_processes")
    if "__train_audio_preprocess" in config:
        task: Task = copy.copy(task)
        task.train_dataset = copy.copy(task.train_dataset)
        task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")

    if not model_def:
        model_def = aed_model_def
    if model_config:
        model_def = ModelDefWithCfg(model_def, model_config)
    if not train_def:
        train_def = aed_training
    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        model_def=model_def,
        train_def=train_def,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else "mpirun",
        time_rqmt=time_rqmt,
        mem_rqmt=mem_rqmt,
        reserve_code=reserve_code,
    )
    recog_training_exp(prefix, task, model_with_checkpoint, recog_def=model_recog)

    return model_with_checkpoint


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


def aed_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim, **kwargs) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    num_enc_layers = config.int("num_enc_layers", 12)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    model_args = kwargs.get("model_args", None)

    return Model(
        in_dim,
        num_enc_layers=num_enc_layers,
        enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
        enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
        enc_att_num_heads=8,
        enc_conformer_layer_opts=dict(
            conv_norm_opts=dict(use_mask=True),
            self_att_opts=dict(
                # Shawn et al 2018 style, old RETURNN way.
                with_bias=False,
                with_linear_pos=False,
                with_pos_bias=False,
                learnable_pos_emb=True,
                separate_pos_emb_per_head=False,
                pos_emb_dropout=pos_emb_dropout,
            ),
            ff_activation=lambda x: rf.relu(x) ** 2.0,
        ),
        target_dim=target_dim,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        enc_aux_logits=enc_aux_logits or (),
        model_args=model_args,
    )


aed_model_def: ModelDef[Model]
aed_model_def.behavior_version = 21
aed_model_def.backend = "torch"
aed_model_def.batch_size_factor = _batch_size_factor


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


def compute_aed_ctc_kd_loss(aed_out: torch.Tensor, ctc_output: torch.Tensor, top_k: int,
                    targets,
                    targets_w_bos,
                    targets_w_eos,
                    target_lengths,
                    input_lengths,
                    eos_mask=True,
                    bos_idx=0,
                    eos_idx=0,
                    blank_idx=10025,
                    backward=True,):
    #shape of aed and ctc?
    batch_size = aed_out.shape[0]
    aed_out_top_k, top_k_list = torch.topk(aed_out, top_k, dim=-1)
    aed_out_top_k = aed_out_top_k[:,:-1,:]
    top_k_list = top_k_list[:, :-1, :]
    ctc_log_prob = ctc_output.log_softmax(dim=-1)


    gamma_fw, (gamma_bw, fw_bw) = ctc_forward(
        log_probs=ctc_log_prob.transpose(0, 1),
        targets=targets,  # (B, S)
        targets_w_bos=targets_w_bos,  # (B S+1)
        targets_w_eos=targets_w_eos,  # (B, S+1)
        input_lengths=input_lengths,  # (B,)
        target_length=target_lengths,  # (B,)
        blank_idx=blank_idx,
        eos_idx=eos_idx,
        bos_idx=bos_idx,
        log_zero=-1e25,  # maybe better than float min for preventing overflowing
        backward=backward,
        top_k_list=top_k_list, )

    if eos_mask:
        fw_bw_eos_mask = (top_k_list != eos_idx).float()
        fw_bw_eos_log_mask = -1e25 * (1. - fw_bw_eos_mask)  # unmasked pos 0, masked pos log_zero
        aed_out_top_k = aed_out_top_k + fw_bw_eos_log_mask
        fw_bw = fw_bw + fw_bw_eos_log_mask
    else:
        assert False # not implemented yet
    aed_out_top_k_renorm = aed_out_top_k.log_softmax(dim=-1)
    fw_bw_renorm = fw_bw.log_softmax(dim=-1)
    fwbw_kl_loss = torch.nn.functional.kl_div(input=fw_bw_renorm,  target=aed_out_top_k_renorm, reduction='none', log_target=True)
    target_mask = get_seq_mask(seq_lens=target_lengths, max_seq_len=targets.shape[1],
                               device=fwbw_kl_loss.device)
    if eos_mask:
        target_mask = target_mask.unsqueeze(-1) * fw_bw_eos_mask
    fwbw_kl_loss = fwbw_kl_loss * target_mask

    return fwbw_kl_loss, gamma_fw









def aed_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim, global_train_step):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)

    aed_ctc_kd_loss = config.bool("aed_ctc_kd_loss", False)
    ctc_kd_layer = config.typed_value("ctc_kd_layer", 12)
    kd_warmup_steps = config.typed_value("ctc_warmup_steps", 2000)
    kd_target_detach = config.bool("kd_target_detach", False)
    kd_scale = config.typed_value("kd_scale", 1.0)
    top_k = config.typed_value("top_k", 10)
    use_normalized_loss = config.bool("use_normalized_loss", True)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    collected_outputs = {}
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            if ctc_kd_layer == layer_idx and aed_ctc_kd_loss:
                ctc_aux_logits = aux_logits
                # compute the loss by ctc fwbw?
            else:
                aux_loss = rf.ctc_loss(
                    logits=aux_logits,
                    targets=targets,
                    input_spatial_dim=enc_spatial_dim,
                    targets_spatial_dim=targets_spatial_dim,
                    blank_index=model.blank_idx,
                )
                aux_loss.mark_as_loss(
                    f"ctc_{layer_idx}",
                    scale=aux_loss_scales[i],
                    custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                    use_normalized_loss=use_normalized_loss,
                )
            # decoded, decoded_spatial_dim = rf.ctc_greedy_decode(aux_logits, in_spatial_dim=enc_spatial_dim)
            # error = rf.edit_distance(
            #     a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
            # )
            # error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=targets_spatial_dim.get_size_tensor())

    batch_dims = data.remaining_dims(data_spatial_dim)
    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx, out_dims=[targets_w_eos_spatial_dim]
    )

    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
    )

    if aed_ctc_kd_loss:

        torch_aed_logits = logits.raw_tensor
        torch_ctc_logits = ctc_aux_logits.raw_tensor # for now, by default the last aux loss layer
        torch_targets = targets.raw_tensor
        torch_targets_w_eos = targets_w_eos.raw_tensor
        torch_targets_w_bos = input_labels.raw_tensor
        torch_target_lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
        torch_input_lengths = enc_spatial_dim.dyn_size_ext.raw_tensor

        aed_logit_scale = config.typed_value("aed_logit_scale", 1.0)
        torch_aed_logits = torch_aed_logits * aed_logit_scale
        if kd_target_detach:
            torch_aed_logits = torch_aed_logits.detach()

        kd_w_bw = config.bool("kd_w_bw", True)

        kd_loss, gamma_fw = compute_aed_ctc_kd_loss(torch_aed_logits, torch_ctc_logits, top_k,
        torch_targets,
        targets_w_bos=torch_targets_w_bos,
        targets_w_eos=torch_targets_w_eos,
        target_lengths=torch_target_lengths,
        input_lengths=torch_input_lengths,
        eos_mask = True,
        bos_idx = model.bos_idx,
        eos_idx = model.eos_idx,
        blank_idx = model.blank_idx,
        backward=kd_w_bw)
        batch_size = torch_aed_logits.shape[0]

        final_score = gamma_fw[torch_input_lengths-1, torch.arange(batch_size), :, torch_target_lengths] # shape (B,2)?
        final_score = final_score.logsumexp(dim=-1)
        step_kd_scale = min(global_train_step/ kd_warmup_steps, 1.0) * kd_scale
        step_ctc_scale = 1.0 - step_kd_scale


        rf.get_run_ctx().mark_as_loss(
            name="aed_kd_loss",
            loss=kd_loss.sum(),
            scale=step_kd_scale,
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )
        if step_ctc_scale > 0:
            rf.get_run_ctx().mark_as_loss(
                name=f"ctc_{ctc_kd_layer}",
                loss=-final_score.sum(),
                scale=step_ctc_scale,
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )

    logits_packed, pack_dim = rf.pack_padded(
        logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    targets_packed, _ = rf.pack_padded(
        targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    )

    log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
    log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
    loss = rf.cross_entropy(
        target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
    )
    loss.mark_as_loss("ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

    best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
    frame_error = best != targets_packed
    frame_error.mark_as_loss(name="fer", as_error=True)


aed_training: TrainDef[Model]
aed_training.learning_rate_control_error_measure = "ce"


def model_recog(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12
    length_normalization_exponent = 1.0
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    decoder_state = model.decoder.default_initial_state(batch_dims=batch_dims_)
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        logits, decoder_state = model.decoder(
            target,
            spatial_dim=single_step_dim,
            encoder=enc,
            state=decoder_state,
        )
        label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=-1.0e30),
            label_log_prob,
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, model.target_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        decoder_state = tree.map_structure(functools.partial(_gather_backrefs, backrefs=backrefs), decoder_state)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        if i > 1 and length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= rf.where(
                ended,
                (i / (i - 1)) ** length_normalization_exponent,
                1.0,
            )

    if i > 0 and length_normalization_exponent != 0:
        seq_log_prob *= (1 / i) ** length_normalization_exponent

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = None
model_recog.batch_size_dependent = False  # not totally correct, but we treat it as such...


def _gather_backrefs(s, *, backrefs: Tensor):
    if isinstance(s, Tensor):
        if backrefs.sparse_dim in s.dims:
            return rf.gather(s, indices=backrefs)  # really the default case
        return s  # e.g. scalar or so, independent from beam
    if isinstance(s, Dim):
        assert s.dimension or backrefs not in s.dyn_size_ext.dims  # currently not supported, also not expected
        return s
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


# class Model(rf.Module):
#     """Model definition"""
#
#     def __init__(
#         self,
#         in_dim: Dim,
#         *,
#         num_enc_layers: int = 12,
#         num_dec_layers: int = 6,
#         target_dim: Dim,
#         wb_target_dim: Optional[Dim] = None,
#         blank_idx: int,
#         eos_idx: int,
#         bos_idx: int,
#         enc_aux_logits: Sequence[int] = (),  # layers
#         enc_model_dim: Dim = Dim(name="enc", dimension=512),
#         dec_model_dim: Dim = Dim(name="dec", dimension=512),
#         enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
#         enc_att_num_heads: int = 4,
#         enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
#         enc_dropout: float = 0.1,
#         enc_att_dropout: float = 0.1,
#         model_args: dict = {},
#     ):
#         super(Model, self).__init__()
#
#         from returnn.config import get_global_config
#
#         config = get_global_config(return_empty_if_none=True)
#
#         enc_layer_drop = config.float("enc_layer_drop", 0.0)
#         if enc_layer_drop:
#             enc_sequential = functools.partial(SequentialLayerDrop, layer_drop=enc_layer_drop)
#         else:
#             enc_sequential = rf.Sequential
#         dec_layer_drop = config.float("dec_layer_drop", 0.0)
#         if dec_layer_drop:
#             dec_sequential = functools.partial(SequentialLayerDrop, layer_drop=dec_layer_drop)
#         else:
#             dec_sequential = rf.Sequential
#
#         self.in_dim = in_dim
#         self.encoder = ConformerEncoder(
#             in_dim,
#             enc_model_dim,
#             ff_dim=enc_ff_dim,
#             input_layer=ConformerConvSubsample(
#                 in_dim,
#                 out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
#                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
#                 pool_sizes=[(1, 2)],
#                 strides=[(1, 1), (3, 1), (2, 1)],
#             ),
#             encoder_layer_opts=enc_conformer_layer_opts,
#             num_layers=num_enc_layers,
#             num_heads=enc_att_num_heads,
#             dropout=enc_dropout,
#             att_dropout=enc_att_dropout,
#             sequential=enc_sequential,
#         )
#         self.decoder = TransformerDecoder(
#             num_layers=num_dec_layers,
#             encoder_dim=enc_model_dim,
#             vocab_dim=target_dim,
#             model_dim=dec_model_dim,
#             sequential=dec_sequential,
#         )
#
#         self.target_dim = target_dim
#         self.blank_idx = blank_idx
#         self.eos_idx = eos_idx
#         self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx
#
#         if enc_aux_logits:
#             if not wb_target_dim:
#                 wb_target_dim = target_dim + 1
#         for i in enc_aux_logits:
#             setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))
#
#         if model_args is not None:
#             ctc_output_args = model_args.get("ctc_output_args", None)
#         else:
#             ctc_output_args = None
#         if ctc_output_args is not None:
#             if not wb_target_dim:
#                 wb_target_dim = target_dim + 1
#             ctc_enc_layer_id = ctc_output_args.get("ctc_enc_layer_id", 12)
#             ctc_output_layer_name = f"enc_aux_logits_{ctc_enc_layer_id}" # same name as enc_aux_logits
#             if int(ctc_enc_layer_id) not in enc_aux_logits:
#                 setattr(self, ctc_output_layer_name, rf.Linear(self.encoder.out_dim, wb_target_dim))
#             self.ctc_output_layer = getattr(self, ctc_output_layer_name)
#
#             self.ctc_enc_layer_id = str(int(ctc_enc_layer_id)-1)
#         else:
#             if enc_aux_logits:
#                 last_ctc_layer_id = str(enc_aux_logits[-1])
#                 self.ctc_output_layer = getattr(self, f"enc_aux_logits_{last_ctc_layer_id}")
#                 self.ctc_enc_layer_id = str(enc_aux_logits[-1]-1)
#             else:
#                 # no ctc output
#                 self.ctc_output_layer = None
#                 self.ctc_enc_layer_id = None
#
#         if not wb_target_dim:
#             self.target_dim_w_blank = wb_target_dim
#         else:
#             self.target_dim_w_blank = target_dim + 1
#
#
#         self.feature_batch_norm = None
#         if config.bool("feature_batch_norm", False):
#             self.feature_batch_norm = rf.BatchNorm(self.in_dim, affine=False, use_mask=True)
#         self.feature_norm = config.bool("feature_norm", False)
#         self.feature_stats = None
#         feature_stats = config.typed_value("feature_stats")
#         if feature_stats:
#             assert isinstance(feature_stats, dict)
#             self.feature_stats = rf.ParameterList(
#                 {
#                     k: rf.Parameter(
#                         rf.convert_to_tensor(numpy.loadtxt(v), dims=[self.in_dim], dtype=rf.get_default_float_dtype()),
#                         auxiliary=True,
#                     )
#                     for k, v in feature_stats.items()
#                 }
#             )
#
#         self._specaugment_opts = {
#             "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
#             "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
#             "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
#             or (_log_mel_feature_dim // 5),
#             "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
#         }
#
#         self._mixup = None
#         if config.typed_value("mixup", None) is not None:
#             from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts
#
#             self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))
#
#     def encode(
#         self,
#         source: Tensor,
#         *,
#         in_spatial_dim: Dim,
#         collected_outputs: Optional[Dict[str, Tensor]] = None,
#     ) -> Tuple[rf.State, Dim]:
#         """encode, and extend the encoder output for things we need in the decoder"""
#         # log mel filterbank features
#         source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
#             source,
#             in_spatial_dim=in_spatial_dim,
#             out_dim=self.in_dim,
#             sampling_rate=16_000,
#         )
#         if self.feature_batch_norm:
#             source = self.feature_batch_norm(source)
#         if self.feature_norm:
#             source = rf.normalize(source, axis=in_spatial_dim)
#         if self.feature_stats:
#             source = (source - self.feature_stats.mean) / self.feature_stats.std_dev
#         if self._mixup:
#             source = self._mixup(source, spatial_dim=in_spatial_dim)
#         # SpecAugment
#         source = rf.audio.specaugment(
#             source,
#             spatial_dim=in_spatial_dim,
#             feature_dim=self.in_dim,
#             **self._specaugment_opts,
#         )
#         # Encoder including convolutional frontend
#         enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
#         return self.decoder.transform_encoder(enc, axis=enc_spatial_dim), enc_spatial_dim
