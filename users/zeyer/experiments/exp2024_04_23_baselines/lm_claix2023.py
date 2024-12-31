"""
LM CLAIX 2023 experiments
"""

from __future__ import annotations

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear

from .configs import (
    config_24gb_v6,
    _get_cfg_lrlin_oclr_by_bs_nep_v3,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
)
from .lm import lm_train_def, lm_model_def

from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset
from i6_experiments.users.zeyer.train_v4 import train, ModelDefWithCfg

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder


def py():
    # ----- LM experiments -----

    # Note: We had the batch_size wrong initially with batch size factor.
    # I think 20k without factor is reasonable with bf16 AMP.
    # with pure bf16, 30k seems to be fine.

    # 35.3 PPL (!!!)
    train(
        "lm/trafo-n96-d512-gelu-drop0-b400_20k-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(20_000, 100, batch_size_factor=1),
                "max_seqs": 400,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            },
        ),
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k", train_epoch_split=20),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=96,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
    )

    # Prepare some small baseline setup. 38.69 PPL, 166_408 steps, 27.1h
    # (trafo-n24-d512-gelu-drop0-b2k_80k-spm10k)
    # (trafo-n24-d512-gelu-drop0-b100_5k on older GPUs with smaller batch size and float32 results in 38.66 PPL)
    # Note: Batch size very large, GPU not used optimally due to laplace too small, lots of padding.
    # However, larger laplace (see below laplace100k) is unstable...
    # (trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-shuffleBatch100-nEp{n_full_ep}-spm10k)
    # 5: 39.85 PPL, 117_362 steps, 16.1h
    # 6: 39.43
    # 7: 39.15 PPL, 164_306 steps, 23.3h
    # 10: 38.88 PPL, 234_732 steps, 33.1h
    train(
        "lm/trafo-n24-d512-gelu-drop0-b2k_80k-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 100, batch_size_factor=1),
                "max_seqs": 2_000,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            },
        ),
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k", train_epoch_split=20),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
    )

    # Use small batch size (b100_5k), just for reference (will be slow, totally underutilizing the GPU...).
    # Note, in lm.py, there is "trafo-n24-d512-gelu-drop0-b100_5k", with the differences:
    # - multi GPU (4 GPUs), param sync after 100 steps
    # - it trains 100 sub-epochs, epoch split 20, but that is on 4 GPUs, so effectively 5*4=20 full epochs (!)
    # - float32
    # -> 38.66 PPL
    # (trafo-n24-d512-gelu-drop0-b100_5k-spm10k) Here with 5 full epochs, we get 39.24 PPL.
    train(
        "lm/trafo-n24-d512-gelu-drop0-b100_5k-nEp20-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(5_000, 400, batch_size_factor=1),
                "max_seqs": 100,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            },
        ),
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k", train_epoch_split=20),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
    )

    # trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-spm10k:
    #   train_sort_laplace_num_seqs larger.
    #   -> total num steps: 117,362
    #   -> 560 sec / subep
    #   -> unstable training, 41.91 PPL (bad).
    #   -> see shuffleBatch100 below
    #   baseline without laplace100k: 38.69 PPL, stable (first subep already 213.2),
    #     total num steps 166,408, 974 sec / subep

    # Again just for reference, small batch size (b100_5k) with laplace100k + shuffleBatch100.
    train(
        "lm/trafo-n24-d512-gelu-drop0-b100_5k-laplace100k-shuffleBatch100-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(5_000, 100, batch_size_factor=1),
                "max_seqs": 100,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "online_shuffle_batches": 100,
            },
        ),
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
    )

    # For grad accum dtype, in RETURNN, create returnn/torch/optim/accum_grads_f32.py with:
    """    
    https://github.com/epfLLM/Megatron-LLM/blob/806a83302cbf4f7d6e147fe34bad5885cf745709/megatron/model/distributed.py#L159-L171

    TODO ...
      unclear:
      - optimizer state is also in bfloat16?
      - without grad accum / distributed training, does this has any effect?
      - the grad_accum, why exactly do we need to keep the reference?
      - is it always the same grad_accum for every train step? so internally it keeps the ref? or because we keep it?
      - so, if grad accum is not relevant, what is actually relevant to make bf16 training work well?
    """
    # TODO ...

    # Different lr, wd
    # (trafo-n24-d512-gelu-drop0-lr{...}-wd{...}-b2k_80k-laplace100k-spm10k)
    # lr, wd:
    # (1e-3, 1e-1),  # 42.43 PPL, unstable
    # (1e-3, 1e-2),  # 41.91 PPL, unstable
    # (1e-3, 1e-3),  # 42.19 PPL, unstable
    # (5e-4, 1e-2),  # 42.55 PPL, unstable

    # Try not-normalized (use_normalized_loss=False): 40.6 PPL, unstable training.
    # Note that the grad norm should be much larger, and thus grad clip is quite different...
    # (Use post_config log_grad_norm:True to check. Should not be much overhead, so can be used in general.)
    # Baseline without lossNoNorm: 41.9 PPL, unstable training.
    # Baseline without laplace100k, without lossNoNorm: 38.69 PPL, stable (first subep already 213.2)

    # Normalize by num seqs, sum over frames.
    # (trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-spm10k-lossSeqNorm)
    # ("use_normalized_loss": "seqs")
    # (Note: small bug, now the exp loss is not correctly calculated...)
    # -> CE 3.693, 40.16 PPL, unstable (vs baseline use_normalized_loss:True, CE 3.735, 41.91 PPL, unstable)
    # Same with shuffleBatch100: -> 38.85 PPL, stable (vs 39.85 PPL, stable) (!)
    # Note: grad clip not adapted here! Grad norm ep1: 43.63, final: 4.63
    train(
        "lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-shuffleBatch100-spm10k-lossSeqNorm",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 100, batch_size_factor=1),
                "max_seqs": 2_000,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "use_normalized_loss": "seqs",
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Again no norm (lossNoNorm).
    train(
        "lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-shuffleBatch100-spm10k-lossNoNorm",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 100, batch_size_factor=1),
                "max_seqs": 2_000,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "use_normalized_loss": False,
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Shuffle batches.
    # -> solves the stability issues!
    # Small laplace1k baseline: 38.69 PPL, stable, 975 sec / subep
    # 1 (no shuffling, baseline): 41.91 PPL, unstable, 563 sec / subep
    # 2: 41.66 PPL, unstable, 572 sec / subep
    # 10: 40.27 PPL, mostly stable, 581 sec / subep
    # 100: 39.85 PPL, stable, 580 sec / subep
    for shuffle_batches in [2, 10, 100]:
        train(
            f"lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-shuffleBatch{shuffle_batches}-spm10k",
            config=dict_update_deep(
                config_96gb_bf16_accgrad1,
                {
                    **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 100, batch_size_factor=1),
                    "max_seqs": 2_000,
                    "optimizer.weight_decay": 1e-2,
                    "calculate_exp_loss": True,
                    "online_shuffle_batches": shuffle_batches,
                },
            ),
            post_config={"log_grad_norm": True},
            train_dataset=get_librispeech_lm_dataset(
                vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
            ),
            model_def=ModelDefWithCfg(
                lm_model_def,
                {
                    "_model_def_dict": rf.build_dict(
                        TransformerDecoder,
                        encoder_dim=None,
                        num_layers=24,
                        model_dim=512,
                        ff_activation=rf.build_dict(rf.gelu),
                        dropout=0.0,
                        att_dropout=0.0,
                    )
                },
            ),
            train_def=lm_train_def,
            # avoid oom
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    from returnn.util.math import PiecewiseLinear, StepFunction

    # Try warmup of batch size (warmupBs).
    # "batch_size": PiecewiseLinear({0: 1_000, 5: 80_000}, kw_name="epoch_continuous", ignore_other_kwargs=True)
    # shuffleBatch1 -> 43.05 PPL. unstable. (vs 41.91 PPL without warmupBs, also unstable.)
    # shuffleBatch100 -> 39.87 PPL, stable (vs 39.85 PPL without warmupBs, stable), initial convergence faster

    # Less grad clip.
    # (trafo-n24-d512-gelu-drop0-gradClip10-b2k_80k-laplace100k-spm10k-lossNoNorm)
    # (But values are way off with this lossNoNorm...)
    # 5 -> 40.64 PPL, unstable training (baseline)
    # 10 -> 40.51 PPL, unstable training

    # Different/Less/More grad clip, no lossNoNorm.
    # 0.1 -> 40.76 PPL, unstable training
    # 1 -> 41.62 PPL, unstable training
    # 5 -> 41.91 PPL, unstable training
    # 20 -> 42.05 PPL, very unstable...
    #   (and avg grad norm way lower, starts at 1.5, goes fast down to 0.3, so rarely has an effect)
    # Now same with shuffleBatch100 (all are stable unless otherwise noted).
    # 1e-4 -> 39.00 PPL (grad norm ep1: 1.983)
    # 1e-3 -> 39.30 PPL (grad norm ep1: 1.98997)
    # 0.01 -> 39.01 PPL (grad norm ep1: 1.984)
    # 0.1 -> 39.32 PPL, faster convergence than grad clip 5, (grad norm ep1: 1.984)
    # 1 -> 39.77 PPL (grad norm ep1: 1.976)
    # 5 -> 39.85 PPL (grad norm ep1: 1.805)
    for grad_clip in [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 5.0]:
        train(
            f"lm/trafo-n24-d512-gelu-drop0-gradClip{str(grad_clip).replace('-', '_')}"
            f"-b2k_80k-laplace100k-shuffleBatch100-spm10k",
            config=dict_update_deep(
                config_96gb_bf16_accgrad1,
                {
                    **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 100, batch_size_factor=1),
                    "max_seqs": 2_000,
                    "gradient_clip_global_norm": grad_clip,
                    "optimizer.weight_decay": 1e-2,
                    "calculate_exp_loss": True,
                    "online_shuffle_batches": 100,
                },
            ),
            post_config={"log_grad_norm": True},
            train_dataset=get_librispeech_lm_dataset(
                vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
            ),
            model_def=ModelDefWithCfg(
                lm_model_def,
                {
                    "_model_def_dict": rf.build_dict(
                        TransformerDecoder,
                        encoder_dim=None,
                        num_layers=24,
                        model_dim=512,
                        ff_activation=rf.build_dict(rf.gelu),
                        dropout=0.0,
                        att_dropout=0.0,
                    )
                },
            ),
            train_def=lm_train_def,
            # avoid oom
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
        )
        del grad_clip

    # Try less seqs in batch, b1k_80k (max_seqs=1_000) instead of b2k_80k. Baseline (b2k_80k) has 39.01 PPL.
    # (lm/trafo-n24-d512-gelu-drop0-gradClip0.01-b1k_80k-laplace100k-shuffleBatch100-spm10k)
    # -> 39.36 PPL.

    # Try longer training.
    # (trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-nEp{n_full_ep}-spm10k-lossNoNorm)
    # 5: 40.639, 6: 40.289, 7: 39.967, 8: 39.783. All still unstable.
    # Now with shuffleBatch100, without lossNoNorm.
    # Note, grad clip not consistent in comparison to baseline due to different batch size.
    # Note, baseline without laplace100k, without shuffleBatch100, with nEp5: 38.69 PPL, 166_408 steps, 27.1h
    # (trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-shuffleBatch100-nEp{n_full_ep}-spm10k)
    # 5: 39.85 PPL, 117_362 steps, 16.1h
    # 6: 39.43
    # 7: 39.15 PPL, 164_306 steps, 23.3h
    # 10: 38.88 PPL, 234_732 steps, 33.1h

    # Maybe more, or slower warmup (lrLowWarm)? Or smaller batch size initially?
    # (trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-lrLowWarm-spm10k-lossNoNorm)
    # -> 40.437 PPL with wd:1e-1. unstable
    # -> 40.61 PPL with wd:1e-2. unstable

    # trafo-n24-d512-gelu-drop0-wd1e_3-b2k_80k-laplace100k-spm10k-lossNoNorm:
    #   Less weight decay wd=1e-3. 41.0 PPL, unstable training.
    #   Baseline with wd=1e-2: 40.6 PPL, unstable training.
    #   Baseline with wd=1e-2, without laplace100k, without lossNoNorm: 38.69 PPL, stable (first subep already 213.2)

    # Try grad accum (accgrad2) (maybe it's bad if there are sometimes big batches with only short seqs).
    # (accum_grad_multiple_step=2)
    # -> 41.30, i.e. slightly worse.

    # Even more grad accum (accgrad100) (no lossNoNorm) such that we cover the laplace100k.
    # (trafo-n24-d512-gelu-drop0-accgrad100-b2k_80k-laplace100k-spm10k)
    # -> 56.60 PPL, but very stable.

    # accgrad2 again, now with shuffleBatch100 and gradClip0.01.
    train(
        f"lm/trafo-n24-d512-gelu-drop0-accgrad2-gradClip0.01-b2k_80k-laplace100k-shuffleBatch100-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 100, batch_size_factor=1),
                "max_seqs": 2_000,
                "gradient_clip_global_norm": 0.01,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "accum_grad_multiple_step": 2,
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # accgrad2 with longer training (nEp10).
    # (trafo-n24-d512-gelu-drop0-accgrad2-gradClip0.01-b2k_80k-laplace100k-nEp10-shuffleBatch100-spm10k)
    # -> 39.01

    # laplace100k is maybe too much. Try laplace10k (train_sort_laplace_num_seqs=10_000).
    # (trafo-n24-d512-gelu-drop0-b2k_80k-accgrad2-laplace10k-spm10k-lossNoNorm)
    # -> 39.42, i.e. much better than laplace100k. (suboptimal here due to accgrad2).
    # laplace1k: 974 sec / subep
    # laplace10k: 582 sec / subep
    # laplace100k: 561 sec / subep

    # laplace10k again, no shuffling, no accgrad. (But note: accgrad2 was slightly better here than accgrad1.)
    # (trafo-n24-d512-gelu-drop0-b2k_80k-laplace10k-spm10k) -> 39.52
    # laplace1k: 38.69, laplace10k: 39.52, laplace100k: 41.91
    # Now without laplace at all (laplaceNone):
    # (trafo-n24-d512-gelu-drop0-b2k_80k-laplaceNone-spm10k) -> 38.45 PPL.
    train(
        "lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplaceNone-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 100, batch_size_factor=1),
                "max_seqs": 2_000,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=None
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # laplace10k with shuffling batches.
    # online_shuffle_batches=10: 39.55 PPL
    # Now try 100 (trafo-n24-d512-gelu-drop0-b2k_80k-laplace10k-shuffleBatch100-spm10k): 39.41 PPL

    # laplace100k in the first 90% epochs, then disable laplace for remaining 10%.
    # (baseline, 100% laplace100k: 39.85 PPL)
    # (trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k_10None-shuffleBatch100-spm10k) 39.55
    # Try beginning and end of the training.
    # train_sort_order=StepFunction(
    #   {9: "random", 10: "laplace:.100000", 90: "laplace:.100000", 91: "random"},
    #   kw_name="epoch", ignore_other_kwargs=True)
    # -> 39.9 PPL
    # Alternate laplace and random (laplace100k_altNone).
    # (trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k_altNone-shuffleBatch100-spm10k)
    # train_sort_order=StepFunction(
    #   {i: {0: "laplace:.100000", 1: "random"}[i % 2] for i in range(1, 101)},
    #   kw_name="epoch", ignore_other_kwargs=True)
    # -> 39.11 PPL, weirdly PPL always goes slightly up every second epoch, but not much.
    # But what about just training half the epochs with random sorting then?
    train(
        "lm/trafo-n24-d512-gelu-drop0-b2k_80k-random-nEp2.5-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 50, batch_size_factor=1),
                "max_seqs": 2_000,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k", train_epoch_split=20, train_sort_order="random"),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    from .optim_ext.zipformer_scaled_adam import ScaledAdam

    # ScaledAdam (optScaledAdam).
    # ("optimizer.class": rf.build_dict(ScaledAdam)["class"], "optimizer.clipping_scale": 2.0)
    # trafo-n24-d512-gelu-drop0-b2k_80k-optScaledAdam-laplace10k-shuffleBatch10-spm10k: 57.62
    # Scaled Adam with higher LR (lr1e_2).
    # trafo-n24-d512-gelu-drop0-b2k_80k-optScaledAdam-lr1e_2-laplace10k-shuffleBatch10-spm10k: 40.23
    # Even higher, default from Icefall (0.045).
    # trafo-n24-d512-gelu-drop0-b2k_80k-optScaledAdam-lr0.045-laplace10k-shuffleBatch10-spm10k: 40.38 PPL
    # TODO what now? tune LR further? tune LR schedule? some of the other hyper params?
    # Try scaling also low, lowest LR.
    train(
        "lm/trafo-n24-d512-gelu-drop0-b2k_80k-optScaledAdam-lr1e_2a-laplace10k-shuffleBatch10-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(
                    80_000, 100, batch_size_factor=1, peak_lr=1e-2, low_lr=1e-4, lowest_lr=1e-5
                ),
                "max_seqs": 2_000,
                "optimizer.class": rf.build_dict(ScaledAdam)["class"],
                "optimizer.clipping_scale": 2.0,
                "calculate_exp_loss": True,
                "online_shuffle_batches": 10,
            },
            [
                # ScaledAdam does not have weight decay (??) (TODO...)
                "optimizer.weight_decay",
                "optimizer.weight_decay_modules_blacklist",
            ],
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=10_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Try some dropout (dropout + att_dropout).
    # (trafo-n24-d512-gelu-drop0.1-gradClip0.01-b2k_80k-laplace100k-shuffleBatch100-spm10k)
    # -> 41.05 PPL (vs 39.01 PPL), so much worse

    # Try Lion.
    # Baseline without Lion: 40.6 PPL, also unstable (due to large batch & laplace100k).
    # Baseline without Lion, without laplace100k, without lossNoNorm: 38.69 PPL, stable (first subep already 213.2)
    #   lion_lr_factor, wd: (lossNoNorm, laplace100k, no shuffleBatch100)
    #         (0.05, 1e-2),  # 42.2 PPL. Unstable training.
    #         (0.1, 1e-2),  # 41.2 PPL. Unstable training.
    #         (1.0, 1e-2),  # broken
    #         # (0.3, 1e-2),  # 43.7 PPL. Unstable training.
    #         # (0.3, 1e-3),  # 43.9 PPL. Unstable training.
    # Now shuffleBatch100, gradClip0.01, no lossNoNorm: baseline (no optLion): 39.01 PPL
    #   lion_lr_factor, wd:
    #     (0.1, 1e-2) -> 39.89 PPL
    for lion_lr_factor, wd in [
        (0.1, 1e-2),
    ]:
        wd = wd / lion_lr_factor
        wd = round(wd, 6)
        train(
            f"lm/trafo-n24-d512-gelu-drop0-gradClip0.01"
            f"-optLion-lr{str(1e-3 * lion_lr_factor).replace('-', '_')}-wd{wd}"
            f"-b2k_80k-laplace100k-shuffleBatch100-spm10k",
            config=dict_update_deep(
                config_96gb_bf16_accgrad1,
                {
                    "calculate_exp_loss": True,
                    **_get_cfg_lrlin_oclr_by_bs_nep_v3(
                        80_000,
                        100,
                        batch_size_factor=1,
                        peak_lr=1e-3 * lion_lr_factor,
                        low_lr=1e-5 * lion_lr_factor,
                        lowest_lr=1e-6 * lion_lr_factor,
                    ),
                    "max_seqs": 2_000,
                    "optimizer.class": "returnn.torch.optim.lion.Lion",
                    "optimizer.weight_decay": wd,
                    "gradient_clip_global_norm": 0.01,
                    "online_shuffle_batches": 100,
                },
                ["optimizer.epsilon"],  # no eps in Lion
            ),
            train_dataset=get_librispeech_lm_dataset(
                vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
            ),
            model_def=ModelDefWithCfg(
                lm_model_def,
                {
                    "_model_def_dict": rf.build_dict(
                        TransformerDecoder,
                        encoder_dim=None,
                        num_layers=24,
                        model_dim=512,
                        ff_activation=rf.build_dict(rf.gelu),
                        dropout=0.0,
                        att_dropout=0.0,
                    )
                },
            ),
            train_def=lm_train_def,
            # avoid oom
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
        )

    # Try RAdam. -> 42.10 PPL, unstable. (vs 41.91 PPL with AdamW)
    # (trafo-n24-d512-gelu-drop0-optRAdam-b2k_80k-laplace100k-spm10k)
    # (..., "optimizer.class": "RAdam", "optimizer.decoupled_weight_decay": True, ...)
    # Now again, with shuffleBatch100, lrNoWarmup. -> 39.62 PPL, stable (vs 39.85 PPL with AdamW, LR warmup, stable)
    # TODO is this now the shorter (no) LR warmup? AdamW with shorter LR warmup?
    n_ep = 100
    peak_lr, low_lr, lowest_lr = 1e-3, 1e-5, 1e-6
    train(
        f"lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-optRAdam-lrNoWarmup-shuffleBatch100-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                "__num_epochs": n_ep,
                "batch_size": 80_000,
                "max_seqs": 2_000,
                "learning_rate": 1.0,
                "dynamic_learning_rate": dyn_lr_piecewise_linear,
                "learning_rate_piecewise_by_epoch_continuous": True,
                "learning_rate_piecewise_steps": [0.45 * n_ep, 0.9 * n_ep, n_ep],
                "learning_rate_piecewise_values": [peak_lr, peak_lr, low_lr, lowest_lr],
                "optimizer.class": "RAdam",
                "optimizer.decoupled_weight_decay": True,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # RAdam, gradClip0.01.
    # 39.39
    train(
        f"lm/trafo-n24-d512-gelu-drop0-gradClip0.01-b2k_80k-laplace100k-optRAdam-lrNoWarmup-shuffleBatch100-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                "__num_epochs": n_ep,
                "batch_size": 80_000,
                "max_seqs": 2_000,
                "learning_rate": 1.0,
                "dynamic_learning_rate": dyn_lr_piecewise_linear,
                "learning_rate_piecewise_by_epoch_continuous": True,
                "learning_rate_piecewise_steps": [0.45 * n_ep, 0.9 * n_ep, n_ep],
                "learning_rate_piecewise_values": [peak_lr, peak_lr, low_lr, lowest_lr],
                "optimizer.class": "RAdam",
                "optimizer.decoupled_weight_decay": True,
                "optimizer.weight_decay": 1e-2,
                "gradient_clip_global_norm": 0.01,
                "calculate_exp_loss": True,
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    from .optim_ext.adopt import ADOPT

    # ADOPT optimizer (optAdopt).
    # 40.89 PPL (vs AdamW 39.85) (but different betas=(0.9, 0.9999))
    # Now again with same betas as our AdamW setup.
    peak_lr, low_lr, lowest_lr = 1e-3, 1e-5, 1e-6
    train(
        f"lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-optAdopt-shuffleBatch100-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                "__num_epochs": n_ep,
                "batch_size": 80_000,
                "max_seqs": 2_000,
                "learning_rate": 1.0,
                "dynamic_learning_rate": dyn_lr_piecewise_linear,
                "learning_rate_piecewise_by_epoch_continuous": True,
                "learning_rate_piecewise_steps": [0.45 * n_ep, 0.9 * n_ep, n_ep],
                "learning_rate_piecewise_values": [low_lr, peak_lr, low_lr, lowest_lr],
                "optimizer.class": rf.build_dict(ADOPT)["class"],
                "optimizer.decoupled": True,
                "optimizer.weight_decay": 1e-2,
                # Adopt defaults: betas: Tuple[float, float] = (0.9, 0.9999), eps: float = 1e-6,
                # AdamW defaults: betas: Tuple[float, float] = (0.9, 0.999),, eps: float = 1e-8,
                # But we anyway overwrite epsilon as 1e-16.
                "optimizer.betas": (0.9, 0.999),
                "calculate_exp_loss": True,
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    from .optim_ext.soap import SOAP

    # Try SOAP.
    peak_lr, low_lr, lowest_lr = 1e-3, 1e-5, 1e-6
    train(
        f"lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-optSoap-shuffleBatch100-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                "__num_epochs": n_ep,
                "batch_size": 80_000,
                "max_seqs": 2_000,
                "learning_rate": 1.0,
                "dynamic_learning_rate": dyn_lr_piecewise_linear,
                "learning_rate_piecewise_by_epoch_continuous": True,
                "learning_rate_piecewise_steps": [0.45 * n_ep, 0.9 * n_ep, n_ep],
                "learning_rate_piecewise_values": [low_lr, peak_lr, low_lr, lowest_lr],
                "optimizer.class": rf.build_dict(SOAP)["class"],
                # Suggested in repo (https://github.com/nikhilvyas/SOAP/tree/main):
                "optimizer.betas": (0.95, 0.95),
                "optimizer.precondition_frequency": 10,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Try Lamb.
    peak_lr, low_lr, lowest_lr = 1e-3, 1e-5, 1e-6
    train(
        f"lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-optLamb-shuffleBatch100-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                "__num_epochs": n_ep,
                "batch_size": 80_000,
                "max_seqs": 2_000,
                "learning_rate": 1.0,
                "dynamic_learning_rate": dyn_lr_piecewise_linear,
                "learning_rate_piecewise_by_epoch_continuous": True,
                "learning_rate_piecewise_steps": [0.45 * n_ep, 0.9 * n_ep, n_ep],
                "learning_rate_piecewise_values": [low_lr, peak_lr, low_lr, lowest_lr],
                "optimizer.class": "torch_optimizer.Lamb",  # pip install torch_optimizer
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Try Shampoo.
    peak_lr, low_lr, lowest_lr = 1e-3, 1e-5, 1e-6
    train(
        f"lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-optShampoo-shuffleBatch100-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                "__num_epochs": n_ep,
                "batch_size": 80_000,
                "max_seqs": 2_000,
                "learning_rate": 1.0,
                "dynamic_learning_rate": dyn_lr_piecewise_linear,
                "learning_rate_piecewise_by_epoch_continuous": True,
                "learning_rate_piecewise_steps": [0.45 * n_ep, 0.9 * n_ep, n_ep],
                "learning_rate_piecewise_values": [low_lr, peak_lr, low_lr, lowest_lr],
                "optimizer.class": "pytorch_optimizer.Shampoo",  # pip install pytorch-optimizer
                "optimizer.preconditioning_compute_steps": 100,  # takes approx 70 secs!
                "optimizer.weight_decouple": True,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Try AdEMAMix.
    # The alpha (default 5) means that the update is 6 times larger than the normal update, thus divide LR by 6.
    peak_lr, low_lr, lowest_lr = (round(lr / 6, 6) for lr in (1e-3, 1e-5, 1e-6))
    train(
        f"lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-optAdEMAMix-shuffleBatch100-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                "__num_epochs": n_ep,
                "batch_size": 80_000,
                "max_seqs": 2_000,
                "learning_rate": 1.0,
                "dynamic_learning_rate": dyn_lr_piecewise_linear,
                "learning_rate_piecewise_by_epoch_continuous": True,
                "learning_rate_piecewise_steps": [0.45 * n_ep, 0.9 * n_ep, n_ep],
                "learning_rate_piecewise_values": [low_lr, peak_lr, low_lr, lowest_lr],
                "optimizer.class": "pytorch_optimizer.AdEMAMix",  # pip install pytorch-optimizer
                "optimizer.weight_decouple": True,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # TODO try timm.optim.adafactor_bv.AdafactorBigVision

    # bf16A
    # Very bad. Stable training but just bad: 49.38 final PPL (compared to 35.58 PPL)
    # train(
    #     "lm/trafo-n96-d512-gelu-bf16A-drop0-b400_20k-spm10k",
    #     config=dict_update_deep(
    #         config_96gb_bf16_accgrad1,
    #         {
    #             "torch_amp": None,
    #             "default_float_dtype": "bfloat16",
    #             # NOTE: wrong batch size factor
    #             **_get_cfg_lrlin_oclr_by_bs_nep_v3(20_000, 100, batch_size_factor=_batch_size_factor),
    #             "max_seqs": 400,
    #             "optimizer.weight_decay": 1e-2,
    #             "calculate_exp_loss": True,
    #         },
    #     ),
    #     train_dataset=get_librispeech_lm_dataset(vocab="spm10k", train_epoch_split=20),
    #     model_def=ModelDefWithCfg(
    #         lm_model_def,
    #         {
    #             "_model_def_dict": rf.build_dict(
    #                 TransformerDecoder,
    #                 encoder_dim=None,
    #                 num_layers=96,
    #                 model_dim=512,
    #                 ff_activation=rf.build_dict(rf.gelu),
    #                 dropout=0.0,
    #                 att_dropout=0.0,
    #             )
    #         },
    #     ),
    #     train_def=lm_train_def,
    # )

    # bf16A, loss_dtype="float32"
    #   50.38 PPL, even worse with loss_dtype float32 (49.39 PPL, which is much worse than baseline 35.58 PPL)?

    # TODO we could very systematically go through the whole net/model and leave some parts as float32
    # Check other (L)LM frameworks, how they do this.
    # https://github.com/search?q=repo%3AepfLLM%2FMegatron-LLM%20params_dtype&type=code
    # https://github.com/search?q=repo%3AepfLLM%2FMegatron-LLM%20accumulate_allreduce_grads_in_fp32&type=code
    # Grad accum in f32:
    # https://github.com/epfLLM/Megatron-LLM/blob/806a83302cbf4f7d6e147fe34bad5885cf745709/megatron/model/distributed.py#L159-L171
    # Logits get casted to f32 before softmax/CE.
    # https://github.com/epfLLM/Megatron-LLM/blob/main/megatron/model/gpt_model.py
    # https://github.com/epfLLM/Megatron-LLM/blob/main/megatron/core/tensor_parallel/cross_entropy.py

    # Try new model (Llama) (noAbsPos-rmsNorm-ffGated-rope-noBias instead of gelu).
    # 38.71 PPL (vs 39.85 PPL)
    train(
        f"lm/trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b2k_80k-laplace100k-shuffleBatch100-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 100, batch_size_factor=1),
                "max_seqs": 2_000,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    pos_enc=None,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    decoder_layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Llama (noAbsPos-rmsNorm-ffGated-rope-noBias) + optRAdam + lrNoWarmup + warmupBs + lossSeqNorm
    # (trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b2k_80k-warmupBs-laplace100k-optRAdam-lrNoWarmup-shuffleBatch100-spm10k-lossSeqNorm)
    # -> 39.74 PPL (maybe suboptimal grad clip here)
    # Now without lossSeqNorm.
    n_ep = 100
    peak_lr, low_lr, lowest_lr = 1e-3, 1e-5, 1e-6
    train(
        "lm/trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b2k_80k"
        "-warmupBs-laplace100k-optRAdam-lrNoWarmup-shuffleBatch100-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                "__num_epochs": n_ep,
                "batch_size": PiecewiseLinear(
                    {0: 1_000, 5: 80_000}, kw_name="epoch_continuous", ignore_other_kwargs=True
                ),
                "max_seqs": 2_000,
                "learning_rate": 1.0,
                "dynamic_learning_rate": dyn_lr_piecewise_linear,
                "learning_rate_piecewise_by_epoch_continuous": True,
                "learning_rate_piecewise_steps": [0.45 * n_ep, 0.9 * n_ep, n_ep],
                "learning_rate_piecewise_values": [peak_lr, peak_lr, low_lr, lowest_lr],
                "optimizer.class": "RAdam",
                "optimizer.decoupled_weight_decay": True,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "online_shuffle_batches": 100,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(
            vocab="spm10k", train_epoch_split=20, train_sort_laplace_num_seqs=100_000
        ),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    pos_enc=None,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    decoder_layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # TODO init out layer to all zeros

    from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe

    # Get the bpe1k vocab exactly as some others from our group (Mohammad, Robin, ...).
    bpe1k = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=1000)
    bpe1k = Bpe(dim=1056, codes=bpe1k.bpe_codes, vocab=bpe1k.bpe_vocab, eos_idx=0, bos_idx=0, unknown_label="<unk>")
    assert bpe1k.codes.creator.job_id() == "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV"
    train(  # 12.79
        "lm/trafo-n32-d1024-gelu-drop0-b400_20k-bpe1k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(20_000, 100, batch_size_factor=1),
                "max_seqs": 400,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(vocab=bpe1k, train_epoch_split=20),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=32,
                    model_dim=1024,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
    )

    # Llama, laplace100k, batch shuffling:
    #   trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b2k_30k-laplace100k-shuffleBatch100-bpe1k: 12.64
    # with more grad clip:
    #   trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-gradClip0.01-b2k_30k-laplace100k-shuffleBatch100-bpe1k:
    #     12.71

    # laplace10k, less batch shuffling.
    train(
        "lm/trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b2k_30k-laplace10k-shuffleBatch10-bpe1k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(100),
                "batch_size": 30_000,
                "max_seqs": 2000,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "online_shuffle_batches": 10,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(vocab=bpe1k, train_epoch_split=20, train_sort_laplace_num_seqs=10_000),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=32,
                    model_dim=1024,
                    pos_enc=None,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    decoder_layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


# https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/9108f4a6f43c40a3a168919afd36839d/
# TODO check weight decay...
config_96gb_bf16_accgrad1 = dict_update_deep(
    config_24gb_v6,
    {
        "__gpu_mem": 96,
        "__cpu_rqmt": 24,  # the whole c23g node has 96 CPUs, and 4 GPUs
        "__mem_rqmt": 100,  # the whole node should have more than 500GB
        "accum_grad_multiple_step": 1,  # per single GPU
    },
)
