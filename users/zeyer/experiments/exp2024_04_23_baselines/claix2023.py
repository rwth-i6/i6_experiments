"""
Config for RWTH IPC CLAIX-2023 cluster experiments.
"""

from __future__ import annotations

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear

from .configs import (
    config_24gb_v6,
    _get_cfg_lrlin_oclr_by_bs_nep_v3,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    _batch_size_factor,
)
from .aed import train_exp as aed_train_exp
from .ctc import train_exp as ctc_train_exp
from .lm import lm_train_def, lm_model_def

from i6_experiments.users.zeyer.experiments.exp2024_10_16_consistency_reg_ctc import cr_ctc_training
from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset
from i6_experiments.users.zeyer.train_v4 import train, ModelDefWithCfg

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import ConformerEncoderLayer, ConformerPositionwiseFeedForward


def py():
    # Note: From the baseline experiment (optimized for 4x11GB GPUs with float32),
    # we made the following changes (partly taking our 1x24GB GPU settings into account)
    # for the H100 GPU with 96GB memory (nodes c23g in the CLAIX-2023 cluster):
    # - __gpu_mem = 96
    # - batch size was increased to 200k (takes about 60-70GB of GPU memory)
    # - bf16 again (AMP) (also checking pure bf16 now...)
    # - (grad accum 1 (no change actually; and obviously, batch size is already large enough...))
    # - (LR scheduling now based on seq_idx (this is not really related to the new GPU, but just simplifies things))
    # - (weight decay = 1e-2 still, no change so far, but checking now...)
    # - partition epoch to 1 (dataset_train_opts.train_epoch_split=1)
    #   (because the GPU is so fast that it trains a single epoch in 20mins;
    #    otherwise, eval is just too often, takes too much time)
    # - more workers for data loading (__multi_proc_dataset_opts.num_workers=25) (check computation time in log!)
    # - __cpu_rqmt = 24 (the whole c23g node has 96 CPUs, and 4 GPUs)
    # - __mem_rqmt = 100 (the whole node should have more than 500GB)
    # Due to the larger batch size, we have less steps per epoch. With bs200k, it is 2016 steps per full epoch.
    # Baseline: SubEp21 start: 21660, SubEp40 end: 46627, thus 24967 steps per full epoch.
    # (But first full epoch has a bit less due to filtering: 21660)

    # Baseline: v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-spm10k-spmSample07
    # {"dev-clean": 2.35, "dev-other": 4.98, "test-clean": 2.21, "test-other": 5.49}
    # Final 'devtrain_loss_ce': 0.11065730461399945, 'devtrain_loss_fer': 0.006211603513944155,
    # -----

    # Note: epoch filtering is wrong, should not do that for 5 full epochs...
    # {"dev-clean": 2.36, "dev-other": 5.35, "test-clean": 2.4, "test-other": 5.72}
    # Final 'devtrain_loss_ce': 0.11504180265534825, 'devtrain_loss_fer': 0.005836691916394713,
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1},
    )

    # No curriculum learning (epoch filtering) (-> train_epoch_wise_filter=None)
    # {"dev-clean": 2.4, "dev-other": 5.22, "test-clean": 2.5, "test-other": 5.55}
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-noCrl-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )

    # TODO instead of having SpecAugment step-based, we can also make this dependent on the continuous epoch

    # SpecAugment adapted
    # {"dev-clean": 2.32, "dev-other": 5.24, "test-clean": 2.41, "test-other": 5.66}
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-noCrl-specAug2k-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "specaugment_steps": (500, 1_000, 2_000),
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )

    # Test default_float_dtype="bfloat16" (bfloat16A) instead of AMP.
    # (96gb-bf16A-bs200k-accgrad1-wd1e_2-lrlinEpCont-noCrl-specAug2k-speedpertV2-spm10k-spmSample07)
    # Consumes about 40GB of GPU memory.
    # {"dev-clean": 4.37, "dev-other": 10.07, "test-clean": 4.39, "test-other": 10.32}
    # TODO what's the problem?

    # bfloat16A with larger batch.
    # Batch size 400k: OOM after some epochs.
    # bfloat16A with larger batch V2.
    # {"dev-clean": 4.28, "dev-other": 10.35, "test-clean": 4.22, "test-other": 10.08}
    aed_train_exp(
        f"96gb-bf16A-bs300k-bsSeq400-accgrad1-wd1e_2-lrlinEpCont-noCrl-specAug2k-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            "torch_amp": None,
            "default_float_dtype": "bfloat16",
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(300_000, 100, batch_size_factor=_batch_size_factor),
            "max_seqs": 400,
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "specaugment_steps": (500, 1_000, 2_000),
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
    )

    # Higher peak LR (96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-lr1e_2-noCrl-speedpertV2-spm10k-spmSample07)
    # {"dev-clean": 5.62, "dev-other": 12.41, "test-clean": 5.56, "test-other": 12.88} -> bad

    # More weight decay
    # {"dev-clean": 2.4, "dev-other": 5.26, "test-clean": 2.54, "test-other": 5.63}
    # -> unclear, maybe slightly better?
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd5e_2-lrlinEpCont-noCrl-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 5e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )

    # Lion optimizer (https://arxiv.org/abs/2302.06675, https://github.com/lucidrains/lion-pytorch/)
    # {"dev-clean": 2.37, "dev-other": 5.52, "test-clean": 2.5, "test-other": 5.68}
    # (Baseline with Adam: "dev-other": 5.22)
    # TODO maybe needs more tuning? different wd, lr
    lion_lr_factor = 0.3
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd0.0333-lrlinEpCont-lr3e_4-optLion-noCrl-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(
                200_000,
                100,
                peak_lr=1e-3 * lion_lr_factor,
                low_lr=1e-5 * lion_lr_factor,
                lowest_lr=1e-6 * lion_lr_factor,
                batch_size_factor=_batch_size_factor,
            ),
            "optimizer.class": "returnn.torch.optim.lion.Lion",
            "optimizer.weight_decay": 1e-2 / lion_lr_factor,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
        config_deletes=["optimizer.epsilon"],  # no eps in Lion
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )

    for opts in [
        # Baseline (n12) has {"dev-clean": 2.35, "dev-other": 5.65, "test-clean": 2.66, "test-other": 5.94}.
        # CLAIX baseline: {"dev-clean": 2.54, "dev-other": 5.93, "test-clean": 2.68, "test-other": 6.27}
        # CLAIX CR: {"dev-clean": 2.49, "dev-other": 5.99, "test-clean": 2.68, "test-other": 6.05}
        # v6-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001
        {
            "name": "v6-relPosAttDef-noBias-aedLoss-bhv20-96gb-bf16-bs200k-accgrad1-wd1e_2"
            "-lrlinEpCont-featBN-speedpertV2-spm10k-bpeSample001",
            "num_enc_layers": 12,
            "batch_size": 200_000,
            "vocab": "spm10k",
        },
        # Baseline (n16, spm10k) has {"dev-clean": 2.26, "dev-other": 5.44, "test-clean": 2.5, "test-other": 5.62}.
        # v6-n16-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs10k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001
        # This here is now spm512 though.
        # Note: In the original CR paper, they don't have time-downsampling!
        {"num_enc_layers": 16, "batch_size": 10_000, "vocab": "spm512"},  # TODO bad?
        {"num_enc_layers": 12, "batch_size": 200_000, "vocab": "spm512"},  # with CR: 7.82, without: 7.85
    ]:
        for cr_ctc in [None, {"cr_loss_scale": 0.2}]:
            # TODO also adapt specaug for CR...
            use_cr_ctc = cr_ctc is not None
            name = f"cr{use_cr_ctc}"
            if use_cr_ctc:
                name += f"-crLoss{cr_ctc['cr_loss_scale']}"
            name += f"-n{opts['num_enc_layers']}-{opts['vocab']}"
            ctc_train_exp(
                name,
                config_96gb_bf16_accgrad1,
                train_def=cr_ctc_training if use_cr_ctc else None,
                model_config={
                    "enc_conformer_layer": rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                    "feature_batch_norm": True,
                    "num_enc_layers": opts["num_enc_layers"],
                },
                config_updates={
                    **_get_cfg_lrlin_oclr_by_bs_nep_v3(
                        opts["batch_size"] // (2 if use_cr_ctc else 1),
                        100 // (2 if use_cr_ctc else 1),
                        batch_size_factor=_batch_size_factor,
                    ),
                    "optimizer.weight_decay": 1e-2,
                    "__train_audio_preprocess": speed_pert_librosa_config,
                    "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                    # purely used for training
                    "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),
                    **(cr_ctc if use_cr_ctc else {}),
                    **({"aed_loss_bug_fix": True} if use_cr_ctc else {}),
                },
                post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
                vocab=opts["vocab"],
                train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
                dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
                # avoid OOM
                # env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
            )

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

    # Prepare some small baseline setup. 38.69 PPL.
    # Note: Batch size very large, not used optimally due to laplace too small.
    # However, larger laplace (see below laplace100k) is unstable...
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

    # trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-spm10k:
    #   train_sort_laplace_num_seqs larger.
    #   -> total num steps: 117,362
    #   -> 560 sec / subep
    #   -> unstable training, 41.91 PPL (bad).
    #   baseline without laplace100k: 38.69 PPL, stable (first subep already 213.2),
    #     total num steps 166,408, 974 sec / subep
    #   TODO why unstable training and bad? loss reduce type? weight decay too much?

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

    # Try not-normalized (use_normalized_loss=False): 40.6 PPL, unstable training.
    # Note that the grad norm should be much larger, and thus grad clip is quite different...
    # (Use post_config log_grad_norm:True to check. Should not be much overhead, so can be used in general.)
    # Baseline without lossNoNorm: 41.9 PPL, unstable training.
    # Baseline without laplace100k, without lossNoNorm: 38.69 PPL, stable (first subep already 213.2)
    train(
        "lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-spm10k-lossNoNorm",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 100, batch_size_factor=1),
                "max_seqs": 2_000,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "use_normalized_loss": False,
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
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
    )

    from returnn.util.math import PiecewiseLinear

    # Try warmup of batch size (warmupBs).
    train(
        "lm/trafo-n24-d512-gelu-drop0-b2k_80k-warmupBs-laplace100k-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(100),
                "batch_size": PiecewiseLinear(
                    {0: 1_000, 1: 80_000}, kw_name="epoch_continuous", ignore_other_kwargs=True
                ),
                "max_seqs": 2_000,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
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
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
    )

    # Less grad clip.
    # 5 -> 40.64 PPL, unstable training (baseline)
    # 10 -> 40.51 PPL, unstable training
    train(
        "lm/trafo-n24-d512-gelu-drop0-gradClip10-b2k_80k-laplace100k-spm10k-lossNoNorm",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 100, batch_size_factor=1),
                "max_seqs": 2_000,
                "gradient_clip_global_norm": 10.0,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "use_normalized_loss": False,
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

    # Less grad clip, no lossNoNorm.
    train(
        "lm/trafo-n24-d512-gelu-drop0-gradClip20-b2k_80k-laplace100k-spm10k",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, 100, batch_size_factor=1),
                "max_seqs": 2_000,
                "gradient_clip_global_norm": 20.0,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
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

    # Try longer training.
    # 5: 40.639, 6: 40.289, 7: 39.967
    for n_full_ep in [5, 7, 8]:
        train(
            f"lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-nEp{n_full_ep}-spm10k-lossNoNorm",
            config=dict_update_deep(
                config_96gb_bf16_accgrad1,
                {
                    **_get_cfg_lrlin_oclr_by_bs_nep_v3(80_000, n_full_ep * 20, batch_size_factor=1),
                    "max_seqs": 2_000,
                    "optimizer.weight_decay": 1e-2,
                    "calculate_exp_loss": True,
                    "use_normalized_loss": False,
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
            # avoid oom
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
        )

    # Maybe more, or slower warmup (lrLowWarm)? Or smaller batch size initially?
    # -> 40.437 PPL with wd:1e-1. unstable
    # -> 40.61 PPL with wd:1e-2. unstable
    n_ep = 100
    peak_lr, low_lr, lowest_lr = 1e-3, 1e-5, 1e-6
    train(
        f"lm/trafo-n24-d512-gelu-drop0-b2k_80k-laplace100k-lrLowWarm-spm10k-lossNoNorm",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                "__num_epochs": n_ep,
                "batch_size": 80_000,
                "max_seqs": 2_000,
                "learning_rate": 1.0,
                "dynamic_learning_rate": dyn_lr_piecewise_linear,
                "learning_rate_piecewise_by_epoch_continuous": True,
                "learning_rate_piecewise_steps": [0.01 * n_ep, 0.05 * n_ep, 0.5 * n_ep, 0.9 * n_ep, n_ep],
                "learning_rate_piecewise_values": [0.0, lowest_lr, low_lr, peak_lr, low_lr, lowest_lr],
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "use_normalized_loss": False,
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
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
    )

    # trafo-n24-d512-gelu-drop0-wd1e_3-b2k_80k-laplace100k-spm10k-lossNoNorm:
    #   Less weight decay wd=1e-3. 41.0 PPL, unstable training.
    #   Baseline with wd=1e-2: 40.6 PPL, unstable training.
    #   Baseline with wd=1e-2, without laplace100k, without lossNoNorm: 38.69 PPL, stable (first subep already 213.2)

    # Try grad accum (accgrad2) (maybe it's bad if there are sometimes big batches with only short seqs).
    # (accum_grad_multiple_step=2)
    # -> 41.30, i.e. slightly worse.

    # laplace100k is maybe too much. Try laplace10k (train_sort_laplace_num_seqs=10_000).
    # (trafo-n24-d512-gelu-drop0-b2k_80k-accgrad2-laplace10k-spm10k-lossNoNorm)
    # -> 39.42, i.e. much better than laplace100k. (suboptimal here due to accgrad2).

    # Try Lion.
    # Baseline without Lion: 40.6 PPL, also unstable (due to large batch & laplace100k).
    # Baseline without Lion, without laplace100k, without lossNoNorm: 38.69 PPL, stable (first subep already 213.2)
    for lion_lr_factor, wd in [
        (0.05, 1e-2),  # 42.2 PPL. Unstable training.
        (0.1, 1e-2),  # 41.2 PPL. Unstable training.
        # (0.3, 1e-2),  # 43.7 PPL. Unstable training.
        # (0.3, 1e-3),  # 43.9 PPL. Unstable training.
    ]:
        wd = wd / lion_lr_factor
        wd = round(wd, 6)
        train(
            f"lm/trafo-n24-d512-gelu-drop0"
            f"-wd{wd}-lr{str(1e-3 * lion_lr_factor).replace('-', '_')}-optLion"
            f"-b2k_80k-laplace100k-spm10k-lossNoNorm",
            config=dict_update_deep(
                config_96gb_bf16_accgrad1,
                {
                    "calculate_exp_loss": True,
                    "use_normalized_loss": False,
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

    from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe

    # Get the bpe1k vocab exactly as some others from our group (Mohammad, Robin, ...).
    bpe1k = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=1000)
    bpe1k = Bpe(dim=1056, codes=bpe1k.bpe_codes, vocab=bpe1k.bpe_vocab, eos_idx=0, bos_idx=0, unknown_label="<unk>")
    assert bpe1k.codes.creator.job_id() == "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV"
    train(
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
