"""Param Import
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any, Union, Tuple, Dict, Sequence, List
import tree
import math
import numpy as np
import hashlib
import contextlib

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.lr_schedules.lin_warmup_invsqrt_decay import dyn_lr_lin_warmup_invsqrt_decay
from i6_experiments.users.zeyer.lr_schedules.combine_eval import dyn_lr_combine_eval
from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, ModelWithCheckpoint

# From Mohammad, 2023-06-29
# dev-clean  2.27
# dev-other  5.39
# test-clean  2.41
# test-other  5.51
# _returnn_tf_config_filename = (
#   "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config")
# E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
# /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb
# Main train (2035 subepochs): /work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.SAh74CLCNJQi
# 15k batch size, accum grad 2 (1350 steps per epoch?)
# (With batch size 40k (here default), I have usually 495 steps/epoch. Same accum grad 2.)
# peak_lr = 0.9e-3 (1e-3 should also be fine), with Adam, optimizer_epsilon = 1e-08
# phase1: peak_lr / 10 -> peak_lr (45%)
# phase2: peak_lr -> peak_lr / 10 (45%)
# phase3: peak_lr / 10 -> 1e-6 (10%)
# all linear decay and step-based
# specaugment like my orig (same here, specaugorig), speed perturb same here.
# weight decay: L2 1e-4 in some layers (not all): FF, depthwise conv, self-att, output, LSTM, readout
# Final from learning_rates file:
# 2035: EpochData(learningRate=<misleading>, error={
# 'dev_error_ctc': 0.0520755184693418,
# 'dev_error_output/output_prob': 0.035661241551042944,
# 'dev_score_ctc': 0.2796084385705723,
# 'dev_score_output/output_prob': 0.1718613621694714,
# 'devtrain_error_ctc': 0.005757552549708462,
# 'devtrain_error_output/output_prob': 0.005408351877314902,
# 'devtrain_score_ctc': 0.022935187616968285,
# 'devtrain_score_output/output_prob': 0.05237826015574962,
# 'train_error_ctc': 0.05592114304093772,
# 'train_error_output/output_prob': 0.041970552995693494,
# 'train_score_ctc': 0.21249712733341475,
# 'train_score_output/output_prob': 0.20816428663741796,
# }),
# Retrain RETURNN training job (600 subepochs): /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.ZhtaEElHqWlr
# Epoch-wise LR:
# Fixed for 20 subepochs: 1e-4
# Linear(?) decay for remaining (?): 1e-4 to 1e-6
# Final from learning_rates file:
# 600: EpochData(learningRate=1e-06, error={
# 'dev_error_ctc': 0.04999311020358747,
# 'dev_error_output/output_prob': 0.03406881170076022,
# 'dev_score_ctc': 0.2881619431223589,
# 'dev_score_output/output_prob': 0.16851828029171323,
# 'devtrain_error_ctc': 0.003611245977923651,
# 'devtrain_error_output/output_prob': 0.004028583366881808,
# 'devtrain_score_ctc': 0.014762402644778178,
# 'devtrain_score_output/output_prob': 0.0458638666428664,
# 'train_error_ctc': 0.051649620746772214,
# 'train_error_output/output_prob': 0.03977601830532325,
# 'train_score_ctc': 0.19722012590584306,
# 'train_score_output/output_prob': 0.19768974342596793,
# }),


# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix(prefix_name)

    # Moh:      dev-clean  2.27, dev-other  5.39, test-clean  2.41,  test-other  5.51
    # RF recog: {"dev-clean": 2.25, "dev-other": 5.34, "test-clean": 2.42, "test-other": 5.56}
    _recog_imported()

    train_exp("from-scratch-train", config, gpu_mem=11)

    train_exp(  # dev-other 7.6
        "base-24gb-bs30k-f32",
        config_24gb,
        config_updates={"batch_size": 30_000 * _batch_size_factor},
        config_deletes=["torch_amp"],
    )

    train_exp("base-24gb-v2-lr1e_3", config_24gb_v2, config_updates={"learning_rate": 0.001})  # dev-other 7.44
    train_exp(  # dev-other 7.24
        "base-24gb-v2-lr1e_3-nogradscaler", config_24gb_v2, config_updates={"learning_rate": 0.001, "grad_scaler": None}
    )

    # base-24gb-v3: diverges at later point
    train_exp(  # 7.01, slightly better than baseline.
        "base-24gb-v3-lr1e_3-wd1e_3",
        config_24gb_v3,
        config_updates={"learning_rate": 0.001, "optimizer.weight_decay": 0.001},
    )
    train_exp("base-24gb-v3-adam", config_24gb_v3, config_updates={"optimizer.class": "adam"})  # 7.56
    train_exp(  # dev-other 7.01 (epoch 1964)
        "base-24gb-v3-lr1e_3",
        config_24gb_v3,
        config_updates={"learning_rate": 0.001},
        fine_tune=[
            # (ep 1280 itself is dev-other 7.34)
            (1280, {"num_epochs": 50}),  # 7.22
            (1280, {"num_epochs": 100}),  # 7.08
            (1280, {"num_epochs": 200}),  # 7.03
            # (ep 2000 is 7.03, 7.31)
            (2000, {"num_epochs": 100}),  # 6.93, 7.12
            (2000, {"num_epochs": 200}),  # dev-other 6.84, test-other 7.06
            (2000, {"num_epochs": 200, "final_lr": 1e-6}),  # dev-other* 6.83, test-other 7.10
            (2000, {"num_epochs": 200, "lr_decay_type": "linspace"}),  # 6.95, 7.11
            (2000, {"num_epochs": 200, "lr_decay_type": "linspace", "final_lr": 1e-6}),  # dev-ot 6.94, test-other* 7.01
            (
                2000,
                {
                    "num_epochs": 200,
                    "_lr_decay_type": "L3_5_150_L7_50",  # dev-other 6.89, test-other 7.20
                    "learning_rates": list(np.linspace(1e-3, 1e-5, num=150)) + list(np.linspace(1e-5, 1e-7, num=50)),
                },
            ),
            (
                2000,
                {
                    "num_epochs": 200,
                    "_lr_decay_type": "L3_5_180_L6_20",  # dev-other 6.87, test-other 7.12
                    "learning_rates": list(np.linspace(1e-3, 1e-5, num=180)) + list(np.linspace(1e-5, 1e-6, num=20)),
                },
            ),
        ],
    )
    train_exp(  # dev/test-other 6.89,7.39 (overfitting on dev? base: dev/test 7.01,7.23). unclear...
        "base-24gb-v3-lr1e_3-lrdecnorm40k",
        config_24gb_v3,
        config_updates={"learning_rate": 0.001, "learning_rate_invsqrt_norm": 40_000},
    )
    train_exp(  # 6.22 (vs base 7.01, so much better)
        "base-24gb-v3-lr1e_3-specaugorig",
        config_24gb_v3,
        config_updates={"learning_rate": 0.001},
        config_deletes=[
            "specaugment_num_spatial_mask_factor",
            "specaugment_max_consecutive_feature_dims",
        ],
    )
    train_exp(  # 8.21 (vs base 7.01, so lossscalesF is worse)
        "base-24gb-v3-lr1e_3-lossscalesF",
        config_24gb_v3,
        config_updates={"learning_rate": 0.001, "aux_loss_scales": [0.1, 0.2], "aed_loss_scale": 0.7},
    )

    train_exp("base-24gb-v3-lr1e_3-wdblacklist", config_24gb_v4)  # 7.07 (vs base 7.01, so worse?)
    train_exp(  # 7.07
        "base-24gb-v4",
        config_24gb_v4,
        fine_tune=[
            (2000, {"num_epochs": 200, "final_lr": 1e-6}),  # 6.84
        ],
    )
    train_exp(  # 6.85 (vs base 7.07), so better, but maybe just because too less regularization in general
        "base-24gb-v4-wdblacklist2",
        config_24gb_v4,
        config_updates={
            "optimizer.weight_decay_modules_blacklist": [
                # Difference to base-24gb-v4: weight decay also for LayerNorm and BatchNorm.
                # The initial thought was that we maybe do it wrong, and it applies to the statistics as well.
                # This is not the case, it only applies on the learnable parameters,
                # and there it makes sense to apply weight decay.
                "rf.Embedding",
                "rf.LearnedRelativePositionalEncoding",
            ],
        },
    )
    train_exp("base-24gb-v4-lr09e_3", config_24gb_v4, config_updates={"learning_rate": 0.0009})  # 6.99 (vs base 7.07)
    train_exp(  # 7.46 (vs base 7.07, so worse)
        "base-24gb-v4-lrcos",
        config_24gb_v4,
        config_updates={
            "dynamic_learning_rate": dyn_lr_combine_eval,
            "learning_rate_eval": "orig * (np.cos(global_train_step / 10_000 * 2 * np.pi) * 0.49995 + 0.50005)",
            "learning_rate_eval_locals": {"orig": dyn_lr_lin_warmup_invsqrt_decay},
        },
    )
    train_exp(  # 6.48 (vs base 7.07, so much better, but this is already with fine-tuning included here)
        "base-24gb-v4-lrlin",
        config_24gb_v4,
        config_updates={
            "learning_rate": 1.0,
            "dynamic_learning_rate": dyn_lr_piecewise_linear,
            # total steps after 2000 epochs: 982.312
            "learning_rate_piecewise_steps": [20_000, 900_000, 982_000],
            "learning_rate_piecewise_values": [0.0, 1e-3, 1e-5, 1e-6],
        },
    )
    train_exp(
        "base-24gb-v4-lrlin1e_5_450k",
        config_24gb_v4,
        config_updates={
            "learning_rate": 1.0,
            "dynamic_learning_rate": dyn_lr_piecewise_linear,
            # total steps after 2000 epochs: 982.312
            "learning_rate_piecewise_steps": [450_000, 900_000, 982_000],
            "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
        },
    )
    # gn01 ("gradient_noise": 0.1), does not converge? that was with old RETURNN, gn after grad clip
    # gn01 before grad clip (new RETURNN) also does not converge.
    # train_exp(
    #     "base-24gb-v4-lrlin-gn01",
    #     config_24gb_v4,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [20_000, 900_000, 982_000],
    #         "learning_rate_piecewise_values": [0.0, 1e-3, 1e-5, 1e-6],
    #         "gradient_noise": 0.1,  # TODO schedule it, only later, gradually more...?
    #     },
    # )
    train_exp(  # 7.08 (vs base 7.07, so unclear)
        "base-24gb-v4-pretrainBug",
        config_24gb_v4,
        config_updates={"pretrain_opts": {"steps": {4 * 500: {"num_layers": 8}, 8 * 500: {"num_layers": 2}}}},
    )
    train_exp(  # 7.30 (vs base 7.07, so worse), much more overfitting
        "base-24gb-v4-pretrain",
        config_24gb_v4,
        config_updates={
            "pretrain_opts": {
                "steps": [(8 * 500, {"num_layers": 2}), (4 * 500, {"num_layers": 4}), (4 * 500, {"num_layers": 8})]
            }
        },
    )
    train_exp(  # 7.02 (vs base 7.07) but dev-clean, test-other worse, unclear
        "base-24gb-v4-posdrop01", config_24gb_v4, config_updates={"pos_emb_dropout": 0.1}
    )
    train_exp(  # 6.52 (vs base 7.07, so much better)
        "base-24gb-v4-pretrain-posdrop01-specaugorig",
        config_24gb_v4,
        config_updates={
            "pretrain_opts": {
                "steps": [(8 * 500, {"num_layers": 2}), (4 * 500, {"num_layers": 4}), (4 * 500, {"num_layers": 8})]
            },
            "pos_emb_dropout": 0.1,
        },
        config_deletes=[
            "specaugment_num_spatial_mask_factor",
            "specaugment_max_consecutive_feature_dims",
        ],
    )
    train_exp(  # 7.02 (vs base 7.07)
        "base-24gb-v4-attdropfixbc", config_24gb_v4, config_updates={"rf_att_dropout_broadcast": False}
    )
    train_exp("base-24gb-v4-bs30k", config_24gb_v4, config_updates={"batch_size": 30_000 * _batch_size_factor})
    train_exp(  # 7.19
        "base-24gb-v4-bs30k-accgrad3",  # accgrad3 instead of (base) accgrad2
        config_24gb_v4,
        config_updates={"batch_size": 30_000 * _batch_size_factor, "accum_grad_multiple_step": 3},
    )
    train_exp(  # 7.21, so nodropbc makes it worse
        "base-24gb-v4-bs30k-accgrad3-nodropbc",
        config_24gb_v4,
        config_updates={
            "batch_size": 30_000 * _batch_size_factor,
            "accum_grad_multiple_step": 3,  # because of reduced batch size
            "rf_dropout_broadcast": False,  # needs more memory, thus reduced batch size
        },
    )

    train_exp("base-24gb-v5", config_24gb_v5)  # 6.35
    train_exp("base-24gb-v5-embInit1", config_24gb_v5, config_updates={"embed_init_stddev": 1.0})  # 6.51
    train_exp(  # 6.36, no effect at all?
        "base-24gb-v5-mixup",
        config_24gb_v5,
        config_updates={"mixup": {}},
        # got GPU OOM in subepoch 1823... so play around here to fix this
        post_config_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"},  # might be even faster?
    )

    train_exp("base-24gb-v6", config_24gb_v6)  # 6.30
    train_exp(  # 5.44
        "base-24gb-v6-lrlin1e_5_450k",
        config_24gb_v6,
        config_updates={
            "learning_rate": 1.0,
            "dynamic_learning_rate": dyn_lr_piecewise_linear,
            # total steps after 2000 epochs: 982.312
            "learning_rate_piecewise_steps": [450_000, 900_000, 982_000],
            "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
        },
    )

    train_exp(
        "v6-11gb-f32-bs15k-accgrad4-mgpu2",
        config_11gb_v6_f32_bs15k_accgrad4_mgpu,
        num_processes=2,  # multi-GPU
    )
    train_exp(  # 5.44
        "v6-11gb-f32-bs15k-accgrad4-mgpu4-lrlin1e_5_295k",
        config_11gb_v6_f32_bs15k_accgrad4_mgpu,
        config_updates={
            **_cfg_lrlin1e_5_295k,  # total steps after 500 epochs: ~652k
        },
        num_processes=4,  # multi-GPU
        num_epochs=500,  # because of multi-GPU, 1 subepoch here is like 4 subepochs in single-GPU
    )
    train_exp(
        "v6-11gb-f32-bs15k-accgrad4-mgpu4-wd1e_4-lrlin1e_5_295k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_wd1e_4_lrlin1e_5_295k,
        config_updates={"accum_grad_multiple_step": 4},
    )
    train_exp(
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-wd1e_4-lrlin1e_5_295k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_wd1e_4_lrlin1e_5_295k,
    )
    train_exp(
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg4-wd1e_4-lrlin1e_5_295k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_wd1e_4_lrlin1e_5_295k,
        config_updates={
            "torch_distributed": {"reduce_type": "param", "param_sync_step": 4},  # multi-GPU
        },
    )
    train_exp(  # 5.63
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg10-wd1e_4-lrlin1e_5_295k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_wd1e_4_lrlin1e_5_295k,
        config_updates={
            "torch_distributed": {"reduce_type": "param", "param_sync_step": 10},  # multi-GPU
        },
    )
    train_exp(  # 5.53, so better than p10?
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_295k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_wd1e_4_lrlin1e_5_295k,
        config_updates={
            "torch_distributed": {"reduce_type": "param", "param_sync_step": 100},  # multi-GPU
        },
    )
    train_exp(
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg500-wd1e_4-lrlin1e_5_295k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_wd1e_4_lrlin1e_5_295k,
        config_updates={
            "torch_distributed": {"reduce_type": "param", "param_sync_step": 500},  # multi-GPU
        },
    )
    train_exp(
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg1000-wd1e_4-lrlin1e_5_295k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_wd1e_4_lrlin1e_5_295k,
        config_updates={
            "torch_distributed": {"reduce_type": "param", "param_sync_step": 1000},  # multi-GPU
        },
    )

    # TODO pretrain with specaugment_steps=(0, 15k, 25k)?


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from .sis_setup import get_prefix_for_config

        prefix_name = get_prefix_for_config(__file__)
    global _sis_prefix
    _sis_prefix = prefix_name


def _recog_imported():
    from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
    from ._moh_att_2023_06_30_import import map_param_func_v2
    from i6_core.returnn.training import Checkpoint as TfCheckpoint, PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import ConvertTfCheckpointToRfPtJob

    task = _get_ls_task()
    extern_data_dict = task.train_dataset.get_extern_data()
    default_target_key = task.train_dataset.get_default_target()
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    target_dim = targets.feature_dim_or_sparse_dim

    new_chkpt_path = ConvertTfCheckpointToRfPtJob(
        checkpoint=TfCheckpoint(index_path=generic_job_output(_returnn_tf_ckpt_filename)),
        make_model_func=MakeModel(
            in_dim=_log_mel_feature_dim,
            target_dim=target_dim.dimension,
            eos_label=_get_eos_idx(target_dim),
        ),
        map_func=map_param_func_v2,
    ).out_checkpoint
    new_chkpt = PtCheckpoint(new_chkpt_path)
    model_with_checkpoint = ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_chkpt)

    _recog("recog_results", model_with_checkpoint)


def _recog(name: str, model_with_checkpoint: ModelWithCheckpoint):
    from sisyphus import tk
    from i6_experiments.users.zeyer.recog import recog_model

    task = _get_ls_task()

    res = recog_model(task, model_with_checkpoint, model_recog)
    tk.register_output(_sis_prefix + "/" + name, res.output)


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    *,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    fine_tune: Optional[Union[int, List[Tuple[int, Dict[str, Any]]]]] = None,
    time_rqmt: Optional[int] = None,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from .train import train
    from i6_experiments.users.zeyer.recog import recog_training_exp

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    task = _get_ls_task()
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    if "__num_epochs" in config:
        num_epochs = config.pop("__num_epochs")
    if "__gpu_mem" in config:
        gpu_mem = config.pop("__gpu_mem")
    if "__num_processes" in config:
        num_processes = config.pop("__num_processes")

    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        model_def=from_scratch_model_def,
        train_def=from_scratch_training,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else "mpirun",
        time_rqmt=time_rqmt,
    )
    recog_training_exp(prefix, task, model_with_checkpoint, recog_def=model_recog)

    if fine_tune:
        if isinstance(fine_tune, int):
            fine_tune = [(fine_tune, {})]
        for ep, opts in fine_tune:
            assert isinstance(ep, int) and isinstance(opts, dict)
            suffix = f"/finetune/{ep}"
            opts = opts.copy()
            if opts:
                for k, v in sorted(opts.items()):
                    k: str
                    suffix += "-" + k.lstrip("_")
                    v = str(v).replace("-", "_")
                    if len(v) > 16 and not k.startswith("_"):
                        suffix += "_" + hashlib.md5(v.encode("utf8")).hexdigest()[:8]
                    else:
                        suffix += v
            num_epochs_ = opts.pop("num_epochs", 50)
            config_ = config.copy()
            config_["import_model_train_epoch1"] = model_with_checkpoint.get_epoch(ep).checkpoint
            config_.pop("dynamic_learning_rate")
            lrs = opts.pop("learning_rates", None)
            if lrs is None:
                lr_decay_type = opts.pop("lr_decay_type", "geomspace")  # geomspace or linspace
                lr_decay_func = getattr(np, lr_decay_type)
                lr = config_["learning_rate"]
                final_lr = opts.pop("final_lr", 1e-7)
                lrs = list(lr_decay_func(lr, final_lr, num=num_epochs_))
            else:
                assert isinstance(lrs, (list, tuple))
                assert len(lrs) == num_epochs_
            config_["learning_rates"] = lrs
            config_["learning_rate"] = float(lrs[-1])
            config_["specaugment_steps"] = (0, 0, 0)
            config_.update({k: v for k, v in opts.items() if not k.startswith("_")})

            finetune_model_with_ckpt = train(
                prefix + suffix,
                task=task,
                config=config_,
                post_config=post_config,
                model_def=from_scratch_model_def,
                train_def=from_scratch_training,
                num_epochs=num_epochs_,
                gpu_mem=gpu_mem,
            )
            # _recog(name + suffix + "/recog/last", finetune_model_with_ckpt.get_last_fixed_epoch())
            recog_training_exp(prefix + suffix, task, finetune_model_with_ckpt, recog_def=model_recog)

    return model_with_checkpoint


_ls_task = None


def _get_ls_task():
    global _ls_task
    if _ls_task:
        return _ls_task

    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_bpe10k_raw

    _ls_task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)
    return _ls_task


py = sis_run_with_prefix  # if run directly via `sis m ...`

_batch_size_factor = 160

config = dict(
    batching="laplace:.1000",
    batch_size=15_000 * _batch_size_factor,
    max_seqs=200,
    max_seq_length_default_target=75,
    specaugment_steps=(10_000, 20_000, 40_000),
    # gradient_clip=0,
    # gradient_clip_global_norm = 1.0
    optimizer={
        "class": "adamw",
        "epsilon": 1e-8,
        "weight_decay": 1e-6,
    },
    accum_grad_multiple_step=4,
    # gradient_noise=0.0,
    learning_rate=0.0025,
    dynamic_learning_rate=dyn_lr_lin_warmup_invsqrt_decay,
    learning_rate_warmup_steps=40_000,
    learning_rate_invsqrt_norm=40_000,
    aux_loss_layers=[4, 8],
)
post_config = dict(
    cleanup_old_models=dict(keep_last_n=5),
    torch_dataloader_opts=dict(num_workers=1),
)

config_24gb = config.copy()
config_24gb.update(
    dict(
        torch_amp="bfloat16",
        batch_size=40_000 * _batch_size_factor,
        accum_grad_multiple_step=2,
        learning_rate=0.002,
        learning_rate_warmup_steps=20_000,
        learning_rate_invsqrt_norm=20_000,
        specaugment_steps=(5_000, 15_000, 25_000),
    )
)
# base-24gb (using config_24gb): converged, but stagnated, and hiccups

config_24gb_v2 = dict_update_deep(
    config_24gb,
    {
        "optimizer.epsilon": 1e-16,
        "specaugment_num_spatial_mask_factor": 200,
        "specaugment_max_consecutive_feature_dims": 10,
    },
)

config_24gb_v3 = config_24gb_v2.copy()
config_24gb_v3.update(
    dict(
        learning_rate=0.0025,
        grad_scaler=None,
        gradient_clip_global_norm=5.0,
    )
)

config_24gb_v4 = dict_update_deep(
    config_24gb_v3,
    {
        "learning_rate": 0.001,
        "optimizer.weight_decay_modules_blacklist": [
            "rf.BatchNorm",  # unclear if really good
            "rf.LayerNorm",  # unclear if really good
            "rf.Embedding",
            "rf.LearnedRelativePositionalEncoding",
        ],
    },
)

config_24gb_v5 = dict_update_deep(
    config_24gb_v4,
    {
        "pretrain_opts": {  # pretrain
            "steps": [(8 * 500, {"num_layers": 2}), (4 * 500, {"num_layers": 4}), (4 * 500, {"num_layers": 8})]
        },
        "pos_emb_dropout": 0.1,  # posdrop01
        "optimizer.weight_decay_modules_blacklist": [  # wdblacklist2
            "rf.Embedding",
            "rf.LearnedRelativePositionalEncoding",
        ],
        "rf_att_dropout_broadcast": False,  # attdropfixbc
    },
    [
        # specaugorig
        "specaugment_num_spatial_mask_factor",
        "specaugment_max_consecutive_feature_dims",
    ],
)

config_24gb_v6 = dict_update_deep(config_24gb_v5, None, ["pretrain_opts"])

_cfg_lrlin1e_5_295k = {  # for bs15k, mgpu4
    "learning_rate": 1.0,
    "dynamic_learning_rate": dyn_lr_piecewise_linear,
    # total steps after 500 epochs: ~652k
    "learning_rate_piecewise_steps": [295_000, 590_000, 652_000],
    "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
}

config_11gb_v6_f32_bs15k_accgrad4_mgpu = dict_update_deep(
    config_24gb_v6,
    {
        "batch_size": 15_000 * _batch_size_factor,  # ~1305 steps/epoch
        "accum_grad_multiple_step": 4,  # per single GPU
        "torch_distributed": {},  # multi-GPU
        "__gpu_mem": 11,
    },
    [
        "torch_amp",  # f32
    ],
)
config_11gb_v6_f32_bs15k_accgrad1_mgpu = dict_update_deep(
    config_11gb_v6_f32_bs15k_accgrad4_mgpu,
    {
        "accum_grad_multiple_step": 1,
    },
)
config_11gb_v6_f32_bs15k_accgrad1_mgpu4_wd1e_4_lrlin1e_5_295k = dict_update_deep(
    config_11gb_v6_f32_bs15k_accgrad1_mgpu,
    {
        "optimizer.weight_decay": 1e-4,
        **_cfg_lrlin1e_5_295k,
        "__num_processes": 4,  # multi-GPU
        "__num_epochs": 500,  # because of multi-GPU, 1 subepoch here is like 4 subepochs in single-GPU
    },
)

# TODO lrlin
# TODO lr09e_3


class MakeModel:
    """for import"""

    def __init__(self, in_dim: int, target_dim: int, *, eos_label: int = 0, num_enc_layers: int = 12):
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.eos_label = eos_label
        self.num_enc_layers = num_enc_layers

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(name="target", dimension=self.target_dim, kind=Dim.Types.Feature)
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(in_dim, target_dim, num_enc_layers=self.num_enc_layers)

    @classmethod
    def make_model(
        cls, in_dim: Dim, target_dim: Dim, *, num_enc_layers: int = 12, pos_emb_dropout: float = 0.0, **extra
    ) -> Model:
        """make"""
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
            **extra,
        )


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        l2: float = 0.0001,
    ):
        super(Model, self).__init__()

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        self.in_dim = in_dim
        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        self.enc_ctx = rf.Linear(self.encoder.out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = Dim(name="enc_win_dim", dimension=5)

        self.inv_fertility = rf.Linear(self.encoder.out_dim, att_num_heads, with_bias=False)

        self.target_embed = rf.Embedding(target_dim, Dim(name="target_embed", dimension=640))
        if config.float("embed_init_stddev", None):
            self.target_embed.weight.initial = rf.init.Normal(stddev=config.float("embed_init_stddev", 0.0))

        self.s = rf.ZoneoutLSTM(
            self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
            Dim(name="lstm", dimension=1024),
            zoneout_factor_cell=0.15,
            zoneout_factor_output=0.05,
            use_zoneout_output=False,  # like RETURNN/TF ZoneoutLSTM old default
            # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
            # parts_order="ifco",
            parts_order="jifo",  # NativeLSTM (the code above converts it...)
            forget_bias=0.0,  # the code above already adds it during conversion
        )

        self.weight_feedback = rf.Linear(att_num_heads, enc_key_total_dim, with_bias=False)
        self.s_transformed = rf.Linear(self.s.out_dim, enc_key_total_dim, with_bias=False)
        self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
        self.readout_in = rf.Linear(
            self.s.out_dim + self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
            Dim(name="readout", dimension=1024),
        )
        self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)

        for p in self.parameters():
            p.weight_decay = l2

        if enc_aux_logits:
            if not wb_target_dim:
                wb_target_dim = target_dim + 1
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        self._pretrain_opts: Optional[Dict[str, Any]] = config.typed_value("pretrain_opts")

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
            log_base=math.exp(2.3026),  # almost 10.0 but not exactly...
        )
        if self._mixup:
            source = self._mixup(source, spatial_dim=in_spatial_dim)
        # SpecAugment
        source = rf.audio.specaugment(
            source,
            spatial_dim=in_spatial_dim,
            feature_dim=self.in_dim,
            **self._specaugment_opts,
        )
        # Encoder including convolutional frontend
        with _opt_apply_pretrain_to_encoder(self.encoder, collected_outputs, self._pretrain_opts):
            enc, enc_spatial_dim = self.encoder(
                source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
            )
        enc_ctx = self.enc_ctx(enc)
        inv_fertility = rf.sigmoid(self.inv_fertility(enc))
        return dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility), enc_spatial_dim

    def decoder_default_initial_state(self, *, batch_dims: Sequence[Dim], enc_spatial_dim: Dim) -> rf.State:
        """Default initial state"""
        state = rf.State(
            s=self.s.default_initial_state(batch_dims=batch_dims),
            att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]),
            accum_att_weights=rf.zeros(
                list(batch_dims) + [enc_spatial_dim, self.att_num_heads], feature_dim=self.att_num_heads
            ),
        )
        state.att.feature_dim_axis = len(state.att.dims) - 1
        return state

    def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
        """loop step out"""
        return {
            "s": Tensor(
                "s", dims=batch_dims + [self.s.out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1
            ),
            "att": Tensor(
                "att",
                dims=batch_dims + [self.att_num_heads * self.encoder.out_dim],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
            ),
        }

    def loop_step(
        self,
        *,
        enc: rf.Tensor,
        enc_ctx: rf.Tensor,
        inv_fertility: rf.Tensor,
        enc_spatial_dim: Dim,
        input_embed: rf.Tensor,
        state: Optional[rf.State] = None,
    ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
        """step of the inner loop"""
        if state is None:
            batch_dims = enc.remaining_dims(
                remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
            )
            state = self.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
        state_ = rf.State()

        prev_att = state.att

        s, state_.s = self.s(rf.concat_features(input_embed, prev_att), state=state.s, spatial_dim=single_step_dim)

        weight_feedback = self.weight_feedback(state.accum_att_weights)
        s_transformed = self.s_transformed(s)
        energy_in = enc_ctx + weight_feedback + s_transformed
        energy = self.energy(rf.tanh(energy_in))
        att_weights = rf.softmax(energy, axis=enc_spatial_dim)
        state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5
        att0 = rf.dot(att_weights, enc, reduce=enc_spatial_dim, use_mask=False)
        att0.feature_dim = self.encoder.out_dim
        att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder.out_dim))
        state_.att = att

        return {"s": s, "att": att}, state_

    def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
        """logits for the decoder"""
        readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
        readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
        readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
        logits = self.output_prob(readout)
        return logits


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


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    return MakeModel.make_model(
        in_dim, target_dim, enc_aux_logits=enc_aux_logits or (), pos_emb_dropout=pos_emb_dropout
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def from_scratch_training(
    *, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
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
    input_embeddings = model.target_embed(targets)
    input_embeddings = rf.shift_right(input_embeddings, axis=targets_spatial_dim, pad_value=0.0)

    def _body(input_embed: Tensor, state: rf.State):
        new_state = rf.State()
        loop_out_, new_state.decoder = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=state.decoder,
        )
        return loop_out_, new_state

    loop_out, _, _ = rf.scan(
        spatial_dim=targets_spatial_dim,
        xs=input_embeddings,
        ys=model.loop_step_output_templates(batch_dims=batch_dims),
        initial=rf.State(
            decoder=model.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim),
        ),
        body=_body,
    )

    logits = model.decode_logits(input_embed=input_embeddings, **loop_out)
    logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False)
    targets_packed, _ = rf.pack_padded(
        targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
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


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


@contextlib.contextmanager
def _opt_apply_pretrain_to_encoder(
    encoder: ConformerEncoder, collected_outputs: Optional[Dict[str, Tensor]], pretrain_opts: Optional[Dict[str, Any]]
):
    """Function is run within RETURNN."""
    if not pretrain_opts:
        yield
        return
    step = rf.get_run_ctx().step
    steps: Union[Sequence[Tuple[int, Dict[str, Any]]], Dict[int, Dict[str, Any]]] = pretrain_opts["steps"]
    if isinstance(steps, (list, tuple)):
        steps_ = {}
        step_bound = 0
        for step_bound_rel, opts in steps:
            step_bound += step_bound_rel
            steps_[step_bound] = opts
        steps = steps_
    assert isinstance(steps, dict)
    for step_bound, opts in sorted(steps.items()):
        if step < step_bound:
            assert isinstance(opts, dict)
            opts_ = opts.copy()
            # somewhat hacky but that is still the easiest way I can think of, without touching a lot of other code
            pretrain_num_layers = opts_.pop("num_layers")
            assert not opts_, f"unhandled opts: {opts_} in opts {opts} for step bound {step_bound}"
            orig_layers = encoder.layers[:]
            del encoder.layers[pretrain_num_layers:]
            yield
            encoder.layers[:] = orig_layers
            if collected_outputs is not None:
                assert len(collected_outputs) == pretrain_num_layers
                for i in range(pretrain_num_layers, len(orig_layers)):
                    collected_outputs[str(i)] = collected_outputs[str(pretrain_num_layers - 1)]
            return
    yield
    return


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
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
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
    decoder_state = model.decoder_default_initial_state(batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        if i == 0:
            input_embed = rf.zeros(batch_dims_ + [model.target_embed.out_dim], feature_dim=model.target_embed.out_dim)
        else:
            input_embed = model.target_embed(target)
        step_out, decoder_state = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=decoder_state,
        )
        logits = model.decode_logits(input_embed=input_embed, **step_out)
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
        decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), decoder_state)
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
# output_blank_label=blank is actually wrong for AED, but now we don't change it anymore
# because it would change all recog hashes.
# Also, it does not matter too much -- it will just cause an extra SearchRemoveLabelJob,
# which will not have any effect here.
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False
