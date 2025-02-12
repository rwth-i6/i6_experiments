"""Copied from Albert Zeyer 25.03.2024, then modified
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
import math
import numpy as np
import hashlib
import copy
import contextlib
import functools
import itertools
from sisyphus import tk

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample


# from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, from_scratch_training_kldiv_sample_batch
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog
from i6_experiments.users.phan.rf_models.default_model_configs import default_extern_lm_config, default_ilm_config
from i6_experiments.users.phan.rf_models.lstm_lm import get_model, train_step

from i6_experiments.users.yang.torch.ctc_ilm_kd.trafo_lm import get_model, train_step, default_trafo_transcription_config
# trafo
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.model_with_checkpoints import (
    ModelWithCheckpoints,
    ModelWithCheckpoint,
)

from i6_core.returnn.training import PtCheckpoint



# config_11gb = copy.deepcopy(config_11gb)
# config_11gb.pop("dynamic_learning_rate", None)
# config_11gb.pop("learning_rate_piecewise_steps", None)
# config_11gb.pop("learning_rate_piecewise_values", None)
# config_11gb.pop("learning_rate_invsqrt_norm", None)
# config_11gb.pop("learning_rate_warmup_steps", None)


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix()
    from i6_experiments.users.yang.torch.ctc_ilm_kd.lbs.transcription_lm_train_config import train_lbs_bpe10k_transcription_lm, lbs_bpe10k_trans_lm_adam_config

    # train_job_1 = train_lbs_bpe10k_transcription_lm(
    #     get_model,
    #     train_step,
    #     num_epochs=10,
    #     hashed_config={
    #         "lm_cfg": default_trafo_transcription_config,
    #         "train_epoch_split":10,
    #     }
    # )
    # train_job_1.add_alias(_sis_prefix + "/train_debug")
    # tk.register_output(_sis_prefix + "/bpe10k_transcription_lm_debug/learning_rates", train_job_1.out_learning_rates)

    # adam optimizer, oclr schedule

    from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear

    def _get_cfg_lrlin_oclr(batch_size: int, num_epochs: int, peak_percentage: float =0.6, epoch_split: int = 10, peak_lr: float = 1e-3, total_steps=None):
        # lbs transcript, batch size 1000, around 17021 steps for one full epoch

        full_epochs = num_epochs//epoch_split
        if total_steps is None:
            total_steps = int(full_epochs * 17021 * 1000//batch_size)
        else:
            total_steps=int(total_steps)
        steps = [total_steps * peak_percentage, total_steps * 0.9, total_steps]
        steps = [int(s) for s in steps]
        return {
            "train_epoch_split": epoch_split,
            "batch_size": batch_size,
            "learning_rate": 1.0,
            "dynamic_learning_rate": dyn_lr_piecewise_linear,
            # If the dict has no entry for the bs_feat,n_ep combination, see above.
            "learning_rate_piecewise_steps": steps,
            "learning_rate_piecewise_values": [peak_lr * 1e-2, peak_lr, peak_lr * 1e-2, peak_lr * 1e-3],
        }

    full_epochs=30
    epoch_split =10
    num_epochs = full_epochs * epoch_split
    # best: lr 0.001, batch 10000 dim 256, but all are close in fact
    # for dim in [256, 512]:
    #     for lr in [1e-3,5e-4]:
    #         for batch_size in [1000,5000,10000]:
    #             train_config = copy.deepcopy(lbs_bpe10k_trans_lm_adam_config)
    #             lm_cfg = copy.deepcopy(default_trafo_transcription_config)
    #             lm_cfg.update(layer_out_dim=dim,
    #                           layer_ff_dim=4*dim,)
    #
    #             hashed_config = {
    #                 "lm_cfg": lm_cfg,
    #             }
    #             hashed_config.update(**_get_cfg_lrlin_oclr(batch_size=batch_size, num_epochs=num_epochs, epoch_split=epoch_split, peak_lr=lr))
    #             train_job_1 = train_lbs_bpe10k_transcription_lm(
    #                 get_model,
    #                 train_step,
    #                 config=train_config,
    #                 num_epochs=num_epochs,
    #                 hashed_config=hashed_config,
    #                 non_hashed_config = {
    #                     "cleanup_old_models": {"keep": [50, 100,150,200,250,300]}
    #                 }
    #             )
    #             train_job_1.add_alias(_sis_prefix + f"/train_trafo_6layer_adam_lr{lr}_batch_size{batch_size}_dim{dim}")
    #             tk.register_output(_sis_prefix + f"/train_trafo_6layer_adam_lr{lr}_batch_size{batch_size}_dim{dim}-learning_rates", train_job_1.out_learning_rates)
    # without dropout

    # batch 10000 : total steps: 46491

    for dim in [256]:
        for lr in [1e-3, 5e-4, 1e-4]:
            for batch_size in [10000]:
                train_config = copy.deepcopy(lbs_bpe10k_trans_lm_adam_config)
                lm_cfg = copy.deepcopy(default_trafo_transcription_config)

                lm_cfg.update(layer_out_dim=dim,
                              layer_ff_dim=4*dim,
                              dropout=0.0,
                              attn_dropout=0.0)

                hashed_config = {
                    "lm_cfg": lm_cfg,
                }
                hashed_config.update(**_get_cfg_lrlin_oclr(batch_size=batch_size, num_epochs=num_epochs, epoch_split=epoch_split, peak_lr=lr, total_steps=47000))
                train_job_1 = train_lbs_bpe10k_transcription_lm(
                    get_model,
                    train_step,
                    config=train_config,
                    num_epochs=num_epochs,
                    hashed_config=hashed_config,
                    non_hashed_config = {
                        "cleanup_old_models": {"keep": [50, 100,150,200,250,300]}
                    }
                )
                train_job_1.add_alias(_sis_prefix + f"/train_trafo_6layer_adam_lr{lr}_batch_size{batch_size}_dim{dim}_drp0.0")
                tk.register_output(_sis_prefix + f"/train_trafo_6layer_adam_lr{lr}_batch_size{batch_size}_dim{dim}_drp0.0-learning_rates", train_job_1.out_learning_rates)


    # train_epoch_split = 20
    # for lr in [1e-4, 1e-5]:
    #     ep = 400
    #     train_job_1 = train_lbs_bpe10k_transcription_lm(
    #         get_model,
    #         train_step,
    #         config=lbs_bpe10k_trans_lm_same_lr_as_kldiv_ilm,
    #         num_epochs=ep,
    #         hashed_config={
    #             "lm_cfg": default_ilm_config,
    #             "learning_rate": lr,
    #             "learning_rates": [lr]*ep,
    #             "train_epoch_split": train_epoch_split,
    #             "batch_size": 630, # 900 * (num_train_step_kldiv/num_train_step_transLM)
    #         },
    #         non_hashed_config={
    #             "cleanup_old_models": {"keep": [20, 40]},
    #         }
    #     )
    #     train_job_1.add_alias(_sis_prefix + f"/train_trans_lm_lr{lr}_kldiv-ilm-hyperparams_split{train_epoch_split}_ep{ep}")
    #     tk.register_output(_sis_prefix + f"/train_trans_lm_lr{lr}_kldiv-ilm-hyperparams_split{train_epoch_split}_ep{ep}/learning_rates", train_job_1.out_learning_rates)



_sis_prefix: Optional[str] = None

def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name
# def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
#     if not prefix_name:
#         from .sis_setup import get_prefix_for_config
#
#         prefix_name = get_prefix_for_config(__file__)
#     global _sis_prefix
#     _sis_prefix = prefix_name


_ls_task = None


def _get_ls_task():
    global _ls_task
    if _ls_task:
        return _ls_task

    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw, get_librispeech_task_raw_v2
    )

    #_ls_task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True) luca's dataloading
    _ls_task = get_librispeech_task_raw_v2(vocab="bpe10k")
    return _ls_task


py = sis_run_with_prefix  # if run directly via `sis m ...`

