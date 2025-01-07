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

from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
    RFModelWithMakeLabelScorer,
)

from i6_experiments.users.yang.torch.luca_ctc.configs import *
from i6_experiments.users.yang.torch.luca_ctc.configs import (
    _batch_size_factor,
    _cfg_lrlin1e_5_295k,
    _get_cfg_lrlin_oclr_by_bs_nep,
    const_linear_learning_rates,
linear_const_linear_learning_rates,
)

# from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, from_scratch_training_kldiv_sample_batch
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog
from i6_experiments.users.phan.rf_models.default_model_configs import default_extern_lm_config, default_ilm_config
from i6_experiments.users.phan.rf_models.lstm_lm import model_def, train_def
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.model_with_checkpoints import (
    ModelWithCheckpoints,
    ModelWithCheckpoint,
)

from i6_core.returnn.training import PtCheckpoint



# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80

# simple linear lr function for fine-tuning


config_11gb = copy.deepcopy(config_11gb)
config_11gb.pop("dynamic_learning_rate", None)
config_11gb.pop("learning_rate_piecewise_steps", None)
config_11gb.pop("learning_rate_piecewise_values", None)
config_11gb.pop("learning_rate_invsqrt_norm", None)
config_11gb.pop("learning_rate_warmup_steps", None)
config_11gb.pop("aux_loss_layers", None)


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix()

    # adamw optim
    lrs = [1e-3]
    epochs = [50]
    for lr in lrs:
        for ep in epochs:
            train_exp( 
                f"tedlium2_transLM_adamw_lr{lr}_ep{ep}",
                config_11gb,
                train_def,
                gpu_mem=11,
                config_updates={
                    "batch_size": 2400000,
                    "learning_rate": lr,
                    "__num_epochs": ep,
                    "lm_cfg": default_ilm_config,
                },
                post_config_updates={
                    "cleanup_old_models": True,
                    "torch_dataloader_opts": { # otherwise it will break after every epoch
                        "num_workers": 0,
                    }
                },
            )

    # sgd optim
    lrs = [1.]
    for lr in lrs:
        for ep in epochs:
            train_exp( 
                f"tedlium2_transLM_sgd_lr{lr}_ep{ep}",
                config_11gb,
                train_def,
                gpu_mem=11,
                config_updates={
                    "batch_size": 2400000,
                    "learning_rate": lr,
                    "__num_epochs": ep,
                    "lm_cfg": default_ilm_config,
                    "optimizer": {"class": "sgd"},
                    "learning_rate_control": "newbob_rel",
                    "learning_rate_control_relative_error_relative_lr": False,
                    "newbob_multi_num_epochs": 1,
                    "newbob_relative_error_div_by_old": True,
                    "newbob_learning_rate_decay": 0.8,
                    "newbob_relative_error_threshold": -0.02,
                    "newbob_multi_update_interval": 1,
                },
                post_config_updates={
                    "cleanup_old_models": True,
                    "torch_dataloader_opts": { # otherwise it will break after every epoch
                        "num_workers": 0,
                    }
                },
            )


_sis_prefix: Optional[str] = None

def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name

# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    train_def: TrainDef,
    *,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    fine_tune: Optional[Union[int, List[Tuple[int, Dict[str, Any]]]]] = None,
    time_rqmt: Optional[int] = None,
    mem_rqmt: Optional[int] = None,
    model_avg: bool = False,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    # from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.train import (
    #     train,
    # )
    from i6_experiments.users.phan.train.train_text_only import train_text_only

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    task = _get_ted2_task()
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    if "__num_epochs" in config:
        num_epochs = config.pop("__num_epochs")
    if "__gpu_mem" in config:
        gpu_mem = config.pop("__gpu_mem")
    if "__num_processes" in config:
        num_processes = config.pop("__num_processes")

    model_with_checkpoint = train_text_only(
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
        mem_rqmt = mem_rqmt,
        disable_epoch_wise_filter=True,
    )

    train_job = model_with_checkpoint.get_training_job()
    tk.register_output(prefix + "/train/learning_rates", train_job.out_learning_rates)
    return model_with_checkpoint


_ted2_task = None

def _get_ted2_task():
    global _ted2_task
    if _ted2_task:
        return _ted2_task
    from i6_experiments.users.phan.datasets.librispeech_tedlium2 import get_tedlium2_task_libri_bpe10k_raw
    _ted2_task = get_tedlium2_task_libri_bpe10k_raw(
        with_eos_postfix=False,
        train_epoch_split=1,
        train_epoch_wise_filter=None
        )
    return _ted2_task


py = sis_run_with_prefix  # if run directly via `sis m ...`

