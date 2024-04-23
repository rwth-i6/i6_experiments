"""
Attention-based encoder-decoder (AED) experiments
"""

from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
import math
import numpy as np
import hashlib
import contextlib

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, TrainDef
from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.zeyer.accum_grad_schedules.piecewise_linear import dyn_accum_grad_piecewise_linear

from .configs import *
from .configs import _get_cfg_lrlin_oclr_by_bs_nep

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, ModelWithCheckpoint
    from i6_experiments.users.zeyer.datasets.task import Task


# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def py():
    train_exp(  # TODO (should give 5.11)
        "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2",
        config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
        model_config={"behavior_version": 20},  # new Trafo decoder defaults
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
    )


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    *,
    model_def: Optional[Union[ModelDefWithCfg, ModelDef[Model]]] = None,
    train_def: Optional[TrainDef[Model]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    time_rqmt: Optional[int] = None,  # set this to 1 or below to get the fast test queue
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
    if "__train_audio_preprocess" in config:
        task: Task = copy.copy(task)
        task.train_dataset = copy.copy(task.train_dataset)
        task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")

    if not model_def:
        model_def = from_scratch_model_def
    if model_config:
        model_def = ModelDefWithCfg(model_def, model_config)
    if not train_def:
        train_def = from_scratch_training
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
        distributed_launch_cmd="torchrun" if num_processes else None,
        time_rqmt=time_rqmt,
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
