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


# from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, from_scratch_training_kldiv_sample_batch
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog
from i6_experiments.users.phan.rf_models.default_model_configs import default_extern_lm_config
from i6_experiments.users.yang.torch.rf_lm.model.lstm_lm import model_def
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.model_with_checkpoints import (
    ModelWithCheckpoints,
    ModelWithCheckpoint,
)

from i6_core.returnn.training import PtCheckpoint

default_lm_config = { # should be used for transcription LM as well
    "class": "LSTMLMRF",
    "symbol_embedding_dim": 128,
    "emebdding_dropout": 0.0,
    "num_lstm_layers": 4,
    "lstm_hidden_dim": 2048,
    "lstm_dropout": 0.0,
    "use_bottleneck": False,
}
# training config
num_epochs = 60

default_training_config = dict(
batching = "random",
batch_size = 1350,
max_seq_length = 1350,
max_seqs = 32,
chunking = "0",
__num_epochs = num_epochs,
optimizer = {"class": "sgd", "weight_decay": 0.0},
gradient_clip_global_norm = 2.,
gradient_noise = 0.,
learning_rate = 1.,
learning_rate_control = "newbob_rel",
learning_rate_control_relative_error_relative_lr = False,
newbob_multi_num_epochs = 4, # hard coded
newbob_relative_error_div_by_old = True,
newbob_learning_rate_decay = 0.9,
newbob_relative_error_threshold = -0.005,
newbob_multi_update_interval = 1,
learning_rate_control_error_measure = "dev_loss_ppl"
)

# simple linear lr function for fine-tuning




def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix()


    # sgd optim
    lrs = [1.]
    for lr in lrs:
        training_config = copy.deepcopy(default_training_config)
        train_exp(
            f"tedlium2_transLM_sgd_lr{lr}_ep60",
            training_config,
            gpu_mem=11,
            config_updates={
                "learning_rate": lr,
                "lm_cfg": default_lm_config,
            },
            # post_config_updates={
            #     "cleanup_old_models": True,
            #     "torch_dataloader_opts": { # otherwise it will break after every epoch
            #         "num_workers": 0,
            #     }
            # },
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
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    # from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.train import (
    #     train,
    # )
    from i6_experiments.users.yang.torch.rf_lm.train_lm import train_text_only
    from i6_experiments.users.yang.torch.rf_lm.datasets.tedlium_lm import get_ted_librispeech_wise_lm_data

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    train_data_dict, dev_data_dict, extern_data_raw =get_ted_librispeech_wise_lm_data()
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
        train_data= train_data_dict,
        dev_data=dev_data_dict,
        extern_data_dict= extern_data_raw,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        model_def=model_def,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else "mpirun",
        time_rqmt=time_rqmt,
        mem_rqmt = mem_rqmt,
    )

    train_job = model_with_checkpoint.get_training_job()
    tk.register_output(prefix + "/train/learning_rates", train_job.out_learning_rates)
    return model_with_checkpoint



py = sis_run_with_prefix  # if run directly via `sis m ...`

