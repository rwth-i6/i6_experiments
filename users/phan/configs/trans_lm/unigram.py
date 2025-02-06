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
from i6_experiments.users.phan.rf_models.unigram_lm import get_model, train_step


from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.model_with_checkpoints import (
    ModelWithCheckpoints,
    ModelWithCheckpoint,
)

from i6_core.returnn.training import PtCheckpoint


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix()
    from i6_experiments.users.phan.lbs_transcription_bpe10k.train_config import train_lbs_bpe10k_transcription_lm, lbs_bpe10k_trans_lm_same_lr_as_kldiv_ilm

    train_job = train_lbs_bpe10k_transcription_lm(
        get_model,
        train_step,
        hashed_config={
            "lm_cfg": {"class": "Unigram_LM_RF"},
        },
        num_epochs=30,
        post_config_update={
            "cleanup_old_models": False,
        }
    )
    train_job.add_alias(_sis_prefix + "/train")
    tk.register_output(_sis_prefix + "/learning_rates", train_job.out_learning_rates)

    # train_job_1 = train_lbs_bpe10k_transcription_lm(
    #     get_model,
    #     train_step,
    #     num_epochs=2,
    #     hashed_config={
    #         "lm_cfg": default_ilm_config,
    #         "this_was_added_to_trick_sisyphus_to_run_the_training_again": 0, # hack to trick it to train
    #     }
    # )
    # train_job_1.add_alias(_sis_prefix + "/train_1_epoch_only")
    # tk.register_output(_sis_prefix + "/bpe10k_transcription_lm_1_epoch/learning_rates", train_job_1.out_learning_rates)

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

