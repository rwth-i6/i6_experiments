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
from i6_experiments.users.phan.rf_models.lstm_lm import get_model, train_step


from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.model_with_checkpoints import (
    ModelWithCheckpoints,
    ModelWithCheckpoint,
)

from i6_core.returnn.training import PtCheckpoint
from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix()
    from i6_experiments.users.phan.prior.prior_config import compute_prior_job, _prior_out_filename
    from i6_experiments.users.phan.prior.model_forward_prior import model_forward_prior
    model = ModelWithCheckpoint(
        definition=from_scratch_model_def,
        checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    )
    prior_config = {
        "batch_size": 25600000,
        "batching": "sorted_reverse",
    }
    lbs_task = _get_ls_task()
    prior_job = compute_prior_job(
        task=lbs_task,
        model=model,
        recog_def=model_forward_prior,
        config=prior_config,
        search_rqmt={"time": 12},
    )
    prior_job.set_vis_name(f"Compute prior job conformer baseline on LBS")
    tk.register_output(_sis_prefix + f"/lbs/{_prior_out_filename}", prior_job.out_files[_prior_out_filename])
    return prior_job




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
    _ls_task = get_librispeech_task_raw_v2(vocab="bpe10k", main_key="train")
    return _ls_task


py = sis_run_with_prefix  # if run directly via `sis m ...`

