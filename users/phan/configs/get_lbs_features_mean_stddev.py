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

from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, from_scratch_training_kldiv_sample_batch
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog
from i6_experiments.users.phan.rf_models.default_model_configs import default_ilm_config, default_extern_lm_config


from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.model_with_checkpoints import (
    ModelWithCheckpoints,
    ModelWithCheckpoint,
)

from i6_core.returnn.training import PtCheckpoint

# From Mohammad, 2023-06-29
# dev-clean  2.27
# dev-other  5.39
# test-clean  2.41
# test-other  5.51
# _returnn_tf_config_filename = (
#   "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config")
# E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"



# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80

# simple linear lr function for fine-tuning


config_11gb = copy.deepcopy(config_11gb)
config_11gb.pop("dynamic_learning_rate", None)
config_11gb.pop("learning_rate_piecewise_steps", None)
config_11gb.pop("learning_rate_piecewise_values", None)
config_11gb.pop("learning_rate_invsqrt_norm", None)
config_11gb.pop("learning_rate_warmup_steps", None)


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """Compute feature statistics for librispeech"""
    from i6_experiments.users.phan.forward_misc import generic_forward_config, compute_features_mean_stddev
    dataset_keys = ["dev-other", "test-other"]
    forward_extra_config = copy.deepcopy(config_11gb)
    forward_extra_config.update({
        "batch_size": 9600000,
        "max_seqs": 200,
    })
    forward_post_config = dict(
        torch_log_memory_usage=True,
        use_lovely_tensors=True,
    )
    dataset_keys = ["train"]
    task = _get_ls_task()
    model = ModelWithCheckpoint(
        definition=from_scratch_model_def,
        checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    )
    for dataset_key in dataset_keys:
        if dataset_key == "train": # not tested
            forward_dataset = task.train_dataset
        else:
            forward_dataset = task.eval_datasets[dataset_key]
        stats_job = generic_forward_config.generic_forward_job(
            dataset=forward_dataset,
            model=model,
            forward_def=compute_features_mean_stddev.forward_compute_features_mean_stddev,
            forward_callback=compute_features_mean_stddev.forward_callback_wrapper,
            forward_extra_config=forward_extra_config,
            forward_post_config=forward_post_config,
            output_files=compute_features_mean_stddev.output_files,
            dataset_key=dataset_key,
            job_vis_name=f"Compute librispeech features mean and std_dev, dataset: {dataset_key}",
        )
        out_file_mean = stats_job.out_files[compute_features_mean_stddev.default_out_file_mean]
        out_file_stddev = stats_job.out_files[compute_features_mean_stddev.default_out_file_stddev]
        out_file_info = stats_job.out_files[compute_features_mean_stddev.default_out_file_info]
        stats_job.add_alias(prefix + "/lbs_feature_stats" + f"/{dataset_key}")
        tk.register_output(prefix + f"/lbs_feature_stats/{dataset_key}/{compute_features_mean_stddev.default_out_file_mean}", out_file_mean)
        tk.register_output(prefix + f"/lbs_feature_stats/{dataset_key}/{compute_features_mean_stddev.default_out_file_stddev}", out_file_stddev)
        tk.register_output(prefix + f"/lbs_feature_stats/{dataset_key}/{compute_features_mean_stddev.default_out_file_info}", out_file_info)


_sis_prefix: Optional[str] = None

def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name

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

