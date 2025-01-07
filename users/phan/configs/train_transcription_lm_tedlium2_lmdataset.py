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
from i6_experiments.users.phan.rf_models.lstm_lm import get_model, train_step
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
    from i6_experiments.users.phan.train_transcription_lm.tedlium2_bpe10k import tedlium2_transcription_bpe10k_dataloader_config
    from i6_experiments.users.phan.train_transcription_lm.train_config import base_adamw_const_lr_config, \
        base_adamw_newbob_lr_config, base_sgd_config, train_transcription_lm_bpe10k_lower_case_job
    name_config_dict = {
        f"tedlium2_transLm_adamw_const_lr{1e-3}_ep{50}": base_adamw_const_lr_config,
        f"tedlium2_transLm_adamw_newbob_lr{1e-3}_ep{50}": base_adamw_newbob_lr_config,
        f"tedlium2_transLm_sgd_newbob_lr{1.}_ep{50}": base_sgd_config,
    }
    num_epochs_list = [50]
    for num_epochs in num_epochs_list:
        for name, config in name_config_dict.items():
            config_w_data = copy.deepcopy(config)
            config_w_data.update(tedlium2_transcription_bpe10k_dataloader_config)
            config_w_data.update({"lm_cfg": default_ilm_config})
            train_job = train_transcription_lm_bpe10k_lower_case_job(
                get_model,
                train_step,
                config=config_w_data,
                num_epochs=num_epochs,
            )
            prefix = _sis_prefix + "/" + name
            train_job.add_alias(prefix + "/train")
            tk.register_output(prefix + "/learning_rates", train_job.out_learning_rates)


_sis_prefix: Optional[str] = None

def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name



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

