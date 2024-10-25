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

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
    RFModelWithMakeLabelScorer,
)

from i6_experiments.users.yang.torch.albert_exp2024_04_23_baselines.configs import *
from i6_experiments.users.yang.torch.albert_exp2024_04_23_baselines.configs import (
    _get_cfg_lrlin_oclr_by_bs_nep,
    _fine_tune_get_cfg_lrlin_oclr_by_bs_nep,
    _batch_size_factor,
)
# from .trafo_lm import trafo_lm_kazuki_import

from i6_experiments.users.yang.torch.luca_ctc.model_conformer_kd_ctc_fwbw import from_scratch_model_def, from_scratch_training
#from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog
from i6_experiments.users.yang.torch.decoding.ctc_greedy import model_recog_greedy

# from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_conformer_ctc import from_scratch_model_def, from_scratch_training
# from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_recog_ctc_greedy import model_recog

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import (
        ModelWithCheckpoints,
        ModelWithCheckpoint,
    )




# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80

# simple linear lr function for fine-tuning


char_config_24gb_v6 = copy.deepcopy(config_24gb_v6)
char_config_24gb_v6.pop("max_seq_length_default_target")

def sis_run_with_prefix(prefix_name: Optional[str] = None):
    for bs in [40]:
    # try smaller batch size, since sequences are longer in general?
        train_exp(
            f"base-24gb-lrlin1e_5_600k_ctc_only_aux4_8_bs{bs}_datalen_312k",
            char_config_24gb_v6,
            config_updates={
                "learning_rate": 1.0,
                "dynamic_learning_rate": dyn_lr_piecewise_linear,
                # total steps after 2000 epochs: 982.312
                "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
                "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
                "mask_eos_output": True,
                "batch_size": bs * 1000 * _batch_size_factor,
                "train_load_extern_lm": None,
                "lm_kd_loss": False,
                "max_seq_length": {"data": 312000.0}, # overwrite seq length constraint
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
    reserve_code: Optional[str] = None,
    model_avg: bool = False,
    greedy_search=True,
    length_norm_scale=0.0,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    # from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.train import (
    #     train,
    # )
    from i6_experiments.users.yang.torch.luca_ctc.train import train
    #from i6_experiments.users.yang.torch.luca_ctc.recog import recog_training_exp
    from i6_experiments.users.yang.torch.decoding.recog import recog_training_exp
    #from i6_experiments.users.zeyer.recog import recog_training_exp

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
        mem_rqmt = mem_rqmt,
        reserve_code=reserve_code,
    )
    # greedy search:
    if greedy_search:
        from i6_experiments.users.yang.torch.decoding.ctc_greedy import model_recog_greedy
        recog_training_exp(
            prefix, task, model_with_checkpoint, recog_def=model_recog_greedy,
        )
    else:
        print('fixed epochs################', model_with_checkpoint.fixed_epochs)
        recog_config_update = {'batch_size':1600000, "length_norm_scale": length_norm_scale}
        from i6_experiments.users.yang.torch.decoding.ctc_aed_joint_simple_version import model_recog
        recog_training_exp(
            prefix + '-label_sync-lennorm_scale_' +f"{length_norm_scale}", task, model_with_checkpoint, search_config=recog_config_update, recog_def=model_recog, model_avg=model_avg
        )

    # ctc label sync search


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
            config_["import_model_train_epoch1"] = model_with_checkpoint.get_epoch(
                ep
            ).checkpoint
            config_.pop("dynamic_learning_rate")
            lrs = opts.pop("learning_rates", None)
            if lrs is None:
                lr_decay_type = opts.pop(
                    "lr_decay_type", "geomspace"
                )  # geomspace or linspace
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
            recog_training_exp(
                prefix + suffix, task, finetune_model_with_ckpt, recog_def=model_recog
            )

    return model_with_checkpoint


_ls_task = None


def _get_ls_task():
    global _ls_task
    if _ls_task:
        return _ls_task

    from i6_experiments.users.yang.torch.datasets.librispeech import get_librispeech_task_raw_v2

    #_ls_task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True) luca's dataloading
    _ls_task = get_librispeech_task_raw_v2(vocab="bpe0")
    return _ls_task


py = sis_run_with_prefix  # if run directly via `sis m ...`


