"""Param Import
"""

from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree
from itertools import product

from sisyphus import tk

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.lm_import_2023_09_03 import (
    LSTM_LM_Model,
    MakeModel,
)
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog import (
    model_recog,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_time_sync import (
    model_recog_time_sync,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_ctc_greedy import (
    model_recog_ctc,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import (
    from_scratch_model_def,
    _get_eos_idx,
)


import torch
import numpy

# from functools import partial


# From Mohammad, 2023-06-29
# dev-clean  2.27
# dev-other  5.39
# test-clean  2.41
# test-other  5.51
# _returnn_tf_config_filename = "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config"
# E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
_torch_ckpt_filename = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/baseline_23_10_19/average.pt"
_torch_ckpt_filename_w_trafo_lm = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/baseline_w_trafo_lm_23_12_28/average.pt"
# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80

from IPython import embed


def sis_run_with_prefix(prefix_name: str = None):
    """run the exp"""
    from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
    from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960._moh_att_2023_06_30_import import (
        map_param_func_v3,
    )
    from .sis_setup import get_prefix_for_config
    from i6_core.returnn.training import Checkpoint as TfCheckpoint, PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.gaudino.recog import recog_model
    from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import (
        ConvertTfCheckpointToRfPtJob,
    )
    from i6_experiments.users.gaudino.datasets.tedlium2 import (
        get_tedlium2_task_bpe1k_raw,
    )

    if not prefix_name:
        prefix_name = get_prefix_for_config(__file__)

    task = get_tedlium2_task_bpe1k_raw(with_eos_postfix=True)

    extern_data_dict = task.train_dataset.get_extern_data()
    default_target_key = task.train_dataset.get_default_target()
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    target_dim = targets.feature_dim_or_sparse_dim

    # load baseline model w trafo lm
    new_chkpt_path = tk.Path(
        _torch_ckpt_filename_w_trafo_lm, hash_overwrite="torch_ckpt"
    )
    new_chkpt = PtCheckpoint(new_chkpt_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=new_chkpt
    )
    model_args = {
        "target_embed_dim": 256,
        "add_ted2_trafo_lm": True,
        "mel_normalization": True,
    }

    # single model experiments
    bsf = 10
    model_name = f"/bsf{bsf}/model_baseline"
    prefix_name = prefix_name + model_name

    # att only
    for beam_size in [12, 18, 32]:
        search_args = {
            "beam_size": beam_size,
            # "lm_scale": lm_scale,
            # "add_trafo_lm": True,
            "bsf": bsf,
            "hash_override": "v1",
        }
        name = (
            prefix_name
            + f"/att_beam{beam_size}"
        )
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev"], # set to None for all
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # ctc greedy
    # ctc greedy prior 0.15
    for beam_size, prior_scale in product([1], [0.0, 0.15]):
        search_args = {
            "beam_size": beam_size,
            "blank_idx": 1057,
            "bsf": bsf,
            # "blank_collapse": True,
            # "blank_threshold": -0.05, # in log space
            "prior_corr": True if prior_scale > 0 else False,
            "prior_scale": prior_scale,
            "ctc_prior_file": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.2UG8sLxHNTMO/output/prior.txt",
            "hash_override": "v1",
        }

        name = (
            prefix_name
            + f"/ctc_greedy"
            + (f"_prior{prior_scale}" if prior_scale > 0 else "")
        )
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog_ctc,
            dev_sets=["dev"], # set to None for all
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # ctc prefix search
    for beam_size, prior_scale in product([], [0.0, 0.1, 0.15, 0.2, 0.3]):
        search_args = {
            "beam_size": beam_size,
            "blank_idx": 1057,
            "bsf": bsf,
            "use_ctc": True,
            "ctc_scale": 1.0,
            "att_scale": 0.0,
            # "blank_collapse": True,
            # "blank_threshold": -0.05, # in log space
            "prior_corr": True if prior_scale > 0 else False,
            "prior_scale": prior_scale,
            "ctc_prior_file": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.2UG8sLxHNTMO/output/prior.txt",
        }

        name = (
            prefix_name
            + f"/ctc_prefix_search"
            + (f"_prior{prior_scale}" if prior_scale > 0 else "")
            + f"_beam{beam_size}"
        )
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev"], # set to None for all
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # att + ctc opts
    for scales, beam_size in product([(0.5, 0.5), (0.6, 0.4), (0.65, 0.35), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)], []):
        att_scale, ctc_scale = scales
        name = (
            prefix_name
            + f"/opts_att{att_scale}_ctc{ctc_scale}_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "bsf": bsf,
        }

        # first recog
        recog_res, recog_out = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            dev_sets=["dev"], # set to None for all
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )

    # att + ctc + trafo lm opts
    for scales, beam_size in product([(0.65, 0.35, 0.1)], []):
        att_scale, ctc_scale, lm_scale = scales
        name = (
            prefix_name
            + f"/bsf40_opts_att{att_scale}_ctc{ctc_scale}_blank_collapse_trafolm{lm_scale}_no_eos_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "add_trafo_lm": True,
            "remove_trafo_lm_eos": True,
            "lm_scale": lm_scale,
            "blank_collapse": True,
            "blank_threshold": -0.05, # in log space
            "bsf": "bsf40",
        }

        dev_sets = ["dev"]  # only dev for testing
        # dev_sets = None  # all

        # first recog
        recog_res, recog_out = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            dev_sets=dev_sets,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )

    # att + ctc opls
    for scales, beam_size in product([(0.6, 0.4), (0.65, 0.35), (0.7, 0.3)], []):
        att_scale, ctc_scale = scales
        name = (
            prefix_name
            + f"/opls_att{att_scale}_ctc{ctc_scale}_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "use_ctc": True,
            "ctc_scale": ctc_scale,
            "bsf": bsf,
        }

        recog_res, recog_out = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev"],
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )

    # ctc + trafo lm
    for scales, beam_size in product([(1.0, 0.0, 0.0), (1.0, 0.1, 0.0)], []):
        ctc_scale, lm_scale, prior_scale = scales
        name = (
            prefix_name
            + f"/bsf10_ctc{ctc_scale}_trafolm{lm_scale}_beam{beam_size}_add_eos"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": 0.0,
            "ctc_scale": ctc_scale,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "bsf": "bsf40_5",
            "remove_trafo_lm_eos": True,
            "add_eos_to_end": True,
            # "blank_collapse": True,
            # "blank_threshold": -0.05, # in log space
        }

        dev_sets = ["dev"]  # only dev for testing
        # dev_sets = None  # all

        # first recog
        recog_res, recog_out = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            dev_sets=dev_sets,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )


    # separate encoders
    new_chkpt_path_sep_enc = tk.Path(
        "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/model_baseline__ctc_only__trafo_lm_24_01_19/average.pt",
        hash_overwrite="torch_ckpt_sep_enc",
    )
    new_chkpt_sep_enc = PtCheckpoint(new_chkpt_path_sep_enc)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=new_chkpt_sep_enc
    )
    model_args_sep_enc = {
        "target_embed_dim": 256,
        "add_ted2_trafo_lm": True,
        "encoder_ctc": True,
    }

    # sep encoder baseline + ctc only - opls + trafo lm
    for scales, beam_size in product([(0.7, 0.3, 0.1), (0.7, 0.3, 0.3)], []):
        model_name = "/model_baseline__ctc_only"
        att_scale, ctc_scale, lm_scale = scales
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "use_ctc": True,
            "ctc_scale": ctc_scale,
            "encoder_ctc": True,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "bsf": "bsf40",
        }

        dev_sets = ["dev"]  # only dev-other for testing
        # dev_sets = None  # all
        name = (
            prefix_name
            + model_name
            # + f"/bsf160_att_beam{beam_size}"
            + f"/bsf10_opls_att{att_scale}_ctc{ctc_scale}_trafo_lm{lm_scale}_beam{beam_size}"
        )
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=dev_sets,
            model_args=model_args_sep_enc,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

py = sis_run_with_prefix
