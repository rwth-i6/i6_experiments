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
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_dump import (
    model_recog_dump
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
    from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960._moh_att_2023_06_30_import import map_param_func_v3
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

    new_chkpt_path = ConvertTfCheckpointToRfPtJob(
        checkpoint=TfCheckpoint(
            index_path=generic_job_output(_returnn_tf_ckpt_filename)
        ),
        make_model_func=MakeModel(
            in_dim=_log_mel_feature_dim,
            target_dim=target_dim.dimension,
            eos_label=_get_eos_idx(target_dim),
        ),
        map_func=map_param_func_v3,
    ).out_checkpoint

    # att + ctc decoding
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
    }

    # att only
    for beam_size in [12]:
        lm_scale = 0.1
        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "add_trafo_lm": True,
            "bsf": "bsf40_1",
        }

        dev_sets = ["dev"]  # only dev-other for testing
        # dev_sets = None  # all
        name = (
            prefix_name
            # + f"/bsf160_att_beam{beam_size}"
            + f"/bsf40_att_trafo_lm{lm_scale}_beam{beam_size}_fix"
        )
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=dev_sets,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name
            + f"/recog_results",
            res.output,
        )

    # att + ctc opts
    for scales, beam_size in product([(0.65, 0.35, 0.3)], [6,12,32]):
        att_scale, ctc_scale, prior_scale = scales
        name = (
                prefix_name
                + f"/bsf40_timesync_att{att_scale}_ctc{ctc_scale}_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "bsf": "bsf40_1",
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

    # ctc + trafo lm
    for scales, beam_size in product([(1.0, 0.1, 0.0), (1.0, 0.3, 0.0)], [6, 12]):
        ctc_scale, lm_scale, prior_scale = scales
        name = (
                prefix_name
                + f"/single_seq_ctc{ctc_scale}_trafolm{lm_scale}_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": 0.0,
            "ctc_scale": ctc_scale,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "bsf": "bsf40_3",
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


    # # att + espnet ctc prefix scorer
    # for scales in [(0.7,0.3), (0.65,0.35), (0.75,0.25)]:
    #     for beam_size in [12]:
    #         att_scale, ctc_scale = scales
    #         search_args = {
    #             "beam_size": beam_size,
    #             # att decoder args
    #             "att_scale": att_scale,
    #             "ctc_scale": ctc_scale,
    #             "use_ctc": True,
    #             "mask_eos": True,
    #             "prior_corr": False,
    #             "prior_scale": 0.2,
    #             "prior_file": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.2UG8sLxHNTMO/output/prior.txt",
    #             "length_normalization_exponent": 1.0,  # 0.0 for disabled
    #             # "window_margin": 10,
    #             "rescore_w_ctc": False,
    #         }
    #         dev_sets = ["dev"]  # only dev-other for testing
    #         # dev_sets = None  # all
    #         res = recog_model(
    #             task,
    #             model_with_checkpoint,
    #             model_recog,
    #             dev_sets=dev_sets,
    #             model_args=model_args,
    #             search_args=search_args,
    #         )
    #         tk.register_output(
    #             prefix_name
    #             + f"/bsf40_espnet_att{att_scale}_ctc{ctc_scale}_beam{beam_size}"
    #             + f"/recog_results",
    #             res.output,
    #         )
    #
    # # opts att + ctc
    # for scales in [(0.7,0.3), (0.65,0.35), (0.75,0.25)]:
    #     for beam_size in [12]:
    #         att_scale, ctc_scale = scales
    #         search_args = {
    #             "beam_size": beam_size,
    #             "att_scale": att_scale,
    #             "ctc_scale": ctc_scale,
    #             "use_ctc": True,
    #             "mask_eos": True,
    #         }
    #         dev_sets = ["dev"]  # only dev-other for testing
    #         # dev_sets = None  # all
    #         res = recog_model(
    #             task,
    #             model_with_checkpoint,
    #             model_recog_time_sync,
    #             dev_sets=dev_sets,
    #             model_args=model_args,
    #             search_args=search_args,
    #         )
    #         tk.register_output(
    #             prefix_name
    #             + f"/bsf40_opts_att{att_scale}_ctc{ctc_scale}_beam{beam_size}"
    #             + f"/recog_results",
    #             res.output,
    #         )



py = sis_run_with_prefix
