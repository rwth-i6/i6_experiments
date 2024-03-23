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

_torch_ckpt_dir_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/without_lm/"

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

    from i6_experiments.users.gaudino.experiments.conformer_att_2023.tedlium2.model_ckpt_info import (
        models,
    )

    models_with_pt_ckpt = {}

    # new_chkpt_path_sep_enc = tk.Path(
    #     "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/model_baseline__ctc_only__trafo_lm_24_01_19/average.pt",
    #     hash_overwrite="torch_ckpt_sep_enc",
    # )
    # new_chkpt_sep_enc = PtCheckpoint(new_chkpt_path_sep_enc)
    # model_with_checkpoint = ModelWithCheckpoint(
    #     definition=from_scratch_model_def, checkpoint=new_chkpt_sep_enc
    # )

    model_names = models.keys()
    for model_name in model_names:
        model_args = {
            "target_embed_dim": 256,
            "add_ted2_trafo_lm": True,
            "mel_normalization": True,
            "no_ctc": models[model_name].get("no_ctc", False),
            "enc_layer_w_ctc": models[model_name].get("enc_layer_w_ctc", None),
            "encoder_ctc": True,
        }
        new_ckpt_path = tk.Path(
            _torch_ckpt_dir_path + model_name + "__ctc_only" + "/average.pt",
            hash_overwrite=model_name + "_torch_ckpt",
        )
        new_ckpt = PtCheckpoint(new_ckpt_path)
        models_with_pt_ckpt[model_name] = {}
        models_with_pt_ckpt[model_name]["ckpt"] = ModelWithCheckpoint(
            definition=from_scratch_model_def, checkpoint=new_ckpt
        )
        models_with_pt_ckpt[model_name]["model_args"] = model_args

    bsf = 10
    prefix_name = prefix_name + f"/bsf{bsf}"

    opls_model_names = {
        # --tuning done--
        # "model_baseline":{
        #     "scales": [(0.82, 0.18, 0.5)],
        # },
        # ----
        "model_ctc0.9_att0.1": {
            "scales": [(0.7, 0.3, 0.7)],
        },
        "model_ctc0.8_att0.2": {
            "scales": [(0.75, 0.25, 0.5)],
        },
        "model_ctc0.7_att0.3": {
            "scales": [(0.8, 0.2, 0.8)],
        },
        "model_ctc0.6_att0.4": {
            "scales": [(0.85, 0.15, 0.4), (0.85, 0.15, 0.8)],
        },
        "model_ctc0.5_att0.5": {
            "scales": [(0.8, 0.2, 0.8), (0.8, 0.2, 0.7)],
        },
        "model_ctc0.4_att0.6": {
            "scales": [(0.8, 0.2, 0.8), (0.8, 0.2, 0.7)],
        },
        "model_ctc0.3_att0.7": {
            "scales": [
                (0.7, 0.3, 0.7),
            ],
        },
        "model_ctc0.2_att0.8": {
            "scales": [(0.85, 0.15, 0.5)], # 0.5
        },
        "model_ctc0.1_att0.9": {
            "scales": [(0.7, 0.3, 0.6)],
        },
        "model_ctc0.001_att0.999": {
            "scales": [(0.75, 0.25, 0.7)],
        },
        # "model_ctc0.3_att0.7_lay6": {
        #     "scales": [],
        # },
        # "model_ctc0.3_att0.7_lay8": {
        #     "scales": [],
        # },
        # "model_ctc0.3_att0.7_lay10": {
        #     "scales": [],
        # },
        # "model_ctc1.0_att1.0_lay6": {
        #     "scales": [],
        # },
        # "model_ctc1.0_att1.0_lay8": {
        #     "scales": [],
        # },
        # "model_ctc1.0_att1.0_lay10": {
        #     "scales": [],
        # },
        "model_att_only_currL": {
            "scales": [(0.6, 0.4, 0.6), (0.65, 0.35, 0.6)],
        },
        "model_att_only_adjSpec": {
            "scales": [(0.6, 0.4, 0.3), (0.65, 0.35, 0.4)],
        },
        # "model_ctc0.43_att1.0": {
        #     "scales": [],
        # },
        # "model_ctc0.25_att1.0": {
        #     "scales": [],
        # },
        # "model_ctc0.2_att1.0": {
        #     "scales": [],
        # },
        # "model_ctc_only",
    }

    # sep enc: opls aed + ctc prefix
    for model_name in opls_model_names.keys():
    # for model_name in ["model_ctc0.2_att0.8"]:
        for scales, beam_size in product(
            opls_model_names[model_name]["scales"],
            [12, 24, 32],
        ):
            att_scale, ctc_scale, prior_scale = scales

            search_args = {
                "beam_size": beam_size,
                "blank_idx": 1057,
                "bsf": bsf,
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "use_ctc": True,
                "encoder_ctc": True,
                "prior_corr": True if prior_scale > 0 else False,
                "prior_scale": prior_scale,
                "ctc_prior_file": models["model_ctc_only"]["prior"],
                # "hash_overwrite": "v2"
            }

            name = (
                prefix_name
                + "/"
                + model_name
                + "__ctc_only"
                + f"/opls_att{att_scale}_ctc{ctc_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + f"_beam{beam_size}"
            )
            res, _ = recog_model(
                task,
                models_with_pt_ckpt[model_name]["ckpt"],
                model_recog,
                dev_sets=["dev", "test"],  # set to None for all
                model_args=models_with_pt_ckpt[model_name]["model_args"],
                search_args=search_args,
                prefix_name=name,
            )
            tk.register_output(
                name + f"/recog_results",
                res.output,
            )


py = sis_run_with_prefix
