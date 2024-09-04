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

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_09_03 import (
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
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_ts_espnet import (
    model_recog_ts_espnet,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import (
    from_scratch_model_def,
    _get_eos_idx,
)

import torch
import numpy

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.tedlium2.scales import *

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

_torch_ckpt_dir_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/"

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
    from i6_experiments.users.gaudino.recog_2 import recog_model as recog_model_2
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

    bsf = 10
    bsf_40 = 40
    prefix_name_single_seq = prefix_name + "/single_seq"
    prefix_name_40 = prefix_name + f"/bsf{bsf_40}"
    prefix_name = prefix_name + f"/bsf{bsf}"

    models_with_pt_ckpt = {}

    model_names = models.keys()
    for model_name in model_names:
        model_args = {
            "target_embed_dim": 256,
            "add_trafo_lm": True,
            "mel_normalization": True,
            "no_ctc": models[model_name].get("no_ctc", False),
            "enc_layer_w_ctc": models[model_name].get("enc_layer_w_ctc", None),
            "encoder_ctc": True,
            "external_language_model": {
                "class": "Trafo_LM_Model",
            },
            "s_use_zoneout_output": True,
        }
        new_ckpt_path = tk.Path(
            _torch_ckpt_dir_path + model_name + "/average.pt",
            hash_overwrite=model_name + "_torch_ckpt",
        )
        new_ckpt = PtCheckpoint(new_ckpt_path)
        models_with_pt_ckpt[model_name] = {}
        models_with_pt_ckpt[model_name]["ckpt"] = ModelWithCheckpoint(
            definition=from_scratch_model_def, checkpoint=new_ckpt
        )
        models_with_pt_ckpt[model_name]["model_args"] = model_args
        models_with_pt_ckpt[model_name]["recog_config"] = {
            "model_args": model_args,
            "preload_from_files": {
                "model_ctc": {
                    "prefix": "sep_enc_ctc.",
                    "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/model_ctc_only/average.pt",
                },
                "01_trafo_lm": {
                    "prefix": "language_model.",
                    "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/trafo_lm_only_24_02_05/network.020.pt",
                },
            },
            "mel_normalization_ted2": True,
            "batch_size": bsf * 20000,  # bsf10
        }

    # opls att + ctc
    for model_name in [name for name in att_model_names if not scales_att_ctc_only_lm_opls[name].get("wer", None)]:
        # lm_scale = [0.2, 0.3,0.4,0.45,0.5, 0.55, 0.6, 0.7]
        # all_scales = [scales_att_ctc_only_opls[model_name]["scales"][0] + [scale] for scale in lm_scale]
        all_scales = scales_att_ctc_only_lm_opls[model_name]["scales"]
        for scales, beam_size in product(all_scales, [12, 32]):
            att_scale, ctc_scale, prior_scale, lm_scale = scales

            # with_lm_ilm, with_lm

            recog_name = (
                ("/with_lm" if lm_scale > 0.0 else "")
                + f"/opls_att{att_scale}_ctc{ctc_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
                + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
                # + (f"_blank{blank_scale}" if blank_scale > 0.0 else "")
                + f"_beam{beam_size}"
            )

            sep_model_name = "/AED-" + model_name + "/CTC-model_ctc_only"
            name = prefix_name + sep_model_name + recog_name
            search_args = {
                "beam_size": beam_size,
                "add_trafo_lm": lm_scale > 0.0,
                "lm_scale": lm_scale,
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "use_ctc": True,
                "bsf": bsf,
                "prior_corr": prior_scale > 0.0,
                "ctc_prior_file": models["model_ctc_only"]["prior"],
                "prior_scale": prior_scale,
                "encoder_ctc": True,
                # "blank_scale_minus": blank_scale,
            }
            recog_config = models_with_pt_ckpt[model_name]["recog_config"]
            recog_config["search_args"] = search_args
            res = recog_model_2(
                task,
                models_with_pt_ckpt[model_name]["ckpt"],
                model_recog,
                dev_sets=["dev"] if len(all_scales) > 1 else None,
                config=recog_config,
                name=name,
                device="gpu",
                # search_mem_rqmt=15,
            )
            tk.register_output(
                name + f"/recog_results",
                res.output,
            )

    # optsr att + ctc
    for model_name in [name for name in att_model_names if not scales_att_ctc_only_lm_optsr[name].get("wer", None)]:
        # lm_scale = [0.2, 0.3,0.4,0.45,0.5, 0.55, 0.6, 0.7]
        # all_scales = [scales_att_ctc_only_lm_optsr[model_name]["scales"][0][:3] + [scale] for scale in lm_scale]
        all_scales = scales_att_ctc_only_lm_optsr[model_name]["scales"]
        for scales, beam_size in product(all_scales, [12, 24, 32]):
            att_scale, ctc_scale, prior_scale, lm_scale = scales

            # with_lm_ilm, with_lm

            recog_name = (
                ("/with_lm" if lm_scale > 0.0 else "")
                + f"/optsr_att{att_scale}_ctc{ctc_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
                + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
                # + (f"_blank{blank_scale}" if blank_scale > 0.0 else "")
                + f"_beam{beam_size}"
            )

            sep_model_name = "/AED-" + model_name + "/CTC-model_ctc_only"
            name = prefix_name_40 + sep_model_name + recog_name
            search_args = {
                "beam_size": beam_size,
                "add_trafo_lm": lm_scale > 0.0,
                "lm_scale": lm_scale,
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "use_ctc": True,
                "bsf": bsf_40,
                "ctc_prior_file": models["model_ctc_only"]["prior"],
                "prior_scale": prior_scale,
                "add_eos_to_end": True,
                "encoder_ctc": True,
                # "blank_scale_minus": blank_scale,
                "lm_skip": True,
                "hash_overwrite": "more time"
            }

            recog_config = models_with_pt_ckpt[model_name]["recog_config"]
            recog_config["search_args"] = search_args
            res = recog_model_2(
                task,
                models_with_pt_ckpt[model_name]["ckpt"],
                model_recog_time_sync,
                dev_sets=["dev"] if len(all_scales) > 1 else None,
                config=recog_config,
                name=name,
                device="gpu",
                search_rqmt={"time": 12},
                # search_mem_rqmt=15,
            )
            tk.register_output(
                name + f"/recog_results",
                res.output,
            )

    #
    # opls_model_names = {
    #     # --tuning done--
    #     "model_baseline":{
    #         "scales": [(0.82, 0.18, 0.5, 0.45), (0.82, 0.18, 0.5, 0.5)],
    #     },
    #     # "model_ctc0.9_att0.1": {
    #     #     "scales": [(0.7, 0.3, 0.7)],
    #     # },
    #     # "model_ctc0.8_att0.2": {
    #     #     "scales": [(0.75, 0.25, 0.5)],
    #     # },
    #     # "model_ctc0.7_att0.3": {
    #     #     "scales": [(0.8, 0.2, 0.8)],
    #     # },
    #     # "model_ctc0.6_att0.4": {
    #     #     "scales": [(0.85, 0.15, 0.4), (0.85, 0.15, 0.8)],
    #     # },
    #     # "model_ctc0.5_att0.5": {
    #     #     "scales": [(0.8, 0.2, 0.8), (0.8, 0.2, 0.7)],
    #     # },
    #     # "model_ctc0.4_att0.6": {
    #     #     "scales": [(0.8, 0.2, 0.8), (0.8, 0.2, 0.7)],
    #     # },
    #     "model_ctc0.3_att0.7": {
    #         "scales": [
    #             (0.7, 0.3, 0.7, 0.4), (0.7, 0.3, 0.7, 0.45),
    #         ],
    #     },
    #     "model_ctc0.2_att0.8": {
    #         "scales": [(0.85, 0.15, 0.5, 0.4), (0.85, 0.15, 0.5, 0.45)], # 0.5
    #     },
    #     "model_ctc0.1_att0.9": {
    #         "scales": [(0.7, 0.3, 0.6, 0.45)],
    #     },
    #     "model_ctc0.001_att0.999": {
    #         "scales": [(0.75, 0.25, 0.7, 0.4)],
    #     },
    #     # "model_att_only_currL": {
    #     #     "scales": [(0.6, 0.4, 0.6), (0.65, 0.35, 0.6)],
    #     # },
    #     # "model_att_only_adjSpec": {
    #     #     "scales": [(0.6, 0.4, 0.3), (0.65, 0.35, 0.4)],
    #     # },
    #     # ----
    #     "model_ctc0.3_att0.7_lay6": {
    #         "scales": [(0.8, 0.2, 0.9, 0.5)],
    #     },
    #     "model_ctc0.3_att0.7_lay8": {
    #         "scales": [(0.85, 0.15, 0.6, 0.6), (0.85, 0.15, 0.7, 0.5)],
    #     },
    #     "model_ctc0.3_att0.7_lay10": {
    #         "scales": [(0.9, 0.1, 0.65, 0.5)],
    #     },
    #     "model_ctc1.0_att1.0_lay6": {
    #         "scales": [(0.7, 0.3, 0.7, 0.45)],
    #     },
    #     # "model_ctc1.0_att1.0_lay8": {
    #     #     "scales": [(0.8, 0.2, 0.3)],
    #     # },
    #     # "model_ctc1.0_att1.0_lay10": {
    #     #     "scales": [(0.7, 0.3, 0.6), (0.7, 0.3, 0.65)],
    #     # },
    #     # "model_ctc0.43_att1.0": {
    #     #     "scales": [],
    #     # },
    #     # "model_ctc0.25_att1.0": {
    #     #     "scales": [],
    #     # },
    #     # "model_ctc0.2_att1.0": {
    #     #     "scales": [],
    #     # },
    #     # "model_ctc_only",
    # }
    #
    # # sep enc: opls aed + ctc prefix
    # for model_name in opls_model_names.keys():
    # # for model_name in ["model_ctc0.2_att0.8"]:
    #     for scales, beam_size in product(
    #         opls_model_names[model_name]["scales"],
    #         [], # [6, 12, 32],
    #     ):
    #         att_scale, ctc_scale, prior_scale = scales
    #
    #         search_args = {
    #             "beam_size": beam_size,
    #             "blank_idx": 1057,
    #             "bsf": bsf,
    #             "att_scale": att_scale,
    #             "ctc_scale": ctc_scale,
    #             "use_ctc": True,
    #             "encoder_ctc": True,
    #             "prior_corr": True if prior_scale > 0 else False,
    #             "prior_scale": prior_scale,
    #             "ctc_prior_file": models["model_ctc_only"]["prior"],
    #             # "hash_overwrite": "v2"
    #         }
    #
    #         name = (
    #             prefix_name
    #             + "/"
    #             + model_name
    #             + "__ctc_only"
    #             + f"/opls_att{att_scale}_ctc{ctc_scale}"
    #             + (f"_prior{prior_scale}" if prior_scale > 0 else "")
    #             + f"_beam{beam_size}"
    #         )
    #         res, _ = recog_model(
    #             task,
    #             models_with_pt_ckpt[model_name]["ckpt"],
    #             model_recog,
    #             dev_sets=["dev", "test"],  # set to None for all
    #             model_args=models_with_pt_ckpt[model_name]["model_args"],
    #             search_args=search_args,
    #             prefix_name=name,
    #         )
    #         tk.register_output(
    #             name + f"/recog_results",
    #             res.output,
    #         )
    #
    # # --- With Trafo LM ---
    #
    # models_with_pt_ckpt = {}
    #
    # model_names = models.keys()
    # for model_name in model_names:
    #     model_args = {
    #         "target_embed_dim": 256,
    #         "add_trafo_lm": True,
    #         "mel_normalization": True,
    #         "no_ctc": models[model_name].get("no_ctc", False),
    #         "enc_layer_w_ctc": models[model_name].get("enc_layer_w_ctc", None),
    #         "encoder_ctc": True,
    #     }
    #     new_ckpt_path = tk.Path(
    #         _torch_ckpt_dir_path + model_name + "__ctc_only__trafo_lm" + "/average.pt",
    #         hash_overwrite=model_name + "_torch_ckpt",
    #     )
    #     new_ckpt = PtCheckpoint(new_ckpt_path)
    #     models_with_pt_ckpt[model_name] = {}
    #     models_with_pt_ckpt[model_name]["ckpt"] = ModelWithCheckpoint(
    #         definition=from_scratch_model_def, checkpoint=new_ckpt
    #     )
    #     models_with_pt_ckpt[model_name]["model_args"] = model_args
    #
    # # sep enc: opls aed + ctc prefix + trafo lm
    # for model_name in opls_model_names.keys():
    # # for model_name in ["model_ctc0.2_att0.8"]:
    #     for scales, beam_size in product(
    #         opls_model_names[model_name]["scales"],
    #         [12, 32],
    #     ):
    #         att_scale, ctc_scale, prior_scale, lm_scale = scales
    #
    #         search_args = {
    #             "beam_size": beam_size,
    #             "blank_idx": 1057,
    #             "bsf": bsf,
    #             "att_scale": att_scale,
    #             "ctc_scale": ctc_scale,
    #             "use_ctc": True,
    #             "encoder_ctc": True,
    #             "prior_corr": True if prior_scale > 0 else False,
    #             "prior_scale": prior_scale,
    #             "ctc_prior_file": models["model_ctc_only"]["prior"],
    #             "add_trafo_lm": True,
    #             "lm_scale": lm_scale,
    #             # "hash_overwrite": "v2"
    #         }
    #
    #         name = (
    #             prefix_name
    #             + "/"
    #             + model_name
    #             + "__ctc_only"
    #             + f"/opls_att{att_scale}_ctc{ctc_scale}_trafolm{lm_scale}"
    #             + (f"_prior{prior_scale}" if prior_scale > 0 else "")
    #             + f"_beam{beam_size}"
    #         )
    #         res, _ = recog_model(
    #             task,
    #             models_with_pt_ckpt[model_name]["ckpt"],
    #             model_recog,
    #             dev_sets=["dev", "test"],  # set to None for all
    #             model_args=models_with_pt_ckpt[model_name]["model_args"],
    #             search_args=search_args,
    #             prefix_name=name,
    #         )
    #         tk.register_output(
    #             name + f"/recog_results",
    #             res.output,
    #         )

    # with mini att ilm

    # ------------------------ with ILM -------------------------
    model_name = "model_baseline"

    model_args = {
        "target_embed_dim": 256,
        "add_trafo_lm": True,
        "mel_normalization": True,
        "no_ctc": models[model_name].get("no_ctc", False),
        "enc_layer_w_ctc": models[model_name].get("enc_layer_w_ctc", None),
        "encoder_ctc": True,
        "external_language_model": {
            "class": "Trafo_LM_Model",
        },
        "internal_language_model": {
            "class": "MiniAtt_ILM_Model",
        },
        "s_use_zoneout_output": True,
    }
    new_ckpt_path = tk.Path(
        _torch_ckpt_dir_path + model_name + "/average.pt",
        hash_overwrite=model_name + "_torch_ckpt",
    )
    new_ckpt = PtCheckpoint(new_ckpt_path)
    model_baseline_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=new_ckpt
    )

    recog_config = {
        "model_args": model_args,
        "preload_from_files": {
            "model_ctc": {
                "prefix": "sep_enc_ctc.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/model_ctc_only/average.pt",
            },
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/trafo_lm_only_24_02_05/network.020.pt",
            },
            "01_mini_att_ilm": {
                "prefix": "ilm.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/mini_att_ilm_24_04_21/average.pt",
            },
        },
        "mel_normalization_ted2": True,
        "batch_size": bsf * 20000,  # bsf10
    }

    orig_recog_config = recog_config.copy()

    model_name = "/AED-model_baseline/CTC-model_ctc_only"

    # ilm ckpt torch: /work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/mini_att_ilm_24_05_28/average.pt

    # # sanity checks
    # for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
    #     [(1.0, 0.0)],
    #     [0.0],  # , 0.05, 0.07],
    #     [0.0], [0.0], [32]
    # ):
    #     att_scale, ctc_scale = scales
    #     # recog_name = (
    #     #     f"/opls_att{att_scale}_ctc{ctc_scale}"
    #     #     + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
    #     #     + f"_trafo_lm{lm_scale}"
    #     #     + f"_ilm{ilm_scale}"
    #     #     + f"_beam{beam_size}_ffix"
    #     # )
    #     recog_name = f"/att_only_check_beam{beam_size}_fix2"
    #     name = prefix_name + model_name + recog_name
    #     search_args = {
    #         "beam_size": beam_size,
    #         "add_trafo_lm": True,
    #         "lm_scale": lm_scale,
    #         "att_scale": att_scale,
    #         "ctc_scale": ctc_scale,
    #         "ilm_scale": ilm_scale,
    #         "use_ctc": False,
    #         "bsf": bsf,
    #         "prior_corr": prior_scale > 0.0,
    #         "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
    #         "prior_scale": prior_scale,
    #         "use_lm_first_label": True,
    #     }
    #     recog_config["search_args"] = search_args
    #     res = recog_model(
    #         task,
    #         model_with_checkpoint,
    #         model_recog,
    #         dev_sets=["dev-other"],
    #         # dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
    #         config=recog_config,
    #         # model_args=model_args,
    #         # search_args=search_args,
    #         name=name,
    #         device="gpu",
    #         # search_mem_rqmt=15,
    #     )
    #     tk.register_output(
    #         name + f"/recog_results",
    #         res.output,
    #     )

    # opls att + ctc
    for scales, prior_scale, lm_scale, ilm_scale, blank_scale, beam_size in product(
        [(0.65, 0.35)],
        [0.5],
        [0.4, 0.43],  # [0.7, 0.73],
        [0.0],  # [0.45],
        [0.0],
        [32],
    ):
        att_scale, ctc_scale = scales

        # with_lm_ilm, with_lm

        recog_name = (
            ("/with_lm" if lm_scale > 0.0 and ilm_scale == 0.0 else "")
            + f"/opls_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + (f"_blank{blank_scale}" if blank_scale > 0.0 else "")
            + f"_beam{beam_size}_fix"
        )
        name = prefix_name + model_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "ilm_scale": ilm_scale,
            "use_ctc": True,
            "bsf": bsf,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": models["model_ctc_only"]["prior"],
            "prior_scale": prior_scale,
            "use_lm_first_label": True,
            "encoder_ctc": True,
            "blank_scale_minus": blank_scale,
        }
        recog_config["search_args"] = search_args
        res = recog_model_2(
            task,
            model_baseline_checkpoint,
            model_recog,
            dev_sets=["dev", "test"],
            config=recog_config,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # optsr att + ctc
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.85, 0.15)],
        [0.1],
        [0.3],  # [0.7, 0.73],
        [0.4],  # [0.45],
        [12, 32],
    ):
        att_scale, ctc_scale = scales

        # with_lm_ilm, with_lm

        recog_name = (
            ("/with_lm" if lm_scale > 0.0 and ilm_scale == 0.0 else "") +
            ("/with_lm_ilm" if ilm_scale > 0.0 else "")
            + f"/optsr_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}"
        )
        name = prefix_name_40 + model_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": lm_scale > 0.0,
            "lm_scale": lm_scale,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "ilm_scale": ilm_scale,
            "bsf": bsf_40,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": models["model_ctc_only"]["prior"],
            "prior_scale": prior_scale,
            "encoder_ctc": True,
            "lm_skip": True,
            "add_eos_to_end": True,
            "hash_overwrite": "more time",
        }
        recog_config["search_args"] = search_args
        res = recog_model_2(
            task,
            model_baseline_checkpoint,
            model_recog_time_sync,
            # dev_sets=["dev"],
            dev_sets=["dev", "test"],
            config=recog_config,
            name=name,
            device="gpu",
            search_rqmt={"time": 12},
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    ctc_bs_recog_config = orig_recog_config.copy()
    ctc_bs_recog_config.pop("batch_size")
    ctc_bs_recog_config["max_seqs"] = 1

    # ctc bs att + ctc
    # ctcbs_att0.5_ctc0.5_prior0.5_beam12/recog_results
    # {"dev": 7.23, "test": 6.66}
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.5, 0.5)],
        [0.5],
        [0.0],  # [0.7, 0.73],
        [0.0],  # [0.45],
        [12],
    ):
        att_scale, ctc_scale = scales

        # with_lm_ilm, with_lm

        recog_name = (
            ("/with_lm" if lm_scale > 0.0 and ilm_scale == 0.0 else "")
            + f"/ctcbs_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}"
        )
        name = prefix_name_single_seq + model_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "ctc_prior_file": models["model_ctc_only"]["prior"],
            "prior_scale": prior_scale,
            "encoder_ctc": True,
            "hash_overwrite": "fix log ctc",
        }
        ctc_bs_recog_config["search_args"] = search_args
        res = recog_model_2(
            task,
            model_baseline_checkpoint,
            model_recog_ts_espnet,
            dev_sets=["dev", "test"],
            config=ctc_bs_recog_config,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )


py = sis_run_with_prefix
