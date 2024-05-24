"""Param Import
"""

from __future__ import annotations

import tree
from itertools import product
import copy

from sisyphus import tk

from returnn.tensor import Tensor

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
)

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


def sis_run_with_prefix(prefix_name: str = None):
    """run the exp"""
    from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960._moh_att_2023_06_30_import import (
        map_param_func_v3,
    )
    from .sis_setup import get_prefix_for_config
    from i6_core.returnn.training import PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.gaudino.recog import recog_model
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
            "mel_normalization": True,
            "no_ctc": models[model_name].get("no_ctc", False),
            "enc_layer_w_ctc": models[model_name].get("enc_layer_w_ctc", None),
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

    bsf = 10
    prefix_name_single_seq = prefix_name + f"/single_seq"
    prefix_name = prefix_name + f"/bsf{bsf}" + "_fix_zoneout_output"

    ### Single model experiments

    # att only
    # for model_name in list(model_names)[:-1]:
    for model_name in ["model_baseline", "model_ctc0.5_att0.5"]:
        for beam_size in []:
            search_args = {
                "beam_size": beam_size,
                # "lm_scale": lm_scale,
                # "add_trafo_lm": True,
                "max_seq": 1,
                "bsf": bsf,
                # "length_normalization_exponent": len_norm,
            }
            name = prefix_name_single_seq + "/" + model_name + f"/att_beam{beam_size}"
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

    # ctc greedy
    # ctc greedy prior 0.15
    # for model_name in list(model_names)[:-3] + ["model_ctc_only"]:
    for model_name in ["model_baseline"]:
        for beam_size, prior_scale in product([], [0.0]):
            search_args = {
                "beam_size": beam_size,
                "blank_idx": 1057,
                "bsf": bsf,
                # "blank_collapse": True,
                # "blank_threshold": -0.05, # in log space
                "prior_corr": True if prior_scale > 0 else False,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
                "hash_overwrite": "debug",
            }

            name = (
                prefix_name
                + "/"
                + model_name
                + f"/ctc_greedy"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
            )
            res, _ = recog_model(
                task,
                models_with_pt_ckpt[model_name]["ckpt"],
                model_recog_ctc,
                dev_sets=["dev", "test"],  # set to None for all
                model_args=models_with_pt_ckpt[model_name]["model_args"],
                search_args=search_args,
                prefix_name=name,
            )
            tk.register_output(
                name + f"/recog_results",
                res.output,
            )

    ctc_prefix_scorer_model_names = {
        "model_baseline": {
            "scales": [0.1],
        },
        "model_ctc0.9_att0.1": {
            "scales": [0.1],
        },
        "model_ctc0.8_att0.2": {
            "scales": [0.1],
        },
        "model_ctc0.7_att0.3": {
            "scales": [0.1],
        },
        "model_ctc0.6_att0.4": {
            "scales": [0.0],
        },
        "model_ctc0.5_att0.5": {
            "scales": [0.2],
        },
        "model_ctc0.4_att0.6": {
            "scales": [0.1],
        },
        "model_ctc0.3_att0.7": {
            "scales": [0.1],
        },
        "model_ctc0.2_att0.8": {
            "scales": [0.1],
        },
        "model_ctc0.1_att0.9": {
            "scales": [0.1],
        },
        "model_ctc0.001_att0.999": {
            "scales": [0.0],
        },
        "model_ctc_only": {
            "scales": [0.1],
        },
        # "model_ctc0.3_att0.7_lay6": {
        #     "scales": [],  # [(0.85, 0.15, 0.3)],
        # },
        # "model_ctc0.3_att0.7_lay8": {
        #     "scales": [],  # [(0.85, 0.15, 0.55)],
        # },
        # "model_ctc0.3_att0.7_lay10": {
        #     "scales": [],  # [(0.8, 0.2, 0.45)],
        # },
        # "model_ctc1.0_att1.0_lay6": {
        #     "scales": [],  # [(0.8, 0.2, 0.3)],
        # },
        # "model_ctc1.0_att1.0_lay8": {
        #     "scales": [],  # [(0.9, 0.1, 0.45)],
        # },
        # "model_ctc1.0_att1.0_lay10": {
        #     "scales": [],  # [(0.9, 0.1, 0.2)],
        # },
        # "model_ctc0.43_att1.0": {
        #     "scales": [],
        # },
        # "model_ctc0.25_att1.0": {
        #     "scales": [],
        # },
        # "model_ctc0.2_att1.0": {
        #     "scales": [],
        # },
    }

    # ctc prefix scorer
    # for model_name in list(model_names)[:-3] + ["model_ctc_only"]:
    for model_name in ctc_prefix_scorer_model_names.keys():
        for beam_size, prior_scale in product(
            [], ctc_prefix_scorer_model_names[model_name]["scales"]
        ):
            search_args = {
                "beam_size": beam_size,
                "bsf": bsf,
                "att_scale": 0.0,
                "ctc_scale": 1.0,
                "use_ctc": True,
                # "blank_collapse": True,
                # "blank_threshold": -0.05, # in log space
                "prior_corr": True if prior_scale > 0 else False,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
            }

            name = (
                # prefix_name_single_seq
                prefix_name
                + "/"
                + model_name
                + f"/ctc_prefix_search"
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

    opls_model_names = {
        # -------- tuning done ----------
        "model_baseline":{
            "scales": [(0.7, 0.3, 0.7, 0.4), (0.7, 0.3, 0.7, 0.5)],
            "scales_w_fix": [],
        },
        # "model_ctc0.43_att1.0": {
        #     "scales": [(0.8,0.2, 0.6), (0.8, 0.2, 0.7), (0.8, 0.2, 0.9)],
        # },
        # "model_ctc0.25_att1.0": {
        #     "scales": [(0.85, 0.15, 0.6)],
        # },
        # "model_ctc0.2_att1.0": {
        #     "scales": [(0.7, 0.3, 0.4), (0.7, 0.3, 0.5)],
        # },
        # "model_ctc0.3_att0.7": {
        #     "scales": [(0.67, 0.33, 0.7, 0.45), (0.67, 0.33, 0.7, 0.4)],
        # },
        # "model_ctc0.2_att0.8": {
        #     "scales": [(0.8, 0.2, 0.6, 0.45)],
        # },
        # "model_ctc0.1_att0.9": {
        #     "scales": [(0.8, 0.2, 0.6, 0.4), (0.8, 0.2, 0.7, 0.3)],
        # },
        # "model_ctc0.001_att0.999": {
        #     "scales": [(0.9, 0.1, 0.9, 0.45)],
        # },
        # "model_ctc0.3_att0.7_lay6": {
        #     "scales": [(0.85, 0.15, 0.5, 0.45)],
        # },
        # "model_ctc0.3_att0.7_lay8": {
        #     "scales": [(0.85, 0.15, 0.7, 0.45)],
        # },
        # "model_ctc0.3_att0.7_lay10": {
        #     "scales": [(0.9, 0.1, 0.9, 0.45)],
        # },
        # "model_ctc1.0_att1.0_lay6": {
        #     "scales": [(0.85, 0.15, 0.8)],
        # },
        # "model_ctc1.0_att1.0_lay8": {
        #     "scales": [(0.8, 0.2, 0.9)],
        # },
        # "model_ctc1.0_att1.0_lay10": {
        #     "scales": [(0.8, 0.2, 0.9)],
        # },
        # -- TODO: convert ckpt w lm
        # "model_ctc0.9_att0.1": {
        #     "scales": [(0.6, 0.4, 0.6)],
        # },
        # "model_ctc0.8_att0.2": {
        #     "scales": [(0.65, 0.35, 0.6)],
        # },
        # "model_ctc0.7_att0.3": {
        #     "scales": [(0.65, 0.35, 0.8)],
        # },
        # "model_ctc0.6_att0.4": {
        #     "scales": [(0.75, 0.25, 0.7)],
        # },
        "model_ctc0.5_att0.5": {
            "scales": [(0.7, 0.3, 0.8, 0.45), (0.7, 0.3, 0.8, 0.42)],
        },
        # "model_ctc0.4_att0.6": {
        #     "scales": [(0.8, 0.2, 0.8)],
        # },
        # ---------------
        # "model_att_only_currL",
        # "model_att_only_adjSpec",
        # "model_ctc_only",
    }

    # opls att + ctc prefix scorer
    for model_name in ["model_baseline"]:
    # for model_name in opls_model_names:
    #     for scales, beam_size in product(opls_model_names[model_name]["scales"], [12]):
        for scales, beam_size in product([(0.6, 0.4), (0.65, 0.35), (0.7, 0.3), (0.75, 0.25), (0.8, 0.2)], [12]):
            att_scale, ctc_scale, = scales
            prior_scale = 0.0

            search_args = {
                "beam_size": beam_size,
                "blank_idx": 1057,
                "bsf": bsf,
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "use_ctc": True,
                "prior_corr": True if prior_scale > 0 else False,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
                "use_zoneout_output": True,
            }

            name = (
                prefix_name
                + "/"
                + model_name
                + f"/opls_att{att_scale}_ctc{ctc_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + f"_beam{beam_size}"
            )
            res, _ = recog_model(
                task,
                models_with_pt_ckpt[model_name]["ckpt"],
                model_recog,
                dev_sets=["dev"],  # set to None for all
                model_args=models_with_pt_ckpt[model_name]["model_args"],
                search_args=search_args,
                prefix_name=name,
            )
            tk.register_output(
                name + f"/recog_results",
                res.output,
            )

    ctc_beam_search_model_names = {
        "model_ctc0.5_att0.5": {
            "scales": [(0.5, 0.5, 0.6), (0.5, 0.5, 0.8)],
        },
        "model_baseline": {"scales": [(0.5, 0.5, 0.4)]},
    }

    # ctc beam search espnet
    for model_name in ctc_beam_search_model_names:
        for scales, beam_size in product(
            ctc_beam_search_model_names[model_name]["scales"], [32] # 32
        ):
            att_scale, ctc_scale, prior_scale = scales

            search_args = {
                "beam_size": beam_size,
                "blank_idx": 1057,
                "bsf": bsf,
                "max_seq": 1,
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "mask_eos": True,
                # "use_ctc": True,
                "prior_corr": True if prior_scale > 0 else False,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
            }

            name = (
                prefix_name_single_seq
                + "/"
                + model_name
                + f"/ctcbs"
                + (f"_att{att_scale}" if att_scale > 0 else "")
                + f"_ctc{ctc_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + f"_beam{beam_size}"
            )
            res, _ = recog_model(
                task,
                models_with_pt_ckpt[model_name]["ckpt"],
                model_recog_ts_espnet,
                dev_sets=["dev", "test"],  # set to None for all
                model_args=models_with_pt_ckpt[model_name]["model_args"],
                search_args=search_args,
                prefix_name=name,
            )
            tk.register_output(
                name + f"/recog_results",
                res.output,
            )
    # ----------------- With ILM -----------------

    for model_name in model_names:
        model_args = {
            "target_embed_dim": 256,
            "internal_language_model": {
                "class": "MiniAtt_ILM_Model",
            },
            "external_language_model": {
                "class": "Trafo_LM_Model",
            },
            "mel_normalization": True,
            "no_ctc": models[model_name].get("no_ctc", False),
            "enc_layer_w_ctc": models[model_name].get("enc_layer_w_ctc", None),
        }
        models_with_pt_ckpt[model_name]["model_args"] = copy.deepcopy(model_args)

    # att + trafo lm + ilm correction
    for model_name, lm_scale, ilm_scale, beam_size in product(
        # ["model_baseline", "model_ctc0.5_att0.5"], [0.36] ,[0.28], [12]
        ["model_baseline"], [0.3, 0.34, 0.36, 0.4] ,[0.28], [12]
    ):
        ilm_model_args = copy.deepcopy(models_with_pt_ckpt[model_name]["model_args"])
        ilm_model_args["preload_from_files"] = {
            "01_mini_att_ilm": {
                "prefix": "ilm.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/mini_att_ilm_24_04_21/average.pt",
            },
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/trafo_lm_only_24_02_05/network.020.pt",
            }
        }

        name = (
            prefix_name
            + "/"
            + model_name
            + f"/att_trafolm{lm_scale}_ilm{ilm_scale}"
            + f"_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": 1.0,
            "ilm_scale": ilm_scale,
            "lm_scale": lm_scale,
            "bsf": bsf,
            "use_first_lm": True,
            "use_zoneout_output": True,
        }
        recog_res, recog_out = recog_model(
            task,
            models_with_pt_ckpt[model_name]["ckpt"],
            model_recog,
            dev_sets=["dev"],
            model_args=ilm_model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )

    # opls att + ctc + trafo lm + ilm
    for model_name, beam_size, ilm_scale in product(["model_baseline"], [12], [0.1, 0.2, 0.25, 0.3, 0.4]):
        for scales in opls_model_names[model_name]["scales"]:
            att_scale, ctc_scale, prior_scale, lm_scale = scales
            name = (
                prefix_name
                + "/"
                + model_name
                + f"/opls_att{att_scale}_ctc{ctc_scale}_trafolm{lm_scale}_ilm{ilm_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + f"_beam{beam_size}"
            )
            search_args = {
                "beam_size": beam_size,
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "use_ctc": True,
                "add_trafo_lm": True,
                "lm_scale": lm_scale,
                "bsf": bsf,
                "prior_corr": True if prior_scale > 0 else False,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
            }

            recog_res, recog_out = recog_model(
                task,
                models_with_pt_ckpt[model_name]["ckpt"],
                model_recog,
                dev_sets=["dev", "test"],
                model_args=models_with_pt_ckpt[model_name]["model_args"],
                search_args=search_args,
                prefix_name=name,
            )
            tk.register_output(
                name + f"/recog_results",
                recog_res.output,
            )

    # ----------------- With Trafo LM -----------------

    for model_name in model_names:
        model_args = {
            "target_embed_dim": 256,
            "external_language_model": {
                "class": "Trafo_LM_Model",
            },
            "preload_from_files": {
                "01_trafo_lm": {
                    "prefix": "language_model.",
                    "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/trafo_lm_only_24_02_05/network.020.pt",
                }
            },
            "mel_normalization": True,
            "no_ctc": models[model_name].get("no_ctc", False),
            "enc_layer_w_ctc": models[model_name].get("enc_layer_w_ctc", None),
        }
        models_with_pt_ckpt[model_name]["model_args"] = model_args

    # att + trafo lm
    for model_name, lm_scale, beam_size in product(
        ["model_baseline"], [0.13, 0.15, 0.18, 0.2], [12]
    ):
        lm_model_args = copy.deepcopy(models_with_pt_ckpt[model_name]["model_args"])
        name = (
            prefix_name
            + "/"
            + model_name
            + f"/att_trafolm{lm_scale}"
            + f"_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": 1.0,
            "lm_scale": lm_scale,
            "bsf": bsf,
            "use_zoneout_output": True,
            "use_first_lm": True,
        }

        recog_res, recog_out = recog_model(
            task,
            models_with_pt_ckpt[model_name]["ckpt"],
            model_recog,
            dev_sets=["dev"],
            model_args=lm_model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )

    # att + ctc + trafo lm opls
    for model_name, beam_size in product(opls_model_names.keys(), []):
        for scales in opls_model_names[model_name]["scales"]:
            att_scale, ctc_scale, prior_scale, lm_scale = scales
            name = (
                prefix_name
                + "/"
                + model_name
                + f"/opls_att{att_scale}_ctc{ctc_scale}_trafolm{lm_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + f"_beam{beam_size}"
            )
            search_args = {
                "beam_size": beam_size,
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "use_ctc": True,
                "add_trafo_lm": True,
                "lm_scale": lm_scale,
                "bsf": bsf,
                "prior_corr": True if prior_scale > 0 else False,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
            }

            recog_res, recog_out = recog_model(
                task,
                models_with_pt_ckpt[model_name]["ckpt"],
                model_recog,
                dev_sets=["dev", "test"],
                model_args=models_with_pt_ckpt[model_name]["model_args"],
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


py = sis_run_with_prefix
