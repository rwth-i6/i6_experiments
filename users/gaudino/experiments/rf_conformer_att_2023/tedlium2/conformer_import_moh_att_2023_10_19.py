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
            "s_use_zoneout_output": True,
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
    prefix_name_single_seq = prefix_name + f"/single_seq" + "_fix_zoneout_output"
    prefix_name_bsf32 = prefix_name + f"/bsf32"
    prefix_name = prefix_name + f"/bsf{bsf}" + "_fix_zoneout_output_240809"

    # ----------------- No LM -------------------

    no_lm_name = "/no_lm"

    # att only
    for model_name in [name for name in att_model_names if not scales_att[name].get("wer", None)]:
        for beam_size in [12, 24]:
            search_args = {
                "beam_size": beam_size,
                "bsf": bsf,
            }
            name = prefix_name + "/" + model_name + no_lm_name + f"/att_beam{beam_size}"
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
    for model_name in [name for name in ctc_model_names if not scales_ctc_prior[name].get("wer", None)]:
        for beam_size, scales in product([], scales_ctc_prior[model_name]["scales"]):
            prior_scale = scales[0]
            search_args = {
                "beam_size": beam_size,
                "blank_idx": 1057,
                "bsf": bsf,
                "prior_corr": True if prior_scale > 0 else False,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
            }
            name = (
                prefix_name
                + "/"
                + model_name
                + no_lm_name
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

    # opls ctc
    for model_name in [name for name in ctc_model_names if not scales_ctc_prior_opls[name].get("wer", None)]:
        for beam_size, scales in product(
            [32], scales_ctc_prior_opls[model_name]["scales"]
        ):
            prior_scale = scales[0]
            search_args = {
                "beam_size": beam_size,
                "bsf": bsf,
                "att_scale": 0.0,
                "ctc_scale": 1.0,
                "use_ctc": True,
                "prior_corr": True if prior_scale > 0 else False,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
            }
            name = (
                prefix_name
                + "/"
                + model_name
                + no_lm_name
                + f"/opls_ctc"
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

    # tsbs ctc
    for model_name in ["model_baseline"]:
        for scales, beam_size in product(
            scales_ctc_prior_tsbs[model_name]["scales"], []  # 32
        ):
            prior_scale = scales[0]
            search_args = {
                "beam_size": beam_size,
                "blank_idx": 1057,
                "max_seq": 1,
                "att_scale": 0.0,
                "ctc_scale": 1.0,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
            }
            name = (
                prefix_name_single_seq
                + "/"
                + model_name
                + no_lm_name
                + f"/tsbs"
                + f"_ctc"
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

    # opls att + ctc
    for model_name in [name for name in both_model_names if not scales_att_ctc_opls[name].get("wer", None)]:
        for scales, beam_size in product(scales_att_ctc_opls[model_name]["scales"], []):  # 12
            (
                att_scale,
                ctc_scale,
                prior_scale
            ) = scales

            search_args = {
                "beam_size": beam_size,
                "blank_idx": 1057,
                "bsf": bsf,
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
            }

            name = (
                prefix_name
                + "/"
                + model_name
                + no_lm_name
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

    # tsbs att + ctc
    for model_name in ["model_baseline"]:
        for scales, beam_size in product(
            scales_ctc_prior_tsbs[model_name]["scales"], []  # 32
        ):
            att_scale, ctc_scale, prior_scale = scales

            search_args = {
                "beam_size": beam_size,
                "blank_idx": 1057,
                "max_seq": 1,
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
            }

            name = (
                prefix_name_single_seq
                + "/"
                + model_name
                + no_lm_name
                + f"/tsbs"
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


    # ----------------- With Trafo LM ----------------- TODO: scales

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
            "s_use_zoneout_output": True,
            "no_ctc": models[model_name].get("no_ctc", False),
            "enc_layer_w_ctc": models[model_name].get("enc_layer_w_ctc", None),
        }
        models_with_pt_ckpt[model_name]["model_args"] = model_args

    with_lm_name = "/with_lm"

    # att + trafo lm
    for model_name, lm_scale, beam_size in product(
        ["model_baseline"], [0.18], []  # 12
    ):
        lm_model_args = copy.deepcopy(models_with_pt_ckpt[model_name]["model_args"])
        name = (
            prefix_name
            + with_lm_name
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
    for model_name, beam_size in product(["model_baseline"], []):  # 12, 32
        for scales in scales_att_ctc_lm_opls[model_name]["scales"]:
            att_scale, ctc_scale, prior_scale, lm_scale = scales
            name = (
                prefix_name
                + with_lm_name
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

    # opls ctc + trafo lm
    for model_name, scales, beam_size in product(
        ["model_ctc_only"],
        [(1.0, 0.6, 0.1), (1.0, 0.7, 0.2)],
        [], # 32
    ):
        ctc_scale, lm_scale, prior_scale = scales
        name = (
            prefix_name
            + with_lm_name
            + "/"
            + model_name
            + f"/opls_ctc{ctc_scale}_trafolm{lm_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0 else "")
            + f"_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": 0.0,
            "ctc_scale": ctc_scale,
            "use_ctc": True,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "bsf": bsf,
            "prior_corr": True if prior_scale > 0 else False,
            "prior_scale": prior_scale,
            "ctc_prior_file": models[model_name]["prior"],
        }

        dev_sets = ["dev", "test"]  # only dev for testing
        # dev_sets = None  # all

        # first recog
        recog_res, recog_out = recog_model(
            task,
            models_with_pt_ckpt[model_name]["ckpt"],
            model_recog,
            dev_sets=dev_sets,
            model_args=models_with_pt_ckpt[model_name]["model_args"],
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
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
            "s_use_zoneout_output": True,
            "no_ctc": models[model_name].get("no_ctc", False),
            "enc_layer_w_ctc": models[model_name].get("enc_layer_w_ctc", None),
        }
        models_with_pt_ckpt[model_name]["model_args"] = copy.deepcopy(model_args)

    preload_from_files_ilm = {
        "01_mini_att_ilm": {
            "prefix": "ilm.",
            "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/mini_att_ilm_24_04_21/average.pt",
        },
        "01_trafo_lm": {
            "prefix": "language_model.",
            "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/trafo_lm_only_24_02_05/network.020.pt",
        },
    }

    # att + trafo lm + ilm correction
    for model_name, lm_scale, ilm_scale, beam_size in product(
        # ["model_baseline", "model_ctc0.5_att0.5"], [0.36] ,[0.28], [12]
        ["model_baseline"],
        [0.36],
        [0.28],
        [],  # 12
    ):
        ilm_model_args = copy.deepcopy(models_with_pt_ckpt[model_name]["model_args"])
        ilm_model_args["preload_from_files"] = preload_from_files_ilm

        name = (
            prefix_name_bsf32
            + "/"
            + model_name
            + f"/att_trafolm{lm_scale}_ilm{ilm_scale}"
            + f"_beam{beam_size}_fffix"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": 1.0,
            "ilm_scale": ilm_scale,
            "lm_scale": lm_scale,
            "bsf": 32,
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
    # 5.74 with att 0.7, ctc 0.3, prior 0.7, trafo 0.6, ilm 0.45
    for model_name, beam_size in product(["model_baseline"], [12, 24]):
        for scales in [(0.7, 0.3, 0.7, 0.6, 0.45)]:
            att_scale, ctc_scale, prior_scale, lm_scale, ilm_scale = scales

            ilm_model_args = copy.deepcopy(
                models_with_pt_ckpt[model_name]["model_args"]
            )
            ilm_model_args["preload_from_files"] = preload_from_files_ilm

            name = (
                prefix_name
                + "/"
                + model_name
                + f"/opls_att{att_scale}_ctc{ctc_scale}_trafolm{lm_scale}_ilm{ilm_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + f"_beam{beam_size}_ffix"
            )
            search_args = {
                "beam_size": beam_size,
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "use_ctc": True,
                "add_trafo_lm": True,
                "ilm_scale": ilm_scale,
                "lm_scale": lm_scale,
                "bsf": bsf,
                "prior_corr": True if prior_scale > 0 else False,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
                "use_first_lm": True,
            }

            recog_res, recog_out = recog_model(
                task,
                models_with_pt_ckpt[model_name]["ckpt"],
                model_recog,
                dev_sets=["dev", "test"],
                model_args=ilm_model_args,
                search_args=search_args,
                prefix_name=name,
            )
            tk.register_output(
                name + f"/recog_results",
                recog_res.output,
            )

    # ts beam search espnet att + ctc + trafo lm + ilm
    for model_name, beam_size in product(["model_baseline"], []): # 1
        for scales in [(0.7, 0.3, 0.7, 0.0, 0.1)]:
            att_scale, ctc_scale, prior_scale, lm_scale, ilm_scale = scales

            ilm_model_args = copy.deepcopy(
                models_with_pt_ckpt[model_name]["model_args"]
            )
            ilm_model_args["preload_from_files"] = preload_from_files_ilm

            name = (
                prefix_name_single_seq
                + "/"
                + model_name
                + f"/optsbs_att{att_scale}_ctc{ctc_scale}_trafolm{lm_scale}_ilm{ilm_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + f"_beam{beam_size}"
            )
            search_args = {
                "beam_size": beam_size,
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "max_seq": 1,
                "use_ctc": True,
                "add_trafo_lm": True,
                "ilm_scale": ilm_scale,
                "lm_scale": lm_scale,
                "bsf": bsf,
                "prior_corr": True if prior_scale > 0 else False,
                "prior_scale": prior_scale,
                "ctc_prior_file": models[model_name]["prior"],
            }

            recog_res, recog_out = recog_model(
                task,
                models_with_pt_ckpt[model_name]["ckpt"],
                model_recog_ts_espnet,
                dev_sets=["dev"],
                model_args=ilm_model_args,
                search_args=search_args,
                prefix_name=name,
            )
            tk.register_output(
                name + f"/recog_results",
                recog_res.output,
            )


py = sis_run_with_prefix
