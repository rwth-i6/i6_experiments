"""
Combining two models in recognition (system combination).

We try different combinations of imported and newly trained models.
"""

from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
from itertools import product

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_11_09 import (
    Trafo_LM_Model,
)
from sisyphus import tk

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_09_03 import (
    LSTM_LM_Model,
    # MakeModel,
)
from i6_experiments.users.gaudino.models.asr.rf.ilm_import_2024_04_17 import (
    MiniAtt_ILM_Model,
)
from i6_experiments.users.gaudino.model_interfaces.model_interfaces import ModelDef, TrainDef

from i6_experiments.users.gaudino.models.asr.rf.joint_model_att_ctc.model_recog_joint import (
    model_recog,
)
from i6_experiments.users.gaudino.models.asr.rf.joint_model_att_ctc.model_recog_joint_time_sync import (
    model_recog_time_sync,
)
# from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_ts_espnet import (
#     model_recog_ts_espnet,
# )

from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_conformer_ctc import from_scratch_model_def as from_scratch_model_def_ctc

from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_recog_ctc_greedy import model_recog as model_recog_ctc

from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_recog_ctc_greedy import model_recog_ctc_zeyer


from i6_experiments.users.gaudino.models.asr.rf.joint_model_att_ctc.model_conformer_joint import Model, from_scratch_model_def


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
_torch_ckpt_filename_w_lstm_lm = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/full_w_lm_import_2023_10_18/average.pt"
_torch_ckpt_filename_w_trafo_lm = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/full_w_trafo_lm_import_2024_02_05/average.pt"
_torch_ckpt_filename_base_model = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/base_model/average.pt"
# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: str = None):
    """run the exp"""
    from .sis_setup import get_prefix_for_config
    from i6_core.returnn.training import PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.gaudino.recog_2 import recog_model

    from i6_experiments.users.gaudino.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw,
    )
    from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.support.search_errors import (
        ComputeSearchErrorsJob,
    )

    if not prefix_name:
        prefix_name = get_prefix_for_config(__file__)

    task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)

    bsf = 10
    bsf_40 = 40
    prefix_name_single_seq = prefix_name + "/single_seq"
    prefix_name_40 = prefix_name + f"/bsf{bsf_40}"
    prefix_name = prefix_name + f"/bsf{bsf}"

    with_lm_name = "/with_lm"
    with_lm_ilm_name = "/with_lm_ilm"

    new_chkpt_path = tk.Path(
        _torch_ckpt_filename_w_lstm_lm, hash_overwrite="torch_ckpt_w_lstm_lm"
    )
    empty_path = tk.Path("/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/sep_enc_aed_ctc_empty_24_06_29/average.pt", hash_overwrite="empty")
    new_chkpt = PtCheckpoint(empty_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=new_chkpt
    )

    # ------------------ Tensorflow imported baseline combined with different standalone CTC models ------------------------

    recog_config = {
        "model_args": {
            "external_language_model": {
                "class": "Trafo_LM_Model",
                "num_layers": 24,
                "layer_out_dim": 1024,
                "att_num_heads": 8,
                "use_pos_enc": True,
                "ff_activation": "relu",
                "pos_enc_diff_pos": True,
            },
            "internal_language_model": {
                "class": "MiniAtt_ILM_Model",
                "s_use_zoneout_output": False,
            },
        },

        "preload_from_files": {
            "model_aed": {
                "prefix": "model_aed.",  #  dev-other 5.36
                "filename": _torch_ckpt_filename_base_model,
            },
            "model_ctc": {
                "prefix": "model_ctc.",  # dev-other
                "filename": "/u/luca.gaudino/setups/2023-08-10--rf-librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.AWwVft0oGy8e/output/models/epoch.1981.pt",
            },
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
            },
            "01_mini_att_ilm": {
                "prefix": "model_aed.ilm.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/mini_att_ilm_24_05_28/average.pt",
            },
        },

        "mel_normalization_ted2": False,
    }

    orig_recog_config = recog_config.copy()

    model_name = "/ATT-baseline_tf-ep2000/CTC-base-24gb-lrlin1e_5_600k_ctc_only_aux4_8-ep1981"

    # ilm ckpt torch: /work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/mini_att_ilm_24_05_28/average.pt

    # sanity checks
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(1.0, 0.0)],
        [0.0], # , 0.05, 0.07],
        [0.0], [0.0], []
    ):
        att_scale, ctc_scale = scales
        # recog_name = (
        #     f"/opls_att{att_scale}_ctc{ctc_scale}"
        #     + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
        #     + f"_trafo_lm{lm_scale}"
        #     + f"_ilm{ilm_scale}"
        #     + f"_beam{beam_size}_ffix"
        # )
        recog_name = f"/att_only_check_beam{beam_size}_fix2"
        name = prefix_name + model_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "ilm_scale": ilm_scale,
            "use_ctc": False,
            "bsf": bsf,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "use_lm_first_label": True,
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-other"],
            # dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            # model_args=model_args,
            # search_args=search_args,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.75, 0.25), (0.7, 0.3), (0.65, 0.35)],
        [0.0, 0.05, 0.1], # , 0.05, 0.07],
        [0.0], [0.0], []
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            f"/opls_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}_fix2"
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
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "use_lm_first_label": True,
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-other"],
            # dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            # model_args=model_args,
            # search_args=search_args,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # opls att + ctc + trafo lm + ilm
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.7, 0.3)],
        [0.0, 0.05], # , 0.05, 0.07],
        [0.55], [], [32] # lm + ilm 0.8, 0.2, 0.05, 0.68, 0.4 best - only lm 0.7, 0.3, 0.0, 0.55 best
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            f"/opls_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + f"_trafo_lm{lm_scale}"
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}_fix2"
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
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "use_lm_first_label": True,
            "hash_overwrite": "123"
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            # dev_sets=["dev-clean", "dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            # model_args=model_args,
            # search_args=search_args,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    #---------------------------- model baseline + 6.8 no eos model----------------------------

    # /u/zeyer/setups/combined/2021-05-31/work/i6_core/returnn/training/ReturnnTrainingJob.AwUVfsEzIqWR/output/models/epoch.500.pt

    recog_config["preload_from_files"]["model_ctc"]["filename"] = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.6XXpx2nMCWDx/output/models/epoch.1982.pt"

    model_name = "/ATT-baseline_tf-ep2000/CTC-base-24gb-lrlin1e_5_600k_ctc_only_aux4_8_no_eos-ep1982"

    # opls att + ctc + trafo lm + ilm
    # opls_att0.7_ctc0.3_trafo_lm0.65_ilm0.4_beam32/recog_results
    # {"dev-clean": 1.66, "dev-other": 3.52, "test-clean": 1.85, "test-other": 3.9}
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.7, 0.3)],
        [0.0, 0.05], # , 0.05, 0.07],
        [0.0],
        [0.0],
        [32]
        # ilm 0.4 best TODO further tune ilm
        # lm only 0.7, 0.3, 0.0, 0.58 best
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            (with_lm_name if lm_scale > 0.0 and ilm_scale ==0.0 else "") +
            (with_lm_ilm_name if ilm_scale > 0.0 else "") +
            f"/opls_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}"
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
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "use_lm_first_label": True,
            # "hash_overwrite": "123"
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            # dev_sets=["dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            # model_args=model_args,
            # search_args=search_args,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # optsr att + ctc + trafo lm + ilm
    #
    # optsr_att0.7_ctc0.3_beam32/recog_results
    # {"dev-clean": 2.11, "dev-other": 4.92, "test-clean": 2.3, "test-other": 5.2}
    # optsr_att0.7_ctc0.3_trafo_lm0.6_ilm0.45_beam32/recog_results
    # {"dev-clean": 1.68, "dev-other": 3.61, "test-clean": 1.88, "test-other": 4.0}
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.7, 0.3)],
        [0.0], # , 0.05, 0.07],
        [0.55, 0.6],
        [0.45],
        [12, 24, 32]
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            (with_lm_name if lm_scale > 0.0 and ilm_scale == 0.0 else "") +
            (with_lm_ilm_name if ilm_scale > 0.0 else "") +
            f"/optsr_att{att_scale}_ctc{ctc_scale}"
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
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "lm_skip": True,
            "length_normalization_exponent": 0.0,
            "add_eos_to_end": True,
            # "hash_overwrite": "123"
        }

        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            # dev_sets=["dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            name=name,
            device="gpu",
            search_rqmt={"time": 12}
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # two pass att + ctc + trafo lm + ilm
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.998, 0.002)],
        [0.0], # , 0.05, 0.07],
        [0.5],
        [0.3],
        [12, 32]
        # ilm 0.4 best TODO further tune ilm
        # lm only 0.7, 0.3, 0.0, 0.58 best
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            (with_lm_name if lm_scale > 0.0 and ilm_scale ==0.0 else "") +
            (with_lm_ilm_name if ilm_scale > 0.0 else "") +
            f"/two_pass_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}"
        )
        name = prefix_name + model_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "att_scale": 1.0,
            # "ctc_scale": ctc_scale,
            "ilm_scale": ilm_scale,
            # "use_ctc": True,
            "bsf": bsf,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "rescore_w_ctc": True,
            "rescore_att_scale": att_scale,
            "rescore_ctc_scale": ctc_scale,
            "hash_overwrite": "123"
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            # dev_sets=["dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            # model_args=model_args,
            # search_args=search_args,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )


    # ctc bs att + ctc
    for scales, prior_scale, beam_size in product(
        [(0.8, 0.2), (0.7, 0.3), (0.75, 0.25), (0.65, 0.35), (0.6, 0.4), (0.5, 0.5)],
        [0.0, 0.1], # , 0.05, 0.07],
        []
        # lm only 0.7, 0.3, 0.0, 0.58 best
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            f"/ctcbs_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            # + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
            # + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}"
        )
        name = prefix_name_single_seq + model_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "use_ctc": True,
            "max_seq": 1,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
            "prior_scale": prior_scale,
            "use_lm_first_label": True,
            # "hash_overwrite": "123"
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog_ts_espnet,
            dev_sets=["dev-other"],
            # dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            # model_args=model_args,
            # search_args=search_args,
            name=name,
            device="gpu",
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )



    #---------------------------- model baseline + 6.3 model with trafo lm and ilm ----------------------------

    # /u/zeyer/setups/combined/2021-05-31/work/i6_core/returnn/training/ReturnnTrainingJob.AwUVfsEzIqWR/output/models/epoch.500.pt

    recog_config["preload_from_files"]["model_ctc"]["filename"] = "/work/asr3/zeineldeen/hiwis/luca.gaudino/checkpoints/zeyer_ctc/v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-bpe10k-bpeSample001/epoch.491.pt"

    recog_config["final_ctc_name"] = "enc_logits"

    model_name = "/ATT-baseline_tf-ep2000/CTC-v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-bpe10k-bpeSample001-ep491"

    # sanity check 6.3 model ctc greedy

    ctc_model_path = tk.Path("/work/asr3/zeineldeen/hiwis/luca.gaudino/checkpoints/zeyer_ctc/v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-bpe10k-bpeSample001/epoch.491.pt", hash_overwrite="ctc_model_albert_491")
    ctc_new_chkpt = PtCheckpoint(ctc_model_path)
    ctc_model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def_ctc, checkpoint=ctc_new_chkpt
    )

    ctc_prior_path = "" # TODO compute prior

    recog_config_ctc = {
        # "preload_from_files": {
        #     # "model_ctc": {
        #     #     "prefix": "model_ctc.",  # dev-other
        #     #     "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/checkpoints/zeyer_ctc/v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-bpe10k-bpeSample001/epoch.500.pt",
        #     # },
        #     # "01_trafo_lm": {
        #     #     "prefix": "language_model.",
        #     #     "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
        #     # },
        # },
        "final_ctc_name": "enc_logits",
        # "hash_overwrite": "ctc_recog_zeyer",
        "mel_normalization_ted2": False,
    }

    for prior_scale in [0.0]:
        recog_name = (
            f"/ctc_greedy"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
        )
        name = prefix_name + model_name + recog_name
        # search_args = {
        #     "beam_size": beam_size,
        #     "add_trafo_lm": True,
        #     "lm_scale": lm_scale,
        #     "att_scale": att_scale,
        #     "ctc_scale": ctc_scale,
        #     "ilm_scale": ilm_scale,
        #     "use_ctc": True,
        #     "bsf": bsf,
        #     "prior_corr": prior_scale > 0.0,
        #     "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
        #     "prior_scale": prior_scale,
        #     "use_lm_first_label": True,
        #     "hash_overwrite": "123"
        # }
        # recog_config["search_args"] = search_args
        res = recog_model(
            task,
            ctc_model_with_checkpoint,
            model_recog_ctc,
            # dev_sets=["dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config_ctc,
            # model_args=model_args,
            # search_args=search_args,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # opls att + ctc + trafo lm + ilm
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.8, 0.2), (0.85, 0.15)],
        [0.0], # , 0.05, 0.07],
        [0.0], [], [32]
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            f"/opls_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}_fix2"
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
            "ctc_prior_file": ctc_prior_path,
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "use_lm_first_label": True,
            "hash_overwrite": "123"
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            # dev_sets=["dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            # model_args=model_args,
            # search_args=search_args,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # ---------------------------- att w aux ctc model 5.5 rf + 6.8 no eos model----------------------------

    # /u/zeyer/setups/combined/2021-05-31/work/i6_core/returnn/training/ReturnnTrainingJob.AwUVfsEzIqWR/output/models/epoch.500.pt

    recog_config = orig_recog_config.copy()

    recog_config["preload_from_files"]["model_aed"]["filename"] = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.CMIQklG8vBT6/output/models/epoch.2000.pt"

    recog_config["preload_from_files"]["model_ctc"][
        "filename"] = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.6XXpx2nMCWDx/output/models/epoch.1982.pt"


    recog_config["model_args"].pop("internal_language_model", None)
    recog_config["preload_from_files"].pop("01_mini_att_ilm", None)

    model_name = "/ATT-base-24gb-v6-lrlin1e_5_600k-ep2000/CTC-base-24gb-lrlin1e_5_600k_ctc_only_aux4_8_no_eos-ep1982"


    # opls att + ctc + trafo lm + ilm
    # opls_att0.6_ctc0.4_trafo_lm0.0_beam32/recog_results
    # {"dev-clean": 2.19, "dev-other": 5.03, "test-clean": 2.39, "test-other": 5.25}
    #
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
            [(0.6, 0.4)],
            [0.0],  # , 0.05, 0.07],
            [0.0], [0.0], [12]
    ):
        att_scale, ctc_scale = scales
        recog_name = (
                f"/opls_att{att_scale}_ctc{ctc_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
                + f"_trafo_lm{lm_scale}"
                + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
                + f"_beam{beam_size}"
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
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "use_lm_first_label": True,
            # "hash_overwrite": "123"
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            # dev_sets=["dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            # model_args=model_args,
            # search_args=search_args,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # optsr att + ctc + trafo lm
    # optsr_att0.7_ctc0.3_beam12/recog_results
    # {"dev-clean": 2.19, "dev-other": 5.07, "test-clean": 2.38, "test-other": 5.25}
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.7, 0.3)],
        [0.0], # , 0.05, 0.07],
        [0.0, 0.5],
        [0.0],
        [12, 32]
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            (with_lm_name if lm_scale > 0.0 and ilm_scale == 0.0 else "") +
            (with_lm_ilm_name if ilm_scale > 0.0 else "") +
            f"/optsr_att{att_scale}_ctc{ctc_scale}"
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
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "lm_skip": True,
            "length_normalization_exponent": 0.0,
            "add_eos_to_end": True,
            # "hash_overwrite": "123"
        }

        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            # dev_sets=["dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            name=name,
            device="gpu",
            search_rqmt={"time": 12}
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )


    # ---------------------------- att only model 5.4 rf + 6.8 no eos model----------------------------

    # /u/zeyer/setups/combined/2021-05-31/work/i6_core/returnn/training/ReturnnTrainingJob.AwUVfsEzIqWR/output/models/epoch.500.pt

    recog_config = orig_recog_config.copy()

    recog_config["preload_from_files"]["model_aed"]["filename"] = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.Kh5PT8N397Fe/output/models/epoch.2000.pt"

    recog_config["preload_from_files"]["model_ctc"][
        "filename"] = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.6XXpx2nMCWDx/output/models/epoch.1982.pt"


    recog_config["model_args"].pop("internal_language_model", None)
    recog_config["preload_from_files"].pop("01_mini_att_ilm", None)

    model_name = "/ATT-base-24gb-v6-lrlin1e_5_600k_noCTC-ep2000/CTC-base-24gb-lrlin1e_5_600k_ctc_only_aux4_8_no_eos-ep1982"


    # opls att + ctc + trafo lm + ilm
    #
    # opls_att0.6_ctc0.4_beam32/recog_results
    # {"dev-clean": 2.1, "dev-other": 4.81, "test-clean": 2.3, "test-other": 5.06}
    #
    # with lm (no prior) 0.65, 0.35, 0.0, 0.45, 0.0, 32: {"dev-clean": 1.82, "dev-other": 3.81, "test-clean": 1.95, "test-other": 4.26}
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
            [(0.6, 0.4)],
            [0.0],  # , 0.05, 0.07],
            [0.0], [0.0], [12, 32]
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            (with_lm_name if lm_scale > 0.0 and ilm_scale == 0.0 else "") +
            (with_lm_ilm_name if ilm_scale > 0.0 else "") +
                f"/opls_att{att_scale}_ctc{ctc_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
                + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
                + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
                + f"_beam{beam_size}"
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
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "use_lm_first_label": True,
            # "hash_overwrite": "123"
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            # dev_sets=["dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            # model_args=model_args,
            # search_args=search_args,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # optsr att + ctc + trafo lm
    # optsr_att0.65_ctc0.35_beam32/recog_results
    # {"dev-clean": 2.11, "dev-other": 4.86, "test-clean": 2.3, "test-other": 5.03}
    #
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.65, 0.35)],
        [0.0], # , 0.05, 0.07],
        [0.0, 0.5],
        [0.0],
        [12, 32]
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            (with_lm_name if lm_scale > 0.0 and ilm_scale == 0.0 else "") +
            (with_lm_ilm_name if ilm_scale > 0.0 else "") +
            f"/optsr_att{att_scale}_ctc{ctc_scale}"
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
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "lm_skip": True,
            "length_normalization_exponent": 0.0,
            "add_eos_to_end": True,
            # "hash_overwrite": "123"
        }

        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            # dev_sets=["dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            name=name,
            device="gpu",
            search_rqmt={"time": 12}
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # two pass att + ctc + trafo lm + ilm
    for scales, prior_scale, lm_scale, ilm_scale, beam_size in product(
        [(0.98, 0.02)], #, (0.995, 0.005), (0.998, 0.002), (0.999, 0.001)],
        [0.0], # , 0.05, 0.07],
        [0.0, 0.3], # 0.5
        [0.0],
        [32]
        # ilm 0.4 best TODO further tune ilm
        # lm only 0.7, 0.3, 0.0, 0.58 best
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            (with_lm_name if lm_scale > 0.0 and ilm_scale ==0.0 else "") +
            (with_lm_ilm_name if ilm_scale > 0.0 else "") +
            f"/two_pass_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + (f"_trafo_lm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_ilm{ilm_scale}" if ilm_scale > 0.0 else "")
            + f"_beam{beam_size}"
        )
        name = prefix_name + model_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "att_scale": 1.0,
            # "ctc_scale": ctc_scale,
            "ilm_scale": ilm_scale,
            # "use_ctc": True,
            "bsf": bsf,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.U1U9MgoNwGYk/output/prior.txt",
            "prior_scale": prior_scale,
            "is_log_prior": False,
            "rescore_w_ctc": True,
            "rescore_att_scale": att_scale,
            "rescore_ctc_scale": ctc_scale,
            "hash_overwrite": "123"
        }
        recog_config["search_args"] = search_args
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            # dev_sets=["dev-other"],
            dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            config=recog_config,
            # model_args=model_args,
            # search_args=search_args,
            name=name,
            device="gpu",
            # search_mem_rqmt=15,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )








py = sis_run_with_prefix  # if run directly via `sis m ...`
