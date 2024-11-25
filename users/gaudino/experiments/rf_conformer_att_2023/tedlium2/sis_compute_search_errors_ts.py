"""Sisyphus entry point to compute the search errors
"""

from __future__ import annotations

import tree

from sisyphus import tk

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_09_03 import (
    LSTM_LM_Model,
    MakeModel,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_forward_ctc_sum import (
    model_forward_ctc_sum,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_forward_ctc_max import (
    model_forward_ctc_max,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import (
    from_scratch_model_def,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.support.search_errors import (
    ComputeSearchErrorsJob,
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
    from i6_core.returnn.training import PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import (
        ConvertTfCheckpointToRfPtJob,
    )
    from i6_experiments.users.gaudino.datasets.tedlium2 import (
        get_tedlium2_task_bpe1k_raw,
        Tedlium2MetaDataset
    )
    from i6_experiments.users.gaudino.datasets.custom_hdf_dataset import CustomHdfDataset

    from i6_experiments.users.gaudino.forward import (
        forward_model,
        forward_custom_dataset,
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
    for model_name in ["model_baseline"]:
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
    prefix_name = prefix_name + f"/bsf{bsf}"

    model_name = "model_baseline"

    ### check for search errors in time sync results with recombination

    search_w_lenNorm = True
    eval_w_lenNorm = True
    compare_w_max = True

    search_args = {
        "beam_size": 12,
        "att_scale": 0.8,
        "ctc_scale": 0.2,
        "use_ctc": True,
        # "lm_scale": lm_scale,
        # "add_trafo_lm": True,
        "bsf": bsf,
        "length_normalization_exponent": 1.0 if eval_w_lenNorm else 0.0,
    }
    name = prefix_name + "/" + model_name + f"/optsr_ctc0.2_att0.8" + ("_lenNorm_fix" if search_w_lenNorm else "") +"_beam12"
    search_args.update({"hash_override": "1"})

    forward_out_gt = forward_model(
        task,
        models_with_pt_ckpt[model_name]["ckpt"],
        model_forward_ctc_sum if not compare_w_max else model_forward_ctc_max,
        dev_sets=["dev"],
        model_args=model_args,
        search_args=search_args,
        prefix_name=name,
    )

    # define dataset somehow

    from i6_experiments.users.gaudino.datasets.tedlium2 import _raw_audio_opts, bpe1k

    dataset_opts = dict(audio=_raw_audio_opts.copy(), audio_dim=1, vocab=bpe1k, main_key="dev")

    if search_w_lenNorm:
        hdf_dataset = CustomHdfDataset("/u/luca.gaudino/debug/recombine/lenNorm/out_best_wo_blank_len_Norm_fix.hdf")
    else:
        hdf_dataset = CustomHdfDataset("/u/luca.gaudino/debug/recombine/lenNorm/out_best_wo_blank.hdf")

    meta_dataset = Tedlium2MetaDataset(ogg_zip_opts=dataset_opts, additional_dataset=hdf_dataset)

    search_args.update({"remove_eos_from_gt": True, "hash_override_2": "1"})

    if eval_w_lenNorm:
        search_args.update({"length_normalization_exponent": 1.0})

    # forward it
    forward_out_search = forward_custom_dataset(
        dataset=meta_dataset,
        model=models_with_pt_ckpt[model_name]["ckpt"],
        recog_def=model_forward_ctc_sum if not compare_w_max else model_forward_ctc_max,
        model_args=model_args,
        search_args=search_args,
        prefix_name=name,
    )

    res = ComputeSearchErrorsJob(
        forward_out_gt.output, forward_out_search.output
    ).out_search_errors
    tk.register_output(
        name + f"/search_errors" + ("_max" if compare_w_max else "") + ("_evalWLenNorm" if eval_w_lenNorm else ""),
        res,
    )


py = sis_run_with_prefix
