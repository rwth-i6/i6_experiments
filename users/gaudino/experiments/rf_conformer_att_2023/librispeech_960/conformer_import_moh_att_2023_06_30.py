"""Param Import
"""

from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree
from itertools import product

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.tedlium2.lm_import_2023_11_09 import (
    Trafo_LM_Model,
)
from sisyphus import tk

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.lm_import_2023_09_03 import (
    LSTM_LM_Model,
    # MakeModel,
)
from i6_experiments.users.gaudino.model_interfaces import ModelDef, RecogDef, TrainDef

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog import (
    model_recog,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_time_sync import (
    model_recog_time_sync,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_dump import (
    model_recog_dump,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_ctc_greedy import (
    model_recog_ctc,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_forward import (
    model_forward,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_forward_time_sync import (
    model_forward_time_sync,
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
_torch_ckpt_filename_w_lstm_lm = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/full_w_lm_import_2023_10_18/average.pt"
_torch_ckpt_filename_w_trafo_lm = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/full_w_trafo_lm_import_2024_02_05/average.pt"
# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: str = None):
    """run the exp"""
    from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
    from ._moh_att_2023_06_30_import import map_param_func_v3
    from .sis_setup import get_prefix_for_config
    from i6_core.returnn.training import Checkpoint as TfCheckpoint, PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.gaudino.recog import recog_model
    from i6_experiments.users.gaudino.forward import forward_model

    from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import (
        ConvertTfCheckpointToRfPtJob,
    )
    from i6_experiments.users.gaudino.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw,
    )
    from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.search_errors import (
        ComputeSearchErrorsJob,
    )

    if not prefix_name:
        prefix_name = get_prefix_for_config(__file__)

    task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)

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
        _torch_ckpt_filename_w_lstm_lm, hash_overwrite="torch_ckpt_w_lstm_lm"
    )
    new_chkpt = PtCheckpoint(new_chkpt_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=new_chkpt
    )

    model_args = {
        "add_lstm_lm": True,
    }

    # att only
    for beam_size in [12, 18]:
        recog_name = f"/bsf20/att_beam{beam_size}"
        name = prefix_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "bsf": 20,
        }

        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-other"],  # None for all
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # att + lstm lm
    for lm_scale in [0.3, 0.33, 0.35]:
        for beam_size in []:
            recog_name = f"/bsf40_att_lm{lm_scale}_beam{beam_size}"
            name = prefix_name + recog_name
            search_args = {
                "beam_size": beam_size,
                "add_lstm_lm": True,
                "lm_scale": lm_scale,
            }
            dev_sets = ["dev-other"]  # only dev-other for testing
            # dev_sets = None  # all
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
                name + f"/recog_results",
                res.output,
            )

    # espnet ctc prefix decoder
    for prior_scale, beam_size in product([0.0], [12, 32]):
        name = (
            prefix_name
            + f"/bsf10/ctc_prefix_fix"
            + (f"_prior{prior_scale}" if prior_scale != 0.0 else "")
            + f"_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": 0.0,
            "use_ctc": True,
            "ctc_scale": 1.0,
            "ctc_state_fix": True,
            "bsf": 10,
            "prior_corr": prior_scale != 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=None,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # att + espnet ctc prefix scorer + lstm lm
    for scales, lm_scale, beam_size in product([(0.7, 0.3)], [0.4], []):
        att_scale, ctc_scale = scales
        name = (
            prefix_name
            + f"/bsf40_espnet_att{att_scale}_ctc{ctc_scale}_lm{lm_scale}_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            # att decoder args
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "use_ctc": True,
            "mask_eos": True,
            "add_lstm_lm": True,
            "lm_scale": lm_scale,
            "prior_corr": False,
            "prior_scale": 0.2,
            "length_normalization_exponent": 1.0,  # 0.0 for disabled
            # "window_margin": 10,
            "rescore_w_ctc": False,
        }
        dev_sets = ["dev-other"]  # only dev-other for testing
        # dev_sets = None  # all
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
            name + f"/recog_results",
            res.output,
        )

    # att + espnet ctc prefix scorer
    for scales, prior_scale, beam_size in product(
        [(0.7, 0.3)], [0.1], [12, 32]
    ):
        att_scale, ctc_scale = scales

        name = (
            prefix_name
            + f"/bsf10/opls_att{att_scale}_ctc{ctc_scale}_fix"
            + (f"_prior{prior_scale}" if prior_scale != 0.0 else "")
            + f"_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "use_ctc": True,
            "ctc_scale": ctc_scale,
            "ctc_state_fix": True,
            "bsf": 10,
            "prior_corr": prior_scale != 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=None,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # check for search errors
    for scales in [(0.7, 0.3)]:
        for beam_size in []:
            att_scale, ctc_scale = scales
            name = (
                prefix_name
                + f"/bsf40_espnet_att{att_scale}_ctc{ctc_scale}_beam{beam_size}"
            )
            search_args = {
                "beam_size": beam_size,
                # att decoder args
                "att_scale": att_scale,
                "ctc_scale": ctc_scale,
                "use_ctc": True,
                "mask_eos": True,
            }
            dev_sets = ["dev-other"]  # only dev-other for testing
            # dev_sets = None  # all

            # first recog
            recog_res, recog_out = recog_model(
                task,
                model_with_checkpoint,
                model_recog,
                dev_sets=dev_sets,
                model_args=model_args,
                search_args=search_args,
                prefix_name=name,
            )
            tk.register_output(
                name + f"/recog_results",
                recog_res.output,
            )

            # then forward
            forward_out = forward_model(
                task,
                model_with_checkpoint,
                model_forward,
                dev_sets=dev_sets,
                model_args=model_args,
                search_args=search_args,
                prefix_name=prefix_name,
            )

            # TODO compute search errors
            res = ComputeSearchErrorsJob(
                forward_out.output, recog_out.output
            ).out_search_errors
            tk.register_output(
                name + f"/search_errors",
                res,
            )

    # ctc only decoding
    search_args = {"bsf": 10}
    name = prefix_name + f"/bsf10/ctc_greedy"
    dev_sets = ["dev-other"]  # only dev-other for testing
    # dev_sets = None  # all
    res, _ = recog_model(
        task,
        model_with_checkpoint,
        model_recog_ctc,
        dev_sets=dev_sets,
        model_args=model_args,
        search_args=search_args,
        prefix_name=name,
    )
    tk.register_output(
        name + f"/recog_results",
        res.output,
    )

    # opts att + ctc
    for scales, blank_scale, beam_size in product([(0.65, 0.35)], [2.0], []):
        att_scale, ctc_scale = scales
        name = (
            prefix_name
            + f"/bsf20/opts_att{att_scale}_ctc{ctc_scale}"
            + (f"_blank{blank_scale}" if blank_scale != 0.0 else "")
            + f"_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "blank_scale": blank_scale,
            "bsf": 20,
        }

        #
        # if prior_scale != 0.0:
        #     search_args.update(
        #         {
        #             "prior_corr": True,
        #             "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
        #             "prior_scale": prior_scale,
        #         }
        #     )

        # first recog
        recog_res, recog_out = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            dev_sets=["dev-other"],
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )

        compute_search_errors = False
        if compute_search_errors:
            # then forward
            forward_out = forward_model(
                task,
                model_with_checkpoint,
                model_forward_time_sync,
                dev_sets=dev_sets,
                model_args=model_args,
                search_args=search_args,
                prefix_name=name,
            )

            # TODO compute search errors
            res = ComputeSearchErrorsJob(
                forward_out.output, recog_out.output
            ).out_search_errors
            tk.register_output(
                name + f"/search_errors",
                res,
            )

    # Model with transformer LM
    model_w_trafo_lm_ckpt_path = tk.Path(
        _torch_ckpt_filename_w_trafo_lm, hash_overwrite="torch_ckpt_w_trafo_lm"
    )
    model_w_trafo_lm_ckpt = PtCheckpoint(model_w_trafo_lm_ckpt_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=model_w_trafo_lm_ckpt
    )

    model_args = {
        "add_trafo_lm": True,
        "trafo_lm_args": {
            "num_layers": 24,
            "layer_out_dim": 1024,
            "att_num_heads": 8,
        },
    }

    # opts ctc + trafo lm
    for scales, beam_size in product([(1.0, 0.5)], []):
        ctc_scale, lm_scale = scales
        name = (
            prefix_name
            + f"/bsf20/opts_ctc{ctc_scale}_trafo_lm{lm_scale}"
            + f"_beam{beam_size}"
        )
        search_args = {
            "beam_size": beam_size,
            "att_scale": 0.0,
            "ctc_scale": ctc_scale,
            "add_trafo_lm": True,
            # "remove_trafo_lm_eos": True,
            # "add_eos_to_end": True,
            "lm_scale": lm_scale,
            "bsf": 20,
        }

        recog_res, recog_out = recog_model(
            task,
            model_with_checkpoint,
            model_recog_time_sync,
            dev_sets=["dev-other"],
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            recog_res.output,
        )

    # att + trafo lm
    for lm_scale, beam_size in product([0.42], []):
        recog_name = f"/bsf20/att_trafo_lm{lm_scale}_beam{beam_size}"
        name = prefix_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "bsf": 20,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=None,
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # ctc prefix + trafo lm
    for ctc_scale, prior_scale, lm_scale, beam_size in product(
        [1.0], [0.0, 0.05, 0.1, 0.2, 0.5], [0.65], [12, 32]
    ):
        recog_name = (
            f"/bsf10/opls_ctc{ctc_scale}_fix"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + f"_trafo_lm{lm_scale}_beam{beam_size}"
        )
        name = prefix_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "att_scale": 0.0,
            "ctc_scale": ctc_scale,
            "use_ctc": True,
            "bsf": 10,
            "prior_corr": prior_scale > 0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "ctc_state_fix": True,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-other"],  # None for all
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )

    # att + trafo lm + espnet ctc prefix scorer
    for scales, prior_scale, lm_scale, beam_size in product(
        [(0.8,0.2), (0.82, 0.18)], [0.0, 0.05, 0.1], [0.47], [12, 32]
    ):
        att_scale, ctc_scale = scales
        recog_name = (
            f"/bsf10/opls_att{att_scale}_ctc{ctc_scale}_fix"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + f"_trafo_lm{lm_scale}_beam{beam_size}"
        )
        name = prefix_name + recog_name
        search_args = {
            "beam_size": beam_size,
            "add_trafo_lm": True,
            "lm_scale": lm_scale,
            "att_scale": att_scale,
            "ctc_scale": ctc_scale,
            "use_ctc": True,
            "bsf": 10,
            "prior_corr": prior_scale>0.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt",
            "prior_scale": prior_scale,
            "ctc_state_fix": True,
        }
        res, _ = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=["dev-other"],  # None for all
            model_args=model_args,
            search_args=search_args,
            prefix_name=name,
            # device="gpu",
            # search_mem_rqmt=10,
        )
        tk.register_output(
            name + f"/recog_results",
            res.output,
        )


py = sis_run_with_prefix  # if run directly via `sis m ...`


def sis_run_dump_scores(prefix_name: str = None):
    """run the exp"""
    from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
    from ._moh_att_2023_06_30_import import map_param_func_v3
    from .sis_setup import get_prefix_for_config
    from i6_core.returnn.training import Checkpoint as TfCheckpoint, PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.gaudino.dump import recog_model_dump
    from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import (
        ConvertTfCheckpointToRfPtJob,
    )
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw,
    )

    if not prefix_name:
        prefix_name = get_prefix_for_config(__file__)

    task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)

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
    search_args = {
        "beam_size": 12,
        # att decoder args
        "att_scale": 1.0,
        "ctc_scale": 1.0,
        "use_ctc": False,
        "mask_eos": True,
        "add_lstm_lm": False,
        "prior_corr": False,
        "prior_scale": 0.2,
        "length_normalization_exponent": 1.0,  # 0.0 for disabled
        # "window_margin": 10,
        "rescore_w_ctc": False,
        "dump_ctc": True,
    }

    # new_chkpt_path = tk.Path(_torch_ckpt_filename_w_ctc, hash_overwrite="torch_ckpt_w_ctc")
    new_chkpt = PtCheckpoint(new_chkpt_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=new_chkpt
    )

    dev_sets = ["dev-other"]  # only dev-other for testing
    # dev_sets = None  # all
    res = recog_model_dump(
        task,
        model_with_checkpoint,
        model_recog_dump,
        dev_sets=dev_sets,
        search_args=search_args,
    )
    tk.register_output(
        prefix_name
        # + f"/espnet_att{search_args['att_scale']}_ctc{search_args['ctc_scale']}_beam{search_args['beam_size']}_maskEos"
        + f"/dump_ctc_scores" + f"/scores",
        res.output,
    )


class MakeModel:
    """for import"""

    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        *,
        eos_label: int = 0,
        num_enc_layers: int = 12,
        model_args: Optional[Dict[str, Any]] = None,
        search_args: Optional[Dict[str, Any]] = None,
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.eos_label = eos_label
        self.num_enc_layers = num_enc_layers
        self.search_args = search_args
        self.model_args = model_args

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(
            name="target", dimension=self.target_dim, kind=Dim.Types.Feature
        )
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(
            in_dim,
            target_dim,
            num_enc_layers=self.num_enc_layers,
            model_args=self.model_args,
            search_args=self.search_args,
        )

    @classmethod
    def make_model(
        cls,
        in_dim: Dim,
        target_dim: Dim,
        *,
        model_args: Optional[Dict[str, Any]] = None,
        search_args: Optional[Dict[str, Any]] = None,
        num_enc_layers: int = 12,
    ) -> Model:
        """make"""
        return Model(
            in_dim,
            num_enc_layers=num_enc_layers,
            enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
            enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
            enc_att_num_heads=8,
            enc_conformer_layer_opts=dict(
                conv_norm_opts=dict(use_mask=True),
                self_att_opts=dict(
                    # Shawn et al 2018 style, old RETURNN way.
                    with_bias=False,
                    with_linear_pos=False,
                    with_pos_bias=False,
                    learnable_pos_emb=True,
                    separate_pos_emb_per_head=False,
                ),
                ff_activation=lambda x: rf.relu(x) ** 2.0,
            ),
            target_dim=target_dim,
            blank_idx=target_dim.dimension,
            bos_idx=_get_bos_idx(target_dim),
            eos_idx=_get_eos_idx(target_dim),
            model_args=model_args,
            search_args=search_args,
        )


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_dim: Dim,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        l2: float = 0.0001,
        model_args: Optional[Dict[str, Any]] = None,
        search_args: Optional[Dict[str, Any]] = None,
    ):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout

        self.target_dim_w_b = Dim(
            name="target_w_b",
            dimension=self.target_dim.dimension + 1,
            kind=Dim.Types.Feature,
        )

        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[
                    Dim(32, name="conv1"),
                    Dim(64, name="conv2"),
                    Dim(64, name="conv3"),
                ],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )

        if model_args.get("encoder_ctc", False):
            self.sep_enc_ctc_encoder = ConformerEncoder(
                in_dim,
                enc_model_dim,
                ff_dim=enc_ff_dim,
                input_layer=ConformerConvSubsample(
                    in_dim,
                    out_dims=[
                        Dim(32, name="conv1"),
                        Dim(64, name="conv2"),
                        Dim(64, name="conv3"),
                    ],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],
                ),
                encoder_layer_opts=enc_conformer_layer_opts,
                num_layers=num_enc_layers,
                num_heads=enc_att_num_heads,
                dropout=enc_dropout,
                att_dropout=enc_att_dropout,
            )
            self.sep_enc_ctc_ctc = rf.Linear(
                self.sep_enc_ctc_encoder.out_dim, self.target_dim_w_b
            )

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        self.enc_ctx = rf.Linear(self.encoder.out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = Dim(name="enc_win_dim", dimension=5)

        self.search_args = search_args
        self.ctc = rf.Linear(self.encoder.out_dim, self.target_dim_w_b)

        if model_args.get("add_lstm_lm", False):
            self.lstm_lm = LSTM_LM_Model(target_dim, target_dim)
        if model_args.get("add_trafo_lm", False):
            self.trafo_lm = Trafo_LM_Model(
                target_dim, target_dim, **model_args.get("trafo_lm_args", {})
            )

        self.inv_fertility = rf.Linear(
            self.encoder.out_dim, att_num_heads, with_bias=False
        )

        self.target_embed = rf.Embedding(
            target_dim,
            Dim(name="target_embed", dimension=model_args.get("target_embed_dim", 640)),
        )

        self.s = rf.ZoneoutLSTM(
            self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
            Dim(name="lstm", dimension=1024),
            zoneout_factor_cell=0.15,
            zoneout_factor_output=0.05,
            use_zoneout_output=False,  # like RETURNN/TF ZoneoutLSTM old default
            # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
            # parts_order="ifco",
            parts_order="jifo",  # NativeLSTM (the code above converts it...)
            forget_bias=0.0,  # the code above already adds it during conversion
        )

        self.weight_feedback = rf.Linear(
            att_num_heads, enc_key_total_dim, with_bias=False
        )
        self.s_transformed = rf.Linear(
            self.s.out_dim, enc_key_total_dim, with_bias=False
        )
        self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
        self.readout_in = rf.Linear(
            self.s.out_dim
            + self.target_embed.out_dim
            + att_num_heads * self.encoder.out_dim,
            Dim(name="readout", dimension=1024),
        )
        self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)

        for p in self.parameters():
            p.weight_decay = l2

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        if source.feature_dim:
            assert source.feature_dim.dimension == 1
            source = rf.squeeze(source, source.feature_dim)
        # log mel filterbank features
        source, in_spatial_dim, in_dim_ = rf.stft(
            source,
            in_spatial_dim=in_spatial_dim,
            frame_step=160,
            frame_length=400,
            fft_length=512,
        )
        source = rf.abs(source) ** 2.0
        source = rf.audio.mel_filterbank(
            source, in_dim=in_dim_, out_dim=self.in_dim, sampling_rate=16000
        )
        source = rf.safe_log(source, eps=1e-10) / 2.3026
        # TODO specaug
        # source = specaugment_wei(source, spatial_dim=in_spatial_dim, feature_dim=self.in_dim)  # TODO
        enc, enc_spatial_dim = self.encoder(
            source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
        )
        enc_ctx = self.enc_ctx(enc)
        inv_fertility = rf.sigmoid(self.inv_fertility(enc))
        ctc = rf.log_softmax(self.ctc(enc), axis=self.target_dim_w_b)
        return (
            dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility, ctc=ctc),
            enc_spatial_dim,
        )

    def encode_ctc(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode ctc"""
        assert self.sep_enc_ctc_encoder is not None, "sep_enc_ctc_encoder is None"
        if source.feature_dim:
            assert source.feature_dim.dimension == 1
            source = rf.squeeze(source, source.feature_dim)
        # log mel filterbank features
        source, in_spatial_dim, in_dim_ = rf.stft(
            source,
            in_spatial_dim=in_spatial_dim,
            frame_step=160,
            frame_length=400,
            fft_length=512,
        )
        source = rf.abs(source) ** 2.0
        source = rf.audio.mel_filterbank(
            source, in_dim=in_dim_, out_dim=self.in_dim, sampling_rate=16000
        )
        source = rf.safe_log(source, eps=1e-10) / 2.3026

        enc, enc_spatial_dim = self.sep_enc_ctc_encoder(
            source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
        )

        ctc = rf.log_softmax(self.ctc(enc), axis=self.target_dim_w_b)
        return (
            dict(enc=enc, ctc=ctc),
            enc_spatial_dim,
        )

    @staticmethod
    def encoder_unstack(ext: Dict[str, rf.Tensor]) -> Dict[str, rf.Tensor]:
        """
        prepare the encoder output for the loop (full-sum or time-sync)
        """
        # We might improve or generalize the interface later...
        # https://github.com/rwth-i6/returnn_common/issues/202
        loop = rf.inner_loop()
        return {k: loop.unstack(v) for k, v in ext.items()}

    def decoder_default_initial_state(
        self, *, batch_dims: Sequence[Dim], enc_spatial_dim: Dim
    ) -> rf.State:
        """Default initial state"""
        state = rf.State(
            s=self.s.default_initial_state(batch_dims=batch_dims),
            att=rf.zeros(
                list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]
            ),
            accum_att_weights=rf.zeros(
                list(batch_dims) + [enc_spatial_dim, self.att_num_heads],
                feature_dim=self.att_num_heads,
            ),
        )
        state.att.feature_dim_axis = len(state.att.dims) - 1
        return state

    def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
        """loop step out"""
        return {
            "s": Tensor(
                "s",
                dims=batch_dims + [self.s.out_dim],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
            ),
            "att": Tensor(
                "att",
                dims=batch_dims + [self.att_num_heads * self.encoder.out_dim],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
            ),
        }

    def loop_step(
        self,
        *,
        enc: rf.Tensor,
        enc_ctx: rf.Tensor,
        inv_fertility: rf.Tensor,
        enc_spatial_dim: Dim,
        input_embed: rf.Tensor,
        state: Optional[rf.State] = None,
    ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
        """step of the inner loop"""
        if state is None:
            batch_dims = enc.remaining_dims(
                remove=(enc.feature_dim, enc_spatial_dim)
                if enc_spatial_dim != single_step_dim
                else (enc.feature_dim,)
            )
            state = self.decoder_default_initial_state(
                batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim
            )
        state_ = rf.State()

        prev_att = state.att

        s, state_.s = self.s(
            rf.concat_features(input_embed, prev_att),
            state=state.s,
            spatial_dim=single_step_dim,
        )

        weight_feedback = self.weight_feedback(state.accum_att_weights)
        s_transformed = self.s_transformed(s)
        energy_in = enc_ctx + weight_feedback + s_transformed
        energy = self.energy(rf.tanh(energy_in))
        att_weights = rf.softmax(energy, axis=enc_spatial_dim)
        state_.accum_att_weights = (
            state.accum_att_weights + att_weights * inv_fertility * 0.5
        )
        att0 = rf.dot(att_weights, enc, reduce=enc_spatial_dim, use_mask=False)
        att0.feature_dim = self.encoder.out_dim
        att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder.out_dim))
        state_.att = att

        return {"s": s, "att": att, "att_weights": att_weights}, state_

    def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
        """logits for the decoder"""
        readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
        readout = rf.reduce_out(
            readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim
        )
        readout = rf.dropout(
            readout, drop_prob=0.3, axis=readout.feature_dim
        )  # why is this here?
        logits = self.output_prob(readout)
        return logits


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


def from_scratch_model_def(
    *,
    epoch: int,
    in_dim: Dim,
    target_dim: Dim,
    model_args: Optional[Dict[str, Any]],
    search_args: Optional[Dict[str, Any]],
) -> Model:
    """Function is run within RETURNN."""
    in_dim, epoch  # noqa
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    return MakeModel.make_model(
        in_dim, target_dim, model_args=model_args, search_args=search_args
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = (  # bsf * 20000
    20  # change batch size here - 20 for att_window - 40 for ctc_prefix
)
from_scratch_model_def.max_seqs = 200  # 1


def from_scratch_training(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
):
    """Function is run within RETURNN."""
    assert not data.feature_dim  # raw samples
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)

    batch_dims = data.remaining_dims(data_spatial_dim)
    input_embeddings = model.target_embed(targets)
    input_embeddings = rf.shift_right(
        input_embeddings, axis=targets_spatial_dim, pad_value=0.0
    )

    def _body(input_embed: Tensor, state: rf.State):
        new_state = rf.State()
        loop_out_, new_state.decoder = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=state.decoder,
        )
        return loop_out_, new_state

    loop_out, _, _ = rf.scan(
        spatial_dim=targets_spatial_dim,
        xs=input_embeddings,
        ys=model.loop_step_output_templates(batch_dims=batch_dims),
        initial=rf.State(
            decoder=model.decoder_default_initial_state(
                batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim
            ),
        ),
        body=_body,
    )

    logits = model.decode_logits(input_embed=input_embeddings, **loop_out)

    log_prob = rf.log_softmax(logits, axis=model.target_dim)
    # log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1)
    loss = rf.cross_entropy(
        target=targets,
        estimated=log_prob,
        estimated_type="log-probs",
        axis=model.target_dim,
    )
    loss.mark_as_loss("ce")


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"
