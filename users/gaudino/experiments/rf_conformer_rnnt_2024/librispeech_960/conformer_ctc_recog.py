"""Copied from Albert Zeyer 25.03.2024, then modified
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
import math
import numpy as np
import hashlib
import contextlib
import functools
from sisyphus import tk
from itertools import product

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
    RFModelWithMakeLabelScorer,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.configs import *
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.configs import (
    _batch_size_factor,
    _cfg_lrlin1e_5_295k,
    _get_cfg_lrlin_oclr_by_bs_nep,
)

# from .trafo_lm import trafo_lm_kazuki_import

from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_conformer_ctc import (
    from_scratch_model_def,
    from_scratch_training,
)
from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_recog_ctc_greedy import (
    model_recog,
)
from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_recog_ctc_ls import (
    model_recog as model_recog_ls,
)
from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_recog_ctc_ts import (
    model_recog_time_sync as model_recog_ts,
)
from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_forward_prior import (
    model_forward_prior,
)


if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.gaudino.model_with_checkpoints import (
    ModelWithCheckpoints,
    ModelWithCheckpoint,
)

# From Mohammad, 2023-06-29
# dev-clean  2.27
# dev-other  5.39
# test-clean  2.41
# test-other  5.51
# _returnn_tf_config_filename = (
#   "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config")
# E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
# /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb
# Main train (2035 subepochs): /work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.SAh74CLCNJQi
# 15k batch size, accum grad 2 (1350 steps per epoch?)
# (With batch size 40k (here default), I have usually 495 steps/epoch. Same accum grad 2.)
# peak_lr = 0.9e-3 (1e-3 should also be fine), with Adam, optimizer_epsilon = 1e-08
# phase1: peak_lr / 10 -> peak_lr (45%)
# phase2: peak_lr -> peak_lr / 10 (45%)
# phase3: peak_lr / 10 -> 1e-6 (10%)
# all linear decay and step-based
# specaugment like my orig (same here, specaugorig), speed perturb same here.
# weight decay: L2 1e-4 in some layers (not all): FF, depthwise conv, self-att, output, LSTM, readout
# Final from learning_rates file:
# 2035: EpochData(learningRate=<misleading>, error={
# 'dev_error_ctc': 0.0520755184693418,
# 'dev_error_output/output_prob': 0.035661241551042944,
# 'dev_score_ctc': 0.2796084385705723,
# 'dev_score_output/output_prob': 0.1718613621694714,
# 'devtrain_error_ctc': 0.005757552549708462,
# 'devtrain_error_output/output_prob': 0.005408351877314902,
# 'devtrain_score_ctc': 0.022935187616968285,
# 'devtrain_score_output/output_prob': 0.05237826015574962,
# 'train_error_ctc': 0.05592114304093772,
# 'train_error_output/output_prob': 0.041970552995693494,
# 'train_score_ctc': 0.21249712733341475,
# 'train_score_output/output_prob': 0.20816428663741796,
# }),
# Retrain RETURNN training job (600 subepochs): /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.ZhtaEElHqWlr
# Epoch-wise LR:
# Fixed for 20 subepochs: 1e-4
# Linear(?) decay for remaining (?): 1e-4 to 1e-6
# Final from learning_rates file:
# 600: EpochData(learningRate=1e-06, error={
# 'dev_error_ctc': 0.04999311020358747,
# 'dev_error_output/output_prob': 0.03406881170076022,
# 'dev_score_ctc': 0.2881619431223589,
# 'dev_score_output/output_prob': 0.16851828029171323,
# 'devtrain_error_ctc': 0.003611245977923651,
# 'devtrain_error_output/output_prob': 0.004028583366881808,
# 'devtrain_score_ctc': 0.014762402644778178,
# 'devtrain_score_output/output_prob': 0.0458638666428664,
# 'train_error_ctc': 0.051649620746772214,
# 'train_error_output/output_prob': 0.03977601830532325,
# 'train_score_ctc': 0.19722012590584306,
# 'train_score_output/output_prob': 0.19768974342596793,
# }),


# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""

    from i6_core.returnn.training import PtCheckpoint

    from sisyphus import tk
    from i6_experiments.users.gaudino.recog_2 import recog_model

    task = _get_ls_task()

    _sis_setup_global_prefix(prefix_name)

    # ------------------- ctc only model w eos -------------------------------

    _torch_ckpt_path = "/u/luca.gaudino/setups/2023-08-10--rf-librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.AWwVft0oGy8e/output/models/epoch.1982.pt"

    new_ckpt_path = tk.Path(
        _torch_ckpt_path,
        hash_overwrite="ctc" + "_torch_ckpt",
    )
    new_ckpt = PtCheckpoint(new_ckpt_path)

    recog_config = {
        "batch_size": 1250 * 160,
        "mel_normalization_ted2": False,
        "hash_override": 1,
        "external_language_model": {
            "class": "Trafo_LM_Model",
            "num_layers": 24,
            "layer_out_dim": 1024,
            "att_num_heads": 8,
            "use_pos_enc": True,
            "ff_activation": "relu",
            "pos_enc_diff_pos": True,
        },
        "preload_from_files": {
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
            },
        },
    }

    orig_recog_config = recog_config.copy()

    model_name = "/model_recogs/bsf10/base-24gb-lrlin1e_5_600k_ctc_only_aux4_8_no_mel_norm/ep1982/"

    for lm_scale, beam_size in product([0.55, 0.6, 0.65], []):
        recog_name = f"opls_ctc1.0_trafolm{lm_scale}_beam{beam_size}"

        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "length_normalization_exponent": 1.0,
        }
        recog_config["search_args"] = search_args

        name = _sis_prefix + model_name + recog_name

        res = recog_model(
            task,
            ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_ckpt),
            recog_def=model_recog_ls,
            config=recog_config,
            dev_sets=["dev-other"],
            name=name,
        )
        tk.register_output(name + "/recog_results", res.output)

    # --------------------------- no eos model ----------------------------

    _torch_ckpt_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.6XXpx2nMCWDx/output/models/epoch.1982.pt"
    new_ckpt_path = tk.Path(
        _torch_ckpt_path,
        hash_overwrite="ctc_no_eos" + "_torch_ckpt",
    )
    new_ckpt = PtCheckpoint(new_ckpt_path)

    recog_config = orig_recog_config.copy()

    # _recog(
    #     "model_recogs/bsf10/base-24gb-lrlin1e_5_600k_ctc_only_aux4_8_no_eos/ep1982/ctc_greedy",
    #     ModelWithCheckpoint(
    #         definition=from_scratch_model_def, checkpoint=new_ckpt
    #     ),
    #     model_recog,
    #     dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
    #     recog_config=recog_config,
    # )

    model_name = (
        "/model_recogs/bsf10/base-24gb-lrlin1e_5_600k_ctc_only_aux4_8_no_eos/ep1982/"
    )
    model_name_ts = (
        "/model_recogs/bsf40/base-24gb-lrlin1e_5_600k_ctc_only_aux4_8_no_eos/ep1982/"
    )

    # opls ctc  + lm
    # lmscale 0.65 beam 32
    for lm_scale, prior_scale, beam_size in product([0.0], [0.0, 0.05, 0.1, 0.2, 0.3, 0.4], []):
        recog_name = (
            f"opls_ctc1.0"
            + (f"(_trafolm{lm_scale}" if lm_scale > 0.0 else "")
            + (f"_prior{prior_scale}_fix" if prior_scale > 0.0 else "")
            + f"_beam{beam_size}"
        )
        # recog_name = f"ctc_greedy" + (
        #     f"_prior{prior_scale}_fix" if prior_scale > 0.0 else ""
        # )

        name = _sis_prefix + model_name + recog_name

        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "length_normalization_exponent": 1.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
            "prior_scale": prior_scale,
            "ctc_log_prior": False,
        }
        recog_config["search_args"] = search_args

        res = recog_model(
            task,
            ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_ckpt),
            recog_def=model_recog_ls,
            config=recog_config,
            search_rqmt=None,
            dev_sets=None,
            # dev_sets=["dev-other"],
            name=name,
        )
        tk.register_output(name + "/recog_results", res.output)

    # optsr ctc + trafo lm
    for lm_scale, prior_scale, beam_size in product([0.65, 0.7, 0.75, 0.8], [0.8, 0.9], [12, 32]):
        recog_name = (
            f"optsr_ctc1.0_trafolm{lm_scale}_fix2"
            + (f"_prior{prior_scale}_fix" if prior_scale > 0.0 else "")
            + f"_beam{beam_size}"
        )
        # recog_name = f"ctc_greedy" + (
        #     f"_prior{prior_scale}_fix" if prior_scale > 0.0 else ""
        # )

        name = _sis_prefix + model_name_ts + recog_name

        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "lm_skip": True,
            "length_normalization_exponent": 1.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
            "prior_scale": prior_scale,
            "ctc_log_prior": False,
            "hash_override": 2,
        }
        recog_config["search_args"] = search_args
        recog_config["batch_size"] = 5000 * 160

        res = recog_model(
            task,
            ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_ckpt),
            recog_def=model_recog_ts,
            config=recog_config,
            search_rqmt=None,
            dev_sets=None,
            # dev_sets=["dev-other"],
            name=name,
        )
        tk.register_output(name + "/recog_results", res.output)

    # ------------------------ albert 6.3 model ------------------------------

    _torch_ckpt_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/checkpoints/zeyer_ctc/v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-bpe10k-bpeSample001/epoch.491.pt"
    new_ckpt_path = tk.Path(
        _torch_ckpt_path,
        hash_overwrite="ctc_no_eos" + "_torch_ckpt",
    )
    new_ckpt = PtCheckpoint(new_ckpt_path)

    recog_config = orig_recog_config.copy()
    recog_config["final_ctc_name"] = "enc_logits"

    model_name = "/model_recogs/bsf10/v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-bpe10k-bpeSample001/ep491/"

    for lm_scale, prior_scale, beam_size in product(
        [0.55, 0.6], [0.0, 0.05], []
    ):
        recog_name = (
            f"opls_ctc1.0_trafolm{lm_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0.0 else "")
            + f"_beam{beam_size}"
        )

        name = _sis_prefix + model_name + recog_name

        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "length_normalization_exponent": 1.0,
            "ctc_prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.H7vpOsL8iXP5/output/prior.txt",
            "prior_scale": prior_scale,
            "ctc_log_prior": False,
        }
        recog_config["search_args"] = search_args

        res = recog_model(
            task,
            ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_ckpt),
            recog_def=model_recog_ls,
            config=recog_config,
            search_rqmt=None,
            dev_sets=None,
            # dev_sets=["dev-other"],
            name=name,
        )
        tk.register_output(name + "/recog_results", res.output)

    # recog ctc only model


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from .sis_setup import get_prefix_for_config

        prefix_name = get_prefix_for_config(__file__)
    global _sis_prefix
    _sis_prefix = prefix_name


def _recog(
    name: str,
    model_with_checkpoint: ModelWithCheckpoint,
    recog_def: RecogDef = None,
    recog_config: Optional[Dict[str, Any]] = None,
    *,
    search_rqmt: Optional[Dict[str, Any]] = None,
    dev_sets: Optional[Collection[str]] = None,
):
    from sisyphus import tk
    from i6_experiments.users.gaudino.recog_2 import recog_model

    if recog_def is None:
        recog_def = model_recog

    task = _get_ls_task()

    res = recog_model(
        task,
        model_with_checkpoint,
        recog_def=recog_def,
        config=recog_config,
        search_rqmt=search_rqmt,
        dev_sets=dev_sets,
        name=name,
    )
    tk.register_output(_sis_prefix + "/" + name + "/recog_results", res.output)


_ls_task = None


def _get_ls_task(with_eos_postfix=True, **dataset_opts):
    global _ls_task
    if _ls_task:
        return _ls_task

    from i6_experiments.users.gaudino.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw,
    )

    _ls_task = get_librispeech_task_bpe10k_raw(
        with_eos_postfix=with_eos_postfix, **dataset_opts
    )
    return _ls_task


py = sis_run_with_prefix  # if run directly via `sis m ...`
