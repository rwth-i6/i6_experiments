"""Copied from Albert Zeyer 25.03.2024, then modified
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
import math
import numpy as np
import torch
import hashlib
import contextlib
import functools
from sisyphus import tk

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
    RFModelWithMakeLabelScorer,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.tedlium2.configs import *
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.tedlium2.configs import (
    _batch_size_factor,
    _cfg_lrlin1e_5_295k,
    _get_cfg_lrlin_oclr_by_bs_nep,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm import trafo_lm_kazuki_import

if TYPE_CHECKING:
    from i6_experiments.users.gaudino.model_interfaces import ModelDef, RecogDef, TrainDef

from i6_experiments.users.gaudino.model_with_checkpoints import (
    ModelWithCheckpoints,
    ModelWithCheckpoint,
)

from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_conformer_ctc import from_scratch_model_def, from_scratch_training
from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_recog_ctc_greedy import model_recog


# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""

    from i6_core.returnn.training import PtCheckpoint

    _sis_setup_global_prefix(prefix_name)

    # Moh:      dev-clean  2.27, dev-other  5.39, test-clean  2.41,  test-other  5.51
    # RF recog: {"dev-clean": 2.25, "dev-other": 5.34, "test-clean": 2.42, "test-other": 5.56}
    # _recog_imported()

    ctc_train_config = dict(
        batching="laplace:.1000",
        batch_size=15_000 * _batch_size_factor,
        max_seqs=200,
        # max_seq_length_default_target=75,
        specaugment_steps=(10_000, 20_000, 40_000),
        # gradient_clip=0,
        # gradient_clip_global_norm = 1.0
        optimizer={
            "class": "adamw",
            "epsilon": 1e-8,
            "weight_decay": 1e-6,
        },
        # accum_grad_multiple_step=4,
        accum_grad_multiple_step=2,
        # gradient_noise=0.0,
        learning_rate=2.5e-3,
        dynamic_learning_rate=dyn_lr_lin_warmup_invsqrt_decay,
        learning_rate_warmup_steps=40_000,
        learning_rate_invsqrt_norm=40_000,
        max_seq_length_default_target=None,
        gradient_clip_global_norm=5.0,
        aux_loss_layers=[4,8],
        # rnnt_loss=False,
    )

    ctc_train_24gb_config = ctc_train_config.copy()
    ctc_train_24gb_config.update(
        dict(
            torch_amp="bfloat16",
            batch_size=40_000 * _batch_size_factor,
            accum_grad_multiple_step=2,
            learning_rate=1e-3,
            learning_rate_warmup_steps=20_000,
            learning_rate_invsqrt_norm=20_000,
            specaugment_steps=(5_000, 15_000, 25_000),
            grad_scaler=None,
            rf_att_dropout_broadcast=False,
        )
    )
    ctc_train_24gb_config["optimizer"]["epsilon"] = 1e-16
    ctc_train_24gb_config["optimizer"]["weight_decay_modules_blacklist"]: [  # wdblacklist2
            "rf.Embedding",
            "rf.LearnedRelativePositionalEncoding",
        ]


    # train_exp("base-11gb", config_11gb, gpu_mem=11)
    # train_exp("base-11gb-v1", my_config_11gb, num_epochs=400, gpu_mem=11)
    # train_exp( # does not converge
    #     "from-scratch-11gb_aux4_8_lrmaxs522k",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         # total steps after 400 epochs:
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #     },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # train_exp(# does not converge
    #     "from-scratch-11gb_lrmaxs522k",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         # total steps after 400 epochs:
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers": [],
    #     },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmaxs522k_lrmax5e-4",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         # total steps after 400 epochs:
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 3e-4, 1e-5, 1e-6],
    #         "aux_loss_layers": [],
    #     },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )
    #
    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmaxs522k_lrmin8e-5_lrmax8e-4",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         # total steps after 400 epochs:
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [8e-5, 8e-4, 8e-5, 1e-6],
    #         "aux_loss_layers": [],
    #     },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # train_exp( # does not converge
    #     "from-scratch-24gb_lrmaxs85k_lrmin1e-5_lrmax1e-3",
    #     ctc_train_24gb_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 400 epochs:
    #         "learning_rate_piecewise_steps": [85_500, 171_000, 190_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers": [],
    #     },
    #     num_epochs=400,
    # )
    #
    # train_exp( # does not converge
    #     "from-scratch-24gb_lrmaxs85k_lrmin8e-5_lrmax8e-4",
    #     ctc_train_24gb_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 400 epochs:
    #         "learning_rate_piecewise_steps": [85_500, 171_000, 190_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [8e-5, 8e-4, 8e-5, 1e-6],
    #         "aux_loss_layers": [],
    #     },
    #     num_epochs=400,
    # )

    # init from tf ctc only model
    _torch_ckpt_dir_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/without_lm/"

    model_args = {
        "mel_normalization": True,
    }
    new_ckpt_path = tk.Path(
        _torch_ckpt_dir_path + "model_ctc_only_rf_compatible" + "/average.pt",
        hash_overwrite= "model_ctc_only_rf_compatible" + "_torch_ckpt",
    )
    new_ckpt = PtCheckpoint(new_ckpt_path)


    # recog ctc only model
    _recog(
        "model_recogs/model_ctc_only_rf_compatible/ctc_greedy/recog_results",
        ModelWithCheckpoint(
            definition=from_scratch_model_def, checkpoint=new_ckpt
        ),
        model_recog,
    )

    # train_exp( # does not improve, different wer from the beginning
    #     "init_from_tf_lin132k_lrmax8e-4_25eps",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         # total steps after 400 epochs:
    #         "learning_rate_piecewise_steps": [66_000, 132_000, 145_000], # 45% 45 % 10%
    #         # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [8e-5, 8e-4, 8e-5, 1e-6],
    #         "aux_loss_layers": [],
    #         "preload_from_files": {
    #             "encoder": {
    #                 "filename": _torch_ckpt_dir_path + "model_ctc_only_rf_compatible" + "/average.pt",
    #                 "ignore_missing": True,
    #                 "init_for_train": True,
    #             },
    #         },
    #     },
    #     num_epochs=100,
    #     gpu_mem=11,
    # )

    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmaxs522k_lrmin8e-5_lrmax8e-4_aux4_8_adjSpec",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [8e-5, 8e-4, 8e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "specaugment_steps": (5_900, 18_000, 36_000),
    # },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )
    #
    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmaxs522k_lrmin1e-5_lrmax1e-3_aux4_8_adjSpec",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "specaugment_steps": (5_900, 18_000, 36_000),
    # },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmaxs522k_lrmin8e-5_lrmax8e-4_aux4_8_adjSpec2",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [8e-5, 8e-4, 8e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "specaugment_steps": (25_000, 50_000, 100_000),
    #     },
    #     config_deletes=["learning_rate_warmup_steps", "learning_rate_invsqrt_norm"],
    #     num_epochs=400,
    #     gpu_mem=11,
    # )
    #
    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmaxs522k_lrmin1e-5_lrmax1e-3_aux4_8_adjSpec2",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "specaugment_steps": (25_000, 50_000, 100_000),
    #     },
    #     config_deletes=["learning_rate_warmup_steps", "learning_rate_invsqrt_norm"],
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # TODO: try with pretrain

    # train_exp( # does not converge
    #     "from-scratch-11gb_pre3_lrmaxs522k_lrmin1e-5_lrmax1e-3_aux4_8_adjSpec",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "specaugment_steps": (5_900, 18_000, 36_000),
    #         "pretrain_opts": {  # pretrain
    #             "steps": [
    #                 (8 * 500, {"num_layers": 2}),
    #                 (4 * 500, {"num_layers": 4}),
    #                 (4 * 500, {"num_layers": 8}),
    #             ]
    #         },
    # },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # TODO: try with epoch base lr schedule
    ep = 400
    lr = 8e-4
    cyc_ep = int(0.45 * ep)

    # train_exp( # does not converge
    #     "from-scratch-11gb_ep_based_lr_aux4_8",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": None,
    #         # total steps after 2000 epochs: 982.312
    #         # "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         # "specaugment_steps": (5_900, 18_000, 36_000),
    #         # "pretrain_opts": {  # pretrain
    #         #     "steps": [
    #         #         (8 * 500, {"num_layers": 2}),
    #         #         (4 * 500, {"num_layers": 4}),
    #         #         (4 * 500, {"num_layers": 8}),
    #         #     ]
    #         # },
    #         "learning_rates": (
    #         list(np.linspace(lr / 10, lr, cyc_ep))
    #         + list(np.linspace(lr, lr / 10, cyc_ep))
    #         + list(np.linspace(lr / 10, 1e-6, ep - 2 * cyc_ep))
    # )
    # },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmin1e-5_lrmax1e-3_aux4_8_adjSpec6k_noCurrL",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "specaugment_steps": (5_900, 18_000, 36_000),
    #         "epoch_wise_filter": {
    # # (1, 5): {"max_mean_len": 1000},  # better?
    # # older settings:
    # # (1, 5): {"max_mean_len": 200},
    # # (6, 10): {"max_mean_len": 500},
    #     },
    # },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmin8e-5_lrmax8e-4_aux4_8_noSpec_noCurrL",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [8e-5, 8e-4, 8e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "use_specaugment": False,
    #         "epoch_wise_filter": {
    # # (1, 5): {"max_mean_len": 1000},  # better?
    # # older settings:
    # # (1, 5): {"max_mean_len": 200},
    # # (6, 10): {"max_mean_len": 500},
    #     },
    # },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmin1e-5_lrmax1e-3_aux4_8_noSpec_noCurrL",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "use_specaugment": False,
    #         "epoch_wise_filter": {
    # # (1, 5): {"max_mean_len": 1000},  # better?
    # # older settings:
    # # (1, 5): {"max_mean_len": 200},
    # # (6, 10): {"max_mean_len": 500},
    #     },
    # },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmin1e-5_lrmax1e-3_aux4_8_noSpec",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "use_specaugment": False,
    # },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmin1e-5_lrmax1e-3_aux4_8_adjSpec6k_adjCurrL",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "specaugment_steps": (5_900, 18_000, 36_000),
    #         "epoch_wise_filter": {
    # # (1, 5): {"max_mean_len": 1000},  # better?
    # # older settings:
    # # (1, 5): {"max_mean_len": 200},
    # # (6, 10): {"max_mean_len": 500},
    #             (1,2): {"max_mean_len": 400},
    #             (2,4): {"max_mean_len": 800},
    #     },
    # },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmin1e-5_lrmax1e-3_aux4_8_adjSpec6k_accumGrad4",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "specaugment_steps": (5_900, 18_000, 36_000),
    #         "accum_grad_multiple_step": 4,
    # },
    #     config_deletes=["learning_rate_warmup_steps", "learning_rate_invsqrt_norm"],
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    # train_exp( # does not converge
    #     "from-scratch-11gb_lrmax2e-3_aux4_8_adjSpec",
    #     ctc_train_config,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
    #         "learning_rate_piecewise_values": [2e-4, 2e-3, 2e-4, 1e-6],
    #         "aux_loss_layers": [4, 8],
    #         "specaugment_steps": (5_900, 18_000, 36_000),
    # },
    #     num_epochs=400,
    #     gpu_mem=11,
    # )

    train_exp( #
        "from-scratch-11gb_lrmax8e-4_aux4_8_adjSpec_no_norm_loss",
        ctc_train_config,
        config_updates={
            "learning_rate": 1.0,
            "dynamic_learning_rate": dyn_lr_piecewise_linear,
            # total steps after 2000 epochs: 982.312
            "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
            "learning_rate_piecewise_values": [8e-5, 8e-4, 8e-5, 1e-6],
            "aux_loss_layers": [4, 8],
            "specaugment_steps": (5_900, 18_000, 36_000),
            "use_normalized_loss": False,
    },
        num_epochs=400,
        gpu_mem=11,
    )

    train_exp( #
        "from-scratch-11gb_lrmax8e-4_aux4_8_adjSpec_no_norm_loss_no_grad_clip",
        ctc_train_config,
        config_updates={
            "learning_rate": 1.0,
            "dynamic_learning_rate": dyn_lr_piecewise_linear,
            # total steps after 2000 epochs: 982.312
            "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
            "learning_rate_piecewise_values": [8e-5, 8e-4, 8e-5, 1e-6],
            "aux_loss_layers": [4, 8],
            "specaugment_steps": (5_900, 18_000, 36_000),
            "use_normalized_loss": False,
    },
        config_deletes=["learning_rate_warmup_steps", "learning_rate_invsqrt_norm", "gradient_clip_global_norm"],
        num_epochs=400,
        gpu_mem=11,
    )

    train_exp( #
        "from-scratch-11gb_lrmax8e-4_aux4_8_adjSpe_eps2e-8",
        ctc_train_config,
        config_updates={
            "learning_rate": 1.0,
            "dynamic_learning_rate": dyn_lr_piecewise_linear,
            # total steps after 2000 epochs: 982.312
            "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
            "learning_rate_piecewise_values": [8e-5, 8e-4, 8e-5, 1e-6],
            "aux_loss_layers": [4, 8],
            "specaugment_steps": (5_900, 18_000, 36_000),
            "optimizer": {
                "class": "adamw",
                "epsilon": 2e-8,
                "weight_decay": 1e-6,
            },
    },
        num_epochs=400,
        gpu_mem=11,
    )

    train_exp( #
        "from-scratch-11gb_lrmax8e-4_aux4_8_adjSpe_adam",
        ctc_train_config,
        config_updates={
            "learning_rate": 1.0,
            "dynamic_learning_rate": dyn_lr_piecewise_linear,
            # total steps after 2000 epochs: 982.312
            "learning_rate_piecewise_steps": [261_000, 522_000, 580_000], # 45% 45 % 10%
            "learning_rate_piecewise_values": [8e-5, 8e-4, 8e-5, 1e-6],
            "aux_loss_layers": [4, 8],
            "specaugment_steps": (5_900, 18_000, 36_000),
            "optimizer": {
                "class": "adam",
                "epsilon": 1e-8,
                "weight_decay": 1e-6,
            },
    },
        num_epochs=400,
        gpu_mem=11,
    )





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

    task = _get_ted2_task()

    res = recog_model(
        task,
        model_with_checkpoint,
        recog_def=recog_def,
        config=recog_config,
        search_rqmt=search_rqmt,
        dev_sets=dev_sets,
    )
    tk.register_output(_sis_prefix + "/" + name, res.output)


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    *,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    fine_tune: Optional[Union[int, List[Tuple[int, Dict[str, Any]]]]] = None,
    time_rqmt: Optional[int] = None,
    model_avg: bool = False,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.train import (
        train,
    )
    from i6_experiments.users.gaudino.recog_2 import recog_training_exp

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    epoch_wise_filter = config.pop("epoch_wise_filter", None)
    task = _get_ted2_task(epoch_wise_filter=epoch_wise_filter)
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    if "__num_epochs" in config:
        num_epochs = config.pop("__num_epochs")
    if "__gpu_mem" in config:
        gpu_mem = config.pop("__gpu_mem")
    if "__num_processes" in config:
        num_processes = config.pop("__num_processes")

    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        model_def=from_scratch_model_def,
        train_def=from_scratch_training,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else "mpirun",
        time_rqmt=time_rqmt,
    )
    recog_training_exp(
        prefix, task, model_with_checkpoint, recog_def=model_recog, model_avg=model_avg
    )

    if fine_tune:
        if isinstance(fine_tune, int):
            fine_tune = [(fine_tune, {})]
        for ep, opts in fine_tune:
            assert isinstance(ep, int) and isinstance(opts, dict)
            suffix = f"/finetune/{ep}"
            opts = opts.copy()
            if opts:
                for k, v in sorted(opts.items()):
                    k: str
                    suffix += "-" + k.lstrip("_")
                    v = str(v).replace("-", "_")
                    if len(v) > 16 and not k.startswith("_"):
                        suffix += "_" + hashlib.md5(v.encode("utf8")).hexdigest()[:8]
                    else:
                        suffix += v
            num_epochs_ = opts.pop("num_epochs", 50)
            config_ = config.copy()
            config_["import_model_train_epoch1"] = model_with_checkpoint.get_epoch(
                ep
            ).checkpoint
            config_.pop("dynamic_learning_rate")
            lrs = opts.pop("learning_rates", None)
            if lrs is None:
                lr_decay_type = opts.pop(
                    "lr_decay_type", "geomspace"
                )  # geomspace or linspace
                lr_decay_func = getattr(np, lr_decay_type)
                lr = config_["learning_rate"]
                final_lr = opts.pop("final_lr", 1e-7)
                lrs = list(lr_decay_func(lr, final_lr, num=num_epochs_))
            else:
                assert isinstance(lrs, (list, tuple))
                assert len(lrs) == num_epochs_
            config_["learning_rates"] = lrs
            config_["learning_rate"] = float(lrs[-1])
            config_["specaugment_steps"] = (0, 0, 0)
            config_.update({k: v for k, v in opts.items() if not k.startswith("_")})

            finetune_model_with_ckpt = train(
                prefix + suffix,
                task=task,
                config=config_,
                post_config=post_config,
                model_def=from_scratch_model_def,
                train_def=from_scratch_training,
                num_epochs=num_epochs_,
                gpu_mem=gpu_mem,
            )
            # _recog(name + suffix + "/recog/last", finetune_model_with_ckpt.get_last_fixed_epoch())
            recog_training_exp(
                prefix + suffix, task, finetune_model_with_ckpt, recog_def=model_recog
            )

    return model_with_checkpoint


_ls_task = None
_ted2_task = None


def _get_ls_task():
    global _ls_task
    if _ls_task:
        return _ls_task

    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw,
    )

    _ls_task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)
    return _ls_task


def _get_ted2_task(epoch_wise_filter=None):
    global _ted2_task
    if _ted2_task:
        return _ted2_task

    from i6_experiments.users.gaudino.datasets.tedlium2 import (
        get_tedlium2_task_bpe1k_raw,
    )

    _ted2_task = get_tedlium2_task_bpe1k_raw(with_eos_postfix=True, train_epoch_wise_filter=epoch_wise_filter)
    return _ted2_task


py = sis_run_with_prefix  # if run directly via `sis m ...


def model_warmup(*, model: Model, **_kwargs):
    """warmup, for more reliable timing measures"""
    import torch
    import time
    from returnn.config import get_global_config
    from returnn.tensor import Dim
    import returnn.frontend as rf

    config = get_global_config()
    start_time = time.monotonic()
    limit = start_time + config.float("model_warmup_time", 10.0)

    print("*** warming up...")
    while time.monotonic() < limit:
        batch_dim = Dim(10, name="dummy_batch")
        time_dim = Dim(rf.full(dims=[batch_dim], fill_value=16_000), name="dummy_time")
        feat_dim = Dim(1, name="dummy_feat")
        source = rf.zeros([batch_dim, time_dim, feat_dim])
        res = model.encode(source=source, in_spatial_dim=time_dim)
        if source.raw_tensor.device.type == "cuda":
            torch.cuda.synchronize(source.raw_tensor.device)
        res  # noqa  # keep ref to make sure it is calculated
