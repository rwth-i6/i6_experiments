from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
from itertools import product
import copy

from sisyphus import tk

from returnn.tensor import Tensor

from i6_core.returnn.training import PtCheckpoint
from i6_experiments.users.yang.torch.decoding.ctc_aed_joint_simple_version import model_recog

from i6_experiments.users.yang.torch.decoding.ctc_greedy import model_recog_greedy
from i6_experiments.users.yang.torch.decoding.ctc_label_sync import model_recog_label_sync
from i6_experiments.users.yang.torch.decoding.recog import recog_model
from i6_experiments.users.yang.torch.luca_ctc.model_conformer_kd_ctc_fwbw import from_scratch_model_def
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.yang.torch.albert_exp2024_04_23_baselines.aed_new import aed_model_def

from i6_experiments.users.yang.torch.luca_ctc.convert._import_model import convert_lstm_lm





_sis_prefix: Optional[str] = None



def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name








def sis_run_with_prefix(prefix_name: Optional[str] = None):
    if _sis_prefix is None:
        _sis_setup_global_prefix()
    # 4-layer LSTM LM from Mohammad for his cross-domain exps, ppl: dev 48
    tf_lm_path = "/work/asr4/michel/setups-data/lm_training/data-train/tedlium_re_i128_m2048_m2048_m2048_m2048.sgd_b32_lr0_cl2.newbobabs.d0.0.1350/net-model/network.020"
    output_path = "/u/zyang/setups/torch_checkpoints/lm/convert/tedlium"
    convert_lstm_lm(tf_lm_path, output_path, model_target_dim=10025, model_args={"num_layers": 4, "lstm_input_dim": 128, "lstm_model_dim": 2048},output_transpose=True)









py = sis_run_with_prefix


