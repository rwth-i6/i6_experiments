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

from i6_experiments.users.gaudino.models.asr.rf.conformer_rnnt.model_conformer_rnnt import from_scratch_model_def, from_scratch_training, from_scratch_model_def_v2
from i6_experiments.users.gaudino.models.asr.rf.conformer_rnnt.model_recog_rnnt import model_recog as model_recog_rnnt
from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_conformer_ctc import from_scratch_model_def_v2 as from_scratch_model_def_ctc
from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_recog_ctc_greedy import model_recog as model_recog_ctc


# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""

    from i6_core.returnn.training import PtCheckpoint

    _sis_setup_global_prefix(prefix_name)

    ## recog rnnt BPE5k nick

    # imported checkpoint
    _torch_ckpt_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/rnnt_nick_240620/epoch.250.pt"

    new_ckpt_path = tk.Path(
        _torch_ckpt_path,
        hash_overwrite= "rnnt_nick" + "_torch_rf_ckpt",
    )
    new_ckpt = PtCheckpoint(new_ckpt_path)
    model_name = f"model_rnnt_nick/"

    _recog(
        model_name + "rnnt_beam12",
        ModelWithCheckpoint(
            definition=from_scratch_model_def_v2, checkpoint=new_ckpt
        ),
        model_recog_rnnt,
        # dev_sets=["dev-other"],
        dev_sets=None,
        recog_config={
            "mel_normalization_ted2": False,
            "hash_overwrite": "1",
        },
        mini_returnn_data=True,
        device="cpu",
    )

    # ctc model nick
    _torch_ckpt_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/ctc_nick_240730/epoch.500.pt"

    new_ckpt_path = tk.Path(
        _torch_ckpt_path,
        hash_overwrite= "ctc_nick" + "_torch_rf_ckpt",
    )
    new_ckpt = PtCheckpoint(new_ckpt_path)
    model_name = f"model_ctc_nick/"

    _recog(
        model_name + "ctc_greedy_fix",
        ModelWithCheckpoint(
            definition=from_scratch_model_def_ctc, checkpoint=new_ckpt
        ),
        model_recog_ctc,
        dev_sets=["dev-other"],
        # dev_sets=None,
        recog_config={
            "mel_normalization_ted2": False,
            "hash_overwrite": "2",
        },
        mini_returnn_data=True,
        device="cpu",
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
    mini_returnn_data: bool = False,
    device: str = "gpu",
):
    from sisyphus import tk
    from i6_experiments.users.gaudino.recog_2 import recog_model

    if recog_def is None:
        recog_def = model_recog

    task = _get_ls_task(bpe_size="BPE5k", mini_returnn=mini_returnn_data)

    res = recog_model(
        task,
        model_with_checkpoint,
        recog_def=recog_def,
        config=recog_config,
        search_rqmt=search_rqmt,
        dev_sets=dev_sets,
        device=device,
        name=name,
    )
    tk.register_output(_sis_prefix + "/" + name+ "/recog_results", res.output)


_ls_task = None

def _get_ls_task(bpe_size, mini_returnn=False):
    global _ls_task
    if _ls_task:
        return _ls_task

    from i6_experiments.users.gaudino.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw, get_librispeech_task_bpe5k_raw
    )

    if bpe_size == "BPE10k":
        _ls_task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)

    if bpe_size == "BPE5k":
        _ls_task = get_librispeech_task_bpe5k_raw(with_eos_postfix=True, mini_returnn=mini_returnn)

    return _ls_task


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
