"""Sis script to compute perplexity of lm
"""

from __future__ import annotations

from typing import Optional, Any, Dict

from sisyphus import tk

from returnn.tensor import Tensor, Dim

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_11_09 import (
    Trafo_LM_Model,
    MakeModel,
)
from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_09_03 import (
LSTM_LM_Model,
MakeModel as MakeModelLSTM,
)
from i6_experiments.users.gaudino.forward import forward_model

from i6_experiments.users.zeyer.model_interfaces import ModelDef

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.model_forward_lm import (
    model_forward_lm,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.support.perplexity import ComputePerplexityJob


def sis_run_with_prefix(prefix_name: str = None):
    """run the exp"""
    from .sis_setup import get_prefix_for_config
    from i6_core.returnn.training import PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.gaudino.datasets.tedlium2 import (
        get_tedlium2_task_bpe1k_raw,
    )

    if not prefix_name:
        prefix_name = get_prefix_for_config(__file__)

    task_ted2 = get_tedlium2_task_bpe1k_raw(with_eos_postfix=True)

    extern_data_dict = task_ted2.train_dataset.get_extern_data()
    default_target_key = task_ted2.train_dataset.get_default_target()
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    target_dim = targets.feature_dim_or_sparse_dim

    # load trafo lm
    new_chkpt_path = tk.Path(
        "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/trafo_lm_only_24_02_05/network.020.pt",
        hash_overwrite="torch_ckpt",
    )
    new_chkpt = PtCheckpoint(new_chkpt_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=lm_model_def, checkpoint=new_chkpt
    )

    model_args = {}
    search_args = {"with_eos": True}

    # compute perplexity
    name = prefix_name + "/ted2lm_gelu_no_pos_enc_eos_fix"

    dev_sets = ["dev", "test"]

    forward_out = forward_model(
        task_ted2,
        model_with_checkpoint,
        model_forward_lm,
        dev_sets=dev_sets,
        model_args=model_args,
        search_args=search_args,
        prefix_name=name,
        forward_lm=True,
    )

    for d_name in dev_sets:
        res = ComputePerplexityJob(forward_out[d_name].output).out_ppl
        tk.register_output(
            name + "/" + d_name + f"/ppl",
            res,
        )

    # transciption only lm
    new_chkpt_path = tk.Path(
        "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/trafo_lm_trans_24_09_04/network.005.pt",
        hash_overwrite="ted2_lm_trans_only_ep5",
    )
    new_chkpt = PtCheckpoint(new_chkpt_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=lm_model_def_trans_only, checkpoint=new_chkpt
    )

    model_args = {"num_layers": 1, "lstm_input_dim": 128, "lstm_model_dim": 1000}
    search_args = {"with_eos": True, "hash_overwrite": 2}

    # compute perplexity
    name = prefix_name + "/ted2lm_trans_only_single_seq"

    dev_sets = ["dev", "test"]

    forward_out = forward_model(
        task_ted2,
        model_with_checkpoint,
        model_forward_lm,
        dev_sets=dev_sets,
        model_args=model_args,
        search_args=search_args,
        prefix_name=name,
        forward_lm=True,
    )

    for d_name in dev_sets:
        res = ComputePerplexityJob(forward_out[d_name].output).out_ppl
        tk.register_output(
            name + "/" + d_name + f"/ppl",
            res,
        )

py = sis_run_with_prefix  # if run directly via `sis m ...`

def lm_model_def(
    *,
    epoch: int,
    in_dim: Dim,
    target_dim: Dim,
    model_args: Optional[Dict[str, Any]],
    search_args: Optional[Dict[str, Any]],
) -> Trafo_LM_Model:
    """Function is run within RETURNN."""
    in_dim, epoch  # noqa
    # real input is raw audio, internally it does logmel
    # in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    return MakeModel.make_model(
        target_dim, target_dim, model_args=model_args
    )


lm_model_def: ModelDef[Trafo_LM_Model]
lm_model_def.behavior_version = 16
lm_model_def.backend = "torch"
lm_model_def.batch_size_factor = (  # bsf * 20000
    20  # change batch size here - 20 for att_window - 40 for ctc_prefix
)
lm_model_def.max_seqs = 200  # 1

def lm_model_def_trans_only(
    *,
    epoch: int,
    in_dim: Dim,
    target_dim: Dim,
    model_args: Optional[Dict[str, Any]],
    search_args: Optional[Dict[str, Any]],
) -> LSTM_LM_Model:
    """Function is run within RETURNN."""
    in_dim, epoch  # noqa
    # real input is raw audio, internally it does logmel
    # in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    return MakeModelLSTM.make_model(
        target_dim, target_dim, model_args=model_args
    )


lm_model_def_trans_only: ModelDef[LSTM_LM_Model]
lm_model_def_trans_only.behavior_version = 16
lm_model_def_trans_only.backend = "torch"
lm_model_def_trans_only.batch_size_factor = (  # bsf * 20000
    20  # change batch size here - 20 for att_window - 40 for ctc_prefix
)
lm_model_def_trans_only.max_seqs = 1  # 1
