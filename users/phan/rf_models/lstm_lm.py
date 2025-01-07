from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection, Dict
import tree
import math
import numpy as np
import torch
import torch.nn as nn
import hashlib
import contextlib
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
    RFModelWithMakeLabelScorer,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm import (
    trafo_lm_kazuki_import,
)
from i6_experiments.users.yang.torch.utils.tensor_ops import mask_eos_label
from i6_experiments.users.yang.torch.utils.masking import get_seq_mask_v2
from i6_experiments.users.yang.torch.loss.ctc_pref_scores_loss import ctc_prefix_posterior_v3
from i6_experiments.users.yang.torch.lm.network.lstm_lm import LSTMLM, LSTMLMConfig

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import (
        ModelWithCheckpoints,
        ModelWithCheckpoint,
    )

_log_mel_feature_dim = 80


###################################################################
def print_gpu_memory_usage(pos='0'):
    print("*********************************************************************************************************")
    unused = torch.cuda.memory_reserved(0) / 1e9 - torch.cuda.memory_allocated(0) / 1e9
    print("Pos: {} Total GPU Memory: {:.2f} GB".format(pos, torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("Pos: {} Allocated GPU Memory: {:.2f} GB".format(pos, torch.cuda.memory_allocated(0) / 1e9))
    print("Pos: {} Cached GPU Memory: {:.2f} GB".format(pos, torch.cuda.memory_reserved(0) / 1e9))
    print("Pos: {} Reserved but Unused GPU Memory: {:.2f} GB".format(pos, unused))
################## extern LM in training ##########################

# default_config
def get_lstm_default_config(**kwargs):
    num_outputs = kwargs.get('num_outputs', 10025)
    embed_dim = kwargs.get('embed_dim', 512)
    hidden_dim = kwargs.get('hidden_dim', 2048)
    num_lstm_layers = kwargs.get('num_lstm_layers',2)
    bottle_neck = kwargs.get('bottle_neck', False)
    bottle_neck_dim = kwargs.get('bottle_neck_dim', 512)
    dropout = kwargs.get('dropout', 0.2)
    default_init_args = {
        'init_args_w':{'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}},
        'init_args_b': {'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}}
    }
    init_args = kwargs.get('init_args', default_init_args)
    model_config = LSTMLMConfig(
        vocab_dim=num_outputs,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_lstm_layers=num_lstm_layers,
        init_args=init_args,
        dropout=dropout,
        trainable=False,
        log_prob_output=False,
    )
    return model_config

# From Luca's rnnt setup
# https://github.com/rwth-i6/i6_experiments/blob/d2b7d3e39d7325f34c5bcc0b4668659e9878fea9/users/gaudino/models/asr/rf/conformer_rnnt/model_conformer_rnnt.py#L241
class LSTMLMRF(rf.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.

    From torchaudio, modified to be used with rf
    """

    def __init__(
        self,
        # cfg: PredictorConfig,
        label_target_size: Dim,
        output_dim: Dim,
        symbol_embedding_dim: int = 128,
        emebdding_dropout: float = 0.0,
        num_lstm_layers: int = 1,
        lstm_hidden_dim: int = 1000,
        lstm_dropout: float = 0.0,
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = 512,
    ) -> None:
        """

        :param cfg: model configuration for the predictor
        :param label_target_size: shared value from model
        :param output_dim: Note that this output dim is for 1(!) single lstm.
            The actual output dim is 2*output_dim since forward and backward lstm
            are concatenated.
        """
        super().__init__()

        self.label_target_size = label_target_size
        self.output_dim = output_dim
        self.embedding_dropout = emebdding_dropout
        self.lstm_dropout = lstm_dropout
        self.num_lstm_layers = num_lstm_layers
        self.use_bottleneck = use_bottleneck

        self.symbol_embedding_dim = Dim(
            name="symbol_embedding", dimension=symbol_embedding_dim
        )
        self.lstm_hidden_dim = Dim(name="lstm_hidden", dimension=lstm_hidden_dim)

        self.embedding = rf.Embedding(label_target_size, self.symbol_embedding_dim)
        # self.input_layer_norm = rf.LayerNorm(self.symbol_embedding_dim)

        self.layers = rf.Sequential(
            rf.LSTM(
                self.symbol_embedding_dim if idx == 0 else self.lstm_hidden_dim,
                self.lstm_hidden_dim,
            )
            for idx in range(self.num_lstm_layers)
        )
        if self.use_bottleneck:
            self.bottleneck_dim = Dim(name="bottleneck", dimension=bottleneck_dim)
            self.bottleneck = rf.Linear(self.lstm_hidden_dim, self.bottleneck_dim)
            self.final_linear = rf.Linear(self.bottleneck_dim, output_dim)
        else:
            self.final_linear = rf.Linear(self.lstm_hidden_dim, output_dim)
        # self.output_layer_norm = rf.LayerNorm(output_dim)

        for name, param in self.named_parameters():
            param.initial = rf.init.Normal(stddev=0.1) # mean already 0.0

    def default_initial_state(
        self, *, batch_dims: Sequence[Dim], use_batch_dims_for_pos: bool = False
    ) -> rf.State:
        """default initial state"""
        state = rf.State(
            {
                k: v.default_initial_state(batch_dims=batch_dims)
                for k, v in self.layers.items()
            }
        )

        return state

    def select_state(self, state: rf.State, backrefs) -> rf.State:
        state = tree.map_structure(
            lambda s: rf.gather(s, indices=backrefs), state
        )
        return state

    def __call__(
        self,
        input: rf.Tensor,
        spatial_dim: Dim,
        # lengths: torch.Tensor,
        state: Optional[rf.State] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        B: batch size;
        U: maximum sequence length in batch;
        D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape `(B, U, output_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``forward``.
        """
        embedding_out = self.embedding(input)
        embedding_out = rf.dropout(
            embedding_out,
            drop_prob=self.embedding_dropout,
            axis=embedding_out.feature_dim,
        )
        # input_layer_norm_out = self.input_layer_norm(embedding_out)

        # lstm_out = input_layer_norm_out
        lstm_out = embedding_out

        if state is None:
            state = self.default_initial_state(batch_dims=input.dims[:-1])

        new_state = rf.State()

        for layer_name, layer in self.layers.items():
            layer: rf.LSTM  # or similar
            # if layer_name in ["0"]: # "0"
            #     breakpoint()
            lstm_out, new_state[layer_name] = layer(
                lstm_out, spatial_dim=spatial_dim, state=state[layer_name]
            )
            # if collected_outputs is not None:
            #     collected_outputs[layer_name] = decoded
        if self.use_bottleneck:
            bottleneck_out = self.bottleneck(lstm_out)
            final_linear_out = self.final_linear(bottleneck_out)
        else:
            final_linear_out = self.final_linear(lstm_out)
        # output_layer_norm_out = self.output_layer_norm(linear_out)
        return {
            "output": final_linear_out,
            "state": new_state,
        }


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



def get_model(*, epoch: int, **_kwargs_unused):
    from returnn.config import get_global_config

    config = get_global_config()
    lm_cfg = config.typed_value("lm_cfg", {})
    lm_cls = lm_cfg.pop("class")
    assert lm_cls == "LSTMLMRF", "Only LSTM LM are supported"
    extern_data_dict = config.typed_value("extern_data")
    data = Tensor(name="data", **extern_data_dict["data"])
    return LSTMLMRF(
        label_target_size=data.sparse_dim,
        output_dim=data.sparse_dim,
        **lm_cfg,
    )

from returnn.tensor import batch_dim
from i6_experiments.users.phan.utils.masking import get_seq_mask
def train_step(*, model: LSTMLMRF, extern_data: TensorDict, **_kwargs_unused):
    targets = extern_data["data"]
    delayed = extern_data["delayed"]
    spatial_dim = delayed.get_time_dim_tag()
    targets_len_rf = spatial_dim.dyn_size_ext
    targets_len = targets_len_rf.raw_tensor
    out = model(delayed, spatial_dim=spatial_dim)
    logits: Tensor = out["output"]
    logits_raw = logits.raw_tensor # (T, B, V)
    targets_raw = targets.raw_tensor.long() # (B, T)
    ce = torch.nn.functional.cross_entropy(
        input = logits_raw.permute(1, 2, 0),
        target = targets_raw,
        reduction="none",
    )
    seq_mask = get_seq_mask(seq_lens=targets_len, max_seq_len=targets_len.max(), device=logits_raw.device)
    loss = (ce*seq_mask).sum()
    ppl = torch.exp(loss/targets_len.sum())
    rf.get_run_ctx().mark_as_loss(
        name="log_ppl", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )
    rf.get_run_ctx().mark_as_loss(
        name="ppl", loss=ppl, as_error=True,
    )

    # doesn't work wtf ?????
    # ce = rf.loss.cross_entropy(
    #     estimated=logits,
    #     target=targets,
    #     axis=targets.sparse_dim,
    #     estimated_type="logits",
    # )
    # rf.get_run_ctx().mark_as_loss(
    #     name="log_ppl",
    #     loss=ce,
    #     custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim),
    # )
    # log_ppl = rf.reduce_mean(ce, axis=spatial_dim) #(B,)
    # ppl = rf.exp(log_ppl) # (B, )
    # ppl = rf.reduce_mean(ppl, axis=batch_dim) # scalar
    # rf.get_run_ctx().mark_as_loss(
    #     name="ppl",
    #     loss=ppl,
    #     as_error=True,
    # )

def model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    lm_cfg = config.typed_value("lm_cfg", {})
    lm_cls = lm_cfg.pop("class")
    assert lm_cls == "LSTMLMRF", "Only LSTM LM are supported"
    extern_data_dict = config.typed_value("extern_data")
    default_target_key = config.typed_value("target")
    targets = Tensor(name="targets", **extern_data_dict[default_target_key])
    return LSTMLMRF(
        label_target_size=targets.sparse_dim,
        output_dim=targets.sparse_dim,
        **lm_cfg,
    )

model_def: ModelDef[Model]
model_def.behavior_version = 21 #16
model_def.backend = "torch"
model_def.batch_size_factor = 160

def train_def(
    *,
    model: Model,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
):
    """Function is run within RETURNN."""
    # <bos> targets
    targets_w_bos, targets_bos_spatial_dim_pad = rf.pad(
        targets,
        padding=[(1, 0)],
        axes=[targets_spatial_dim],
        value=0, # hardcode EOS BOS value
    )
    #targets <eos>
    targets_w_eos, targets_eos_spatial_dim_pad = rf.pad(
        targets,
        padding=[(0, 1)],
        axes=[targets_spatial_dim],
        value=0,
    )
    targets_w_eos_raw = targets_w_eos.raw_tensor.long()
    out = model(targets_w_bos, spatial_dim=targets_bos_spatial_dim_pad[0])
    logits: Tensor = out["output"]
    logits_raw = logits.raw_tensor # (T, B, V)
    ce = torch.nn.functional.cross_entropy(
        input = logits_raw.permute(1, 2, 0),
        target = targets_w_eos_raw,
        reduction="none",
    )
    targets_len_raw = targets_spatial_dim.dyn_size_ext.raw_tensor + 1 # +1 for EOS
    seq_mask = get_seq_mask(
        seq_lens=targets_len_raw,
        max_seq_len=targets_len_raw.max(),
        device=logits_raw.device
        )
    loss = (ce*seq_mask).sum() / targets_len_raw.float().sum()
    ppl = torch.exp(loss)
    rf.get_run_ctx().mark_as_loss(
        name="log_ppl", loss=loss,
    )
    rf.get_run_ctx().mark_as_loss(
        name="ppl", loss=ppl, as_error=True,
    )

train_def: TrainDef[Model]
train_def.learning_rate_control_error_measure = "dev_loss_ppl"

