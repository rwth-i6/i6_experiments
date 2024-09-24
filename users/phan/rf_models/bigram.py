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


class BigramLMRF(rf.Module):
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
        num_ff_layers: int = 2,
        ff_hidden_dim: int = 1000,
        ff_dropout: float = 0.0,
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
        self.ff_dropout = ff_dropout
        self.num_ff_layers = num_ff_layers
        self.use_bottleneck = use_bottleneck

        self.symbol_embedding_dim = Dim(
            name="symbol_embedding", dimension=symbol_embedding_dim
        )
        self.ff_hidden_dim = Dim(name="ff_hidden", dimension=ff_hidden_dim)

        self.embedding = rf.Embedding(label_target_size, self.symbol_embedding_dim)
        # self.input_layer_norm = rf.LayerNorm(self.symbol_embedding_dim)

        self.layers = rf.Sequential(
            rf.Linear(
                self.symbol_embedding_dim if idx == 0 else self.ff_hidden_dim,
                self.ff_hidden_dim,
            )
            for idx in range(self.num_ff_layers)
        )
        if self.use_bottleneck:
            self.bottleneck_dim = Dim(name="bottleneck", dimension=bottleneck_dim)
            self.bottleneck = rf.Linear(self.ff_hidden_dim, self.bottleneck_dim)
            self.final_linear = rf.Linear(self.bottleneck_dim, output_dim)
        else:
            self.final_linear = rf.Linear(self.ff_hidden_dim, output_dim)
        # self.output_layer_norm = rf.LayerNorm(output_dim)

        for name, param in self.named_parameters():
            param.initial = rf.init.Normal(stddev=0.1) # mean already 0.0

    def default_initial_state(
        self, *, batch_dims: Sequence[Dim], use_batch_dims_for_pos: bool = False
    ) -> rf.State:
        """
        all states are None. Need this to be (maybe) compatible
        with some LM interfaces or RF
        """
        state = rf.State(
            {
                k: None for k, v in self.layers.items()
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
        spatial_dim: Optional[Dim] = None,
        # lengths: torch.Tensor,
        state: Optional[rf.State] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        All states are None. Need states to be (maybe) compatible
        with some LM interfaces or RF

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
        ff_out = embedding_out

        if state is None:
            state = self.default_initial_state(batch_dims=input.dims[:-1])

        new_state = rf.State()

        for layer_name, layer in self.layers.items():
            layer: rf.Linear  # or similar
            # if layer_name in ["0"]: # "0"
            #     breakpoint()
            ff_out = layer(ff_out)
            new_state[layer_name] = None
            # if collected_outputs is not None:
            #     collected_outputs[layer_name] = decoded
        if self.use_bottleneck:
            bottleneck_out = self.bottleneck(ff_out)
            final_linear_out = self.final_linear(bottleneck_out)
        else:
            final_linear_out = self.final_linear(ff_out)
        # output_layer_norm_out = self.output_layer_norm(linear_out)
        return {
            "output": final_linear_out,
            "state": new_state,
        }



def get_model(*, epoch: int, **_kwargs_unused):
    from returnn.config import get_global_config

    config = get_global_config()
    lm_cfg = config.typed_value("lm_cfg", {})
    lm_cls = lm_cfg.pop("class")
    assert lm_cls == "BigramLMRF", "Only Bigram LM are supported"
    extern_data_dict = config.typed_value("extern_data")
    data = Tensor(name="data", **extern_data_dict["data"])
    return BigramLMRF(
        label_target_size=data.sparse_dim,
        output_dim=data.sparse_dim,
        **lm_cfg,
    )

from returnn.tensor import batch_dim
from i6_experiments.users.phan.utils.masking import get_seq_mask
def train_step(*, model: BigramLMRF, extern_data: TensorDict, **_kwargs_unused):
    targets = extern_data["data"]
    delayed = extern_data["delayed"]
    spatial_dim = delayed.get_time_dim_tag()
    targets_len_rf = spatial_dim.dyn_size_ext
    targets_len = targets_len_rf.raw_tensor
    out = model(delayed)
    logits: Tensor = out["output"]
    logits_raw = logits.raw_tensor # (B, T, V)
    targets_raw = targets.raw_tensor.long() # (B, T)
    ce = torch.nn.functional.cross_entropy(
        input = logits_raw.transpose(1, 2),
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

# intended for Albert setup
# for transcription setup use get_model and train_step
def from_scratch_training(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    # import for training only, will fail on CPU servers
    # from i6_native_ops import warp_rnnt

    config = get_global_config()  # noqa
    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx,
        out_dims=[targets_w_eos_spatial_dim]
    )
    out = model(input_labels)
    estimated_type = "logits" if not model.log_prob_output else "log-probs"
    ce = rf.cross_entropy(out, targets_w_eos, targets_w_eos_spatial_dim, estimated_type)
    ce.mark_as_loss(
        "ce",
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
        use_normalized_loss=True,
    )
    # calculate PPL



from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"
