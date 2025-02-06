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

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm import (
    trafo_lm_kazuki_import,
)

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import (
        ModelWithCheckpoints,
        ModelWithCheckpoint,
    )


class FFNN_LM_RF(rf.Module):
    r"""Feed-forward neural n-gram LM
    """

    def __init__(
        self,
        label_target_size: Dim,
        output_dim: Dim,
        eos_idx: int,
        context_size: int,
        symbol_embedding_dim: int = 128,
        activation: str = "tanh",
        emebdding_dropout: float = 0.0,
        num_ff_layers: int = 2,
        ff_hidden_dim: int = 1000,
        ff_dropout: float = 0.0,
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = 512,
        input_bottleneck_dim: Optional[int] = None,
    ) -> None:
        """
        :param order: the n in n-gram (>=1)
        :param eos_idx: the index of the end-of-sentence token (assuming this is same as BOS)
        :param label_target_size: shared value from model
        :param output_dim: output vocab size
        """
        super().__init__()
        assert isinstance(context_size, int) and context_size >= 1, "context_size must be int >= 1"
        self.context_size = context_size
        self.eos_idx = eos_idx 
        self.label_target_size = label_target_size
        self.output_dim = output_dim
        self.embedding_dropout = emebdding_dropout
        self.ff_dropout = ff_dropout
        self.num_ff_layers = num_ff_layers
        self.use_bottleneck = use_bottleneck

        self.context_dim = Dim(name="context", dimension=self.context_size) # dim to store the context (the N tokens)
        self.state_dim = Dim(name="state_context", dimension=self.context_size - 1) # dim to store the state (N-1)

        self.symbol_embedding_dim = Dim(
            name="symbol_embedding", dimension=symbol_embedding_dim
        )

        if activation == "tanh":
            self.activation = rf.tanh
        else:
            raise ValueError(f"Activation {activation} not supported")

        self.ff_hidden_dim = Dim(name="ff_hidden", dimension=ff_hidden_dim)

        self.embedding = rf.Embedding(label_target_size, self.symbol_embedding_dim)
        # self.input_layer_norm = rf.LayerNorm(self.symbol_embedding_dim)

        # This is simply the concatenation of the embeddings of the context tokens
        self.context_embedding_dim = Dim(
            name="context_embedding", dimension=symbol_embedding_dim * self.context_size
        )

        if input_bottleneck_dim is None:
            hidden_layers = [
                rf.Linear(
                    self.context_embedding_dim if idx == 0 else self.ff_hidden_dim,
                    self.ff_hidden_dim,
                )
                for idx in range(self.num_ff_layers)
            ]
        else:
            self.input_bottleneck_dim = Dim(name="input_bottleneck", dimension=input_bottleneck_dim)
            input_bottleneck_layer = rf.Linear(self.context_embedding_dim, self.input_bottleneck_dim)
            hidden_layers = [input_bottleneck_layer] + [
                rf.Linear(
                    self.input_bottleneck_dim if idx == 0 else self.ff_hidden_dim,
                    self.ff_hidden_dim,
                )
                for idx in range(self.num_ff_layers)
            ]


        self.layers = rf.Sequential(*hidden_layers)
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
        States are simply the last context-1 tokens of the input.
        Default states are simply (batches, context-1) filled with EOS.
        This is appended to the beginning of the input.
        """
        state = rf.full(
            dims=batch_dims + [self.state_dim],
            fill_value=self.eos_idx,
            sparse_dim=self.label_target_size,
        )
        return state

    def select_state(self, state: rf.State, backrefs) -> rf.State:
        # Seems OK
        # print("state", state.raw_tensor)
        # print("backrefs", backrefs.raw_tensor)
        if self.context_size > 1:
            state = rf.gather(state, indices=backrefs)
        else:
            state = rf.zeros(list(backrefs.dims) + [self.state_dim], dtype="int64") # this is anyway empty
        # print("state after select", state.raw_tensor)
        return state

    def __call__(
        self,
        input: rf.Tensor,
        spatial_dim: Optional[Dim] = None,
        # lengths: torch.Tensor,
        state: Optional[rf.State] = None,
        batch_dims: Optional[Sequence[Dim]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        IMPORTANT: In case of forwarding the whole sequence at once,
        assuming the input is already padded with only one EOS token at the beginning.
        Make sure your sequences already have the EOS token at the beginning.
        This is mostly for compatibility.

        B: batch size;
        S: maximum sequence length in batch;

        Example input:
        [[EOS, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [EOS, 1, 2, 3, 4, 5, 6, 7, PAD, PAD, PAD]]

        Two cases of spatial_dim to handle:
        - single_step_dim: input in (batches,). The code basically makes it (batches, 1)
        - otherwise: input in (batches, S)

        Output shape is (S, B, V) for compatibility with LSTM output format
        and the search implementation.

        Args:
            input: input tensor of shape `(B, S)`.
            spatial_dim: spatial dim of the input tensor (S dim).
                Not needed, just for compatibility.
            state: state tensor of shape `(B, context_size). 

        Returns: dict of {"output", "state"}
            output: output tensor of shape `(S, B, V)` (no S if single step dim).
            state: state tensor of shape `(B, context_size-1)`.
        """
        # makes input (batches, 1) if single_step_dim
        if spatial_dim == single_step_dim:
            input = rf.expand_dim(input, spatial_dim)

        batch_dims = input.remaining_dims(spatial_dim)

        # get default state
        if state is None:
            state = self.default_initial_state(batch_dims=batch_dims)

        # How to concat two RF tensors with dims [batch, context-1], [batch, spatial] along dimension -1?
        # Fuck it we concat using raw tensors
        input_padded_raw = torch.cat([state.raw_tensor, input.raw_tensor], dim=-1).long() # [batch, context-1+spatial]

        if spatial_dim == single_step_dim:
            # make artifical spatial dim for single step
            spatial_dim_with_state = Dim(self.context_size, name="single-step-with-state")
        else:
            # extend to include the context, used later for the sliding windows 
            spatial_dim_with_state = (self.context_size-1) + spatial_dim

        # Fuck it we rawdog in torch
        # windowing using torch.unfold
        # Example with context 3:
        # [EOS, EOS, EOS, 1, 2, 3, ...] -> [[EOS, EOS, 1], [EOS, 1, 2], [1, 2, 3], ...]
        input_padded_windowed_raw = input_padded_raw.unfold(-1, self.context_size, 1)
        input_padded_windowed = rf.Tensor(
            "input_padded_windowed",
            dims=batch_dims + [spatial_dim, self.context_dim],
            sparse_dim=self.label_target_size,
            raw_tensor=input_padded_windowed_raw,
            dtype="int64",
        )

        # # print("spatial dim with state", spatial_dim_with_state.dyn_size_ext.raw_tensor)
        input_padded = rf.Tensor(
            "input_padded",
            dims=batch_dims + [spatial_dim_with_state],
            sparse_dim=self.label_target_size,
            raw_tensor=input_padded_raw,
            dtype="int64",
        )
        # # print(input_padded)
        # # print("input_padded raw", input_padded.raw_tensor)

        # # windowing. Example with context 3:
        # # [EOS, EOS, EOS, 1, 2, 3, ...] -> [[EOS, EOS, 1], [EOS, 1, 2], [1, 2, 3], ...]
        # input_padded_windowed, out_window_spatial_dim = rf.window( # (B, S, context)
        #     input_padded,
        #     spatial_dim=spatial_dim_with_state,
        #     window_dim=self.context_dim,
        #     padding="valid",
        #     stride=1,
        # )
        # # returnn tensor is (Batches, context, S). We want (Batches, S, context)
        # input_padded_windowed = input_padded_windowed.copy_transpose(batch_dims + [out_window_spatial_dim, self.context_dim])

        # print(input_padded_windowed)
        # print(input_padded_windowed.raw_tensor)
        # print("out spatial dim", out_window_spatial_dim)

        # embed the context windows
        input_embed = self.embedding(input_padded_windowed) # (B, S, context, symbol_embedding)
        input_context_embed, context_feature_dim = rf.merge_dims(
            input_embed,
            dims=[self.context_dim, self.symbol_embedding_dim],
            out_dim=self.context_embedding_dim,
        )
        ff_out = input_context_embed
        # --------- Normal forwarding stuffs --------
        if self.embedding_dropout > 0.0:
            ff_out = rf.dropout(
                ff_out,
                drop_prob=self.embedding_dropout,
                axis=context_feature_dim,
            )

        for layer_name, layer in self.layers.items():
            layer: rf.Linear  # or similar
            ff_out = layer(ff_out)
            ff_out = self.activation(ff_out)
            # new_state[layer_name] = None
        if self.use_bottleneck:
            bottleneck_out = self.bottleneck(ff_out)
            final_linear_out = self.final_linear(bottleneck_out)
        else:
            final_linear_out = self.final_linear(ff_out)
        # ---------- End of normal forwarding stuffs --------

        # God knows why I wrote this code
        # Get the last context-1 tokens of input_padded as new state
        if spatial_dim == single_step_dim:
            start = 1
            end = self.context_size
            new_state, new_state_dim = rf.slice(
                source=input_padded,
                axis=spatial_dim_with_state,
                start=start,
                end=end,
                out_dim=self.state_dim,
            )
        else:
            start = spatial_dim_with_state.dyn_size_ext.raw_tensor - self.context_size + 1
            expand_ = [-1]*start.dim() + [self.context_size-1]
            state_idx = start.unsqueeze(-1).expand(*expand_) + torch.arange(self.context_size-1)
            new_state = rf.Tensor(
                "new_state",
                dims=batch_dims + [self.state_dim],
                sparse_dim=self.label_target_size,
                raw_tensor=input_padded_raw.gather(-1, state_idx.to(input_padded_raw.device)),
                dtype="int64",
            )

        # Transpose from (batches, spatial, V) to (spatial, batches, V)
        # This is for compatibility with LSTM output format and the search implementation
        final_linear_spatial_axis = final_linear_out.get_axis_by_tag_name(spatial_dim.name)
        final_linear_out = final_linear_out.copy_move_axis(final_linear_spatial_axis, 0)

        # Remove spatial dim if single_step_dim (same as LSTM output)
        if spatial_dim == single_step_dim:
            final_linear_out = rf.squeeze(final_linear_out, spatial_dim)

        # print("input_padded", input_padded.raw_tensor)
        # print("input_padded_windowed", input_padded_windowed.raw_tensor)
        # print("last state", new_state.raw_tensor)

        return {
            "output": final_linear_out,
            "state": new_state,
        }



def get_model(*, epoch: int, **_kwargs_unused):
    from returnn.config import get_global_config

    config = get_global_config()
    lm_cfg = config.typed_value("lm_cfg", {})
    lm_cls = lm_cfg.pop("class")
    assert lm_cls == "FFNN_LM_RF", "Only FFNN LM is supported"
    extern_data_dict = config.typed_value("extern_data")
    data = Tensor(name="data", **extern_data_dict["data"])
    return FFNN_LM_RF(
        label_target_size=data.sparse_dim,
        output_dim=data.sparse_dim,
        **lm_cfg,
    )

from returnn.tensor import batch_dim
from i6_experiments.users.phan.utils.masking import get_seq_mask
def train_step(*, model: FFNN_LM_RF, extern_data: TensorDict, **_kwargs_unused):
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
