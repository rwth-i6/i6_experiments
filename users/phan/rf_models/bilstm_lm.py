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
from i6_experiments.users.phan.rf_models.bilstm import BiLSTM

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import (
        ModelWithCheckpoints,
        ModelWithCheckpoint,
    )

_log_mel_feature_dim = 80


class BiLSTMLMRF(rf.Module):
    r"""
    Bi-directional masked LSTM LM
    """

    def __init__(
        self,
        # cfg: PredictorConfig,
        input_dim: Dim,
        output_dim: Dim,
        symbol_embedding_dim: int = 128,
        emebdding_dropout: float = 0.0,
        num_blstm_layers: int = 1,
        blstm_hidden_dim: int = 1000,
        blstm_dropout: float = 0.0,
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = 512,
    ) -> None:
        """

        :param cfg: model configuration for the predictor
        :param label_target_size: shared value from model
        :param output_dim: Note that this output dim is for 1(!) single lstm
            in one direction.
            The actual output dim is 2*output_dim since forward and backward lstm
            are concatenated.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dropout = emebdding_dropout
        self.blstm_dropout = blstm_dropout
        self.num_blstm_layers = num_blstm_layers
        self.use_bottleneck = use_bottleneck

        self.symbol_embedding_dim = Dim(
            name="symbol_embedding", dimension=symbol_embedding_dim
        )
        self.blstm_hidden_dim = Dim(name="blstm_hidden", dimension=blstm_hidden_dim)

        self.embedding = rf.Embedding(input_dim, self.symbol_embedding_dim)
        # self.input_layer_norm = rf.LayerNorm(self.symbol_embedding_dim)

        self.layers = rf.Sequential(
            BiLSTM(
                self.symbol_embedding_dim if idx == 0 else self.blstm_hidden_dim,
                self.blstm_hidden_dim,
            )
            for idx in range(self.num_blstm_layers)
        )
        if self.use_bottleneck:
            self.bottleneck_dim = Dim(name="bottleneck", dimension=bottleneck_dim)
            self.bottleneck = rf.Linear(self.blstm_hidden_dim, self.bottleneck_dim)
            self.final_linear = rf.Linear(self.bottleneck_dim, output_dim)
        else:
            self.final_linear = rf.Linear(self.blstm_hidden_dim, output_dim)

        for name, param in self.named_parameters():
            param.initial = rf.init.Normal(stddev=0.1) # mean already 0.0

    # def default_initial_state(
    #     self, *, batch_dims: Sequence[Dim], use_batch_dims_for_pos: bool = False
    # ) -> rf.State:
    #     """default initial state"""
    #     state = rf.State(
    #         {
    #             k: v.default_initial_state(batch_dims=batch_dims)
    #             for k, v in self.layers.items()
    #         }
    #     )

    #     return state

    # def select_state(self, state: rf.State, backrefs) -> rf.State:
    #     state = tree.map_structure(
    #         lambda s: rf.gather(s, indices=backrefs), state
    #     )
    #     return state

    def __call__(
        self,
        input: rf.Tensor,
        spatial_dim: Dim,
        batch_dims: Sequence[Dim],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        Note that a bi lstm does not, and should not have states.
        It should not operate on single_step_dim as well.
        """
        embedding_out = self.embedding(input)
        embedding_out = rf.dropout(
            embedding_out,
            drop_prob=self.embedding_dropout,
            axis=embedding_out.feature_dim,
        )
        # input_layer_norm_out = self.input_layer_norm(embedding_out)

        # lstm_out = input_layer_norm_out
        blstm_out = embedding_out

        for layer_name, layer in self.layers.items():
            layer: BiLSTM  # or similar
            blstm_out, _ = layer(
                blstm_out, spatial_dim=spatial_dim,
                state=layer.default_initial_state(batch_dims=[batch_dims]),
            )
        if self.use_bottleneck:
            bottleneck_out = self.bottleneck(blstm_out)
            final_linear_out = self.final_linear(bottleneck_out)
        else:
            final_linear_out = self.final_linear(blstm_out)
        # output_layer_norm_out = self.output_layer_norm(linear_out)
        return {
            "output": final_linear_out,
            "state": rf.State(), # placeholder
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


# TODO: adapt for bi lstm lm
# def get_model(*, epoch: int, **_kwargs_unused):
#     from returnn.config import get_global_config

#     config = get_global_config()
#     lm_cfg = config.typed_value("lm_cfg", {})
#     lm_cls = lm_cfg.pop("class")
#     assert lm_cls == "LSTMLMRF", "Only LSTM LM are supported"
#     extern_data_dict = config.typed_value("extern_data")
#     data = Tensor(name="data", **extern_data_dict["data"])
#     return LSTMLMRF(
#         label_target_size=data.sparse_dim,
#         output_dim=data.sparse_dim,
#         **lm_cfg,
#     )

# from returnn.tensor import batch_dim
# from i6_experiments.users.phan.utils.masking import get_seq_mask
# def train_step(*, model: LSTMLMRF, extern_data: TensorDict, **_kwargs_unused):
#     targets = extern_data["data"]
#     delayed = extern_data["delayed"]
#     spatial_dim = delayed.get_time_dim_tag()
#     targets_len_rf = spatial_dim.dyn_size_ext
#     targets_len = targets_len_rf.raw_tensor
#     out = model(delayed, spatial_dim=spatial_dim)
#     logits: Tensor = out["output"]
#     logits_raw = logits.raw_tensor # (T, B, V)
#     targets_raw = targets.raw_tensor.long() # (B, T)
#     ce = torch.nn.functional.cross_entropy(
#         input = logits_raw.permute(1, 2, 0),
#         target = targets_raw,
#         reduction="none",
#     )
#     seq_mask = get_seq_mask(seq_lens=targets_len, max_seq_len=targets_len.max(), device=logits_raw.device)
#     loss = (ce*seq_mask).sum()
#     ppl = torch.exp(loss/targets_len.sum())
#     rf.get_run_ctx().mark_as_loss(
#         name="log_ppl", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
#     )
#     rf.get_run_ctx().mark_as_loss(
#         name="ppl", loss=ppl, as_error=True,
#     )
