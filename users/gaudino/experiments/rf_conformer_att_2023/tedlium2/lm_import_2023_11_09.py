"""Import Tedlium2 LM from TF checkpoint to RETURNN frontend model with PT backend."""
from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from typing import Optional, Union, Any, Tuple, List, Dict, Callable
import copy as _copy
from returnn.util.basic import NotSpecified

# from .base import ISeqDownsamplingEncoder


class TrafoLMLayer(rf.Module):
    """
    Represents a conformer block
    """

    def __init__(
        self,
        out_dim: Dim,
        *,
        ff_dim: Dim,
        ff_activation: Callable[[Tensor], Tensor] = rf.relu,
        dropout: float = 0.0,
        conv_kernel_size: int = 32,
        conv_norm: Union[rf.BatchNorm, type, Any] = NotSpecified,
        conv_norm_opts: Optional[Dict[str, Any]] = None,
        num_heads: int = 12,
        self_att: Optional[Union[rf.RelPosSelfAttention, rf.Module, type, Any]] = None,
        self_att_opts: Optional[Dict[str, Any]] = None,
        att_dropout: float = 0.0,
    ):
        """
        :param out_dim: the output feature dimension
        :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
        :param ff_activation: activation function for feed-forward network
        :param dropout: the dropout value for the FF block
        :param conv_kernel_size: the kernel size of depthwise convolution in the conv block
        :param conv_norm: used for the conv block. Batch norm originally
        :param conv_norm_opts: for nn.BatchNorm or other conv_norm type.
          In case of nn.BatchNorm, uses use_mask=False by default.
            use_mask means whether to properly mask the spatial dim in batch norm.
            Most existing implementations don't do this. Except of RETURNN.
            It's faster when you don't do this.
        :param num_heads: the number of attention heads
        :param self_att: the self-attention layer. RelPosSelfAttention originally and default
        :param self_att_opts: options for the self-attention layer, for :class:`nn.RelPosSelfAttention`
        :param att_dropout: attention dropout value
        """
        super().__init__()

        self.dropout = dropout
        self.out_dim = out_dim
        self.ff_dim = ff_dim

        self.ff_layer_norm = rf.LayerNorm(out_dim)
        self.ff_conv1 = rf.Linear(out_dim, ff_dim)
        self.ff_conv2 = rf.Linear(ff_dim, out_dim)
        self.ff_activation = ff_activation

        if self_att is None or isinstance(self_att, type):
            self_att_opts_ = dict(
                in_dim=out_dim,
                proj_dim=out_dim,
                key_dim_total=out_dim,
                value_dim_total=out_dim,
                num_heads=num_heads,
                att_dropout=att_dropout,
            )
            if self_att_opts:
                self_att_opts_.update(self_att_opts)
            if self_att is None:
                self.self_att = rf.SelfAttention(**self_att_opts_)
            else:
                self.self_att = self_att(**self_att_opts_)
        else:
            self.self_att = self_att
        self.self_att_lin = rf.Linear(out_dim, out_dim)
        self.self_att_layer_norm = rf.LayerNorm(out_dim)

    def __call__(self, inp: Tensor, *, spatial_dim: Dim) -> Tensor:
        """forward"""

        # MHSA
        x_mhsa_ln = self.self_att_layer_norm(inp)
        x_mhsa = self.self_att(x_mhsa_ln, axis=spatial_dim)
        x_mhsa = self.self_att_lin(x_mhsa)
        x_mhsa = rf.dropout(x_mhsa, axis=self.out_dim, drop_prob=self.dropout)
        x_mhsa_out = x_mhsa + inp

        # FFN
        x_ff_ln = self.ff_layer_norm(inp)
        x_ff1 = self.ff_conv1(x_ff_ln)
        x_act = self.ff_activation(x_ff1)
        x_ff2 = self.ff_conv2(x_act)
        x_drop = rf.dropout(x_ff2, axis=self.ff_conv2.out_dim, drop_prob=self.dropout)

        res = x_drop + x_mhsa_out

        return res


class Ted2_Trafo_LM_Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        target_dim: Dim,
        *,
        layer_out_dim: Dim = Dim(768, name="trafo-lm-default-out-dim"),
        layer_ff_dim: Dim = Dim(4096, name="trafo-lm-default-ff-dim"),
        embed_dim: Dim = Dim(128, name="trafo-lm-default-embed-dim"),
        num_layers: int = 30,
        search_args: Optional[Dict[str, Any]] = None,
    ):
        super(Ted2_Trafo_LM_Model, self).__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers

        self.target_embed_raw = rf.Embedding(in_dim, embed_dim)
        self.target_embed_with_pos = rf.LearnedRelativePositionalEncoding(
            self.target_embed_raw.out_dim
        )

        # self.target_embed = rf.Dropout -> called in step
        self.target_embed_lin = rf.Linear(
            self.target_embed_with_pos.feat_dim, layer_out_dim.dimension
        )

        trafo_layer_opts_ = dict(
            out_dim=layer_out_dim,
            ff_dim=layer_ff_dim,
            ff_activation=rf.relu,
            dropout=0.0,
            num_heads=12,
            att_dropout=0.0,
        )

        trafo_lm_layer = TrafoLMLayer(**trafo_layer_opts_)

        self.layers = rf.Sequential(
            _copy.deepcopy(trafo_lm_layer) for _ in range(num_layers)
        )

        self.output = rf.Linear(layer_out_dim, target_dim)

    def loop_step(
        self,
        prev_target,
        prev_state,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ):
        """loop step"""
        # lm_state = rf.State()

        target_embed_raw = self.target_embed_raw(prev_target)
        target_embed_with_pos, pos_emb_spatial_dim = self.target_embed_with_pos(
            target_embed_raw
        )

        target_embed = rf.dropout(target_embed_with_pos, 0.0)

        target_embed_lin = self.target_embed_lin(target_embed)

        out_spatial_dim = pos_emb_spatial_dim

        x = self.layers(
            target_embed_lin,
            spatial_dim=out_spatial_dim,
            collected_outputs=collected_outputs,
        )

        output = self.output(x)

        return {"output": output}

    # def lm_default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
    #     """Default initial state"""
    #     state = rf.State(
    #         lstm_0=self.lstm_0.default_initial_state(batch_dims=batch_dims),
    #         lstm_1=self.lstm_1.default_initial_state(batch_dims=batch_dims),
    #         lstm_2=self.lstm_2.default_initial_state(batch_dims=batch_dims),
    #         lstm_3=self.lstm_3.default_initial_state(batch_dims=batch_dims),
    #     )
    #     return state


class MakeModel:
    """for import"""

    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        # eos_label: int = 0,
        # num_enc_layers: int = 12,
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim

    def __call__(self) -> Ted2_Trafo_LM_Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(
            name="target", dimension=self.target_dim, kind=Dim.Types.Feature
        )
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(in_dim, target_dim)

    @classmethod
    def make_model(
        cls,
        in_dim: Dim,
        target_dim: Dim,
        # *,
        # search_args: Optional[Dict[str, Any]],
        # num_enc_layers: int = 12,
    ) -> Ted2_Trafo_LM_Model:
        """make"""
        return Ted2_Trafo_LM_Model(
            in_dim,
            # num_enc_layers=num_enc_layers,
            target_dim=target_dim,
        )
