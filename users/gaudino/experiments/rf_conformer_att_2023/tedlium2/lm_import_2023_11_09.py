"""Import Tedlium2 LM from TF checkpoint to RETURNN frontend model with PT backend."""

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

class TrafoLMLayer(rf.Module):
    """
    Represents a conformer block
    """

    def __init__(
        self,
        out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        ff_dim: Dim = NotSpecified,
        ff_activation: Callable[[Tensor], Tensor] = rf.swish,
        dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_norm: Union[rf.BatchNorm, type, Any] = NotSpecified,
        conv_norm_opts: Optional[Dict[str, Any]] = None,
        num_heads: int = 4,
        self_att: Optional[Union[rf.RelPosSelfAttention, rf.Module, type, Any]] = None,
        self_att_opts: Optional[Dict[str, Any]] = None,
        att_dropout: float = 0.1,
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

        if ff_dim is None:
            ff_dim = 4 * out_dim
        self.ffn1 = ConformerPositionwiseFeedForward(
            out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=ff_activation
        )
        self.ffn1_layer_norm = rf.LayerNorm(out_dim)

        self.ffn2 = ConformerPositionwiseFeedForward(
            out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=ff_activation
        )
        self.ffn2_layer_norm = rf.LayerNorm(out_dim)

        if conv_norm is NotSpecified or conv_norm is rf.BatchNorm:
            conv_norm_opts = conv_norm_opts.copy() if conv_norm_opts else {}
            conv_norm_opts.setdefault("use_mask", False)
            conv_norm = rf.BatchNorm(out_dim, **conv_norm_opts)
        elif isinstance(conv_norm, type):
            conv_norm = conv_norm(out_dim, **(conv_norm_opts or {}))
        self.conv_block = ConformerConvBlock(out_dim=out_dim, kernel_size=conv_kernel_size, norm=conv_norm)
        self.conv_layer_norm = rf.LayerNorm(out_dim)

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
                self.self_att = rf.RelPosSelfAttention(**self_att_opts_)
            else:
                self.self_att = self_att(**self_att_opts_)
        else:
            self.self_att = self_att
        self.self_att_layer_norm = rf.LayerNorm(out_dim)

        self.final_layer_norm = rf.LayerNorm(out_dim)

    def __call__(self, inp: Tensor, *, spatial_dim: Dim) -> Tensor:
        """forward"""
        # FFN
        x_ffn1_ln = self.ffn1_layer_norm(inp)
        x_ffn1 = self.ffn1(x_ffn1_ln)
        x_ffn1_out = 0.5 * rf.dropout(x_ffn1, axis=self.out_dim, drop_prob=self.dropout) + inp

        # MHSA
        x_mhsa_ln = self.self_att_layer_norm(x_ffn1_out)
        x_mhsa = self.self_att(x_mhsa_ln, axis=spatial_dim)
        x_mhsa = rf.dropout(x_mhsa, axis=self.out_dim, drop_prob=self.dropout)
        x_mhsa_out = x_mhsa + x_ffn1_out

        # Conv
        x_conv_ln = self.conv_layer_norm(x_mhsa_out)
        x_conv = self.conv_block(x_conv_ln, spatial_dim=spatial_dim)
        x_conv_out = rf.dropout(x_conv, axis=self.out_dim, drop_prob=self.dropout) + x_mhsa_out

        # FFN
        x_ffn2_ln = self.ffn2_layer_norm(x_conv_out)
        x_ffn2 = self.ffn2(x_ffn2_ln)
        x_ffn2_out = 0.5 * rf.dropout(x_ffn2, axis=self.out_dim, drop_prob=self.dropout) + x_conv_out

        # last LN layer
        return self.final_layer_norm(x_ffn2_out)

class Ted2_Trafo_LM_Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        target_dim: Dim,
        *,
        num_layers: int = 30,

        lstm_input_dim: Dim = Dim(name="lstm-input", dimension=128),
        lstm_model_dim: Dim = Dim(name="lstm-model", dimension=2048),
        # enc_att_num_heads: int = 4,
        # enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        # enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        # att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        # att_dropout: float = 0.1,
        # enc_dropout: float = 0.1,
        # enc_att_dropout: float = 0.1,
        # l2: float = 0.0001,
        search_args: Optional[Dict[str, Any]] = None,
    ):
        super(Ted2_Trafo_LM_Model, self).__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers

        self.target_embed_raw = rf.Embedding(in_dim, 128)
        self.target_embed_with_pos = rf.LearnedRelativePositionalEncoding(self.target_embed_raw.out_dim)

        # self.target_embed = rf.Dropout -> called in step
        target_embed_lin_dim = Dim(name="target_embed_lin_dim", dimension=768)
        self.target_embed_lin = rf.Linear(self.target_embed_with_pos.feat_dim, target_embed_lin_dim)

        trafo_layer_opts_ = dict(
            out_dim=out_dim,
            ff_dim=ff_dim,
            ff_activation=ff_activation,
            dropout=0.0,
            conv_kernel_size=conv_kernel_size,
            conv_norm=conv_norm,
            num_heads=num_heads,
            att_dropout=att_dropout,
        )

        encoder_layer = TrafoLMLayer(**trafo_layer_opts_)

        self.layers = rf.Sequential(_copy.deepcopy(encoder_layer) for _ in range(num_layers))




        for i in range(self.num_layers):
            prefix = f"dec_{i}"

            prefix_self_att = prefix + "_self_att"
            prefix_ff = prefix + "_ff"

            # create masked mhsa
            setattr(self, prefix_self_att + "_laynorm", rf.NormLayer)
            setattr(self, prefix_self_att + "_att", rf.AttentionLayer)
            setattr(self, prefix_self_att + "_lin", rf.Linear)
            setattr(self, prefix_self_att + "_drop", rf.Dropout)

            # in loop set _drop + target_embed_lin/dec_i-1_out

            # create ff block
            setattr(self, prefix_self_att + "_laynorm", rf.NormLayer)
            setattr(self, prefix_self_att + "_conv1", rf.Linear)
            setattr(self, prefix_self_att + "_conv2", rf.Linear)
            setattr(self, prefix_self_att + "_drop", rf.Dropout)

            # in loop set _drop + mhsa_out
        self.decoder = rf.NormLayer
        self.output = rf.Linear

        #
        # self.input = rf.Embedding(in_dim, lstm_input_dim)
        # self.input_bias = rf.Parameter((lstm_input_dim,))
        #
        # self.lstm_0 = rf.LSTM(lstm_input_dim, lstm_model_dim, with_bias=True)
        # self.lstm_1 = rf.LSTM(lstm_model_dim, lstm_model_dim, with_bias=True)
        # self.lstm_2 = rf.LSTM(lstm_model_dim, lstm_model_dim, with_bias=True)
        # self.lstm_3 = rf.LSTM(lstm_model_dim, lstm_model_dim, with_bias=True)
        #
        # self.output = rf.Linear(lstm_model_dim, target_dim)

    def loop_step(self, prev_target, prev_state):
        """loop step"""
        lm_state = rf.State()

        target_embed_raw = self.target_embed_raw(prev_target)
        target_embed_with_pos, pos_emb_spatial_dim = self.target_embed_with_pos(target_embed_raw)

        target_embed = rf.dropout(target_embed_with_pos, 0.0)

        target_embed_lin = self.target_embed_lin(target_embed)

        for

        # input = self.input(prev_target)
        # input += self.input_bias
        # # breakpoint()
        # lstm_0, lstm_0_state = self.lstm_0(input, state=prev_state.lstm_0, spatial_dim=single_step_dim)
        # lm_state.lstm_0 = lstm_0_state
        # lstm_1, lstm_1_state = self.lstm_1(lstm_0, state=prev_state.lstm_1, spatial_dim=single_step_dim)
        # lm_state.lstm_1 = lstm_1_state
        # lstm_2, lstm_2_state = self.lstm_2(lstm_1, state=prev_state.lstm_2, spatial_dim=single_step_dim)
        # lm_state.lstm_2 = lstm_2_state
        # lstm_3, lstm_3_state = self.lstm_3(lstm_2, state=prev_state.lstm_3, spatial_dim=single_step_dim)
        # lm_state.lstm_3 = lstm_3_state
        # output = self.output(lstm_3)
        return {"output": output}, lm_state

    def lm_default_initial_state(self, *, batch_dims: Sequence[Dim]
    ) -> rf.State:
        """Default initial state"""
        state = rf.State(
            lstm_0=self.lstm_0.default_initial_state(batch_dims=batch_dims),
            lstm_1=self.lstm_1.default_initial_state(batch_dims=batch_dims),
            lstm_2=self.lstm_2.default_initial_state(batch_dims=batch_dims),
            lstm_3=self.lstm_3.default_initial_state(batch_dims=batch_dims),
        )
        return state


class MakeModel:
    """for import"""

    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        *,
        eos_label: int = 0,
        # num_enc_layers: int = 12,
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim

        self.eos_label = eos_label


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