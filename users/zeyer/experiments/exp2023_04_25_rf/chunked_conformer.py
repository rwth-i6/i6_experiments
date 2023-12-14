"""
Chunked Conformer encoder - adapted from original RF Conformer code

Moh base example config:
chunked_att_chunk-35_step-20_linDecay300_0.0002_decayPt0.3333333333333333_bs15000_accum2_winLeft0_endSliceStart0_endSlice20_memVariant1_memSize2_convCache2_useCachedKV_memSlice0-20_L0_C20_R15
- chunk-35
- step-20
- winLeft0
- endSliceStart0 -> end_slice_start = 0
- endSlice20 -> end_slice_size = 20
- memVariant1
- memSize2 -> ConformerMemoryVariantOpts mem_size = 2
- convCache2 -> ConformerMemoryVariantOpts conv_cache_size = 2
- useCachedKV -> use_cached_prev_kv
- memSlice0-20 -> mem_slice_start, mem_slice_size
- L0
- C20
- R15

See code i6_experiments.users.zeineldeen.models.asr.encoder.conformer_encoder.ConformerEncoder._self_att_v2.
"""


from __future__ import annotations
from typing import Optional, Union, Any, Tuple, Dict, Callable
import copy as _copy
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.util.basic import NotSpecified
from returnn.frontend.encoder.base import ISeqDownsamplingEncoder
from returnn.frontend.encoder.conformer import ConformerPositionwiseFeedForward, ConformerConvSubsample


class ChunkedConformerConvBlock(rf.Module):
    """
    Conformer convolution block
        FF -> GLU -> depthwise conv -> BN -> Swish -> FF
    """

    def __init__(
        self,
        out_dim: Dim,
        *,
        kernel_size: int,
        norm: Union[rf.BatchNorm, Any],
        chunk_history: int,
        end_chunk_size_dim: Dim,
    ):
        """
        :param out_dim: output feature dimension
        :param kernel_size: kernel size of depthwise convolution
        :param norm: Batch norm originally
        :param chunk_history:
        :param end_chunk_size_dim:
        """
        super().__init__()
        self.out_dim = out_dim

        self.positionwise_conv1 = rf.Linear(out_dim, 2 * out_dim)
        self.depthwise_conv = rf.Conv1d(
            out_dim, out_dim, filter_size=kernel_size, groups=out_dim.dimension, padding="same"
        )
        self.positionwise_conv2 = rf.Linear(out_dim, out_dim)
        self.norm = norm

        self.chunk_history = chunk_history
        self.end_chunk_size_dim = end_chunk_size_dim

    def __call__(self, inp: Tensor, *, spatial_dim: Dim, chunked_time_dim: Dim) -> Tensor:
        """forward"""
        x_conv1 = self.positionwise_conv1(inp)
        x_act, _ = rf.gating(x_conv1)
        x_act, ext_spatial_dim = _mem_chunks(
            x_act,
            spatial_dim=spatial_dim,
            chunked_time_dim=chunked_time_dim,
            mem_size=self.chunk_history,
            end_chunk_size_dim=self.end_chunk_size_dim,
        )
        x_depthwise_conv, _ = self.depthwise_conv(x_act, in_spatial_dim=ext_spatial_dim)
        x_depthwise_conv, _ = rf.slice(
            x_depthwise_conv,
            axis=ext_spatial_dim,
            start=self.chunk_history * self.end_chunk_size_dim.get_dim_value_tensor(),
            out_dim=spatial_dim,
        )
        x_normed = self.norm(x_depthwise_conv)
        x_swish = rf.swish(x_normed)
        x_conv2 = self.positionwise_conv2(x_swish)
        return x_conv2


class ChunkedConformerEncoderLayer(rf.Module):
    """
    Represents a conformer block
    """

    def __init__(
        self,
        out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        chunk_history: int,
        end_chunk_size_dim: Dim,
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
        :param chunk_history:
        :param end_chunk_size_dim:
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
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.out_dim = out_dim
        self.chunk_history = chunk_history
        self.end_chunk_size_dim = end_chunk_size_dim

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
        self.conv_block = ChunkedConformerConvBlock(
            out_dim=out_dim,
            kernel_size=conv_kernel_size,
            norm=conv_norm,
            chunk_history=chunk_history,
            end_chunk_size_dim=end_chunk_size_dim,
        )
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
                self.self_att = ChunkedRelPosSelfAttention(
                    chunk_history=chunk_history, end_chunk_size_dim=end_chunk_size_dim, **self_att_opts_
                )
            else:
                self.self_att = self_att(**self_att_opts_)
        else:
            self.self_att = self_att
        self.self_att_layer_norm = rf.LayerNorm(out_dim)

        self.final_layer_norm = rf.LayerNorm(out_dim)

    def __call__(self, inp: Tensor, *, spatial_dim: Dim, chunked_time_dim: Dim) -> Tensor:
        """forward"""
        # FFN
        x_ffn1_ln = self.ffn1_layer_norm(inp)
        x_ffn1 = self.ffn1(x_ffn1_ln)
        x_ffn1_out = 0.5 * rf.dropout(x_ffn1, self.dropout, axis=self.dropout_broadcast and self.out_dim) + inp

        # MHSA
        x_mhsa_ln = self.self_att_layer_norm(x_ffn1_out)
        x_mhsa = self.self_att(x_mhsa_ln, axis=spatial_dim, chunked_time_dim=chunked_time_dim)
        x_mhsa = rf.dropout(x_mhsa, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x_mhsa_out = x_mhsa + x_ffn1_out

        # Conv
        x_conv_ln = self.conv_layer_norm(x_mhsa_out)
        x_conv = self.conv_block(x_conv_ln, spatial_dim=spatial_dim, chunked_time_dim=chunked_time_dim)
        x_conv_out = rf.dropout(x_conv, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_mhsa_out

        # FFN
        x_ffn2_ln = self.ffn2_layer_norm(x_conv_out)
        x_ffn2 = self.ffn2(x_ffn2_ln)
        x_ffn2_out = 0.5 * rf.dropout(x_ffn2, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_conv_out

        # last LN layer
        return self.final_layer_norm(x_ffn2_out)


class ChunkedConformerEncoder(rf.Module):
    """
    Represents Conformer encoder architecture
    """

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        num_layers: int,
        input_layer: Union[ConformerConvSubsample, ISeqDownsamplingEncoder, rf.Module, Any],
        input_dropout: float = 0.1,
        ff_dim: Dim = NotSpecified,
        ff_activation: Callable[[Tensor], Tensor] = rf.swish,
        dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_norm: Union[rf.BatchNorm, type, Any] = NotSpecified,
        num_heads: int = 4,
        att_dropout: float = 0.1,
        encoder_layer: Optional[Union[ChunkedConformerEncoderLayer, rf.Module, type, Any]] = None,
        encoder_layer_opts: Optional[Dict[str, Any]] = None,
        chunk_history: int,
        end_chunk_size_dim: Dim,
    ):
        """
        :param out_dim: the output feature dimension
        :param num_layers: the number of encoder layers
        :param input_layer: input/frontend/prenet with potential subsampling.
            (x, in_spatial_dim) -> (y, out_spatial_dim)
        :param input_dropout: applied after input_projection(input_layer(x))
        :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
        :param ff_activation: activation function for feed-forward network
        :param dropout: the dropout value for the FF block
        :param conv_kernel_size: the kernel size of depthwise convolution in the conv block
        :param conv_norm: used for the conv block. Batch norm originally
        :param num_heads: the number of attention heads
        :param att_dropout: attention dropout value
        :param encoder_layer: an instance of :class:`ConformerEncoderLayer` or similar
        :param encoder_layer_opts: options for the encoder layer
        :param chunk_history:
        :param end_chunk_size_dim:
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        # TODO once we figured out good defaults, we would create ConformerConvSubsample here when not given
        self.input_layer = input_layer
        self.input_projection = rf.Linear(
            self.input_layer.out_dim if self.input_layer else self.in_dim, self.out_dim, with_bias=False
        )
        self.input_dropout = input_dropout

        if not encoder_layer or isinstance(encoder_layer, type):
            encoder_layer_opts_ = dict(
                out_dim=out_dim,
                ff_dim=ff_dim,
                ff_activation=ff_activation,
                dropout=dropout,
                conv_kernel_size=conv_kernel_size,
                conv_norm=conv_norm,
                num_heads=num_heads,
                att_dropout=att_dropout,
                chunk_history=chunk_history,
                end_chunk_size_dim=end_chunk_size_dim,
            )
            if encoder_layer_opts:
                encoder_layer_opts_.update(encoder_layer_opts)
            if not encoder_layer:
                encoder_layer = ChunkedConformerEncoderLayer(**encoder_layer_opts_)
            elif isinstance(encoder_layer, type):
                encoder_layer = encoder_layer(**encoder_layer_opts_)
            else:
                raise TypeError(f"unexpected encoder_layer {encoder_layer!r}")

        self.layers = rf.Sequential(_copy.deepcopy(encoder_layer) for _ in range(num_layers))

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        chunked_time_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dim]:
        """forward"""
        if self.input_layer:
            x_subsample, out_spatial_dim = self.input_layer(source, in_spatial_dim=in_spatial_dim)
        else:
            x_subsample, out_spatial_dim = source, in_spatial_dim
        x_linear = self.input_projection(x_subsample)
        x = rf.dropout(x_linear, self.input_dropout, axis=self.dropout_broadcast and self.input_projection.out_dim)
        x = self.layers(
            x, spatial_dim=out_spatial_dim, chunked_time_dim=chunked_time_dim, collected_outputs=collected_outputs
        )
        return x, out_spatial_dim


class ChunkedRelPosSelfAttention(rf.RelPosSelfAttention):
    def __init__(self, *, chunk_history: int, end_chunk_size_dim: Dim, **kwargs):
        super().__init__(**kwargs)
        self.chunk_history = chunk_history
        self.end_chunk_size_dim = end_chunk_size_dim

    def __call__(self, source: Tensor, *, axis: Dim, chunked_time_dim: Dim, **_kwargs) -> Tensor:
        """forward"""
        q, k, v = self.forward_qkv(source)
        hist_dim = Dim(None, name=f"{axis.description}:kv")
        k, _ = rf.replace_dim(k, in_dim=axis, out_dim=hist_dim)
        v, _ = rf.replace_dim(v, in_dim=axis, out_dim=hist_dim)
        k, hist_dim_ = _mem_chunks(
            k,
            spatial_dim=hist_dim,
            chunked_time_dim=chunked_time_dim,
            mem_size=self.chunk_history,
            end_chunk_size_dim=self.end_chunk_size_dim,
        )
        v, _ = _mem_chunks(
            v,
            spatial_dim=hist_dim,
            chunked_time_dim=chunked_time_dim,
            mem_size=self.chunk_history,
            end_chunk_size_dim=self.end_chunk_size_dim,
            out_spatial_dim=hist_dim_,
        )
        q_with_bias_u = (q + self.pos_bias_u) if self.pos_bias_u is not None else q  # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v) if self.pos_bias_v is not None else q  # (batch, head, time1, d_k)

        # TODO actually wrong...! but wrong just like in orig cfg? should be end_chunk_size_dim?
        query_offset = self.chunk_history * axis.dimension

        if self.learned_pos_emb is not None:
            pos_emb, pos_emb_spatial_dim = self.learned_pos_emb(
                query_spatial_dim=axis, key_value_spatial_dim=hist_dim_, query_offset=query_offset
            )
        else:
            pos_emb, pos_emb_spatial_dim = rf.relative_positional_encoding(
                query_spatial_dim=axis,
                key_value_spatial_dim=hist_dim_,
                feat_dim=self.pos_emb_feat_dim,
                query_offset=query_offset,
            )
        if self.pos_emb_dropout:
            pos_emb = rf.dropout(pos_emb, self.pos_emb_dropout)
        if self.linear_pos is not None:
            pos_emb = self.linear_pos(pos_emb)
        if self.separate_pos_emb_per_head:
            pos_emb = rf.split_dims(pos_emb, axis=self.key_dim_total, dims=(self.num_heads, self.key_dim_per_head))
        # pos_emb: (head, 2*time1-1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = rf.matmul(q_with_bias_u, k, reduce=self.key_dim_per_head)

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = rf.matmul(q_with_bias_v, pos_emb, reduce=self.key_dim_per_head)
        matrix_bd = self._rel_shift(matrix_bd, axis, pos_emb_spatial_dim, hist_dim_)

        scores = matrix_ac + matrix_bd  # (batch, head, time1, time2)
        scores *= self.key_dim_per_head.dimension**-0.5
        att_weights = rf.softmax(scores, axis=hist_dim_)
        att_weights = rf.dropout(att_weights, self.att_dropout, axis=self.att_dropout_broadcast and hist_dim_)
        # Masking not needed because softmax should already have masked,
        # so we have 0.0 att weights for padded frames.
        att = rf.matmul(att_weights, v, reduce=hist_dim_, use_mask=False)
        output, _ = rf.merge_dims(att, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
        if self.proj:
            output = self.proj(output)
        return output


def _mem_chunks(
    source: rf.Tensor,
    *,
    spatial_dim: Dim,
    chunked_time_dim: Dim,
    mem_size: int,
    end_chunk_size_dim: Dim,
    out_spatial_dim: Optional[Dim] = None,
) -> Tuple[Tensor, Dim]:
    """
    :return: concatenated prev chunks, concatenated spatial dim
    """
    concats = []
    source_sliced, _ = rf.slice(source, axis=spatial_dim, size=end_chunk_size_dim)
    for shift_amount in range(mem_size, 0, -1):
        shifted = rf.shift_right(source_sliced, axis=chunked_time_dim, amount=shift_amount, pad_value=0.0)
        concats.append((shifted, end_chunk_size_dim))
    concats.append((source, spatial_dim))
    return rf.concat(*concats, out_dim=out_spatial_dim)
