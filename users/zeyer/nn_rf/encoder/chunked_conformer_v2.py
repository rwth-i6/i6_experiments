"""
V2:
- Allow dynamic chunk sizes, strides, left/right contexts.
- Changed configuration options.
- Support both online and offline mode.
- Support different kinds of overlap handling:
  concat, average, ...
- Fix masking within chunks.

Earlier configuration options:

- input_chunk_size_dim: on input (10ms) level. chunk size (including right context). default: 210
- chunk_stride: on input (10ms) level. chunk stride. default: 120
- end_chunk_size_dim: on encoder (60ms) level. chunk size excluding right context.
  The right context is cut off for AED cross-att and when adding history (left) context.
  Because this is also used for the history context (concat prev chunks),
  it should match the chunk_stride (which is on input level though, before downsampling).
  default: 20
- chunk_history: num prev chunks to add for history context. default: 2

New configuration options:

TODO do this:

- input_chunk_stride: on input (10ms) level. chunk stride. default: 120
- enc_right_chunk_size: on encoder (60ms) level. chunk size right context size.
  The right context is cut off for AED cross-att and when adding history (left) context.
  default: 20
- enc_left_chunk_size: on encoder (60ms) level. chunk size left context size.
  This is used for the history context (concat prev chunks).
  default: 40

TODO make it support both offline and online mode
TODO diff kinds of overlap handling (for right ctx): concat, average, ...
   (currently it just throws it away)
TODO overlap handling also for mem ctx?
TODO fix masking within chunks. set right window dim

TODO conv mem ctx optional
TODO conv mem ctx only as much as we really need
TODO abs pos enc instead (optional, configurable)
TODO diff impl for mem ctx: separate matmuls for each prev chunk

TODO self-conditioning
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union
import copy as _copy
from dataclasses import dataclass

from returnn.util.basic import NotSpecified
from returnn.util.math import ceil_div
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.encoder.base import ISeqDownsamplingEncoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
    make_ff,
    make_norm,
)


@dataclass
class _BatchChunkingSettings:
    input_chunk_size_dim: Dim
    chunk_stride: int
    chunk_history: int
    end_chunk_size_dim: Dim
    chunked_time_dim: Dim


class ChunkedConformerConvBlockV2(rf.Module):
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
    ):
        """
        :param out_dim: output feature dimension
        :param kernel_size: kernel size of depthwise convolution
        :param norm: Batch norm originally
        """
        super().__init__()
        self.out_dim = out_dim

        self.positionwise_conv1 = rf.Linear(out_dim, 2 * out_dim)
        self.depthwise_conv = rf.Conv1d(
            out_dim, out_dim, filter_size=kernel_size, groups=out_dim.dimension, padding="same"
        )
        self.positionwise_conv2 = rf.Linear(out_dim, out_dim)
        self.norm = norm

    def __call__(self, inp: Tensor, *, spatial_dim: Dim, chunking: Optional[_BatchChunkingSettings]) -> Tensor:
        """forward"""
        x_conv1 = self.positionwise_conv1(inp)
        x_act, _ = rf.gating(x_conv1)
        if chunking:
            needed_left_ctx = self.depthwise_conv.filter_size[0].dimension // 2
            needed_mem_size = (
                needed_left_ctx + chunking.end_chunk_size_dim.dimension - 1
            ) // chunking.end_chunk_size_dim.dimension
            mem_size = min(chunking.chunk_history, needed_mem_size)
        else:
            mem_size = None
        if chunking and mem_size:
            x_act, ext_spatial_dim = _mem_chunks(
                x_act,
                spatial_dim=spatial_dim,
                chunked_time_dim=chunking.chunked_time_dim,
                mem_size=mem_size,
                end_chunk_size_dim=chunking.end_chunk_size_dim,
            )
        else:
            ext_spatial_dim = spatial_dim
        x_depthwise_conv, _ = self.depthwise_conv(x_act, in_spatial_dim=ext_spatial_dim)
        if chunking and mem_size:
            x_depthwise_conv, _ = rf.slice(
                x_depthwise_conv,
                axis=ext_spatial_dim,
                start=mem_size * chunking.end_chunk_size_dim.get_dim_value_tensor(),
                out_dim=spatial_dim,
            )
        x_normed = self.norm(x_depthwise_conv)
        x_swish = rf.swish(x_normed)
        x_conv2 = self.positionwise_conv2(x_swish)
        return x_conv2


class ChunkedConformerEncoderLayerV2(rf.Module):
    """
    Represents a conformer block
    """

    def __init__(
        self,
        out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        ff: Union[type, Dict[str, Any], rf.Module] = NotSpecified,
        ff_dim: Dim = NotSpecified,
        ff_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = NotSpecified,
        dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_norm: Union[rf.BatchNorm, type, Any] = NotSpecified,
        conv_norm_opts: Optional[Dict[str, Any]] = None,
        num_heads: int = 4,
        self_att: Optional[Union[rf.RelPosSelfAttention, rf.Module, type, Any]] = None,
        self_att_opts: Optional[Dict[str, Any]] = None,
        att_dropout: float = 0.1,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
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

        self.ffn1 = make_ff(ff=ff, out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, ff_activation=ff_activation)
        self.ffn1_layer_norm = make_norm(norm, out_dim)

        self.ffn2 = make_ff(ff=ff, out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, ff_activation=ff_activation)
        self.ffn2_layer_norm = make_norm(norm, out_dim)

        if conv_norm is NotSpecified or conv_norm is rf.BatchNorm:
            conv_norm_opts = conv_norm_opts.copy() if conv_norm_opts else {}
            conv_norm_opts.setdefault("use_mask", False)
            conv_norm = rf.BatchNorm(out_dim, **conv_norm_opts)
        elif isinstance(conv_norm, type):
            conv_norm = conv_norm(out_dim, **(conv_norm_opts or {}))
        self.conv_block = ChunkedConformerConvBlockV2(
            out_dim=out_dim,
            kernel_size=conv_kernel_size,
            norm=conv_norm,
        )
        self.conv_layer_norm = make_norm(norm, out_dim)

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
                self.self_att = ChunkedRelPosSelfAttentionV2(**self_att_opts_)
            else:
                self.self_att = self_att(**self_att_opts_)
        else:
            self.self_att = self_att
        self.self_att_layer_norm = make_norm(norm, out_dim)

        self.final_layer_norm = make_norm(norm, out_dim)

    def __call__(self, inp: Tensor, *, spatial_dim: Dim, chunking: Optional[_BatchChunkingSettings]) -> Tensor:
        """forward"""
        # FFN
        x_ffn1_ln = self.ffn1_layer_norm(inp)
        x_ffn1 = self.ffn1(x_ffn1_ln)
        x_ffn1_out = 0.5 * rf.dropout(x_ffn1, self.dropout, axis=self.dropout_broadcast and self.out_dim) + inp

        # MHSA
        x_mhsa_ln = self.self_att_layer_norm(x_ffn1_out)
        x_mhsa = self.self_att(x_mhsa_ln, axis=spatial_dim, chunking=chunking)
        x_mhsa = rf.dropout(x_mhsa, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x_mhsa_out = x_mhsa + x_ffn1_out

        # Conv
        x_conv_ln = self.conv_layer_norm(x_mhsa_out)
        x_conv = self.conv_block(x_conv_ln, spatial_dim=spatial_dim, chunking=chunking)
        x_conv_out = rf.dropout(x_conv, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_mhsa_out

        # FFN
        x_ffn2_ln = self.ffn2_layer_norm(x_conv_out)
        x_ffn2 = self.ffn2(x_ffn2_ln)
        x_ffn2_out = 0.5 * rf.dropout(x_ffn2, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_conv_out

        # last LN layer
        return self.final_layer_norm(x_ffn2_out)


class ChunkedConformerEncoderV2(rf.Module):
    """
    Represents Conformer encoder architecture
    """

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Union[int, Dim] = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        num_layers: int,
        input_layer: Optional[Union[ConformerConvSubsample, ISeqDownsamplingEncoder, rf.Module, Any]],
        input_dropout: float = 0.1,
        ff_dim: Dim = NotSpecified,
        ff_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = NotSpecified,
        dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_norm: Union[rf.BatchNorm, type, Any] = NotSpecified,
        num_heads: int = 4,
        att_dropout: float = 0.1,
        encoder_layer: Optional[Union[ConformerEncoderLayer, rf.Module, type, Dict[str, Any], Any]] = None,
        encoder_layer_opts: Optional[Dict[str, Any]] = None,
        input_chunk_size_dim: Union[int, Dim],
        chunk_stride: int,
        chunk_history: int,
        end_chunk_size_dim: Union[int, Dim],
        version: int = 1,
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
        :param input_chunk_size_dim:
        :param chunk_stride:
        :param chunk_history:
        :param end_chunk_size_dim:
        :param version: version of chunked conformer
        """
        super().__init__()

        if isinstance(out_dim, int):
            out_dim = Dim(out_dim, name="model")
        if isinstance(input_chunk_size_dim, int):
            input_chunk_size_dim = Dim(input_chunk_size_dim, name="input_chunk_size")
        if isinstance(end_chunk_size_dim, int):
            end_chunk_size_dim = Dim(end_chunk_size_dim, name="end_chunk_size")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        self.input_chunk_size_dim = input_chunk_size_dim
        self.chunk_stride = chunk_stride
        self.chunk_history = chunk_history
        self.end_chunk_size_dim = end_chunk_size_dim
        self.version = version
        assert version <= 2

        if isinstance(input_layer, dict):
            input_layer = rf.build_from_dict(input_layer, in_dim)
            input_layer: ConformerConvSubsample  # maybe not true, but assume for some attribs

        self.input_layer = input_layer
        self.input_projection = rf.Linear(
            self.input_layer.out_dim if self.input_layer else self.in_dim, self.out_dim, with_bias=False
        )
        self.input_dropout = input_dropout

        if not encoder_layer or isinstance(encoder_layer, (dict, type)):
            encoder_layer_opts_ = dict(
                out_dim=out_dim,
                ff_dim=ff_dim,
                ff_activation=ff_activation,
                dropout=dropout,
                conv_kernel_size=conv_kernel_size,
                conv_norm=conv_norm,
                num_heads=num_heads,
                att_dropout=att_dropout,
            )
            encoder_layer_opts_ = {k: v for (k, v) in encoder_layer_opts_.items() if v is not NotSpecified}
            if encoder_layer_opts:
                encoder_layer_opts_.update(encoder_layer_opts)
            if not encoder_layer:
                encoder_layer = ChunkedConformerEncoderLayerV2(**encoder_layer_opts_)
            elif isinstance(encoder_layer, type):
                encoder_layer = encoder_layer(**encoder_layer_opts_)
            elif isinstance(encoder_layer, dict):
                # Note: Reuse all the encoder_layer_opts_.
                # If this does not make sense for the specific encoder_layer class here,
                # we would suggest to use a different ConformerEncoder class.
                # (The alternative, to not reuse encoder_layer_opts_ here,
                #  would probably be more confusing, as those options are all ignored then.
                #  It's also not clear what args to pass then and what not.)
                # (Maybe we should do a ConformerEncoderV2 if this is confusing here...)
                encoder_layer_opts_ = {k: v for (k, v) in encoder_layer_opts_.items() if k not in encoder_layer}
                encoder_layer = rf.build_from_dict(encoder_layer, **encoder_layer_opts_)
            else:
                raise TypeError(f"unexpected encoder_layer {encoder_layer!r}")
        else:
            if not callable(encoder_layer):
                raise TypeError(f"{self}: invalid non-callable encoder_layer {encoder_layer!r}")

        self.layers = rf.Sequential(_copy.deepcopy(encoder_layer) for _ in range(num_layers))

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dim]:
        """forward"""
        if rf.get_run_ctx().step % 2 == 0 and rf.get_run_ctx().train_flag:
            chunking = None
            spatial_dim = in_spatial_dim
        else:
            # Chunk
            input_chunk_size_dim = self.input_chunk_size_dim
            chunk_stride = self.chunk_stride
            chunk_history = self.chunk_history
            end_chunk_size_dim = self.end_chunk_size_dim

            if self.version >= 2:
                # First potentially reduce chunk sizes, history, if the input is not long enough.
                max_input_chunk_size_dim = Dim(int(in_spatial_dim.get_dim_value()), name="max_input_chunk_size")
                max_chunk_size_dim = (
                    self.input_layer.get_out_spatial_dim(max_input_chunk_size_dim)
                    if self.input_layer
                    else max_input_chunk_size_dim
                )
                if end_chunk_size_dim.dimension > max_chunk_size_dim.dimension:
                    end_chunk_size_dim = max_chunk_size_dim
                if input_chunk_size_dim.dimension > max_input_chunk_size_dim.dimension:
                    input_chunk_size_dim = max_input_chunk_size_dim
                    chunk_history = 0
                elif end_chunk_size_dim.dimension * chunk_history > max_chunk_size_dim.dimension - 1:
                    chunk_history = ceil_div(max_chunk_size_dim.dimension - 1, end_chunk_size_dim.dimension)

            source, chunked_time_dim = rf.window(
                source,
                spatial_dim=in_spatial_dim,
                window_dim=input_chunk_size_dim,
                window_left=0,
                stride=chunk_stride,
                pad_value=0.0,
            )
            spatial_dim = input_chunk_size_dim

            chunking = _BatchChunkingSettings(
                input_chunk_size_dim=input_chunk_size_dim,
                chunk_stride=chunk_stride,
                chunk_history=chunk_history,
                end_chunk_size_dim=end_chunk_size_dim,
                chunked_time_dim=chunked_time_dim,
            )

        if self.input_layer:
            x_subsample, spatial_dim = self.input_layer(source, in_spatial_dim=spatial_dim)
        else:
            x_subsample = source
        x_linear = self.input_projection(x_subsample)

        x = rf.dropout(x_linear, self.input_dropout, axis=self.dropout_broadcast and self.input_projection.out_dim)
        x = self.layers(
            x,
            spatial_dim=spatial_dim,
            chunking=chunking,
            collected_outputs=collected_outputs,
        )

        if chunking:
            # Unchunk
            x, _ = rf.slice(x, axis=spatial_dim, size=chunking.end_chunk_size_dim)
            x, out_spatial_dim = rf.merge_dims(x, dims=(chunking.chunked_time_dim, chunking.end_chunk_size_dim))
        else:
            out_spatial_dim = spatial_dim

        if collected_outputs:
            for k, v in list(collected_outputs.items()):
                if chunking:
                    v, _ = rf.slice(v, axis=spatial_dim, size=chunking.end_chunk_size_dim)
                    v, _ = rf.merge_dims(
                        v, dims=(chunking.chunked_time_dim, chunking.end_chunk_size_dim), out_dim=out_spatial_dim
                    )
                collected_outputs[k] = v

        return x, out_spatial_dim


class ChunkedRelPosSelfAttentionV2(rf.RelPosSelfAttention):
    def __call__(self, source: Tensor, *, axis: Dim, chunking: Optional[_BatchChunkingSettings], **_kwargs) -> Tensor:
        """forward"""
        q, k, v = self.forward_qkv(source)
        hist_dim = Dim(None, name=f"{axis.description}:kv")
        k, _ = rf.replace_dim(k, in_dim=axis, out_dim=hist_dim)
        v, _ = rf.replace_dim(v, in_dim=axis, out_dim=hist_dim)
        q_with_bias_u = (q + self.pos_bias_u) if self.pos_bias_u is not None else q  # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v) if self.pos_bias_v is not None else q  # (batch, head, time1, d_k)

        if chunking:
            # NOTE: This changed from the earlier RF/TF implementation.
            query_offset = chunking.chunk_history * (
                chunking.end_chunk_size_dim.dimension
                if chunking.end_chunk_size_dim.is_static()
                else chunking.end_chunk_size_dim.get_size_tensor(device=source.device)
            )
            hist_dim_ = chunking.chunk_history * chunking.end_chunk_size_dim + hist_dim
        else:
            query_offset = None
            hist_dim_ = hist_dim

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
                device=source.device,
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
        # (batch, head, time1, time2')
        matrix_ac = rf.matmul(q_with_bias_u, k, reduce=self.key_dim_per_head)

        if chunking:
            matrix_ac, _ = _mem_chunks(
                matrix_ac,
                spatial_dim=hist_dim,
                chunked_time_dim=chunking.chunked_time_dim,
                mem_size=chunking.chunk_history,
                end_chunk_size_dim=chunking.end_chunk_size_dim,
                out_spatial_dim=hist_dim_,
            )

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = rf.matmul(q_with_bias_v, pos_emb, reduce=self.key_dim_per_head)
        matrix_bd = self._rel_shift(matrix_bd, axis, pos_emb_spatial_dim, hist_dim_)

        scores = matrix_ac + matrix_bd  # (batch, head, time1, time2)
        scores *= self.key_dim_per_head.dimension**-0.5
        att_weights = rf.softmax(scores, axis=hist_dim_)
        att_weights = rf.dropout(att_weights, self.att_dropout, axis=self.att_dropout_broadcast and hist_dim_)

        if chunking:
            # Undo the _mem_chunks for att_weights to save memory
            # and to be able to split the history dimension.
            att_weights_parts = rf.split(
                att_weights,
                axis=hist_dim_,
                out_dims=chunking.chunk_history * [chunking.end_chunk_size_dim] + [hist_dim],
            )
            att_weights_parts = [
                rf.pad(
                    part,
                    axes=[chunking.end_chunk_size_dim],
                    padding=[(0, hist_dim - chunking.end_chunk_size_dim)],
                    value=0.0,
                    out_dims=[hist_dim],
                )[0]
                for part in att_weights_parts[:-1]
            ] + [att_weights_parts[-1]]
            att_weights, history_parts_dim = rf.stack(att_weights_parts)
            # (batch, head, time1, time2', history_parts)

            # Masking not needed because softmax should already have masked,
            # so we have 0.0 att weights for padded frames.
            att = rf.matmul(att_weights, v, reduce=(hist_dim, history_parts_dim), use_mask=False)
            # (batch, head, time1, feat)

        else:
            # Masking not needed because softmax should already have masked,
            # so we have 0.0 att weights for padded frames.
            att = rf.matmul(att_weights, v, reduce=hist_dim, use_mask=False)

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
    Concat the prev chunks to the current chunk, i.e. add history / memory.

    :param source: (batch..., chunked_time, spatial_dim=chunk_size, feat)
    :param spatial_dim: chunk size / window size
    :param chunked_time_dim: the chunks
    :param mem_size: how many previous chunks to concat
    :param end_chunk_size_dim: ...?
    :param out_spatial_dim: if given, use this as output spatial dim
    :return: concatenated prev chunks, concatenated spatial dim
    """
    concats = []
    source_sliced, _ = rf.slice(source, axis=spatial_dim, size=end_chunk_size_dim)
    for shift_amount in range(mem_size, 0, -1):
        shifted = rf.shift_right(source_sliced, axis=chunked_time_dim, amount=shift_amount, pad_value=0.0)
        concats.append((shifted, end_chunk_size_dim))
    concats.append((source, spatial_dim))
    return rf.concat(*concats, out_dim=out_spatial_dim)
