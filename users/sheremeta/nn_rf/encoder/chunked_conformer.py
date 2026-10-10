import returnn.frontend as rf
from returnn.tensor import Dim as ReturnnDim
from returnn.tensor import Tensor as ReturnnTensor


_kernel_attention_class = None
_kernel_attention_class_v2 = None
_kept_rows_conv_block_class = None


def conv_history_for_kernel(kernel_size: int, center_size: int, chunk_history: int) -> int:
    """
    How many previous chunks the depthwise conv of the upstream conv block needs.

    The block builds its history with its ``chunk_history`` and keeps the outputs of the current chunk,
    which see ``(kernel_size - 1) // 2`` frames to the left, so that many frames of history give the
    same outputs as the attention's full history at a fraction of the conv work.

    :param kernel_size: depthwise conv kernel size
    :param center_size: frames a previous chunk contributes
    :param chunk_history: the attention's history in chunks, the upper bound
    :return: the history in chunks for the conv block
    """
    left_context = (kernel_size - 1) // 2
    return min(chunk_history, -(-left_context // center_size))


def kernel_attention_class():
    """
    Builds the chunked rel-pos self-attention of the i6_experiments chunked Conformer on the rf op, once.

    :return: the attention class, a subclass of ChunkedRelPosSelfAttention
    """
    global _kernel_attention_class
    if _kernel_attention_class is not None:
        return _kernel_attention_class
    from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v1 import ChunkedRelPosSelfAttention

    class ChunkedRelPosSelfAttentionKernel(ChunkedRelPosSelfAttention):
        """Runs the chunked attention through the rf op, which takes the Triton kernel where it applies."""

        def __call__(self, source: ReturnnTensor, *, axis: ReturnnDim, chunked_time_dim: ReturnnDim, **kwargs):
            del kwargs
            return chunked_attention(
                self,
                source,
                axis=axis,
                chunked_time_dim=chunked_time_dim,
                chunk_history=self.chunk_history,
                end_chunk_size_dim=self.end_chunk_size_dim,
            )

    _kernel_attention_class = ChunkedRelPosSelfAttentionKernel
    return _kernel_attention_class


def kernel_attention_class_v2():
    """
    Builds the V2 chunked rel-pos self-attention of the i6_experiments chunked Conformer on the rf op, once.

    V2 hands the chunk geometry in at call time instead of holding it on the module, and passes no
    geometry at all for an offline step, which falls through to the upstream math.

    :return: the attention class, a subclass of ChunkedRelPosSelfAttentionV2
    """
    global _kernel_attention_class_v2
    if _kernel_attention_class_v2 is not None:
        return _kernel_attention_class_v2
    from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import ChunkedRelPosSelfAttentionV2

    class ChunkedRelPosSelfAttentionV2Kernel(ChunkedRelPosSelfAttentionV2):
        """Runs the chunked attention through the rf op, which takes the Triton kernel where it applies."""

        def __call__(self, source: ReturnnTensor, *, axis: ReturnnDim, chunking, **kwargs):
            if chunking is None:
                return super().__call__(source, axis=axis, chunking=chunking, **kwargs)
            return chunked_attention(
                self,
                source,
                axis=axis,
                chunked_time_dim=chunking.chunked_time_dim,
                chunk_history=chunking.chunk_history,
                end_chunk_size_dim=chunking.end_chunk_size_dim,
            )

    _kernel_attention_class_v2 = ChunkedRelPosSelfAttentionV2Kernel
    return _kernel_attention_class_v2


def kept_rows_conv_block_class():
    """
    Builds the conv block of the i6_experiments chunked Conformer that convolves only the rows it keeps, once.

    The upstream block convolves the history window of every chunk with same padding and keeps the chunk's own rows.
    Those rows are the conv of the window with the left padding shortened by the history and the right padding of
    same, so this block computes them alone, with the same parameters.

    :return: the conv block class, a subclass of ChunkedConformerConvBlock
    """
    global _kept_rows_conv_block_class
    if _kept_rows_conv_block_class is not None:
        return _kept_rows_conv_block_class
    from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v1 import ChunkedConformerConvBlock, _mem_chunks

    class ChunkedConformerConvBlockKeptRows(ChunkedConformerConvBlock):
        """Convolves the rows the upstream block keeps, which gives its outputs for a fraction of the conv work."""

        def __call__(self, inp: ReturnnTensor, *, spatial_dim: ReturnnDim, chunked_time_dim: ReturnnDim):
            x_conv1 = self.positionwise_conv1(inp)
            x_act, _ = rf.gating(x_conv1)
            x_act, window_dim = _mem_chunks(
                x_act,
                spatial_dim=spatial_dim,
                chunked_time_dim=chunked_time_dim,
                mem_size=self.chunk_history,
                end_chunk_size_dim=self.end_chunk_size_dim,
            )
            conv = self.depthwise_conv
            width = conv.filter_size[0].dimension
            pad_left = (width - 1) // 2
            history = self.chunk_history * self.end_chunk_size_dim.dimension
            if history > pad_left:
                x_act, window_dim = rf.slice(x_act, axis=window_dim, start=history - pad_left)
            x_depthwise_conv, _ = rf.conv(
                x_act,
                in_dim=conv.in_dim,
                out_dim=conv.out_dim,
                in_spatial_dims=[window_dim],
                out_spatial_dims=[spatial_dim],
                filter=conv.filter,
                filter_size=conv.filter_size,
                padding=[(max(pad_left - history, 0), width - 1 - pad_left)],
                strides=conv.strides,
                dilation_rate=conv.dilation_rate,
                groups=conv.groups,
                bias=conv.bias if conv.with_bias else None,
            )
            x_normed = self.norm(x_depthwise_conv)
            x_swish = rf.swish(x_normed)
            return self.positionwise_conv2(x_swish)

    _kept_rows_conv_block_class = ChunkedConformerConvBlockKeptRows
    return _kept_rows_conv_block_class


def chunked_attention(
    att,
    source: ReturnnTensor,
    *,
    axis: ReturnnDim,
    chunked_time_dim: ReturnnDim,
    chunk_history: int,
    end_chunk_size_dim: ReturnnDim,
) -> ReturnnTensor:
    """
    Runs the chunked rel-pos self-attention of a module through ``rf.chunked_rel_pos_self_attention``,
    which reads the history of a chunk from the chunk buffer where its backend has a kernel for it and
    materializes that history otherwise. The module keeps the projections, the position encoding and the biases.

    :param att: the ChunkedRelPosSelfAttention module
    :param source: [batch, chunked_time, axis, in_dim]
    :param axis: the chunk dim (center plus right context)
    :param chunked_time_dim: the chunks
    :param chunk_history: previous chunks every chunk attends over
    :param end_chunk_size_dim: rows a previous chunk contributes, the center
    :return: the attention output [batch, chunked_time, axis, out_dim]
    """
    center_rows = end_chunk_size_dim.dimension
    assert center_rows and axis.dimension, (
        f"chunked attention needs static chunk sizes, got {axis} and {end_chunk_size_dim}"
    )
    hist_len = chunk_history * center_rows + axis.dimension
    hist_dim = getattr(att, "hist_dim", None)
    if hist_dim is None or hist_dim.dimension != hist_len:
        hist_dim = ReturnnDim(hist_len, name="chunk_hist")
        att.hist_dim = hist_dim

    q, k, v = att.forward_qkv(source)
    query_offset = chunk_history * center_rows

    if att.learned_pos_emb is not None:
        pos_emb, pos_emb_spatial_dim = att.learned_pos_emb(
            query_spatial_dim=axis, key_value_spatial_dim=hist_dim, query_offset=query_offset
        )
    else:
        pos_emb, pos_emb_spatial_dim = rf.relative_positional_encoding(
            query_spatial_dim=axis,
            key_value_spatial_dim=hist_dim,
            feat_dim=att.pos_emb_feat_dim,
            query_offset=query_offset,
            device=source.device,
        )
    if att.pos_emb_dropout:
        pos_emb = rf.dropout(pos_emb, att.pos_emb_dropout)
    if att.linear_pos is not None:
        pos_emb = att.linear_pos(pos_emb)
    if att.separate_pos_emb_per_head:
        pos_emb = rf.split_dims(pos_emb, axis=att.key_dim_total, dims=(att.num_heads, att.key_dim_per_head))

    att_out = rf.chunked_rel_pos_self_attention(
        q,
        k,
        v,
        pos_emb,
        pos_bias_u=att.pos_bias_u,
        pos_bias_v=att.pos_bias_v,
        att_dropout=att.att_dropout,
        att_dropout_broadcast=att.att_dropout_broadcast,
        v_feat_dim=att.value_dim_per_head,
        qk_feat_dim=att.key_dim_per_head,
        chunk_dim=axis,
        chunked_time_dim=chunked_time_dim,
        hist_dim=hist_dim,
        pos_emb_spatial_dim=pos_emb_spatial_dim,
        chunk_history=chunk_history,
        end_chunk_size_dim=end_chunk_size_dim,
    )

    output, _ = rf.merge_dims(att_out, dims=(att.num_heads, att.value_dim_per_head), out_dim=att.value_dim_total)
    if att.proj:
        output = att.proj(output)
    return output
