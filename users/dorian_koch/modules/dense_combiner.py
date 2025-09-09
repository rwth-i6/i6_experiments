from __future__ import annotations
from typing import Any, Callable, Dict, Tuple, Union, Optional, Sequence
import logging
from returnn.frontend.attention import _att_dropout_broadcast_default
from returnn.frontend.decoder.transformer import FeedForward, make_norm
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend._cache import Cache
from returnn.util.basic import NotSpecified
import copy as _copy


def dot_attention_with_bias(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    weight_bias: Tensor | None = None,
    key_dim: Dim,
    axis: Dim,
    att_dropout: float = 0.0,
    att_dropout_broadcast: Optional[bool] = None,
) -> Tensor:
    """
    Dot attention, but with weight bias. Can be used to implement ALiBi
    """
    query *= key_dim.dimension**-0.5
    energy = rf.matmul(query, keys, reduce=key_dim)
    if weight_bias is not None:
        assert all([d in energy.dims_set for d in weight_bias.dims_set]), "do not add additional dims to energy"
        energy = energy + weight_bias
    att_weights = rf.softmax(energy, axis=axis)
    if att_dropout_broadcast is None:
        att_dropout_broadcast = _att_dropout_broadcast_default()
    att_weights = rf.dropout(att_weights, att_dropout, axis=att_dropout_broadcast and axis)
    # Masking not needed because softmax should already have masked,
    # so we have 0.0 att weights for padded frames.
    att = rf.matmul(att_weights, values, reduce=axis, use_mask=False)
    if values.feature_dim in att.dims:
        att.feature_dim = values.feature_dim
    return att


class DenseCombinerAttention(rf.Module):
    """
    Used to combine dense k-probs for Denoising Language Model training
    similar to cross attention, but with bias in attention energies
    """

    def __init__(
        self,
        k_embed_dim: Dim,
        query_in_dim: Dim,
        proj_dim: Optional[Dim],
        *,
        key_dim_total: Dim,
        value_dim_total: Dim,
        num_heads: Union[int, Dim],
        with_bias: bool = True,
        att_dropout: float = 0.1,
        att_dropout_broadcast: Optional[bool] = None,
    ):
        """
        :param encoder_dim: encoder output dim = input dim for key-value
        :param query_in_dim: input dim for query
        :param proj_dim: if given, will add a final linear projection to this dim.
            otherwise no projection after the attention
        :param key_dim_total: total key dim. should be a multiple of num_heads
        :param value_dim_total: total value dim. should be a multiple of num_heads
        :param num_heads: number of heads
        :param with_bias: whether to add bias to qkv and proj linear projections.
            Was False in original Transformer, but many recent implementations use True by default.
            Also see: https://github.com/rwth-i6/returnn_common/issues/234.
        :param att_dropout: dropout for attention weights
        :param att_dropout_broadcast: whether to broadcast over all but ``axis``.
            normally not wanted. disabled by default since behavior version 19.
        """
        super().__init__()
        self.k_embed_dim = k_embed_dim
        self.query_in_dim = query_in_dim
        self.out_dim = proj_dim if proj_dim else value_dim_total
        if isinstance(num_heads, int):
            num_heads = Dim(num_heads, name="num_heads")
        self.key_dim_total = key_dim_total
        self.key_dim_per_head = key_dim_total.div_left(num_heads)
        self.value_dim_total = value_dim_total
        self.value_dim_per_head = value_dim_total.div_left(num_heads)
        self.num_heads = num_heads
        self.kv_dim_total = key_dim_total + value_dim_total
        self.kv_dim_per_head = self.key_dim_per_head + self.value_dim_per_head
        self.kv = rf.Linear(k_embed_dim, self.kv_dim_total, with_bias=with_bias)
        self.q = rf.Linear(query_in_dim, self.key_dim_total, with_bias=with_bias)
        # In Fairseq MultiheadAttention, they use:
        #   nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))  (same for q_proj, v_proj),
        # where (Xavier Glorot) xavier_uniform_ means:
        #   std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        #   a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        #   _no_grad_uniform_(tensor, -a, a)
        # Out nn.init.VarianceScaling with mode="fan_avg", distribution="uniform":
        #   scale = scale * 2.0 / float(fan_in + fan_out)
        #   limit = math.sqrt(3.0 * scale)
        #   nn.random(distribution="uniform", minval=-limit, maxval=limit, ...)
        # Xavier Glorot: VarianceScaling with mode="fan_avg", distribution="uniform", scale=1.0.
        self.kv.weight.initial = rf.init.Glorot(scale=3 / 4)
        self.q.weight.initial = rf.init.Glorot(scale=1 / 2)
        # The bias init is different, but not sure how important this is.
        if proj_dim:
            self.proj = rf.Linear(value_dim_total, proj_dim, with_bias=with_bias)
        else:
            self.proj = None
        self.att_dropout = att_dropout
        if att_dropout_broadcast is None:
            att_dropout_broadcast = _att_dropout_broadcast_default()
        self.att_dropout_broadcast = att_dropout_broadcast

        self.bias_embedding = rf.Parameter([k_embed_dim])
        self.bias_embedding.initial = rf.init.Glorot()

    def forward_kv(self, source: Tensor) -> Tuple[Tensor, Tensor]:
        """
        This would be calculated once for the whole sequence (batch)
        and then always reused for :func:`attention`.

        :return: k,v
        """
        qkv = self.kv(source)
        qkv = rf.split_dims(qkv, axis=self.kv_dim_total, dims=(self.num_heads, self.kv_dim_per_head))
        k, v = rf.split(
            qkv,
            axis=self.kv_dim_per_head,
            out_dims=(self.key_dim_per_head, self.value_dim_per_head),
        )
        return k, v

    def forward_query(self, source: Tensor) -> Tensor:
        """
        This is calculated for every different query.

        :return: q
        """
        q = self.q(source)
        q = rf.split_dims(q, axis=self.key_dim_total, dims=(self.num_heads, self.key_dim_per_head))
        return q

    def __call__(self, x: Tensor, *, kprobs: Tensor, k_embeds: Tensor, k_dim: Dim) -> Tensor:
        """
        :param x: Some hidden state
        :param kprobs: probabilities, should probably add up to one (so not in logspace)
        :param k_embeds: embeddings for those probabilities
        :param k_dim: dimension of k
        """
        q = self.forward_query(x)

        assert all([d in k_embeds.dims_set for d in kprobs.dims_set])
        # not super clear if we even need to do this
        # could probably just re-use the embeds for both keys and values, and not do all this additional computation
        # complexity increases by a lot because of k_dim... (but i didnt check)
        k, v = self.forward_kv(k_embeds + kprobs * self.bias_embedding)

        att = dot_attention_with_bias(
            q,
            k,
            v,
            weight_bias=kprobs,
            key_dim=self.key_dim_per_head,
            axis=k_dim,
            att_dropout=self.att_dropout,
            att_dropout_broadcast=self.att_dropout_broadcast,
        )
        output, _ = rf.merge_dims(att, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
        if self.proj:
            output = self.proj(output)
        return output


class DenseCombinerBlock(rf.Module):
    """ """

    def __init__(
        self,
        embed_dim: Dim,
        out_dim: Dim = Dim(512, name="dense-default-out-dim"),
        *,
        ff: Union[type, Dict[str, Any], rf.Module] = NotSpecified,
        ff_dim: Union[Dim, int] = NotSpecified,
        ff_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = NotSpecified,
        dropout: float = 0.1,
        num_heads: int = 8,
        att_dropout: float = 0.1,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.out_dim = out_dim

        if ff is NotSpecified:
            ff = FeedForward
        if isinstance(ff, rf.Module):
            ff = _copy.deepcopy(ff)
        else:
            ff_kwargs = dict(out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=ff_activation)
            ff_kwargs = {k: v for (k, v) in ff_kwargs.items() if v is not NotSpecified}
            if isinstance(ff, type):
                ff = ff(**ff_kwargs)
            elif isinstance(ff, dict):
                ff = rf.build_from_dict(ff, **ff_kwargs)
            else:
                raise TypeError(f"unexpected ff type {ff!r}")
        assert isinstance(ff, rf.Module)

        self.ff = ff
        self.ff_layer_norm = make_norm(norm, out_dim)

        self.dense_att: Optional[DenseCombinerAttention] = (
            None  # type might be inaccurate, but we expect this interface
        )
        self.dense_att_layer_norm = None
        assert embed_dim is not None
        cross_att_opts = dict(
            k_embed_dim=self.embed_dim,
            query_in_dim=out_dim,
            proj_dim=out_dim,
            key_dim_total=out_dim,
            value_dim_total=out_dim,
            num_heads=num_heads,
            att_dropout=att_dropout,
        )

        self.dense_att = DenseCombinerAttention(**cross_att_opts)
        self.dense_att_layer_norm = make_norm(norm, out_dim)

    def __call__(self, x: Tensor, *, k_probs: Tensor, k_embeds: Tensor, k_dim: Dim) -> Tensor:
        """forward"""

        # (multi-head) dense-attention (DA)

        x_da_ln = self.dense_att_layer_norm(x)
        x_da = self.dense_att(x_da_ln, k_embeds=k_embeds, kprobs=k_probs, k_dim=k_dim)
        x_da = rf.dropout(x_da, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x_da + x

        # feed-forward (FF)
        x_ff_ln = self.ff_layer_norm(x)
        x_ff = self.ff(x_ff_ln)
        x_ff = rf.dropout(x_ff, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x_ff + x

        return x
