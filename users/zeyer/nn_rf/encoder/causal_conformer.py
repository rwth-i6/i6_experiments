"""
Fully-causal Conformer encoder building blocks:
causal relpos self-attention + causal depthwise conv.

A zero-lookahead streaming baseline with unlimited left context,
except for the small fixed lookahead (~50 ms) from the (unchanged) ConformerConvSubsample frontend.

Contrast with the chunked encoders (block attention, limited left context + lookahead):
here every layer is strictly causal, so the model is frame-synchronous.
"""

from __future__ import annotations

from typing import Any, Union

import returnn.frontend as rf
from returnn.tensor import Dim, Tensor
from returnn.frontend.encoder.conformer import ConformerConvBlock, ConformerEncoderLayer


class CausalRelPosSelfAttention(rf.RelPosCausalSelfAttention):
    """
    Causal relative-positional self-attention, returning just the output tensor.

    RETURNN's causal self-attention returns ``(output, state)`` (for step-wise decoding),
    but :class:`ConformerEncoderLayer` calls ``self.self_att(x, axis=...)`` and expects a single tensor
    (as the non-causal :class:`rf.RelPosSelfAttention` returns).
    In the full-sequence (training) path,
    each position attends to itself and all past positions only
    (proper causal masking; see ``_causal_self_att_step``).
    """

    def __call__(self, source: Tensor, *, axis: Dim, **kwargs) -> Tensor:
        """forward (drops the returned state)"""
        out, _ = super().__call__(source, axis=axis)
        return out


class CausalConformerConvBlock(ConformerConvBlock):
    """
    Conformer convolution block with a CAUSAL depthwise conv.

    The default block uses ``padding="same"``,
    so the depthwise conv sees +/-(kernel_size // 2) frames --
    i.e. ~kernel_size/2 frames of FUTURE context
    (15 encoder frames ~= 900 ms for the default kernel_size=32).
    Here we left-pad by ``kernel_size - 1`` and use a ``"valid"`` conv,
    so each output frame depends only on current + past frames.
    """

    def __init__(self, out_dim: Dim, *, kernel_size: int, norm: Union[rf.BatchNorm, Any]):
        super().__init__(out_dim, kernel_size=kernel_size, norm=norm)
        # Replace the same-padding conv;
        # we pad causally on the left in __call__ instead.
        self.depthwise_conv = rf.Conv1d(
            out_dim, out_dim, filter_size=kernel_size, groups=out_dim.dimension, padding="valid"
        )

    def __call__(self, inp: Tensor, *, spatial_dim: Dim) -> Tensor:
        """forward"""
        x_conv1 = self.positionwise_conv1(inp)
        x_act, _ = rf.gating(x_conv1)
        kernel_size = self.depthwise_conv.filter_size[0].dimension
        # Causal: left-pad kernel_size-1 frames,
        # then a valid conv over (L + kernel_size - 1) frames yields L frames again,
        # each depending only on current + past.
        x_padded, (ext_spatial_dim,) = rf.pad(x_act, axes=[spatial_dim], padding=[(kernel_size - 1, 0)], value=0.0)
        x_depthwise_conv, out_spatial_dim = self.depthwise_conv(x_padded, in_spatial_dim=ext_spatial_dim)
        out_spatial_dim.declare_same_as(spatial_dim)
        x_normed = self.norm(x_depthwise_conv)
        x_swish = rf.swish(x_normed)
        x_conv2 = self.positionwise_conv2(x_swish)
        return x_conv2


class CausalConformerEncoderLayer(ConformerEncoderLayer):
    """
    Conformer block, fully causal:
    causal depthwise conv + (by default) causal relpos self-att.

    Drop-in replacement for :class:`ConformerEncoderLayer` in ``rf.encoder.conformer.ConformerEncoder``.
    """

    def __init__(self, *args, self_att: Any = None, conv_kernel_size: int = 32, **kwargs):
        if self_att is None:
            self_att = rf.build_dict(CausalRelPosSelfAttention)
        super().__init__(*args, self_att=self_att, conv_kernel_size=conv_kernel_size, **kwargs)
        # Swap the same-padding conv block for the causal one,
        # reusing the norm super() built.
        self.conv_block = CausalConformerConvBlock(
            self.out_dim, kernel_size=conv_kernel_size, norm=self.conv_block.norm
        )
