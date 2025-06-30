from typing import Optional, Dict, Any, Sequence, Tuple, List, Union, TYPE_CHECKING
import contextlib
import math
import functools

import numpy as np
import torch
import numpy

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample, ConformerEncoderLayer, ConformerConvBlock

_log_mel_feature_dim = 80
_batch_size_factor = 160


class GlobalConformerEncoder(ConformerEncoder):
  def __init__(
          self,
          in_dim: Dim,
          out_dim: Dim = Dim(name="enc", dimension=512),
          *,
          num_layers: int = 12,
          target_dim: Dim,
          wb_target_dim: Optional[Dim] = None,
          aux_logits: Sequence[int] = (),  # layers
          ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
          num_heads: int = 4,
          encoder_layer_opts: Optional[Dict[str, Any]] = None,
          enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
          dec_att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
          dropout: float = 0.1,
          att_dropout: float = 0.1,
          use_weight_feedback: bool = True,
          feature_extraction_opts: Optional[Dict[str, Any]] = None,
          encoder_layer: Optional[Union[ConformerEncoderLayer, rf.Module, type, Dict[str, Any], Any]] = None,
          input_layer_cls: rf.Module = ConformerConvSubsample,
          enc_ctx_layer: Optional[str] = None,
  ):
    super(GlobalConformerEncoder, self).__init__(
      in_dim,
      out_dim,
      ff_dim=ff_dim,
      input_layer=input_layer_cls(
        in_dim,
        out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
        filter_sizes=[(3, 3), (3, 3), (3, 3)],
        pool_sizes=[(1, 2)],
        strides=[(1, 1), (3, 1), (2, 1)],
      ),
      encoder_layer_opts=encoder_layer_opts,
      encoder_layer=encoder_layer,
      num_layers=num_layers,
      num_heads=num_heads,
      dropout=dropout,
      att_dropout=att_dropout,
    )

    from returnn.config import get_global_config

    config = get_global_config(return_empty_if_none=True)

    if not feature_extraction_opts:
      feature_extraction_opts = {}
    self.feature_extraction_opts = feature_extraction_opts

    self.enc_ctx_layer = enc_ctx_layer
    self.enc_ctx = rf.Linear(self.out_dim, enc_key_total_dim)
    self.enc_ctx_dropout = 0.2

    self.use_weight_feedback = use_weight_feedback
    if use_weight_feedback:
      self.inv_fertility = rf.Linear(self.out_dim, dec_att_num_heads, with_bias=False)

    if aux_logits:
      if not wb_target_dim:
        wb_target_dim = target_dim + 1
    for i in aux_logits:
      setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.out_dim, wb_target_dim))

    self._specaugment_opts = {
      "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
      "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
      "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
                                      or (_log_mel_feature_dim // 5),
      "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
    }


class GlobalConformerEncoderWAbsolutePos(GlobalConformerEncoder):
  def __call__(
          self,
          source: Tensor,
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Tensor, Dim]:
    """forward"""
    if self.input_layer:
      x_subsample, out_spatial_dim = self.input_layer(source, in_spatial_dim=in_spatial_dim)
    else:
      x_subsample, out_spatial_dim = source, in_spatial_dim

    x_linear = self.input_projection(x_subsample)
    x = rf.dropout(x_linear, self.input_dropout, axis=self.dropout_broadcast and self.input_projection.out_dim)

    # abs pos encoding
    x = x * np.sqrt(self.out_dim.dimension) + rf.sinusoidal_positional_encoding(
      spatial_dim=out_spatial_dim, feat_dim=self.out_dim)

    x = self.layers(x, spatial_dim=out_spatial_dim, collected_outputs=collected_outputs)
    return x, out_spatial_dim


class GlobalConformerEncoderWFinalLayerNorm(GlobalConformerEncoder):
  def __init__(self, *args, **kwargs):
    super(GlobalConformerEncoderWFinalLayerNorm, self).__init__(*args, **kwargs)

    self.final_layer_norm = rf.LayerNorm(self.out_dim)

  def __call__(
          self,
          source: Tensor,
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Tensor, Dim]:
    """forward"""
    if self.input_layer:
      x_subsample, out_spatial_dim = self.input_layer(source, in_spatial_dim=in_spatial_dim)
    else:
      x_subsample, out_spatial_dim = source, in_spatial_dim

    x_subsample = x_subsample + rf.sinusoidal_positional_encoding(
      spatial_dim=out_spatial_dim, feat_dim=x_subsample.feature_dim)

    x_linear = self.input_projection(x_subsample)
    x = rf.dropout(x_linear, self.input_dropout, axis=self.dropout_broadcast and self.input_projection.out_dim)
    x = self.layers(x, spatial_dim=out_spatial_dim, collected_outputs=collected_outputs)

    x = self.final_layer_norm(x)

    return x, out_spatial_dim


class ConformerEncoderLayerWoFinalLayerNorm(ConformerEncoderLayer):
  def __init__(self, *args, **kwargs):
    super(ConformerEncoderLayerWoFinalLayerNorm, self).__init__(*args, **kwargs)

    delattr(self, "final_layer_norm")

  def __call__(self, inp: Tensor, *, spatial_dim: Dim) -> Tensor:
    """forward"""
    # FFN
    x_ffn1_ln = self.ffn1_layer_norm(inp)
    x_ffn1 = self.ffn1(x_ffn1_ln)
    x_ffn1_out = 0.5 * rf.dropout(x_ffn1, self.dropout, axis=self.dropout_broadcast and self.out_dim) + inp

    # MHSA
    x_mhsa_ln = self.self_att_layer_norm(x_ffn1_out)
    x_mhsa = self.self_att(x_mhsa_ln, axis=spatial_dim)
    x_mhsa = rf.dropout(x_mhsa, self.dropout, axis=self.dropout_broadcast and self.out_dim)
    x_mhsa_out = x_mhsa + x_ffn1_out

    # Conv
    x_conv_ln = self.conv_layer_norm(x_mhsa_out)
    x_conv = self.conv_block(x_conv_ln, spatial_dim=spatial_dim)
    x_conv_out = rf.dropout(x_conv, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_mhsa_out

    # FFN
    x_ffn2_ln = self.ffn2_layer_norm(x_conv_out)
    x_ffn2 = self.ffn2(x_ffn2_ln)
    x_ffn2_out = 0.5 * rf.dropout(x_ffn2, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_conv_out

    return x_ffn2_out


class ConformerEncoderLayerWoConvolution(ConformerEncoderLayer):
  def __init__(self, *args, **kwargs):
    super(ConformerEncoderLayerWoConvolution, self).__init__(*args, **kwargs)

    delattr(self, "conv_block")
    delattr(self, "conv_layer_norm")

  def __call__(self, inp: Tensor, *, spatial_dim: Dim) -> Tensor:
    """forward"""
    # FFN
    x_ffn1_ln = self.ffn1_layer_norm(inp)
    x_ffn1 = self.ffn1(x_ffn1_ln)
    x_ffn1_out = 0.5 * rf.dropout(x_ffn1, self.dropout, axis=self.dropout_broadcast and self.out_dim) + inp

    # MHSA
    x_mhsa_ln = self.self_att_layer_norm(x_ffn1_out)
    x_mhsa = self.self_att(x_mhsa_ln, axis=spatial_dim)
    x_mhsa = rf.dropout(x_mhsa, self.dropout, axis=self.dropout_broadcast and self.out_dim)
    x_mhsa_out = x_mhsa + x_ffn1_out

    # FFN
    x_ffn2_ln = self.ffn2_layer_norm(x_mhsa_out)
    x_ffn2 = self.ffn2(x_ffn2_ln)
    x_ffn2_out = 0.5 * rf.dropout(x_ffn2, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_mhsa_out

    # last LN layer
    return self.final_layer_norm(x_ffn2_out)


class ConformerConvBlockWZeroPadding(ConformerConvBlock):
  def __call__(self, inp: Tensor, *, spatial_dim: Dim) -> Tensor:
    """forward"""
    x_conv1 = self.positionwise_conv1(inp)
    x_act, _ = rf.gating(x_conv1)
    x_act = x_act.copy_masked(mask_value=0.0)
    x_depthwise_conv, _ = self.depthwise_conv(x_act, in_spatial_dim=spatial_dim)
    x_normed = self.norm(x_depthwise_conv)
    x_swish = rf.swish(x_normed)
    x_conv2 = self.positionwise_conv2(x_swish)
    return x_conv2


class ConformerConvSubsampleWZeroPadding(ConformerConvSubsample):
  def __call__(self, source: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
    """forward"""
    assert self.in_dim in source.dims
    in_spatial_dims = [in_spatial_dim, self.in_dim]
    in_dim = self._dummy_in_dim
    x = rf.expand_dim(source, dim=in_dim)
    for i, conv_layer in enumerate(self.conv_layers):
      x = x.copy_masked(mask_value=0.0)
      x, in_spatial_dims = conv_layer(x, in_spatial_dims=in_spatial_dims)
      in_dim = conv_layer.out_dim
      x = self.activation(x)
      if self.pool_sizes and i < len(self.pool_sizes):
        x = x.copy_masked(mask_value=0.0)
        x, in_spatial_dims = rf.pool2d(
          x, in_spatial_dims=in_spatial_dims, pool_size=self.pool_sizes[i], padding="same", mode="max"
        )
    x, in_spatial_dims[-1] = rf.replace_dim(x, out_dim=self._final_second_spatial_dim, in_dim=in_spatial_dims[-1])
    out, _ = rf.merge_dims(x, dims=[self._final_second_spatial_dim, in_dim])
    return out, in_spatial_dims[0]
