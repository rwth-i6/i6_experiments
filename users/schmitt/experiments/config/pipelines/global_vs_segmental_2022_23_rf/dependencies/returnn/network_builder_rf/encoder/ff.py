import copy
from typing import Dict, Optional, Tuple, Any
import math

import returnn.frontend as rf
from returnn.frontend import Dim, Tensor
from returnn.frontend.encoder.conformer import ConformerConvSubsample

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.encoder.global_ import (
  GlobalConformerEncoder,
)

_log_mel_feature_dim = 80


class LinearEncoderBlock(rf.Module):
  def __init__(self, model_dim: Dim, dropout: float = 0.1):
    super().__init__()

    self.model_dim = model_dim
    self.ff = rf.Linear(model_dim, model_dim)
    self.dropout = dropout
    self.dropout_broadcast = rf.dropout_broadcast_default()

  def __call__(self, x, use_relu: bool = True):
    x = self.ff(x)
    x = rf.dropout(x, self.dropout, axis=self.dropout_broadcast and self.ff.out_dim)
    x = rf.relu(x)

    return x


class LinearEncoder(rf.Module):
  """
  Represents Conformer encoder architecture
  """

  def __init__(
          self,
          in_dim: Dim,
          out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
          *,
          num_layers: int,
          dropout: float = 0.1,
          input_dropout: float = 0.1,
          sequential=rf.Sequential,
          feature_extraction_opts: Optional[Dict[str, Any]] = None,
          decoder_type: str = "lstm",
          use_weight_feedback: bool = True,
          need_enc_ctx: bool = True,
          enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
          dec_att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
          enc_ctx_layer: Optional[str] = None,
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
    :param sequential:
    """
    super().__init__()

    from returnn.config import get_global_config
    config = get_global_config()

    self.in_dim = in_dim
    self.out_dim = out_dim
    self.dropout = dropout
    self.input_dropout = input_dropout
    self.dropout_broadcast = rf.dropout_broadcast_default()
    if not feature_extraction_opts:
      feature_extraction_opts = {}
    self.feature_extraction_opts = feature_extraction_opts
    self._specaugment_opts = {
      "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
      "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
      "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
                                      or (_log_mel_feature_dim // 5),
      "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
    }

    self.enc_ctx_layer = enc_ctx_layer

    self.decoder_type = decoder_type
    if decoder_type == "lstm":

      self.need_enc_ctx = need_enc_ctx
      if need_enc_ctx:
        self.enc_ctx = rf.Linear(self.out_dim, enc_key_total_dim)
      self.enc_ctx_dropout = 0.2

      self.use_weight_feedback = use_weight_feedback
      if use_weight_feedback:
        self.inv_fertility = rf.Linear(self.out_dim, dec_att_num_heads, with_bias=False)

    self.input_layer = ConformerConvSubsample(
      in_dim,
      out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
      filter_sizes=[(3, 3), (3, 3), (3, 3)],
      pool_sizes=[(1, 2)],
      strides=[(1, 1), (3, 1), (2, 1)],
    )
    self.input_projection = rf.Linear(
      self.input_layer.out_dim if self.input_layer else self.in_dim, self.out_dim, with_bias=False
    )

    encoder_layer = LinearEncoderBlock(out_dim, dropout=self.dropout)
    self.layers = sequential(copy.deepcopy(encoder_layer) for _ in range(num_layers - 1))
    self.final_layer = rf.Linear(out_dim, out_dim)
    self.final_layer_norm = rf.LayerNorm(out_dim)

  def __call__(
          self,
          source: Tensor,
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Tensor, Dim]:
    """forward"""

    x_subsample, out_spatial_dim = self.input_layer(source, in_spatial_dim=in_spatial_dim)
    x_linear = self.input_projection(x_subsample)

    if collected_outputs is not None:
      collected_outputs["encoder_input"] = x_linear

    x = rf.dropout(x_linear, self.input_dropout, axis=self.dropout_broadcast and self.input_projection.out_dim)
    x = self.layers(x, collected_outputs=collected_outputs)
    x = self.final_layer(x)
    x = rf.dropout(x, self.dropout, axis=self.dropout_broadcast and self.final_layer.out_dim)
    x = self.final_layer_norm(x)
    return x, out_spatial_dim

  def encode(
          self,
          source: Tensor,
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Dict[str, Tensor], Dim]:
    """encode, and extend the encoder output for things we need in the decoder"""
    # log mel filterbank features
    source, in_spatial_dim = GlobalConformerEncoder.log_mel_filterbank_from_raw(
      raw_audio=source,
      in_spatial_dim=in_spatial_dim,
      out_dim=self.in_dim,
      log_base=math.exp(
        2.3026  # almost 10.0 but not exactly...
      ) if self.decoder_type == "lstm" else 10.0,
      **self.feature_extraction_opts
    )
    # SpecAugment
    source = rf.audio.specaugment(
      source,
      spatial_dim=in_spatial_dim,
      feature_dim=self.in_dim,
      **self._specaugment_opts,
    )

    # Encoder including convolutional frontend
    enc, enc_spatial_dim = self(
      source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
    )

    if self.decoder_type == "lstm":
      if self.need_enc_ctx:
        if self.enc_ctx_layer is None:
          enc_ctx = self.enc_ctx(enc)
        else:
          enc_ctx = self.enc_ctx(collected_outputs[self.enc_ctx_layer])
      else:
        enc_ctx = None
      if self.use_weight_feedback:
        inv_fertility = rf.sigmoid(self.inv_fertility(enc))
      else:
        inv_fertility = None
    else:
      enc_ctx = None
      inv_fertility = None

    return dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility), enc_spatial_dim
