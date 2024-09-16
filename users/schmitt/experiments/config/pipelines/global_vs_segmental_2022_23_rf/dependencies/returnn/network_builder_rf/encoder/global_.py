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
          l2: float = 0.0001,
          use_weight_feedback: bool = True,
          need_enc_ctx: bool = True,
          decoder_type: str = "lstm",
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

    self.decoder_type = decoder_type
    if decoder_type == "lstm":

      self.need_enc_ctx = need_enc_ctx
      if need_enc_ctx:
        self.enc_ctx = rf.Linear(self.out_dim, enc_key_total_dim)
      self.enc_ctx_dropout = 0.2

      self.use_weight_feedback = use_weight_feedback
      if use_weight_feedback:
        self.inv_fertility = rf.Linear(self.out_dim, dec_att_num_heads, with_bias=False)

    for p in self.parameters():
      p.weight_decay = l2

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

    self._pretrain_opts: Optional[Dict[str, Any]] = config.typed_value("pretrain_opts")

    self._mixup = None
    if config.typed_value("mixup", None) is not None:
      from i6_experiments.users.schmitt.returnn_frontend.models.rf_mixup import Mixup, MixupOpts

      self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

  @staticmethod
  def log_mel_filterbank_from_raw(
          raw_audio: Tensor,
          in_spatial_dim: Dim,
          out_dim: Dim,
          sampling_rate: int = 16_000,
          window_len: float = 0.025,
          step_len: float = 0.010,
          n_fft: Optional[int] = None,
          log_base: Union[int, float] = 10,
          f_min: Optional[Union[int, float]] = None,
          f_max: Optional[Union[int, float]] = None,
          mel_normalization: bool = False,
  ) -> Tuple[Tensor, Dim]:
    from returnn.util import math as util_math

    if raw_audio.feature_dim and raw_audio.feature_dim.dimension == 1:
      raw_audio = rf.squeeze(raw_audio, axis=raw_audio.feature_dim)
    window_num_frames = int(window_len * sampling_rate)
    step_num_frames = int(step_len * sampling_rate)
    if not n_fft:
      n_fft = util_math.next_power_of_two(window_num_frames)
    spectrogram, out_spatial_dim, in_dim_ = rf.stft(raw_audio, in_spatial_dim=in_spatial_dim,
      frame_step=step_num_frames, frame_length=window_num_frames, fft_length=n_fft, )
    power_spectrogram = rf.abs(spectrogram) ** 2.0
    mel_fbank = rf.audio.mel_filterbank(
      power_spectrogram, in_dim=in_dim_, out_dim=out_dim, sampling_rate=sampling_rate, f_min=f_min, f_max=f_max)
    log_mel_fbank = rf.safe_log(mel_fbank, eps=1e-10)
    if log_base != math.e:
      log_mel_fbank = log_mel_fbank * (1.0 / math.log(log_base))

    if mel_normalization:
      # hard-coded for spanish task
      es8khz_global_mean = rf.Tensor(
        name="es8khz_global_mean",
        dims=[log_mel_fbank.feature_dim],
        dtype=log_mel_fbank.dtype,
        raw_tensor=torch.tensor(
          numpy.loadtxt(
            "/nas/models/asr/rschmitt/setups/spanish/2024-07-09--trans-att-i6/stats/feature_mean",
            dtype="float32",
          )
        ),
      )
      es8khz_global_stddev = rf.Tensor(
        name="es8khz_global_stddev",
        dims=[log_mel_fbank.feature_dim],
        dtype=log_mel_fbank.dtype,
        raw_tensor=torch.tensor(
          numpy.loadtxt(
            "/nas/models/asr/rschmitt/setups/spanish/2024-07-09--trans-att-i6/stats/feature_std_dev",
            dtype="float32",
          )
        ),
      )

      log_mel_fbank = (
        log_mel_fbank - rf.copy_to_device(es8khz_global_mean)
      ) / rf.copy_to_device(es8khz_global_stddev)

    return log_mel_fbank, out_spatial_dim

  def encode(
          self,
          source: Tensor,
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Dict[str, Tensor], Dim]:
    """encode, and extend the encoder output for things we need in the decoder"""
    # log mel filterbank features
    source, in_spatial_dim = self.log_mel_filterbank_from_raw(
      raw_audio=source,
      in_spatial_dim=in_spatial_dim,
      out_dim=self.in_dim,
      log_base=math.exp(
        2.3026  # almost 10.0 but not exactly...
      ) if self.decoder_type == "lstm" else 10.0,
      **self.feature_extraction_opts
    )
    if self._mixup:
      source = self._mixup(source, spatial_dim=in_spatial_dim)
    # SpecAugment
    source = rf.audio.specaugment(
      source,
      spatial_dim=in_spatial_dim,
      feature_dim=self.in_dim,
      **self._specaugment_opts,
    )

    if collected_outputs is None:
      collected_outputs = {}
    # Encoder including convolutional frontend
    with _opt_apply_pretrain_to_encoder(self, collected_outputs, self._pretrain_opts):
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


@contextlib.contextmanager
def _opt_apply_pretrain_to_encoder(
    encoder: ConformerEncoder, collected_outputs: Optional[Dict[str, Tensor]], pretrain_opts: Optional[Dict[str, Any]]
):
    """Function is run within RETURNN."""
    if not pretrain_opts:
        yield
        return
    step = rf.get_run_ctx().step
    steps: Union[Sequence[Tuple[int, Dict[str, Any]]], Dict[int, Dict[str, Any]]] = pretrain_opts["steps"]
    if isinstance(steps, (list, tuple)):
        steps_ = {}
        step_bound = 0
        for step_bound_rel, opts in steps:
            step_bound += step_bound_rel
            steps_[step_bound] = opts
        steps = steps_
    assert isinstance(steps, dict)
    for step_bound, opts in sorted(steps.items()):
        if step < step_bound:
            assert isinstance(opts, dict)
            opts_ = opts.copy()
            # somewhat hacky but that is still the easiest way I can think of, without touching a lot of other code
            pretrain_num_layers = opts_.pop("num_layers")
            assert not opts_, f"unhandled opts: {opts_} in opts {opts} for step bound {step_bound}"
            orig_layers = encoder.layers[:]
            del encoder.layers[pretrain_num_layers:]
            yield
            encoder.layers[:] = orig_layers
            if collected_outputs is not None:
                assert len(collected_outputs) == pretrain_num_layers
                for i in range(pretrain_num_layers, len(orig_layers)):
                    collected_outputs[str(i)] = collected_outputs[str(pretrain_num_layers - 1)]
            return
    yield
    return


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
