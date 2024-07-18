from typing import Optional, Dict, Any, Sequence, Tuple, List, Union, TYPE_CHECKING
import contextlib
import math
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

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
          decoder_type: str = "lstm",
  ):
    super(GlobalConformerEncoder, self).__init__(
      in_dim,
      out_dim,
      ff_dim=ff_dim,
      input_layer=ConformerConvSubsample(
        in_dim,
        out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
        filter_sizes=[(3, 3), (3, 3), (3, 3)],
        pool_sizes=[(1, 2)],
        strides=[(1, 1), (3, 1), (2, 1)],
      ),
      encoder_layer_opts=encoder_layer_opts,
      num_layers=num_layers,
      num_heads=num_heads,
      dropout=dropout,
      att_dropout=att_dropout,
    )

    from returnn.config import get_global_config

    config = get_global_config(return_empty_if_none=True)

    # self.in_dim = in_dim

    self.decoder_type = decoder_type
    if decoder_type == "lstm":
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

  def encode(
          self,
          source: Tensor,
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Dict[str, Tensor], Dim]:
    """encode, and extend the encoder output for things we need in the decoder"""
    # log mel filterbank features
    source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
      source,
      in_spatial_dim=in_spatial_dim,
      out_dim=self.in_dim,
      sampling_rate=16_000,
      log_base=math.exp(
        2.3026  # almost 10.0 but not exactly...
      ) if self.decoder_type == "lstm" else 10.0,
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
    # Encoder including convolutional frontend
    with _opt_apply_pretrain_to_encoder(self, collected_outputs, self._pretrain_opts):
      enc, enc_spatial_dim = self(
        source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
      )
    if self.decoder_type == "lstm":
      enc_ctx = self.enc_ctx(enc)
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
