from typing import Optional, Dict, Any, Sequence, Tuple, List, Union, TYPE_CHECKING
import contextlib
import math
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.supports_label_scorer_torch import RFModelWithMakeLabelScorer

_log_mel_feature_dim = 80
_batch_size_factor = 160


class BaseModel(rf.Module):
  def __init__(
          self,
          in_dim: Dim,
          *,
          num_enc_layers: int = 12,
          target_dim: Dim,
          wb_target_dim: Optional[Dim] = None,
          blank_idx: int,
          enc_aux_logits: Sequence[int] = (),  # layers
          enc_model_dim: Dim = Dim(name="enc", dimension=512),
          enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
          enc_att_num_heads: int = 4,
          enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
          enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
          att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
          att_dropout: float = 0.1,
          enc_dropout: float = 0.1,
          enc_att_dropout: float = 0.1,
          l2: float = 0.0001,
          language_model: Optional[RFModelWithMakeLabelScorer] = None,
  ):
    super(BaseModel, self).__init__()

    from returnn.config import get_global_config

    config = get_global_config(return_empty_if_none=True)

    self.in_dim = in_dim
    self.encoder = ConformerEncoder(
      in_dim,
      enc_model_dim,
      ff_dim=enc_ff_dim,
      input_layer=ConformerConvSubsample(
        in_dim,
        out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
        filter_sizes=[(3, 3), (3, 3), (3, 3)],
        pool_sizes=[(1, 2)],
        strides=[(1, 1), (3, 1), (2, 1)],
      ),
      encoder_layer_opts=enc_conformer_layer_opts,
      num_layers=num_enc_layers,
      num_heads=enc_att_num_heads,
      dropout=enc_dropout,
      att_dropout=enc_att_dropout,
    )

    self.target_dim = target_dim
    self.blank_idx = blank_idx

    self.enc_key_total_dim = enc_key_total_dim
    self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
    self.att_num_heads = att_num_heads
    self.att_dropout = att_dropout
    self.dropout_broadcast = rf.dropout_broadcast_default()

    self.enc_ctx = rf.Linear(self.encoder.out_dim, enc_key_total_dim)
    self.enc_ctx_dropout = 0.2
    self.enc_win_dim = Dim(name="enc_win_dim", dimension=5)

    self.inv_fertility = rf.Linear(self.encoder.out_dim, att_num_heads, with_bias=False)

    self.target_embed = rf.Embedding(target_dim, Dim(name="target_embed", dimension=640))

    self.s = rf.ZoneoutLSTM(
      self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
      Dim(name="lstm", dimension=1024),
      zoneout_factor_cell=0.15,
      zoneout_factor_output=0.05,
      use_zoneout_output=False,  # like RETURNN/TF ZoneoutLSTM old default
      # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
      # parts_order="ifco",
      parts_order="jifo",  # NativeLSTM (the code above converts it...)
      forget_bias=0.0,  # the code above already adds it during conversion
    )

    self.weight_feedback = rf.Linear(att_num_heads, enc_key_total_dim, with_bias=False)
    self.s_transformed = rf.Linear(self.s.out_dim, enc_key_total_dim, with_bias=False)
    self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
    self.readout_in = rf.Linear(
      self.s.out_dim + self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
      Dim(name="readout", dimension=1024),
    )
    self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)

    for p in self.parameters():
      p.weight_decay = l2

    if enc_aux_logits:
      if not wb_target_dim:
        wb_target_dim = target_dim + 1
    for i in enc_aux_logits:
      setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))

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

    # Note: Even though we have this here, it is not used in loop_step or decode_logits.
    # Instead, it is intended to make a separate label scorer for it.
    self.language_model = None
    self.language_model_make_label_scorer = None
    if language_model:
      self.language_model, self.language_model_make_label_scorer = language_model

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
      log_base=math.exp(2.3026),  # almost 10.0 but not exactly...
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
    with _opt_apply_pretrain_to_encoder(self.encoder, collected_outputs, self._pretrain_opts):
      enc, enc_spatial_dim = self.encoder(
        source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
      )
    enc_ctx = self.enc_ctx(enc)
    inv_fertility = rf.sigmoid(self.inv_fertility(enc))
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
