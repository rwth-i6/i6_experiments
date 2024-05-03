from typing import Optional, Dict, Any, Sequence, Tuple, List
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor, _log_mel_feature_dim
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import BaseModel


class SegmentalAttentionModel(BaseModel):
  def __init__(
          self,
          length_model_state_dim: Dim,
          length_model_embed_dim: Dim,
          center_window_size: int,
          align_target_dim: Dim,
          **kwargs
  ):
    super(SegmentalAttentionModel, self).__init__(**kwargs)

    self.align_target_dim = align_target_dim

    self.length_model_state_dim = length_model_state_dim
    self.length_model_embed_dim = length_model_embed_dim
    self.emit_prob_dim = Dim(name="emit_prob", dimension=1)
    self.center_window_size = center_window_size
    self.accum_att_weights_dim = Dim(name="accum_att_weights", dimension=center_window_size)

    self.target_embed_length_model = rf.Embedding(align_target_dim, self.length_model_embed_dim)
    # when using rf.LSTM, something with the parameter import from TF checkpoint was not right
    # i.e. the import worked but the LSTM output was different than in TF even though the inputs were the same
    # self.s_length_model = rf.LSTM(self.encoder.out_dim + self.length_model_embed_dim, self.length_model_state_dim)
    self.s_length_model = rf.ZoneoutLSTM(
      self.encoder.out_dim + self.length_model_embed_dim,
      self.length_model_state_dim,
      parts_order="jifo",
      forget_bias=0.0,
    )
    self.emit_prob = rf.Linear(self.length_model_state_dim, self.emit_prob_dim)

  def label_decoder_default_initial_state(
          self,
          *,
          batch_dims: Sequence[Dim],
          segment_starts_sparse_dim: Optional[Dim] = None,
          segment_lens_sparse_dim: Optional[Dim] = None,
  ) -> rf.State:
    """Default initial state"""
    state = rf.State(
      s=self.s.default_initial_state(batch_dims=batch_dims),
      att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]),
      accum_att_weights=rf.zeros(
        list(batch_dims) + [self.accum_att_weights_dim, self.att_num_heads], feature_dim=self.att_num_heads
      ),
      segment_starts=rf.zeros(batch_dims, sparse_dim=segment_starts_sparse_dim, dtype="int32"),
      segment_lens=rf.zeros(batch_dims, sparse_dim=segment_lens_sparse_dim, dtype="int32"),
    )
    state.att.feature_dim_axis = len(state.att.dims) - 1
    return state

  def label_loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
    """loop step out"""
    return {
      "s": Tensor(
        "s", dims=batch_dims + [self.s.out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1
      ),
      "att": Tensor(
        "att",
        dims=batch_dims + [self.att_num_heads * self.encoder.out_dim],
        dtype=rf.get_default_float_dtype(),
        feature_dim_axis=-1,
      ),
    }

  def _get_prev_accum_att_weights_scattered(
          self,
          prev_accum_att_weights: Tensor,
          segment_starts: Tensor,
          prev_segment_starts: Tensor,
          prev_segment_lens: Tensor,
  ) -> Tensor:

    overlap_len = rf.cast(prev_segment_starts + prev_segment_lens - segment_starts, "int32")
    overlap_len = rf.where(
      rf.logical_or(overlap_len < 0, overlap_len > prev_segment_lens),
      rf.convert_to_tensor(0),
      overlap_len
    )
    overlap_start = prev_segment_lens - overlap_len

    slice_dim = Dim(name="slice", dimension=overlap_len)
    gather_positions = rf.range_over_dim(slice_dim)
    gather_positions += overlap_start
    prev_accum_att_weights_overlap = rf.gather(
      prev_accum_att_weights, axis=self.accum_att_weights_dim, indices=gather_positions, clip_to_valid=True
    )
    overlap_range = rf.range_over_dim(slice_dim)

    prev_accum_att_weights_scattered = rf.scatter(
      prev_accum_att_weights_overlap,
      out_dim=self.accum_att_weights_dim,
      indices=overlap_range,
      indices_dim=slice_dim,
    )

    return prev_accum_att_weights_scattered

  def _get_accum_att_weights(
          self,
          att_t_dim: Dim,
          enc_spatial_dim: Dim,
          inv_fertility: Tensor,
          att_weights: Tensor,
          prev_accum_att_weights_scattered: Tensor,
          gather_positions: Tensor,
  ) -> Tensor:
    att_weights_range = rf.range_over_dim(att_t_dim)
    att_weights_scattered = rf.scatter(
      att_weights,
      out_dim=self.accum_att_weights_dim,
      indices=att_weights_range,
      indices_dim=att_t_dim,
    )

    inv_fertility_sliced = rf.gather(inv_fertility, axis=enc_spatial_dim, indices=gather_positions, clip_to_valid=True)
    inv_fertility_scattered = rf.scatter(
      inv_fertility_sliced,
      out_dim=self.accum_att_weights_dim,
      indices=att_weights_range,
      indices_dim=att_t_dim,
    )

    accum_att_weights = prev_accum_att_weights_scattered + att_weights_scattered * inv_fertility_scattered * 0.5

    return accum_att_weights

  def _get_weight_feedback(
          self,
          prev_accum_att_weights_scattered: Tensor,
          att_t_dim: Dim,
  ) -> Tensor:
    gather_positions = rf.range_over_dim(att_t_dim)
    prev_accum_att_weights_sliced = rf.gather(
      prev_accum_att_weights_scattered,
      axis=self.accum_att_weights_dim,
      indices=gather_positions,
      clip_to_valid=True
    )

    return self.weight_feedback(prev_accum_att_weights_sliced)

  def label_sync_loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_ctx: rf.Tensor,
          inv_fertility: rf.Tensor,
          enc_spatial_dim: Dim,
          input_embed: rf.Tensor,
          segment_starts: rf.Tensor,
          segment_lens: rf.Tensor,
          state: Optional[rf.State] = None,
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    """step of the inner loop"""
    if state is None:
      batch_dims = enc.remaining_dims(
        remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
      )
      state = self.label_decoder_default_initial_state(batch_dims=batch_dims)
    state_ = rf.State()

    # during search, these need to be the values from the previous "emit" step (not necessarily the previous time step)
    prev_att = state.att
    prev_s_state = state.s
    prev_accum_att_weights = state.accum_att_weights
    prev_segment_starts = state.segment_starts
    prev_segment_lens = state.segment_lens

    s, state_.s = self.s(rf.concat_features(input_embed, prev_att), state=prev_s_state, spatial_dim=single_step_dim)
    s_transformed = self.s_transformed(s)

    slice_dim = Dim(name="slice", dimension=segment_lens)
    gather_positions = rf.range_over_dim(slice_dim)
    gather_positions += segment_starts

    enc_ctx_sliced = rf.gather(enc_ctx, axis=enc_spatial_dim, indices=gather_positions, clip_to_valid=True)
    enc_sliced = rf.gather(enc, axis=enc_spatial_dim, indices=gather_positions, clip_to_valid=True)

    prev_accum_att_weights_scattered = self._get_prev_accum_att_weights_scattered(
      prev_accum_att_weights=prev_accum_att_weights,
      segment_starts=segment_starts,
      prev_segment_starts=prev_segment_starts,
      prev_segment_lens=prev_segment_lens,
    )
    weight_feedback = self._get_weight_feedback(
      prev_accum_att_weights_scattered=prev_accum_att_weights_scattered,
      att_t_dim=slice_dim,
    )

    energy_in = enc_ctx_sliced + weight_feedback + s_transformed
    energy = self.energy(rf.tanh(energy_in))
    att_weights = rf.softmax(energy, axis=slice_dim)
    # we do not need use_mask because the softmax output is already padded with zeros
    att0 = rf.dot(att_weights, enc_sliced, reduce=slice_dim, use_mask=False)
    att0.feature_dim = self.encoder.out_dim
    att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder.out_dim))
    state_.att = att

    accum_att_weights = self._get_accum_att_weights(
      att_t_dim=slice_dim,
      enc_spatial_dim=enc_spatial_dim,
      inv_fertility=inv_fertility,
      att_weights=att_weights,
      prev_accum_att_weights_scattered=prev_accum_att_weights_scattered,
      gather_positions=gather_positions,
    )
    accum_att_weights.feature_dim = self.att_num_heads
    state_.accum_att_weights = accum_att_weights

    state_.segment_starts = segment_starts
    state_.segment_lens = segment_lens

    return {"s": s, "att": att}, state_

  def decode_label_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
    """logits for the decoder"""
    readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
    readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
    readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
    logits = self.output_prob(readout)
    return logits

  def blank_decoder_default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
    """Default initial state"""
    state = rf.State(
      s_length_model=self.s_length_model.default_initial_state(batch_dims=batch_dims),
      i=rf.zeros(batch_dims, dtype="int32"),
    )
    return state

  def blank_loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
    """loop step out"""
    return {
      "s_length_model": Tensor(
        "s_length_model",
        dims=batch_dims + [self.s_length_model.out_dim],
        dtype=rf.get_default_float_dtype(),
        feature_dim_axis=-1
      ),
    }

  def time_sync_loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_spatial_dim: Dim,
          input_embed: rf.Tensor,
          state: Optional[rf.State] = None,
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    """step of the inner loop"""
    if state is None:
      batch_dims = enc.remaining_dims(
        remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
      )
      state = self.blank_decoder_default_initial_state(batch_dims=batch_dims)
    state_ = rf.State()

    am = rf.gather(enc, axis=enc_spatial_dim, indices=state.i, clip_to_valid=True)
    s_length_model, state_.s_length_model = self.s_length_model(
      rf.concat_features(am, input_embed),
      state=state.s_length_model,
      spatial_dim=single_step_dim
    )

    state_.i = state.i + 1

    return {"s_length_model": s_length_model}, state_

  def decode_blank_logits(self, *, s_length_model: Tensor) -> Tensor:
    """logits for the decoder"""
    logits = self.emit_prob(s_length_model)
    return logits


class MakeModel:
  """for import"""

  def __init__(self, in_dim: int, align_target_dim: int, target_dim: int, *, center_window_size: int, eos_label: int = 0, num_enc_layers: int = 12):
    self.in_dim = in_dim
    self.align_target_dim = align_target_dim
    self.target_dim = target_dim
    self.center_window_size = center_window_size
    self.eos_label = eos_label
    self.num_enc_layers = num_enc_layers

  def __call__(self) -> SegmentalAttentionModel:
    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
    align_target_dim = Dim(name="align_target", dimension=self.align_target_dim, kind=Dim.Types.Feature)
    target_dim = Dim(name="non_blank_target", dimension=self.target_dim, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels(
      [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
    )

    return self.make_model(in_dim, align_target_dim, target_dim, center_window_size=self.center_window_size)

  @classmethod
  def make_model(
          cls,
          in_dim: Dim,
          align_target_dim: Dim,
          target_dim: Dim,
          *,
          center_window_size: int,
          num_enc_layers: int = 12,
          pos_emb_dropout: float = 0.0,
          language_model: Optional[Dict[str, Any]] = None,
          **extra,
  ) -> SegmentalAttentionModel:
    """make"""
    lm = None
    if language_model:
      assert isinstance(language_model, dict)
      language_model = language_model.copy()
      cls_name = language_model.pop("class")
      assert cls_name == "TransformerDecoder"
      language_model.pop("vocab_dim", None)  # will just overwrite

      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.lm.trafo import model as trafo_lm

      lm = trafo_lm.MakeModel(vocab_dim=target_dim, **language_model)()
      lm = (lm, functools.partial(trafo_lm.make_time_sync_label_scorer_torch, model=lm, align_target_dim=align_target_dim))

    return SegmentalAttentionModel(
      in_dim=in_dim,
      num_enc_layers=num_enc_layers,
      enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
      enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
      enc_att_num_heads=8,
      enc_conformer_layer_opts=dict(
        conv_norm_opts=dict(use_mask=True),
        self_att_opts=dict(
          # Shawn et al 2018 style, old RETURNN way.
          with_bias=False,
          with_linear_pos=False,
          with_pos_bias=False,
          learnable_pos_emb=True,
          separate_pos_emb_per_head=False,
          pos_emb_dropout=pos_emb_dropout,
        ),
        ff_activation=lambda x: rf.relu(x) ** 2.0,
      ),
      target_dim=target_dim,
      align_target_dim=align_target_dim,
      blank_idx=target_dim.dimension,
      language_model=lm,
      length_model_state_dim=Dim(name="length_model_state", dimension=128, kind=Dim.Types.Feature),
      length_model_embed_dim=Dim(name="length_model_embed", dimension=128, kind=Dim.Types.Feature),
      center_window_size=center_window_size,
      **extra,
    )


def from_scratch_model_def(
        *, epoch: int, in_dim: Dim, align_target_dim: Dim, target_dim: Dim) -> SegmentalAttentionModel:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  in_dim, epoch  # noqa
  config = get_global_config()  # noqa
  enc_aux_logits = config.typed_value("aux_loss_layers")
  pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
  # real input is raw audio, internally it does logmel
  in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
  lm_opts = config.typed_value("external_lm")
  center_window_size = config.typed_value("center_window_size")
  if center_window_size is None:
    raise ValueError("center_window_size is not set!")
  return MakeModel.make_model(
    in_dim,
    align_target_dim,
    target_dim,
    center_window_size=center_window_size,
    enc_aux_logits=enc_aux_logits or (),
    pos_emb_dropout=pos_emb_dropout,
    language_model=lm_opts
  )


from_scratch_model_def: ModelDef[SegmentalAttentionModel]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
  from returnn.tensor import Tensor, Dim
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  extern_data_dict = config.typed_value("extern_data")
  non_blank_vocab = config.typed_value("non_blank_vocab")
  data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
  targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
  non_blank_targets = Tensor(
    name="non_blank_targets",
    sparse_dim=Dim(description="non_blank_vocab", dimension=targets.sparse_dim.dimension - 1, kind=Dim.Types.Spatial),
    vocab=non_blank_vocab,
  )

  model_def = config.typed_value("_model_def")
  model = model_def(
    epoch=epoch, in_dim=data.feature_dim, align_target_dim=targets.sparse_dim, target_dim=non_blank_targets.sparse_dim)
  return model
