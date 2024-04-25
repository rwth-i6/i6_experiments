from typing import Optional, Dict, Any, Sequence, Tuple, List
import functools
import math
import tree
import numpy

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import BaseModel

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.recog import RecogDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor, _log_mel_feature_dim
from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf._moh_att_2023_06_30_import import map_param_func_v2 as map_param_func_v2_albert


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

  def __init__(self, in_dim: int, align_target_dim: int, target_dim: int, *, eos_label: int = 0, num_enc_layers: int = 12):
    self.in_dim = in_dim
    self.align_target_dim = align_target_dim
    self.target_dim = target_dim
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

    return self.make_model(in_dim, align_target_dim, target_dim)

  @classmethod
  def make_model(
          cls,
          in_dim: Dim,
          align_target_dim: Dim,
          target_dim: Dim,
          *,
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

      from . import trafo_lm

      lm = trafo_lm.MakeModel(vocab_dim=target_dim, **language_model)()
      lm = (lm, functools.partial(trafo_lm.make_label_scorer_torch, model=lm))

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
      center_window_size=5,
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
  lm_opts = config.typed_value("external_language_model")
  return MakeModel.make_model(
    in_dim,
    align_target_dim,
    target_dim,
    enc_aux_logits=enc_aux_logits or (),
    pos_emb_dropout=pos_emb_dropout,
    language_model=lm_opts
  )


from_scratch_model_def: ModelDef[SegmentalAttentionModel]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def _get_non_blank_mask(x: Tensor, blank_idx: int):
  non_blank_mask = x != rf.convert_to_tensor(blank_idx)
  return rf.where(non_blank_mask, rf.sequence_mask(x.dims), rf.convert_to_tensor(False))


def _get_masked(
        input: Tensor, mask: Tensor, mask_dim: Dim, batch_dims: Sequence[Dim], result_spatial_dim: Optional[Dim] = None
) -> Tuple[Tensor, Dim]:

  import torch

  if not result_spatial_dim:
    new_lens = rf.reduce_sum(rf.cast(mask, "int32"), axis=mask_dim)
    result_spatial_dim = Dim(name=f"{mask_dim.name}_masked", dimension=rf.copy_to_device(new_lens, "cpu"))
  else:
    new_lens = rf.copy_to_device(result_spatial_dim.get_size_tensor(), input.device)
  # max number of non-blank targets in the batch
  result_spatial_size = rf.cast(rf.reduce_max(new_lens, axis=batch_dims), "int32")
  mask_axis = mask.get_axis_from_description(mask_dim)
  # scatter indices
  idxs = rf.cast(mask, "int32").copy_template()
  idxs.raw_tensor = torch.cumsum(mask.raw_tensor.to(
    torch.int32), dim=mask_axis, dtype=torch.int32) - 1
  idxs = rf.where(mask, idxs, result_spatial_size)
  # scatter non-blank targets
  # blanks are scattered to the last position of each batch
  result_spatial_dim_temp = result_spatial_dim + 1
  result = rf.scatter(
    input, indices=idxs, indices_dim=mask_dim, out_dim=result_spatial_dim_temp)
  # remove accumulated blanks at the last position
  result = result.copy_transpose([result_spatial_dim_temp] + batch_dims)
  result_raw_tensor = result.raw_tensor
  result = result.copy_template_replace_dim_tag(0, result_spatial_dim)
  result.raw_tensor = result_raw_tensor[:-1]
  result = result.copy_transpose(batch_dims + [result_spatial_dim])

  return result, result_spatial_dim


def from_scratch_training(
        *,
        model: SegmentalAttentionModel,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        align_targets: rf.Tensor,
        align_targets_spatial_dim: Dim
):
  """Function is run within RETURNN."""
  from returnn.config import get_global_config
  import torch

  config = get_global_config()  # noqa
  aux_loss_layers = config.typed_value("aux_loss_layers")
  aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
  aed_loss_scale = config.float("aed_loss_scale", 1.0)
  use_normalized_loss = config.bool("use_normalized_loss", True)

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio

  batch_dims = data.remaining_dims(data_spatial_dim)

  def _get_segment_starts_and_lens(out_spatial_dim: Dim):
    non_blank_mask = _get_non_blank_mask(align_targets, model.blank_idx)
    targets_range = rf.range_over_dim(align_targets_spatial_dim, dtype="int32")
    targets_range = rf.expand_dim(targets_range, batch_dims[0])
    non_blank_positions, _ = _get_masked(
      targets_range, non_blank_mask, align_targets_spatial_dim, batch_dims, out_spatial_dim
    )
    starts = rf.maximum(
      rf.convert_to_tensor(0, dtype="int32"), non_blank_positions - model.center_window_size // 2)
    ends = rf.minimum(
      rf.copy_to_device(align_targets_spatial_dim.get_size_tensor() - 1, non_blank_positions.device),
      non_blank_positions + model.center_window_size // 2
    )
    lens = ends - starts + 1

    return starts, lens

  def _get_emit_ground_truth():
    non_blank_mask = _get_non_blank_mask(align_targets, model.blank_idx)
    result = rf.where(non_blank_mask, rf.convert_to_tensor(1), rf.convert_to_tensor(0))
    sparse_dim = Dim(name="emit_ground_truth", dimension=2)
    # result = rf.expand_dim(result, sparse_dim)
    result.sparse_dim = sparse_dim
    torch.set_printoptions(threshold=10000)

    return result, sparse_dim

  non_blank_targets, non_blank_targets_spatial_dim = _get_masked(
    align_targets, _get_non_blank_mask(align_targets, model.blank_idx), align_targets_spatial_dim, batch_dims
  )
  non_blank_targets.sparse_dim = model.target_dim
  segment_starts, segment_lens = _get_segment_starts_and_lens(non_blank_targets_spatial_dim)

  # ------------------- encoder aux loss -------------------

  collected_outputs = {}
  enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
  if aux_loss_layers:
    for i, layer_idx in enumerate(aux_loss_layers):
      if layer_idx > len(model.encoder.layers):
        continue
      linear = getattr(model, f"enc_aux_logits_{layer_idx}")
      aux_logits = linear(collected_outputs[str(layer_idx - 1)])
      aux_loss = rf.ctc_loss(
        logits=aux_logits,
        targets=non_blank_targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=non_blank_targets_spatial_dim,
        blank_index=model.blank_idx,
      )
      aux_loss.mark_as_loss(
        f"ctc_{layer_idx}",
        scale=aux_loss_scales[i],
        custom_inv_norm_factor=align_targets_spatial_dim.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
      )

  non_blank_input_embeddings = model.target_embed(non_blank_targets)
  non_blank_input_embeddings = rf.shift_right(
    non_blank_input_embeddings, axis=non_blank_targets_spatial_dim, pad_value=0.0)

  align_input_embeddings = model.target_embed_length_model(align_targets)
  align_input_embeddings = rf.shift_right(
    align_input_embeddings, axis=align_targets_spatial_dim, pad_value=0.0)

  # ------------------- label loop -------------------

  def _label_loop_body(xs, state: rf.State):
    new_state = rf.State()
    loop_out_, new_state.decoder = model.label_sync_loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=xs["input_embed"],
      segment_starts=xs["segment_starts"],
      segment_lens=xs["segment_lens"],
      state=state.decoder,
    )
    return loop_out_, new_state

  label_loop_out, _, _ = rf.scan(
    spatial_dim=non_blank_targets_spatial_dim,
    xs={
      "input_embed": non_blank_input_embeddings,
      "segment_starts": segment_starts,
      "segment_lens": segment_lens,
    },
    ys=model.label_loop_step_output_templates(batch_dims=batch_dims),
    initial=rf.State(
      decoder=model.label_decoder_default_initial_state(
        batch_dims=batch_dims,
        # TODO: do we need these sparse dims? they are automatically added by rf.range_over_dim
        segment_starts_sparse_dim=segment_starts.sparse_dim,
        segment_lens_sparse_dim=segment_lens.sparse_dim,
      ),
    ),
    body=_label_loop_body,
  )

  logits = model.decode_label_logits(input_embed=non_blank_input_embeddings, **label_loop_out)
  logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [non_blank_targets_spatial_dim], enforce_sorted=False)
  non_blank_targets_packed, _ = rf.pack_padded(
    non_blank_targets, dims=batch_dims + [non_blank_targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
  )

  log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
  log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
  loss = rf.cross_entropy(
    target=non_blank_targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
  )
  loss.mark_as_loss("non_blank_ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

  best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
  frame_error = best != non_blank_targets_packed
  frame_error.mark_as_loss(name="non_blank_fer", as_error=True)

  # ------------------- blank loop -------------------

  def _blank_loop_body(xs, state: rf.State):
    new_state = rf.State()
    loop_out_, new_state.decoder = model.time_sync_loop_step(
      enc=enc_args["enc"],
      enc_spatial_dim=enc_spatial_dim,
      input_embed=xs["input_embed"],
      state=state.decoder,
    )
    return loop_out_, new_state

  label_loop_out, _, _ = rf.scan(
    spatial_dim=align_targets_spatial_dim,
    xs={
      "input_embed": align_input_embeddings,
    },
    ys=model.blank_loop_step_output_templates(batch_dims=batch_dims),
    initial=rf.State(
      decoder=model.blank_decoder_default_initial_state(
        batch_dims=batch_dims,
      ),
    ),
    body=_blank_loop_body,
  )

  blank_logits = model.decode_blank_logits(**label_loop_out)
  blank_logits_packed, pack_dim = rf.pack_padded(blank_logits, dims=batch_dims + [align_targets_spatial_dim], enforce_sorted=False)
  emit_ground_truth, emit_blank_target_dim = _get_emit_ground_truth()
  emit_ground_truth_packed, _ = rf.pack_padded(
    emit_ground_truth, dims=batch_dims + [align_targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
  )

  # rf.log_sigmoid not implemented for torch backend
  emit_log_prob = rf.log(rf.sigmoid(blank_logits_packed))
  blank_log_prob = rf.log(rf.sigmoid(-blank_logits_packed))
  blank_logit_dim = blank_logits_packed.remaining_dims((pack_dim,))[0]
  emit_blank_log_prob, _ = rf.concat(
    (blank_log_prob, blank_logit_dim), (emit_log_prob, blank_logit_dim), out_dim=emit_blank_target_dim)
  blank_loss = rf.cross_entropy(
    target=emit_ground_truth_packed,
    estimated=emit_blank_log_prob,
    estimated_type="log-probs",
    axis=emit_blank_target_dim
  )
  blank_loss.mark_as_loss("emit_blank_ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

  best = rf.reduce_argmax(emit_blank_log_prob, axis=emit_blank_target_dim)
  frame_error = best != emit_ground_truth_packed
  frame_error.mark_as_loss(name="emit_blank_fer", as_error=True)


from_scratch_training: TrainDef[SegmentalAttentionModel]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


def model_recog(
        *,
        model: SegmentalAttentionModel,
        data: Tensor,
        data_spatial_dim: Dim,
        max_seq_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
  """
  Function is run within RETURNN.

  Earlier we used the generic beam_search function,
  but now we just directly perform the search here,
  as this is overall simpler and shorter.

  :return:
      recog results including beam {batch, beam, out_spatial},
      log probs {batch, beam},
      out_spatial_dim,
      final beam_dim
  """
  assert not model.language_model  # not implemented here. use the pure PyTorch search instead

  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
  beam_size = 12
  if max_seq_len is None:
    max_seq_len = enc_spatial_dim.get_size_tensor()
  else:
    max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
  print("** max seq len:", max_seq_len.raw_tensor)
  max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)

  # Eager-mode implementation of beam search.
  # Initial state.
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims
  label_decoder_state = model.label_decoder_default_initial_state(batch_dims=batch_dims_,)

  blank_decoder_state = model.blank_decoder_default_initial_state(batch_dims=batch_dims_)
  bos_idx = 0
  target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.align_target_dim)
  target_non_blank = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
  # ended = rf.constant(False, dims=batch_dims_)
  seq_log_prob = rf.constant(0.0, dims=batch_dims_)

  i = 0
  seq_targets = []
  seq_backrefs = []
  while i < max_seq_len.raw_tensor:
    if i == 0:
      input_embed = rf.zeros(batch_dims_ + [model.target_embed.out_dim], feature_dim=model.target_embed.out_dim, dtype="float32")
      input_embed_length_model = rf.zeros(
        batch_dims_ + [model.target_embed_length_model.out_dim], feature_dim=model.target_embed_length_model.out_dim)
    else:
      input_embed_length_model = model.target_embed_length_model(target)

    # ------------------- label step -------------------
    center_position = rf.minimum(
      rf.full(dims=[beam_dim] + batch_dims, fill_value=i, dtype="int32"),
      rf.copy_to_device(data_spatial_dim.get_size_tensor() - 1, data.device)
    )
    segment_starts = rf.maximum(
      rf.convert_to_tensor(0, dtype="int32"), center_position - model.center_window_size // 2)
    segment_ends = rf.minimum(
      rf.copy_to_device(data_spatial_dim.get_size_tensor() - 1, data.device),
      center_position + model.center_window_size // 2
    )
    segment_lens = segment_ends - segment_starts + 1
    label_step_out, label_decoder_state_updated = model.label_sync_loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed,
      segment_lens=segment_lens,
      segment_starts=segment_starts,
      state=label_decoder_state,
    )
    label_logits = model.decode_label_logits(input_embed=input_embed, **label_step_out)
    label_log_prob = rf.log_softmax(label_logits, axis=model.target_dim)

    # ------------------- blank step -------------------

    blank_step_out, blank_decoder_state = model.time_sync_loop_step(
      enc=enc_args["enc"],
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed_length_model,
      state=blank_decoder_state,
    )
    blank_logits = model.decode_blank_logits(**blank_step_out)
    emit_log_prob = rf.log(rf.sigmoid(blank_logits))
    emit_log_prob = rf.squeeze(emit_log_prob, axis=emit_log_prob.feature_dim)
    blank_log_prob = rf.log(rf.sigmoid(-blank_logits))

    # combine blank and label probs
    label_log_prob += emit_log_prob
    output_log_prob, _ = rf.concat(
      (label_log_prob, model.target_dim), (blank_log_prob, blank_log_prob.feature_dim),
      out_dim=model.align_target_dim
    )

    # top-k
    seq_log_prob = seq_log_prob + output_log_prob  # Batch, InBeam, Vocab
    old_beam_dim = beam_dim.copy()
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, model.align_target_dim]
    )  # seq_log_prob, backrefs, target: Batch, Beam
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    update_state_mask = rf.convert_to_tensor(target != model.blank_idx)

    def _get_masked_state(old, new, mask):
      old = rf.gather(old, indices=backrefs, axis=old_beam_dim)
      new = rf.gather(new, indices=backrefs, axis=old_beam_dim)
      return rf.where(mask, new, old)

    for key, old_state in label_decoder_state.items():
      if key == "s":
        for s_key, s_val in old_state.items():
          label_decoder_state[key][s_key] = _get_masked_state(
            s_val, label_decoder_state_updated[key][s_key], update_state_mask)
      else:
        label_decoder_state[key] = _get_masked_state(old_state, label_decoder_state_updated[key], update_state_mask)
        # old_state = rf.gather(old_state, indices=backrefs, axis=old_beam_dim)
        # new_state = rf.gather(label_decoder_state_updated[key], indices=backrefs, axis=old_beam_dim)
        # label_decoder_state[key] = rf.where(update_state_mask, new_state, old_state)
    target_non_blank = rf.where(update_state_mask, target, rf.convert_to_tensor(0, dtype="int32"))
    target_non_blank.sparse_dim = model.target_embed.in_dim
    input_embed = rf.where(
      update_state_mask,
      model.target_embed(target_non_blank),
      rf.gather(input_embed, indices=backrefs, axis=old_beam_dim)
    )

    blank_decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), blank_decoder_state)

    i += 1

  # Backtrack via backrefs, resolve beams.
  seq_targets_ = []
  indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
  for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
    # indices: FinalBeam -> Beam
    # backrefs: Beam -> PrevBeam
    seq_targets_.insert(0, rf.gather(target, indices=indices))
    indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

  seq_targets__ = TensorArray(seq_targets_[0])
  for target in seq_targets_:
    seq_targets__ = seq_targets__.push_back(target)
  seq_targets = seq_targets__.stack(axis=enc_spatial_dim)

  non_blank_targets, non_blank_targets_spatial_dim = _get_masked(
    seq_targets,
    _get_non_blank_mask(seq_targets, model.blank_idx),
    enc_spatial_dim,
    [beam_dim] + batch_dims,
  )
  non_blank_targets.sparse_dim = model.target_dim

  return non_blank_targets, seq_log_prob, non_blank_targets_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[SegmentalAttentionModel]
model_recog.output_with_beam = True
# output_blank_label=blank is actually wrong for AED, but now we don't change it anymore
# because it would change all recog hashes.
# Also, it does not matter too much -- it will just cause an extra SearchRemoveLabelJob,
# which will not have any effect here.
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False
