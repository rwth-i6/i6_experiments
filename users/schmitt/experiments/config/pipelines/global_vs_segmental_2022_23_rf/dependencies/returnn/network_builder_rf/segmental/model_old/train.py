from typing import Dict, List

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import FramewiseTrainDef

from returnn.tensor import TensorDict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.utils import get_non_blank_mask, get_masked
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_old.model import SegmentalAttentionModel

from returnn.tensor import Dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef


def _returnn_v2_train_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  data = extern_data[default_input_key]
  data_spatial_dim = data.get_time_dim_tag()
  targets = extern_data[default_target_key]
  targets_spatial_dim = targets.get_time_dim_tag()
  train_def: FramewiseTrainDef = config.typed_value("_train_def")
  train_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    align_targets=targets,
    align_targets_spatial_dim=targets_spatial_dim,
  )


def get_blank_loop_out_unoptimized(
        model: SegmentalAttentionModel,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        align_targets_spatial_dim: Dim,
        align_input_embeddings: rf.Tensor,
        batch_dims: List[Dim]
):
  def _blank_loop_body(xs, state: rf.State):
    new_state = rf.State()
    loop_out_, new_state.decoder = model.time_sync_loop_step(
      enc=enc_args["enc"],
      enc_spatial_dim=enc_spatial_dim,
      input_embed=xs["input_embed"],
      state=state.decoder,
    )
    return loop_out_, new_state

  blank_loop_out, _, _ = rf.scan(
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
  return blank_loop_out


def get_blank_loop_out_optimized(
        model: SegmentalAttentionModel,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        align_input_embeddings: rf.Tensor,
        batch_dims: List[Dim],
        align_targets_spatial_dim: Dim
):
  blank_loop_out, _ = model.time_sync_loop_step(
    enc=enc_args["enc"],
    enc_spatial_dim=enc_spatial_dim,
    input_embed=align_input_embeddings,
    state=model.blank_decoder_default_initial_state(batch_dims=batch_dims,),
    spatial_dim=align_targets_spatial_dim,
  )
  return blank_loop_out


def from_scratch_training(
        *,
        model: SegmentalAttentionModel,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        align_targets: rf.Tensor,
        align_targets_spatial_dim: Dim
):
  """
  Here
  """
  from returnn.config import get_global_config
  import torch

  config = get_global_config()  # noqa
  aux_loss_layers = config.typed_value("aux_loss_layers")
  aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
  aed_loss_scale = config.float("aed_loss_scale", 1.0)
  use_normalized_loss = config.bool("use_normalized_loss", True)
  use_optimized_length_model_loop = config.bool("use_optimized_length_model_loop", True)

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio

  batch_dims = data.remaining_dims(data_spatial_dim)

  def _get_segment_starts_and_lens(out_spatial_dim: Dim):
    non_blank_mask = get_non_blank_mask(align_targets, model.blank_idx)
    targets_range = rf.range_over_dim(align_targets_spatial_dim, dtype="int32")
    targets_range = rf.expand_dim(targets_range, batch_dims[0])
    non_blank_positions, _ = get_masked(
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
    non_blank_mask = get_non_blank_mask(align_targets, model.blank_idx)
    result = rf.where(non_blank_mask, rf.convert_to_tensor(1), rf.convert_to_tensor(0))
    sparse_dim = Dim(name="emit_ground_truth", dimension=2)
    # result = rf.expand_dim(result, sparse_dim)
    result.sparse_dim = sparse_dim
    torch.set_printoptions(threshold=10000)

    return result, sparse_dim

  non_blank_targets, non_blank_targets_spatial_dim = get_masked(
    align_targets, get_non_blank_mask(align_targets, model.blank_idx), align_targets_spatial_dim, batch_dims
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
  # deactivate to compare to old returnn training (label smoothing is done differently)
  # log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
  loss = rf.cross_entropy(
    target=non_blank_targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
  )
  loss.mark_as_loss("non_blank_ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

  best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
  frame_error = best != non_blank_targets_packed
  frame_error.mark_as_loss(name="non_blank_fer", as_error=True)

  # ------------------- blank loop -------------------

  if use_optimized_length_model_loop:
    blank_loop_out = get_blank_loop_out_optimized(
      model=model,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      align_targets_spatial_dim=align_targets_spatial_dim,
      align_input_embeddings=align_input_embeddings,
      batch_dims=batch_dims
    )
  else:
    blank_loop_out = get_blank_loop_out_unoptimized(
      model=model,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      align_targets_spatial_dim=align_targets_spatial_dim,
      align_input_embeddings=align_input_embeddings,
      batch_dims=batch_dims
    )

  blank_logits = model.decode_blank_logits(**blank_loop_out)
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
