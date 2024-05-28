from typing import Optional, Dict, Any, Sequence, Tuple, List

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder,
  SegmentalAttEfficientLabelDecoder
)

from returnn.tensor import Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef


def _calc_ce_loss_and_fer(
        logits: rf.Tensor,
        targets: rf.Tensor,
        batch_dims: List[Dim],
        targets_spatial_dim: Dim,
        target_dim: Dim,
):
  logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False)
  non_blank_targets_packed, _ = rf.pack_padded(
    targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
  )

  log_prob = rf.log_softmax(logits_packed, axis=target_dim)
  log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=target_dim)
  loss = rf.cross_entropy(
    target=non_blank_targets_packed, estimated=log_prob, estimated_type="log-probs", axis=target_dim
  )
  loss.mark_as_loss("non_blank_ce", scale=1.0, use_normalized_loss=True)

  best = rf.reduce_argmax(logits_packed, axis=target_dim)
  frame_error = best != non_blank_targets_packed
  frame_error.mark_as_loss(name="non_blank_fer", as_error=True)


def viterbi_training(
        *,
        model: SegmentalAttLabelDecoder,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,
        segment_lens: rf.Tensor,
        batch_dims: List[Dim],
        return_label_model_states: bool = False,
) -> Optional[Tuple[rf.Tensor, Dim]]:
  non_blank_input_embeddings = model.target_embed(non_blank_targets)
  non_blank_input_embeddings_shifted = rf.shift_right(
    non_blank_input_embeddings, axis=non_blank_targets_spatial_dim, pad_value=0.0)

  # ------------------- label loop -------------------

  def _label_loop_body(xs, state: rf.State):
    new_state = rf.State()
    loop_out_, new_state.decoder = model.loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=xs["input_embed"],
      segment_starts=xs["segment_starts"],
      segment_lens=xs["segment_lens"],
      state=state.decoder,
    )
    return loop_out_, new_state

  label_loop_out, final_state, _ = rf.scan(
    spatial_dim=non_blank_targets_spatial_dim,
    xs={
      "input_embed": non_blank_input_embeddings_shifted,
      "segment_starts": segment_starts,
      "segment_lens": segment_lens,
    },
    ys=model.loop_step_output_templates(batch_dims=batch_dims),
    initial=rf.State(
      decoder=model.default_initial_state(
        batch_dims=batch_dims,
        # TODO: do we need these sparse dims? they are automatically added by rf.range_over_dim
        segment_starts_sparse_dim=segment_starts.sparse_dim,
        segment_lens_sparse_dim=segment_lens.sparse_dim,
      ),
    ),
    body=_label_loop_body,
  )

  logits = model.decode_logits(input_embed=non_blank_input_embeddings_shifted, **label_loop_out)
  _calc_ce_loss_and_fer(logits, non_blank_targets, batch_dims, non_blank_targets_spatial_dim, model.target_dim)

  if return_label_model_states:
    # need to run the loop one more time to get the last output (which is not needed for the loss computation)
    last_embedding = rf.gather(
        non_blank_input_embeddings,
        axis=non_blank_targets_spatial_dim,
        indices=rf.copy_to_device(
          non_blank_targets_spatial_dim.get_size_tensor() - 1, non_blank_input_embeddings.device)
    )
    last_loop_out, _ = model.loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=last_embedding,
      segment_starts=final_state.decoder.segment_starts,
      segment_lens=final_state.decoder.segment_lens,
      state=final_state.decoder,
    )
    return rf.concat(
      (label_loop_out["s"], non_blank_targets_spatial_dim),
      (rf.expand_dim(last_loop_out["s"], single_step_dim), single_step_dim),
    )

  return None


def viterbi_training_efficient(
        *,
        model: SegmentalAttEfficientLabelDecoder,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        targets: rf.Tensor,
        targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,
        segment_lens: rf.Tensor,
        batch_dims: List[Dim],
        ce_targets: rf.Tensor,
        ce_spatial_dim: Dim,
        non_blank_mask: Optional[rf.Tensor] = None,
        non_blank_mask_spatial_dim: Optional[Dim] = None,
        return_label_model_states: bool = False,
) -> Optional[Tuple[rf.Tensor, Dim]]:
  input_embeddings = model.target_embed(targets)
  input_embeddings_shifted = rf.shift_right(
    input_embeddings, axis=targets_spatial_dim, pad_value=0.0)

  label_lstm_out, final_state = model.s_wo_att(
    input_embeddings_shifted,
    state=model.s_wo_att.default_initial_state(batch_dims=batch_dims),
    spatial_dim=targets_spatial_dim,
  )

  if non_blank_mask is not None:
    label_lstm_out = utils.get_unmasked(
      input=label_lstm_out,
      input_spatial_dim=targets_spatial_dim,
      mask=non_blank_mask,
      mask_spatial_dim=non_blank_mask_spatial_dim,
    )
    input_embeddings_shifted = utils.get_unmasked(
      input=input_embeddings_shifted,
      input_spatial_dim=targets_spatial_dim,
      mask=non_blank_mask,
      mask_spatial_dim=non_blank_mask_spatial_dim,
    )

  # need to move size tensor to GPU since otherwise there is an error in some merge_dims call inside rf.gather
  # because two tensors have different devices
  # TODO: fix properly in the gather implementation
  targets_spatial_dim.dyn_size_ext = rf.copy_to_device(targets_spatial_dim.dyn_size_ext, label_lstm_out.device)
  if non_blank_mask_spatial_dim is not None:
    non_blank_mask_spatial_dim.dyn_size_ext = rf.copy_to_device(non_blank_mask_spatial_dim.dyn_size_ext, label_lstm_out.device)
  att = model(
    enc=enc_args["enc"],
    enc_ctx=enc_args["enc_ctx"],
    enc_spatial_dim=enc_spatial_dim,
    s=label_lstm_out,
    segment_starts=segment_starts,
    segment_lens=segment_lens,
  )
  targets_spatial_dim.dyn_size_ext = rf.copy_to_device(targets_spatial_dim.dyn_size_ext, "cpu")
  if non_blank_mask_spatial_dim is not None:
    non_blank_mask_spatial_dim.dyn_size_ext = rf.copy_to_device(non_blank_mask_spatial_dim.dyn_size_ext, "cpu")

  logits = model.decode_logits(
    input_embed=input_embeddings_shifted,
    att=att,
    s=label_lstm_out,
  )

  _calc_ce_loss_and_fer(logits, ce_targets, batch_dims, ce_spatial_dim, model.target_dim)

  if return_label_model_states:
    # need to run the lstm one more time to get the last output (which is not needed for the loss computation)
    last_embedding = rf.gather(
        input_embeddings,
        axis=targets_spatial_dim,
        indices=rf.copy_to_device(
          targets_spatial_dim.get_size_tensor() - 1, input_embeddings.device),
        clip_to_valid=True,
    )
    last_lstm_out, _ = model.s_wo_att(
      last_embedding,
      state=final_state,
      spatial_dim=single_step_dim,
    )
    return rf.concat(
      (label_lstm_out, targets_spatial_dim),
      (rf.expand_dim(last_lstm_out, single_step_dim), single_step_dim),
    )

  return None


def full_sum_training(
        *,
        model: SegmentalAttEfficientLabelDecoder,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,
        segment_lens: rf.Tensor,
        batch_dims: List[Dim],
) -> Optional[Dict[str, Tuple[rf.Tensor, Dim]]]:
  # print("full_sum_training")
  # print("model", model)

  import torch
  from torch.profiler import profile, record_function, ProfilerActivity

  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:

    non_blank_input_embeddings = model.target_embed(non_blank_targets)  # [B, S, D]
    singleton_dim = Dim(name="singleton", dimension=1)
    singleton_zeros = rf.zeros(batch_dims + [singleton_dim, model.target_embed.out_dim])
    non_blank_input_embeddings_shifted, non_blank_targets_spatial_dim_ext = rf.concat(
      (singleton_zeros, singleton_dim),
      (non_blank_input_embeddings, non_blank_targets_spatial_dim),
      allow_broadcast=True
    )  # [B, S+1, D]
    non_blank_input_embeddings_shifted.feature_dim = non_blank_input_embeddings.feature_dim

    label_lstm_out, _ = model.s_wo_att(
      non_blank_input_embeddings_shifted,
      state=model.s_wo_att.default_initial_state(batch_dims=batch_dims),
      spatial_dim=non_blank_targets_spatial_dim_ext,
    )  # [B, S+1, D]

    att = model(
      enc=enc_args["enc"],
      enc_ctx=enc_args["enc_ctx"],
      enc_spatial_dim=enc_spatial_dim,
      s=label_lstm_out,
      segment_starts=segment_starts,
      segment_lens=segment_lens,
    )  # [B, S+1, T, D]

    logits = model.decode_logits(
      input_embed=non_blank_input_embeddings_shifted,
      att=att,
      s=label_lstm_out,
    )  # [B, S+1, T, D]

    print("logits", logits.raw_tensor.shape)

    logits_packed, pack_dim = rf.pack_padded(
      logits,
      dims=batch_dims + [enc_spatial_dim, non_blank_targets_spatial_dim_ext],
      enforce_sorted=False
    )  # [B * T * (S+1), D]

    print("logits_packed", logits_packed.raw_tensor.shape)

  print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_memory_usage", row_limit=10))
  exit()

  from returnn.extern_private.BergerMonotonicRNNT.monotonic_rnnt.pytorch_binding import monotonic_rnnt_loss

  loss = monotonic_rnnt_loss(
    acts=logits_packed.raw_tensor,
    labels=non_blank_targets.copy_transpose(batch_dims + [non_blank_targets_spatial_dim]).raw_tensor,
    input_lengths=rf.copy_to_device(enc_spatial_dim.dyn_size_ext, logits.device).raw_tensor,
    label_lengths=rf.copy_to_device(non_blank_targets_spatial_dim.dyn_size_ext, logits.device).raw_tensor.int(),
    blank_label=model.blank_idx,
  )

  # print("loss", loss.shape)

  exit()

  loss = rf.convert_to_tensor(loss, name="full_sum_loss")
  loss.mark_as_loss("full_sum_loss", scale=1.0, use_normalized_loss=True)

  return None
