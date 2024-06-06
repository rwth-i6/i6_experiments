from typing import Optional, Dict, Any, Sequence, Tuple, List
import tree

import torch

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import recombination
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder,
  SegmentalAttEfficientLabelDecoder
)

from returnn.tensor import Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

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
    singleton_dim = Dim(name="singleton", dimension=1)
    return rf.concat(
      (label_loop_out["s"], non_blank_targets_spatial_dim),
      (rf.expand_dim(last_loop_out["s"], singleton_dim), singleton_dim),
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
    singleton_dim = Dim(name="singleton", dimension=1)
    last_lstm_out, _ = model.s_wo_att(
      last_embedding,
      state=final_state,
      spatial_dim=singleton_dim,
    )
    return rf.concat(
      (label_lstm_out, targets_spatial_dim),
      (rf.expand_dim(last_lstm_out, singleton_dim), singleton_dim),
    )

  return None


def full_sum_training(
        *,
        model: SegmentalAttEfficientLabelDecoder,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,  # [B, S, V]
        non_blank_targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,  # [B, T]
        segment_lens: rf.Tensor,  # [B, T]
        batch_dims: List[Dim],
) -> Optional[Dict[str, Tuple[rf.Tensor, Dim]]]:
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


  logits_packed, pack_dim = rf.pack_padded(
    logits,
    dims=batch_dims + [enc_spatial_dim, non_blank_targets_spatial_dim_ext],
    enforce_sorted=False
  )  # [B * T * (S+1), D]

  from returnn.extern_private.BergerMonotonicRNNT.monotonic_rnnt.pytorch_binding import monotonic_rnnt_loss

  loss = monotonic_rnnt_loss(
    acts=logits_packed.raw_tensor,
    labels=non_blank_targets.copy_transpose(batch_dims + [non_blank_targets_spatial_dim]).raw_tensor,
    input_lengths=rf.copy_to_device(enc_spatial_dim.dyn_size_ext, logits.device).raw_tensor,
    label_lengths=rf.copy_to_device(non_blank_targets_spatial_dim.dyn_size_ext, logits.device).raw_tensor.int(),
    blank_label=model.blank_idx,
  )

  loss = rf.convert_to_tensor(loss, name="full_sum_loss")
  loss.mark_as_loss("full_sum_loss", scale=1.0, use_normalized_loss=True)

  return None


def full_sum_training_w_beam(
        *,
        model: SegmentalAttEfficientLabelDecoder,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,  # [B, S, V]
        non_blank_targets_spatial_dim: Dim,
        # segment_starts: rf.Tensor,  # [B, T]
        # segment_lens: rf.Tensor,  # [B, T]
        batch_dims: List[Dim],
        beam_size: int,
) -> Optional[Dict[str, Tuple[rf.Tensor, Dim]]]:
  assert len(batch_dims) == 1, "not supported yet"

  non_blank_input_embeddings = model.target_embed(non_blank_targets)  # [B, S, D]
  non_blank_input_embeddings, non_blank_targets_padded_spatial_dim = rf.pad(
    non_blank_input_embeddings,
    axes=[non_blank_targets_spatial_dim],
    padding=[(1, 0)],
    value=0.0,
  )
  non_blank_targets_padded_spatial_dim = non_blank_targets_padded_spatial_dim[0]

  # add blank idx on the right
  # this way, when the label index for gathering reached the last non-blank index, it will gather blank after that
  # which then only allows corresponding hypotheses to be extended by blank
  non_blank_targets_padded, _ = rf.pad(
    non_blank_targets,
    axes=[non_blank_targets_spatial_dim],
    padding=[(0, 1)],
    value=model.blank_idx,
    out_dims=[non_blank_targets_padded_spatial_dim]
  )

  non_blank_targets_padded_sizes = rf.copy_to_device(
    non_blank_targets_padded_spatial_dim.dyn_size_ext, non_blank_targets.device
  )
  non_blank_targets_spatial_sizes = rf.copy_to_device(
    non_blank_targets_spatial_dim.dyn_size_ext, non_blank_targets.device)
  enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext, non_blank_targets.device)

  linear_label_positions = enc_spatial_sizes / non_blank_targets_spatial_sizes
  linear_label_positions = linear_label_positions * rf.range_over_dim(non_blank_targets_spatial_dim)
  # print("linear_label_positions", linear_label_positions.raw_tensor)
  # exit()

  # print("non_blank_targets_padded", non_blank_targets_padded.raw_tensor)

  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims
  bos_idx = 0
  seq_log_prob = rf.constant(0.0, dims=batch_dims_)

  max_seq_len = enc_spatial_dim.get_size_tensor()
  max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)

  label_lstm_state = model.s_wo_att.default_initial_state(batch_dims=batch_dims_)
  target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
  # target_non_blank = target.copy()
  update_state_mask = rf.convert_to_tensor(target != model.blank_idx)
  label_indices = rf.zeros(batch_dims_, dtype="int64")

  # input_embed = rf.zeros(
  #   batch_dims_ + [model.target_embed.out_dim],
  #   feature_dim=model.target_embed.out_dim,
  #   dtype="float32"
  # )

  vocab_range = rf.range_over_dim(model.target_dim)
  blank_tensor = rf.convert_to_tensor(model.blank_idx, dtype=vocab_range.dtype)
  log_lambda = rf.log(rf.convert_to_tensor(0.004)) * rf.ones([model.target_dim], dtype="float32")
  # lambda_ = rf.shift_right(lambda_, axis=model.target_dim, pad_value=0.0)
  log_lambda = rf.where(
    vocab_range == model.blank_idx,
    rf.constant(0.0, dims=[model.target_dim], dtype="float32"),
    log_lambda
  )

  old_beam_dim = beam_dim.copy()
  backrefs = rf.zeros(batch_dims_, dtype="int32")

  i = 0
  seq_targets = []
  seq_backrefs = []
  while i < max_seq_len.raw_tensor:
    if i > 0:
      # target_non_blank = rf.where(update_state_mask, target, rf.gather(target_non_blank, indices=backrefs))
      # input_embed = rf.where(
      #   update_state_mask,
      #   model.target_embed(target_non_blank),
      #   rf.gather(input_embed, indices=backrefs)
      # )
      prev_label_indices = rf.gather(label_indices, indices=backrefs)
      label_indices = rf.where(
        update_state_mask,
        rf.where(
          prev_label_indices == non_blank_targets_padded_sizes - 1,
          prev_label_indices,
          prev_label_indices + 1
        ),
        prev_label_indices
      )

    ground_truth = rf.gather(
      non_blank_targets_padded,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )
    input_embed = rf.gather(
      non_blank_input_embeddings,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )

    label_lstm_out, label_lstm_state_updated = model.s_wo_att(
      input_embed,
      state=label_lstm_state,
      spatial_dim=single_step_dim,
    )

    center_position = rf.minimum(
      rf.full(dims=[beam_dim] + batch_dims, fill_value=i, dtype="int32"),
      rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, label_lstm_out.device)
    )
    segment_starts = rf.maximum(
      rf.convert_to_tensor(0, dtype="int32"), center_position - model.center_window_size // 2)
    segment_ends = rf.minimum(
      rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, label_lstm_out.device),
      center_position + model.center_window_size // 2
    )
    segment_lens = segment_ends - segment_starts + 1

    att = model(
      enc=enc_args["enc"],
      enc_ctx=enc_args["enc_ctx"],
      enc_spatial_dim=enc_spatial_dim,
      s=label_lstm_out,
      segment_starts=segment_starts,
      segment_lens=segment_lens,
    )  # [B, S+1, T, D]
    # print("att", att)

    logits = model.decode_logits(
      input_embed=input_embed,
      att=att,
      s=label_lstm_out,
    )  # [B, S+1, T, D]
    # print("logits", logits)

    label_log_prob = rf.log_softmax(logits, axis=model.target_dim)

    def custom_backward(grad):
      grad[:, :, 0] *= 0.001
      return grad

    if rf.get_run_ctx().train_flag:
      label_log_prob.raw_tensor.register_hook(custom_backward)

    # log prob needs to correspond to the next non-blank label...
    log_prob_mask = vocab_range == ground_truth
    rem_frames = enc_spatial_sizes - i
    rem_labels = non_blank_targets_spatial_sizes - label_indices
    # ... or to blank if there are more frames than labels left
    log_prob_mask = rf.logical_or(
      log_prob_mask,
      rf.logical_and(
        vocab_range == blank_tensor,
        rem_frames > rem_labels
      )
    )
    label_log_prob = rf.where(
      log_prob_mask,
      label_log_prob,
      rf.constant(-float("inf"), dims=batch_dims + [beam_dim, model.target_dim])
    )

    seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
    old_beam_dim = beam_dim.copy()
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob,
      k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
      axis=[beam_dim, model.target_dim]
    )  # seq_log_prob, backrefs, target: Batch, Beam
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    update_state_mask = rf.logical_and(
      rf.convert_to_tensor(target != model.blank_idx),
      seq_log_prob != rf.convert_to_tensor(-float("inf"), dtype="float32")
    )

    def _get_masked_state(old, new, mask):
      old = rf.gather(old, indices=backrefs, axis=old_beam_dim)
      new = rf.gather(new, indices=backrefs, axis=old_beam_dim)
      return rf.where(mask, new, old)

    label_lstm_state = tree.map_structure(
      lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
      label_lstm_state, label_lstm_state_updated
    )

    i += 1

  # # Backtrack via backrefs, resolve beams.
  # seq_targets_ = []
  # indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
  # for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
  #   # indices: FinalBeam -> Beam
  #   # backrefs: Beam -> PrevBeam
  #   seq_targets_.insert(0, rf.gather(target, indices=indices))
  #   indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam
  #
  # seq_targets__ = TensorArray(seq_targets_[0])
  # for target in seq_targets_:
  #   seq_targets__ = seq_targets__.push_back(target)
  # seq_targets = seq_targets__.stack(axis=enc_spatial_dim)

  # torch.set_printoptions(threshold=10_000)
  # print("seq_targets", seq_targets.copy_transpose(batch_dims + [beam_dim, enc_spatial_dim]).raw_tensor[0, 0])
  # print("seq_log_prob", seq_log_prob.raw_tensor[0])

  # calculate full-sum loss using the log-sum-exp trick
  max_log_prob = rf.reduce_max(seq_log_prob, axis=beam_dim)
  loss = -1 * (max_log_prob + rf.log(rf.reduce_sum(rf.exp(seq_log_prob - max_log_prob), axis=beam_dim)))

  # loss = -rf.log(rf.reduce_sum(rf.exp(seq_log_prob), axis=beam_dim))
  loss.mark_as_loss("full_sum_loss", scale=1.0, use_normalized_loss=True)

  return None


def full_sum_training_w_beam_eff(
        *,
        model: SegmentalAttEfficientLabelDecoder,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,  # [B, S, V]
        non_blank_targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,  # [B, T]
        segment_lens: rf.Tensor,  # [B, T]
        batch_dims: List[Dim],
        beam_size: int,
) -> Optional[Dict[str, Tuple[rf.Tensor, Dim]]]:
  assert len(batch_dims) == 1, "not supported yet"
  assert model.blank_idx == 0, "blank idx needs to be zero because of the way the gradient is scaled"

  # ------------------------ init some variables ------------------------
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims
  bos_idx = 0
  seq_log_prob = rf.constant(0.0, dims=batch_dims_)
  max_seq_len = enc_spatial_dim.get_size_tensor()
  max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)
  label_lstm_state = model.s_wo_att.default_initial_state(batch_dims=batch_dims)
  target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
  update_state_mask = rf.convert_to_tensor(target != model.blank_idx)
  label_indices = rf.zeros(batch_dims_, dtype="int32")
  vocab_range = rf.range_over_dim(model.target_dim)
  blank_tensor = rf.convert_to_tensor(model.blank_idx, dtype=vocab_range.dtype)
  backrefs = rf.zeros(batch_dims_, dtype="int32")
  seq_hash = rf.constant(0, dims=batch_dims_, dtype="int64")

  # ------------------------ targets/embeddings ------------------------

  non_blank_input_embeddings = model.target_embed(non_blank_targets)  # [B, S, D]
  non_blank_input_embeddings, non_blank_targets_padded_spatial_dim = rf.pad(
    non_blank_input_embeddings,
    axes=[non_blank_targets_spatial_dim],
    padding=[(1, 0)],
    value=0.0,
  )  # [B, S+1, D]
  non_blank_targets_padded_spatial_dim = non_blank_targets_padded_spatial_dim[0]

  # add blank idx on the right
  # this way, when the label index for gathering reached the last non-blank index, it will gather blank after that
  # which then only allows corresponding hypotheses to be extended by blank
  non_blank_targets_padded, _ = rf.pad(
    non_blank_targets,
    axes=[non_blank_targets_spatial_dim],
    padding=[(0, 1)],
    value=model.blank_idx,
    out_dims=[non_blank_targets_padded_spatial_dim]
  )

  # ------------------------ sizes ------------------------

  non_blank_targets_padded_spatial_sizes = rf.copy_to_device(
    non_blank_targets_padded_spatial_dim.dyn_size_ext, non_blank_targets.device
  )
  non_blank_targets_spatial_sizes = rf.copy_to_device(
    non_blank_targets_spatial_dim.dyn_size_ext, non_blank_targets.device)
  max_num_labels = rf.reduce_max(
    non_blank_targets_spatial_sizes, axis=non_blank_targets_spatial_sizes.dims
  ).raw_tensor.item()
  enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext, non_blank_targets.device)

  # ------------------------ compute LSTM sequence ------------------------

  label_lstm_out_seq, _ = model.s_wo_att(
    non_blank_input_embeddings,
    state=label_lstm_state,
    spatial_dim=non_blank_targets_padded_spatial_dim,
  )

  # ------------------------ chunk dim ------------------------

  chunk_size = 20
  chunk_dim = Dim(chunk_size, name="chunk")
  chunk_range = rf.expand_dim(rf.range_over_dim(chunk_dim), batch_dims[0])

  i = 0
  seq_targets = []
  seq_backrefs = []
  while i < max_seq_len.raw_tensor:
    # get current number of labels for each hypothesis
    if i > 0:
      prev_label_indices = rf.gather(label_indices, indices=backrefs)
      label_indices = rf.where(
        update_state_mask,
        rf.where(
          prev_label_indices == non_blank_targets_padded_spatial_sizes - 1,
          prev_label_indices,
          prev_label_indices + 1
        ),
        prev_label_indices
      )

    # gather ground truth, input embeddings and LSTM output for current label index
    ground_truth = rf.gather(
      non_blank_targets_padded,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )
    input_embed = rf.gather(
      non_blank_input_embeddings,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )
    label_lstm_out = rf.gather(
      label_lstm_out_seq,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )

    # precompute attention for the current chunk (more efficient than computing it individually for each label index)
    if i % chunk_size == 0:
      seg_starts = rf.gather(
        segment_starts,
        indices=chunk_range,
        axis=enc_spatial_dim,
        clip_to_valid=True
      )
      seg_lens = rf.gather(
        segment_lens,
        indices=chunk_range,
        axis=enc_spatial_dim,
        clip_to_valid=True
      )
      att = model(
        enc=enc_args["enc"],
        enc_ctx=enc_args["enc_ctx"],
        enc_spatial_dim=enc_spatial_dim,
        s=label_lstm_out_seq,
        segment_starts=seg_starts,
        segment_lens=seg_lens,
      )  # [B, S+1, T, D]
      chunk_range += chunk_size

    # gather attention for the current label index
    att_step = rf.gather(
      att,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )
    att_step = rf.gather(
      att_step,
      indices=rf.constant(i % chunk_size, dims=batch_dims, device=att_step.device),
      axis=chunk_dim,
      clip_to_valid=True
    )

    logits = model.decode_logits(
      input_embed=input_embed,
      att=att_step,
      s=label_lstm_out,
    )  # [B, S+1, T, D]

    label_log_prob = rf.log_softmax(logits, axis=model.target_dim)

    # alpha = 0.
    # label_log_prob = label_log_prob + rf.stop_gradient(label_log_prob * (alpha - 1))

    # # scale down blank gradient to avoid outputting only blanks in the beginning
    # # and then all other labels in the end
    # def custom_backward(grad):
    #   grad[:, :, 0] *= 0.00005
    #   return grad
    #
    # if rf.get_run_ctx().train_flag:
    #   label_log_prob.raw_tensor.register_hook(custom_backward)

    # mask label log prob in order to only allow hypotheses corresponding to the ground truth:
    # log prob needs to correspond to the next non-blank label...
    log_prob_mask = vocab_range == ground_truth
    rem_frames = enc_spatial_sizes - i
    rem_labels = non_blank_targets_spatial_sizes - label_indices
    # ... or to blank if there are more frames than labels left
    log_prob_mask = rf.logical_or(
      log_prob_mask,
      rf.logical_and(
        vocab_range == blank_tensor,
        rem_frames > rem_labels
      )
    )
    label_log_prob = rf.where(
      log_prob_mask,
      label_log_prob,
      rf.constant(-1.0e30, dims=batch_dims + [beam_dim, model.target_dim])
    )

    # recombine hypotheses corresponding to the same node in the lattice (= same hash value -> same label history)
    # do this by setting the log prob of all but the best hypothesis to -inf
    # and setting the log prob of the best hypothesis to either the max or the sum of the equivalent hypotheses
    seq_log_prob = recombination.recombine_seqs(seq_targets, seq_log_prob, seq_hash, beam_dim, batch_dims[0])

    # set the beam size as low as possible according to the following rules (using recombination):
    # 1) in frame i, there are i+1 nodes in the lattice and from each node, we can spawn 2 hypotheses
    # 2) if T-i frames remain, only (T-i)*2 hypotheses can survive in order to reach the last node
    # 3) in a frame, there are at most S+1 nodes, i.e. (S+1)*2 hypotheses can be spawned (see 1))
    # 4) the beam size should not exceed the given beam size
    beam_size_ = min(
      min((i + 1) * 2, rf.reduce_max(rem_frames, axis=rem_frames.dims).raw_tensor.item() * 2),
      min((max_num_labels + 1) * 2 - 1, beam_size)
    )

    # update sequence log prob and beam indices
    seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob,
      k_dim=Dim(beam_size_, name=f"dec-step{i}-beam"),
      axis=[beam_dim, model.target_dim]
    )  # seq_log_prob, backrefs, target: Batch, Beam
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    seq_hash = recombination.update_seq_hash(seq_hash, target, backrefs, model.blank_idx)

    # mask blank label
    update_state_mask = rf.convert_to_tensor(target != model.blank_idx)

    i += 1

  # last recombination
  seq_log_prob = recombination.recombine_seqs(seq_targets, seq_log_prob, seq_hash, beam_dim, batch_dims[0])

  # # Backtrack via backrefs, resolve beams.
  # seq_targets_ = []
  # indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
  # for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
  #   # indices: FinalBeam -> Beam
  #   # backrefs: Beam -> PrevBeam
  #   seq_targets_.insert(0, rf.gather(target, indices=indices))
  #   indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam
  #
  # seq_targets__ = TensorArray(seq_targets_[0])
  # for target in seq_targets_:
  #   seq_targets__ = seq_targets__.push_back(target)
  # seq_targets = seq_targets__.stack(axis=enc_spatial_dim)
  #
  # torch.set_printoptions(threshold=10_000)
  # print("seq_targets", seq_targets.copy_transpose(batch_dims + [beam_dim, enc_spatial_dim]).raw_tensor[0, :])
  # print("seq_log_prob", seq_log_prob.raw_tensor[0])

  # calculate full-sum loss using the log-sum-exp trick
  max_log_prob = rf.reduce_max(seq_log_prob, axis=beam_dim)
  loss = -1 * (max_log_prob + rf.log(rf.reduce_sum(rf.exp(seq_log_prob - max_log_prob), axis=beam_dim)))

  # loss = -rf.log(rf.reduce_sum(rf.exp(seq_log_prob), axis=beam_dim))
  loss.mark_as_loss("full_sum_loss", scale=1.0, use_normalized_loss=True)

  return None


def full_sum_training_w_beam_eff_w_recomb(
        *,
        model: SegmentalAttEfficientLabelDecoder,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,  # [B, S, V]
        non_blank_targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,  # [B, T]
        segment_lens: rf.Tensor,  # [B, T]
        batch_dims: List[Dim],
        beam_size: int,
) -> Optional[Dict[str, Tuple[rf.Tensor, Dim]]]:
  assert len(batch_dims) == 1, "not supported yet"
  assert model.blank_idx == 0, "blank idx needs to be zero because of the way the gradient is scaled"

  # ------------------------ init some variables ------------------------
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims
  bos_idx = 0
  seq_log_prob = rf.constant(0.0, dims=batch_dims_)
  max_seq_len = enc_spatial_dim.get_size_tensor()
  max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)
  label_lstm_state = model.s_wo_att.default_initial_state(batch_dims=batch_dims)
  target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
  vocab_range = rf.range_over_dim(model.target_dim)
  blank_tensor = rf.convert_to_tensor(model.blank_idx, dtype=vocab_range.dtype)
  backrefs = rf.zeros(batch_dims_, dtype="int32")

  # ------------------------ targets/embeddings ------------------------

  non_blank_input_embeddings = model.target_embed(non_blank_targets)  # [B, S, D]
  non_blank_input_embeddings, non_blank_targets_padded_spatial_dim = rf.pad(
    non_blank_input_embeddings,
    axes=[non_blank_targets_spatial_dim],
    padding=[(1, 0)],
    value=0.0,
  )  # [B, S+1, D]
  non_blank_targets_padded_spatial_dim = non_blank_targets_padded_spatial_dim[0]

  # add blank idx on the right
  # this way, when the label index for gathering reached the last non-blank index, it will gather blank after that
  # which then only allows corresponding hypotheses to be extended by blank
  non_blank_targets_padded, _ = rf.pad(
    non_blank_targets,
    axes=[non_blank_targets_spatial_dim],
    padding=[(0, 1)],
    value=model.blank_idx,
    out_dims=[non_blank_targets_padded_spatial_dim]
  )

  # ------------------------ sizes ------------------------

  non_blank_targets_padded_spatial_sizes = rf.copy_to_device(
    non_blank_targets_padded_spatial_dim.dyn_size_ext, non_blank_targets.device
  )
  non_blank_targets_spatial_sizes = rf.copy_to_device(
    non_blank_targets_spatial_dim.dyn_size_ext, non_blank_targets.device)
  max_num_labels = rf.reduce_max(
    non_blank_targets_spatial_sizes, axis=non_blank_targets_spatial_sizes.dims
  ).raw_tensor.item()
  single_col_dim = Dim(dimension=max_num_labels + 1, name="max-num-labels")
  label_indices = rf.zeros(batch_dims_, dtype="int32", sparse_dim=single_col_dim)

  enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext, non_blank_targets.device)

  # ------------------------ compute LSTM sequence ------------------------

  label_lstm_out_seq, _ = model.s_wo_att(
    non_blank_input_embeddings,
    state=label_lstm_state,
    spatial_dim=non_blank_targets_padded_spatial_dim,
  )

  # ------------------------ chunk dim ------------------------

  chunk_size = 20
  chunk_dim = Dim(chunk_size, name="chunk")
  chunk_range = rf.expand_dim(rf.range_over_dim(chunk_dim), batch_dims[0])

  i = 0
  seq_targets = []
  seq_backrefs = []
  while i < max_seq_len.raw_tensor:
    # get current number of labels for each hypothesis
    if i > 0:
      prev_label_indices = rf.gather(label_indices, indices=backrefs)
      # mask blank label
      update_state_mask = rf.convert_to_tensor(target != prev_label_indices)
      label_indices = rf.where(
        update_state_mask,
        rf.where(
          prev_label_indices == non_blank_targets_padded_spatial_sizes - 1,
          prev_label_indices,
          prev_label_indices + 1
        ),
        prev_label_indices
      )

    # gather ground truth, input embeddings and LSTM output for current label index
    ground_truth = rf.gather(
      non_blank_targets_padded,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )
    input_embed = rf.gather(
      non_blank_input_embeddings,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )
    label_lstm_out = rf.gather(
      label_lstm_out_seq,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )

    # precompute attention for the current chunk (more efficient than computing it individually for each label index)
    if i % chunk_size == 0:
      seg_starts = rf.gather(
        segment_starts,
        indices=chunk_range,
        axis=enc_spatial_dim,
        clip_to_valid=True
      )
      seg_lens = rf.gather(
        segment_lens,
        indices=chunk_range,
        axis=enc_spatial_dim,
        clip_to_valid=True
      )
      att = model(
        enc=enc_args["enc"],
        enc_ctx=enc_args["enc_ctx"],
        enc_spatial_dim=enc_spatial_dim,
        s=label_lstm_out_seq,
        segment_starts=seg_starts,
        segment_lens=seg_lens,
      )  # [B, S+1, T, D]
      chunk_range += chunk_size

    # gather attention for the current label index
    att_step = rf.gather(
      att,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )
    att_step = rf.gather(
      att_step,
      indices=rf.constant(i % chunk_size, dims=batch_dims, device=att_step.device),
      axis=chunk_dim,
      clip_to_valid=True
    )

    logits = model.decode_logits(
      input_embed=input_embed,
      att=att_step,
      s=label_lstm_out,
    )  # [B, S+1, T, D]

    label_log_prob = rf.log_softmax(logits, axis=model.target_dim)

    # mask label log prob in order to only allow hypotheses corresponding to the ground truth:
    # log prob needs to correspond to the next non-blank label...
    log_prob_mask = vocab_range == ground_truth
    rem_frames = enc_spatial_sizes - i
    rem_labels = non_blank_targets_spatial_sizes - label_indices
    # ... or to blank if there are more frames than labels left
    log_prob_mask = rf.logical_or(
      log_prob_mask,
      rf.logical_and(
        vocab_range == blank_tensor,
        rem_frames > rem_labels
      )
    )
    label_log_prob = rf.where(
      log_prob_mask,
      label_log_prob,
      rf.constant(-1.0e30, dims=batch_dims + [beam_dim, model.target_dim])
    )

    label_log_prob = rf.where(
      rf.convert_to_tensor(i >= rf.copy_to_device(enc_spatial_dim.get_size_tensor(), label_log_prob.device)),
      rf.sparse_to_dense(
        model.blank_idx,
        axis=model.target_dim,
        label_value=0.0,
        other_value=-1.0e30
      ),
      label_log_prob
    )

    seq_log_prob = recombination.recombine_seqs_train(
      seq_log_prob=seq_log_prob,
      label_log_prob=label_log_prob,
      label_indices=label_indices,
      ground_truth=ground_truth,
      target_dim=model.target_dim,
      single_col_dim=single_col_dim,
      labels_padded_spatial_sizes=non_blank_targets_padded_spatial_sizes,
      beam_dim=beam_dim,
      batch_dims=batch_dims,
      blank_idx=model.blank_idx,
    )

    beam_size_ = min(
      min((i + 2), rf.reduce_max(rem_frames, axis=rem_frames.dims).raw_tensor.item()),
      min((max_num_labels + 1), beam_size)
    )

    # update sequence log prob and beam indices
    # seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob,
      k_dim=Dim(beam_size_, name=f"dec-step{i}-beam"),
      axis=[beam_dim, single_col_dim]
    )  # seq_log_prob, backrefs, target: Batch, Beam
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

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

  torch.set_printoptions(threshold=10_000)
  print("seq_targets", seq_targets.copy_transpose(batch_dims + [beam_dim, enc_spatial_dim]).raw_tensor[0, 0])

  loss = -1 * seq_log_prob

  # loss = -rf.log(rf.reduce_sum(rf.exp(seq_log_prob), axis=beam_dim))
  loss.mark_as_loss("full_sum_loss", scale=1.0, use_normalized_loss=True)

  # print("loss", loss.raw_tensor)
  # print("single_col_dim", single_col_dim.dimension)
  # exit()

  return None

