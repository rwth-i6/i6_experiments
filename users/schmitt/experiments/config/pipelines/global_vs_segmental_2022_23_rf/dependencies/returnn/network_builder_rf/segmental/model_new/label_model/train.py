from typing import Optional, Dict, Any, Sequence, Tuple, List

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.utils import get_non_blank_mask, get_masked
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import SegmentalAttLabelDecoder

from returnn.tensor import Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef


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
        output_tensors: Optional[List[str]] = None
) -> Optional[Dict[str, Tuple[rf.Tensor, Dim]]]:
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
  logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [non_blank_targets_spatial_dim], enforce_sorted=False)
  non_blank_targets_packed, _ = rf.pack_padded(
    non_blank_targets, dims=batch_dims + [non_blank_targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
  )

  log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
  log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
  loss = rf.cross_entropy(
    target=non_blank_targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
  )
  loss.mark_as_loss("non_blank_ce", scale=1.0, use_normalized_loss=True)

  best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
  frame_error = best != non_blank_targets_packed
  frame_error.mark_as_loss(name="non_blank_fer", as_error=True)

  if output_tensors is not None:
    extended_outputs = {}
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
    for key, val in last_loop_out.items():
      if key not in output_tensors:
        continue
      extended_outputs[key] = rf.concat(
        (label_loop_out[key], non_blank_targets_spatial_dim),
        (rf.expand_dim(val, single_step_dim), single_step_dim),
      )
    return extended_outputs

  return None
