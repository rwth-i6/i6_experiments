from typing import Optional, Dict, Any, Sequence, Tuple, List, Union
import tree

import torch

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import recombination
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder,
  SegmentalAttEfficientLabelDecoder
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import (
  GlobalAttDecoder,
  GlobalAttEfficientDecoder
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.train import (
  forward_sequence as forward_sequence_global_att,
)

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import (
  BlankDecoderV4
)

from returnn.tensor import Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef


def _calc_ce_loss_and_fer(
        logits: rf.Tensor,
        targets: rf.Tensor,
        batch_dims: List[Dim],
        targets_spatial_dim: Dim,
        target_dim: Dim,
        blank_idx: int,
        separate_blank_loss: bool = False,
        beam_dim: Optional[Dim] = None,
        separate_blank_from_softmax: bool = False,
        loss_alias: Optional[str] = None,
):
  from returnn.config import get_global_config
  config = get_global_config()  # noqa
  nb_loss_scale = config.typed_value("nb_loss_scale", 1.0)
  b_loss_scale = config.typed_value("b_loss_scale", 1.0)

  if beam_dim is not None:
    batch_dims = logits.remaining_dims([beam_dim, targets_spatial_dim, target_dim])
  logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False)

  non_blank_targets_packed, _ = rf.pack_padded(
    targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
  )

  if separate_blank_from_softmax:
    log_prob = utils.log_softmax_sep_blank(logits=logits_packed, blank_idx=blank_idx, target_dim=target_dim)
  else:
    log_prob = rf.log_softmax(logits_packed, axis=target_dim)

  log_prob_label_smoothed = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=target_dim)
  loss = rf.cross_entropy(
    target=non_blank_targets_packed, estimated=log_prob_label_smoothed, estimated_type="log-probs", axis=target_dim
  )

  nb_mask = rf.convert_to_tensor(non_blank_targets_packed != blank_idx)

  # scale loss
  loss = rf.where(
    nb_mask,
    loss * nb_loss_scale,
    loss * b_loss_scale,
  )

  # additionally report separate losses for blank and non-blank targets
  # these additional losses are marked as an error and are therefore not used for optimization
  if separate_blank_loss:
    # non-blank loss
    nb_loss, _ = rf.masked_select(loss, mask=nb_mask, dims=[pack_dim])
    nb_loss.mark_as_loss("non_blank_ce", scale=1.0, use_normalized_loss=True, as_error=True)

    # non-blank frame error
    nb_targets, nb_spatial_dim = rf.masked_select(non_blank_targets_packed, mask=nb_mask, dims=[pack_dim])
    nb_logits, _ = rf.masked_select(logits_packed, mask=nb_mask, dims=[pack_dim], out_dim=nb_spatial_dim)
    nb_best = rf.reduce_argmax(nb_logits, axis=target_dim)
    nb_frame_error = nb_best != nb_targets
    nb_frame_error.mark_as_loss(name="non_blank_fer", as_error=True)

    # blank ground-truth
    emit_ground_truth, emit_blank_target_dim = utils.get_emit_ground_truth(targets, blank_idx)
    emit_ground_truth_packed, _ = rf.pack_padded(
      emit_ground_truth, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    )
    # emit-blank log-prob (2 target dims: blank and non-blank (1 - blank))
    singleton_dim = Dim(name="blank_singleton", dimension=1)
    blank_log_prob = rf.gather(
      log_prob_label_smoothed,
      indices=rf.expand_dim(rf.convert_to_tensor(blank_idx), singleton_dim),
      axis=target_dim
    )
    emit_log_prob = rf.log(1 - rf.exp(blank_log_prob))
    emit_blank_log_prob, _ = rf.concat(
      (blank_log_prob, singleton_dim), (emit_log_prob, singleton_dim), out_dim=emit_blank_target_dim)

    # blank loss
    b_loss = rf.cross_entropy(
      target=emit_ground_truth_packed,
      estimated=emit_blank_log_prob,
      estimated_type="log-probs",
      axis=emit_blank_target_dim
    )
    b_loss.mark_as_loss("emit_blank_ce", scale=1.0, use_normalized_loss=True, as_error=True)

    # blank frame error
    b_best = rf.reduce_argmax(emit_blank_log_prob, axis=emit_blank_target_dim)
    b_frame_error = b_best != emit_ground_truth_packed
    b_frame_error.mark_as_loss(name="emit_blank_fer", as_error=True)

    loss_prefix = "total"
  else:
    loss_prefix = "non_blank"

  loss_name = f"{loss_alias}_{loss_prefix}" if loss_alias else loss_prefix
  loss.mark_as_loss(f"{loss_name}_ce", scale=1.0, use_normalized_loss=True)

  best = rf.reduce_argmax(logits_packed, axis=target_dim)
  frame_error = best != non_blank_targets_packed
  frame_error.mark_as_loss(name=f"{loss_name}_fer", as_error=True)


def forward_sequence(
        *,
        model: SegmentalAttLabelDecoder,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,
        segment_lens: rf.Tensor,
        center_positions: rf.Tensor,
        batch_dims: List[Dim],
        return_label_model_states: bool = False,
        detach_att: bool = False,
        detach_h_t: bool = False,
) -> Tuple[rf.Tensor, Optional[Tuple[rf.Tensor, Dim]]]:
  non_blank_input_embeddings = model.target_embed(non_blank_targets)
  non_blank_input_embeddings_shifted = rf.shift_right(
    non_blank_input_embeddings, axis=non_blank_targets_spatial_dim, pad_value=0.0)
  non_blank_input_embeddings = rf.dropout(non_blank_input_embeddings, drop_prob=model.target_embed_dropout, axis=None)

  # ------------------- label loop -------------------

  def _label_loop_body(xs, state: rf.State):
    new_state = rf.State()
    loop_out_, new_state.decoder = model.loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=xs["input_embed"],
      segment_starts=xs["segment_starts"],
      segment_lens=xs["segment_lens"],
      center_positions=xs["center_positions"],
      state=state.decoder,
      detach_prev_att=detach_att,
    )
    return loop_out_, new_state

  label_loop_out, final_state, _ = rf.scan(
    spatial_dim=non_blank_targets_spatial_dim,
    xs={
      "input_embed": non_blank_input_embeddings_shifted,
      "segment_starts": segment_starts,
      "segment_lens": segment_lens,
      "center_positions": center_positions,
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

  if model.use_current_frame_in_readout or model.use_current_frame_in_readout_w_gate or model.use_current_frame_in_readout_random:
    h_t = rf.gather(enc_args["enc"], axis=enc_spatial_dim, indices=center_positions)
  else:
    h_t = None
  logits, _ = model.decode_logits(
    input_embed=non_blank_input_embeddings_shifted,
    **label_loop_out,
    h_t=h_t,
    detach_att=detach_att,
    detach_h_t=detach_h_t,
  )

  if return_label_model_states:
    # need to run the loop one more time to get the last output (which is not needed for the loss computation)
    last_embedding = rf.gather(
        non_blank_input_embeddings,
        axis=non_blank_targets_spatial_dim,
        indices=rf.copy_to_device(
          non_blank_targets_spatial_dim.get_size_tensor() - 1, non_blank_input_embeddings.device)
    )
    last_center_position = rf.gather(
        center_positions,
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
      center_positions=last_center_position,
      state=final_state.decoder,
    )
    singleton_dim = Dim(name="singleton", dimension=1)
    return logits, rf.concat(
      (label_loop_out["s"], non_blank_targets_spatial_dim),
      (rf.expand_dim(last_loop_out["s"], singleton_dim), singleton_dim),
    )

  return logits, None


def viterbi_training(
        *,
        model: Union[SegmentalAttLabelDecoder, GlobalAttDecoder],
        enc_args: Dict,
        att_enc_args: Dict,
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,
        segment_lens: rf.Tensor,
        center_positions: rf.Tensor,
        batch_dims: List[Dim],
        separate_blank_loss: bool = False,
        return_label_model_states: bool = False,
        beam_dim: Optional[Dim] = None,
) -> Tuple[rf.Tensor, Optional[Tuple[rf.Tensor, Dim]]]:

  if isinstance(model, SegmentalAttLabelDecoder):
    logits, label_model_states = forward_sequence(
      model=model,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      non_blank_targets=non_blank_targets,
      non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
      segment_starts=segment_starts,
      segment_lens=segment_lens,
      center_positions=center_positions,
      batch_dims=batch_dims,
      return_label_model_states=return_label_model_states,
    )
    h_t_logits = None
  else:
    logits, label_model_states, h_t_logits = forward_sequence_global_att(
      model=model,
      targets=non_blank_targets,
      targets_spatial_dim=non_blank_targets_spatial_dim,
      enc_args=enc_args,
      att_enc_args=att_enc_args,
      enc_spatial_dim=enc_spatial_dim,
      batch_dims=batch_dims,
      return_label_model_states=return_label_model_states,
      center_positions=center_positions,
    )

  _calc_ce_loss_and_fer(
    logits,
    non_blank_targets,
    batch_dims,
    non_blank_targets_spatial_dim,
    model.target_dim,
    blank_idx=model.blank_idx,
    separate_blank_loss=separate_blank_loss,
    beam_dim=beam_dim,
    separate_blank_from_softmax=model.separate_blank_from_softmax,
  )

  if h_t_logits:
    _calc_ce_loss_and_fer(
      h_t_logits,
      non_blank_targets,
      batch_dims,
      non_blank_targets_spatial_dim,
      model.target_dim,
      blank_idx=model.blank_idx,
      separate_blank_loss=separate_blank_loss,
      beam_dim=beam_dim,
      separate_blank_from_softmax=model.separate_blank_from_softmax,
      loss_alias="sep-h_t"
    )

  return logits, label_model_states


def forward_sequence_efficient(
        *,
        model: SegmentalAttEfficientLabelDecoder,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        targets: rf.Tensor,
        targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,
        segment_lens: rf.Tensor,
        center_positions: rf.Tensor,
        batch_dims: List[Dim],
        non_blank_mask: Optional[rf.Tensor] = None,
        non_blank_mask_spatial_dim: Optional[Dim] = None,
        return_label_model_states: bool = False,
        detach_att: bool = False,
        detach_h_t: bool = False,
) -> Tuple[rf.Tensor, Optional[Tuple[rf.Tensor, Dim]]]:

  input_embeddings = model.target_embed(targets)
  input_embeddings_shifted = rf.shift_right(
    input_embeddings, axis=targets_spatial_dim, pad_value=0.0)

  if "lstm" in model.decoder_state:
    s_out, s_final_state = model.s_wo_att(
      input_embeddings_shifted,
      state=model.s_wo_att.default_initial_state(batch_dims=batch_dims),
      spatial_dim=targets_spatial_dim,
    )
  else:
    s_out = model.s_wo_att_linear(input_embeddings_shifted)
    s_final_state = None

  if non_blank_mask is not None:
    s_out = utils.get_unmasked(
      input=s_out,
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
  targets_spatial_dim.dyn_size_ext = rf.copy_to_device(targets_spatial_dim.dyn_size_ext, s_out.device)
  if non_blank_mask_spatial_dim is not None:
    non_blank_mask_spatial_dim.dyn_size_ext = rf.copy_to_device(non_blank_mask_spatial_dim.dyn_size_ext, s_out.device)

  att = model(
    enc=enc_args["enc"],
    enc_ctx=enc_args.get("enc_ctx"),
    enc_spatial_dim=enc_spatial_dim,
    s=s_out,
    segment_starts=segment_starts,
    segment_lens=segment_lens,
    center_positions=center_positions,
  )
  targets_spatial_dim.dyn_size_ext = rf.copy_to_device(targets_spatial_dim.dyn_size_ext, "cpu")
  if non_blank_mask_spatial_dim is not None:
    non_blank_mask_spatial_dim.dyn_size_ext = rf.copy_to_device(non_blank_mask_spatial_dim.dyn_size_ext, "cpu")

  if model.use_current_frame_in_readout or model.use_current_frame_in_readout_w_gate or model.use_current_frame_in_readout_random:
    h_t = rf.gather(enc_args["enc"], axis=enc_spatial_dim, indices=center_positions)
  else:
    h_t = None

  logits, _ = model.decode_logits(
    input_embed=input_embeddings_shifted,
    att=att,
    s=s_out,
    h_t=h_t,
    detach_att=detach_att,
    detach_h_t=detach_h_t,
  )

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
    if "lstm" in model.decoder_state:
      last_s_out, _ = model.s_wo_att(
        last_embedding,
        state=s_final_state,
        spatial_dim=single_step_dim,
      )
    else:
      last_s_out = model.s_wo_att_linear(last_embedding)

    return logits, rf.concat(
      (s_out, targets_spatial_dim),
      (rf.expand_dim(last_s_out, singleton_dim), singleton_dim),
    )

  return logits, None


def viterbi_training_efficient(
        *,
        model: Union[SegmentalAttEfficientLabelDecoder, GlobalAttEfficientDecoder],
        enc_args: Dict,
        enc_spatial_dim: Dim,
        targets: rf.Tensor,
        targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,
        segment_lens: rf.Tensor,
        center_positions: rf.Tensor,
        batch_dims: List[Dim],
        ce_targets: rf.Tensor,
        ce_spatial_dim: Dim,
        non_blank_mask: Optional[rf.Tensor] = None,
        non_blank_mask_spatial_dim: Optional[Dim] = None,
        return_label_model_states: bool = False,
        beam_dim: Optional[Dim] = None,
        separate_blank_loss: bool = False,
) -> Tuple[rf.Tensor, Optional[Tuple[rf.Tensor, Dim]]]:

  if isinstance(model, SegmentalAttEfficientLabelDecoder):
    logits, label_model_states = forward_sequence_efficient(
      model=model,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      targets=targets,
      targets_spatial_dim=targets_spatial_dim,
      segment_starts=segment_starts,
      segment_lens=segment_lens,
      center_positions=center_positions,
      batch_dims=batch_dims,
      non_blank_mask=non_blank_mask,
      non_blank_mask_spatial_dim=non_blank_mask_spatial_dim,
      return_label_model_states=return_label_model_states,
    )
  else:
    logits, label_model_states = forward_sequence_global_att(
      model=model,
      targets=targets,
      targets_spatial_dim=targets_spatial_dim,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      batch_dims=batch_dims,
      return_label_model_states=return_label_model_states,
      center_positions=center_positions,
    )

  _calc_ce_loss_and_fer(
    logits,
    ce_targets,
    batch_dims,
    ce_spatial_dim,
    model.target_dim,
    beam_dim=beam_dim,
    blank_idx=model.blank_idx,
    separate_blank_loss=separate_blank_loss,
    separate_blank_from_softmax=model.separate_blank_from_softmax,
  )

  return logits, label_model_states


def full_sum_training(
        *,
        model: SegmentalAttentionModel,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,  # [B, S, V]
        non_blank_targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,  # [B, T]
        segment_lens: rf.Tensor,  # [B, T]
        center_positions: rf.Tensor,  # [B, T]
        batch_dims: List[Dim],
) -> Optional[Dict[str, Tuple[rf.Tensor, Dim]]]:
  assert isinstance(
    model.label_decoder, SegmentalAttEfficientLabelDecoder) or isinstance(
    model.label_decoder, GlobalAttEfficientDecoder) or isinstance(
    model.label_decoder, TransformerDecoder) or isinstance(
    model.label_decoder, GlobalAttDecoder
  )

  from returnn.config import get_global_config
  config = get_global_config()  # noqa

  if model.label_decoder_state == "trafo":
    enc_transformed = model.label_decoder.transform_encoder(enc_args["enc"], axis=enc_spatial_dim)

    non_blank_targets_ext, non_blank_targets_spatial_dim_ext = rf.pad(
      non_blank_targets,
      axes=[non_blank_targets_spatial_dim],
      padding=[(1, 0)],
      value=model.bos_idx
    )
    non_blank_targets_spatial_dim_ext = non_blank_targets_spatial_dim_ext[0]
    logits, _, s_out = model.label_decoder(
      non_blank_targets_ext,
      spatial_dim=non_blank_targets_spatial_dim_ext,
      encoder=enc_transformed,
      state=model.label_decoder.default_initial_state(batch_dims=batch_dims),
    )
  else:
    non_blank_input_embeddings = model.label_decoder.target_embed(non_blank_targets)  # [B, S, D]
    singleton_dim = Dim(name="singleton", dimension=1)
    singleton_zeros = rf.zeros(batch_dims + [singleton_dim, model.label_decoder.target_embed.out_dim])
    non_blank_input_embeddings_shifted, non_blank_targets_spatial_dim_ext = rf.concat(
      (singleton_zeros, singleton_dim),
      (non_blank_input_embeddings, non_blank_targets_spatial_dim),
      allow_broadcast=True
    )  # [B, S+1, D]
    non_blank_input_embeddings_shifted.feature_dim = non_blank_input_embeddings.feature_dim

    if "lstm" in model.label_decoder.decoder_state:
      if type(model.label_decoder) is GlobalAttDecoder:
        def _body(input_embed: rf.Tensor, state: rf.State):
          new_state = rf.State()
          loop_out_, new_state.decoder = model.label_decoder.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=state.decoder,
          )
          return loop_out_, new_state

        loop_out, _, _ = rf.scan(
          spatial_dim=non_blank_targets_spatial_dim_ext,
          xs=non_blank_input_embeddings_shifted,
          ys=model.label_decoder.loop_step_output_templates(batch_dims=batch_dims),
          initial=rf.State(
            decoder=model.label_decoder.decoder_default_initial_state(
              batch_dims=batch_dims,
              enc_spatial_dim=enc_spatial_dim,
            ),
          ),
          body=_body,
        )
        s_out = loop_out["s"]
      else:
        s_out, _ = model.label_decoder.s_wo_att(
          non_blank_input_embeddings_shifted,
          state=model.label_decoder.s_wo_att.default_initial_state(batch_dims=batch_dims),
          spatial_dim=non_blank_targets_spatial_dim_ext,
        )  # [B, S+1, D]
    else:
      if type(model.label_decoder) is GlobalAttDecoder:
        def _body(input_embed: rf.Tensor, state: rf.State):
          new_state = rf.State()
          loop_out_, new_state.decoder = model.label_decoder.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=state.decoder,
          )
          return loop_out_, new_state

        loop_out, _, _ = rf.scan(
          spatial_dim=non_blank_targets_spatial_dim_ext,
          xs=non_blank_input_embeddings_shifted,
          ys=model.label_decoder.loop_step_output_templates(batch_dims=batch_dims),
          initial=rf.State(
            decoder=model.label_decoder.decoder_default_initial_state(
              batch_dims=batch_dims,
              enc_spatial_dim=enc_spatial_dim,
            ),
          ),
          body=_body,
        )
        s_out = loop_out["s"]
      else:
        s_out = model.label_decoder.s_wo_att_linear(non_blank_input_embeddings_shifted)  # [B, S+1, D]

    if isinstance(model.label_decoder, GlobalAttEfficientDecoder):
      att = model.label_decoder(
        enc=enc_args["enc"],
        enc_ctx=enc_args["enc_ctx"],
        enc_spatial_dim=enc_spatial_dim,
        s=s_out,
        input_embed=non_blank_input_embeddings_shifted,
        input_embed_spatial_dim=non_blank_targets_spatial_dim_ext,
      )
    elif type(model.label_decoder) is GlobalAttDecoder:
      att = loop_out["att"]
    else:
      assert isinstance(model.label_decoder, SegmentalAttEfficientLabelDecoder)
      if model.center_window_size == 1:
        att = enc_args["enc"]
      else:
        att = model.label_decoder(
          enc=enc_args["enc"],
          enc_ctx=enc_args["enc_ctx"],
          enc_spatial_dim=enc_spatial_dim,
          s=s_out,
          segment_starts=segment_starts,
          segment_lens=segment_lens,
          center_positions=center_positions,
        )  # [B, S+1, T, D]

    if (
            model.label_decoder.use_current_frame_in_readout or
            model.label_decoder.use_current_frame_in_readout_w_gate or
            model.label_decoder.use_current_frame_in_readout_random
    ):
      h_t = rf.gather(enc_args["enc"], axis=enc_spatial_dim, indices=center_positions)
    else:
      h_t = None

    logits, _ = model.label_decoder.decode_logits(
      input_embed=non_blank_input_embeddings_shifted,
      att=att,
      s=s_out,
      h_t=h_t,
    )  # [B, S+1, T, D]

    if config.bool("use_sep_ce_loss", False):
      ce_logits, _ = model.label_decoder.decode_logits(
        input_embed=rf.shift_right(non_blank_input_embeddings, axis=non_blank_targets_spatial_dim, pad_value=0.0),
        att=att,
        s=s_out,
        h_t=None,
      )
      ce_logits_packed, pack_dim = rf.pack_padded(
        ce_logits, dims=batch_dims + [non_blank_targets_spatial_dim], enforce_sorted=False)
      non_blank_targets_packed, _ = rf.pack_padded(
        non_blank_targets, dims=batch_dims + [non_blank_targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
      )

      ce_log_prob = rf.log_softmax(ce_logits_packed, axis=model.target_dim)
      ce_log_prob = rf.label_smoothed_log_prob_gradient(ce_log_prob, 0.1, axis=model.target_dim)
      ce_loss = rf.cross_entropy(
        target=non_blank_targets_packed, estimated=ce_log_prob, estimated_type="log-probs", axis=model.target_dim
      )
      ce_loss.mark_as_loss("aed_ce", scale=1.0, use_normalized_loss=True)

      best = rf.reduce_argmax(ce_logits_packed, axis=model.target_dim)
      ce_frame_error = best != non_blank_targets_packed
      ce_frame_error.mark_as_loss(name="fer", as_error=True)

  if model.blank_decoder is not None:
    assert isinstance(model.blank_decoder, BlankDecoderV4)
    blank_logits = model.blank_decoder.decode_logits(
      enc=enc_args["enc"], label_model_states_unmasked=s_out, allow_broadcast=True
    )  # [B, T, S, 1]

    if model.label_decoder_state == "trafo":
      label_decoder_target_dim = model.label_decoder.vocab_dim
    else:
      label_decoder_target_dim = model.label_decoder.target_dim

    # normally, we combine the labels with blank in log space: label_log_prob = log(softmax(label_logits)) + log(sigmoid(blank_logits))
    # However, Simon's full-sum loss implementation expects logits as input. Therefore, we need to do the combination
    # in logit space
    blank_log_prob = rf.safe_log(rf.sigmoid(-blank_logits))
    log_alt_norm = -blank_logits - blank_log_prob
    max_logit = rf.reduce_max(logits, axis=label_decoder_target_dim)
    # subtract max_logit to avoid infinities in softmax
    log_softmax = logits - rf.safe_log(
      rf.reduce_sum(rf.exp(logits - max_logit), axis=label_decoder_target_dim)) - max_logit
    logits = log_softmax + rf.safe_log(rf.sigmoid(blank_logits)) + log_alt_norm
    logits, _ = rf.concat(
      (logits, label_decoder_target_dim),
      (-blank_logits, model.blank_decoder.emit_prob_dim),
      out_dim=model.align_target_dim,
      allow_broadcast=True
    )
    logits = rf.squeeze(logits, axis=model.blank_decoder.emit_prob_dim)

  def _get_packed_logits_v1():
    # manually pack logits by looping over batch dim
    logits_raw = logits.copy_transpose(batch_dims + [enc_spatial_dim, non_blank_targets_spatial_dim_ext, model.align_target_dim]).raw_tensor
    enc_lens = enc_spatial_dim.dyn_size_ext.raw_tensor
    non_blank_lens = non_blank_targets_spatial_dim_ext.dyn_size_ext.raw_tensor
    vocab_len = model.align_target_dim.dimension

    batch_tensors = []

    for b in range(logits_raw.shape[0]):
      enc_len = enc_lens[b]
      non_blank_len = non_blank_lens[b]
      combined_len = enc_len * non_blank_len
      logits_single = logits_raw[b, :enc_len, :non_blank_len]
      logits_single = torch.reshape(logits_single, (combined_len, vocab_len))
      batch_tensors.append(logits_single)

    return torch.cat(batch_tensors, dim=0)

  def _get_packed_logits_v2():
    # use built-in pack_padded (now efficient, since Albert improved it)
    logits_packed, pack_dim = rf.pack_padded(
      logits,
      dims=batch_dims + [enc_spatial_dim, non_blank_targets_spatial_dim_ext],
      enforce_sorted=False
    )  # [B * T * (S+1), D]
    logits_packed_raw = logits_packed.raw_tensor

    return logits_packed_raw, logits_packed

  logits_packed_raw, logits_packed = _get_packed_logits_v2()

  from returnn.extern_private.BergerMonotonicRNNT.monotonic_rnnt.pytorch_binding import monotonic_rnnt_loss

  loss = monotonic_rnnt_loss(
    acts=logits_packed_raw,
    labels=non_blank_targets.copy_transpose(batch_dims + [non_blank_targets_spatial_dim]).raw_tensor,
    input_lengths=rf.copy_to_device(enc_spatial_dim.dyn_size_ext, logits.device).raw_tensor,
    label_lengths=rf.copy_to_device(non_blank_targets_spatial_dim.dyn_size_ext, logits.device).raw_tensor.int(),
    blank_label=model.blank_idx,
  )

  loss = rf.convert_to_tensor(loss, name="full_sum_loss")
  loss.mark_as_loss("full_sum_loss", scale=1.0, use_normalized_loss=True)

  return None


def get_score_lattice(
        *,
        model: SegmentalAttentionModel,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,  # [B, S, V]
        non_blank_targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,  # [B, T]
        segment_lens: rf.Tensor,  # [B, T]
        center_positions: rf.Tensor,  # [B, T]
        batch_dims: List[Dim],
) -> Tuple[rf.Tensor, rf.Dim]:
  assert isinstance(
    model.label_decoder, SegmentalAttEfficientLabelDecoder) or isinstance(
    model.label_decoder, GlobalAttEfficientDecoder) or isinstance(
    model.label_decoder, TransformerDecoder) or isinstance(
    model.label_decoder, GlobalAttDecoder
  )

  from returnn.config import get_global_config
  config = get_global_config()  # noqa

  if model.label_decoder_state == "trafo":
    enc_transformed = model.label_decoder.transform_encoder(enc_args["enc"], axis=enc_spatial_dim)

    non_blank_targets_ext, non_blank_targets_spatial_dim_ext = rf.pad(
      non_blank_targets,
      axes=[non_blank_targets_spatial_dim],
      padding=[(1, 0)],
      value=model.bos_idx
    )
    non_blank_targets_spatial_dim_ext = non_blank_targets_spatial_dim_ext[0]
    logits, _, s_out = model.label_decoder(
      non_blank_targets_ext,
      spatial_dim=non_blank_targets_spatial_dim_ext,
      encoder=enc_transformed,
      state=model.label_decoder.default_initial_state(batch_dims=batch_dims),
    )
  else:
    non_blank_input_embeddings = model.label_decoder.target_embed(non_blank_targets)  # [B, S, D]
    singleton_dim = Dim(name="singleton", dimension=1)
    singleton_zeros = rf.zeros(batch_dims + [singleton_dim, model.label_decoder.target_embed.out_dim])
    non_blank_input_embeddings_shifted, non_blank_targets_spatial_dim_ext = rf.concat(
      (singleton_zeros, singleton_dim),
      (non_blank_input_embeddings, non_blank_targets_spatial_dim),
      allow_broadcast=True
    )  # [B, S+1, D]
    non_blank_input_embeddings_shifted.feature_dim = non_blank_input_embeddings.feature_dim

    if "lstm" in model.label_decoder.decoder_state:
      if type(model.label_decoder) is GlobalAttDecoder:
        def _body(input_embed: rf.Tensor, state: rf.State):
          new_state = rf.State()
          loop_out_, new_state.decoder = model.label_decoder.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=state.decoder,
          )
          return loop_out_, new_state

        loop_out, _, _ = rf.scan(
          spatial_dim=non_blank_targets_spatial_dim_ext,
          xs=non_blank_input_embeddings_shifted,
          ys=model.label_decoder.loop_step_output_templates(batch_dims=batch_dims),
          initial=rf.State(
            decoder=model.label_decoder.decoder_default_initial_state(
              batch_dims=batch_dims,
              enc_spatial_dim=enc_spatial_dim,
            ),
          ),
          body=_body,
        )
        s_out = loop_out["s"]
      else:
        s_out, _ = model.label_decoder.s_wo_att(
          non_blank_input_embeddings_shifted,
          state=model.label_decoder.s_wo_att.default_initial_state(batch_dims=batch_dims),
          spatial_dim=non_blank_targets_spatial_dim_ext,
        )  # [B, S+1, D]
    else:
      if type(model.label_decoder) is GlobalAttDecoder:
        def _body(input_embed: rf.Tensor, state: rf.State):
          new_state = rf.State()
          loop_out_, new_state.decoder = model.label_decoder.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=state.decoder,
          )
          return loop_out_, new_state

        loop_out, _, _ = rf.scan(
          spatial_dim=non_blank_targets_spatial_dim_ext,
          xs=non_blank_input_embeddings_shifted,
          ys=model.label_decoder.loop_step_output_templates(batch_dims=batch_dims),
          initial=rf.State(
            decoder=model.label_decoder.decoder_default_initial_state(
              batch_dims=batch_dims,
              enc_spatial_dim=enc_spatial_dim,
            ),
          ),
          body=_body,
        )
        s_out = loop_out["s"]
      else:
        s_out = model.label_decoder.s_wo_att_linear(non_blank_input_embeddings_shifted)  # [B, S+1, D]

    if isinstance(model.label_decoder, GlobalAttEfficientDecoder):
      att = model.label_decoder(
        enc=enc_args["enc"],
        enc_ctx=enc_args["enc_ctx"],
        enc_spatial_dim=enc_spatial_dim,
        s=s_out,
        input_embed=non_blank_input_embeddings_shifted,
        input_embed_spatial_dim=non_blank_targets_spatial_dim_ext,
      )
    elif type(model.label_decoder) is GlobalAttDecoder:
      att = loop_out["att"]
    else:
      assert isinstance(model.label_decoder, SegmentalAttEfficientLabelDecoder)
      if model.center_window_size == 1:
        att = enc_args["enc"]
      else:
        att = model.label_decoder(
          enc=enc_args["enc"],
          enc_ctx=enc_args["enc_ctx"],
          enc_spatial_dim=enc_spatial_dim,
          s=s_out,
          segment_starts=segment_starts,
          segment_lens=segment_lens,
          center_positions=center_positions,
        )  # [B, S+1, T, D]

    if (
            model.label_decoder.use_current_frame_in_readout or
            model.label_decoder.use_current_frame_in_readout_w_gate or
            model.label_decoder.use_current_frame_in_readout_random
    ):
      h_t = rf.gather(enc_args["enc"], axis=enc_spatial_dim, indices=center_positions)
    else:
      h_t = None

    logits, _ = model.label_decoder.decode_logits(
      input_embed=non_blank_input_embeddings_shifted,
      att=att,
      s=s_out,
      h_t=h_t,
    )  # [B, S+1, T, D]

  if model.blank_decoder is not None:
    assert isinstance(model.blank_decoder, BlankDecoderV4)
    blank_logits = model.blank_decoder.decode_logits(
      enc=enc_args["enc"], label_model_states_unmasked=s_out, allow_broadcast=True
    )  # [B, T, S, 1]

    if model.label_decoder_state == "trafo":
      label_decoder_target_dim = model.label_decoder.vocab_dim
    else:
      label_decoder_target_dim = model.label_decoder.target_dim

    # normally, we combine the labels with blank in log space: label_log_prob = log(softmax(label_logits)) + log(sigmoid(blank_logits))
    # However, Simon's full-sum loss implementation expects logits as input. Therefore, we need to do the combination
    # in logit space
    blank_log_prob = rf.safe_log(rf.sigmoid(-blank_logits))
    log_alt_norm = -blank_logits - blank_log_prob
    max_logit = rf.reduce_max(logits, axis=label_decoder_target_dim)
    # subtract max_logit to avoid infinities in softmax
    log_softmax = logits - rf.safe_log(
      rf.reduce_sum(rf.exp(logits - max_logit), axis=label_decoder_target_dim)) - max_logit
    logits = log_softmax + rf.safe_log(rf.sigmoid(blank_logits)) + log_alt_norm
    logits, _ = rf.concat(
      (logits, label_decoder_target_dim),
      (-blank_logits, model.blank_decoder.emit_prob_dim),
      out_dim=model.align_target_dim,
      allow_broadcast=True
    )
    logits = rf.squeeze(logits, axis=model.blank_decoder.emit_prob_dim)

  return logits, non_blank_targets_spatial_dim_ext
