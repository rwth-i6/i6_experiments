from typing import Optional, Dict, Any, Tuple, Sequence
import tree

from returnn.tensor import Tensor, Dim, single_step_dim
from returnn.frontend.state import State
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.recog import RecogDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import GlobalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import SegmentalAttLabelDecoder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import get_score_lattice
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import GlobalAttDecoder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import recombination
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.beam_search import utils as beam_search_utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import (
  BlankDecoderV1,
  BlankDecoderV3,
  BlankDecoderV4,
  BlankDecoderV5,
  BlankDecoderV6,
  BlankDecoderV7,
  BlankDecoderV8,
  BlankDecoderV9,
)


def _get_init_trafo_state(model: TransformerDecoder, batch_dims: Sequence[Dim]) -> State:
  trafo_state = model.default_initial_state(batch_dims=batch_dims)
  for state in trafo_state:
    if state == "pos":
      trafo_state[state] = rf.zeros(batch_dims, dtype="int32")
    else:
      self_att_expand_dim = Dim(rf.zeros(batch_dims, dtype="int32"), name="self_att_expand_dim_init")
      trafo_state[state].self_att.accum_axis = self_att_expand_dim

      k_accum = trafo_state[state].self_att.k_accum  # type: rf.Tensor
      k_accum_raw = k_accum.raw_tensor
      trafo_state[state].self_att.k_accum = k_accum.copy_template_replace_dim_tag(
        k_accum.get_axis_from_description("stag:self_att_expand_dim_init"), self_att_expand_dim
      )
      trafo_state[state].self_att.k_accum.raw_tensor = k_accum_raw

      v_accum = trafo_state[state].self_att.v_accum  # type: rf.Tensor
      v_accum_raw = v_accum.raw_tensor
      trafo_state[state].self_att.v_accum = v_accum.copy_template_replace_dim_tag(
        v_accum.get_axis_from_description("stag:self_att_expand_dim_init"), self_att_expand_dim
      )
      trafo_state[state].self_att.v_accum.raw_tensor = v_accum_raw

  return trafo_state


def _get_masked_trafo_state(
        trafo_state: State,
        trafo_state_updated,
        update_state_mask: Tensor,
        backrefs: Tensor,
) -> State:
  for state in trafo_state:
    if state == "pos":
      trafo_state[state] = rf.where(
        update_state_mask,
        rf.gather(trafo_state_updated[state], indices=backrefs),
        rf.gather(trafo_state[state], indices=backrefs)
      )
    else:
      updated_accum_axis = trafo_state_updated[state].self_att.accum_axis

      updated_self_att_expand_dim_dyn_size_ext = rf.gather(updated_accum_axis.dyn_size_ext, indices=backrefs)
      masked_self_att_expand_dim_dyn_size_ext = rf.where(
        update_state_mask,
        updated_self_att_expand_dim_dyn_size_ext,
        updated_self_att_expand_dim_dyn_size_ext - 1
      )
      masked_self_att_expand_dim = Dim(masked_self_att_expand_dim_dyn_size_ext, name="self_att_expand_dim_init")
      trafo_state[state].self_att.accum_axis = masked_self_att_expand_dim

      def _mask_lm_state(tensor: rf.Tensor):
        tensor = rf.gather(tensor, indices=backrefs)
        tensor = tensor.copy_transpose(
          [updated_accum_axis] + tensor.remaining_dims(updated_accum_axis))
        tensor_raw = tensor.raw_tensor
        tensor_raw = tensor_raw[:rf.reduce_max(
          masked_self_att_expand_dim_dyn_size_ext,
          axis=masked_self_att_expand_dim_dyn_size_ext.dims
        ).raw_tensor.item()]
        tensor = tensor.copy_template_replace_dim_tag(
          tensor.get_axis_from_description(updated_accum_axis), masked_self_att_expand_dim
        )
        tensor.raw_tensor = tensor_raw
        return tensor

      trafo_state[state].self_att.k_accum = _mask_lm_state(trafo_state_updated[state].self_att.k_accum)
      trafo_state[state].self_att.v_accum = _mask_lm_state(trafo_state_updated[state].self_att.v_accum)

  return trafo_state


def update_state(
        model: SegmentalAttentionModel,
        update_state_mask: Tensor,
        backrefs: Tensor,
        label_decoder_state: State,
        label_decoder_state_updated: State,
        blank_decoder_state: Optional[State],
        blank_decoder_state_updated: Optional[State],
        lm_state: Optional[State],
        lm_state_updated: Optional[State],
        ilm_state: Optional[State],
        ilm_state_updated: Optional[State],
        aed_decoder_state: Optional[State],
        aed_decoder_state_updated: Optional[State],
) -> Tuple[State, Optional[State], Optional[State], Optional[State], Optional[State]]:

  # ------------------- update blank decoder state -------------------

  if blank_decoder_state is not None:
    blank_decoder_state = tree.map_structure(
      lambda s: rf.gather(s, indices=backrefs), blank_decoder_state_updated)

  # ------------------- update label decoder state and ILM state -------------------

  if model.label_decoder_state == "trafo":
    label_decoder_state = _get_masked_trafo_state(
      trafo_state=label_decoder_state,
      trafo_state_updated=label_decoder_state_updated,
      update_state_mask=update_state_mask,
      backrefs=backrefs
    )
  else:
    if model.label_decoder.trafo_att:
      trafo_att_state = label_decoder_state.pop("trafo_att")
      trafo_att_state_updated = label_decoder_state_updated.pop("trafo_att")

      trafo_att_state = _get_masked_trafo_state(
        trafo_state=trafo_att_state,
        trafo_state_updated=trafo_att_state_updated,
        update_state_mask=update_state_mask,
        backrefs=backrefs
      )

    def _get_masked_state(old, new, mask):
      old = rf.gather(old, indices=backrefs)
      new = rf.gather(new, indices=backrefs)
      return rf.where(mask, new, old)

    if model.label_decoder_state == "joint-lstm":
      label_decoder_state = tree.map_structure(
        lambda s: rf.gather(s, indices=backrefs), label_decoder_state_updated)
    else:
      label_decoder_state = tree.map_structure(
        lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
        label_decoder_state, label_decoder_state_updated
      )

    if model.label_decoder.trafo_att:
      label_decoder_state["trafo_att"] = trafo_att_state

  # ILM
  if ilm_state is not None:
    ilm_state = tree.map_structure(
      lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
      ilm_state, ilm_state_updated
    )

  # ------------------- update external AED state -------------------

  if aed_decoder_state is not None:
    aed_decoder_state = tree.map_structure(
      lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
      aed_decoder_state, aed_decoder_state_updated
    )

  # ------------------- update external LM state -------------------

  if lm_state is not None:
    lm_state = _get_masked_trafo_state(
      trafo_state=lm_state,
      trafo_state_updated=lm_state_updated,
      update_state_mask=update_state_mask,
      backrefs=backrefs
    )

  return label_decoder_state, blank_decoder_state, lm_state, ilm_state, aed_decoder_state


def get_score(
        model: SegmentalAttentionModel,
        i: int,
        chunk_idx: Optional[Tensor],
        num_windows: Optional[Tensor],
        input_embed_label_model: Optional[Tensor],
        input_embed_blank_model: Optional[Tensor],
        input_embed_aed_model: Optional[Tensor],
        nb_target: Tensor,
        emit_positions: Tensor,
        label_decoder_state: State,
        blank_decoder_state: Optional[State],
        lm_state: Optional[State],
        ilm_state: Optional[State],
        ilm_type: Optional[str],
        aed_decoder_state: Optional[State],
        enc_args: Dict[str, Tensor],
        att_enc_args: Dict[str, Tensor],
        aed_enc_args: Dict[str, Tensor],
        enc_spatial_dim: Dim,
        beam_dim: Dim,
        batch_dims: Sequence[Dim],
        base_model_scale: float,
        external_lm_scale: Optional[float] = None,
        ilm_correction_scale: Optional[float] = None,
        external_aed_scale: Optional[float] = None,
        subtract_ilm_eos_score: bool = False,
        separate_readout_alpha: Optional[float] = None,
) -> Tuple[Tensor, State, Optional[State], Optional[State], Optional[State], Optional[State]]:
  # ------------------- label step -------------------

  center_positions = rf.minimum(
    rf.full(dims=[beam_dim] + batch_dims, fill_value=i, dtype="int32"),
    rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1)
  )
  if model.center_window_size is not None:
    if chunk_idx is None:
      segment_starts = rf.maximum(
        rf.convert_to_tensor(0, dtype="int32"), center_positions - model.center_window_size // 2)
      segment_ends = rf.minimum(
        rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1),
        center_positions + model.center_window_size // 2
      )
      segment_lens = segment_ends - segment_starts + 1
    else:
      segment_starts = chunk_idx * model.window_step_size
      segment_ends = rf.minimum(
        rf.copy_to_device(enc_spatial_dim.get_size_tensor()),
        segment_starts + model.center_window_size
      )
      segment_lens = segment_ends - segment_starts
  else:
    segment_starts = None
    segment_lens = None

  if model.label_decoder_state == "trafo":
    label_logits, label_decoder_state, label_step_s_out = model.label_decoder(
      nb_target,
      spatial_dim=single_step_dim,
      encoder=att_enc_args["enc_transformed"],
      state=label_decoder_state,
    )
  else:
    if model.center_window_size is None:
      label_step_out, label_decoder_state = model.label_decoder.loop_step(
        **att_enc_args,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed_label_model,
        state=label_decoder_state,
      )
    else:
      label_step_out, label_decoder_state = model.label_decoder.loop_step(
        **att_enc_args,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed_label_model,
        segment_lens=segment_lens,
        segment_starts=segment_starts,
        center_positions=center_positions,
        state=label_decoder_state,
      )
    label_step_s_out = label_step_out["s"]

    if not isinstance(model.label_decoder, TransformerDecoder) and (
            model.label_decoder.use_current_frame_in_readout or
            model.label_decoder.use_current_frame_in_readout_w_gate or
            model.label_decoder.use_current_frame_in_readout_random or
            model.label_decoder.use_current_frame_in_readout_w_double_gate or
            model.label_decoder.use_sep_h_t_readout
    ):
      h_t = rf.gather(enc_args["enc"], axis=enc_spatial_dim, indices=center_positions)
    else:
      h_t = None

    label_logits, h_t_logits = model.label_decoder.decode_logits(
      input_embed=input_embed_label_model,
      s=label_step_out["s"],
      att=label_step_out["att"],
      h_t=h_t
    )

  if model.label_decoder_state != "trafo" and model.label_decoder.separate_blank_from_softmax:
    label_log_prob = utils.log_softmax_sep_blank(
      logits=label_logits, blank_idx=model.blank_idx, target_dim=model.target_dim)
  else:
    label_log_prob = rf.log_softmax(label_logits, axis=model.target_dim)

  # combine two softmaxes in case of the random readout
  if not isinstance(model.label_decoder, TransformerDecoder) and model.label_decoder.use_current_frame_in_readout_random:
    label_logits2 = model.label_decoder.decode_logits(
      input_embed=input_embed_label_model,
      s=label_step_out["s"],
      att=label_step_out["att"],
    )
    label_log_prob2 = rf.log_softmax(label_logits2, axis=model.target_dim)
    alpha = separate_readout_alpha
    label_log_prob = alpha * label_log_prob + (1 - alpha) * label_log_prob2

  # combine two softmaxes in case of two separate readouts
  if not isinstance(model.label_decoder, TransformerDecoder) and model.label_decoder.use_sep_h_t_readout:
    h_t_label_log_prob = rf.log_softmax(h_t_logits, axis=model.target_dim)
    alpha = separate_readout_alpha
    label_log_prob = alpha * label_log_prob + (1 - alpha) * h_t_label_log_prob

  label_log_prob *= base_model_scale
  # print("label_log_prob", label_log_prob.raw_tensor)

  # ------------------- external AED step -------------------
  aed_eos_log_prob = rf.zeros(batch_dims, dtype="float32")
  if aed_decoder_state is not None:
    aed_step_out, aed_decoder_state = model.aed_model.label_decoder.loop_step(
      **aed_enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed_aed_model,
      state=aed_decoder_state,
    )
    aed_logits, _ = model.aed_model.label_decoder.decode_logits(input_embed=input_embed_aed_model, **aed_step_out)
    aed_label_log_prob = rf.log_softmax(aed_logits, axis=model.target_dim)

    # do not apply LM scores to blank
    if model.use_joint_model:
      aed_label_log_prob_ = rf.where(
        rf.range_over_dim(model.target_dim) == model.blank_idx,
        rf.zeros(batch_dims, dtype="float32"),
        aed_label_log_prob
      )
      aed_label_log_prob = rf.where(
        rf.convert_to_tensor(i == rf.copy_to_device(enc_spatial_dim.get_size_tensor()) - 1),
        aed_label_log_prob,
        aed_label_log_prob_
      )
    else:
      aed_eos_log_prob = rf.where(
        rf.convert_to_tensor(
          i == rf.copy_to_device(enc_spatial_dim.get_size_tensor()) - 1),
        rf.gather(
          aed_label_log_prob,
          indices=rf.constant(model.aed_model.eos_idx, dtype="int32", dims=batch_dims, sparse_dim=nb_target.sparse_dim)
        ),
        aed_eos_log_prob
      )

    label_log_prob += external_aed_scale * aed_label_log_prob

  # ------------------- external LM step -------------------

  lm_eos_log_prob = rf.zeros(batch_dims, dtype="float32")
  if lm_state is not None:
    lm_logits, lm_state, = model.language_model(
      nb_target,
      spatial_dim=single_step_dim,
      state=lm_state,
    )
    lm_label_log_prob = rf.log_softmax(lm_logits, axis=model.target_dim)
    # print("chunk_idx", chunk_idx.raw_tensor)
    # print("lm_label_log_prob", lm_label_log_prob.raw_tensor)

    # do not apply LM scores to blank
    if model.use_joint_model:
      # set EOS log prob of LM to 0
      lm_label_log_prob_ = rf.where(
        rf.range_over_dim(model.target_dim) == model.blank_idx,
        rf.zeros(batch_dims, dtype="float32"),
        lm_label_log_prob
      )
      # keep EOS log prob of LM in case of last chunk/ last frame
      if chunk_idx is None:
        lm_label_log_prob = rf.where(
          rf.convert_to_tensor(i == rf.copy_to_device(enc_spatial_dim.get_size_tensor()) - 1),
          lm_label_log_prob,
          lm_label_log_prob_
        )
      else:
        # print("chunk_idx", chunk_idx.raw_tensor)
        # print("num_windows", num_windows.raw_tensor)
        lm_label_log_prob = rf.where(
          chunk_idx == num_windows - 1,
          lm_label_log_prob,
          lm_label_log_prob_
        )
        # print("lm_label_log_prob", lm_label_log_prob.raw_tensor)
    else:
      lm_eos_log_prob = rf.where(
        rf.convert_to_tensor(
          i == rf.copy_to_device(enc_spatial_dim.get_size_tensor()) - 1),
        # TODO: change to non hard-coded BOS index
        rf.gather(lm_label_log_prob, indices=rf.constant(0, dtype="int32", dims=batch_dims, sparse_dim=nb_target.sparse_dim)),
        lm_eos_log_prob
      )

    label_log_prob += external_lm_scale * lm_label_log_prob

  # --------------------------------- ILM step ---------------------------------

  ilm_eos_log_prob = rf.zeros(batch_dims, dtype="float32")
  if ilm_state is not None:
    ilm_step_out, ilm_state = model.label_decoder.loop_step(
      **att_enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed_label_model,
      segment_lens=segment_lens,
      segment_starts=segment_starts,
      center_positions=center_positions,
      state=ilm_state,
      use_mini_att=ilm_type == "mini_att",
      use_zero_att=ilm_type == "zero_att"
    )
    ilm_logits, _ = model.label_decoder.decode_logits(
      input_embed=input_embed_label_model,
      s=ilm_step_out["s"],
      att=ilm_step_out["att"],
      h_t=h_t
    )
    ilm_label_log_prob = rf.log_softmax(ilm_logits, axis=model.target_dim)

    # do not apply ILM correction to blank
    if model.use_joint_model:
      ilm_label_log_prob_ = rf.where(
        rf.range_over_dim(model.target_dim) == model.blank_idx,
        rf.zeros(batch_dims, dtype="float32"),
        ilm_label_log_prob
      )
      if subtract_ilm_eos_score:
        if chunk_idx is None:
          ilm_label_log_prob = rf.where(
            rf.convert_to_tensor(i == rf.copy_to_device(enc_spatial_dim.get_size_tensor()) - 1),
            ilm_label_log_prob,
            ilm_label_log_prob_
          )
        else:
          ilm_label_log_prob = rf.where(
            chunk_idx == num_windows - 1,
            ilm_label_log_prob,
            ilm_label_log_prob_
          )
      else:
        ilm_label_log_prob = ilm_label_log_prob_.copy()
    else:
      ilm_eos_log_prob = rf.where(
        rf.convert_to_tensor(
          i == rf.copy_to_device(enc_spatial_dim.get_size_tensor()) - 1),
        # TODO: change to non hard-coded BOS index
        rf.gather(ilm_label_log_prob,
                  indices=rf.constant(0, dtype="int32", dims=batch_dims, sparse_dim=nb_target.sparse_dim)),
        ilm_eos_log_prob
      )

    label_log_prob -= ilm_correction_scale * ilm_label_log_prob

  # ------------------- blank step -------------------

  if blank_decoder_state is not None:
    if model.blank_decoder_version in (1, 3):
      blank_loop_step_kwargs = dict(
        enc=enc_args["enc"],
        enc_spatial_dim=enc_spatial_dim,
        state=blank_decoder_state,
      )
      if isinstance(model.blank_decoder, BlankDecoderV1):
        blank_loop_step_kwargs["input_embed"] = input_embed_blank_model
      else:
        blank_loop_step_kwargs["label_model_state"] = label_step_s_out

      blank_step_out, blank_decoder_state = model.blank_decoder.loop_step(**blank_loop_step_kwargs)
      blank_logits = model.blank_decoder.decode_logits(**blank_step_out)
    else:
      enc_position = rf.minimum(
        rf.full(dims=batch_dims, fill_value=i, dtype="int32"),
        rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1)
      )
      enc_frame = rf.gather(enc_args["enc"], indices=enc_position, axis=enc_spatial_dim)
      enc_frame = rf.expand_dim(enc_frame, beam_dim)
      if isinstance(model.blank_decoder, BlankDecoderV4):
        blank_logits = model.blank_decoder.decode_logits(enc=enc_frame, label_model_states_unmasked=label_step_s_out)
      elif isinstance(model.blank_decoder, BlankDecoderV8):
        blank_logits = model.blank_decoder.decode_logits(enc=enc_frame)
      elif isinstance(model.blank_decoder, BlankDecoderV9):
        energy_in = rf.gather(
          energy_in,
          indices=rf.constant(i, dtype="int32", dims=batch_dims),
          axis=enc_spatial_dim
        )
        blank_logits = model.blank_decoder.decode_logits(energy_in=energy_in)
      elif isinstance(model.blank_decoder, BlankDecoderV5):
        # no LSTM -> no state -> just leave (empty) state as is
        blank_logits = model.blank_decoder.emit_prob(
          rf.concat_features(enc_frame, label_step_s_out))
      elif isinstance(model.blank_decoder, BlankDecoderV6):
        prev_lstm_state = blank_decoder_state.s_blank
        blank_decoder_state = rf.State()
        s_blank, blank_decoder_state.s_blank = model.blank_decoder.s(
          enc_frame,
          state=prev_lstm_state,
          spatial_dim=single_step_dim
        )
        blank_logits = model.blank_decoder.emit_prob(rf.concat_features(s_blank, label_step_s_out))
      else:
        assert isinstance(model.blank_decoder, BlankDecoderV7)
        prev_emit_distance = i - emit_positions - 1
        prev_emit_distance = rf.clip_by_value(
          prev_emit_distance, 0, model.blank_decoder.distance_dim.dimension - 1)
        prev_emit_distance.sparse_dim = model.blank_decoder.distance_dim
        blank_logits = model.blank_decoder.decode_logits(
          enc=enc_frame, label_model_states_unmasked=label_step_s_out, prev_emit_distances=prev_emit_distance)

    emit_log_prob = rf.log(rf.sigmoid(blank_logits))
    emit_log_prob = rf.squeeze(emit_log_prob, axis=emit_log_prob.feature_dim)
    blank_log_prob = rf.log(rf.sigmoid(-blank_logits))
    blank_log_prob += lm_eos_log_prob + aed_eos_log_prob
    if subtract_ilm_eos_score:
      blank_log_prob -= ilm_eos_log_prob

    # ------------------- combination -------------------

    label_log_prob += emit_log_prob
    output_log_prob, _ = rf.concat(
      (label_log_prob, model.target_dim), (blank_log_prob, blank_log_prob.feature_dim),
      out_dim=model.align_target_dim
    )
  else:
    output_log_prob = label_log_prob

  return output_log_prob, label_decoder_state, blank_decoder_state, lm_state, ilm_state, aed_decoder_state


def model_recog(
        *,
        model: SegmentalAttentionModel,
        data: Tensor,
        data_spatial_dim: Dim,
        beam_size: int,
        use_recombination: Optional[str] = None,
        base_model_scale: float = 1.0,
        external_lm_scale: Optional[float] = None,
        external_aed_scale: Optional[float] = None,
        ilm_type: Optional[str] = None,
        ilm_correction_scale: Optional[float] = None,
        subtract_ilm_eos_score: bool = False,
        cheating_targets: Optional[Tensor] = None,
        cheating_targets_spatial_dim: Optional[Dim] = None,
        return_non_blank_seqs: bool = True,
        separate_readout_alpha: Optional[float] = None,
        length_normalization_exponent: float = 0.0,
) -> Tuple[Tensor, Tensor, Dim, Tensor, Dim, Dim]:
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
  assert any(
    isinstance(model.blank_decoder, cls) for cls in (
      BlankDecoderV1,
      BlankDecoderV3,
      BlankDecoderV4,
      BlankDecoderV5,
      BlankDecoderV6,
      BlankDecoderV7,
      BlankDecoderV8,
      BlankDecoderV9,
    )
  ) or model.blank_decoder is None, "blank_decoder not supported"
  if model.blank_decoder is None:
    assert model.use_joint_model, "blank_decoder is None, so use_joint_model must be True"
  if model.language_model:
    assert external_lm_scale is not None, "external_lm_scale must be defined with LM"
  assert model.label_decoder_state in {"nb-lstm", "joint-lstm", "nb-2linear-ctx1", "trafo"}

  assert (cheating_targets is None) == (cheating_targets_spatial_dim is None)

  # --------------------------------- init encoder, dims, etc ---------------------------------
  encoders = [model.encoder]
  if model.att_encoder:
    encoders.append(model.att_encoder)

  enc_args_list = []
  enc_spatial_dim_list = []
  for encoder in encoders:
    enc_args, enc_spatial_dim = encoder.encode(data, in_spatial_dim=data_spatial_dim)
    if model.label_decoder_state == "trafo":
      enc_args["enc_transformed"] = model.label_decoder.transform_encoder(enc_args["enc"], axis=enc_spatial_dim)

    enc_args_list.append(enc_args)
    enc_spatial_dim_list.append(enc_spatial_dim)

  enc_spatial_dim = enc_spatial_dim_list[0]
  enc_args = enc_args_list[0]
  if model.att_encoder:
    att_enc_args = enc_args_list[1]

    att_enc_args["enc"] = utils.copy_tensor_replace_dim_tag(
      att_enc_args["enc"], enc_spatial_dim_list[1], enc_spatial_dim)
    att_enc_args["enc_ctx"] = utils.copy_tensor_replace_dim_tag(
      att_enc_args["enc_ctx"], enc_spatial_dim_list[1], enc_spatial_dim)
  else:
    att_enc_args = enc_args

  enc_spatial_sizes = enc_spatial_dim.get_size_tensor()
  max_seq_len = rf.reduce_max(enc_spatial_sizes, axis=enc_spatial_sizes.dims)

  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims
  backrefs = rf.zeros(batch_dims_, dtype="int32")

  bos_idx = model.bos_idx

  ended = rf.constant(False, dims=batch_dims_)
  out_seq_len = rf.constant(0, dims=batch_dims_, dtype="int32")
  seq_log_prob = rf.constant(0.0, dims=batch_dims_)

  # for blank decoder v7
  emit_positions = rf.full(dims=batch_dims_, fill_value=-1, dtype="int32")

  if use_recombination:
    assert len(batch_dims) == 1
    assert use_recombination in {"sum", "max"}
    seq_hash = rf.constant(0, dims=batch_dims_, dtype="int64")
  else:
    seq_hash = None

  # lists of [B, beam] tensors
  seq_targets = []
  seq_backrefs = []

  update_state_mask = rf.constant(True, dims=batch_dims_)

  output_dim = model.target_dim if model.use_joint_model else model.align_target_dim
  if model.label_decoder_state == "trafo":
    target_non_blank_dim = model.label_decoder.vocab_dim
  else:
    target_non_blank_dim = model.target_dim

  # --------------------------------- init states ---------------------------------

  # label decoder
  if model.label_decoder_state == "trafo":
    label_decoder_state = _get_init_trafo_state(model.label_decoder, batch_dims_)
  else:
    if isinstance(model.label_decoder, GlobalAttDecoder):
      label_decoder_state = model.label_decoder.decoder_default_initial_state(
        batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)

      if model.label_decoder.trafo_att:
        label_decoder_state.trafo_att = _get_init_trafo_state(model.label_decoder.trafo_att, batch_dims_)
    else:
      assert isinstance(model.label_decoder, SegmentalAttLabelDecoder)
      label_decoder_state = model.label_decoder.default_initial_state(batch_dims=batch_dims_)

  # blank decoder
  if model.blank_decoder is not None:
    blank_decoder_state = model.blank_decoder.default_initial_state(batch_dims=batch_dims_)
  else:
    blank_decoder_state = None

  # external LM
  if model.language_model:
    lm_state = _get_init_trafo_state(model.language_model, batch_dims_)
  else:
    lm_state = None

  # ILM
  if ilm_type is not None:
    ilm_state = model.label_decoder.default_initial_state(
      batch_dims=batch_dims_,
      use_mini_att=ilm_type == "mini_att",
      use_zero_att=ilm_type == "zero_att"
    )
  else:
    ilm_state = None

  # external aed model
  if model.aed_model:
    aed_enc_args, aed_enc_spatial_dim = model.aed_model.encoder.encode(data, in_spatial_dim=data_spatial_dim)
    aed_enc_args["enc"] = utils.copy_tensor_replace_dim_tag(aed_enc_args["enc"], aed_enc_spatial_dim, enc_spatial_dim)
    aed_enc_args["enc_ctx"] = utils.copy_tensor_replace_dim_tag(
      aed_enc_args["enc_ctx"], aed_enc_spatial_dim, enc_spatial_dim)
    aed_decoder_state = model.aed_model.label_decoder.decoder_default_initial_state(
      batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
    input_embed_aed = rf.zeros(
      batch_dims_ + [model.aed_model.label_decoder.target_embed.out_dim],
      feature_dim=model.aed_model.label_decoder.target_embed.out_dim,
      dtype="float32"
    )
  else:
    aed_decoder_state = None
    input_embed_aed = None
    aed_enc_args = None

  # --------------------------------- init targets, embeddings ---------------------------------

  if model.use_joint_model:
    target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    target_non_blank = target.copy()
  else:
    target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.align_target_dim)
    target_non_blank = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)

  if model.label_decoder_state != "trafo":
    input_embed = rf.zeros(
      batch_dims_ + [model.label_decoder.target_embed.out_dim],
      feature_dim=model.label_decoder.target_embed.out_dim,
      dtype="float32"
    )
  else:
    input_embed = None

  if isinstance(model.blank_decoder, BlankDecoderV1):
    input_embed_length_model = rf.zeros(
      batch_dims_ + [model.blank_decoder.target_embed.out_dim], feature_dim=model.blank_decoder.target_embed.out_dim)
  else:
    input_embed_length_model = None

  # --------------------------------- cheating targets ---------------------------------

  if cheating_targets is not None:
    # add blank idx on the right
    # this way, when the label index for gathering reached the last non-blank index, it will gather blank after that
    # which then only allows corresponding hypotheses to be extended by blank
    cheating_targets_padded, cheating_targets_padded_spatial_dim = rf.pad(
      cheating_targets,
      axes=[cheating_targets_spatial_dim],
      padding=[(0, 1)],
      value=model.blank_idx,
    )
    cheating_targets_padded_spatial_dim = cheating_targets_padded_spatial_dim[0]

    # rf.pad falsely pads right after the padding. this means that for shorter seqs, the padding is behind the padding.
    cheating_targets_padded = rf.where(
      rf.range_over_dim(cheating_targets_padded_spatial_dim) < rf.copy_to_device(cheating_targets_padded_spatial_dim.dyn_size_ext) - 1,
      cheating_targets_padded,
      model.blank_idx
    )

    cheating_targets_padded_spatial_sizes = rf.copy_to_device(cheating_targets_padded_spatial_dim.dyn_size_ext)
    cheating_targets_spatial_sizes = rf.copy_to_device(cheating_targets_spatial_dim.dyn_size_ext)
    max_num_labels = rf.reduce_max(
      cheating_targets_spatial_sizes, axis=cheating_targets_spatial_sizes.dims
    ).raw_tensor.item()
    single_col_dim = Dim(dimension=max_num_labels + 1, name="max-num-labels")
    label_indices = rf.zeros(batch_dims_, dtype="int32", sparse_dim=single_col_dim)
    prev_label_indices = label_indices.copy()
    enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext)

    vocab_range = rf.range_over_dim(output_dim)
    blank_tensor = rf.convert_to_tensor(model.blank_idx, dtype=vocab_range.dtype)

  # --------------------------------- main loop ---------------------------------

  i = 0
  if model.window_step_size != 1 or model.use_vertical_transitions:
    chunk_idx = rf.zeros(batch_dims_, dtype="int32")
    num_windows = enc_spatial_sizes // model.window_step_size
    num_windows += enc_spatial_sizes % model.window_step_size > 0
    num_windows = rf.copy_to_device(num_windows)  # for some reason, num_windows is still on CPU otherwise...
    while_cond = "True"
  else:
    chunk_idx = None
    num_windows = None
    while_cond = f"i < {max_seq_len.raw_tensor}"
  while eval(while_cond):
    # print("i", i)
    if i > 0:
      target_non_blank = rf.where(update_state_mask, target, rf.gather(target_non_blank, indices=backrefs))
      target_non_blank.sparse_dim = target_non_blank_dim

      if model.label_decoder_state != "trafo":
        if model.label_decoder_state == "joint-lstm":
          input_embed = model.label_decoder.target_embed(target)
        else:
          input_embed = rf.where(
            update_state_mask,
            model.label_decoder.target_embed(target_non_blank),
            rf.gather(input_embed, indices=backrefs)
          )
      if model.aed_model:
        input_embed_aed = rf.where(
          update_state_mask,
          model.aed_model.label_decoder.target_embed(target_non_blank),
          rf.gather(input_embed_aed, indices=backrefs)
        )
      if isinstance(model.blank_decoder, BlankDecoderV1):
        input_embed_length_model = model.blank_decoder.target_embed(target)

      emit_positions = rf.where(
        update_state_mask,
        rf.full(dims=batch_dims, fill_value=i - 1, dtype="int32"),
        rf.gather(emit_positions, indices=backrefs)
      )

      if cheating_targets is not None:
        label_indices = rf.where(
          update_state_mask,
          rf.where(
            prev_label_indices == cheating_targets_padded_spatial_sizes - 1,
            prev_label_indices,
            prev_label_indices + 1
          ),
          prev_label_indices
        )

    (
      output_log_prob,
      label_decoder_state_updated,
      blank_decoder_state_updated,
      lm_state_updated,
      ilm_state_updated,
      aed_decoder_state_updated
    ) = get_score(
      model=model,
      i=i,
      chunk_idx=chunk_idx,
      num_windows=num_windows,
      input_embed_label_model=input_embed,
      input_embed_blank_model=input_embed_length_model,
      input_embed_aed_model=input_embed_aed,
      nb_target=target_non_blank,
      emit_positions=emit_positions,
      label_decoder_state=label_decoder_state,
      blank_decoder_state=blank_decoder_state,
      lm_state=lm_state,
      ilm_state=ilm_state,
      ilm_type=ilm_type,
      aed_decoder_state=aed_decoder_state,
      enc_args=enc_args,
      att_enc_args=att_enc_args,
      aed_enc_args=aed_enc_args,
      enc_spatial_dim=enc_spatial_dim,
      beam_dim=beam_dim,
      batch_dims=batch_dims,
      base_model_scale=base_model_scale,
      external_lm_scale=external_lm_scale,
      ilm_correction_scale=ilm_correction_scale,
      external_aed_scale=external_aed_scale,
      subtract_ilm_eos_score=subtract_ilm_eos_score,
      separate_readout_alpha=separate_readout_alpha,
    )

    # ------------------- filter finished beam --------------------

    if chunk_idx is not None:
      # for chunked, beam is finished if ended is True
      output_log_prob = rf.where(
        ended,
        rf.sparse_to_dense(model.blank_idx, axis=model.target_dim, label_value=0.0, other_value=-1.0e30),
        output_log_prob
      )
    else:
      # for time-sync, beam is finished if i == enc_spatial_dim
      # for shorter seqs in the batch, set the blank score to zero and the others to ~-inf
      output_log_prob = rf.where(
        rf.convert_to_tensor(i >= rf.copy_to_device(enc_spatial_dim.get_size_tensor())),
        rf.sparse_to_dense(
          model.blank_idx,
          axis=output_dim,
          label_value=0.0,
          other_value=-1.0e30
        ),
        output_log_prob
      )

    if cheating_targets is not None:
      label_ground_truth = rf.gather(
        cheating_targets_padded,
        indices=label_indices,
        axis=cheating_targets_padded_spatial_dim,
        clip_to_valid=True
      )
      # mask label log prob in order to only allow hypotheses corresponding to the ground truth:
      # log prob needs to correspond to the next non-blank label...
      output_log_prob_mask = vocab_range == label_ground_truth
      rem_frames = enc_spatial_sizes - i
      rem_labels = cheating_targets_spatial_sizes - label_indices
      # ... or to blank if there are more frames than labels left
      output_log_prob_mask = rf.logical_or(
        output_log_prob_mask,
        rf.logical_and(
          vocab_range == blank_tensor,
          rem_frames > rem_labels
        )
      )
      output_log_prob = rf.where(
        output_log_prob_mask,
        output_log_prob,
        rf.constant(-1.0e30, dims=batch_dims + [beam_dim, output_dim])
      )

    # ------------------- recombination -------------------

    if use_recombination:
      seq_log_prob, _ = recombination.recombine_seqs(
        seq_targets,
        seq_log_prob,
        seq_hash,
        beam_dim,
        batch_dims[0],
        use_sum=use_recombination == "sum"
      )

    # ------------------- top-k -------------------

    seq_log_prob = seq_log_prob + output_log_prob  # Batch, InBeam, Vocab

    # print("seq_log_prob", seq_log_prob.raw_tensor)
    # print("------------------------------------------------------------------")

    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob,
      k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
      axis=[beam_dim, output_dim]
    )
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    if cheating_targets is not None:
      prev_label_indices = rf.gather(label_indices, indices=backrefs)

    # ------------------- update hash for recombination -------------------

    if use_recombination:
      seq_hash = recombination.update_seq_hash(seq_hash, target, backrefs, model.blank_idx)

    # mask for updating label-sync states
    update_state_mask = rf.convert_to_tensor(target != model.blank_idx)

    # update chunk logic
    if chunk_idx is not None:
      chunk_idx = rf.gather(chunk_idx, indices=backrefs)
      # update the seq length before we update "ended", because we do not want to cut off the last EOC label
      # otherwise, we get problems with the utils.get_non_blank_mask function below
      out_seq_len = out_seq_len + rf.where(ended, 0, 1)
      out_seq_len = rf.gather(out_seq_len, indices=backrefs)
      ended = rf.gather(ended, indices=backrefs)
      ended = rf.logical_or(
        ended,
        rf.logical_and(
          ~update_state_mask,
          chunk_idx >= num_windows - 1
        )
      )
      ended = rf.logical_or(
        ended,
        rf.copy_to_device(i >= enc_spatial_sizes * 2)
      )

      if i > 1 and length_normalization_exponent != 0:
        # Length-normalized scores, so we evaluate score_t/len.
        # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
        # Because we count with EOS symbol, shifted by one.
        seq_log_prob *= rf.where(
          ended,
          (i / (i - 1)) ** length_normalization_exponent,
          1.0,
        )

      # update chunk idx, if model output blank and chunk_idx < num_windows - 1
      chunk_idx = rf.where(
        rf.logical_and(~update_state_mask, chunk_idx < num_windows - 1),
        chunk_idx + 1,
        chunk_idx
      )

      # Backtrack via backrefs, resolve beams.
      seq_targets_ = []
      indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
      for backrefs_, target_ in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target_, indices=indices))
        indices = rf.gather(backrefs_, indices=indices)  # FinalBeam -> PrevBeam

      i += 1
      if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
        break

    label_decoder_state, blank_decoder_state, lm_state, ilm_state, aed_decoder_state = update_state(
      model=model,
      update_state_mask=update_state_mask,
      backrefs=backrefs,
      label_decoder_state=label_decoder_state,
      label_decoder_state_updated=label_decoder_state_updated,
      blank_decoder_state=blank_decoder_state,
      blank_decoder_state_updated=blank_decoder_state_updated,
      lm_state=lm_state,
      lm_state_updated=lm_state_updated,
      ilm_state=ilm_state,
      ilm_state_updated=ilm_state_updated,
      aed_decoder_state=aed_decoder_state,
      aed_decoder_state_updated=aed_decoder_state_updated
    )

  # last recombination
  if use_recombination:
    seq_log_prob, _ = recombination.recombine_seqs(
      seq_targets,
      seq_log_prob,
      seq_hash,
      beam_dim,
      batch_dims[0],
      use_sum=use_recombination == "sum"
    )

  if i > 0 and length_normalization_exponent != 0:
    seq_log_prob *= (1 / i) ** length_normalization_exponent

  best_hyps = rf.reduce_argmax(seq_log_prob, axis=beam_dim)

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
  if model.window_step_size == 1 and not model.use_vertical_transitions:
    out_spatial_dim = enc_spatial_dim
    best_seq_targets_spatial_dim = enc_spatial_dim
  else:
    out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    best_seq_targets_spatial_dim = out_spatial_dim.copy()
    best_seq_targets_spatial_dim.dyn_size_ext = rf.gather(
      out_spatial_dim.dyn_size_ext,
      indices=best_hyps,
      axis=beam_dim
    )

  seq_targets = seq_targets__.stack(axis=out_spatial_dim)

  # if return_non_blank_seqs:
  non_blank_targets, non_blank_targets_spatial_dim = utils.get_masked(
    seq_targets,
    utils.get_non_blank_mask(seq_targets, model.blank_idx),
    out_spatial_dim,
    [beam_dim] + batch_dims,
  )
  non_blank_targets.sparse_dim = model.target_dim

  best_alignment = rf.gather(
    seq_targets,
    indices=best_hyps,
    axis=beam_dim,
  )
  best_alignment = utils.copy_tensor_replace_dim_tag(
    best_alignment, out_spatial_dim, best_seq_targets_spatial_dim
  )

  return best_alignment, seq_log_prob, best_seq_targets_spatial_dim, non_blank_targets, non_blank_targets_spatial_dim, beam_dim
