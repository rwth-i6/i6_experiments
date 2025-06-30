from typing import Optional, Dict, Any, Tuple, Sequence

import torch
import tree

import numpy as np
from torch import dtype

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


def get_max_num_recombinations(
        batch_dim: Dim,
        S: Tensor,
        T: Tensor,
):
  max_recomb_paths = rf.zeros(dims=[batch_dim], dtype="float32")
  for b in range(batch_dim.dyn_size_ext.raw_tensor.item()):
    S_ = S.raw_tensor[b].item() + 1  # noqa
    T_ = T.raw_tensor[b].item()  # noqa
    lattice = torch.full((S_, T_), dtype=torch.float32, fill_value=-torch.inf)
    lattice[0, :T_ - S_ + 1] = 0

    for s in range(1, S_):
      for t in range(1, T_):
        if (T_ - t) >= (S_ - s):
          lattice[s, t] = recombination.logsumexp(lattice[s - 1, t - 1], lattice[s, t - 1])

    max_recomb_paths.raw_tensor[b] = lattice[-1, -1]

  return max_recomb_paths


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


def get_score(
        model: SegmentalAttentionModel,
        i: int,
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
        aed_ilm_state: Optional[State],
        enc_args: Dict[str, Tensor],
        att_enc_args: Dict[str, Tensor],
        aed_enc_args: Dict[str, Tensor],
        ext_trans_enc_args: Dict[str, Tensor],
        enc_spatial_dim: Dim,
        beam_dim: Dim,
        batch_dims: Sequence[Dim],
        base_model_scale: float,
        external_lm_scale: Optional[float] = None,
        external_lm_eos_scale: Optional[float] = None,
        ilm_correction_scale: Optional[float] = None,
        external_aed_scale: Optional[float] = None,
        subtract_ilm_eos_score: bool = False,
        att_readout_scale: Optional[float] = None,
        h_t_readout_scale: Optional[float] = None,
        blank_penalty: Optional[float] = None,
        blank_scale: float = 1.0,
        emit_scale: float = 1.0,
        ext_trans_scale: Optional[float] = None,
        ext_trans_label_decoder_state: Optional[State] = None,
        ext_trans_ilm_state: Optional[State] = None,
        input_embed_ext_trans: Optional[Tensor] = None,
) -> Tuple[Tensor, State, Optional[State], Optional[State], Optional[State], Optional[State], Optional[State], Optional[State], Optional[State], Tensor]:
  # ------------------- label step -------------------

  center_positions = rf.minimum(
    rf.full(dims=[beam_dim] + batch_dims, fill_value=i, dtype="int32"),
    rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1)
  )
  if model.center_window_size is not None:
    segment_starts = rf.maximum(
      rf.convert_to_tensor(0, dtype="int32"), center_positions - model.center_window_size // 2)
    segment_ends = rf.minimum(
      rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1),
      center_positions + model.center_window_size // 2
    )
    segment_lens = segment_ends - segment_starts + 1
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
    alpha = att_readout_scale
    label_log_prob = alpha * label_log_prob + (1 - alpha) * label_log_prob2

  # combine two softmaxes in case of two separate readouts
  if not isinstance(model.label_decoder, TransformerDecoder) and model.label_decoder.use_sep_h_t_readout:
    h_t_label_log_prob = rf.log_softmax(h_t_logits, axis=model.target_dim)
    label_log_prob = att_readout_scale * label_log_prob + h_t_readout_scale * h_t_label_log_prob

  label_log_prob *= base_model_scale

  # ------------------- external AED step -------------------
  aed_eos_log_prob = rf.zeros(batch_dims, dtype="float32")
  if aed_decoder_state is not None:
    if model.aed_model.decoder_state != "trafo":
      aed_step_out, aed_decoder_state = model.aed_model.label_decoder.loop_step(
        **aed_enc_args,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed_aed_model,
        state=aed_decoder_state,
      )
      aed_logits, _ = model.aed_model.label_decoder.decode_logits(
        input_embed=input_embed_aed_model,
        att=aed_step_out["att"],
        s=aed_step_out["s"],
      )
    else:
      aed_logits, aed_decoder_state = model.aed_model.label_decoder(
        nb_target,
        spatial_dim=single_step_dim,
        encoder=aed_enc_args["enc"],
        state=aed_decoder_state,
      )
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

  # ------------------- external transducer step -------------------
  if ext_trans_label_decoder_state is not None:
    ext_trans_label_step_out, ext_trans_label_decoder_state = model.external_transducer_model.label_decoder.loop_step(
      **ext_trans_enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed_ext_trans,
      segment_lens=segment_lens,
      segment_starts=segment_starts,
      center_positions=center_positions,
      state=ext_trans_label_decoder_state,
    )
    ext_trans_label_step_s_out = ext_trans_label_step_out["s"]

    ext_trans_label_logits, _ = model.external_transducer_model.label_decoder.decode_logits(
      input_embed=input_embed_ext_trans,
      s=ext_trans_label_step_out["s"],
      att=ext_trans_label_step_out["att"],
    )
    ext_trans_label_log_prob = rf.log_softmax(ext_trans_label_logits, axis=model.external_transducer_model.target_dim)
    label_log_prob += ext_trans_scale * ext_trans_label_log_prob

  # ------------------- external LM step -------------------

  lm_eos_log_prob = rf.zeros(batch_dims, dtype="float32")
  if lm_state is not None:
    lm_logits, lm_state = model.language_model(
      nb_target,
      spatial_dim=single_step_dim,
      state=lm_state,
    )
    lm_label_log_prob = rf.log_softmax(lm_logits, axis=model.target_dim)

    # do not apply LM scores to blank
    if model.use_joint_model:
      lm_label_log_prob_ = rf.where(
        rf.range_over_dim(model.target_dim) == model.blank_idx,
        rf.zeros(batch_dims, dtype="float32"),
        lm_label_log_prob
      )
      lm_label_log_prob = rf.where(
        rf.convert_to_tensor(i == rf.copy_to_device(enc_spatial_dim.get_size_tensor()) - 1),
        lm_label_log_prob,
        lm_label_log_prob_
      )
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
    if model.center_window_size is None:
      ilm_step_out, ilm_state = model.label_decoder.loop_step(
        **att_enc_args,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed_label_model,
        state=ilm_state,
        use_mini_att=ilm_type == "mini_att",
        use_zero_att=ilm_type == "zero_att",
      )
    else:
      ilm_step_out, ilm_state = model.label_decoder.loop_step(
        **att_enc_args,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed_label_model,
        segment_lens=segment_lens,
        segment_starts=segment_starts,
        center_positions=center_positions,
        state=ilm_state,
        use_mini_att=ilm_type == "mini_att",
        use_zero_att=ilm_type == "zero_att",
      )

    if h_t is None:
      ilm_h_t = None
    else:
      ilm_h_t = utils.copy_tensor_replace_dim_tag(ilm_step_out["att"], model.label_decoder.att_dim, model.label_decoder.enc_out_dim)
    ilm_logits, ilm_h_t_logits = model.label_decoder.decode_logits(
      input_embed=input_embed_label_model,
      s=ilm_step_out["s"],
      att=ilm_step_out["att"],  # mini LSTM
      h_t=ilm_h_t,
    )
    ilm_label_log_prob = rf.log_softmax(ilm_logits, axis=model.target_dim)

    # combine two softmaxes in case of two separate readouts
    if not isinstance(model.label_decoder, TransformerDecoder) and model.label_decoder.use_sep_h_t_readout:
      ilm_h_t_label_log_prob = rf.log_softmax(ilm_h_t_logits, axis=model.target_dim)
      ilm_label_log_prob = att_readout_scale * ilm_label_log_prob + h_t_readout_scale * ilm_h_t_label_log_prob

    # do not apply ILM correction to blank
    if model.use_joint_model:
      ilm_label_log_prob_ = rf.where(
        rf.range_over_dim(model.target_dim) == model.blank_idx,
        rf.zeros(batch_dims, dtype="float32"),
        ilm_label_log_prob
      )
      if subtract_ilm_eos_score:
        ilm_label_log_prob = rf.where(
          rf.convert_to_tensor(i == rf.copy_to_device(enc_spatial_dim.get_size_tensor()) - 1),
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

    label_log_prob -= base_model_scale * ilm_correction_scale * ilm_label_log_prob

  # --------------------------------- AED ILM step ---------------------------------

  aed_ilm_eos_log_prob = rf.zeros(batch_dims, dtype="float32")
  if aed_ilm_state is not None:
    aed_ilm_step_out, aed_ilm_state = model.aed_model.label_decoder.loop_step(
      **aed_enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed_aed_model,
      state=aed_ilm_state,
      use_mini_att=ilm_type == "mini_att",
      use_zero_att=ilm_type == "zero_att",
    )
    aed_ilm_logits, _ = model.aed_model.label_decoder.decode_logits(
      input_embed=input_embed_aed_model,
      s=aed_ilm_step_out["s"],
      att=aed_ilm_step_out["att"],
    )
    aed_ilm_label_log_prob = rf.log_softmax(aed_ilm_logits, axis=model.target_dim)

    # do not apply ILM correction to blank
    if model.use_joint_model:
      aed_ilm_label_log_prob_ = rf.where(
        rf.range_over_dim(model.target_dim) == model.blank_idx,
        rf.zeros(batch_dims, dtype="float32"),
        aed_ilm_label_log_prob
      )
      aed_ilm_label_log_prob = rf.where(
        rf.convert_to_tensor(i == rf.copy_to_device(enc_spatial_dim.get_size_tensor()) - 1),
        aed_ilm_label_log_prob,
        aed_ilm_label_log_prob_
      )
    else:
      aed_ilm_eos_log_prob = rf.where(
        rf.convert_to_tensor(
          i == rf.copy_to_device(enc_spatial_dim.get_size_tensor()) - 1),
        # TODO: change to non hard-coded BOS index
        rf.gather(aed_ilm_label_log_prob,
                  indices=rf.constant(0, dtype="int32", dims=batch_dims, sparse_dim=nb_target.sparse_dim)),
        aed_ilm_eos_log_prob
      )

    label_log_prob -= external_aed_scale * ilm_correction_scale * aed_ilm_label_log_prob

  # --------------------------------- ext transducer ILM step ---------------------------------

  if ext_trans_ilm_state is not None:
    ext_trans_ilm_step_out, ext_trans_ilm_state = model.external_transducer_model.label_decoder.loop_step(
      **ext_trans_enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed_ext_trans,
      segment_lens=segment_lens,
      segment_starts=segment_starts,
      center_positions=center_positions,
      state=ext_trans_ilm_state,
      use_mini_att=ilm_type == "mini_att",
      use_zero_att=ilm_type == "zero_att",
    )

    ext_trans_ilm_logits, _ = model.external_transducer_model.label_decoder.decode_logits(
      input_embed=input_embed_ext_trans,
      s=ext_trans_ilm_step_out["s"],
      att=ext_trans_ilm_step_out["att"],
    )
    ext_trans_ilm_label_log_prob = rf.log_softmax(ext_trans_ilm_logits, axis=model.target_dim)

    label_log_prob -= ext_trans_scale * ilm_correction_scale * ext_trans_ilm_label_log_prob

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
    emit_log_prob *= emit_scale

    blank_log_prob = rf.log(rf.sigmoid(-blank_logits))
    if blank_penalty:
      blank_log_prob -= blank_penalty
    blank_log_prob *= blank_scale

    blank_log_prob += external_lm_eos_scale * lm_eos_log_prob + aed_eos_log_prob
    if subtract_ilm_eos_score:
      blank_log_prob -= ilm_eos_log_prob

    # always subtract the AED ILM EOS score from blank
    if aed_ilm_state is not None:
      blank_log_prob -= aed_ilm_eos_log_prob

    # ------------------- external transducer blank step -------------------

    if ext_trans_label_decoder_state is not None:
      ext_trans_enc_frame = rf.gather(ext_trans_enc_args["enc"], indices=enc_position, axis=enc_spatial_dim)
      ext_trans_enc_frame = rf.expand_dim(ext_trans_enc_frame, beam_dim)
      ext_trans_blank_logits = model.external_transducer_model.blank_decoder.decode_logits(
        enc=ext_trans_enc_frame, label_model_states_unmasked=ext_trans_label_step_s_out)
      ext_trans_emit_log_prob = rf.log(rf.sigmoid(ext_trans_blank_logits))
      ext_trans_emit_log_prob = rf.squeeze(ext_trans_emit_log_prob, axis=ext_trans_emit_log_prob.feature_dim)
      ext_trans_blank_log_prob = rf.log(rf.sigmoid(-ext_trans_blank_logits))

      ext_trans_blank_log_prob = utils.copy_tensor_replace_dim_tag(
        ext_trans_blank_log_prob,
        model.external_transducer_model.blank_decoder.emit_prob_dim,
        model.blank_decoder.emit_prob_dim
      )

      blank_log_prob += ext_trans_scale * ext_trans_blank_log_prob
      emit_log_prob += ext_trans_scale * ext_trans_emit_log_prob

    # ------------------- combination -------------------

    label_log_prob += emit_log_prob
    output_log_prob, _ = rf.concat(
      (label_log_prob, model.target_dim), (blank_log_prob, blank_log_prob.feature_dim),
      out_dim=model.align_target_dim
    )
  else:
    output_log_prob = label_log_prob

  return (
    output_log_prob,
    label_decoder_state,
    blank_decoder_state,
    lm_state,
    ilm_state,
    aed_decoder_state,
    aed_ilm_state,
    ext_trans_label_decoder_state,
    ext_trans_ilm_state,
    lm_eos_log_prob
  )


def model_recog(
        *,
        model: SegmentalAttentionModel,
        data: Tensor,
        data_spatial_dim: Dim,
        beam_size: int,
        use_recombination: Optional[str] = None,
        base_model_scale: float = 1.0,
        external_lm_scale: Optional[float] = None,
        external_lm_eos_scale: Optional[float] = None,
        external_aed_scale: Optional[float] = None,
        external_transducer_scale: Optional[float] = None,
        ilm_type: Optional[str] = None,
        ilm_correction_scale: Optional[float] = None,
        subtract_ilm_eos_score: bool = False,
        cheating_targets: Optional[Tensor] = None,
        cheating_targets_spatial_dim: Optional[Dim] = None,
        att_readout_scale: Optional[float] = None,
        h_t_readout_scale: Optional[float] = None,
        blank_penalty: Optional[float] = None,
        blank_scale: float = 1.0,
        emit_scale: float = 1.0,
        add_lm_eos_to_non_blank_end_hyps: bool = False,
        cheating_target_is_alignment: bool = False,
        length_normalization_exponent: float = 0.0,
) -> Tuple[Tensor, Tensor, Dim, Tensor, Dim, Dim, Tensor]:
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

  max_seq_len = enc_spatial_dim.get_size_tensor()
  max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)

  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims
  backrefs = rf.zeros(batch_dims_, dtype="int32")

  bos_idx = model.bos_idx

  seq_log_prob = rf.constant(0.0, dims=batch_dims_)

  # for each hypothesis, count the number of paths (in log-space) which went into it through recombination
  recomb_path_counter = rf.zeros(batch_dims_, dtype="float32")

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
  num_labels = rf.constant(0, dims=batch_dims_)

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
  if external_lm_eos_scale is None:
    external_lm_eos_scale = 1.0

  # ILM
  if ilm_type is not None:
    if isinstance(model.label_decoder, GlobalAttDecoder):
      ilm_state = model.label_decoder.decoder_default_initial_state(
        batch_dims=batch_dims_,
        use_mini_att=ilm_type == "mini_att",
        use_zero_att=ilm_type == "zero_att",
        enc_spatial_dim=enc_spatial_dim
      )
    else:
      ilm_state = model.label_decoder.default_initial_state(
        batch_dims=batch_dims_,
        use_mini_att=ilm_type == "mini_att",
        use_zero_att=ilm_type == "zero_att",
      )
  else:
    ilm_state = None

  # external aed model
  if model.aed_model:
    aed_enc_args, aed_enc_spatial_dim = model.aed_model.encoder.encode(data, in_spatial_dim=data_spatial_dim)
    aed_enc_args["enc"] = utils.copy_tensor_replace_dim_tag(aed_enc_args["enc"], aed_enc_spatial_dim, enc_spatial_dim)

    if model.aed_model.decoder_state == "trafo":
      aed_enc_args["enc"] = model.aed_model.label_decoder.transform_encoder(aed_enc_args["enc"], axis=enc_spatial_dim)
      aed_decoder_state = _get_init_trafo_state(model.aed_model.label_decoder, batch_dims_)
      input_embed_aed = None
    else:
      aed_enc_args["enc_ctx"] = utils.copy_tensor_replace_dim_tag(
        aed_enc_args["enc_ctx"], aed_enc_spatial_dim, enc_spatial_dim)
      aed_decoder_state = model.aed_model.label_decoder.decoder_default_initial_state(
        batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
      input_embed_aed = rf.zeros(
        batch_dims_ + [model.aed_model.label_decoder.target_embed.out_dim],
        feature_dim=model.aed_model.label_decoder.target_embed.out_dim,
        dtype="float32"
      )

    if ilm_type is not None:
      aed_ilm_state = model.aed_model.label_decoder.decoder_default_initial_state(
        batch_dims=batch_dims_,
        use_mini_att=ilm_type == "mini_att",
        use_zero_att=ilm_type == "zero_att",
        enc_spatial_dim=enc_spatial_dim
      )
    else:
      aed_ilm_state = None
  else:
    aed_decoder_state = None
    aed_ilm_state = None
    input_embed_aed = None
    aed_enc_args = None

  # external transducer model
  if model.external_transducer_model:
    ext_trans_enc_args, ext_trans_enc_spatial_dim = model.external_transducer_model.encoder.encode(data, in_spatial_dim=data_spatial_dim)
    ext_trans_enc_args["enc"] = utils.copy_tensor_replace_dim_tag(ext_trans_enc_args["enc"], ext_trans_enc_spatial_dim, enc_spatial_dim)
    ext_trans_decoder_state = model.external_transducer_model.label_decoder.default_initial_state(
      batch_dims=batch_dims_)
    input_embed_ext_trans = rf.zeros(
      batch_dims_ + [model.external_transducer_model.label_decoder.target_embed.out_dim],
      feature_dim=model.external_transducer_model.label_decoder.target_embed.out_dim,
      dtype="float32"
    )

    if ilm_type is not None:
      ext_trans_ilm_state = model.external_transducer_model.label_decoder.default_initial_state(
        batch_dims=batch_dims_,
        use_mini_att=ilm_type == "mini_att",
        use_zero_att=ilm_type == "zero_att",
      )
    else:
      ext_trans_ilm_state = None
  else:
    ext_trans_decoder_state = None
    ext_trans_ilm_state = None
    input_embed_ext_trans = None
    ext_trans_enc_args = None

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
    vocab_range = rf.range_over_dim(output_dim)

    if not cheating_target_is_alignment:
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

      blank_tensor = rf.convert_to_tensor(model.blank_idx, dtype=vocab_range.dtype)

  # --------------------------------- main loop ---------------------------------

  i = 0
  T = max_seq_len.raw_tensor.item()
  if add_lm_eos_to_non_blank_end_hyps:
    # loop until T + 1, because we need to get the LM EOS prob for the hypotheses which end with non-blank
    # because we don't add the EOS score for these hyps in get_score. so we do it after the loop
    T_ = T + 1
  else:
    T_ = T

  while i < T_:
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
      if model.aed_model and model.aed_model.decoder_state != "trafo":
        input_embed_aed = rf.where(
          update_state_mask,
          model.aed_model.label_decoder.target_embed(target_non_blank),
          rf.gather(input_embed_aed, indices=backrefs)
        )
      if model.external_transducer_model:
        input_embed_ext_trans = rf.where(
          update_state_mask,
          model.external_transducer_model.label_decoder.target_embed(target_non_blank),
          rf.gather(input_embed_ext_trans, indices=backrefs)
        )
      if isinstance(model.blank_decoder, BlankDecoderV1):
        input_embed_length_model = model.blank_decoder.target_embed(target)

      emit_positions = rf.where(
        update_state_mask,
        rf.full(dims=batch_dims, fill_value=i - 1, dtype="int32"),
        rf.gather(emit_positions, indices=backrefs)
      )

      if cheating_targets is not None and not cheating_target_is_alignment:
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
      aed_decoder_state_updated,
      aed_ilm_state_updated,
      ext_trans_decoder_state_updated,
      ext_trans_ilm_state_updated,
      lm_eos_log_prob,
    ) = get_score(
      model=model,
      i=i,
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
      aed_ilm_state=aed_ilm_state,
      enc_args=enc_args,
      att_enc_args=att_enc_args,
      aed_enc_args=aed_enc_args,
      enc_spatial_dim=enc_spatial_dim,
      beam_dim=beam_dim,
      batch_dims=batch_dims,
      base_model_scale=base_model_scale,
      external_lm_scale=external_lm_scale,
      external_lm_eos_scale=external_lm_eos_scale,
      ilm_correction_scale=ilm_correction_scale,
      external_aed_scale=external_aed_scale,
      subtract_ilm_eos_score=subtract_ilm_eos_score,
      att_readout_scale=att_readout_scale,
      h_t_readout_scale=h_t_readout_scale,
      blank_penalty=blank_penalty,
      blank_scale=blank_scale,
      emit_scale=emit_scale,
      ext_trans_scale=external_transducer_scale,
      ext_trans_enc_args=ext_trans_enc_args,
      ext_trans_label_decoder_state=ext_trans_decoder_state,
      ext_trans_ilm_state=ext_trans_ilm_state,
      input_embed_ext_trans=input_embed_ext_trans,
    )

    if i == T:
      break

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

    if cheating_targets is not None and not cheating_target_is_alignment:
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
    elif cheating_target_is_alignment:
      align_position = rf.minimum(
        rf.full(dims=[beam_dim] + batch_dims, fill_value=i, dtype="int32"),
        rf.copy_to_device(cheating_targets_spatial_dim.get_size_tensor() - 1)
      )
      align_ground_truth = rf.gather(
        cheating_targets,
        indices=align_position,
        axis=cheating_targets_spatial_dim,
        clip_to_valid=True
      )

      # mask label log prob in order to only allow hypotheses corresponding to the ground truth:
      output_log_prob_mask = vocab_range == align_ground_truth
      output_log_prob = rf.where(
        output_log_prob_mask,
        output_log_prob,
        rf.constant(-1.0e30, dims=batch_dims + [beam_dim, output_dim])
      )

    # ------------------- recombination -------------------

    if use_recombination:
      seq_log_prob, recomb_path_counter = recombination.recombine_seqs(
        seq_targets,
        seq_log_prob,
        seq_hash,
        beam_dim,
        batch_dims[0],
        use_sum=use_recombination == "sum",
        # recomb_path_counter=recomb_path_counter,
      )

    # ------------------- top-k -------------------

    seq_log_prob = seq_log_prob + output_log_prob  # Batch, InBeam, Vocab
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob,
      k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
      axis=[beam_dim, output_dim]
    )
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    if cheating_targets is not None and not cheating_target_is_alignment:
      prev_label_indices = rf.gather(label_indices, indices=backrefs)

    # ------------------- update hash for recombination -------------------

    if use_recombination:
      seq_hash = recombination.update_seq_hash(seq_hash, target, backrefs, model.blank_idx)

    # mask for updating label-sync states
    update_state_mask = rf.convert_to_tensor(target != model.blank_idx)
    num_labels = rf.gather(num_labels, indices=backrefs)
    num_labels = rf.where(update_state_mask, num_labels + 1, num_labels)

    (
      label_decoder_state,
      blank_decoder_state,
      lm_state,
      ilm_state,
      aed_decoder_state,
      aed_ilm_state,
      ext_trans_decoder_state,
      ext_trans_ilm_state,
    ) = update_state(
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
      aed_decoder_state_updated=aed_decoder_state_updated,
      aed_ilm_state=aed_ilm_state,
      aed_ilm_state_updated=aed_ilm_state_updated,
      ext_trans_label_decoder_state=ext_trans_decoder_state,
      ext_trans_label_decoder_state_updated=ext_trans_decoder_state_updated,
      ext_trans_ilm_state=ext_trans_ilm_state,
      ext_trans_ilm_state_updated=ext_trans_ilm_state_updated,
    )

    # recomb_path_counter = rf.gather(recomb_path_counter, indices=backrefs)

    if i > 1 and length_normalization_exponent != 0:
      # Length-normalized scores, so we evaluate score_t/len.
      # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
      # Because we count with EOS symbol, shifted by one.
      seq_log_prob *= rf.where(
        num_labels > 1,
        # add 1e-30 to avoid division by zero
        (num_labels / (num_labels - 1 + 1e-30)) ** length_normalization_exponent,
        1.0,
      )

    i += 1

  if add_lm_eos_to_non_blank_end_hyps:
    seq_log_prob = rf.where(
      update_state_mask,
      seq_log_prob + lm_eos_log_prob,
      seq_log_prob
    )

  # last recombination
  if use_recombination:
    seq_log_prob, recomb_path_counter = recombination.recombine_seqs(
      seq_targets,
      seq_log_prob,
      seq_hash,
      beam_dim,
      batch_dims[0],
      use_sum=use_recombination == "sum",
      # recomb_path_counter=recomb_path_counter,
    )

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

  non_blank_targets, non_blank_targets_spatial_dim = utils.get_masked(
    seq_targets,
    utils.get_non_blank_mask(seq_targets, model.blank_idx),
    enc_spatial_dim,
    [beam_dim] + batch_dims,
  )
  non_blank_targets.sparse_dim = model.target_dim

  best_hyps = rf.reduce_argmax(seq_log_prob, axis=beam_dim)
  best_alignment = rf.gather(
    seq_targets,
    indices=best_hyps,
    axis=beam_dim,
  )

  # # calculate theoretical maximum number of recombined paths
  # best_recomb_path_count = rf.gather(recomb_path_counter, indices=best_hyps)
  # best_S = rf.gather(non_blank_targets_spatial_dim.dyn_size_ext, indices=best_hyps)
  #
  # assert len(batch_dims) == 1
  # max_recomb_paths = get_max_num_recombinations(
  #   batch_dim=batch_dims[0],
  #   S=best_S,
  #   T=enc_spatial_dim.dyn_size_ext
  # )
  #
  # ratio_recomb_paths = rf.exp(best_recomb_path_count - max_recomb_paths)
  ratio_recomb_paths = rf.constant(0.0, dims=batch_dims)

  return (
    best_alignment,
    seq_log_prob,
    enc_spatial_dim,
    non_blank_targets,
    non_blank_targets_spatial_dim,
    beam_dim,
    ratio_recomb_paths
  )


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
        aed_ilm_state: Optional[State],
        aed_ilm_state_updated: Optional[State],
        ext_trans_label_decoder_state: Optional[State],
        ext_trans_label_decoder_state_updated: Optional[State],
        ext_trans_ilm_state: Optional[State],
        ext_trans_ilm_state_updated: Optional[State],
) -> Tuple[State, Optional[State], Optional[State], Optional[State], Optional[State], Optional[State], Optional[State], Optional[State]]:
  def _get_masked_state(old, new, mask):
    old = rf.gather(old, indices=backrefs)
    new = rf.gather(new, indices=backrefs)
    return rf.where(mask, new, old)

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

  # ------------------ update external transducer state -------------------
  if ext_trans_label_decoder_state is not None:
    ext_trans_label_decoder_state = tree.map_structure(
      lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
      ext_trans_label_decoder_state, ext_trans_label_decoder_state_updated
    )

  if ext_trans_ilm_state is not None:
    ext_trans_ilm_state = tree.map_structure(
      lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
      ext_trans_ilm_state, ext_trans_ilm_state_updated
    )

  # ------------------- update external AED state -------------------

  if aed_decoder_state is not None:
    if model.aed_model.decoder_state != "trafo":
      aed_decoder_state = tree.map_structure(
        lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
        aed_decoder_state, aed_decoder_state_updated
      )
    else:
      aed_decoder_state = _get_masked_trafo_state(
        trafo_state=aed_decoder_state,
        trafo_state_updated=aed_decoder_state_updated,
        update_state_mask=update_state_mask,
        backrefs=backrefs
      )

  if aed_ilm_state is not None:
    aed_ilm_state = tree.map_structure(
      lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
      aed_ilm_state, aed_ilm_state_updated
    )

  # ------------------- update external LM state -------------------

  if lm_state is not None:
    lm_state = _get_masked_trafo_state(
      trafo_state=lm_state,
      trafo_state_updated=lm_state_updated,
      update_state_mask=update_state_mask,
      backrefs=backrefs
    )

  return label_decoder_state, blank_decoder_state, lm_state, ilm_state, aed_decoder_state, aed_ilm_state, ext_trans_label_decoder_state, ext_trans_ilm_state


def model_recog_on_lattice(
        *,
        model: SegmentalAttentionModel,
        data: Tensor,
        data_spatial_dim: Dim,
        beam_size: int,
        cheating_targets: Tensor,
        cheating_targets_spatial_dim: Dim,
        use_recombination: Optional[str] = None,
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
  assert model.label_decoder_state in {"nb-lstm", "joint-lstm", "nb-2linear-ctx1", "trafo"}

  # --------------------------------- init encoder, dims, etc ---------------------------------

  enc_args, enc_spatial_dim = model.encoder.encode(data, in_spatial_dim=data_spatial_dim)

  max_seq_len = enc_spatial_dim.get_size_tensor()
  max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)

  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims
  # backrefs = rf.zeros(batch_dims_, dtype="int32")

  segment_starts, segment_lens, center_positions = utils.get_segment_starts_and_lens(
    rf.sequence_mask(batch_dims + [enc_spatial_dim]),  # this way, every frame is interpreted as non-blank
    enc_spatial_dim,
    model.center_window_size,
    batch_dims,
    enc_spatial_dim
  )

  seq_log_prob = rf.constant(0.0, dims=batch_dims_, device="cpu")

  if use_recombination:
    assert len(batch_dims) == 1
    assert use_recombination in {"sum", "max"}
    seq_hash = rf.constant(0, dims=batch_dims_, dtype="int64", device="cpu")
  else:
    seq_hash = None

  # lists of [B, beam] tensors
  seq_targets = []
  seq_backrefs = []

  output_dim = model.target_dim if model.use_joint_model else model.align_target_dim

  score_lattice, cheating_targets_padded_spatial_dim = get_score_lattice(
    model=model,
    enc_args=enc_args,
    enc_spatial_dim=enc_spatial_dim,
    non_blank_targets=cheating_targets,
    non_blank_targets_spatial_dim=cheating_targets_spatial_dim,
    segment_starts=segment_starts,
    segment_lens=segment_lens,
    center_positions=center_positions,
    batch_dims=batch_dims,
  )
  log_prob_lattice = rf.log_softmax(score_lattice, axis=output_dim)
  log_prob_lattice = rf.copy_to_device(log_prob_lattice, "cpu")

  # --------------------------------- cheating targets ---------------------------------

  # add blank idx on the right
  # this way, when the label index for gathering reached the last non-blank index, it will gather blank after that
  # which then only allows corresponding hypotheses to be extended by blank
  cheating_targets_padded, _ = rf.pad(
    cheating_targets,
    axes=[cheating_targets_spatial_dim],
    padding=[(0, 1)],
    value=model.blank_idx,
    out_dims=[cheating_targets_padded_spatial_dim],
  )
  cheating_targets_padded = rf.copy_to_device(cheating_targets_padded, "cpu")

  # rf.pad falsely pads right after the padding. this means that for shorter seqs, the padding is behind the padding.
  cheating_targets_padded = rf.where(
    rf.range_over_dim(cheating_targets_padded_spatial_dim, device="cpu") < cheating_targets_padded_spatial_dim.dyn_size_ext - 1,
    cheating_targets_padded,
    model.blank_idx
  )

  cheating_targets_padded_spatial_sizes = cheating_targets_padded_spatial_dim.dyn_size_ext
  cheating_targets_spatial_sizes = cheating_targets_spatial_dim.dyn_size_ext
  max_num_labels = rf.reduce_max(
    cheating_targets_spatial_sizes, axis=cheating_targets_spatial_sizes.dims
  ).raw_tensor.item()
  single_col_dim = Dim(dimension=max_num_labels + 1, name="max-num-labels")
  label_indices = rf.zeros(batch_dims_, dtype="int32", sparse_dim=single_col_dim, device="cpu")
  prev_label_indices = label_indices.copy()
  enc_spatial_sizes = enc_spatial_dim.dyn_size_ext

  vocab_range = rf.range_over_dim(output_dim, device="cpu")
  blank_tensor = rf.convert_to_tensor(model.blank_idx, dtype=vocab_range.dtype, device="cpu")

  # --------------------------------- main loop ---------------------------------

  i = 0
  while i < max_seq_len.raw_tensor:
    output_log_prob_t = rf.gather(log_prob_lattice, indices=rf.constant(i, dims=batch_dims, device="cpu"), axis=enc_spatial_dim)
    output_log_prob = rf.gather(output_log_prob_t, indices=label_indices, axis=cheating_targets_padded_spatial_dim)

    # for shorter seqs in the batch, set the blank score to zero and the others to ~-inf
    output_log_prob = rf.where(
      rf.convert_to_tensor(i >= enc_spatial_sizes, device="cpu"),
      rf.copy_to_device(rf.sparse_to_dense(
        model.blank_idx,
        axis=output_dim,
        label_value=0.0,
        other_value=-1.0e30
      ), device="cpu"),
      output_log_prob
    )

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
      rf.constant(-1.0e30, dims=batch_dims + [beam_dim, output_dim], device="cpu")
    )

    # ------------------- recombination -------------------

    if use_recombination:
      seq_log_prob = recombination.recombine_seqs(
        seq_targets,
        seq_log_prob,
        seq_hash,
        beam_dim,
        batch_dims[0],
        use_sum=use_recombination == "sum"
      )

    beam_size = min(
      min((i + 1) * 2, rf.reduce_max(rem_frames, axis=rem_frames.dims).raw_tensor.item() * 2), max_num_labels * 2)

    # ------------------- top-k -------------------

    seq_log_prob = seq_log_prob + output_log_prob  # Batch, InBeam, Vocab
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob,
      k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
      axis=[beam_dim, output_dim]
    )
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    prev_label_indices = rf.gather(label_indices, indices=backrefs)

    # ------------------- update hash for recombination -------------------

    if use_recombination:
      seq_hash = recombination.update_seq_hash(seq_hash, target, backrefs, model.blank_idx)

    # mask for updating label-sync states
    update_state_mask = rf.convert_to_tensor(target != model.blank_idx)

    label_indices = rf.where(
      update_state_mask,
      rf.where(
        prev_label_indices == cheating_targets_padded_spatial_sizes - 1,
        prev_label_indices,
        prev_label_indices + 1
      ),
      prev_label_indices
    )

    i += 1

  # last recombination
  if use_recombination:
    seq_log_prob = recombination.recombine_seqs(
      seq_targets,
      seq_log_prob,
      seq_hash,
      beam_dim,
      batch_dims[0],
      use_sum=use_recombination == "sum"
    )

  # Backtrack via backrefs, resolve beams.
  seq_targets_ = []
  indices = rf.range_over_dim(beam_dim, device="cpu")  # FinalBeam -> FinalBeam
  for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
    # indices: FinalBeam -> Beam
    # backrefs: Beam -> PrevBeam
    seq_targets_.insert(0, rf.gather(target, indices=indices))
    indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

  seq_targets__ = TensorArray(seq_targets_[0])
  for target in seq_targets_:
    seq_targets__ = seq_targets__.push_back(target)
  seq_targets = seq_targets__.stack(axis=enc_spatial_dim)

  best_hyps = rf.reduce_argmax(seq_log_prob, axis=beam_dim)
  best_alignment = rf.gather(
    seq_targets,
    indices=best_hyps,
    axis=beam_dim,
  )

  return best_alignment, seq_log_prob, enc_spatial_dim, cheating_targets, cheating_targets_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[SegmentalAttentionModel]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False


def model_recog_pure_torch(
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
      recog results info: key -> {batch, beam},
      out_spatial_dim,
      final beam_dim
  """
  import torch
  from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.beam_search.time_sync import BeamSearchOpts, time_sync_beam_search
  from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.label_scorer import ShallowFusedLabelScorers
  from returnn.config import get_global_config

  config = get_global_config()

  torch.cuda.set_sync_debug_mode(1)  # debug CUDA sync. does not hurt too much to leave this always in?

  data_concat_zeros = config.float("data_concat_zeros", 0)
  if data_concat_zeros:
    data_concat_zeros_dim = Dim(int(data_concat_zeros * _batch_size_factor * 100), name="data_concat_zeros")
    data, data_spatial_dim = rf.concat(
      (data, data_spatial_dim), (rf.zeros([data_concat_zeros_dim]), data_concat_zeros_dim), allow_broadcast=True
    )

  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  assert len(batch_dims) == 1, batch_dims  # not implemented otherwise, simple to add...
  batch_dim = batch_dims[0]
  enc, enc_spatial_dim = model.encoder.encode(data, in_spatial_dim=data_spatial_dim)
  if max_seq_len is None:
    max_seq_len = enc_spatial_dim.get_size_tensor()
  else:
    max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")

  beam_search_opts = (config.typed_value("beam_search_opts", None) or {}).copy()
  if beam_search_opts.get("beam_size") is None:
    beam_search_opts["beam_size"] = config.int("beam_size", 12)
  if beam_search_opts.get("length_normalization_exponent") is None:
    beam_search_opts["length_normalization_exponent"] = config.float("length_normalization_exponent", 1.0)

  label_scorer = ShallowFusedLabelScorers()
  label_scorer.label_scorers["label_sync_decoder"] = (
    get_label_sync_scorer_pure_torch(model=model, batch_dim=batch_dim, enc=enc, enc_spatial_dim=enc_spatial_dim),
    1.0,
  )
  label_scorer.label_scorers["time_sync_decoder"] = (
    get_time_sync_scorer_pure_torch(model=model, batch_dim=batch_dim, enc=enc, enc_spatial_dim=enc_spatial_dim),
    1.0,
  )
  if model.label_decoder.language_model:
    lm_scale = beam_search_opts.pop("lm_scale")  # must be defined with LM
    label_scorer.label_scorers["lm"] = (model.label_decoder.language_model_make_label_scorer(), lm_scale)

  print("** max seq len:", max_seq_len.raw_tensor)

  # Beam search happening here:
  (
    seq_targets,  # [Batch,FinalBeam,OutSeqLen]
    seq_log_prob,  # [Batch,FinalBeam]
  ) = time_sync_beam_search(
    label_scorer,
    label_sync_keys=["label_sync_decoder", "lm"] if model.label_decoder.language_model else ["label_sync_decoder"],
    time_sync_keys=["time_sync_decoder"],
    batch_size=int(batch_dim.get_dim_value()),
    blank_idx=model.blank_idx,
    max_seq_len=max_seq_len.copy_compatible_to_dims_raw([batch_dim]),
    device=data.raw_tensor.device,
    opts=BeamSearchOpts(
      **beam_search_opts,
      bos_label=0,
      eos_label=0,
      num_labels=model.target_dim.dimension,
    ),
  )

  beam_dim = Dim(seq_log_prob.shape[1], name="beam")
  seq_targets_t = rf.convert_to_tensor(
    seq_targets, dims=[batch_dim, beam_dim, enc_spatial_dim], sparse_dim=model.target_dim
  )
  seq_log_prob_t = rf.convert_to_tensor(seq_log_prob, dims=[batch_dim, beam_dim])

  non_blank_targets, non_blank_targets_spatial_dim = utils.get_masked(
    seq_targets_t,
    utils.get_non_blank_mask(seq_targets_t, model.blank_idx),
    enc_spatial_dim,
    [beam_dim] + batch_dims,
  )
  non_blank_targets.sparse_dim = model.target_dim

  return non_blank_targets, seq_log_prob_t, non_blank_targets_spatial_dim, beam_dim


# RecogDef API
model_recog_pure_torch: RecogDef[SegmentalAttentionModel]
model_recog_pure_torch.output_with_beam = True
model_recog_pure_torch.output_blank_label = None
model_recog_pure_torch.batch_size_dependent = False


def get_label_sync_scorer_pure_torch(
        *,
        model: SegmentalAttentionModel,
        batch_dim: Dim,
        enc: Dict[str, Tensor],
        enc_spatial_dim: Dim,
):
  import torch
  import functools
  from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.label_scorer import (
    LabelScorerIntf,
    StateObjTensorExt,
    StateObjIgnored,
  )

  class LabelScorer(LabelScorerIntf):
    """label scorer"""

    def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
      """Initial state."""
      beam_dim = Dim(1, name="initial-beam")
      batch_dims_ = [batch_dim, beam_dim]
      decoder_state = model.label_decoder.default_initial_state(batch_dims=batch_dims_, )
      return tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state)

    def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
            prev_align_label: Optional[torch.Tensor] = None,
            t: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any]:
      """update state"""
      beam_dim = Dim(prev_label.shape[1], name="beam")
      assert t is not None

      def _map_raw_to_tensor(v):
        if isinstance(v, StateObjTensorExt):
          tensor: Tensor = v.extra
          tensor = tensor.copy_template_new_dim_tags(
            (batch_dim, beam_dim) + tensor.dims[2:], keep_special_axes=True
          )
          tensor.raw_tensor = v.tensor
          return tensor
        elif isinstance(v, StateObjIgnored):
          return v.content
        else:
          raise TypeError(f"_map_raw_to_tensor: unexpected {v} ({type(v).__name__})")

      center_position = rf.minimum(
        rf.full(dims=[beam_dim, batch_dim], fill_value=t, dtype="int32"),
        rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, enc["enc"].device)
      )
      segment_starts = rf.maximum(
        rf.convert_to_tensor(0, dtype="int32"), center_position - model.center_window_size // 2)
      segment_ends = rf.minimum(
        rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, enc["enc"].device),
        center_position + model.center_window_size // 2
      )
      segment_lens = segment_ends - segment_starts + 1

      zeros_embed = rf.zeros(
        [batch_dim, beam_dim, model.label_decoder.target_embed.out_dim],
        feature_dim=model.label_decoder.target_embed.out_dim,
        dtype="float32"
      )
      initial_output_mask = rf.convert_to_tensor(prev_label == -1, dims=[batch_dim, beam_dim])
      prev_label = rf.convert_to_tensor(prev_label, dims=[batch_dim, beam_dim], sparse_dim=model.target_dim)
      prev_label = rf.where(
        initial_output_mask,
        rf.zeros_like(prev_label),
        prev_label
      )
      input_embed = rf.where(
        initial_output_mask,
        zeros_embed,
        model.label_decoder.target_embed(prev_label)
      )

      decode_out, decoder_state = model.label_decoder.loop_step(
        **enc,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed,
        segment_lens=segment_lens,
        segment_starts=segment_starts,
        state=tree.map_structure(_map_raw_to_tensor, prev_state),
      )
      logits = model.label_decoder.decode_logits(input_embed=input_embed, **decode_out)
      label_log_prob = rf.log_softmax(logits, axis=model.target_dim)

      blank_log_prob = rf.zeros(
        [Dim(1, name="blank_log_prob_label_scorer")],
        dtype="float32"
      )
      output_log_prob, _ = rf.concat(
        (label_log_prob, model.target_dim), (blank_log_prob, blank_log_prob.dims[0]),
        out_dim=model.align_target_dim,
        allow_broadcast=True
      )
      assert set(output_log_prob.dims) == {batch_dim, beam_dim, model.align_target_dim}

      return (
        self._map_tensor_to_raw(output_log_prob, beam_dim=beam_dim).tensor,
        tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state),
      )

    @staticmethod
    def _map_tensor_to_raw(v, *, beam_dim: Dim):
      if isinstance(v, Tensor):
        if beam_dim not in v.dims:
          return StateObjIgnored(v)
        batch_dims_ = [batch_dim, beam_dim]
        v = v.copy_transpose(batch_dims_ + [dim for dim in v.dims if dim not in batch_dims_])
        raw = v.raw_tensor
        return StateObjTensorExt(raw, v.copy_template())
      elif isinstance(v, Dim):
        return StateObjIgnored(v)
      else:
        raise TypeError(f"_map_tensor_to_raw: unexpected {v} ({type(v).__name__})")

  return LabelScorer()


def get_time_sync_scorer_pure_torch(
        *,
        model: SegmentalAttentionModel,
        batch_dim: Dim,
        enc: Dict[str, Tensor],
        enc_spatial_dim: Dim,
):
  import torch
  import functools
  from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.label_scorer import (
    LabelScorerIntf,
    StateObjTensorExt,
    StateObjIgnored,
  )

  class LabelScorer(LabelScorerIntf):
    """label scorer"""

    def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
      """Initial state."""
      beam_dim = Dim(1, name="initial-beam")
      batch_dims_ = [batch_dim, beam_dim]
      decoder_state = model.blank_decoder.default_initial_state(batch_dims=batch_dims_, )
      return tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state)

    def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
            prev_align_label: Optional[torch.Tensor] = None,
            t: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any]:
      """update state"""
      beam_dim = Dim(prev_label.shape[1], name="beam")
      assert prev_align_label is not None

      def _map_raw_to_tensor(v):
        if isinstance(v, StateObjTensorExt):
          tensor: Tensor = v.extra
          tensor = tensor.copy_template_new_dim_tags(
            (batch_dim, beam_dim) + tensor.dims[2:], keep_special_axes=True
          )
          tensor.raw_tensor = v.tensor
          return tensor
        elif isinstance(v, StateObjIgnored):
          return v.content
        else:
          raise TypeError(f"_map_raw_to_tensor: unexpected {v} ({type(v).__name__})")

      zeros_embed = rf.zeros(
        [batch_dim, beam_dim, model.blank_decoder.target_embed.out_dim],
        feature_dim=model.blank_decoder.target_embed.out_dim,
        dtype="float32"
      )
      initial_output_mask = rf.convert_to_tensor(prev_align_label == -1, dims=[batch_dim, beam_dim])
      prev_align_label = rf.convert_to_tensor(prev_align_label, dims=[batch_dim, beam_dim], sparse_dim=model.align_target_dim)
      prev_align_label = rf.where(
        initial_output_mask,
        rf.zeros_like(prev_align_label),
        prev_align_label
      )
      input_embed = rf.where(
        initial_output_mask,
        zeros_embed,
        model.blank_decoder.target_embed(prev_align_label)
      )

      decode_out, decoder_state = model.blank_decoder.loop_step(
        enc=enc["enc"],
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed,
        state=tree.map_structure(_map_raw_to_tensor, prev_state),
      )
      blank_logits = model.blank_decoder.decode_logits(**decode_out)
      emit_log_prob = rf.log(rf.sigmoid(blank_logits))
      emit_log_prob = rf.squeeze(emit_log_prob, axis=emit_log_prob.feature_dim)
      blank_log_prob = rf.log(rf.sigmoid(-blank_logits))

      label_log_prob = rf.zeros(
        dims=[batch_dim, beam_dim, model.target_dim],
        dtype="float32"
      )
      label_log_prob += emit_log_prob
      output_log_prob, _ = rf.concat(
        (label_log_prob, model.target_dim), (blank_log_prob, blank_log_prob.feature_dim),
        out_dim=model.align_target_dim
      )
      assert set(output_log_prob.dims) == {batch_dim, beam_dim, model.align_target_dim}

      return (
        self._map_tensor_to_raw(output_log_prob, beam_dim=beam_dim).tensor,
        tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state),
      )

    @staticmethod
    def _map_tensor_to_raw(v, *, beam_dim: Dim):
      if isinstance(v, Tensor):
        if beam_dim not in v.dims:
          return StateObjIgnored(v)
        batch_dims_ = [batch_dim, beam_dim]
        v = v.copy_transpose(batch_dims_ + [dim for dim in v.dims if dim not in batch_dims_])
        raw = v.raw_tensor
        return StateObjTensorExt(raw, v.copy_template())
      elif isinstance(v, Dim):
        return StateObjIgnored(v)
      else:
        raise TypeError(f"_map_tensor_to_raw: unexpected {v} ({type(v).__name__})")

  return LabelScorer()
