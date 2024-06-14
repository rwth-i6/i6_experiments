from typing import Optional, Dict, Any, Tuple, Sequence
import tree

from returnn.tensor import Tensor, Dim, single_step_dim
from returnn.frontend.state import State
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.recog import RecogDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import recombination
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.utils import get_masked, get_non_blank_mask
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.beam_search import utils as beam_search_utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import (
  BlankDecoderV1,
  BlankDecoderV3,
  BlankDecoderV5,
  BlankDecoderV6,
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
) -> Tuple[State, Optional[State], Optional[State]]:

  # ------------------- update blank decoder state -------------------

  if blank_decoder_state is not None:
    blank_decoder_state = tree.map_structure(
      lambda s: rf.gather(s, indices=backrefs), blank_decoder_state_updated)

  # ------------------- update label decoder state -------------------

  if model.label_decoder_state == "joint-lstm":
    label_decoder_state = tree.map_structure(
      lambda s: rf.gather(s, indices=backrefs), label_decoder_state_updated)
  else:
    def _get_masked_state(old, new, mask):
      old = rf.gather(old, indices=backrefs)
      new = rf.gather(new, indices=backrefs)
      return rf.where(mask, new, old)

    label_decoder_state = tree.map_structure(
      lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
      label_decoder_state, label_decoder_state_updated
    )

  # ------------------- update external LM state -------------------

  if lm_state is not None:
    for state in lm_state:
      if state == "pos":
        lm_state[state] = rf.where(
          update_state_mask,
          rf.gather(lm_state_updated[state], indices=backrefs),
          rf.gather(lm_state[state], indices=backrefs)
        )
      else:
        updated_accum_axis = lm_state_updated[state].self_att.accum_axis

        updated_self_att_expand_dim_dyn_size_ext = rf.gather(updated_accum_axis.dyn_size_ext, indices=backrefs)
        masked_self_att_expand_dim_dyn_size_ext = rf.where(
          update_state_mask,
          updated_self_att_expand_dim_dyn_size_ext,
          updated_self_att_expand_dim_dyn_size_ext - 1
        )
        masked_self_att_expand_dim = Dim(masked_self_att_expand_dim_dyn_size_ext, name="self_att_expand_dim_init")
        lm_state[state].self_att.accum_axis = masked_self_att_expand_dim

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

        lm_state[state].self_att.k_accum = _mask_lm_state(lm_state_updated[state].self_att.k_accum)
        lm_state[state].self_att.v_accum = _mask_lm_state(lm_state_updated[state].self_att.v_accum)

  return label_decoder_state, blank_decoder_state, lm_state


def get_score(
        model: SegmentalAttentionModel,
        i: int,
        input_embed_label_model: Tensor,
        input_embed_blank_model: Optional[Tensor],
        nb_target: Tensor,
        label_decoder_state: State,
        blank_decoder_state: Optional[State],
        lm_state: Optional[State],
        enc_args: Dict[str, Tensor],
        enc_spatial_dim: Dim,
        beam_dim: Dim,
        batch_dims: Sequence[Dim],
        external_lm_scale: Optional[float] = None,
) -> Tuple[Tensor, State, Optional[State], Optional[State]]:
  # ------------------- label step -------------------

  center_position = rf.minimum(
    rf.full(dims=[beam_dim] + batch_dims, fill_value=i, dtype="int32"),
    rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, input_embed_label_model.device)
  )
  segment_starts = rf.maximum(
    rf.convert_to_tensor(0, dtype="int32"), center_position - model.center_window_size // 2)
  segment_ends = rf.minimum(
    rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, input_embed_label_model.device),
    center_position + model.center_window_size // 2
  )
  segment_lens = segment_ends - segment_starts + 1

  label_step_out, label_decoder_state = model.label_decoder.loop_step(
    **enc_args,
    enc_spatial_dim=enc_spatial_dim,
    input_embed=input_embed_label_model,
    segment_lens=segment_lens,
    segment_starts=segment_starts,
    state=label_decoder_state,
  )
  label_logits = model.label_decoder.decode_logits(input_embed=input_embed_label_model, **label_step_out)
  label_log_prob = rf.log_softmax(label_logits, axis=model.target_dim)

  # ------------------- external LM step -------------------

  lm_eos_log_prob = rf.zeros(batch_dims, dtype="float32")
  if lm_state is not None:
    lm_logits, lm_state = model.language_model(
      nb_target,
      spatial_dim=single_step_dim,
      state=lm_state,
    )
    lm_label_log_prob = rf.log_softmax(lm_logits, axis=model.target_dim)
    label_log_prob += external_lm_scale * lm_label_log_prob

    lm_eos_log_prob = rf.where(
      rf.convert_to_tensor(i == rf.copy_to_device(enc_spatial_dim.get_size_tensor(), input_embed_label_model.device) - 1),
      # TODO: change to non hard-coded BOS index
      rf.gather(lm_label_log_prob, indices=rf.constant(0, dtype="int32", dims=batch_dims, sparse_dim=nb_target.sparse_dim)),
      lm_eos_log_prob
    )

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
        blank_loop_step_kwargs["label_model_state"] = label_step_out["s"]

      blank_step_out, blank_decoder_state = model.blank_decoder.loop_step(**blank_loop_step_kwargs)
      blank_logits = model.blank_decoder.decode_logits(**blank_step_out)
    else:
      assert isinstance(model.blank_decoder, BlankDecoderV5) or isinstance(model.blank_decoder, BlankDecoderV6)
      enc_position = rf.minimum(
        rf.full(dims=batch_dims, fill_value=i, dtype="int32"),
        rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, input_embed_label_model.device)
      )
      enc_frame = rf.gather(enc_args["enc"], indices=enc_position, axis=enc_spatial_dim)
      enc_frame = rf.expand_dim(enc_frame, beam_dim)
      if isinstance(model.blank_decoder, BlankDecoderV5):
        # no LSTM -> no state -> just leave (empty) state as is
        blank_logits = model.blank_decoder.emit_prob(
          rf.concat_features(enc_frame, label_step_out["s"]))
      else:
        prev_lstm_state = blank_decoder_state.s_blank
        blank_decoder_state = rf.State()
        s_blank, blank_decoder_state.s_blank = model.blank_decoder.s(
          enc_frame,
          state=prev_lstm_state,
          spatial_dim=single_step_dim
        )
        blank_logits = model.blank_decoder.emit_prob(rf.concat_features(s_blank, label_step_out["s"]))

    emit_log_prob = rf.log(rf.sigmoid(blank_logits))
    emit_log_prob = rf.squeeze(emit_log_prob, axis=emit_log_prob.feature_dim)
    blank_log_prob = rf.log(rf.sigmoid(-blank_logits))
    blank_log_prob += lm_eos_log_prob

    # ------------------- combination -------------------

    label_log_prob += emit_log_prob
    output_log_prob, _ = rf.concat(
      (label_log_prob, model.target_dim), (blank_log_prob, blank_log_prob.feature_dim),
      out_dim=model.align_target_dim
    )
  else:
    output_log_prob = label_log_prob

  # for shorter seqs in the batch, set the blank score to zero and the others to ~-inf
  output_log_prob = rf.where(
    rf.convert_to_tensor(i >= rf.copy_to_device(enc_spatial_dim.get_size_tensor(), input_embed_label_model.device)),
    rf.sparse_to_dense(
      model.blank_idx,
      axis=model.target_dim if model.use_joint_model else model.align_target_dim,
      label_value=0.0,
      other_value=-1.0e30
    ),
    output_log_prob
  )

  return output_log_prob, label_decoder_state, blank_decoder_state, lm_state


def model_recog(
        *,
        model: SegmentalAttentionModel,
        data: Tensor,
        data_spatial_dim: Dim,
        beam_size: int,
        use_recombination: Optional[str] = None,
        external_lm_scale: Optional[float] = None,
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
  assert any(
    isinstance(model.blank_decoder, cls) for cls in (BlankDecoderV1, BlankDecoderV3, BlankDecoderV5, BlankDecoderV6)
  ) or model.blank_decoder is None, "blank_decoder not supported"
  if model.blank_decoder is None:
    assert model.use_joint_model, "blank_decoder is None, so use_joint_model must be True"
  if model.language_model:
    assert external_lm_scale is not None, "external_lm_scale must be defined with LM"
  assert model.label_decoder_state in {"nb-lstm", "joint-lstm", "nb-linear1"}

  # --------------------------------- init encoder, dims, etc ---------------------------------

  enc_args, enc_spatial_dim = model.encoder.encode(data, in_spatial_dim=data_spatial_dim)

  max_seq_len = enc_spatial_dim.get_size_tensor()
  max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)

  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims
  backrefs = rf.zeros(batch_dims_, dtype="int32")

  bos_idx = 0

  seq_log_prob = rf.constant(0.0, dims=batch_dims_)

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

  # --------------------------------- init states ---------------------------------

  # label decoder
  label_decoder_state = model.label_decoder.default_initial_state(batch_dims=batch_dims_, )

  # blank decoder
  if model.blank_decoder is not None:
    blank_decoder_state = model.blank_decoder.default_initial_state(batch_dims=batch_dims_)
  else:
    blank_decoder_state = None

  # external LM
  if model.language_model:
    lm_state = model.language_model.default_initial_state(batch_dims=batch_dims_)
    for state in lm_state:
      if state == "pos":
        lm_state[state] = rf.zeros(batch_dims_, dtype="int32")
      else:
        self_att_expand_dim = Dim(rf.zeros(batch_dims_, dtype="int32"), name="self_att_expand_dim_init")
        lm_state[state].self_att.accum_axis = self_att_expand_dim

        k_accum = lm_state[state].self_att.k_accum  # type: rf.Tensor
        k_accum_raw = k_accum.raw_tensor
        lm_state[state].self_att.k_accum = k_accum.copy_template_replace_dim_tag(
          k_accum.get_axis_from_description("stag:self_att_expand_dim_init"), self_att_expand_dim
        )
        lm_state[state].self_att.k_accum.raw_tensor = k_accum_raw

        v_accum = lm_state[state].self_att.v_accum  # type: rf.Tensor
        v_accum_raw = v_accum.raw_tensor
        lm_state[state].self_att.v_accum = v_accum.copy_template_replace_dim_tag(
          v_accum.get_axis_from_description("stag:self_att_expand_dim_init"), self_att_expand_dim
        )
        lm_state[state].self_att.v_accum.raw_tensor = v_accum_raw
  else:
    lm_state = None

  # --------------------------------- init targets, embeddings ---------------------------------

  if model.use_joint_model:
    target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    if model.label_decoder_state in ("nb-lstm", "nb-linear1"):
      target_non_blank = target.copy()
  else:
    target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.align_target_dim)
    target_non_blank = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)

  input_embed = rf.zeros(
    batch_dims_ + [model.label_decoder.target_embed.out_dim],
    feature_dim=model.label_decoder.target_embed.out_dim,
    dtype="float32"
  )

  if isinstance(model.blank_decoder, BlankDecoderV1):
    input_embed_length_model = rf.zeros(
      batch_dims_ + [model.blank_decoder.target_embed.out_dim], feature_dim=model.blank_decoder.target_embed.out_dim)
  else:
    input_embed_length_model = None

  # --------------------------------- main loop ---------------------------------

  i = 0
  while i < max_seq_len.raw_tensor:
    if i > 0:
      if model.label_decoder_state == "joint-lstm":
        input_embed = model.label_decoder.target_embed(target)
      else:
        target_non_blank = rf.where(update_state_mask, target, rf.gather(target_non_blank, indices=backrefs))
        target_non_blank.sparse_dim = model.label_decoder.target_embed.in_dim
        input_embed = rf.where(
          update_state_mask,
          model.label_decoder.target_embed(target_non_blank),
          rf.gather(input_embed, indices=backrefs)
        )
      if isinstance(model.blank_decoder, BlankDecoderV1):
        input_embed_length_model = model.blank_decoder.target_embed(target)

    output_log_prob, label_decoder_state_updated, blank_decoder_state_updated, lm_state_updated = get_score(
      model=model,
      i=i,
      input_embed_label_model=input_embed,
      input_embed_blank_model=input_embed_length_model,
      nb_target=target_non_blank,
      label_decoder_state=label_decoder_state,
      blank_decoder_state=blank_decoder_state,
      lm_state=lm_state,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      beam_dim=beam_dim,
      batch_dims=batch_dims,
      external_lm_scale=external_lm_scale,
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

    # ------------------- top-k -------------------

    seq_log_prob = seq_log_prob + output_log_prob  # Batch, InBeam, Vocab
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob,
      k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
      axis=[beam_dim, model.target_dim if model.use_joint_model else model.align_target_dim]
    )
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    # ------------------- update hash for recombination -------------------

    if use_recombination:
      seq_hash = recombination.update_seq_hash(seq_hash, target, backrefs, model.blank_idx)

    # mask for updating label-sync states
    update_state_mask = rf.convert_to_tensor(target != model.blank_idx)

    label_decoder_state, blank_decoder_state, lm_state = update_state(
      model=model,
      update_state_mask=update_state_mask,
      backrefs=backrefs,
      label_decoder_state=label_decoder_state,
      label_decoder_state_updated=label_decoder_state_updated,
      blank_decoder_state=blank_decoder_state,
      blank_decoder_state_updated=blank_decoder_state_updated,
      lm_state=lm_state,
      lm_state_updated=lm_state_updated,
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

  non_blank_targets, non_blank_targets_spatial_dim = get_masked(
    seq_targets,
    get_non_blank_mask(seq_targets, model.blank_idx),
    enc_spatial_dim,
    [beam_dim] + batch_dims,
  )
  non_blank_targets.sparse_dim = model.target_dim

  return non_blank_targets, seq_log_prob, non_blank_targets_spatial_dim, beam_dim


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

  non_blank_targets, non_blank_targets_spatial_dim = get_masked(
    seq_targets_t,
    get_non_blank_mask(seq_targets_t, model.blank_idx),
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
