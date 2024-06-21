from typing import Optional, Dict, Any, Tuple
import tree

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.recog import RecogDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_old.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.utils import get_masked, get_non_blank_mask


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
      rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, data.device)
    )
    segment_starts = rf.maximum(
      rf.convert_to_tensor(0, dtype="int32"), center_position - model.center_window_size // 2)
    segment_ends = rf.minimum(
      rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, data.device),
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

    label_decoder_state = tree.map_structure(
      lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
      label_decoder_state, label_decoder_state_updated
    )

    target_non_blank = rf.where(update_state_mask, target, rf.gather(target_non_blank, indices=backrefs))
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
# output_blank_label=blank is actually wrong for AED, but now we don't change it anymore
# because it would change all recog hashes.
# Also, it does not matter too much -- it will just cause an extra SearchRemoveLabelJob,
# which will not have any effect here.
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
  enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
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
  if model.language_model:
    lm_scale = beam_search_opts.pop("lm_scale")  # must be defined with LM
    label_scorer.label_scorers["lm"] = (model.language_model_make_label_scorer(), lm_scale)

  print("** max seq len:", max_seq_len.raw_tensor)

  # Beam search happening here:
  (
    seq_targets,  # [Batch,FinalBeam,OutSeqLen]
    seq_log_prob,  # [Batch,FinalBeam]
  ) = time_sync_beam_search(
    label_scorer,
    label_sync_keys=["label_sync_decoder", "lm"] if model.language_model else ["label_sync_decoder"],
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
      decoder_state = model.label_decoder_default_initial_state(batch_dims=batch_dims_,)
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
        [batch_dim, beam_dim, model.target_embed.out_dim],
        feature_dim=model.target_embed.out_dim,
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
        model.target_embed(prev_label)
      )

      decode_out, decoder_state = model.label_sync_loop_step(
        **enc,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed,
        segment_lens=segment_lens,
        segment_starts=segment_starts,
        state=tree.map_structure(_map_raw_to_tensor, prev_state),
      )
      logits = model.decode_label_logits(input_embed=input_embed, **decode_out)
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
      decoder_state = model.blank_decoder_default_initial_state(batch_dims=batch_dims_,)
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
        [batch_dim, beam_dim, model.target_embed_length_model.out_dim],
        feature_dim=model.target_embed_length_model.out_dim,
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
        model.target_embed_length_model(prev_align_label)
      )

      decode_out, decoder_state = model.time_sync_loop_step(
        enc=enc["enc"],
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed,
        state=tree.map_structure(_map_raw_to_tensor, prev_state),
      )
      blank_logits = model.decode_blank_logits(**decode_out)
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
