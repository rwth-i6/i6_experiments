from typing import Optional, Dict, Any, Tuple
import tree

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.recog import RecogDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.utils import get_masked, get_non_blank_mask
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.beam_search import utils as beam_search_utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import (
  BlankDecoderV1,
  BlankDecoderV3,
)


def recombine_seqs(
        seq_targets: list,
        seq_log_prob: Tensor,
        seq_backrefs: list,
        seq_hash: Tensor,
        beam_dim: Dim,
        batch_dim: Dim,
        i: int
) -> Tensor:
  if len(seq_targets) in (0, 1):
    return seq_log_prob

  print("seq_hash: ", seq_hash.raw_tensor)
  print("seq_log_prob before: ", seq_log_prob.raw_tensor)

  seq_hash_cpu = rf.copy_to_device(seq_hash.copy_transpose([batch_dim, beam_dim]), device="cpu")
  seq_log_prob = rf.copy_to_device(seq_log_prob.copy_transpose([batch_dim, beam_dim]), device="cpu")

  for b in range(batch_dim.get_dim_value()):
    seq_sets = {}
    for h in range(beam_dim.dimension):
      seq_hash_value = seq_hash_cpu.raw_tensor[b, h]
      if seq_hash_value not in seq_sets:
        seq_sets[seq_hash_value] = []
      seq_sets[seq_hash_value].append(h)

    for seq_set in seq_sets.values():
      if len(seq_set) == 1:
        continue
      best_score = 0
      best_idx = -1
      for idx in seq_set:
        if seq_log_prob.raw_tensor[b, idx] > best_score:
          best_score = seq_log_prob.raw_tensor[b, idx]
          best_idx = idx
      for idx in seq_set:
        if idx != best_idx:
          seq_log_prob.raw_tensor[b, idx] = -float("inf")
        else:
          seq_log_prob.raw_tensor[b, idx] = best_score

  print("seq_log_prob after: ", seq_log_prob.raw_tensor)
  exit()

  return rf.copy_to_device(seq_log_prob, device="gpu")


def update_seq_hash(seq_hash: Tensor, target: Tensor, backrefs: Tensor) -> Tensor:
  print("update_seq_hash")
  print("old seq_hash", seq_hash.raw_tensor)
  print("target", target.raw_tensor)
  print("backrefs", backrefs.raw_tensor)
  print("\n\n")

  old_seq_hash = rf.gather(seq_hash, indices=backrefs)
  seq_hash = rf.where(
    target == 10025,
    old_seq_hash,
    (old_seq_hash * 257 + (target + 1)) % (10 ** 9 + 7)
  )
  return seq_hash


def model_recog(
        *,
        model: SegmentalAttentionModel,
        data: Tensor,
        data_spatial_dim: Dim,
        use_recombination: bool = False,
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
  # assert not model.language_model  # not implemented here. use the pure PyTorch search instead
  assert any(
    isinstance(model.blank_decoder, cls) for cls in (BlankDecoderV1, BlankDecoderV3)
  ) or model.blank_decoder is None, "blank_decoder not supported"
  if model.blank_decoder is None:
    assert model.use_joint_model, "blank_decoder is None, so use_joint_model must be True"

  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  enc_args, enc_spatial_dim = model.encoder.encode(data, in_spatial_dim=data_spatial_dim)
  beam_size = 12
  max_seq_len = enc_spatial_dim.get_size_tensor()
  print("** max seq len:", max_seq_len.raw_tensor)
  max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)

  # Initial state.
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims

  label_decoder_state = model.label_decoder.default_initial_state(batch_dims=batch_dims_, )
  if model.blank_decoder is not None:
    blank_decoder_state = model.blank_decoder.default_initial_state(batch_dims=batch_dims_)
  if model.language_model:
    lm_state = model.language_model.default_initial_state(batch_dims=batch_dims_)

  bos_idx = 0

  if model.use_joint_model:
    target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
  else:
    target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.align_target_dim)
    update_state_mask = rf.convert_to_tensor(target != model.blank_idx)
    target_non_blank = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)

  seq_log_prob = rf.constant(0.0, dims=batch_dims_)
  if use_recombination:
    assert len(batch_dims) == 1
    seq_hash = rf.constant(0, dims=batch_dims_, dtype="int64")

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

  old_beam_dim = beam_dim.copy()
  backrefs = rf.zeros(batch_dims_, dtype="int32")

  i = 0
  seq_targets = []
  seq_backrefs = []
  while i < max_seq_len.raw_tensor:
    if i > 0:
      if model.use_joint_model:
        input_embed = model.label_decoder.target_embed(target)
      else:
        target_non_blank = rf.where(update_state_mask, target, rf.gather(target_non_blank, indices=backrefs))
        target_non_blank.sparse_dim = model.label_decoder.target_embed.in_dim
        input_embed = rf.where(
          update_state_mask,
          model.label_decoder.target_embed(target_non_blank),
          rf.gather(input_embed, indices=backrefs, axis=old_beam_dim)
        )
      if isinstance(model.blank_decoder, BlankDecoderV1):
        input_embed_length_model = model.blank_decoder.target_embed(target)

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

    label_step_out, label_decoder_state_updated = model.label_decoder.loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed,
      segment_lens=segment_lens,
      segment_starts=segment_starts,
      state=label_decoder_state,
    )
    label_logits = model.label_decoder.decode_logits(input_embed=input_embed, **label_step_out)
    label_log_prob = rf.log_softmax(label_logits, axis=model.target_dim)

    # ------------------- external LM step -------------------

    if model.language_model:
      lm_logits, lm_state_updated = model.language_model(
        target_non_blank,
        spatial_dim=single_step_dim,
        state=lm_state,
      )
      label_log_prob += rf.log_softmax(lm_logits, axis=model.target_dim)

    # ------------------- blank step -------------------

    if not model.use_joint_model:
      blank_loop_step_kwargs = dict(
        enc=enc_args["enc"],
        enc_spatial_dim=enc_spatial_dim,
        state=blank_decoder_state,
      )
      if isinstance(model.blank_decoder, BlankDecoderV1):
        blank_loop_step_kwargs["input_embed"] = input_embed_length_model
      else:
        blank_loop_step_kwargs["label_model_state"] = label_step_out["s"]

      blank_step_out, blank_decoder_state = model.blank_decoder.loop_step(**blank_loop_step_kwargs)
      blank_logits = model.blank_decoder.decode_logits(**blank_step_out)
      emit_log_prob = rf.log(rf.sigmoid(blank_logits))
      emit_log_prob = rf.squeeze(emit_log_prob, axis=emit_log_prob.feature_dim)
      blank_log_prob = rf.log(rf.sigmoid(-blank_logits))

    # ------------------- combination -------------------

      label_log_prob += emit_log_prob
      output_log_prob, _ = rf.concat(
        (label_log_prob, model.target_dim), (blank_log_prob, blank_log_prob.feature_dim),
        out_dim=model.align_target_dim
      )
    else:
      output_log_prob = label_log_prob

    # ------------------- top-k -------------------

    if use_recombination:
      seq_log_prob = recombine_seqs(seq_targets, seq_log_prob, seq_backrefs, seq_hash, beam_dim, batch_dims[0], i)
      if i== 3:
        exit()

    seq_log_prob = seq_log_prob + output_log_prob  # Batch, InBeam, Vocab
    old_beam_dim = beam_dim.copy()
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob,
      k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
      axis=[beam_dim, model.target_dim if model.use_joint_model else model.align_target_dim]
    )  # seq_log_prob, backrefs, target: Batch, Beam
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    if use_recombination:
      seq_hash = update_seq_hash(seq_hash, target, backrefs)

    # mask for updating label-sync states
    update_state_mask = rf.convert_to_tensor(target != model.blank_idx)

    # ------------------- update blank decoder state -------------------

    if not model.use_joint_model:
      blank_decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), blank_decoder_state)

    # ------------------- update label decoder state -------------------

    if model.use_joint_model:
      label_decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), label_decoder_state_updated)
    else:
      def _get_masked_state(old, new, mask):
        old = rf.gather(old, indices=backrefs, axis=old_beam_dim)
        new = rf.gather(new, indices=backrefs, axis=old_beam_dim)
        return rf.where(mask, new, old)

      label_decoder_state = tree.map_structure(
        lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
        label_decoder_state, label_decoder_state_updated
      )

    # ------------------- update external LM state -------------------

    if model.language_model:
      def _get_masked_state_lm(old: rf.Tensor, new: rf.Tensor, mask: rf.Tensor):
        if isinstance(old, Dim):
          return new

        def _update(tensor: rf.Tensor):
          tensor = tensor.copy_transpose(batch_dims + [old_beam_dim] + tensor.remaining_dims(batch_dims_))
          tensor_raw_tensor = beam_search_utils.batch_gather(
            tensor.raw_tensor,
            indices=backrefs.copy_transpose(batch_dims + [beam_dim]).raw_tensor
          )
          tensor = tensor.copy_template_replace_dim_tag(1, beam_dim)
          tensor.raw_tensor = tensor_raw_tensor
          return tensor

        old = _update(old)
        new = _update(new)

        return rf.where(mask, new, old)

      lm_state = tree.map_structure(
        lambda old_state, new_state: _get_masked_state_lm(old_state, new_state, update_state_mask),
        lm_state, lm_state_updated
      )

      exit()

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
