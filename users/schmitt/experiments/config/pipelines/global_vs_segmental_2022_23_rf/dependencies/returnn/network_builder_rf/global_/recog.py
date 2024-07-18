from typing import Optional, Dict, Any, Tuple
import tree
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.recog import RecogDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import GlobalAttentionModel


def _trafo_gather_backrefs(s, *, backrefs: Tensor):
  if isinstance(s, Tensor):
    if backrefs.sparse_dim in s.dims:
      return rf.gather(s, indices=backrefs)  # really the default case
    return s  # e.g. scalar or so, independent from beam
  if isinstance(s, Dim):
    assert s.dimension or backrefs not in s.dyn_size_ext.dims  # currently not supported, also not expected
    return s
  raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


def model_recog(
        *,
        model: GlobalAttentionModel,
        data: Tensor,
        data_spatial_dim: Dim,
        beam_size: int,
        max_seq_len: Optional[int] = None,
        external_lm_scale: Optional[float] = None,
        ilm_type: Optional[str] = None,
        ilm_correction_scale: Optional[float] = None,
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

  if ilm_type is not None:
    assert ilm_type in ("mini_att",)
    assert ilm_correction_scale is not None

  # --------------------------------- init encoder, dims, etc ---------------------------------

  enc_args, enc_spatial_dim = model.encoder.encode(data, in_spatial_dim=data_spatial_dim)
  if model.decoder_type == "trafo":
    enc_args["enc"] = model.label_decoder.transform_encoder(enc_args["enc"], axis=enc_spatial_dim)

  if max_seq_len is None:
    max_seq_len = enc_spatial_dim.get_size_tensor()
  else:
    max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")

  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims

  length_normalization_exponent = 1.0

  ended = rf.constant(False, dims=batch_dims_)
  out_seq_len = rf.constant(0, dims=batch_dims_)
  seq_log_prob = rf.constant(0.0, dims=batch_dims_)

  # lists of [B, beam] tensors
  seq_targets = []
  seq_backrefs = []

  # --------------------------------- init states ---------------------------------
  if model.decoder_type == "lstm":
    decoder_state = model.label_decoder.decoder_default_initial_state(
      batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
  else:
    decoder_state = model.label_decoder.default_initial_state(batch_dims=batch_dims_)

  # external LM
  if model.language_model:
    lm_state = model.language_model.default_initial_state(batch_dims=batch_dims_)
  else:
    lm_state = None

  # ILM
  if ilm_type is not None:
    ilm_state = model.label_decoder.decoder_default_initial_state(
      batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
  else:
    ilm_state = None

  # --------------------------------- init targets ---------------------------------

  target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)

  # --------------------------------- main loop ---------------------------------

  i = 0
  while True:
    if model.decoder_type == "lstm":
      # --------------------------------- get embeddings ---------------------------------
      if i == 0:
        input_embed = rf.zeros(
          batch_dims_ + [model.label_decoder.target_embed.out_dim],
          feature_dim=model.label_decoder.target_embed.out_dim)
      else:
        input_embed = model.label_decoder.target_embed(target)

      # --------------------------------- decoder step ---------------------------------

      step_out, decoder_state = model.label_decoder.loop_step(
        **enc_args,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed,
        state=decoder_state,
      )
      logits = model.label_decoder.decode_logits(input_embed=input_embed, **step_out)
    else:
      logits, decoder_state = model.label_decoder(
        target,
        spatial_dim=single_step_dim,
        encoder=enc_args["enc"],
        state=decoder_state,
      )

    label_log_prob = rf.log_softmax(logits, axis=model.target_dim)

    # --------------------------------- external LM step ---------------------------------

    if lm_state is not None:
      lm_logits, lm_state = model.language_model(
        target,
        spatial_dim=single_step_dim,
        state=lm_state,
      )
      lm_label_log_prob = rf.log_softmax(lm_logits, axis=model.target_dim)
      label_log_prob += external_lm_scale * lm_label_log_prob

    # --------------------------------- ILM step ---------------------------------

    if ilm_state is not None:
      assert model.decoder_type == "lstm", "not implemented yet"

      ilm_step_out, ilm_state = model.label_decoder.loop_step(
        **enc_args,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed,
        state=ilm_state,
        use_mini_att=True
      )
      ilm_logits = model.label_decoder.decode_logits(input_embed=input_embed, **ilm_step_out)
      ilm_label_log_prob = rf.log_softmax(ilm_logits, axis=model.target_dim)
      label_log_prob -= ilm_correction_scale * ilm_label_log_prob

    # --------------------------------- filter finished beams, pick top-k ---------------------------------

    # Filter out finished beams
    label_log_prob = rf.where(
      ended,
      rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=-1.0e30),
      label_log_prob,
    )

    seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, model.target_dim]
    )  # seq_log_prob, backrefs, target: Batch, Beam
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    # --------------------------------- update states ---------------------------------

    # decoder
    if model.decoder_type == "lstm":
      decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), decoder_state)
    else:
      decoder_state = tree.map_structure(
        functools.partial(_trafo_gather_backrefs, backrefs=backrefs), decoder_state)

    # external LM
    if lm_state is not None:
      def _get_lm_state(state):
        if isinstance(state, Dim):
          return state

        assert isinstance(state, Tensor)
        if len(state.dims) == 0:
          return state

        return rf.gather(state, indices=backrefs)

      lm_state = tree.map_structure(lambda state: _get_lm_state(state), lm_state)

    # ILM
    if ilm_state is not None:
      ilm_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), ilm_state)

    ended = rf.gather(ended, indices=backrefs)
    out_seq_len = rf.gather(out_seq_len, indices=backrefs)
    i += 1

    ended = rf.logical_or(ended, rf.convert_to_tensor(target == model.eos_idx))
    ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
    if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
      break
    out_seq_len = out_seq_len + rf.where(ended, 0, 1)

    if i > 1 and length_normalization_exponent != 0:
      # Length-normalized scores, so we evaluate score_t/len.
      # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
      # Because we count with EOS symbol, shifted by one.
      seq_log_prob *= rf.where(
        ended,
        (i / (i - 1)) ** length_normalization_exponent,
        1.0,
      )

  if i > 0 and length_normalization_exponent != 0:
    seq_log_prob *= (1 / i) ** length_normalization_exponent

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
  out_spatial_dim = Dim(out_seq_len, name="out-spatial")
  seq_targets = seq_targets__.stack(axis=out_spatial_dim)

  # if model.decoder_type == "trafo":
  #   print("seq_targets:", seq_targets.raw_tensor)
  #   exit()

  return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[GlobalAttentionModel]
model_recog.output_with_beam = True
# output_blank_label=blank is actually wrong for AED, but now we don't change it anymore
# because it would change all recog hashes.
# Also, it does not matter too much -- it will just cause an extra SearchRemoveLabelJob,
# which will not have any effect here.
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False


def model_recog_pure_torch(
        *,
        model: GlobalAttentionModel,
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
  from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.beam_search.label_sync import BeamSearchOpts, label_sync_beam_search
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
  label_scorer.label_scorers["decoder"] = (
    get_label_scorer_pure_torch(model=model, batch_dim=batch_dim, enc=enc, enc_spatial_dim=enc_spatial_dim),
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
    out_seq_len,  # [Batch,FinalBeam]
  ) = label_sync_beam_search(
    label_scorer,
    batch_size=int(batch_dim.get_dim_value()),
    max_seq_len=max_seq_len.copy_compatible_to_dims_raw([batch_dim]),
    device=data.raw_tensor.device,
    opts=BeamSearchOpts(
      **beam_search_opts,
      bos_label=model.label_decoder.bos_idx,
      eos_label=model.label_decoder.eos_idx,
      num_labels=model.target_dim.dimension,
    ),
  )

  beam_dim = Dim(seq_log_prob.shape[1], name="beam")
  out_spatial_dim = Dim(rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim], name="out_spatial"))
  seq_targets_t = rf.convert_to_tensor(
    seq_targets, dims=[batch_dim, beam_dim, out_spatial_dim], sparse_dim=model.target_dim
  )
  seq_log_prob_t = rf.convert_to_tensor(seq_log_prob, dims=[batch_dim, beam_dim])

  return seq_targets_t, seq_log_prob_t, out_spatial_dim, beam_dim


# RecogDef API
model_recog_pure_torch: RecogDef[GlobalAttentionModel]
model_recog_pure_torch.output_with_beam = True
model_recog_pure_torch.output_blank_label = None
model_recog_pure_torch.batch_size_dependent = False


def get_label_scorer_pure_torch(
        *,
        model: GlobalAttentionModel,
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
      decoder_state = model.label_decoder.decoder_default_initial_state(batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
      return tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state)

    def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
            prev_align_label: Optional[torch.Tensor] = None,  # not used
            t: Optional[int] = None,  # not used
    ) -> Tuple[torch.Tensor, Any]:
      """update state"""
      beam_dim = Dim(prev_label.shape[1], name="beam")

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

      input_embed = model.label_decoder.target_embed(
        rf.convert_to_tensor(prev_label, dims=[batch_dim, beam_dim], sparse_dim=model.target_dim)
      )
      decode_out, decoder_state = model.label_decoder.loop_step(
        **enc,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed,
        state=tree.map_structure(_map_raw_to_tensor, prev_state),
      )
      logits = model.label_decoder.decode_logits(input_embed=input_embed, **decode_out)
      label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
      assert set(label_log_prob.dims) == {batch_dim, beam_dim, model.target_dim}

      return (
        self._map_tensor_to_raw(label_log_prob, beam_dim=beam_dim).tensor,
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
