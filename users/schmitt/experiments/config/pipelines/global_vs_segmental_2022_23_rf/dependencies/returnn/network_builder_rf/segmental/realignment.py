from typing import Optional, Dict, Any, Sequence, Tuple, List

import torch

from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import recombination
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder,
  SegmentalAttEfficientLabelDecoder
)

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

from returnn.tensor import Dim, single_step_dim, TensorDict
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray


def model_realign_(
        *,
        model: SegmentalAttentionModel,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: Dim,
):

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio

  batch_dims = data.remaining_dims(data_spatial_dim)
  enc_args, enc_spatial_dim = model.encoder.encode(data, in_spatial_dim=data_spatial_dim)

  max_num_labels = rf.reduce_max(
    non_blank_targets_spatial_dim.dyn_size_ext,
    axis=batch_dims
  )
  max_num_labels = max_num_labels.raw_tensor.item()

  if model.use_joint_model and isinstance(model.label_decoder, SegmentalAttEfficientLabelDecoder):
    segment_starts, segment_lens = utils.get_segment_starts_and_lens(
      rf.sequence_mask(batch_dims + [enc_spatial_dim]),  # this way, every frame is interpreted as non-blank
      enc_spatial_dim,
      model,
      batch_dims,
      enc_spatial_dim
    )

    seq_log_prob, viterbi_alignment, viterbi_alignment_spatial_dim = model_realign_efficient(
      model=model.label_decoder,
      enc=enc_args["enc"],
      enc_ctx=enc_args["enc_ctx"],
      enc_spatial_dim=enc_spatial_dim,
      non_blank_targets=non_blank_targets,
      non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
      segment_starts=segment_starts,
      segment_lens=segment_lens,
      batch_dims=batch_dims,
      beam_size=max_num_labels,
      downsampling=1,
      precompute_chunk_size=10,
      interpolation_alignment=None,
      interpolation_alignment_factor=0.0,
      use_recombination="max",
      return_realignment=True,
    )
  else:
    seq_log_prob, viterbi_alignment, viterbi_alignment_spatial_dim = model_realign(
      model=model,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      non_blank_targets=non_blank_targets,
      non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
      batch_dims=batch_dims,
      beam_size=max_num_labels,
      # use_recombination=None,
      use_recombination="max",
    )
    print("seq_log_prob", seq_log_prob.raw_tensor)
    exit()

  return viterbi_alignment, seq_log_prob, viterbi_alignment_spatial_dim


# def model_realign(
#         *,
#         model: SegmentalAttentionModel,
#         enc_args: Dict[str, rf.Tensor],
#         enc_spatial_dim: Dim,
#         non_blank_targets: rf.Tensor,  # [B, S, V]
#         non_blank_targets_spatial_dim: Dim,
#         batch_dims: List[Dim],
#         beam_size: int,
#         use_recombination: Optional[str] = "sum",
# ) -> Tuple[rf.Tensor, Optional[rf.Tensor], Optional[Dim]]:
#   assert any(
#     isinstance(model.blank_decoder, cls) for cls in (BlankDecoderV1, BlankDecoderV3, BlankDecoderV5, BlankDecoderV6)
#   ) or model.blank_decoder is None, "blank_decoder not supported"
#   if model.blank_decoder is None:
#     assert model.use_joint_model, "blank_decoder is None, so use_joint_model must be True"
#   assert model.label_decoder_state in {"nb-lstm", "joint-lstm", "nb-2linear-ctx1"}
#
#   # --------------------------------- init dims, etc ---------------------------------
#
#   max_seq_len = enc_spatial_dim.get_size_tensor()
#   max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)
#
#   beam_dim = Dim(1, name="initial-beam")
#   batch_dims_ = [beam_dim] + batch_dims
#   backrefs = rf.zeros(batch_dims_, dtype="int32")
#
#   bos_idx = 0
#
#   seq_log_prob = rf.constant(0.0, dims=batch_dims_)
#
#   # lists of [B, beam] tensors
#   seq_targets = []
#   seq_backrefs = []
#
#   update_state_mask = rf.constant(True, dims=batch_dims_)
#
#   # --------------------------------- init states ---------------------------------
#
#   # label decoder
#   label_decoder_state = model.label_decoder.default_initial_state(batch_dims=batch_dims_, )
#
#   # blank decoder
#   if model.blank_decoder is not None:
#     blank_decoder_state = model.blank_decoder.default_initial_state(batch_dims=batch_dims_)
#   else:
#     blank_decoder_state = None
#
#   # --------------------------------- init targets, embeddings ---------------------------------
#
#   if model.use_joint_model:
#     target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
#     if model.label_decoder_state in ("nb-lstm", "nb-2linear-ctx1"):
#       target_non_blank = target.copy()
#   else:
#     target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.align_target_dim)
#     target_non_blank = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
#
#   input_embed = rf.zeros(
#     batch_dims_ + [model.label_decoder.target_embed.out_dim],
#     feature_dim=model.label_decoder.target_embed.out_dim,
#     dtype="float32"
#   )
#
#   if isinstance(model.blank_decoder, BlankDecoderV1):
#     input_embed_length_model = rf.zeros(
#       batch_dims_ + [model.blank_decoder.target_embed.out_dim], feature_dim=model.blank_decoder.target_embed.out_dim)
#   else:
#     input_embed_length_model = None
#
#   # add blank idx on the right
#   # this way, when the label index for gathering reached the last non-blank index, it will gather blank after that
#   # which then only allows corresponding hypotheses to be extended by blank
#   non_blank_targets_padded, non_blank_targets_padded_spatial_dim = rf.pad(
#     non_blank_targets,
#     axes=[non_blank_targets_spatial_dim],
#     padding=[(0, 1)],
#     value=model.blank_idx,
#   )
#   non_blank_targets_padded_spatial_dim = non_blank_targets_padded_spatial_dim[0]
#
#   # ------------------------ sizes ------------------------
#
#   non_blank_targets_padded_spatial_sizes = rf.copy_to_device(
#     non_blank_targets_padded_spatial_dim.dyn_size_ext, non_blank_targets.device
#   )
#   non_blank_targets_spatial_sizes = rf.copy_to_device(
#     non_blank_targets_spatial_dim.dyn_size_ext, non_blank_targets.device)
#   max_num_labels = rf.reduce_max(
#     non_blank_targets_spatial_sizes, axis=non_blank_targets_spatial_sizes.dims
#   ).raw_tensor.item()
#   single_col_dim = Dim(dimension=max_num_labels + 1, name="max-num-labels")
#   label_indices = rf.zeros(batch_dims_, dtype="int32", sparse_dim=single_col_dim)
#   prev_label_indices = label_indices.copy()
#   enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext, non_blank_targets.device)
#
#   target_dim = model.target_dim if model.use_joint_model else model.align_target_dim
#   vocab_range = rf.range_over_dim(target_dim)
#   blank_tensor = rf.convert_to_tensor(model.blank_idx, dtype=vocab_range.dtype)
#
#   # --------------------------------- main loop ---------------------------------
#
#   i = 0
#   while i < max_seq_len.raw_tensor:
#     if i > 0:
#       if model.label_decoder_state == "joint-lstm":
#         input_embed = model.label_decoder.target_embed(target)
#       else:
#         target_non_blank = rf.where(update_state_mask, target, rf.gather(target_non_blank, indices=backrefs))
#         target_non_blank.sparse_dim = model.label_decoder.target_embed.in_dim
#         input_embed = rf.where(
#           update_state_mask,
#           model.label_decoder.target_embed(target_non_blank),
#           rf.gather(input_embed, indices=backrefs)
#         )
#       if isinstance(model.blank_decoder, BlankDecoderV1):
#         input_embed_length_model = model.blank_decoder.target_embed(target)
#
#       label_indices = rf.where(
#         update_state_mask,
#         rf.where(
#           prev_label_indices == non_blank_targets_padded_spatial_sizes - 1,
#           prev_label_indices,
#           prev_label_indices + 1
#         ),
#         prev_label_indices
#       )
#
#     # ------------------- label step -------------------
#
#     center_position = rf.minimum(
#       rf.full(dims=[beam_dim] + batch_dims, fill_value=i, dtype="int32"),
#       rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, input_embed.device)
#     )
#     segment_starts = rf.maximum(
#       rf.convert_to_tensor(0, dtype="int32"), center_position - model.center_window_size // 2)
#     segment_ends = rf.minimum(
#       rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, input_embed.device),
#       center_position + model.center_window_size // 2
#     )
#     segment_lens = segment_ends - segment_starts + 1
#
#     label_step_out, label_decoder_state_updated = model.label_decoder.loop_step(
#       **enc_args,
#       enc_spatial_dim=enc_spatial_dim,
#       input_embed=input_embed,
#       segment_lens=segment_lens,
#       segment_starts=segment_starts,
#       state=label_decoder_state,
#     )
#     label_logits = model.label_decoder.decode_logits(input_embed=input_embed, **label_step_out)
#     label_log_prob = rf.log_softmax(label_logits, axis=model.target_dim)
#
#     # ------------------- blank step -------------------
#
#     if blank_decoder_state is not None:
#       if model.blank_decoder_version in (1, 3):
#         blank_loop_step_kwargs = dict(
#           enc=enc_args["enc"],
#           enc_spatial_dim=enc_spatial_dim,
#           state=blank_decoder_state,
#         )
#         if isinstance(model.blank_decoder, BlankDecoderV1):
#           blank_loop_step_kwargs["input_embed"] = input_embed_length_model
#         else:
#           blank_loop_step_kwargs["label_model_state"] = label_step_out["s"]
#
#         blank_step_out, blank_decoder_state = model.blank_decoder.loop_step(**blank_loop_step_kwargs)
#         blank_logits = model.blank_decoder.decode_logits(**blank_step_out)
#       else:
#         assert isinstance(model.blank_decoder, BlankDecoderV5) or isinstance(model.blank_decoder, BlankDecoderV6)
#         enc_position = rf.minimum(
#           rf.full(dims=batch_dims, fill_value=i, dtype="int32"),
#           rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, input_embed.device)
#         )
#         enc_frame = rf.gather(enc_args["enc"], indices=enc_position, axis=enc_spatial_dim)
#         enc_frame = rf.expand_dim(enc_frame, beam_dim)
#         if isinstance(model.blank_decoder, BlankDecoderV5):
#           # no LSTM -> no state -> just leave (empty) state as is
#           blank_logits = model.blank_decoder.emit_prob(
#             rf.concat_features(enc_frame, label_step_out["s"]))
#         else:
#           prev_lstm_state = blank_decoder_state.s_blank
#           blank_decoder_state = rf.State()
#           s_blank, blank_decoder_state.s_blank = model.blank_decoder.s(
#             enc_frame,
#             state=prev_lstm_state,
#             spatial_dim=single_step_dim
#           )
#           blank_logits = model.blank_decoder.emit_prob(rf.concat_features(s_blank, label_step_out["s"]))
#
#       emit_log_prob = rf.log(rf.sigmoid(blank_logits))
#       emit_log_prob = rf.squeeze(emit_log_prob, axis=emit_log_prob.feature_dim)
#       blank_log_prob = rf.log(rf.sigmoid(-blank_logits))
#
#       # ------------------- combination -------------------
#
#       label_log_prob += emit_log_prob
#       output_log_prob, _ = rf.concat(
#         (label_log_prob, model.target_dim), (blank_log_prob, blank_log_prob.feature_dim),
#         out_dim=model.align_target_dim
#       )
#     else:
#       output_log_prob = label_log_prob
#
#     label_ground_truth = rf.gather(
#       non_blank_targets_padded,
#       indices=label_indices,
#       axis=non_blank_targets_padded_spatial_dim,
#       clip_to_valid=True
#     )
#     # mask label log prob in order to only allow hypotheses corresponding to the ground truth:
#     # log prob needs to correspond to the next non-blank label...
#     output_log_prob_mask = vocab_range == label_ground_truth
#     rem_frames = enc_spatial_sizes - i
#     rem_labels = non_blank_targets_spatial_sizes - label_indices
#     # ... or to blank if there are more frames than labels left
#     output_log_prob_mask = rf.logical_or(
#       output_log_prob_mask,
#       rf.logical_and(
#         vocab_range == blank_tensor,
#         rem_frames > rem_labels
#       )
#     )
#     output_log_prob = rf.where(
#       output_log_prob_mask,
#       output_log_prob,
#       rf.constant(-1.0e30, dims=batch_dims + [beam_dim, target_dim])
#     )
#
#     # for shorter seqs in the batch, set the blank score to zero and the others to ~-inf
#     output_log_prob = rf.where(
#       rf.convert_to_tensor(i >= rf.copy_to_device(enc_spatial_dim.get_size_tensor(), input_embed.device)),
#       rf.sparse_to_dense(
#         model.blank_idx,
#         axis=target_dim,
#         label_value=0.0,
#         other_value=-1.0e30
#       ),
#       output_log_prob
#     )
#
#     # ------------------- recombination -------------------
#
#     if use_recombination is not None:
#       seq_log_prob = recombination.recombine_seqs_train(
#         seq_log_prob=seq_log_prob,
#         label_log_prob=output_log_prob,
#         label_indices=label_indices,
#         ground_truth=label_ground_truth,
#         target_dim=target_dim,
#         single_col_dim=single_col_dim,
#         beam_dim=beam_dim,
#         batch_dims=batch_dims,
#         blank_idx=model.blank_idx,
#         use_sum_recombination=use_recombination == "sum"
#       )
#       beam_size_ = min(
#         min((i + 2), rf.reduce_max(rem_frames, axis=rem_frames.dims).raw_tensor.item()),
#         min((max_num_labels + 1), beam_size)
#       )
#       seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
#         seq_log_prob,
#         k_dim=Dim(beam_size_, name=f"dec-step{i}-beam"),
#         axis=[beam_dim, single_col_dim]
#       )
#
#       prev_label_indices = rf.gather(label_indices, indices=backrefs)
#       label_indices = target
#       # mask blank label
#       update_state_mask = rf.convert_to_tensor(target != prev_label_indices)
#
#       target = rf.where(
#         update_state_mask,
#         rf.gather(
#           non_blank_targets_padded,
#           indices=rf.cast(label_indices, "int32"),
#           axis=non_blank_targets_padded_spatial_dim,
#           clip_to_valid=True
#         ),
#         model.blank_idx
#       )
#
#       # print("target", target.raw_tensor)
#       # exit()
#     else:
#       beam_size_ = beam_size
#
#       seq_log_prob = seq_log_prob + output_log_prob  # Batch, InBeam, Vocab
#       seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
#         seq_log_prob,
#         k_dim=Dim(beam_size_, name=f"dec-step{i}-beam"),
#         axis=[beam_dim, target_dim]
#       )
#
#       prev_label_indices = rf.gather(label_indices, indices=backrefs)
#       # mask blank label
#       update_state_mask = rf.convert_to_tensor(target != model.blank_idx)
#
#     seq_targets.append(target)
#     seq_backrefs.append(backrefs)
#
#     # mask for updating label-sync states
#     update_state_mask = rf.convert_to_tensor(target != model.blank_idx)
#
#     # ------------------- update blank decoder state -------------------
#
#     if blank_decoder_state is not None:
#       blank_decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), blank_decoder_state)
#
#     # ------------------- update label decoder state and ILM state -------------------
#
#     def _get_masked_state(old, new, mask):
#       old = rf.gather(old, indices=backrefs)
#       new = rf.gather(new, indices=backrefs)
#       return rf.where(mask, new, old)
#
#     # label decoder
#     if model.label_decoder_state == "joint-lstm":
#       label_decoder_state = tree.map_structure(
#         lambda s: rf.gather(s, indices=backrefs), label_decoder_state_updated)
#     else:
#       label_decoder_state = tree.map_structure(
#         lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
#         label_decoder_state, label_decoder_state_updated
#       )
#
#     i += 1
#
#   # Backtrack via backrefs, resolve beams.
#   seq_targets_ = []
#   indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
#   for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
#     # indices: FinalBeam -> Beam
#     # backrefs: Beam -> PrevBeam
#     seq_targets_.insert(0, rf.gather(target, indices=indices))
#     indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam
#
#   seq_targets__ = TensorArray(seq_targets_[0])
#   for target in seq_targets_:
#     seq_targets__ = seq_targets__.push_back(target)
#   seq_targets = seq_targets__.stack(axis=enc_spatial_dim)
#
#   return seq_log_prob, seq_targets, enc_spatial_dim

import time
def model_realign(
        *,
        model: SegmentalAttentionModel,
        enc_args: Dict[str, rf.Tensor],
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,  # [B, S, V]
        non_blank_targets_spatial_dim: Dim,
        batch_dims: List[Dim],
        beam_size: int,
        use_recombination: Optional[str] = "sum",
        only_use_blank_model: bool = False,
) -> Tuple[rf.Tensor, Optional[rf.Tensor], Optional[Dim]]:
  assert any(
    isinstance(model.blank_decoder, cls) for cls in (BlankDecoderV1, BlankDecoderV3, BlankDecoderV5, BlankDecoderV6)
  ) or model.blank_decoder is None, "blank_decoder not supported"
  if model.blank_decoder is None:
    assert model.use_joint_model, "blank_decoder is None, so use_joint_model must be True"
  assert model.label_decoder_state in {"nb-lstm", "joint-lstm", "nb-2linear-ctx1"}

  # --------------------------------- init dims, etc ---------------------------------

  max_seq_len = enc_spatial_dim.get_size_tensor()
  max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)

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

  # --------------------------------- init targets, embeddings ---------------------------------

  if model.use_joint_model:
    target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    if model.label_decoder_state in ("nb-lstm", "nb-2linear-ctx1"):
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

  # add blank idx on the right
  # this way, when the label index for gathering reached the last non-blank index, it will gather blank after that
  # which then only allows corresponding hypotheses to be extended by blank
  non_blank_targets_padded, non_blank_targets_padded_spatial_dim = rf.pad(
    non_blank_targets,
    axes=[non_blank_targets_spatial_dim],
    padding=[(0, 1)],
    value=model.blank_idx,
  )
  non_blank_targets_padded_spatial_dim = non_blank_targets_padded_spatial_dim[0]

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
  label_indices = rf.zeros(batch_dims_, dtype="int64", sparse_dim=single_col_dim)
  prev_label_indices = label_indices.copy()
  enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext, non_blank_targets.device)

  target_dim = model.target_dim if model.use_joint_model else model.align_target_dim
  vocab_range = rf.range_over_dim(target_dim)
  blank_tensor = rf.convert_to_tensor(model.blank_idx, dtype=vocab_range.dtype)

  # label_step_time = 0
  # embedding_time = 0
  # recombination_time = 0
  # segment_boundary_time = 0
  # label_ground_truth_stuff_time = 0
  # top_k_time = 0
  # update_state_time = 0
  # total_time = time.time()

  # --------------------------------- main loop ---------------------------------

  i = 0
  while i < max_seq_len.raw_tensor:
    if i > 0:

      # torch.cuda.synchronize()
      # before = time.time()

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

      label_indices = rf.where(
        update_state_mask,
        rf.where(
          prev_label_indices == non_blank_targets_padded_spatial_sizes - 1,
          prev_label_indices,
          prev_label_indices + 1
        ),
        prev_label_indices
      )

      # torch.cuda.synchronize()
      # embedding_time += time.time() - before

    # ------------------- label step -------------------

    # before = time.time()

    if not only_use_blank_model:
      center_position = rf.minimum(
        rf.full(dims=[beam_dim] + batch_dims, fill_value=i, dtype="int32"),
        rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, input_embed.device)
      )
      segment_starts = rf.maximum(
        rf.convert_to_tensor(0, dtype="int32"), center_position - model.center_window_size // 2)
      segment_ends = rf.minimum(
        rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, input_embed.device),
        center_position + model.center_window_size // 2
      )
      segment_lens = segment_ends - segment_starts + 1

      # segment_boundary_time += time.time() - before
      #
      # before = time.time()
      # torch.cuda.synchronize()

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

    # torch.cuda.synchronize()
    # label_step_time += time.time() - before

    # ------------------- blank step -------------------

    if blank_decoder_state is not None:
      if model.blank_decoder_version in (1, 3):
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
      else:
        assert isinstance(model.blank_decoder, BlankDecoderV5) or isinstance(model.blank_decoder, BlankDecoderV6)
        enc_position = rf.minimum(
          rf.full(dims=batch_dims, fill_value=i, dtype="int32"),
          rf.copy_to_device(enc_spatial_dim.get_size_tensor() - 1, input_embed.device)
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

      # ------------------- combination -------------------

      if only_use_blank_model:
        label_log_prob = rf.zeros(dims=batch_dims + [beam_dim, model.target_dim])

      label_log_prob += emit_log_prob
      output_log_prob, _ = rf.concat(
        (label_log_prob, model.target_dim), (blank_log_prob, blank_log_prob.feature_dim),
        out_dim=model.align_target_dim
      )
    else:
      output_log_prob = label_log_prob

    # before = time.time()
    # torch.cuda.synchronize()

    label_ground_truth = rf.gather(
      non_blank_targets_padded,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )
    # mask label log prob in order to only allow hypotheses corresponding to the ground truth:
    # log prob needs to correspond to the next non-blank label...
    output_log_prob_mask = vocab_range == label_ground_truth
    rem_frames = enc_spatial_sizes - i
    rem_labels = non_blank_targets_spatial_sizes - label_indices
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
      rf.constant(-1.0e30, dims=batch_dims + [beam_dim, target_dim])
    )

    # for shorter seqs in the batch, set the blank score to zero and the others to ~-inf
    output_log_prob = rf.where(
      rf.convert_to_tensor(i >= rf.copy_to_device(enc_spatial_dim.get_size_tensor(), input_embed.device)),
      rf.sparse_to_dense(
        model.blank_idx,
        axis=target_dim,
        label_value=0.0,
        other_value=-1.0e30
      ),
      output_log_prob
    )

    # torch.cuda.synchronize()
    # label_ground_truth_stuff_time += time.time() - before

    # ------------------- recombination -------------------

    if use_recombination is not None:
      # before = time.time()
      # torch.cuda.synchronize()

      seq_log_prob = recombination.recombine_seqs(
        seq_targets=seq_targets,
        seq_log_prob=seq_log_prob,
        seq_hash=seq_hash,
        beam_dim=beam_dim,
        batch_dim=batch_dims[0],
        use_sum=use_recombination == "sum",
      )

      # torch.cuda.synchronize()
      # recombination_time += time.time() - before

      # beam_size_ = min(
      #   min((i + 2), rf.reduce_max(rem_frames, axis=rem_frames.dims).raw_tensor.item()),
      #   min((max_num_labels + 1), beam_size)
      # )
      beam_size_ = beam_size
    else:
      beam_size_ = beam_size

    # before = time.time()
    # torch.cuda.synchronize()

    seq_log_prob = seq_log_prob + output_log_prob  # Batch, InBeam, Vocab
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob,
      k_dim=Dim(beam_size_, name=f"dec-step{i}-beam"),
      axis=[beam_dim, target_dim]
    )

    prev_label_indices = rf.gather(label_indices, indices=backrefs)
    # mask blank label
    update_state_mask = rf.convert_to_tensor(target != model.blank_idx)

    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    # torch.cuda.synchronize()
    # top_k_time += time.time() - before

    if use_recombination:
      # torch.cuda.synchronize()
      # before = time.time()

      seq_hash = recombination.update_seq_hash(seq_hash, target, backrefs, model.blank_idx)

      # torch.cuda.synchronize()
      # recombination_time += time.time() - before

    # torch.cuda.synchronize()
    # before = time.time()

    # mask for updating label-sync states
    update_state_mask = rf.convert_to_tensor(target != model.blank_idx)

    # ------------------- update blank decoder state -------------------

    if blank_decoder_state is not None:
      blank_decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), blank_decoder_state)

    # ------------------- update label decoder state and ILM state -------------------

    if not only_use_blank_model:
      def _get_masked_state(old, new, mask):
        old = rf.gather(old, indices=backrefs)
        new = rf.gather(new, indices=backrefs)
        return rf.where(mask, new, old)

      # label decoder
      if model.label_decoder_state == "joint-lstm":
        label_decoder_state = tree.map_structure(
          lambda s: rf.gather(s, indices=backrefs), label_decoder_state_updated)
      else:
        label_decoder_state = tree.map_structure(
          lambda old_state, new_state: _get_masked_state(old_state, new_state, update_state_mask),
          label_decoder_state, label_decoder_state_updated
        )

    # torch.cuda.synchronize()
    # update_state_time += time.time() - before

    i += 1

  # last recombination
  if use_recombination:
    # torch.cuda.synchronize()
    # before = time.time()

    seq_log_prob = recombination.recombine_seqs(
      seq_targets,
      seq_log_prob,
      seq_hash,
      beam_dim,
      batch_dims[0],
      use_sum=use_recombination == "sum"
    )

    # print("seq_log_prob", seq_log_prob.raw_tensor)
    # print("backrefs", seq_backrefs[-1].raw_tensor)
    # print("seq_targets", seq_targets[-1].raw_tensor)
    # exit()

    # torch.cuda.synchronize()
    # recombination_time += time.time() - before

  # torch.cuda.synchronize()
  # print("label_step_time", label_step_time)
  # print("embedding_time", embedding_time)
  # print("segment_boundary_time", segment_boundary_time)
  # print("label_ground_truth_stuff_time", label_ground_truth_stuff_time)
  # print("top_k_time", top_k_time)
  # print("update_state_time", update_state_time)
  # print("recombination_time", recombination_time)
  # print("total_time_manual", label_step_time + segment_boundary_time + label_ground_truth_stuff_time + top_k_time + update_state_time + recombination_time + embedding_time)
  # print("total_time", time.time() - total_time)

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

  if use_recombination:
    # only one hyp with score > -1.0e30 because of recombination
    best_hyps = rf.reduce_argmax(seq_log_prob, axis=beam_dim)
    seq_log_prob = rf.reduce_max(seq_log_prob, axis=beam_dim)
    seq_targets = rf.gather(
      seq_targets,
      indices=best_hyps,
      axis=beam_dim,
    )

  return seq_log_prob, seq_targets, enc_spatial_dim

import time
def model_realign_efficient(
        *,
        model: SegmentalAttEfficientLabelDecoder,
        enc: rf.Tensor,
        enc_ctx: rf.Tensor,
        enc_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,  # [B, S, V]
        non_blank_targets_spatial_dim: Dim,
        segment_starts: rf.Tensor,  # [B, T]
        segment_lens: rf.Tensor,  # [B, T]
        batch_dims: List[Dim],
        beam_size: int,
        downsampling: int,
        precompute_chunk_size: int,
        interpolation_alignment: Optional[rf.Tensor],
        interpolation_alignment_factor: float,
        use_recombination: Optional[str] = "sum",
        return_realignment: bool = False,
) -> Tuple[rf.Tensor, Optional[rf.Tensor], Optional[Dim]]:
  assert len(batch_dims) == 1, "not supported yet"
  assert model.decoder_state == "nb-lstm"
  if interpolation_alignment_factor > 0.0:
    assert interpolation_alignment is not None

  # ------------------------ downsample encoder ------------------------

  if downsampling > 1:
    enc_spatial_dim_downsampled = enc_spatial_dim // downsampling
    downsample_indices = rf.range_over_dim(enc_spatial_dim_downsampled) * downsampling
    downsample_indices = rf.expand_dim(downsample_indices, batch_dims[0])
    segment_starts_downsampled = rf.gather(segment_starts, indices=downsample_indices, axis=enc_spatial_dim, clip_to_valid=True)
    segment_lens_downsampled = rf.gather(segment_lens, indices=downsample_indices, axis=enc_spatial_dim, clip_to_valid=True)

    segment_starts = segment_starts_downsampled
    segment_lens = segment_lens_downsampled
  else:
    enc_spatial_dim_downsampled = enc_spatial_dim

  # ------------------------ init some variables ------------------------
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims
  bos_idx = 0
  seq_log_prob = rf.constant(0.0, dims=batch_dims_)
  max_seq_len = enc_spatial_dim_downsampled.get_size_tensor()
  max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)
  label_lstm_state = model.s_wo_att.default_initial_state(batch_dims=batch_dims)
  target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
  vocab_range = rf.range_over_dim(model.target_dim)
  blank_tensor = rf.convert_to_tensor(model.blank_idx, dtype=vocab_range.dtype)
  backrefs = rf.zeros(batch_dims_, dtype="int32")
  update_state_mask = rf.ones(batch_dims_, dtype="bool")

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
  prev_label_indices = label_indices.copy()

  enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim_downsampled.dyn_size_ext, non_blank_targets.device)

  # ------------------------ compute LSTM sequence ------------------------

  start_time = time.time()
  label_lstm_out_seq, _ = model.s_wo_att(
    non_blank_input_embeddings,
    state=label_lstm_state,
    spatial_dim=non_blank_targets_padded_spatial_dim,
  )
  lstm_time = time.time() - start_time
  att_time = 0
  logit_time = 0
  softmax_time = 0
  gather_time = 0
  recombination_time = 0
  rem_time = 0

  # ------------------------ chunk dim ------------------------

  chunk_dim = Dim(precompute_chunk_size, name="chunk")
  chunk_range = rf.expand_dim(rf.range_over_dim(chunk_dim), batch_dims[0])

  i = 0
  seq_targets = []
  seq_backrefs = []

  torch.cuda.synchronize()
  before_loop_time = time.time()
  while i < max_seq_len.raw_tensor:
    # get current number of labels for each hypothesis
    if i > 0:
      before_where_time = time.time()
      label_indices = rf.where(
        update_state_mask,
        rf.where(
          prev_label_indices == non_blank_targets_padded_spatial_sizes - 1,
          prev_label_indices,
          prev_label_indices + 1
        ),
        prev_label_indices
      )
      torch.cuda.synchronize()
      rem_time += time.time() - before_where_time

    torch.cuda.synchronize()
    before_gather_time = time.time()
    # gather ground truth, input embeddings and LSTM output for current label index
    label_ground_truth = rf.gather(
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
    torch.cuda.synchronize()
    gather_time += time.time() - before_gather_time

    # precompute attention for the current chunk (more efficient than computing it individually for each label index)
    if i % precompute_chunk_size == 0:
      torch.cuda.synchronize()
      before_gather_time = time.time()
      seg_starts = rf.gather(
        segment_starts,
        indices=chunk_range,
        axis=enc_spatial_dim_downsampled,
        clip_to_valid=True
      )
      seg_lens = rf.gather(
        segment_lens,
        indices=chunk_range,
        axis=enc_spatial_dim_downsampled,
        clip_to_valid=True
      )
      torch.cuda.synchronize()
      gather_time += time.time() - before_gather_time
      before_att_time = time.time()
      att = model(
        enc=enc,
        enc_ctx=enc_ctx,
        enc_spatial_dim=enc_spatial_dim,
        s=label_lstm_out_seq,
        segment_starts=seg_starts,
        segment_lens=seg_lens,
      )  # [B, S+1, T, D]
      torch.cuda.synchronize()
      att_time += time.time() - before_att_time
      chunk_range += precompute_chunk_size

    # gather attention for the current label index
    torch.cuda.synchronize()
    before_gather_time = time.time()
    att_step = rf.gather(
      att,
      indices=label_indices,
      axis=non_blank_targets_padded_spatial_dim,
      clip_to_valid=True
    )
    att_step = rf.gather(
      att_step,
      indices=rf.constant(i % precompute_chunk_size, dims=batch_dims, device=att_step.device),
      axis=chunk_dim,
      clip_to_valid=True
    )
    torch.cuda.synchronize()
    gather_time += time.time() - before_gather_time

    torch.cuda.synchronize()
    before_logits_time = time.time()
    logits = model.decode_logits(
      input_embed=input_embed,
      att=att_step,
      s=label_lstm_out,
    )  # [B, S+1, T, D]
    torch.cuda.synchronize()
    logit_time += time.time() - before_logits_time
    before_softmax_time = time.time()
    label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
    torch.cuda.synchronize()
    softmax_time += time.time() - before_softmax_time

    # mask label log prob in order to only allow hypotheses corresponding to the ground truth:
    # log prob needs to correspond to the next non-blank label...
    log_prob_mask = vocab_range == label_ground_truth
    rem_frames = enc_spatial_sizes - i
    rem_labels = non_blank_targets_spatial_sizes - label_indices
    torch.cuda.synchronize()
    before_or_time = time.time()
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

    # interpolate with given alignment: a * one_hot + (1 - a) * label_log_prob
    if interpolation_alignment_factor > 0.0:
      interpolation_alignment_ground_truth = rf.gather(
        interpolation_alignment,
        indices=i,
        axis=enc_spatial_dim_downsampled,
      )
      label_log_prob = interpolation_alignment_factor * rf.log(rf.sparse_to_dense(
        interpolation_alignment_ground_truth, label_value=0.9, other_value=0.1 / model.target_dim.dimension
      )) + (1 - interpolation_alignment_factor) * label_log_prob

    label_log_prob = rf.where(
      rf.convert_to_tensor(i >= enc_spatial_sizes),
      rf.sparse_to_dense(
        model.blank_idx,
        axis=model.target_dim,
        label_value=0.0,
        other_value=-1.0e30
      ),
      label_log_prob
    )
    torch.cuda.synchronize()
    rem_time += time.time() - before_or_time

    if use_recombination is not None:
      torch.cuda.synchronize()
      before_recombination_time = time.time()
      seq_log_prob = recombination.recombine_seqs_train(
        seq_log_prob=seq_log_prob,
        label_log_prob=label_log_prob,
        label_indices=label_indices,
        ground_truth=label_ground_truth,
        target_dim=model.target_dim,
        single_col_dim=single_col_dim,
        beam_dim=beam_dim,
        batch_dims=batch_dims,
        blank_idx=model.blank_idx,
        use_sum_recombination=use_recombination == "sum"
      )
      torch.cuda.synchronize()
      recombination_time += time.time() - before_recombination_time
      beam_size_ = min(
        min((i + 2), rf.reduce_max(rem_frames, axis=rem_frames.dims).raw_tensor.item()),
        min((max_num_labels + 1), beam_size)
      )
      torch.cuda.synchronize()
      before_top_k_time = time.time()
      seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
        seq_log_prob,
        k_dim=Dim(beam_size_, name=f"dec-step{i}-beam"),
        axis=[beam_dim, single_col_dim]
      )

      prev_label_indices = rf.gather(label_indices, indices=backrefs)
      # mask blank label
      update_state_mask = rf.convert_to_tensor(target != prev_label_indices)
      torch.cuda.synchronize()
      rem_time += time.time() - before_top_k_time
    else:
      beam_size_ = beam_size

      seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
      seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
        seq_log_prob,
        k_dim=Dim(beam_size_, name=f"dec-step{i}-beam"),
        axis=[beam_dim, model.target_dim]
      )

      prev_label_indices = rf.gather(label_indices, indices=backrefs)
      # mask blank label
      update_state_mask = rf.convert_to_tensor(target != model.blank_idx)

    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    i += 1

  print("att time", att_time)
  print("logit time", logit_time)
  print("softmax time", softmax_time)
  print("gather time", gather_time)
  print("recombination time", recombination_time)
  print("rem time", rem_time)
  print("------------------------------------------")
  print("sum of the above", att_time + logit_time + softmax_time + gather_time + recombination_time + rem_time)
  print("loop time", time.time() - before_loop_time)
  print("lstm time", lstm_time)
  print()

  if return_realignment:
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
    seq_targets = seq_targets__.stack(axis=enc_spatial_dim_downsampled)

    if use_recombination is not None:
      seq_targets_padded = rf.shift_right(seq_targets, axis=enc_spatial_dim_downsampled, pad_value=0)
      seq_targets = rf.where(
        seq_targets == seq_targets_padded,
        model.blank_idx,
        rf.gather(
          non_blank_targets, indices=rf.cast(seq_targets - 1, "int32"), axis=non_blank_targets_spatial_dim, clip_to_valid=True)
      )
      realignment = rf.squeeze(seq_targets, beam_dim)
    else:
      realignment = seq_targets

    realignment_spatial_dim = enc_spatial_dim_downsampled
  else:
    realignment = None
    realignment_spatial_dim = None

  if use_recombination is not None:
    seq_log_prob = rf.squeeze(seq_log_prob, beam_dim)

  return seq_log_prob, realignment, realignment_spatial_dim


def _returnn_v2_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  import returnn.frontend as rf
  from returnn.tensor import Tensor, Dim, batch_dim
  from returnn.config import get_global_config

  if rf.is_executing_eagerly():
    batch_size = int(batch_dim.get_dim_value())
    for batch_idx in range(batch_size):
      seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
      print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  data = extern_data[default_input_key]
  data_spatial_dim = data.get_time_dim_tag()
  realign_def = config.typed_value("_realign_def")

  default_target_key = config.typed_value("target")
  targets = extern_data[default_target_key]
  targets_spatial_dim = targets.get_time_dim_tag()

  realign_out = realign_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    non_blank_targets=targets,
    non_blank_targets_spatial_dim=targets_spatial_dim,
  )

  if len(realign_out) == 3:
    # realign results including viterbi_align,
    # log probs {batch,},
    # out_spatial_dim,
    viterbi_align, scores, out_spatial_dim = realign_out
  else:
    raise ValueError(f"unexpected num outputs {len(realign_out)} from recog_def")
  assert isinstance(viterbi_align, Tensor) and isinstance(scores, Tensor)
  assert isinstance(out_spatial_dim, Dim)
  rf.get_run_ctx().mark_as_output(viterbi_align, "viterbi_align", dims=[batch_dim, out_spatial_dim])
  rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim,])


_v2_forward_out_scores_filename = "scores.py.gz"
_v2_forward_out_alignment_filename = "realignment.hdf"


def _returnn_v2_get_forward_callback():
  from typing import TextIO
  import numpy as np
  from returnn.tensor import Tensor, TensorDict
  from returnn.forward_iface import ForwardCallbackIface
  from returnn.config import get_global_config
  from returnn.datasets.hdf import SimpleHDFWriter

  class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
    def __init__(self):
      self.score_file: Optional[TextIO] = None
      self.alignment_file: Optional[SimpleHDFWriter] = None

    def init(self, *, model):
      import gzip

      self.score_file = gzip.open(_v2_forward_out_scores_filename, "wt")
      self.score_file.write("{\n")

      self.alignment_file = SimpleHDFWriter(
        filename=_v2_forward_out_alignment_filename, dim=model.target_dim.dimension, ndim=1
      )

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
      viterbi_align: Tensor = outputs["viterbi_align"]  # [T]
      scores: Tensor = outputs["scores"]  # []
      assert len(viterbi_align.dims) == 1, f"expected hyps to be 1D, but got {viterbi_align.dims}"
      assert viterbi_align.dims[0].dyn_size_ext, f"viterbi_align {viterbi_align} does not define seq lengths"
      self.score_file.write(f"{seq_tag!r}: ")
      score = float(scores.raw_tensor)
      self.score_file.write(f"{score!r},\n")

      seq_len = viterbi_align.dims[0].dyn_size_ext.raw_tensor.item()
      viterbi_align_raw = viterbi_align.raw_tensor[:seq_len]

      hdf.dump_hdf_numpy(
        hdf_dataset=self.alignment_file,
        data=viterbi_align_raw[None],  # [1, T]
        seq_lens=np.array([seq_len]),  # [1]
        seq_tags=[seq_tag],
      )

    def finish(self):
      self.score_file.write("}\n")
      self.score_file.close()
      self.alignment_file.close()

  return _ReturnnRecogV2ForwardCallbackIface()
