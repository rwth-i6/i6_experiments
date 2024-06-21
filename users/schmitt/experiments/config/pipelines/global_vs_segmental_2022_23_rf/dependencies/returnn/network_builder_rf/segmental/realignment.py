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
  assert model.use_joint_model
  assert isinstance(model.label_decoder, SegmentalAttEfficientLabelDecoder)
  assert model.label_decoder_state == "nb-lstm"

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio

  batch_dims = data.remaining_dims(data_spatial_dim)
  enc_args, enc_spatial_dim = model.encoder.encode(data, in_spatial_dim=data_spatial_dim)

  segment_starts, segment_lens = utils.get_segment_starts_and_lens(
    rf.sequence_mask(batch_dims + [enc_spatial_dim]),  # this way, every frame is interpreted as non-blank
    enc_spatial_dim,
    model,
    batch_dims,
    enc_spatial_dim
  )

  max_num_labels = rf.reduce_max(
    non_blank_targets_spatial_dim.dyn_size_ext,
    axis=batch_dims
  )
  max_num_labels = max_num_labels.raw_tensor.item()

  seq_log_prob, viterbi_alignment, viterbi_alignment_spatial_dim = model_realign(
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

  return viterbi_alignment, seq_log_prob, viterbi_alignment_spatial_dim


def model_realign(
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
  assert model.blank_idx == 0, "blank idx needs to be zero because of the way the gradient is scaled"
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

  label_lstm_out_seq, _ = model.s_wo_att(
    non_blank_input_embeddings,
    state=label_lstm_state,
    spatial_dim=non_blank_targets_padded_spatial_dim,
  )

  # ------------------------ chunk dim ------------------------

  chunk_dim = Dim(precompute_chunk_size, name="chunk")
  chunk_range = rf.expand_dim(rf.range_over_dim(chunk_dim), batch_dims[0])

  i = 0
  seq_targets = []
  seq_backrefs = []
  while i < max_seq_len.raw_tensor:
    # get current number of labels for each hypothesis
    if i > 0:
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

    # precompute attention for the current chunk (more efficient than computing it individually for each label index)
    if i % precompute_chunk_size == 0:
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
      att = model(
        enc=enc,
        enc_ctx=enc_ctx,
        enc_spatial_dim=enc_spatial_dim,
        s=label_lstm_out_seq,
        segment_starts=seg_starts,
        segment_lens=seg_lens,
      )  # [B, S+1, T, D]
      chunk_range += precompute_chunk_size

    # gather attention for the current label index
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

    logits = model.decode_logits(
      input_embed=input_embed,
      att=att_step,
      s=label_lstm_out,
    )  # [B, S+1, T, D]

    label_log_prob = rf.log_softmax(logits, axis=model.target_dim)

    # mask label log prob in order to only allow hypotheses corresponding to the ground truth:
    # log prob needs to correspond to the next non-blank label...
    log_prob_mask = vocab_range == label_ground_truth
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

    if use_recombination is not None:
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
      beam_size_ = min(
        min((i + 2), rf.reduce_max(rem_frames, axis=rem_frames.dims).raw_tensor.item()),
        min((max_num_labels + 1), beam_size)
      )
      seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
        seq_log_prob,
        k_dim=Dim(beam_size_, name=f"dec-step{i}-beam"),
        axis=[beam_dim, single_col_dim]
      )

      prev_label_indices = rf.gather(label_indices, indices=backrefs)
      # mask blank label
      update_state_mask = rf.convert_to_tensor(target != prev_label_indices)
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
