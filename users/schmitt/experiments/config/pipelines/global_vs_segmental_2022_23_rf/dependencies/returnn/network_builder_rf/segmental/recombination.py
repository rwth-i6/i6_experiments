from typing import Optional, Dict, Any, Tuple, Sequence
import tree
import numpy as np
import torch

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


def recombine_seqs(
        seq_targets: list,
        seq_log_prob: Tensor,
        seq_hash: Tensor,
        beam_dim: Dim,
        batch_dim: Dim,
        use_sum: bool = True,
) -> Tensor:
  if len(seq_targets) in (0, 1):
    return seq_log_prob

  seq_hash_cpu = rf.copy_to_device(seq_hash.copy_transpose([batch_dim, beam_dim]), device="cpu")
  # convert from neg log prob to log prob
  seq_log_prob_cpu = rf.copy_to_device(seq_log_prob.copy_transpose([batch_dim, beam_dim]), device="cpu")

  for b in range(batch_dim.dyn_size_ext.raw_tensor.item()):
    # for each batch dim, we need to find the seqs that have the same hash value
    seq_sets = {}
    for h in range(beam_dim.dimension):
      # hash value of current hypothesis
      seq_hash_value = seq_hash_cpu.raw_tensor[b, h].item()
      if seq_hash_value not in seq_sets:
        seq_sets[seq_hash_value] = []
      # insert hypothesis index into the list of hypotheses with the same hash value
      seq_sets[seq_hash_value].append(h)
    # for each set of hypotheses with the same hash value, we keep the one with the highest log prob
    for seq_set in seq_sets.values():
      # skip if there is only one hypothesis in the set
      if len(seq_set) == 1:
        continue
      best_score = -1.0e30
      best_idx = -1
      # find the hypothesis with the highest log prob
      for idx in seq_set:
        if seq_log_prob_cpu.raw_tensor[b, idx] > best_score:
          best_score = seq_log_prob_cpu.raw_tensor[b, idx]
          best_idx = idx

      if use_sum:
        sum_score = torch.zeros(1, device="cpu")
        # calculate log of sum of probs via log-sum-exp trick
        for idx in seq_set:
          sum_score += torch.exp(seq_log_prob_cpu.raw_tensor[b, idx] - best_score)
        recomb_score = torch.log(sum_score) + best_score
      else:
        recomb_score = best_score

      for idx in seq_set:
        if idx != best_idx:
          seq_log_prob_cpu.raw_tensor[b, idx] = -1.0e30
        else:
          seq_log_prob_cpu.raw_tensor[b, idx] = recomb_score

  return rf.copy_to_device(seq_log_prob_cpu, device=seq_log_prob.device)


def recombine_seqs_train(
        seq_log_prob: Tensor,
        label_log_prob: Tensor,
        label_indices: Tensor,
        ground_truth: Tensor,
        target_dim: Dim,
        single_col_dim: Dim,
        beam_dim: Dim,
        batch_dims: Sequence[Dim],
        blank_idx: int,
        use_sum_recombination: bool = True,
) -> Tensor:
  # local horizontal scores for each hyp: [B, beam]
  horizontal_scores = rf.gather(
    label_log_prob,
    indices=rf.constant(blank_idx, dtype="int32", dims=batch_dims),
    axis=target_dim
  )

  # combined horizontal scores for each hyp: [B, beam]
  horizontal_scores = seq_log_prob + horizontal_scores

  # if a hypothesis has score < -1.0e30, it means that it got recombined with another hypothesis in the previous step
  # in this case, it has the same label index as the hypothesis it got combined with and therefore, the scatter
  # below would result in a horizontal score < -1.0e30, which is wrong.
  # therefore, set the label index to S+2 and cut off this score afterwards
  label_indices_ext = rf.where(
    seq_log_prob <= -1.0e30,
    single_col_dim.dimension,
    label_indices,
  )
  single_col_dim_ext = single_col_dim + 1
  label_indices_ext.sparse_dim = single_col_dim_ext

  # lattice column with horizontal scores: [B, S+2]
  # horizontal -> label index stays the same
  horizontal_scores = rf.scatter(
    horizontal_scores,
    indices=label_indices_ext,
    indices_dim=beam_dim,
  )
  horizontal_scores = horizontal_scores.copy_transpose(batch_dims + [single_col_dim_ext])
  # cut off the last row as mentioned above
  horizontal_scores_raw = horizontal_scores.raw_tensor[:, :-1]
  # lattice column with horizontal scores: [B, S+1]
  horizontal_scores = horizontal_scores.copy_template_replace_dim_tag(
    horizontal_scores.get_axis_from_description(single_col_dim_ext),
    single_col_dim
  )
  horizontal_scores.raw_tensor = horizontal_scores_raw

  # lattice column with horizontal scores for each hyp: [B, beam, S+1]
  # each hyp only has one node with score > -1.0e30, which is the node corresponding to the horizontal transition
  horizontal_scores = rf.where(
    rf.range_over_dim(single_col_dim) == label_indices_ext,
    horizontal_scores,
    rf.constant(-1.0e30, dims=batch_dims)
  )

  # local diagonal scores for each hyp: [B, beam]
  diagonal_scores = rf.gather(
    label_log_prob,
    indices=ground_truth,
    axis=target_dim
  )

  # if the ground truth is blank, it means the hypothesis is already at the top-most row of the lattice
  # in this case, set the diagonal score to -1.0e30
  diagonal_scores = rf.where(
    ground_truth == blank_idx,
    rf.constant(-1.0e30, dims=batch_dims),
    diagonal_scores
  )
  # combined diagonal scores for each hyp: [B, beam]
  diagonal_scores = seq_log_prob + diagonal_scores

  # the updated label indices after diagonal transition: [B, beam]
  label_indices_updated = label_indices + 1


  # same as with horizontal_scores, see above
  label_indices_updated_ext = rf.where(
    seq_log_prob <= -1.0e30,
    single_col_dim.dimension,
    label_indices_updated,
  )
  label_indices_updated_ext.sparse_dim = single_col_dim_ext

  # lattice column with diagonal scores: [B, S+2]
  # diagonal -> label index is updated
  diagonal_scores = rf.scatter(
    diagonal_scores,
    indices=label_indices_updated_ext,
    indices_dim=beam_dim,
  )
  diagonal_scores = diagonal_scores.copy_transpose(batch_dims + [single_col_dim_ext])
  # cut off last row, see above
  diagonal_scores_raw = diagonal_scores.raw_tensor[:, :-1]
  # lattice column with diagonal scores: [B, S+1]
  diagonal_scores = diagonal_scores.copy_template_replace_dim_tag(
    diagonal_scores.get_axis_from_description(single_col_dim_ext),
    single_col_dim
  )
  diagonal_scores.raw_tensor = diagonal_scores_raw

  # lattice column with diagonal scores for each hyp: [B, beam, S+1]
  # each hyp only has one node with score > -1.0e30, which is the node corresponding to the diagonal transition
  diagonal_scores = rf.where(
    rf.range_over_dim(single_col_dim) == label_indices_updated_ext,
    diagonal_scores,
    rf.constant(-1.0e30, dims=batch_dims)
  )

  # for each hyp, merge horizontal and diagonal scores into one column
  # i.e. [-inf, ..., d, ..., -inf] + [-inf, ..., h, ..., -inf] -> [-inf, ..., d, h, ..., -inf]
  best_scores = rf.maximum(horizontal_scores, diagonal_scores)
  merged_scores = rf.exp(horizontal_scores - best_scores) + rf.exp(diagonal_scores - best_scores)
  merged_scores = best_scores + rf.safe_log(merged_scores)

  # recombine the scores of different hypotheses
  best_scores = rf.reduce_max(merged_scores, axis=beam_dim)
  is_max = merged_scores == best_scores

  if use_sum_recombination:
    recombined_score = best_scores + rf.log(rf.reduce_sum(rf.exp(merged_scores - best_scores), axis=beam_dim))
  else:
    recombined_score = best_scores

  # add back the beam dimension and set the score of the worse hypotheses to -1.0e30
  recombined_score = rf.expand_dim(recombined_score, beam_dim)
  recombined_score = rf.where(
    is_max,
    recombined_score,
    rf.constant(-1.0e30, dims=batch_dims)
  )

  return recombined_score


def update_seq_hash(seq_hash: Tensor, target: Tensor, backrefs: Tensor, blank_idx: int) -> Tensor:
  old_seq_hash = rf.gather(seq_hash, indices=backrefs)
  seq_hash = rf.where(
    target == blank_idx,
    old_seq_hash,
    (old_seq_hash * 257 + (target + 1)) % (10 ** 9 + 7)
  )
  return seq_hash
