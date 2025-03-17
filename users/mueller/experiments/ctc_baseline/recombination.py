from typing import Optional, Dict, Any, Tuple, Sequence
import tree
import warnings
import numpy as np
import torch

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


def recombine_seqs(
        seq_log_prob: Tensor,
        seq_hash: Tensor,
        beam_dim: Dim,
        batch_dim: Dim,
        vocab_dim: Dim,
        blank_idx: int,
        recomb_blank: bool,
        use_sum: bool = True,
        is_blank: Optional[Tensor] = None,
) -> Tensor:
  has_vocab = vocab_dim is not None
  if has_vocab:
    seq_hash_cpu = rf.copy_to_device(seq_hash.copy_transpose([batch_dim, beam_dim, vocab_dim]), device="cpu")
    seq_log_prob_cpu = rf.copy_to_device(seq_log_prob.copy_transpose([batch_dim, beam_dim, vocab_dim]), device="cpu")
  else:
    seq_hash_cpu = rf.copy_to_device(seq_hash.copy_transpose([batch_dim, beam_dim]), device="cpu")
    # convert from neg log prob to log prob
    seq_log_prob_cpu = rf.copy_to_device(seq_log_prob.copy_transpose([batch_dim, beam_dim]), device="cpu")
    assert is_blank is not None
    is_blank_cpu = rf.copy_to_device(is_blank.copy_transpose([batch_dim, beam_dim]), device="cpu")
    
  for b in range(batch_dim.dyn_size_ext.raw_tensor.item()):
    # for each batch dim, we need to find the seqs that have the same hash value
    seq_sets = {}
    for h in range(beam_dim.dimension):
      if has_vocab:
        for v in range(vocab_dim.dimension):
          # hash value of current hypothesis
          seq_hash_value = seq_hash_cpu.raw_tensor[b, h, v].item()
          if v == blank_idx and not recomb_blank:
            seq_hash_value = "blank_" + str(seq_hash_value)
          if seq_hash_value not in seq_sets:
            seq_sets[seq_hash_value] = []
          # insert hypothesis index into the list of hypotheses with the same hash value
          seq_sets[seq_hash_value].append((h, v))
      else:
        # hash value of current hypothesis
        seq_hash_value = seq_hash_cpu.raw_tensor[b, h].item()
        if is_blank_cpu.raw_tensor[b, h] and not recomb_blank:
          seq_hash_value = "blank_" + str(seq_hash_value)
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
      if has_vocab:
        for idx_h, idx_v in seq_set:
          if seq_log_prob_cpu.raw_tensor[b, idx_h, idx_v] > best_score:
            best_score = seq_log_prob_cpu.raw_tensor[b, idx_h, idx_v]
            best_idx = (idx_h, idx_v)
      else:
        for idx in seq_set:
          if seq_log_prob_cpu.raw_tensor[b, idx] > best_score:
            best_score = seq_log_prob_cpu.raw_tensor[b, idx]
            best_idx = idx
      
      if use_sum:
        sum_score = torch.zeros(1, device="cpu")
        # calculate log of sum of probs via log-sum-exp trick
        if has_vocab:
          for idx_h, idx_v in seq_set:
            sum_score += torch.exp(seq_log_prob_cpu.raw_tensor[b, idx_h, idx_v] - best_score)
        else:
          for idx in seq_set:
            sum_score += torch.exp(seq_log_prob_cpu.raw_tensor[b, idx] - best_score)
        recomb_score = torch.log(sum_score) + best_score
      else:
        recomb_score = best_score

      if has_vocab:
        for idx_h, idx_v in seq_set:
          if (idx_h, idx_v) != best_idx:
            seq_log_prob_cpu.raw_tensor[b, idx_h, idx_v] = -1.0e30
          else:
            seq_log_prob_cpu.raw_tensor[b, idx_h, idx_v] = recomb_score
      else:
        for idx in seq_set:
          if idx != best_idx:
            seq_log_prob_cpu.raw_tensor[b, idx] = -1.0e30
          else:
            seq_log_prob_cpu.raw_tensor[b, idx] = recomb_score
      
  return rf.copy_to_device(seq_log_prob_cpu)

def update_seq_hash(seq_hash: Tensor, new_target: Tensor, backrefs: Tensor, old_target: Tensor, blank_idx: int, gather_old_target: bool = True) -> Tensor:
  if backrefs is not None and old_target is not None:
    old_seq_hash = rf.gather(seq_hash, indices=backrefs)
    if gather_old_target:
      old_seq_hash = rf.gather(old_seq_hash, indices=old_target)
  else:
    old_seq_hash = seq_hash
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    seq_hash = rf.where(
      (new_target == blank_idx) | (new_target == old_target),
      old_seq_hash,
      (old_seq_hash * 257 + (new_target + 1)) % (10 ** 9 + 7)
    )
  return seq_hash