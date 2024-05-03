from typing import Tuple, List
import functools
import tree
import torch
from dataclasses import dataclass
import dataclasses

from returnn.tensor import Tensor

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.label_scorer import LabelScorerIntf, StateObjTensorExt
from .utils import top_k_nd, batch_gather, batch_gather_


@dataclass
class BeamSearchOpts:
  beam_size: int  # e.g. 12
  bos_label: int
  eos_label: int
  num_labels: int

  length_normalization_exponent: float = 0.0  # e.g. 1 to enable, 0 to disable


def time_sync_beam_search(
        label_scorer: LabelScorerIntf,
        *,
        label_sync_keys: List[str],
        time_sync_keys: List[str],
        batch_size: int,
        blank_idx: int,
        max_seq_len: torch.Tensor,
        device: torch.device,
        opts: BeamSearchOpts,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
  bad_score = -1.0e30
  max_seq_len = max_seq_len.to("cpu")
  max_seq_len = max_seq_len.max().item()

  # Initial state.
  beam_size = 1
  state = label_scorer.get_initial_state(batch_size=batch_size, device=device)
  # i here use the index -1 to indicate the initial state
  # this way, i implement an initial embedding of zeros by checking for -1 in the label scorers
  # this special handling needs to be implemented for every scorer because otherwise it will lead to index out of bounds
  # if we directly pass the -1 to an embedding layer
  # TODO: implement this in a nicer way?
  target = torch.full([batch_size, beam_size], -1, device=device)
  target_non_blank = torch.full([batch_size, beam_size], -1, device=device)
  seq_log_prob = torch.full([batch_size, beam_size], 0.0, device=device)

  t = 0
  seq_targets = []
  seq_backrefs = []
  while t < max_seq_len:
    seq_log_prob_ext, individual_scores, new_state = label_scorer.seq_score_ext_and_update_state(
      prev_seq_scores=seq_log_prob, prev_state=state, prev_label=target_non_blank, prev_align_label=target, t=t
    )

    # TODO: del state?
    # del state
    # seq_log_prob_ext: [Batch,InBeam,Vocab]
    # individual_scores: all tensors have [Batch|1,InBeam|1,Vocab|1]
    # new_state: all tensors have [Batch,InBeam,...]

    seq_log_prob, (backrefs, target) = top_k_nd(seq_log_prob_ext, k=opts.beam_size, dim=[1, 2])  # all [Batch,Beam]

    update_state_mask = target != blank_idx

    def _get_masked_state(old, new, mask):
      old = batch_gather_(old, indices=backrefs)
      new = batch_gather_(new, indices=backrefs)

      if isinstance(old, Tensor):
        assert isinstance(new, Tensor)
        assert mask.shape[:2] == old.shape[:2]
        mask_reshaped = torch.reshape(mask, mask.shape + (1,) * (old.ndim - mask.ndim))
        return torch.where(mask_reshaped, new, old)
      elif isinstance(old, StateObjTensorExt):
        assert isinstance(new, StateObjTensorExt)
        assert mask.shape[:2] == old.tensor.shape[:2]
        mask_reshaped = torch.reshape(mask, mask.shape + (1,) * (old.tensor.ndim - mask.ndim))
        return dataclasses.replace(new, tensor=torch.where(mask_reshaped, new.tensor, old.tensor))
      else:
       return new

    del seq_log_prob_ext
    beam_size = seq_log_prob.shape[1]
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    for key in time_sync_keys:
      state[key] = tree.map_structure(functools.partial(batch_gather_, indices=backrefs), new_state[key])  # [Batch,Beam,...]
    for key in label_sync_keys:
      state[key] = tree.map_structure(functools.partial(_get_masked_state, mask=update_state_mask), state[key], new_state[key])  # [Batch,Beam,...]
    target_non_blank = torch.where(update_state_mask, target, batch_gather_(target_non_blank, indices=backrefs))

    del new_state
    t += 1

  # Backtrack via backrefs, resolve beams.
  seq_targets_ = []
  indices = torch.arange(beam_size, device=device)[None, :].expand(batch_size, -1)  # [Batch,FinalBeam] -> FinalBeam
  for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
    # indices: [Batch,FinalBeam] -> Beam
    # backrefs: [Batch,Beam] -> PrevBeam
    seq_targets_.insert(0, batch_gather(target, indices=indices))  # [Batch,FinalBeam]
    indices = batch_gather(backrefs, indices=indices)  # [Batch,FinalBeam] -> PrevBeam

  seq_targets = torch.stack(seq_targets_, dim=2)  # [Batch,FinalBeam,OutSeqLen]

  return seq_targets, seq_log_prob
