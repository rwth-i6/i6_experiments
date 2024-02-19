"""
Beam search
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import functools
import torch
import tree

from .interface import LabelScorerIntf
from .utils import top_k_nd, batch_gather, batch_gather_, combine_individual_seq_scores
from .beam_search import BeamSearchOpts


def beam_search_v5(
    label_scorer: LabelScorerIntf,
    *,
    batch_size: int,
    max_seq_len: torch.Tensor,
    device: torch.device,
    opts: BeamSearchOpts,
    out_individual_seq_scores: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    beam search

    :param label_scorer:
    :param batch_size:
    :param max_seq_len: e.g. use encoder length. shape [Batch]
    :param device:
    :param opts:
    :param out_individual_seq_scores: if set, fills in: key -> [Batch,FinalBeam]
    :return: seq_targets, seq_log_prob, out_seq_len:
        seq_targets: [Batch,FinalBeam,OutSeqLen]
        seq_log_prob: [Batch,FinalBeam]
        out_seq_len: [Batch,FinalBeam]
    """
    # Eager-mode implementation of beam search.
    assert opts.length_reward == 0  # length reward is supposed to be done via LengthRewardScorer

    # Initial state.
    beam_size = 1
    state = label_scorer.get_initial_state(batch_size=batch_size, device=device)
    target = torch.full([batch_size, beam_size], opts.bos_label, device=device)
    ended = torch.full([batch_size, beam_size], False, device=device)
    out_seq_len = torch.full([batch_size, beam_size], 0, device=device)
    seq_log_prob = torch.full([batch_size, beam_size], 0.0, device=device)

    masked_finished_log_prob = torch.where(
        torch.arange(0, opts.num_labels, device=device) == opts.eos_label, 0.0, -1.0e30
    )  # [Vocab]

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        seq_log_prob_ext, individual_scores, new_state = label_scorer.seq_score_ext_and_update_state(
            prev_seq_scores=seq_log_prob, prev_state=state, prev_label=target
        )
        # seq_log_prob_ext: [Batch,InBeam,Vocab]
        # individual_scores: all tensors have [Batch|1,InBeam|1,Vocab|1]
        # new_state: all tensors have [Batch,InBeam,...]

        # Filter out finished beams
        seq_log_prob_ext = torch.where(
            ended[:, :, None], seq_log_prob[:, :, None] + masked_finished_log_prob[None, None, :], seq_log_prob_ext
        )
        seq_log_prob, (backrefs, target) = top_k_nd(seq_log_prob_ext, k=opts.beam_size, dim=[1, 2])  # all [Batch,Beam]
        beam_size = seq_log_prob.shape[1]
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        if out_individual_seq_scores is not None:
            out_individual_seq_scores.update(
                combine_individual_seq_scores(
                    out_individual_seq_scores, individual_scores, beam_backrefs=backrefs, labels=target
                )
            )
        state = tree.map_structure(functools.partial(batch_gather_, indices=backrefs), new_state)  # [Batch,Beam,...]
        ended = batch_gather(ended, indices=backrefs)  # [Batch,Beam]
        out_seq_len = batch_gather(out_seq_len, indices=backrefs)  # [Batch,Beam]
        i += 1

        ended = ended | (target == opts.eos_label)
        ended = ended | (i >= max_seq_len)[:, None].to(device)  # [Batch,Beam]
        if ended.all():
            break

        if opts.length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= torch.where(
                ended,
                ((i + 1) / i) ** opts.length_normalization_exponent,
                1.0,
            )

        out_seq_len = out_seq_len + torch.where(ended, 0, 1)

    if opts.length_normalization_exponent != 0:
        # All seq_log_prob will be normalized by (1/(out_seq_len+1)**length_normalization_exponent.
        seq_log_prob *= (1 / i) ** opts.length_normalization_exponent

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = torch.arange(beam_size, device=device)[None, :].expand(batch_size, -1)  # [Batch,FinalBeam] -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: [Batch,FinalBeam] -> Beam
        # backrefs: [Batch,Beam] -> PrevBeam
        seq_targets_.insert(0, batch_gather(target, indices=indices))  # [Batch,FinalBeam]
        indices = batch_gather(backrefs, indices=indices)  # [Batch,FinalBeam] -> PrevBeam

    seq_targets = torch.stack(seq_targets_, dim=2)  # [Batch,FinalBeam,OutSeqLen]

    return seq_targets, seq_log_prob, out_seq_len
