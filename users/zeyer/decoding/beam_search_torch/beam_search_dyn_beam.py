"""
Beam search
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import functools
from dataclasses import dataclass
import torch
import tree

from .interface_dyn_beam import LabelScorerDynBeamIntf
from .utils import top_k_nd, batch_gather, gather_, combine_individual_seq_scores


@dataclass
class BeamSearchDynBeamOpts:
    beam_size: int  # e.g. 12, for active hyps
    beam_and_ended_size: int  # e.g. 12 for both active+ended hyps
    length_normalization_exponent: float  # e.g. 1 to enable, 0 to disable
    bos_label: int
    eos_label: int
    num_labels: int


def beam_search_dyn_beam(
    label_scorer: LabelScorerDynBeamIntf,
    *,
    batch_size: int,
    max_seq_len: torch.Tensor,
    device: torch.device,
    opts: BeamSearchDynBeamOpts,
    out_individual_seq_scores: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Beam search with dynamic beam size and ended hypotheses kept separately.

    Based on beam_search_v5.

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

    # Initial state.
    max_act_beam_size = 1
    active_beam_sizes = torch.full([batch_size], 1, device=device)  # [Batch] (not Batch_)
    batch_idx = torch.arange(batch_size, dtype=torch.int32, device=device)  # [Batch_] -> Batch
    state = label_scorer.get_initial_state(batch_size=batch_size, device=device)  # [Batch_]
    target = torch.full([batch_size], opts.bos_label, device=device)  # [Batch_]
    seq_log_prob = torch.full([batch_size], 0.0, device=device)  # [Batch_]
    ended_seq_log_prob = torch.full([0], 0.0, device=device)  # [Batch_E]
    max_ended_beam_size = 0
    ended_beam_sizes = torch.full([batch_size], 0, device=device)  # [Batch]
    out_seq_len = torch.full([batch_size, 1], 0, device=device)

    bad_score = -1.0e30

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        seq_log_prob_ext, individual_scores, new_state = label_scorer.seq_score_ext_and_update_state(
            batch_idx=batch_idx, prev_seq_scores=seq_log_prob, prev_state=state, prev_label=target
        )
        # seq_log_prob_ext: [Batch__(InBeam),Vocab]
        # individual_scores: all tensors have [Batch__|1,Vocab__|1]
        # new_state: all tensors have [Batch__,...]

        active = torch.arange(max_act_beam_size, device=device)[None, :] < active_beam_sizes[:, None]  # [Batch,InBeam]
        prev_active = active
        prev_act_beam_sizes = active_beam_sizes
        seq_log_prob_ext_ = torch.full([batch_size, max_act_beam_size, opts.num_labels], bad_score, device=device)
        seq_log_prob_ext_.masked_scatter_(active[:, :, None], seq_log_prob_ext)  # [Batch,Max(InBeam),Vocab]
        del seq_log_prob_ext

        seq_log_prob, (backrefs, target) = top_k_nd(seq_log_prob_ext_, k=opts.beam_size, dim=[1, 2])  # all [Batch,Beam]
        # backrefs: [Batch,Beam] -> PrevBeam, should be in [0...InBeam[b]-1] for each b
        del seq_log_prob_ext_
        max_pre_act_beam_size = seq_log_prob.shape[1]

        ended = (
            torch.arange(max_ended_beam_size, device=device)[None, :] < ended_beam_sizes[:, None]
        )  # [Batch,EndedInBeam]
        ended_seq_log_prob_ = torch.full([batch_size, max_ended_beam_size], bad_score, device=device)
        ended_seq_log_prob_.masked_scatter_(ended, ended_seq_log_prob)  # [Batch,Max(EndedBeam)]
        seq_log_prob = torch.concat([seq_log_prob, ended_seq_log_prob_], dim=1)  # [Batch,Max(ActBeam)+Max(EndedBeam)]
        backrefs = torch.concat(
            [
                backrefs,
                (
                    torch.arange(max_ended_beam_size, dtype=backrefs.dtype, device=backrefs.device)[None, :]
                    + max_act_beam_size
                ).expand(batch_size, opts.beam_size + max_ended_beam_size),
            ],
            dim=1,
        )  # [Batch,Max(ActBeam)+Max(EndedBeam)]
        target = torch.concat(
            [
                target,
                torch.full([batch_size, max_ended_beam_size], opts.eos_label, dtype=target.dtype, device=target.device),
            ],
            dim=1,
        )  # [Batch,Max(ActBeam)+Max(EndedBeam)]
        ended = torch.concat(
            [
                torch.full([1, max_pre_act_beam_size], False, device=device),
                torch.full([1, max_ended_beam_size], True, device=device),
            ],
            dim=1,
        ).expand(
            batch_size, target.shape[1]
        )  # [Batch,Max(ActBeam)+Max(EndedBeam)]

        # seq_log_prob.shape[1] >= min(opts.num_labels * (opts.num_labels - 1) ** i, opts.beam_size)
        # before we concatenated ended_seq_log_prob_.
        if seq_log_prob.shape[1] > opts.beam_and_ended_size > 0:
            seq_log_prob, backrefs_ = torch.topk(
                seq_log_prob, k=opts.beam_and_ended_size, dim=1
            )  # both [Batch,OutCombBeam]
            backrefs = batch_gather(backrefs, indices=backrefs_)  # [Batch,OutCombBeam] -> PrevBeam
            target = batch_gather(target, indices=backrefs_)  # [Batch,OutCombBeam]
            ended = batch_gather(ended, indices=backrefs_)  # [Batch,OutCombBeam]

        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        out_seq_len = batch_gather(out_seq_len, indices=backrefs)  # [Batch,OutCombBeam]

        if out_individual_seq_scores is not None:
            out_individual_seq_scores.update(
                {
                    k: torch.where(ended, out_individual_seq_scores[k], v) if out_individual_seq_scores else v
                    for k, v in combine_individual_seq_scores(
                        out_individual_seq_scores, individual_scores, beam_backrefs=backrefs, labels=target
                    ).items()
                }
            )

        i += 1
        ended |= target == opts.eos_label
        ended |= (i >= max_seq_len)[:, None].to(device)  # [Batch,OutCombBeam]
        ended_or_invalid = ended | (seq_log_prob <= bad_score)  # padded area
        if ended_or_invalid.all():
            break
        active = ~ended_or_invalid  # [Batch,OutCombBeam]
        out_seq_len = out_seq_len + torch.where(ended, 0, 1)

        # First we want to split active and ended.
        batch_idx = torch.arange(batch_size, dtype=batch_idx.dtype, device=device)[:, None]  # [Batch,1]
        batch_idx = torch.masked_select(batch_idx, active)  # [Batch_]
        (seq_log_prob, ended_seq_log_prob) = (
            torch.masked_select(seq_log_prob, active),  # [Batch_]
            torch.masked_select(seq_log_prob, ended),  # [Batch_E]
        )
        target = torch.masked_select(target, active)  # [Batch_]
        active_beam_sizes = active.sum(dim=1)  # [Batch]
        max_act_beam_size = active_beam_sizes.max()  # scalar
        ended_beam_sizes = ended.sum(dim=1)  # [Batch]
        max_ended_beam_size = ended_beam_sizes.max()  # scalar

        # backrefs are [Batch,OutCombBeam] -> PrevBeam = Max(PrevActBeam).
        # But we want Batch_ -> Batch__.
        idx_ = torch.arange(prev_act_beam_sizes.sum())  # [Batch__] -> Batch__, i.e. index [0...Batch__-1]
        idx = torch.full(prev_active.shape, -1, device=device)  # [Batch,Max(PrevActBeam)]
        idx.masked_scatter_(prev_active, idx_)  # [Batch,Max(PrevActBeam)] -> Batch__
        backrefs = batch_gather(idx, indices=backrefs)  # [Batch,OutCombBeam] -> Batch__
        backrefs = torch.masked_select(backrefs, active)  # Batch_ -> Batch__

        state = tree.map_structure(functools.partial(gather_, indices=backrefs), new_state)  # [Batch_,...]

        if opts.length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            ended_seq_log_prob *= ((i + 1) / i) ** opts.length_normalization_exponent

    if opts.length_normalization_exponent != 0:
        # All seq_log_prob will be normalized by 1/(out_seq_len+1)**length_normalization_exponent.
        ended_seq_log_prob *= (1 / i) ** opts.length_normalization_exponent

    # seq_log_prob: [Batch,FinalBeam] where we break.
    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    beam_size = seq_log_prob.shape[1]
    indices = torch.arange(beam_size, device=device)[None, :].expand(batch_size, -1)  # [Batch,FinalBeam] -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: [Batch,FinalBeam] -> Beam
        # backrefs: [Batch,Beam] -> PrevBeam
        seq_targets_.insert(0, batch_gather(target, indices=indices))  # [Batch,FinalBeam]
        indices = batch_gather(backrefs, indices=indices)  # [Batch,FinalBeam] -> PrevBeam

    seq_targets = torch.stack(seq_targets_, dim=2)  # [Batch,FinalBeam,OutSeqLen]

    return seq_targets, seq_log_prob, out_seq_len
