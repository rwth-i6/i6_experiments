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

    # Note: There are multiple different beams:
    # - Batch_ / Batch_E:
    #       Batch_: Packed active hyps, each active_beam_sizes, i.e. len = sum(active_beam_sizes).
    #       Batch_E: Separately, packed ended hyps, each ended_beam_sizes, len = sum(ended_beam_sizes).
    # - Batch__(InActBeam) = Batch__: refers to previous Batch_.
    # - InActBeam:
    #       Padded active hyps, unpacked from Batch_.
    # - Beam:
    #       After topk on active hyps (Max(InBeam) * Vocab).
    #       k = opts.beam_size, i.e. size = min(opts.num_labels * (opts.num_labels - 1) ** i, opts.beam_size).
    # - InEndedBeam:
    #       Padded ended hyps, unpacked from Batch_E.
    # - Beam+InEndedBeam:
    #       Both concatenated together. Padded area might occur in between.
    # - OutCombBeam:
    #       After topk on combined hyps (Beam+Max(EndedBeam)).
    #       k = opts.beam_and_ended_size.

    # Initial state.
    max_act_beam_size = 1  # size of InActBeam
    active_beam_sizes = torch.full([batch_size], 1, device=device)  # [Batch] (not Batch_)
    batch_idx_ = torch.arange(batch_size, dtype=torch.int32, device=device)  # [Batch_] -> Batch
    state_ = label_scorer.get_initial_state(batch_size=batch_size, device=device)  # [Batch_]
    target_ = torch.full([batch_size], opts.bos_label, device=device)  # [Batch_]
    seq_log_prob_ = torch.full([batch_size], 0.0, device=device)  # [Batch_]
    ended_seq_log_prob = None  # [Batch,InEndedBeam]
    ended_seq_len = None  # [Batch,InEndedBeam]
    max_ended_beam_size = 0
    ended = None
    active = torch.full([batch_size, 1], True, device=device)  # [Batch,InActBeam]

    bad_score = -1.0e30

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        seq_log_prob_ext_, individual_scores_, new_state_ = label_scorer.seq_score_ext_and_update_state(
            batch_idx=batch_idx_, prev_seq_scores=seq_log_prob_, prev_state=state_, prev_label=target_
        )
        # seq_log_prob_ext: [Batch__(InActBeam),Vocab]
        # individual_scores: all tensors have [Batch__|1,Vocab__|1]
        # new_state: all tensors have [Batch__,...]

        # Unfortunately, need to expand seq_log_prob_ext to [Batch,InActBeam,Vocab]
        # because top_k_nd (or equivalent) would not work otherwise.
        seq_log_prob_ext = torch.full([batch_size, max_act_beam_size, opts.num_labels], bad_score, device=device)
        seq_log_prob_ext.masked_scatter_(active[:, :, None], seq_log_prob_ext_)  # [Batch,InActBeam,Vocab]
        del seq_log_prob_ext_

        prev_active = active
        prev_act_beam_sizes = active_beam_sizes

        seq_log_prob, (backrefs, target) = top_k_nd(seq_log_prob_ext, k=opts.beam_size, dim=[1, 2])  # all [Batch,Beam]
        # backrefs: [Batch,Beam] -> InActBeam, should be in [0...InActBeam[b]-1] for each b
        del seq_log_prob_ext
        beam_size = seq_log_prob.shape[1]
        seq_len = torch.full([batch_size, beam_size], i, device=device)  # [Batch,Beam]

        if ended is not None:
            seq_log_prob = torch.concat([seq_log_prob, ended_seq_log_prob], dim=1)  # [Batch,Beam+InEndedBeam]
            seq_len = torch.concat([seq_len, ended_seq_len], dim=1)  # [Batch,Beam+InEndedBeam]
            backrefs = torch.concat(
                [
                    backrefs,  # [Batch,Beam] -> InActBeam
                    (
                        torch.arange(max_ended_beam_size, dtype=backrefs.dtype, device=backrefs.device)[None, :]
                        + max_act_beam_size
                    ).expand(batch_size, max_ended_beam_size),
                ],
                dim=1,
            )  # [Batch,Beam+InEndedBeam] -> InActBeam+InEndedBeam
            target = torch.concat(
                [
                    target,
                    torch.full(
                        [batch_size, max_ended_beam_size], opts.eos_label, dtype=target.dtype, device=target.device
                    ),
                ],
                dim=1,
            )  # [Batch,Beam+InEndedBeam]

        ended_comb = torch.concat(
            [
                torch.full([1, beam_size], False, device=device),
                torch.full([1, max_ended_beam_size], True, device=device),
            ],
            dim=1,
        ).expand(
            batch_size, target.shape[1]
        )  # [Batch,Beam+InEndedBeam]

        # seq_log_prob.shape[1] >= min(opts.num_labels * (opts.num_labels - 1) ** i, opts.beam_size)
        # before we concatenated ended_seq_log_prob_.
        if seq_log_prob.shape[1] > opts.beam_and_ended_size > 0:
            seq_log_prob, backrefs_2nd = torch.topk(
                seq_log_prob, k=opts.beam_and_ended_size, dim=1
            )  # both [Batch,OutCombBeam]
            # backrefs_2nd: [Batch,OutCombBeam] -> Beam+InEndedBeam
            # backrefs (before): [Batch,Beam+InEndedBeam] -> InActBeam+InEndedBeam
            backrefs = batch_gather(backrefs, indices=backrefs_2nd)  # [Batch,OutCombBeam] -> InActBeam+InEndedBeam
            target = batch_gather(target, indices=backrefs_2nd)  # [Batch,OutCombBeam]
            ended_comb = batch_gather(ended_comb, indices=backrefs_2nd)  # [Batch,OutCombBeam]
            seq_len = batch_gather(seq_len, indices=backrefs_2nd)  # [Batch,OutCombBeam]

        i += 1
        ended_comb = ended_comb | (target == opts.eos_label)
        ended_comb = ended_comb | (i >= max_seq_len)[:, None].to(device)  # [Batch,OutCombBeam]
        ended_or_invalid_comb = ended_comb | (seq_log_prob <= bad_score)  # padded area
        active_comb = ~ended_or_invalid_comb  # [Batch,OutCombBeam]

        # First we want to split active and ended.
        batch_idx_ = torch.arange(batch_size, dtype=batch_idx_.dtype, device=device)[:, None]  # [Batch,1] -> Batch
        batch_idx_ = torch.masked_select(batch_idx_, active_comb)  # [Batch_] -> Batch
        (seq_log_prob_, ended_seq_log_prob_) = (
            torch.masked_select(seq_log_prob, active_comb),  # [Batch_]
            torch.masked_select(seq_log_prob, ended_comb),  # [Batch_E]
        )
        target_ = torch.masked_select(target, active_comb)  # [Batch_]
        ended_seq_len_ = torch.masked_select(seq_len, ended_comb)  # [Batch_E]
        backrefs_active_ = torch.masked_select(backrefs, active_comb)  # [Batch_] -> InActBeam
        backrefs_ended_ = torch.masked_select(backrefs, ended_comb)  # [Batch_] -> InActBeam+InEndedBeam
        active_beam_sizes = active_comb.sum(dim=1)  # [Batch]
        max_act_beam_size = active_beam_sizes.max()  # scalar
        ended_beam_sizes = ended_comb.sum(dim=1)  # [Batch]
        max_ended_beam_size = ended_beam_sizes.max()  # scalar

        active = torch.arange(max_act_beam_size, device=device)[None, :] < active_beam_sizes[:, None]  # [Batch,ActBeam]
        ended = (
            torch.arange(max_ended_beam_size, device=device)[None, :] < ended_beam_sizes[:, None]
        )  # [Batch,EndedBeam]

        target = torch.full(active.shape, opts.eos_label, device=device)  # [Batch,ActBeam]
        target.masked_scatter_(active, target_)
        target = torch.concat(
            [
                target,
                torch.full([batch_size, max_ended_beam_size], opts.eos_label, dtype=target.dtype, device=target.device),
            ],
            dim=1,
        )  # [Batch,ActBeam+EndedBeam]
        backrefs_active = torch.full(active.shape, 0, device=device)  # [Batch,ActBeam]
        backrefs_active.masked_scatter_(active, backrefs_active_)  # [Batch,ActBeam] -> InActBeam
        backrefs_ended = torch.full(ended.shape, 0, device=device)  # [Batch,EndedBeam]
        backrefs_ended.masked_scatter_(ended, backrefs_ended_)  # [Batch,EndedBeam] -> InActBeam+InEndedBeam
        backrefs = torch.concat(
            [backrefs_active, backrefs_ended], dim=1
        )  # [Batch,ActBeam+EndedBeam] -> InActBeam+InEndedBeam
        seq_log_prob = torch.full(active.shape, bad_score, device=device)  # [Batch,ActBeam]
        seq_log_prob.masked_scatter_(active, seq_log_prob_)
        ended_seq_log_prob = torch.full(ended.shape, bad_score, device=device)  # [Batch,EndedBeam]
        ended_seq_log_prob.masked_scatter_(ended, ended_seq_log_prob_)
        seq_log_prob = torch.concat([seq_log_prob, ended_seq_log_prob], dim=1)  # [Batch,ActBeam+EndedBeam]
        seq_len = torch.full(active.shape, i, device=device)  # [Batch,ActBeam]
        ended_seq_len = torch.full(ended.shape, 0, device=device)  # [Batch,EndedBeam]
        ended_seq_len.masked_scatter_(ended, ended_seq_len_)
        seq_len = torch.concat([seq_len, ended_seq_len], dim=1)  # [Batch,ActBeam+EndedBeam]

        seq_targets.append(target)
        seq_backrefs.append(backrefs)

        # backrefs_active are [Batch,ActBeam] -> InActBeam.
        # We need Batch_ -> Batch__ (i.e. packed indices and only the active ones)
        # for transforming the state.
        idx_ = torch.arange(
            prev_act_beam_sizes.sum(), dtype=torch.int64, device=device
        )  # [Batch__] -> Batch__, i.e. index [0...Batch__-1]
        idx = torch.full(prev_active.shape, -1, dtype=torch.int64, device=device)  # [Batch,InActBeam]
        # prev_active: [Batch,InActBeam] mask, sum(prev_active) == len(Batch__)
        idx.masked_scatter_(prev_active, idx_)  # [Batch,InActBeam] -> Batch__
        backrefs_ = batch_gather(idx, indices=backrefs_active)  # [Batch,ActBeam] -> Batch__
        backrefs_ = torch.masked_select(backrefs_, active)  # Batch_ -> Batch__

        if out_individual_seq_scores is not None:
            # Similar as combine_individual_seq_scores but adapted for the packed format.
            # individual_scores: [Batch__,Vocab]
            # prev out_individual_seq_scores: [Batch,InActBeam+InEndedBeam]
            # want: out_individual_seq_scores: [Batch,ActBeam+EndedBeam]

            prev_was_active = (backrefs < prev_active.shape[1]) & (
                seq_log_prob > bad_score
            )  # [Batch,ActBeam+EndedBeam] -> active in prev
            backrefs__ = torch.where(prev_was_active, backrefs, 0)  # the prev-ended ones don't matter
            backrefs__ = batch_gather(idx, indices=backrefs__)  # [Batch,ActBeam+EndedBeam] -> Batch__
            backrefs_prev_was_active_ = torch.masked_select(backrefs__, prev_was_active)  # [Batch_PA] -> Batch__
            target_prev_was_active_ = torch.masked_select(target, prev_was_active)  # [Batch_PA] -> Vocab
            backrefs_prev_was_active__ = (
                backrefs_prev_was_active_ * idx_.shape[0] + target_prev_was_active_
            )  # [Batch_PA] -> Batch__ * Vocab + Vocab (flat indices)

            for k in list(individual_scores_.keys()):

                seq_score = individual_scores_.pop(k)  # [Batch__|1,Vocab|1]
                if (
                    seq_score.shape[0] == 1 < idx_.shape[0] and seq_score.shape[1] == opts.num_labels
                ):  # [Batch__=1,Vocab]
                    raise NotImplementedError(f"seq_score shape {seq_score.shape}, bc Batch__")
                elif (
                    seq_score.shape[0] == idx_.shape[0] and seq_score.shape[1] == 1 < opts.num_labels
                ):  # [Batch__,Vocab=1]
                    raise NotImplementedError(f"seq_score shape {seq_score.shape}, bc Vocab")
                elif seq_score.shape[0] == idx_.shape[0] and seq_score.shape[1] == opts.num_labels:  # [Batch__,Vocab]
                    seq_score_ = seq_score.flatten()[backrefs_prev_was_active__]  # [Batch_PA]
                    seq_score = torch.full(target.shape, 0.0, device=device)  # [Batch,ActBeam+EndedBeam]
                    seq_score.masked_scatter_(prev_was_active, seq_score_)
                else:
                    assert seq_score.shape == (1, 1)
                    seq_score = seq_score.expand(*target.shape)

                if k in out_individual_seq_scores:
                    prev_seq_score = out_individual_seq_scores[k]
                    # prev_seq_score: [1,1] or [Batch,InActBeam+InEndedBeam]
                    if prev_seq_score.shape[1] > 1:
                        prev_seq_score = batch_gather(prev_seq_score, indices=backrefs)
                    # prev_seq_score: [Batch,ActBeam+EndedBeam]
                    seq_score = seq_score + prev_seq_score

                seq_score.masked_fill_(seq_log_prob <= bad_score, bad_score)
                out_individual_seq_scores[k] = seq_score

        if ended_or_invalid_comb.all():
            break

        state_ = tree.map_structure(functools.partial(gather_, indices=backrefs_), new_state_)  # [Batch_,...]

        if opts.length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            ended_seq_log_prob_ *= ((i + 1) / i) ** opts.length_normalization_exponent

    if opts.length_normalization_exponent != 0:
        # All seq_log_prob will be normalized by 1/(out_seq_len+1)**length_normalization_exponent.
        seq_log_prob *= (1 / i) ** opts.length_normalization_exponent

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

    return seq_targets, seq_log_prob, seq_len
