"""
Beam search
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import functools
import torch
import tree

from .interface_dyn_beam import LabelScorerDynBeamIntf
from .utils import top_k_nd, batch_gather, gather_, masked_select, nonzero
from .beam_search_dyn_beam import BeamSearchDynBeamOpts


def beam_search_dyn_beam_v2(
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

    Based on beam_search_dyn_beam (v1).

    Via torch.cuda.set_sync_debug_mode, found out:
    masked_select is a synchronizing CUDA operation.
    We do a lot of them, which probably makes it slow.

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
    #       Batch_: Packed active hyps, each act_beam_sizes, i.e. len = sum(act_beam_sizes).
    #       Batch_E: Separately, packed ended hyps, each end_beam_sizes, len = sum(end_beam_sizes).
    # - Batch__(InActBeam) = Batch__: refers to previous Batch_.
    # - InActBeam:
    #       Padded active hyps, unpacked from Batch_.
    # - Beam:
    #       After topk on active hyps (Max(InBeam) * Vocab).
    #       k = opts.beam_size, i.e. size = min(opts.num_labels * (opts.num_labels - 1) ** i, opts.beam_size).
    # - InEndBeam:
    #       Padded ended hyps, unpacked from Batch_E.
    # - Beam+InEndBeam:
    #       Both concatenated together. Padded area might occur in between.
    # - OutCombBeam:
    #       After topk on combined hyps (Beam+Max(EndBeam)).
    #       k = opts.beam_and_ended_size.

    bad_score = -1.0e30
    max_seq_len = max_seq_len.to(device)
    length_normalization_exponent_dev = torch.full((), opts.length_normalization_exponent, device=device)

    # Initial state.
    max_act_beam_size = 1  # size of InActBeam
    sum_act_beam_sizes = batch_size
    active = None  # [Batch,InActBeam]
    batch_idx_ = torch.arange(batch_size, dtype=torch.int32, device=device)  # [Batch_] -> Batch
    state_ = label_scorer.get_initial_state(batch_size=batch_size, device=device)  # [Batch_]
    target_ = torch.full([batch_size], opts.bos_label, device=device)  # [Batch_]
    seq_log_prob_ = torch.full([batch_size], 0.0, device=device)  # [Batch_]
    end_seq_log_prob = None  # [Batch,InEndBeam]
    end_seq_len = None  # [Batch,InEndBeam]
    max_end_beam_size = 0

    i = 0
    i_dev = torch.zeros((), dtype=torch.int64, device=device)
    seq_targets = []
    seq_backrefs = []
    while True:
        seq_log_prob_ext_, individual_scores_, new_state_ = label_scorer.seq_score_ext_and_update_state(
            batch_idx=batch_idx_, prev_seq_scores=seq_log_prob_, prev_state=state_, prev_label=target_
        )
        # seq_log_prob_ext: [Batch__(InActBeam),Vocab]
        # individual_scores: all tensors have [Batch__|1,Vocab__|1]
        # new_state: all tensors have [Batch__,...]

        if active is None:
            seq_log_prob_ext = seq_log_prob_ext_.unflatten(
                0, (batch_size, max_act_beam_size)
            )  # [Batch,InActBeam,Vocab]
        else:
            # Unfortunately, need to expand seq_log_prob_ext to [Batch,InActBeam,Vocab]
            # because top_k_nd (or equivalent) would not work otherwise.
            seq_log_prob_ext = torch.full([batch_size, max_act_beam_size, opts.num_labels], bad_score, device=device)
            seq_log_prob_ext.masked_scatter_(active[:, :, None], seq_log_prob_ext_)  # [Batch,InActBeam,Vocab]
        del seq_log_prob_ext_

        prev_active = active
        prev_sum_act_beam_sizes = sum_act_beam_sizes
        prev_max_act_beam_size = max_act_beam_size

        seq_log_prob, (backrefs, target) = top_k_nd(seq_log_prob_ext, k=opts.beam_size, dim=[1, 2])  # all [Batch,Beam]
        # backrefs: [Batch,Beam] -> InActBeam, should be in [0...InActBeam[b]-1] for each b
        del seq_log_prob_ext
        beam_size = seq_log_prob.shape[1]
        seq_len = torch.full([batch_size, beam_size], i, device=device)  # [Batch,Beam]

        if max_end_beam_size > 0:
            seq_log_prob = torch.concat([seq_log_prob, end_seq_log_prob], dim=1)  # [Batch,Beam+InEndBeam]
            seq_len = torch.concat([seq_len, end_seq_len], dim=1)  # [Batch,Beam+InEndBeam]
            backrefs = torch.concat(
                [
                    backrefs,  # [Batch,Beam] -> InActBeam
                    (
                        torch.arange(max_end_beam_size, dtype=backrefs.dtype, device=backrefs.device)[None, :]
                        + max_act_beam_size
                    ).expand(batch_size, max_end_beam_size),
                ],
                dim=1,
            )  # [Batch,Beam+InEndedBeam] -> InActBeam+InEndBeam
            target = torch.concat(
                [
                    target,
                    torch.full(
                        [batch_size, max_end_beam_size], opts.eos_label, dtype=target.dtype, device=target.device
                    ),
                ],
                dim=1,
            )  # [Batch,Beam+InEndBeam]

            end_comb = torch.concat(
                [
                    torch.full([1, beam_size], False, device=device),
                    torch.full([1, max_end_beam_size], True, device=device),
                ],
                dim=1,
            ).expand(
                *target.shape
            )  # [Batch,Beam+InEndBeam]

        else:
            end_comb = None

        # seq_log_prob.shape[1] >= min(opts.num_labels * (opts.num_labels - 1) ** i, opts.beam_size)
        # before we concatenated ended_seq_log_prob.
        if seq_log_prob.shape[1] > opts.beam_and_ended_size > 0:
            seq_log_prob, backrefs_2nd = torch.topk(
                seq_log_prob, k=opts.beam_and_ended_size, dim=1
            )  # both [Batch,OutCombBeam]
            # backrefs_2nd: [Batch,OutCombBeam] -> Beam+InEndBeam
            # backrefs (before): [Batch,Beam+InEndedBeam] -> InActBeam+InEndBeam
            backrefs = batch_gather(backrefs, indices=backrefs_2nd)  # [Batch,OutCombBeam] -> InActBeam+InEndBeam
            target = batch_gather(target, indices=backrefs_2nd)  # [Batch,OutCombBeam]
            if end_comb is not None:
                end_comb = batch_gather(end_comb, indices=backrefs_2nd)  # [Batch,OutCombBeam]
            seq_len = batch_gather(seq_len, indices=backrefs_2nd)  # [Batch,OutCombBeam]

        i += 1
        i_dev += 1

        # Make mapping 0->active, 1->ended, 100->invalid.
        state_comb = torch.full(target.shape, 0, dtype=torch.int8, device=device)  # [Batch,OutCombBeam]
        if end_comb is not None:
            torch.where(end_comb, torch.full((), 1, device=device), state_comb, out=state_comb)
        torch.where(target == opts.eos_label, torch.full((), 1, device=device), state_comb, out=state_comb)
        torch.where((i_dev >= max_seq_len)[:, None], torch.full((), 1, device=device), state_comb, out=state_comb)
        torch.where(seq_log_prob <= bad_score, torch.full((), 100, device=device), state_comb, out=state_comb)

        act_comb = state_comb == 0  # [Batch,OutCombBeam]
        act_beam_sizes = act_comb.sum(dim=1)  # [Batch]
        end_comb = state_comb == 1  # [Batch,OutCombBeam]
        end_beam_sizes = end_comb.sum(dim=1)  # [Batch]

        state_comb_cpu = state_comb.cpu()  # single CUDA sync point
        act_beam_sizes_cpu = (state_comb_cpu == 0).sum(dim=1)  # [Batch]
        max_act_beam_size = act_beam_sizes_cpu.max()  # scalar
        end_beam_sizes_cpu = (state_comb_cpu == 1).sum(dim=1)  # [Batch]
        max_end_beam_size = end_beam_sizes_cpu.max()  # scalar
        sum_act_beam_sizes = act_beam_sizes_cpu.sum()  # scalar
        full_act_beam = sum_act_beam_sizes == max_act_beam_size * act_beam_sizes_cpu.shape[0]

        if full_act_beam and act_comb.shape[1] == max_act_beam_size and max_end_beam_size == 0:
            act_idx = act_end_idx = act_comb_idx = None  # Not needed.
        else:
            act_idx = torch.argsort(state_comb, dim=1, stable=True)[
                :, :max_act_beam_size
            ]  # [Batch,ActBeam] -> OutCombBeam
            if max_end_beam_size > 0:
                torch.where(act_comb, torch.full((), 2, device=device), state_comb, out=state_comb)  # remap 2->active
                end_idx = torch.argsort(state_comb, dim=1, stable=True)[
                    :, :max_end_beam_size
                ]  # [Batch,EndBeam] -> OutCombBeam
                act_end_idx = torch.concat([act_idx, end_idx], dim=1)  # [Batch,ActBeam+EndBeam] -> OutCombBeam
            else:
                act_end_idx = act_idx

            # Do not use torch.masked_select in the following to avoid CUDA synchronization.
            # Also do not use torch.nonzero to avoid CUDA synchronization.
            act_comb_idx = nonzero(act_comb.flatten(), out_len=sum_act_beam_sizes)  # [Batch_] -> Batch*OutCombBeam

        # First we want to split active and ended.
        batch_idx_ = torch.arange(batch_size, dtype=batch_idx_.dtype, device=device)[:, None].expand(
            *target.shape
        )  # [Batch,OutCombBeam] -> Batch
        if act_end_idx is None:
            batch_idx_ = batch_idx_.flatten()  # [Batch_] -> Batch
            seq_log_prob_ = seq_log_prob.flatten()  # [Batch_]
            target_ = target.flatten()  # [Batch_]
        else:
            batch_idx_ = batch_idx_.flatten()[act_comb_idx]  # [Batch_] -> Batch
            seq_log_prob_ = seq_log_prob.flatten()[act_comb_idx]  # [Batch_]
            target_ = target.flatten()[act_comb_idx]  # [Batch_]

        if full_act_beam and max_end_beam_size == 0:
            active = valid = None  # not needed
        else:
            active = (
                torch.arange(max_act_beam_size, device=device)[None, :] < act_beam_sizes[:, None]
            )  # [Batch,ActBeam]
            if max_end_beam_size > 0:
                ended = (
                    torch.arange(max_end_beam_size, device=device)[None, :] < end_beam_sizes[:, None]
                )  # [Batch,EndBeam]
                valid = torch.concat([active, ended], dim=1)  # [Batch,ActBeam+EndBeam]
            else:
                valid = active

        if act_idx is not None:
            target = batch_gather(target, indices=act_idx)  # [Batch,ActBeam]
            if active is not None:
                target = torch.where(
                    active, target, torch.full((), opts.eos_label, dtype=target.dtype, device=device)
                )  # [Batch,ActBeam]
            if max_end_beam_size > 0:
                target = torch.concat(
                    [
                        target,
                        torch.full([batch_size, max_end_beam_size], opts.eos_label, dtype=target.dtype, device=device),
                    ],
                    dim=1,
                )  # [Batch,ActBeam+EndedBeam]
            backrefs = batch_gather(backrefs, indices=act_end_idx)  # [Batch,ActBeam+EndedBeam] -> InActBeam+InEndedBeam
            torch.where(valid, backrefs, torch.zeros((), dtype=backrefs.dtype, device=device), out=backrefs)
            seq_log_prob = batch_gather(seq_log_prob, indices=act_end_idx)  # [Batch,ActBeam+EndBeam]
            torch.where(valid, seq_log_prob, torch.full((), bad_score, device=device), out=seq_log_prob)
            seq_len = batch_gather(seq_len, indices=act_end_idx)  # [Batch,ActBeam+EndBeam]
            torch.where(valid, seq_len, torch.zeros((), dtype=seq_len.dtype, device=device), out=seq_len)
        end_seq_log_prob = seq_log_prob[:, max_act_beam_size:]  # [Batch,EndBeam]
        end_seq_len = seq_len[:, max_act_beam_size:]  # [Batch,EndBeam]

        seq_targets.append(target)
        seq_backrefs.append(backrefs)

        # backrefs are [Batch,ActBeam+EndBeam] -> InActBeam+InEndBeam.
        backrefs_ = backrefs[:, :max_act_beam_size]  # [Batch,ActBeam] -> InActBeam+InEndBeam
        backrefs_ = torch.clip(backrefs_, 0, prev_max_act_beam_size - 1)  # pad out-of-range, only -> InActBeam
        # We need Batch_ -> Batch__ (i.e. packed indices and only the active ones)
        # for transforming the state.
        if prev_active is None:
            backrefs_ += torch.arange(batch_size)[:, None] * prev_max_act_beam_size  # [Batch,ActBeam] -> Batch__
            idx = None
        else:
            idx_ = torch.arange(prev_sum_act_beam_sizes, dtype=torch.int64, device=device)  # [Batch__] -> Batch__
            idx = torch.full(prev_active.shape, -1, dtype=torch.int64, device=device)  # [Batch,InActBeam]
            # prev_active: [Batch,InActBeam] mask, sum(prev_active) == len(Batch__)
            idx.masked_scatter_(prev_active, idx_)  # [Batch,InActBeam] -> Batch__
            backrefs_ = batch_gather(idx, indices=backrefs_)  # [Batch,ActBeam] -> Batch__
        if active is None:
            backrefs_ = backrefs_.flatten()
        else:
            backrefs_ = masked_select(backrefs_, active, out_len=sum_act_beam_sizes)  # Batch_ -> Batch__

        if out_individual_seq_scores is not None:
            # Similar as combine_individual_seq_scores but adapted for the packed format.
            # individual_scores: [Batch__,Vocab]
            # prev out_individual_seq_scores: [Batch,InActBeam+InEndedBeam]
            # want: out_individual_seq_scores: [Batch,ActBeam+EndedBeam]

            prev_was_active = valid & (backrefs < prev_max_act_beam_size)  # [Batch,ActBeam+EndBeam]
            backrefs__ = torch.clip(backrefs, 0, prev_max_act_beam_size - 1)
            if prev_active is None:
                backrefs__ += (
                    torch.arange(batch_size)[:, None] * prev_max_act_beam_size
                )  # [Batch,ActBeam+EndBeam] -> Batch__
            else:
                backrefs__ = batch_gather(idx, indices=backrefs__)  # [Batch,ActBeam+EndBeam] -> Batch__
            backrefs_flat = (
                backrefs__ * opts.num_labels + target
            )  # [Batch,ActBeam+EndedBeam] -> Batch__ * Vocab + Vocab (flat indices)

            for k in list(individual_scores_.keys()):

                seq_score = individual_scores_.pop(k)  # [Batch__|1,Vocab|1]
                if seq_score.shape == (1, 1):  # [Batch__=1,Vocab=1]
                    pass  # leave it, but it will be interpreted as [Batch=1,(ActBeam+EndedBeam)=1]
                elif (
                    seq_score.shape[0] == 1 < idx_.shape[0] and seq_score.shape[1] == opts.num_labels
                ):  # [Batch__=1,Vocab]
                    raise NotImplementedError(
                        f"seq_score shape {seq_score.shape}, bc Batch__,"
                        f" batch__ {idx_.shape[0]}, target shape {target.shape}, vocab {opts.num_labels}"
                    )
                elif (
                    seq_score.shape[0] == idx_.shape[0] and seq_score.shape[1] == 1 < opts.num_labels
                ):  # [Batch__,Vocab=1]
                    raise NotImplementedError(
                        f"seq_score shape {seq_score.shape}, bc Vocab,"
                        f" batch__ {idx_.shape[0]}, target shape {target.shape}, vocab {opts.num_labels}"
                    )
                elif seq_score.shape[0] == idx_.shape[0] and seq_score.shape[1] == opts.num_labels:  # [Batch__,Vocab]
                    seq_score = seq_score.flatten()[backrefs_flat]  # [Batch,ActBeam+EndedBeam]
                else:
                    raise RuntimeError(
                        f"did not expect seq_score shape {seq_score.shape},"
                        f" batch__ {idx_.shape[0]}, target shape {target.shape}, vocab {opts.num_labels}"
                    )
                seq_score = torch.where(
                    prev_was_active, seq_score, torch.zeros((), device=device)
                )  # [Batch,ActBeam+EndBeam]

                if k in out_individual_seq_scores:
                    prev_seq_score = out_individual_seq_scores[k]
                    # prev_seq_score: [1,1] or [Batch,InActBeam+InEndedBeam]
                    if prev_seq_score.shape[1] > 1:
                        prev_seq_score = batch_gather(prev_seq_score, indices=backrefs)
                    # prev_seq_score: [Batch,ActBeam+EndedBeam]
                    seq_score = seq_score + prev_seq_score

                out_individual_seq_scores[k] = seq_score

        if max_act_beam_size == 0:
            break

        state_ = tree.map_structure(functools.partial(gather_, indices=backrefs_), new_state_)  # [Batch_,...]

        if opts.length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            end_seq_log_prob *= ((i_dev + 1) / i_dev) ** length_normalization_exponent_dev

    if opts.length_normalization_exponent != 0:
        # All seq_log_prob will be normalized by 1/(out_seq_len+1)**length_normalization_exponent.
        seq_log_prob *= (1 / i_dev) ** length_normalization_exponent_dev

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

    return seq_targets, seq_log_prob, seq_len.cpu()
