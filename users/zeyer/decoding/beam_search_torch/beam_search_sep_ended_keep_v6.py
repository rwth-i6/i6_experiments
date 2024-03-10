"""
Beam search
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import functools
from dataclasses import dataclass
import torch
import tree

from .interface import LabelScorerIntf
from .utils import top_k_nd, batch_gather, batch_gather_


@dataclass
class BeamSearchSepEndedKeepOpts:
    beam_size: int  # e.g. 12, for active hyps
    beam_ended_size: int  # e.g. 1 or 12, for ended hyps. influences ending condition, and what we return.

    bos_label: int
    eos_label: int
    num_labels: int

    pruning_threshold: Optional[float] = None  # prune active hyps away compared to best ended hyp
    adaptive_pruning: bool = False
    pruning_threshold_worst: Optional[float] = None  # prune active hyps away compared to worst ended hyp
    length_normalization_exponent: float = 0.0  # e.g. 1 to enable, 0 to disable
    length_normalization_offset: int = 0  # calc e.g. ((N + offset) / (offset + 1)) ** exp instead. Google NMT: offset=5


def beam_search_sep_ended_keep_v6(
    label_scorer: LabelScorerIntf,
    *,
    batch_size: int,
    max_seq_len: torch.Tensor,
    device: torch.device,
    opts: BeamSearchSepEndedKeepOpts,
    out_individual_seq_scores: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Beam search with dynamic beam size and ended hypotheses kept separately.

    Based on beam_search_sep_ended,
    but we keep always N best ended hyps.

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
    # - InActBeam:
    #       Padded active hyps, unpacked from Batch_.
    # - Beam:
    #       After topk on active hyps (Max(InBeam) * Vocab).
    #       k = opts.beam_size, i.e. size = min(opts.num_labels * (opts.num_labels - 1) ** i, opts.beam_size).
    # - InEndBeam:
    #       Padded ended hyps, unpacked from Batch_E.
    # - Beam+InEndBeam:
    #       Both concatenated together. Padded area might occur in between.

    bad_score = -1.0e30
    bad_score_dev = torch.full((), bad_score, device=device)
    max_seq_len = max_seq_len.to(device)
    len_norm_exp_dev = torch.full((), opts.length_normalization_exponent, device=device)
    len_norm_offset_dev = torch.full((), opts.length_normalization_offset, dtype=len_norm_exp_dev.dtype, device=device)
    len_norm_offset1_dev = len_norm_offset_dev + 1

    # Initial state.
    act_state = label_scorer.get_initial_state(batch_size=batch_size, device=device)  # [Batch,InActBeam=1]
    act_target = torch.full([batch_size, 1], opts.bos_label, device=device)  # [Batch,InActBeam=1]
    act_valid = torch.full([batch_size, 1], True, device=device)  # [Batch,InActBeam=1]
    act_seq_log_prob = torch.full([batch_size, 1], 0.0, device=device)  # [Batch,InActBeam=1]
    end_seq_log_prob = torch.full([batch_size, 0], 0.0, device=device)  # [Batch,InEndBeam=0]

    i = 0
    i_dev = torch.zeros((), dtype=torch.int64, device=device)
    seq_targets = []
    seq_backrefs = []
    while True:
        seq_log_prob_ext, individual_scores, new_state = label_scorer.seq_score_ext_and_update_state(
            prev_seq_scores=act_seq_log_prob, prev_state=act_state, prev_label=act_target
        )
        del act_state
        # seq_log_prob_ext: [Batch,InActBeam,Vocab]
        # individual_scores: all tensors have [Batch|1,InActBeam|1,Vocab|1]
        # new_state: all tensors have [Batch,InActBeam,...]

        torch.where(act_valid[:, :, None], seq_log_prob_ext, bad_score_dev, out=seq_log_prob_ext)

        end_seq_log_prob = torch.concat(
            [
                seq_log_prob_ext[:, :, opts.eos_label],  # [Batch,InActBeam]
                end_seq_log_prob,  # [Batch,InEndBeam]
            ],
            dim=1,
        )  # [Batch,InActBeam+InEndBeam]
        seq_log_prob_ext[:, :, opts.eos_label] = bad_score_dev

        prev_max_act_beam_size = seq_log_prob_ext.shape[1]  # InActBeam

        act_seq_log_prob, (act_backrefs, act_target) = top_k_nd(
            seq_log_prob_ext, k=opts.beam_size, dim=[1, 2]
        )  # all [Batch,ActBeam]
        # act_backrefs: [Batch,ActBeam] -> InActBeam, should be in [0...InActBeam[b]-1] for each b
        del seq_log_prob_ext
        act_valid = batch_gather(act_valid, indices=act_backrefs)  # [Batch,ActBeam]

        end_seq_log_prob, end_backrefs = torch.topk(
            end_seq_log_prob, k=min(end_seq_log_prob.shape[1], opts.beam_ended_size), dim=1
        )  # all [Batch,EndBeam]
        # backrefs_eos: [Batch,EndBeam] -> InActBeam+InEndBeam

        act_valid &= (i_dev < max_seq_len)[:, None]
        if (
            opts.pruning_threshold_worst is not None and end_seq_log_prob.shape[1] >= opts.beam_ended_size
        ):  # filled the ended beam
            # Filter out active which are worse than the worst ended hyp.
            worst_ended_seq_log_prob = end_seq_log_prob[:, -1]  # [Batch]
            pruning_threshold = worst_ended_seq_log_prob - opts.pruning_threshold_worst  # [Batch]
            act_valid &= act_seq_log_prob > pruning_threshold[:, None]

        if opts.pruning_threshold is not None and end_seq_log_prob.shape[1] > 0:
            # Prune in relation to best ended hyp.
            best_ended_seq_log_prob = end_seq_log_prob[:, 0]  # [Batch]
            pruning_threshold = best_ended_seq_log_prob - opts.pruning_threshold  # [Batch]
            act_valid &= act_seq_log_prob > pruning_threshold[:, None]

        if opts.adaptive_pruning and end_seq_log_prob.shape[1] > 0:
            # Prune in relation to best potential future score.
            best_ended_seq_log_prob = end_seq_log_prob[:, 0]  # [Batch]
            max_remaining_steps = (max_seq_len - i_dev)[:, None]  # [Batch,ActBeam=InActBeam=1]
            max_gain = label_scorer.max_remaining_seq_score(
                state=new_state, max_remaining_steps=max_remaining_steps, device=device
            )  # [Batch|1,InActBeam|1]
            if max_gain.shape[1] > 1:
                max_gain = batch_gather(max_gain, indices=act_backrefs)  # [Batch,ActBeam]
            max_future_seq_log_prob = act_seq_log_prob + max_gain  # [Batch,ActBeam]
            if opts.length_normalization_exponent != 0:
                # Normalize with (1/(out_seq_len+1))**exp (for best ended, see also same logic at the end).
                best_ended_seq_log_prob *= (len_norm_offset1_dev / (i_dev + len_norm_offset1_dev)) ** len_norm_exp_dev
                max_future_seq_log_prob *= (
                    len_norm_offset1_dev / (max_seq_len[:, None] + len_norm_offset1_dev)
                ) ** len_norm_exp_dev
            act_valid &= max_future_seq_log_prob > best_ended_seq_log_prob[:, None]

        torch.where(act_valid, act_seq_log_prob, bad_score_dev, out=act_seq_log_prob)
        act_beam_sizes = act_valid.sum(dim=1)  # [Batch]
        act_beam_sizes_cpu = act_beam_sizes.cpu()  # single CUDA sync
        max_act_beam_size = act_beam_sizes_cpu.max()  # scalar

        # Seqs are sorted per score, thus we can just slice the best.
        # Slice for the next iteration. `backrefs`, `target` still contain all.
        act_valid = act_valid[:, :max_act_beam_size]
        act_seq_log_prob = act_seq_log_prob[:, :max_act_beam_size]
        act_backrefs = act_backrefs[:, :max_act_beam_size]
        act_target = act_target[:, :max_act_beam_size]

        seq_log_prob = torch.concat([act_seq_log_prob, end_seq_log_prob], dim=1)  # [Batch,ActBeam+EndBeam]
        backrefs = torch.concat(
            [
                act_backrefs,  # [Batch,ActBeam] -> InActBeam
                end_backrefs,  # [Batch,EndBeam] -> InActBeam+InEndBeam
            ],
            dim=1,
        )  # [Batch,ActBeam+EndBeam] -> InActBeam+InEndBeam
        target = torch.concat(
            [
                act_target,
                torch.full(end_seq_log_prob.shape, opts.eos_label, dtype=act_target.dtype, device=device),
            ],
            dim=1,
        )  # [Batch,ActBeam+EndBeam]

        i += 1
        i_dev += 1

        seq_targets.append(target)
        seq_backrefs.append(backrefs)

        if out_individual_seq_scores is not None:
            # Similar as combine_individual_seq_scores but adapted for the packed format.
            # individual_scores: [Batch,InActBeam,Vocab]
            # prev out_individual_seq_scores: [Batch,InActBeam+InEndedBeam]
            # want: out_individual_seq_scores: [Batch,ActBeam+EndedBeam]

            prev_was_active = backrefs < prev_max_act_beam_size  # [Batch,ActBeam+EndBeam]
            prev_was_active &= seq_log_prob > bad_score_dev
            backrefs_flat = torch.clip(backrefs, 0, prev_max_act_beam_size - 1)
            backrefs_flat += (
                torch.arange(batch_size, dtype=backrefs.dtype, device=device)[:, None] * prev_max_act_beam_size
            )  # [Batch,ActBeam+EndBeam] -> flat indices in (Batch,InActBeam), i.e. Batch * InActBeam + InActBeam
            backrefs_flat *= opts.num_labels
            backrefs_flat += target  # [Batch,ActBeam+EndedBeam] -> flat indices in (Batch,InActBeam,Vocab)

            for k in list(individual_scores.keys()):

                seq_score = individual_scores.pop(k)  # [Batch|1,InActBeam|1,Vocab|1]
                if seq_score.shape == (1, 1, 1):  # [Batch=1,InActBeam=1,Vocab=1]
                    seq_score = seq_score[0]  # interpret it as [Batch=1,(ActBeam+EndedBeam)=1]
                elif (
                    seq_score.shape[0] * seq_score.shape[1] == 1 < batch_size * prev_max_act_beam_size
                    and seq_score.shape[2] == opts.num_labels
                ):  # [Batch=1,InActBeam=1,Vocab]
                    raise NotImplementedError(
                        f"seq_score shape {seq_score.shape}, bc Batch&Beam,"
                        f" InActBeam {prev_max_act_beam_size}, target shape {target.shape}, vocab {opts.num_labels}"
                    )
                elif (
                    seq_score.shape[0] == batch_size
                    and seq_score.shape[1] == prev_max_act_beam_size
                    and seq_score.shape[2] == 1 < opts.num_labels
                ):  # [Batch,InActBeam,Vocab=1]
                    seq_score = seq_score.flatten()[backrefs_flat // opts.num_labels]  # [Batch,ActBeam+EndedBeam]
                elif seq_score.shape == (
                    batch_size,
                    prev_max_act_beam_size,
                    opts.num_labels,
                ):  # [Batch,InActBeam,Vocab]
                    seq_score = seq_score.flatten()[backrefs_flat]  # [Batch,ActBeam+EndedBeam]
                else:
                    raise RuntimeError(
                        f"did not expect seq_score shape {seq_score.shape},"
                        f" InActBeam {prev_max_act_beam_size}, target shape {target.shape}, vocab {opts.num_labels}"
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

        act_state = tree.map_structure(
            functools.partial(batch_gather_, indices=act_backrefs), new_state
        )  # [Batch,ActBeam,...]
        del new_state

        if opts.length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            end_seq_log_prob *= ((i_dev + len_norm_offset1_dev) / (i_dev + len_norm_offset_dev)) ** len_norm_exp_dev

    if opts.length_normalization_exponent != 0:
        # All seq_log_prob will be normalized by 1/(out_seq_len+1)**length_normalization_exponent.
        seq_log_prob *= (len_norm_offset1_dev / (i_dev + len_norm_offset_dev)) ** len_norm_exp_dev

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
    seq_len = seq_targets.shape[2] - (seq_targets == opts.eos_label).sum(dim=2)  # [Batch,FinalBeam]
    return seq_targets, seq_log_prob, seq_len.cpu()
