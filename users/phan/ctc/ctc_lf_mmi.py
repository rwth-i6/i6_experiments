"""
Implement lattice-free MMI training for CTC
"""

import torch
from i6_experiments.users.phan.utils.math import log_matmul, modified_log_matmul
from i6_experiments.users.phan.utils.masking import get_seq_mask

def ctc_lf_mmi_context_1(
    log_probs, # (T, B, F)
    log_lm_probs, # (F, F)
    targets, # (B, S) # WITHOUT BOS EOS WHATSOEVER
    input_lengths, # (B,)
    target_lengths, # (B,)
    am_scale,
    lm_scale,
    blank_idx=0, # should be same as eos index
    log_zero=-1e15,
):
    """
    Lattice-free MMI training for CTC, given by
    L_MMI = q(target_seq) / sum_{all seq} q(seq)
    where q(seq) = p_AM^alpha(seq) * p_LM^beta(seq).

    This is for the case the LM is a bigram (context 1).

    The numerator is standard CTC loss plus LM score.

    The denominator is calculated by sum_{u in V} [Q(T, u, blank) + Q(T, u, non-blank)],
    where Q(t, u, {N or B}) is the some of partial CTC alignments up to timeframe t
    with u being the last emitted label, and the last emitted frame is non-blank or blank.

    This Q is calculated by the two recursions:
    Q(t, u, blank) = [Q(t-1, u, blank) + Q(t-1, u, non-blank)]*p_AM(blank | x_t)
    Q(t, u, non-blank) = p_AM(u|x_t) * [horizontal + diagonal + skip], where 
    horizontal = Q(t-1, u, non-blank)
    diagonal = sum_v Q(t-1, v, blank)*p_LM(u|v)
    skip = sum{w!=u} Q(t-1, w, non-blank)*p_LM(u|w)

    Initialization:
    Q(1, u, blank) = 0
    Q(1, u, non-blank) = p_AM(u | x1) * p_LM(u | bos)

    :param log_probs: log CTC output probs (T, B, F)
    :param log_lm_probs: log bigram LM probs of all possible context (F, F)
    :param targets: Target sequences (B, S) WITHOUT ANY SOS EOS SYMBOL
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,)
    :param am_scale: AM scale
    :param lm_scale: LM scale
    :param blank_idx: Blank index in F dim
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :returns: log loss MMI
    """
    device = log_probs.device
    batch_size, max_seq_len = targets.shape
    max_audio_time, _, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs_scaled = am_scale*log_probs
    log_lm_probs_scaled = lm_scale*log_lm_probs

    # numerator am score
    neg_log_p_ctc = torch.nn.functional.ctc_loss(
        log_probs_scaled,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_idx,
        reduction="none",
    )
    # numerator lm score, calculate by indexing from targets (B, S) and log LM probs (F, F)
    eos_targets = torch.cat([torch.full((batch_size, 1), blank_idx, dtype=torch.long).to(device), targets], dim=1)
    eos_targets_BS1F = eos_targets.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n_out)
    targets_eos = torch.cat([targets, torch.full((batch_size, 1), blank_idx, dtype=torch.long).to(device)], dim=1)
    targets_eos_BS1 = targets_eos.unsqueeze(-1)
    log_lm_probs_BSFF = log_lm_probs_scaled.unsqueeze(0).unsqueeze(0).expand(batch_size, max_seq_len+1, -1, -1)
    eos_targets_BSF = log_lm_probs_BSFF.gather(2, eos_targets_BS1F).squeeze(2) # log beam score for each pos in eos_targets (B, S, F)
    log_targets_lm_pos_score = eos_targets_BSF.gather(2, targets_eos_BS1).squeeze(2) # position-wise log loss of targets (B, S)

    target_mask = get_seq_mask(target_lengths+1, max_seq_len+1, device)
    log_targets_lm_score = (log_targets_lm_pos_score*target_mask).sum(dim=-1) # (B,)
    numer_score = -neg_log_p_ctc + log_targets_lm_score # (B,)

    # denominator score
    # calculate empty sequence score
    # Empty am score = sum log prob blank
    log_partial_empty_seq_prob = log_probs_scaled[:, :, blank_idx].cumsum(dim=0)
    log_empty_seq_prob = log_partial_empty_seq_prob.gather(0, (input_lengths-1).unsqueeze(0)).squeeze(0)
    # Empty lm score = p_LM(eos | eos)
    log_q_empty_seq = log_empty_seq_prob + log_lm_probs_scaled[blank_idx][blank_idx]

    # to remove blank from the last dim
    out_idx = torch.arange(n_out)
    out_idx_wo_blank = out_idx[out_idx != blank_idx].long().to(device)

    # denom score by DP
    # dim 2: 0 is non-blank, 1 is blank
    log_q = torch.full((max_audio_time, batch_size, 2, n_out-1), log_zero, device=device) # (T, B, 2, F-1), no blank in last dim
    # Init Q for t=1
    log_q[0, :, 0, :] = log_probs_scaled[0, :, out_idx_wo_blank] + log_lm_probs_scaled[0, out_idx_wo_blank].unsqueeze(0).expand(batch_size, -1)
    log_lm_probs_scaled_wo_blank = log_lm_probs_scaled.index_select(0, out_idx_wo_blank).index_select(1, out_idx_wo_blank)
    for t in range(1, max_audio_time):
        # case 1: emit a blank at t
        new_log_q = torch.full((batch_size, 2, n_out-1), log_zero).to(device)
        new_log_q[:, 1, :] = log_q[t-1, :, :, :].clone().logsumexp(dim=1) + log_probs_scaled[t, :, blank_idx].unsqueeze(-1).expand(-1, n_out-1)
        # case 2: emit a non-blank at t
        # horizontal transition Q(t-1, u, non-blank)
        log_mass_horizontal = log_q[t-1, :, 0, :].clone()
        # diagonal transition sum_v Q(t-1, v, blank)*p_LM(u|v)
        # take batch index b into account, this is equivalent to compute
        # mass_diagonal[b, u] = sum_v Q(t-1, b, blank, v)*p_LM(u|v)
        # mass_diagonal = Q(t-1, :, blank, :) @ M, where M(v,u) = p_LM(u|v) = lm_probs[v][u]
        # important: in this transition, there is a prefix empty^(t-1) that is not covered in the Q(t-1,v,blank)
        # this is covered in log_partial_empty_seq_prob[t-1]
        log_prev_partial_seq_probs = torch.concat([log_partial_empty_seq_prob[t-1].unsqueeze(-1), log_q[t-1, :, 1, :].clone()], dim=-1)
        log_mass_diagonal = log_matmul(log_prev_partial_seq_probs, log_lm_probs_scaled.index_select(1, out_idx_wo_blank)) # (B, F) @ (F, F-1)
        # skip transition sum{w!=u} Q(t-1, w, non-blank)*p_LM(u|w)
        # same consideration as diagonal transition
        log_mass_skip = modified_log_matmul(log_q[t-1, :, 0, :].clone(), log_lm_probs_scaled_wo_blank)
        # multiply with p_AM(u|x_t)
        new_log_q[:, 0, :] = log_probs_scaled[t, :, out_idx_wo_blank] + (torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1).logsumexp(dim=-1))
        time_mask = (t < input_lengths).unsqueeze(-1).unsqueeze(-1).expand(-1, 2, n_out-1).to(device)
        log_q[t] = torch.where(time_mask, new_log_q, log_q[t-1])
    # multiply last Q with p_LM(eos | u)
    log_q[-1, :, :, :] = log_q[-1, :, :, :].clone() + log_lm_probs_scaled[out_idx_wo_blank, 0].unsqueeze(0).unsqueeze(0).expand(batch_size, 2, -1)
    denom_score = torch.logaddexp(log_q[-1, :, :, :].logsumexp(dim=1).logsumexp(dim=1), log_q_empty_seq)
    loss = (-numer_score + denom_score).sum()
    return loss
