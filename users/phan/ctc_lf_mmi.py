"""
Implement lattice-free MMI training for CTC
"""

import torch
from i6_experiments.users.phan.utils.math import log_matmul, modified_log_matmul, batch_log_matmul
from i6_experiments.users.phan.utils.masking import get_seq_mask
####### Be careful about stuffs related to EOS and blank index with the BPE setup
def ctc_lf_mmi_context_1(
    log_probs, # (T, B, F)
    log_lm_probs, # (F, F)
    targets, # (B, S) # WITHOUT BOS EOS WHATSOEVER
    input_lengths, # (B,)
    target_lengths, # (B,)
    am_scale,
    lm_scale,
    blank_idx=0, # should be same as eos index
    eos_idx=None,
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

    NOTE: In case there is EOS in the vocab, its ctc posterior should be very small,
    because the transitions in the denominator does not consider EOS

    :param log_probs: log CTC output probs (T, B, F)
    :param log_lm_probs: (F, F), then log bigram LM probs of all possible context
    :param targets: Target sequences (B, S) WITHOUT ANY SOS EOS SYMBOL
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,)
    :param am_scale: AM scale
    :param lm_scale: LM scale
    :param blank_idx: Blank index in F dim
    :param eos_idx: EOS idx in the vocab. None if no EOS in log_probs vocab.
        If None, then blank_idx in log_lm_probs should be EOS
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
    if eos_idx is None or eos_idx == blank_idx: # vocab means no EOS and blank
        vocab_size = n_out - 1
        eos_symbol = blank_idx
    else:
        vocab_size = n_out - 2
        eos_symbol = eos_idx
    # numerator lm score, calculate by indexing from targets (B, S) and log LM probs (F, F)
    # ------------ memory consuming, must fix ------------
    eos_targets = torch.cat([torch.full((batch_size, 1), eos_symbol, dtype=torch.long).to(device), targets], dim=1)
    targets_eos = torch.cat([targets, torch.full((batch_size, 1), eos_symbol, dtype=torch.long).to(device)], dim=1)
    eos_targets_BS1F = eos_targets.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, vocab_size+1)
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
    log_empty_seq_prob = log_partial_empty_seq_prob.gather(0, (input_lengths-1).to(device).long().unsqueeze(0)).squeeze(0)
    # Empty lm score = p_LM(eos | bos)
    log_q_empty_seq = log_empty_seq_prob + log_lm_probs_scaled[eos_symbol][eos_symbol]

    # to remove blank and eos from the last dim
    out_idx = torch.arange(n_out)
    out_idx_vocab = out_idx[out_idx != blank_idx].long().to(device) # "vocab" means no EOS and blank
    if eos_idx is not None and eos_idx != blank_idx:
        out_idx_vocab = out_idx_vocab[out_idx_vocab != eos_idx]

    # denom score by DP
    # dim 2: 0 is non-blank, 1 is blank
    
    log_q = torch.full((max_audio_time, batch_size, 2, vocab_size), log_zero, device=device) # (T, B, 2, F-1), no blank in last dim
    # Init Q for t=1
    log_q[0, :, 0, :] = log_probs_scaled[0, :, out_idx_vocab] + log_lm_probs_scaled[eos_symbol, out_idx_vocab].unsqueeze(0).expand(batch_size, -1)
    log_lm_probs_scaled_wo_eos = log_lm_probs_scaled.index_select(0, out_idx_vocab).index_select(1, out_idx_vocab)
    for t in range(1, max_audio_time):
        # case 1: emit a blank at t
        new_log_q = torch.full((batch_size, 2, vocab_size), log_zero).to(device)
        new_log_q[:, 1, :] = log_q[t-1, :, :, :].clone().logsumexp(dim=1) + log_probs_scaled[t, :, blank_idx].unsqueeze(-1).expand(-1, vocab_size)
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
        log_mass_diagonal = log_matmul(log_prev_partial_seq_probs, log_lm_probs_scaled.index_select(1, out_idx_vocab)) # (B, F) @ (F, F-1)
        # skip transition sum{w!=u} Q(t-1, w, non-blank)*p_LM(u|w)
        # same consideration as diagonal transition
        log_mass_skip = modified_log_matmul(log_q[t-1, :, 0, :].clone(), log_lm_probs_scaled_wo_eos)
        # multiply with p_AM(u|x_t)
        new_log_q[:, 0, :] = log_probs_scaled[t, :, out_idx_vocab] + (torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1).logsumexp(dim=-1))
        time_mask = (t < input_lengths).unsqueeze(-1).unsqueeze(-1).expand(-1, 2, vocab_size).to(device)
        log_q[t] = torch.where(time_mask, new_log_q, log_q[t-1])
        torch.cuda.empty_cache()
    # multiply last Q with p_LM(eos | u)
    log_q[-1, :, :, :] = log_q[-1, :, :, :].clone() + log_lm_probs_scaled[out_idx_vocab, eos_idx].unsqueeze(0).unsqueeze(0).expand(batch_size, 2, -1)
    denom_score = torch.logaddexp(log_q[-1, :, :, :].logsumexp(dim=1).logsumexp(dim=1), log_q_empty_seq)
    loss = (-numer_score + denom_score).sum()
    return loss


def ctc_lf_mmi_context_1_topk(
    log_probs, # (T, B, V)
    targets, # (B, S) # WITHOUT BOS EOS
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_target_probs, # (B, S+1)
    log_bigram_probs, # (V, V)
    am_scale,
    lm_scale,
    top_k,
    blank_idx=10025, # should be same as eos index
    eos_idx=None,
    log_zero=-1e15,
):
    """
    Seems to only work for eos=0, blank=last...

    This version tries to reduce mem usage by having top K hypotheses
    at each time step.

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

    NOTE: In case there is EOS in the vocab, its ctc posterior should be very small,
    because the transitions in the denominator does not consider EOS

    :param log_probs: log CTC output probs (T, B, F)
    :param log_lm_probs: (F, F), then log bigram LM probs of all possible context
    :param targets: Target sequences (B, S) WITHOUT ANY SOS EOS SYMBOL
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,)
    :param am_scale: AM scale
    :param lm_scale: LM scale
    :param top_k: At each time step keeps top best K scores.
    :param blank_idx: Blank index in F dim
    :param eos_idx: EOS idx in the vocab. None if no EOS in log_probs vocab.
        If None, then blank_idx in log_lm_probs should be EOS
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :returns: log loss MMI
    """
    device = log_probs.device
    batch_size, max_seq_len = targets.shape
    max_audio_time, _, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs = am_scale*log_probs
    log_bigram_probs = lm_scale*log_bigram_probs
    log_bigram_probs_masked = log_bigram_probs.clone().fill_diagonal_(log_zero)

    # numerator am score
    neg_log_p_ctc = torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_idx,
        reduction="none",
    )
    if eos_idx is None or eos_idx == blank_idx: # vocab means no EOS and blank
        vocab_size = n_out - 1
        eos_symbol = blank_idx
    else:
        vocab_size = n_out - 2
        eos_symbol = eos_idx
    target_mask = get_seq_mask(target_lengths+1, max_seq_len+1, device)
    log_targets_lm_score = (lm_scale*log_target_probs*target_mask).sum(dim=-1) # (B,)
    numer_score = -neg_log_p_ctc + log_targets_lm_score # (B,)
    # denominator score
    # calculate empty sequence score
    # Empty am score = sum log prob blank
    log_partial_empty_seq_prob = log_probs[:, :, blank_idx].cumsum(dim=0)
    log_empty_seq_prob = log_partial_empty_seq_prob.gather(0, (input_lengths-1).to(device).long().unsqueeze(0)).squeeze(0)
    # Empty lm score = p_LM(eos | bos)
    log_q_empty_seq = log_empty_seq_prob + log_bigram_probs[eos_symbol][eos_symbol]

    # to remove blank and eos from the last dim
    out_idx = torch.arange(n_out)
    out_idx_vocab = out_idx[out_idx != blank_idx].long().to(device) # "vocab" means no EOS and blank
    out_idx_vocab_w_eos = out_idx_vocab
    if eos_idx is not None and eos_idx != blank_idx:
        out_idx_vocab = out_idx_vocab[out_idx_vocab != eos_idx]
    eos_idx_tensor = torch.tensor([eos_symbol]).long()

    # denom score by DP
    # dim 2: 0 is non-blank, 1 is blank
    # reminder: "vocab" and "vocab_size" are without EOS and blank
    
    # Init Q for t=1
    log_bigram_probs_no_eos = log_bigram_probs.index_select(0, out_idx_vocab).index_select(1, out_idx_vocab)
    log_bigram_probs_no_eos_masked = log_bigram_probs_no_eos.clone().fill_diagonal_(log_zero)
    all_hyp_scores_label = log_probs[0, :, out_idx_vocab] + log_bigram_probs[eos_symbol:eos_symbol+1, out_idx_vocab].expand(batch_size, -1) # (B, V)
    # all_hyp_scores_blank = log_probs[0, :, blank_idx].unsqueeze(-1).expand(-1, vocab_size)
    # all_hyp_scores = torch.logaddexp(all_hyp_scores_label, all_hyp_scores_blank)
    all_hyp_scores = all_hyp_scores_label
    all_hyp_scores_blank = torch.full((batch_size, vocab_size), fill_value=log_zero, device=log_probs.device)
    topk_scores, topk_idx = torch.topk(all_hyp_scores, k=top_k, dim=-1) # (B, K)
    # log_bigram_probs_no_eos = log_bigram_probs.index_select(0, out_idx_vocab).index_select(1, out_idx_vocab)
    for t in range(1, max_audio_time):
        # case 1: emit a blank at t
        all_hyp_scores_blank_new = all_hyp_scores + log_probs[t, :, blank_idx].unsqueeze(1).expand(-1, vocab_size)
        
        # case 2: emit a non-blank at t

        # horizontal transition Q(t-1, u, non-blank)
        log_mass_horizontal = all_hyp_scores_label # (B, V)

        # diagonal transition sum_v Q(t-1, v, blank)*p_LM(u|v)
        topk_scores_blank = all_hyp_scores_blank.gather(1, topk_idx)
        topk_scores_blank_with_empty_hyp = torch.concat([log_partial_empty_seq_prob[t-1].unsqueeze(-1), topk_scores_blank], dim=-1) # (B, 1+K)
        log_bigram_probs_from_topk = log_bigram_probs_no_eos.unsqueeze(0).expand(
            batch_size, -1, -1
        ).gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size)
        ) # (B, K, V)
        log_bigram_probs_from_topk_w_eos = torch.cat( # (B, 1+K, V)
            [
                log_bigram_probs[eos_symbol:eos_symbol+1, :].index_select(1, out_idx_vocab).unsqueeze(0).expand(batch_size, -1, -1), # (B, 1, V)
                log_bigram_probs_from_topk,
            ],
            dim=1,
        )
        log_mass_diagonal = batch_log_matmul(topk_scores_blank_with_empty_hyp.unsqueeze(1), log_bigram_probs_from_topk_w_eos).squeeze(1) # (B, 1, 1+K) @ (B, 1+K, V) --> (B, V)
        
        # skip transition
        topk_scores_label = all_hyp_scores_label.gather(1, topk_idx)
        log_bigram_probs_masked_from_topk = log_bigram_probs_no_eos_masked.unsqueeze(0).expand(
            batch_size, -1, -1
        ).gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size)
        ) # (B, K, V)
        log_mass_skip = batch_log_matmul(topk_scores_label.unsqueeze(1), log_bigram_probs_masked_from_topk).squeeze(1) # (B, 1, K) @ (B, K, V) --> (B, V)
        
        # multiply with p_AM(u|x_t)
        all_hyp_scores_label_new = log_probs[t, :, out_idx_vocab] + (torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1).logsumexp(dim=-1)) # (B, V)
        # prepare for next time step
        time_mask = (t < input_lengths).unsqueeze(-1).expand(-1, vocab_size).to(device)
        all_hyp_scores_label = torch.where(time_mask, all_hyp_scores_label_new, all_hyp_scores_label)
        all_hyp_scores_blank = torch.where(time_mask, all_hyp_scores_blank_new, all_hyp_scores_blank)
        all_hyp_scores = torch.logaddexp(all_hyp_scores_label, all_hyp_scores_blank)
        topk_scores, topk_idx = torch.topk(all_hyp_scores, k=top_k, dim=-1)
        torch.cuda.empty_cache()
    # multiply last Q with p_LM(eos | u)
    log_bigram_eos_prob_from_topk = log_bigram_probs[:, eos_symbol].unsqueeze(0).expand(
        batch_size, -1,
    ).gather(1, topk_idx) # (B, K)
    topk_scores = topk_scores + log_bigram_eos_prob_from_topk
    denom_score = torch.logaddexp(topk_scores.logsumexp(-1), log_q_empty_seq)
    loss = (-numer_score + denom_score).sum()
    return loss


def ctc_lf_mmi_context_1_topk_strict_v2(
    log_probs, # (T, B, V)
    targets, # (B, S) # WITHOUT BOS EOS
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_target_probs, # (B, S+1)
    log_bigram_probs, # (V, V)
    am_scale,
    lm_scale,
    top_k,
    blank_idx=10025, # should be same as eos index
    eos_idx=None,
    log_zero=-1e15,
):
    """
    Seems to only work for eos=0, blank=last...

    This version tries to reduce mem usage by having top K hypotheses
    at each time step AND ignore the horizontal transitions for hypothese
    not in the top K to have a strict top K approximation.

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

    NOTE: In case there is EOS in the vocab, its ctc posterior should be very small,
    because the transitions in the denominator does not consider EOS

    :param log_probs: log CTC output probs (T, B, F)
    :param log_lm_probs: (F, F), then log bigram LM probs of all possible context
    :param targets: Target sequences (B, S) WITHOUT ANY SOS EOS SYMBOL
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,)
    :param am_scale: AM scale
    :param lm_scale: LM scale
    :param top_k: At each time step keeps top best K scores.
    :param blank_idx: Blank index in F dim
    :param eos_idx: EOS idx in the vocab. None if no EOS in log_probs vocab.
        If None, then blank_idx in log_lm_probs should be EOS
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :returns: log loss MMI
    """
    device = log_probs.device
    batch_size, max_seq_len = targets.shape
    max_audio_time, _, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs = am_scale*log_probs
    log_bigram_probs = lm_scale*log_bigram_probs
    log_bigram_probs_masked = log_bigram_probs.clone().fill_diagonal_(log_zero)

    # numerator am score
    neg_log_p_ctc = torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_idx,
        reduction="none",
    )
    if eos_idx is None or eos_idx == blank_idx: # vocab means no EOS and blank
        vocab_size = n_out - 1
        eos_symbol = blank_idx
    else:
        vocab_size = n_out - 2
        eos_symbol = eos_idx
    target_mask = get_seq_mask(target_lengths+1, max_seq_len+1, device)
    log_targets_lm_score = (lm_scale*log_target_probs*target_mask).sum(dim=-1) # (B,)
    numer_score = -neg_log_p_ctc + log_targets_lm_score # (B,)
    # denominator score
    # calculate empty sequence score
    # Empty am score = sum log prob blank
    log_partial_empty_seq_prob = log_probs[:, :, blank_idx].cumsum(dim=0)
    log_empty_seq_prob = log_partial_empty_seq_prob.gather(0, (input_lengths-1).to(device).long().unsqueeze(0)).squeeze(0)
    # Empty lm score = p_LM(eos | bos)
    log_q_empty_seq = log_empty_seq_prob + log_bigram_probs[eos_symbol][eos_symbol]

    # to remove blank and eos from the last dim
    out_idx = torch.arange(n_out)
    out_idx_vocab = out_idx[out_idx != blank_idx].long().to(device) # "vocab" means no EOS and blank
    out_idx_vocab_w_eos = out_idx_vocab
    if eos_idx is not None and eos_idx != blank_idx:
        out_idx_vocab = out_idx_vocab[out_idx_vocab != eos_idx]
    eos_idx_tensor = torch.tensor([eos_symbol]).long()

    # denom score by DP
    # dim 2: 0 is non-blank, 1 is blank
    # reminder: "vocab" and "vocab_size" are without EOS and blank
    
    # Init Q for t=1
    log_bigram_probs_no_eos = log_bigram_probs.index_select(0, out_idx_vocab).index_select(1, out_idx_vocab)
    log_bigram_probs_no_eos_masked = log_bigram_probs_no_eos.clone().fill_diagonal_(log_zero)
    all_hyp_scores_label = log_probs[0, :, out_idx_vocab] + log_bigram_probs[eos_symbol:eos_symbol+1, out_idx_vocab].expand(batch_size, -1) # (B, V)
    # all_hyp_scores_blank = log_probs[0, :, blank_idx].unsqueeze(-1).expand(-1, vocab_size)
    # all_hyp_scores = torch.logaddexp(all_hyp_scores_label, all_hyp_scores_blank)
    all_hyp_scores = all_hyp_scores_label
    all_hyp_scores_blank = torch.full((batch_size, vocab_size), fill_value=log_zero, device=log_probs.device)
    topk_scores, topk_idx = torch.topk(all_hyp_scores, k=top_k, dim=-1) # (B, K)
    # log_bigram_probs_no_eos = log_bigram_probs.index_select(0, out_idx_vocab).index_select(1, out_idx_vocab)
    for t in range(1, max_audio_time):
        # calculate top k mask, for horizontal transitions
        topk_mask = torch.full_like(all_hyp_scores_label, fill_value=log_zero)
        topk_mask.scatter_(1, topk_idx, 0.0)  # put 0.0 in log space (log(1) = 0) at top-k positions

        # case 1: emit a blank at t
        all_hyp_scores_blank_new =  torch.logaddexp(
            all_hyp_scores_blank,
            all_hyp_scores_label + topk_mask,
        ) + log_probs[t, :, blank_idx].unsqueeze(1).expand(-1, vocab_size)
        
        # case 2: emit a non-blank at t

        # horizontal transition Q(t-1, u, non-blank)
        log_mass_horizontal = all_hyp_scores_label + topk_mask # (B, V)

        # diagonal transition sum_v Q(t-1, v, blank)*p_LM(u|v)
        topk_scores_blank = all_hyp_scores_blank.gather(1, topk_idx)
        topk_scores_blank_with_empty_hyp = torch.concat([log_partial_empty_seq_prob[t-1].unsqueeze(-1), topk_scores_blank], dim=-1) # (B, 1+K)
        log_bigram_probs_from_topk = log_bigram_probs_no_eos.unsqueeze(0).expand(
            batch_size, -1, -1
        ).gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size)
        ) # (B, K, V)
        log_bigram_probs_from_topk_w_eos = torch.cat( # (B, 1+K, V)
            [
                log_bigram_probs[eos_symbol:eos_symbol+1, :].index_select(1, out_idx_vocab).unsqueeze(0).expand(batch_size, -1, -1), # (B, 1, V)
                log_bigram_probs_from_topk,
            ],
            dim=1,
        )
        log_mass_diagonal = batch_log_matmul(topk_scores_blank_with_empty_hyp.unsqueeze(1), log_bigram_probs_from_topk_w_eos).squeeze(1) # (B, 1, 1+K) @ (B, 1+K, V) --> (B, V)
        
        # skip transition
        topk_scores_label = all_hyp_scores_label.gather(1, topk_idx)
        log_bigram_probs_masked_from_topk = log_bigram_probs_no_eos_masked.unsqueeze(0).expand(
            batch_size, -1, -1
        ).gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size)
        ) # (B, K, V)
        log_mass_skip = batch_log_matmul(topk_scores_label.unsqueeze(1), log_bigram_probs_masked_from_topk).squeeze(1) # (B, 1, K) @ (B, K, V) --> (B, V)
        
        # multiply with p_AM(u|x_t)
        all_hyp_scores_label_new = log_probs[t, :, out_idx_vocab] + (torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1).logsumexp(dim=-1)) # (B, V)
        # prepare for next time step
        time_mask = (t < input_lengths).unsqueeze(-1).expand(-1, vocab_size).to(device)
        all_hyp_scores_label = torch.where(time_mask, all_hyp_scores_label_new, all_hyp_scores_label)
        all_hyp_scores_blank = torch.where(time_mask, all_hyp_scores_blank_new, all_hyp_scores_blank)
        all_hyp_scores = torch.logaddexp(all_hyp_scores_label, all_hyp_scores_blank)
        topk_scores, topk_idx = torch.topk(all_hyp_scores, k=top_k, dim=-1)
        torch.cuda.empty_cache()
    # multiply last Q with p_LM(eos | u)
    log_bigram_eos_prob_from_topk = log_bigram_probs[:, eos_symbol].unsqueeze(0).expand(
        batch_size, -1,
    ).gather(1, topk_idx) # (B, K)
    topk_scores = topk_scores + log_bigram_eos_prob_from_topk
    denom_score = torch.logaddexp(topk_scores.logsumexp(-1), log_q_empty_seq)
    loss = (-numer_score + denom_score).sum()
    return loss
