"""
Implement lattice-free MMI training for CTC
"""

import torch
import numpy as np

####### Be careful about stuffs related to EOS and blank index with the BPE setup
def sum_loss(
    *,
    log_probs: torch.Tensor, # (T, B, V)
    log_lm_probs: torch.Tensor, # (V, V)
    log_prior: torch.Tensor | None, # (V,)
    input_lengths: torch.Tensor, # (B,)
    top_k: int = 0,
    LM_order: int,
    am_scale: float,
    lm_scale: float,
    prior_scale: float,
    horizontal_prior: bool,
    blank_prior: bool = True,
    blank_idx:int = 0, # should be same as eos index
    eos_idx: int | None = None,
    unk_idx: int = 1,
    log_zero: float = float("-inf"),
    device: torch.device = torch.device("cpu"),
    print_best_path_for_idx: list[int] = [],
    alignment_topk: bool = False,
):
    """
    Sum criterion training for CTC, given by
    L = sum_{all seq} q(seq)
    where q(seq) = p_AM^alpha(seq) * p_LM^beta(seq) / p_PR(seq)
    and p_AM = prod_n posterior.

    The loss is calculated by sum_{u in V} [Q(T, u, blank) + Q(T, u, non-blank)],
    where Q(t, u, {N or B}) is the sum of partial CTC alignments up to timeframe t
    with u being the last emitted label, and the last emitted frame is non-blank or blank.

    This Q is calculated by the two recursions (NOTE: the prior is used in all transitions):
    Q(t, u, blank) = [p_AM(blank | x_t) / p_PR(blank)] * [Q(t-1, u, blank) + Q(t-1, u, non-blank)]
    Q(t, u, non-blank) = [p_AM(u|x_t) / p_PR(u)] * [horizontal + diagonal + skip], where
    horizontal = Q(t-1, u, non-blank)
    diagonal = sum_v Q(t-1, v, blank) * p_LM(u|v)
    skip = sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w)
    
    Alternative calculation (NOTE: the prior is only used when the LM is used):
    Q(t, u, blank) = p_AM(blank | x_t) * [Q(t-1, u, blank) + Q(t-1, u, non-blank)]
    Q(t, u, non-blank) = p_AM(u|x_t) * [horizontal + diagonal + skip], where
    horizontal = Q(t-1, u, non-blank)
    diagonal = sum_v Q(t-1, v, blank) * p_LM(u|v) / p_PR(u)
    skip = sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w) / p_PR(u)

    Initialization:
    Q(1, u, blank) = 0
    Q(1, u, non-blank) = p_AM(u | x1) * p_LM(u | bos) / p_PR(u)

    NOTE: In case there is EOS in the vocab, its ctc posterior should be very small,
    because the transitions in the sum do not consider EOS.

    :param log_probs: log CTC output probs (T, B, V)
    :param log_lm_probs: (V, V), then log bigram LM probs of all possible context
    :param log_prior: vocab log prior probs (V,)
    :param input_lengths: Input lengths (B,)
    :param LM_order: Order of the LM
    :param am_scale: AM scale
    :param lm_scale: LM scale
    :param blank_idx: Blank index in V dim
    :param eos_idx: EOS idx in the vocab. None if no EOS in log_probs vocab.
        If None, then blank_idx in log_lm_probs should be EOS
    :param unk_idx: Unknown index in V dim
    :param log_zero: Value of log zero.
    :returns: log sum loss
    """
    # TODO generalize LM usage to any order
    
    use_prior = log_prior is not None
    
    old_device = log_probs.device
    log_probs = log_probs.to(device)
    log_lm_probs = log_lm_probs.to(device)
    if use_prior:
        log_prior = log_prior.to(device)
    input_lengths = input_lengths.to(device)
    
    max_audio_time, batch_size, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs = am_scale * log_probs
    if lm_scale == 0.0:
        log_lm_probs = torch.zeros_like(log_lm_probs, device=device)
    else:
        log_lm_probs = lm_scale * log_lm_probs
    if use_prior:
        log_prior = prior_scale * log_prior
    
    # print_gradients = PrintGradients.apply
    # grad_assert = AssertGradients.apply
    
    if eos_idx is None or eos_idx == blank_idx: # vocab means no EOS and blank
        vocab_size = n_out - 1
        eos_symbol = blank_idx
    else:
        vocab_size = n_out - 2
        eos_symbol = eos_idx
        assert blank_idx == n_out - 1, "blank should be the last symbol"
    assert unk_idx is not None, "unk_idx should be defined"
    vocab_size -= 1 # remove unk from vocab size
    assert log_lm_probs.size() == (vocab_size + 2, vocab_size + 2), f"LM shape is not correct, should be {vocab_size + 2}x{vocab_size + 2} but is {log_lm_probs.size()}"
    if use_prior:
        assert log_prior.size() == (n_out + 1,) or log_prior.size() == (n_out,), f"Prior shape is not correct, should be {n_out} or {n_out + 1} but is {log_prior.size()}"
    
    # calculate empty sequence score
    # Empty am score = sum log prob blank
    if use_prior and blank_prior:
        log_partial_empty_seq_prob = (log_probs[:, :, blank_idx] - log_prior[blank_idx]).cumsum(dim=0)
    else:
        log_partial_empty_seq_prob = log_probs[:, :, blank_idx].cumsum(dim=0)
    log_empty_seq_prob = log_partial_empty_seq_prob.gather(0, (input_lengths-1).long().unsqueeze(0)).squeeze(0)
    # Empty lm score = p_LM(eos | bos), prior score = p_PR(eos)
    log_q_empty_seq = log_empty_seq_prob + log_lm_probs[eos_symbol, eos_symbol]

    # Unbind the log_probs and log_partial_empty_seq_prob for each timestep so it is faster during backprop
    log_probs = log_probs.unbind(0)
    log_partial_empty_seq_prob = log_partial_empty_seq_prob.unbind(0)

    # to remove blank, unk and eos from the last dim (vocab)
    out_idx = torch.arange(n_out, device=device)
    out_idx_vocab = out_idx[out_idx != blank_idx].long() # "vocab" means no EOS, unk and blank
    out_idx_vocab = out_idx_vocab[out_idx_vocab != unk_idx]
    out_idx_vocab_w_eos = out_idx[out_idx != unk_idx].long()
    if eos_idx is not None and eos_idx != blank_idx:
        out_idx_vocab = out_idx_vocab[out_idx_vocab != eos_idx]
        out_idx_vocab_w_eos = out_idx_vocab_w_eos[out_idx_vocab_w_eos != blank_idx]

    # sum score by DP
    
    # List in which we store the log Q values as tensors of the last N timesteps
    # dim 2: 0 is non-blank, 1 is blank
    # Init Q for t=1
    # Q(1, u, blank) = 0
    # Q(1, u, non-blank) = p_AM(u | x1) * p_LM(u | bos) / p_PR(u)
    log_q_label = log_probs[0][:, out_idx_vocab] + log_lm_probs[eos_symbol, out_idx_vocab].unsqueeze(0)
    if use_prior:
        log_q_label = log_q_label - log_prior[out_idx_vocab].unsqueeze(0)
    log_q_blank = torch.full((batch_size, vocab_size), log_zero, device=device) # (B, 2, V-1), no blank and eos in last dim
    log_q = log_q_label
    
    # Calculate initial top k if needed
    if top_k > 0:
        if alignment_topk:
            tmp_log_q = torch.cat([log_q_label, log_q_blank, log_partial_empty_seq_prob[0].unsqueeze(-1)], dim=-1)
        else:
            tmp_log_q = torch.cat([log_q, log_partial_empty_seq_prob[0].unsqueeze(-1)], dim=-1)
        topk_scores, topk_idx = torch.topk(tmp_log_q, top_k, dim=-1, sorted=False)
        # print(topk_idx[print_best_path_for_idx[0]], topk_scores[print_best_path_for_idx[0]])
    
    # Set up the best path print
    if print_best_path_for_idx:
        with torch.no_grad():
            best_path_print = {}
            max_val, max_idx = torch.max(log_q, dim=-1)
            for idx in print_best_path_for_idx:
                best_path_print[idx] = {"str": f"{max_idx[idx] + 2}", "am_str": "{:.2f}".format(log_probs[0][idx][max_idx[idx] + 2].tolist()), "prior": "{:.2f}".format(log_prior[max_idx[idx] + 2].tolist() if use_prior else 0.0), "LM": "{:.2f}".format(log_lm_probs[eos_symbol, max_idx[idx] + 2].tolist()), "score": "{:.2f}".format(max_val[idx].tolist()), "AM": log_probs[0][idx][max_idx[idx] + 2].tolist()}
    
    log_lm_probs_wo_eos = log_lm_probs[out_idx_vocab][:, out_idx_vocab].fill_diagonal_(log_zero)
    for t in range(1, max_audio_time):
        # case 1: emit a blank at t
        # Q(t, u, blank) = [Q(t-1, u, blank) + Q(t-1, u, non-blank)]*p_AM(blank | x_t)
        if top_k > 0:
            # Only consider blanks from top k indexes (excluding the blank sequence)
            mask_label = torch.zeros((batch_size, vocab_size), device=device, dtype=torch.bool)
            if alignment_topk:
                mask_blank = torch.zeros((batch_size, vocab_size), device=device, dtype=torch.bool)
            for b in range(batch_size):
                mask_label[b, topk_idx[b][topk_idx[b] < vocab_size]] = True
                if alignment_topk:
                    mask_blank[b, topk_idx[b][(topk_idx[b] >= vocab_size) & (topk_idx[b] < 2 * vocab_size)] - vocab_size] = True
            if alignment_topk:
                new_log_q_blank_l = log_q_label + log_probs[t][:, blank_idx].unsqueeze(-1)
                new_log_q_blank_l = torch.where(mask_label, new_log_q_blank_l, log_zero)
                new_log_q_blank_b = log_q_blank + log_probs[t][:, blank_idx].unsqueeze(-1)
                new_log_q_blank_b = torch.where(mask_blank, new_log_q_blank_b, log_zero)
                new_log_q_blank = safe_logaddexp(new_log_q_blank_l, new_log_q_blank_b)
            else:
                new_log_q_blank = log_q + log_probs[t][:, blank_idx].unsqueeze(-1)
                new_log_q_blank = torch.where(mask_label, new_log_q_blank, log_zero)
        else:
            new_log_q_blank = log_q + log_probs[t][:, blank_idx].unsqueeze(-1)
        if use_prior and blank_prior:
            new_log_q_blank = new_log_q_blank - log_prior[blank_idx]

        # case 2: emit a non-blank at t
        # Q(t, u, non-blank) = p_AM(u|x_t) * [horizontal + diagonal + skip] 
        
        # horizontal transition Q(t-1, u, non-blank)
        if top_k > 0:
            log_mass_horizontal = log_q_label
            log_mass_horizontal = torch.where(mask_label, log_mass_horizontal, log_zero)
        else:
            log_mass_horizontal = log_q_label
        if horizontal_prior and use_prior:
            log_mass_horizontal = log_mass_horizontal - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
        # diagonal transition sum_v Q(t-1, v, blank) * p_LM(u|v) / p_PR(u)
        # take batch index b into account, this is equivalent to compute
        # mass_diagonal[b, u] = sum_v Q(t-1, b, blank, v) * p_LM(u|v) / p_PR(u)
        # mass_diagonal = Q(t-1, :, blank, :) @ M / p_PR(u), where M(v,u) = p_LM(u|v) = lm_probs[v][u]
        # important: in this transition, there is a prefix empty^(t-1) that is not covered in the Q(t-1,v,blank)
        # this is covered in log_partial_empty_seq_prob[t-1]
        if top_k > 0:
            if alignment_topk:
                empty = torch.full((batch_size, vocab_size), log_zero, device=device)
                log_q_blank_topk = torch.cat([empty, log_q_blank, log_partial_empty_seq_prob[t-1].unsqueeze(-1)], dim=-1).gather(-1, topk_idx) # (B, K)
            else:
                log_q_blank_topk = torch.cat([log_q_blank, log_partial_empty_seq_prob[t-1].unsqueeze(-1)], dim=-1).gather(-1, topk_idx) # (B, K)
            log_lm_probs_topk = log_lm_probs[out_idx_vocab][:, out_idx_vocab].unsqueeze(0).expand(batch_size, -1, -1) # (B, V-1, V-1)
            if alignment_topk:
                log_lm_probs_topk = torch.cat([log_lm_probs_topk, log_lm_probs_topk, log_lm_probs[eos_symbol, out_idx_vocab].unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)], dim=1) # (B, V, V-1)
            else:
                log_lm_probs_topk = torch.cat([log_lm_probs_topk, log_lm_probs[eos_symbol, out_idx_vocab].unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)], dim=1) # (B, V, V-1)
            log_lm_probs_topk = log_lm_probs_topk.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size)) # (B, K, V-1)
            log_mass_diagonal = log_matmul(log_q_blank_topk, log_lm_probs_topk, batch_given=True) # (B, K+1) @ (B, K+1, V-1) -> (B, V-1)
        else:
            log_prev_partial_seq_probs = torch.cat([log_partial_empty_seq_prob[t-1].unsqueeze(-1), log_q_blank], dim=-1) # (B, V)
            log_mass_diagonal = log_matmul(log_prev_partial_seq_probs, log_lm_probs[out_idx_vocab_w_eos][:, out_idx_vocab]) # (B, V) @ (V, V-1) -> (B, V-1)
        if use_prior:
            log_mass_diagonal = log_mass_diagonal - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
        # skip transition sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w) / p_PR(u)
        # same consideration as diagonal transition
        if top_k > 0:
            if alignment_topk:
                empty = torch.full((batch_size, vocab_size), log_zero, device=device)
                log_q_label_topk = torch.cat([log_q_label, empty, torch.full((batch_size, 1), log_zero, device=device)], dim=-1).gather(-1, topk_idx)
                log_lm_probs_wo_eos_topk = torch.cat([log_lm_probs_wo_eos, log_lm_probs_wo_eos, torch.full((1, vocab_size), log_zero, device=device)], dim=0).unsqueeze(0).expand(batch_size, -1, -1).gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size)) # (B, K, V-1)
            else:
                log_q_label_topk = torch.cat([log_q_label, torch.full((batch_size, 1), log_zero, device=device)], dim=-1).gather(-1, topk_idx)
                log_lm_probs_wo_eos_topk = torch.cat([log_lm_probs_wo_eos, torch.full((1, vocab_size), log_zero, device=device)], dim=0).unsqueeze(0).expand(batch_size, -1, -1).gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size)) # (B, K, V-1)
            log_mass_skip = log_matmul(log_q_label_topk, log_lm_probs_wo_eos_topk, batch_given=True) # (B, K) @ (B, K, V-1) -> (B, V-1)
        else:
            log_mass_skip = log_matmul(log_q_label, log_lm_probs_wo_eos) # (B, V-1) @ (V-1, V-1) -> (B, V-1)
        if use_prior:
            log_mass_skip = log_mass_skip - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
        # multiply with p_AM(u|x_t)
        new_log_q_label = log_probs[t][:, out_idx_vocab] + safe_logsumexp(torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1), dim=-1)
        
        # set masked results to log_q
        time_mask = (t < input_lengths).unsqueeze(-1).expand(-1, vocab_size)
        log_q_blank = torch.where(time_mask, new_log_q_blank, log_q_blank)
        log_q_label = torch.where(time_mask, new_log_q_label, log_q_label)
        log_q = safe_logaddexp(log_q_label, log_q_blank)
        
        if print_best_path_for_idx:
            with torch.no_grad():
                max_val, max_idx = torch.max(log_q, dim=-1)
                for idx in print_best_path_for_idx:
                    best_path_print[idx]["str"] += f" {max_idx[idx] + 2}"
                    best_path_print[idx]["am_str"] += " {:.2f}".format(log_probs[t][idx][max_idx[idx] + 2].tolist())
                    best_path_print[idx]["prior"] += " {:.2f}".format(log_prior[max_idx[idx] + 2].tolist() if use_prior else 0.0)
                    best_path_print[idx]["LM"] += " {:.2f}".format(safe_logsumexp(log_lm_probs_wo_eos[:, max_idx[idx]], dim=-1).tolist() if lm_scale > 0.0 else 0.0)
                    best_path_print[idx]["score"] += " {:.2f}".format(max_val[idx].tolist()) #  / (t+1)
                    best_path_print[idx]["AM"] += log_probs[t][idx][max_idx[idx] + 2].tolist()
        
        if top_k > 0:
            tmp_log_partial_empty_seq_prob = torch.where((t < input_lengths), log_partial_empty_seq_prob[t], log_empty_seq_prob).unsqueeze(-1)
            if alignment_topk:
                tmp_log_q = torch.cat([log_q_label, log_q_blank, tmp_log_partial_empty_seq_prob], dim=-1) # TODO: we should only add it if it also was in top k up until now
                # If we are in the last timestep, we also have to add the EOS LM probabaility
                last_mask = (t == input_lengths - 1).unsqueeze(-1).expand(-1, vocab_size * 2 + 1)
                tmp_log_q = torch.where(last_mask, tmp_log_q + torch.cat([log_lm_probs[out_idx_vocab, eos_symbol], log_lm_probs[out_idx_vocab, eos_symbol], log_lm_probs[eos_symbol, eos_symbol].unsqueeze(0)], dim=0).unsqueeze(0), tmp_log_q)
            else:
                tmp_log_q = torch.cat([log_q, tmp_log_partial_empty_seq_prob], dim=-1) # TODO: we should only add it if it also was in top k up until now
                # If we are in the last timestep, we also have to add the EOS LM probabaility
                last_mask = (t == input_lengths - 1).unsqueeze(-1).expand(-1, vocab_size + 1)
                tmp_log_q = torch.where(last_mask, tmp_log_q + torch.cat([log_lm_probs[out_idx_vocab, eos_symbol], log_lm_probs[eos_symbol, eos_symbol].unsqueeze(0)], dim=0).unsqueeze(0), tmp_log_q)
            # Calculate top k and apply time mask
            time_mask_k = (t < input_lengths).unsqueeze(-1).expand(-1, top_k)
            new_topk_scores, new_topk_idx = torch.topk(tmp_log_q, top_k, dim=-1, sorted=False)
            topk_scores = torch.where(time_mask_k, new_topk_scores, topk_scores)
            topk_idx = torch.where(time_mask_k, new_topk_idx, topk_idx)
            # print(topk_idx[print_best_path_for_idx[0]], topk_scores[print_best_path_for_idx[0]])
        
        torch.cuda.empty_cache()
    
    # multiply last Q with p_LM(eos | u) and devide by prior of EOS
    if top_k > 0:
        log_q = topk_scores
    else:
        log_q = log_q + log_lm_probs[out_idx_vocab, eos_symbol].unsqueeze(0)
    if print_best_path_for_idx:
        with torch.no_grad():
            for idx in print_best_path_for_idx:
                print(f"Best path for {idx}: {best_path_print[idx]['str']}\nAM str: {best_path_print[idx]['am_str']}\nPrior: {best_path_print[idx]['prior']}\nLM: {best_path_print[idx]['LM']}\nScore: {best_path_print[idx]['score']}\nAM: {best_path_print[idx]['AM']}")
    
    # sum over the vocab dimension
    sum_score = safe_logsumexp(log_q, dim=-1)
    if top_k <= 0:
        # add empty sequence score
        sum_score = safe_logaddexp(sum_score, log_q_empty_seq) # (B,)
    
    loss = -sum_score
    if old_device != device:
        loss = loss.to(old_device)
    
    return loss

# TODO generalize LM usage to any order
def sum_loss2(
    *,
    log_probs: torch.Tensor, # (T, B, V)
    log_lm_probs: torch.Tensor, # (V, V)
    log_prior: torch.Tensor | None, # (V,)
    input_lengths: torch.Tensor, # (B,)
    top_k: int = 0,
    LM_order: int,
    am_scale: float,
    lm_scale: float,
    prior_scale: float,
    horizontal_prior: bool,
    blank_prior: bool = True,
    blank_idx:int = 0, # should be same as eos index
    eos_idx: int | None = None,
    unk_idx: int = 1,
    log_zero: float = float("-inf"),
    device: torch.device = torch.device("cpu"),
    print_best_path_for_idx: list[int] = [],
    alignment_topk: bool = False,
    blank_correction_version = 0
):
    """
    Sum criterion training for CTC, given by
    L = sum_{all seq} q(seq)
    where q(seq) = p_AM^alpha(seq) * p_LM^beta(seq) / p_PR(seq)
    and p_AM = prod_n posterior.

    The loss is calculated by sum_{u in V} [Q(T, u, blank) + Q(T, u, non-blank)],
    where Q(t, u, {N or B}) is the sum of partial CTC alignments up to timeframe t
    with u being the last emitted label, and the last emitted frame is non-blank or blank.

    This Q is calculated by the two recursions (NOTE: the prior is used in all transitions):
    Q(t, u, blank) = [p_AM(blank | x_t) / p_PR(blank)] * [Q(t-1, u, blank) + Q(t-1, u, non-blank)]
    Q(t, u, non-blank) = [p_AM(u|x_t) / p_PR(u)] * [horizontal + diagonal + skip], where
    horizontal = Q(t-1, u, non-blank)
    diagonal = sum_v Q(t-1, v, blank) * p_LM(u|v)
    skip = sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w)
    
    Alternative calculation (NOTE: the prior is only used when the LM is used):
    Q(t, u, blank) = p_AM(blank | x_t) * [Q(t-1, u, blank) + Q(t-1, u, non-blank)]
    Q(t, u, non-blank) = p_AM(u|x_t) * [horizontal + diagonal + skip], where
    horizontal = Q(t-1, u, non-blank)
    diagonal = sum_v Q(t-1, v, blank) * p_LM(u|v) / p_PR(u)
    skip = sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w) / p_PR(u)

    Initialization:
    Q(1, u, blank) = 0
    Q(1, u, non-blank) = p_AM(u | x1) * p_LM(u | bos) / p_PR(u)

    NOTE: In case there is EOS in the vocab, its ctc posterior should be very small,
    because the transitions in the sum do not consider EOS.

    :param log_probs: log CTC output probs (T, B, V)
    :param log_lm_probs: (V, V), then log bigram LM probs of all possible context
    :param log_prior: vocab log prior probs (V,)
    :param input_lengths: Input lengths (B,)
    :param LM_order: Order of the LM
    :param am_scale: AM scale
    :param lm_scale: LM scale
    :param blank_idx: Blank index in V dim
    :param eos_idx: EOS idx in the vocab. None if no EOS in log_probs vocab.
        If None, then blank_idx in log_lm_probs should be EOS
    :param unk_idx: Unknown index in V dim
    :param log_zero: Value of log zero.
    :returns: log sum loss
    """
    use_prior = log_prior is not None
    
    old_device = log_probs.device
    log_probs = log_probs.to(device)
    log_lm_probs = log_lm_probs.to(device)
    if use_prior:
        log_prior = log_prior.to(device)
    input_lengths = input_lengths.to(device)
    
    max_audio_time, batch_size, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs = am_scale * log_probs
    if lm_scale == 0.0:
        log_lm_probs = torch.zeros_like(log_lm_probs, device=device)
    else:
        log_lm_probs = lm_scale * log_lm_probs
    if use_prior:
        log_prior = prior_scale * log_prior
    
    # print_gradients = PrintGradients.apply
    # grad_assert = AssertGradients.apply
    
    if eos_idx is None or eos_idx == blank_idx: # vocab means no EOS and blank
        vocab_size = n_out - 1
        eos_symbol = blank_idx
        assert eos_idx == 0, "EOS should be the first symbol"
    else:
        vocab_size = n_out - 2
        eos_symbol = eos_idx
        assert eos_idx == 0, "EOS should be the first symbol"
        assert blank_idx == n_out - 1, "blank should be the last symbol"
    assert unk_idx is not None, "unk_idx should be defined"
    vocab_size -= 1 # remove unk from vocab size
    # BoS / EoS and UNK are in the LM
    assert log_lm_probs.size() == (vocab_size + 2,) * LM_order, f"LM shape is not correct, should be {vocab_size + 2} in all dimensions but is {log_lm_probs.size()}"
    if use_prior:
        assert log_prior.size() == (n_out + 1,) or log_prior.size() == (n_out,), f"Prior shape is not correct, should be {n_out} or {n_out + 1} but is {log_prior.size()}"
        assert horizontal_prior, "Not using the horizontal prior is not implemented"

    # Unbind the log_probs for each timestep so it is faster during backprop
    log_probs = log_probs.unbind(0)

    # Used to remove blank, unk and eos from the last dim (vocab)
    out_idx = torch.arange(n_out, device=device)
    out_idx_vocab = out_idx[out_idx != blank_idx].long() # "vocab" means no EOS, unk and blank
    out_idx_vocab = out_idx_vocab[out_idx_vocab != unk_idx]
    out_idx_vocab_w_eos = out_idx[out_idx != unk_idx].long()
    if eos_idx is not None and eos_idx != blank_idx:
        out_idx_vocab = out_idx_vocab[out_idx_vocab != eos_idx]
        out_idx_vocab_w_eos = out_idx_vocab_w_eos[out_idx_vocab_w_eos != blank_idx]

    # sum score by DP
    # log_q is the list in which we store the log Q values as tensors of the last N timesteps
    # It is split into ending on a blank and ending on a non-blank
    
    # Init Q for t=0
    # Q(0, u, blank) = 0
    # Q(0, u, non-blank) = p_AM(u | x1) * p_LM(u | bos) / p_PR(u)
    log_q_label_init = log_probs[0][:, out_idx_vocab] + log_lm_probs[*[eos_symbol] * (LM_order - 1), out_idx_vocab].unsqueeze(0)
    if use_prior:
        log_q_label_init = log_q_label_init - log_prior[out_idx_vocab].unsqueeze(0)
    # We have to prepend the BoS symbol even though it is not used in the q_label calculation, but it is need for higher order LMs
    log_q_label_init = torch.cat([torch.full((batch_size, 1), log_zero, device=device), log_q_label_init], dim=1)
    log_q_label = torch.full((batch_size, *(vocab_size + 1,) * (LM_order - 1)), log_zero, device=device) # (B, V, ..., V), no blank in vocab dims
    log_q_label[:, *(0,) * (LM_order - 2)] = log_q_label_init
    log_q_blank = torch.full_like(log_q_label, log_zero, device=device) # (B, V, ..., V)
    # Calculate partial empty sequence score
    # Empty am score = sum log prob blank, lm score = p_LM(eos | bos), prior score = p_PR(eos)
    log_partial_empty_seq_prob = log_probs[0][:, blank_idx]
    if use_prior and blank_prior:
        log_partial_empty_seq_prob = log_partial_empty_seq_prob - log_prior[blank_idx]
    log_q_blank[:, *(0,) * (LM_order - 1)] = log_partial_empty_seq_prob
    log_q = safe_logaddexp(log_q_label, log_q_blank)
    original_shape = log_q.shape
    
    # Calculate initial top k if needed
    if top_k > 0:
        log_q_label = log_q_label.view(batch_size, -1)
        log_q_blank = log_q_blank.view(batch_size, -1)
        log_q = log_q.view(batch_size, -1)
        
        if alignment_topk:
            raise NotImplementedError("Alignment topk is not implemented yet")
            tmp_log_q = torch.stack([log_q_label, log_q_blank], dim=-1).view(batch_size, -1)
        else:
            tmp_log_q = log_q
        topk_scores, topk_idx = torch.topk(tmp_log_q, top_k, dim=1, sorted=False) # TODO replace log_q with topk_scores
        topk_idx = torch.tensor(np.array([np.unravel_index(topk_idx[b].cpu().numpy(), (log_q.size(1), 2)) for b in range(batch_size)]), device=device).transpose(1,2) if alignment_topk else topk_idx
        # print(topk_idx.shape)
        
        new_last_idx = torch.arange(vocab_size + 1, device=device)[None, None, None, :].expand(batch_size, top_k, 1, vocab_size + 1)
        new_idx = torch.tensor(np.array([np.unravel_index(topk_idx[b].cpu().numpy(), original_shape[1:]) for b in range(batch_size)]), device=device).transpose(1,2)[:, :, 1:].unsqueeze(-1).expand(-1, -1, -1, vocab_size + 1)
        new_idx = torch.cat([new_idx, new_last_idx], dim=2)
        new_idx = torch.tensor(np.array([[np.ravel_multi_index(new_idx[b, k].cpu().numpy(), original_shape[1:]) for k in range(top_k)] for b in range(batch_size)]), device=device)
        new_idx = new_idx.view(batch_size, -1)
        # print(topk_idx[print_best_path_for_idx[0]], topk_scores[print_best_path_for_idx[0]])
    
    # Set up the best path print
    if print_best_path_for_idx:
        with torch.no_grad():
            best_path_print = {}
            max_val, max_idx = torch.max(log_q.view(batch_size, -1), dim=-1)
            max_idx = torch.tensor([np.unravel_index(max_idx[b].cpu().numpy(), (vocab_size + 1,) * (LM_order - 1)) for b in range(batch_size)], device=device)
            for idx in print_best_path_for_idx:
                m_idx = max_idx[idx]
                m_idx[m_idx > 0] += 1
                a_idx = m_idx.clone()
                a_idx[a_idx == 0] = blank_idx
                best_path_print[idx] = {"str": f"{m_idx.tolist()}", "am_str": "{:.2f}".format(log_probs[0][idx][a_idx[-1]].tolist()), "prior": "{:.2f}".format(log_prior[a_idx[-1]].tolist() if use_prior else 0.0), "LM": "{:.2f}".format(log_lm_probs[*[eos_symbol] * (LM_order - 1), m_idx[-1]].tolist()), "score": "{:.2f}".format(max_val[idx].tolist()), "AM": log_probs[0][idx][a_idx[-1]].tolist()}
    
    # Prepare lm tensor for the diagonal transition
    log_lm_probs_wo_last_eos = dynamic_slice(log_lm_probs, [out_idx_vocab_w_eos] * (LM_order - 1) + [out_idx_vocab])
    log_lm_probs_wo_last_eos = torch.cat([torch.full((*log_lm_probs_wo_last_eos.size()[:-1], 1), log_zero, device=device), log_lm_probs_wo_last_eos], dim=-1) # EoS in last dimension is set to log_zero
    # Prepare lm tensor for the skip transition
    log_lm_probs_wo_diag = dynamic_slice(log_lm_probs, [out_idx_vocab_w_eos] * (LM_order - 2) + [out_idx_vocab] * 2)
    log_lm_probs_wo_diag = torch.cat([torch.full((*log_lm_probs_wo_diag.size()[:-2], 1, vocab_size), log_zero, device=device), log_lm_probs_wo_diag], dim=-2) # EoS in penultimate dimension is set to log_zero
    log_lm_probs_wo_diag = torch.cat([torch.full((*log_lm_probs_wo_diag.size()[:-1], 1), log_zero, device=device), log_lm_probs_wo_diag], dim=-1) # EoS in last dimension is set to log_zero
    log_lm_probs_wo_diag = (1 - torch.eye(vocab_size + 1, device=device)).log() + log_lm_probs_wo_diag # Fill diagonal in last two dimensions with log_zero as we don't allow repetitions of the same label here
    # Prepare lm tensor for EoS transition
    log_lm_probs_eos = dynamic_slice(log_lm_probs, [out_idx_vocab_w_eos] * (LM_order - 1) + [torch.tensor([eos_symbol], device=device)]).squeeze(-1).unsqueeze(0)
    if top_k > 0:
        log_lm_probs_eos[log_lm_probs_eos == float("-inf")] = -1000000.0 # Set to a very low value to avoid having -inf scores in the top k
    # Prepare prior
    if use_prior:
        log_prior_wo_bos = torch.cat([torch.full((1,), 0.0, device=device), log_prior[out_idx_vocab]], dim=0).unsqueeze(0)[:, *(None,) * (LM_order - 2), :].expand(original_shape)
        if top_k > 0:
            log_prior_wo_bos = log_prior_wo_bos.reshape(batch_size, -1)
    
    for t in range(1, max_audio_time):
        # case 1: emit a blank at t
        # Q(t, u, blank) = [Q(t-1, u, blank) + Q(t-1, u, non-blank)]*p_AM(blank | x_t)
        if top_k > 0:
            # Only consider blanks following top k sequences
            new_log_q_blank = torch.full_like(log_q, log_zero, device=device)
            if alignment_topk:
                label_topk_idx = topk_idx[topk_idx[:, :, -1] == 0][:-1]
                new_log_q_blank[label_topk_idx] = log_q_label[label_topk_idx] + log_probs[t][:, blank_idx][:, *(None,) * (LM_order - 1)]
                blank_topk_idx = topk_idx[topk_idx[-1] == 1][:-1]
                # We could already have entries from the topk labels, so we have to add
                new_log_q_blank[blank_topk_idx] = safe_logaddexp(new_log_q_blank[blank_topk_idx], log_q_blank[blank_topk_idx] + log_probs[t][:, blank_idx][:, *(None,) * (LM_order - 1)])
            else:
                new_log_q_blank.scatter_(1, topk_idx, log_q.gather(1, topk_idx) + log_probs[t][:, blank_idx].unsqueeze(-1))
        else:
            new_log_q_blank = log_q + log_probs[t][:, blank_idx][:, *(None,) * (LM_order - 1)]
        if use_prior and blank_prior:
            new_log_q_blank = new_log_q_blank - log_prior[blank_idx]

        # case 2: emit a non-blank at t
        # Q(t, u, non-blank) = p_AM(u|x_t) * [horizontal + diagonal + skip]
        
        # horizontal transition Q(t-1, u, non-blank)
        if top_k > 0:
            new_log_q_label = torch.full_like(log_q, log_zero, device=device)
            if alignment_topk:
                log_mass_horizontal[label_topk_idx] = log_q_label[label_topk_idx] # TODO maybe we need gather instead
            else:
                new_log_q_label.scatter_(1, topk_idx, log_q_label.gather(1, topk_idx))
        else:
            log_mass_horizontal = log_q_label
        
        # diagonal transition sum_v Q(t-1, v, blank) * p_LM(u|v) / p_PR(u)
        # take batch index b into account, this is equivalent to compute
        # mass_diagonal[b, u] = sum_v Q(t-1, b, blank, v) * p_LM(u|v) / p_PR(u)
        # mass_diagonal = Q(t-1, :, blank, :) @ M / p_PR(u), where M(v,u) = p_LM(u|v) = lm_probs[v][u]
        if top_k > 0:
            log_lm_probs_topk = log_lm_probs_wo_last_eos.view(-1, vocab_size + 1).unsqueeze(0).expand(batch_size, -1, -1) # (B, V, V)
            if alignment_topk:
                log_q_blank_topk = log_q_blank.gather(1, blank_topk_idx) # (B, K)
                log_lm_probs_topk = log_lm_probs_topk.gather(1, blank_topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size + 1)) # (B, K, V-1)
            else:
                log_q_blank_topk = log_q_blank.gather(1, topk_idx) # (B, K)
                log_lm_probs_topk = log_lm_probs_topk.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size + 1)) # (B, K, V)
            log_q_blank_topk = log_q_blank_topk.unsqueeze(-1).expand_as(log_lm_probs_topk)
            log_mass_diagonal_add = log_q_blank_topk + log_lm_probs_topk # (B, K, V)
            log_mass_diagonal_add = log_mass_diagonal_add.view(batch_size, -1)
            new_log_q_label = scatter_safe_logsumexp(new_log_q_label, 1, new_idx, log_mass_diagonal_add, include_self=True)
        else:
            log_mass_diagonal = log_matmul(log_q_blank, log_lm_probs_wo_last_eos) # (B, V) @ (V, V) -> (B, V)
        
        # skip transition sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w) / p_PR(u)
        if top_k > 0:
            log_lm_probs_topk = log_lm_probs_wo_diag.view(-1, vocab_size + 1).unsqueeze(0).expand(batch_size, -1, -1) # (B, V1, ..., Vm)
            if alignment_topk:
                log_q_label_topk = log_q_label.gather(1, label_topk_idx) # (B, K)
                log_lm_probs_topk = log_lm_probs_topk.gather(1, label_topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size + 1)) # (B, K, V-1)
            else:
                log_q_label_topk = log_q_label.gather(1, topk_idx) # (B, K)
                log_lm_probs_topk = log_lm_probs_topk.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size + 1)) # (B, K, V)
            log_q_label_topk = log_q_label_topk.unsqueeze(-1).expand_as(log_lm_probs_topk)
            log_mass_skip_add = log_q_label_topk + log_lm_probs_topk # (B, K, V)
            log_mass_skip_add = log_mass_skip_add.view(batch_size, -1)
            new_log_q_label = scatter_safe_logsumexp(new_log_q_label, 1, new_idx, log_mass_skip_add, include_self=True)
        else:
            log_mass_skip = log_matmul(log_q_label, log_lm_probs_wo_diag) # (B, V) @ (V, V) -> (B, V)
        
        # add up the three transition types
        if top_k <= 0:
            new_log_q_label = safe_logsumexp(torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1), dim=-1)
        
        # correct the prior
        if use_prior:
            new_log_q_label -= log_prior_wo_bos
        
        # multiply with p_AM(u|x_t)
        if top_k > 0:
            new_log_q_label = new_log_q_label + torch.cat([torch.full((batch_size, 1), log_zero, device=device), log_probs[t][:, out_idx_vocab]], dim=-1)[:, *(None,) * (LM_order - 2), :].expand(original_shape).reshape(batch_size, -1)
        else:
            new_log_q_label = new_log_q_label + torch.cat([torch.full((batch_size, 1), log_zero, device=device), log_probs[t][:, out_idx_vocab]], dim=-1)[:, *(None,) * (LM_order - 2), :]
        
        # set masked results to log_q
        time_mask = (t < input_lengths)[:, *(None,) * (LM_order - 1)] if top_k <= 0 else (t < input_lengths).unsqueeze(-1)
        log_q_blank = torch.where(time_mask.expand_as(log_q), new_log_q_blank, log_q_blank)
        log_q_label = torch.where(time_mask.expand_as(log_q), new_log_q_label, log_q_label)
        log_q = safe_logaddexp(log_q_label, log_q_blank)
        
        assert torch.all(torch.isneginf(log_q_label[..., 0])), "There should be no probability for the BoS symbol in log_q_label"
        
        if top_k > 0:
            if alignment_topk:
                tmp_log_q = torch.stack([log_q_label, log_q_blank], dim=-1)
            else:
                tmp_log_q = log_q
            # If we are in the last timestep, we also have to add the EOS LM probability
            last_mask = (t == input_lengths - 1).unsqueeze(-1).expand_as(tmp_log_q)
            tmp_log_q = torch.where(last_mask, tmp_log_q + log_lm_probs_eos.view(1, -1).expand_as(tmp_log_q), tmp_log_q)
            # Calculate top k and apply time mask
            new_topk_scores, new_topk_idx = torch.topk(tmp_log_q, top_k, dim=1, sorted=False)
            new_topk_idx = torch.tensor(np.array([np.unravel_index(new_topk_idx[b].cpu().numpy(), (log_q.size(1), 2)) for b in range(batch_size)]), device=device).transpose(1,2) if alignment_topk else new_topk_idx
            topk_scores = torch.where(time_mask.expand_as(topk_scores), new_topk_scores, topk_scores)
            topk_idx = torch.where(time_mask.expand_as(topk_idx), new_topk_idx, topk_idx)

            new_last_idx = torch.arange(vocab_size + 1, device=device)[None, None, None, :].expand(batch_size, top_k, 1, vocab_size + 1)
            new_idx = torch.tensor(np.array([np.unravel_index(topk_idx[b].cpu().numpy(), original_shape[1:]) for b in range(batch_size)]), device=device).transpose(1,2)[:, :, 1:].unsqueeze(-1).expand(-1, -1, -1, vocab_size + 1)
            new_idx = torch.cat([new_idx, new_last_idx], dim=2)
            new_idx = torch.tensor(np.array([[np.ravel_multi_index(new_idx[b, k].cpu().numpy(), original_shape[1:]) for k in range(top_k)] for b in range(batch_size)]), device=device)
            new_idx = new_idx.view(batch_size, -1)
            # print(topk_idx[print_best_path_for_idx[0]], topk_scores[print_best_path_for_idx[0]])
            
        if print_best_path_for_idx:
            with torch.no_grad():
                max_val, max_idx = torch.max(log_q.view(batch_size, -1), dim=-1)
                max_idx = torch.tensor([np.unravel_index(max_idx[b].cpu().numpy(), (vocab_size + 1,) * (LM_order - 1)) for b in range(batch_size)], device=device)
                for idx in print_best_path_for_idx:
                    m_idx = max_idx[idx]
                    m_idx[m_idx > 0] += 1
                    a_idx = m_idx.clone()
                    a_idx[a_idx == 0] = blank_idx
                    
                    best_path_print[idx]["str"] += f" {m_idx.tolist()}"
                    best_path_print[idx]["am_str"] += " {:.2f}".format(log_probs[t][idx][a_idx[-1]].tolist())
                    best_path_print[idx]["prior"] += " {:.2f}".format(log_prior[a_idx[-1]].tolist() if use_prior else 0.0)
                    best_path_print[idx]["LM"] += " {:.2f}".format(safe_logsumexp(log_lm_probs_wo_last_eos[:, *max_idx[idx]], dim=-1).tolist() if lm_scale > 0.0 else 0.0)
                    best_path_print[idx]["score"] += " {:.2f}".format(max_val[idx].tolist()) #  / (t+1)
                    best_path_print[idx]["AM"] += log_probs[t][idx][a_idx[-1]].tolist()
        
        torch.cuda.empty_cache()
    
    if top_k > 0:
        sum_score = safe_logsumexp(topk_scores, dim=-1)
    else:
        # multiply last Q with p_LM(eos | u)
        log_q = log_q + log_lm_probs_eos
        # sum over the vocab dimensions
        sum_score = log_q
        for _ in range(LM_order - 1):
            sum_score = safe_logsumexp(sum_score, dim=-1)
    
    if print_best_path_for_idx:
        with torch.no_grad():
            for idx in print_best_path_for_idx:
                print(f"Best path for {idx}: {get_bpes(best_path_print[idx]['str'])}\nAM str: {best_path_print[idx]['am_str']}\nPrior: {best_path_print[idx]['prior']}\nLM: {best_path_print[idx]['LM']}\nScore: {best_path_print[idx]['score']}\nAM: {best_path_print[idx]['AM']}")
    
    loss = -sum_score
    if old_device != device:
        loss = loss.to(old_device)
    
    return loss

def sum_loss_approx(
    *,
    log_probs: torch.Tensor, # (T, B, V)
    log_lm_probs: torch.Tensor, # (V, V)
    log_prior: torch.Tensor | None, # (V,)
    input_lengths: torch.Tensor, # (B,)
    top_k: int = 0,
    LM_order: int,
    am_scale: float,
    lm_scale: float,
    prior_scale: float,
    horizontal_prior: bool,
    blank_prior: bool = True,
    blank_idx:int = 0, # should be same as eos index
    eos_idx: int | None = None,
    unk_idx: int = 1,
    log_zero: float = float("-inf"),
    device: torch.device = torch.device("cpu"),
    print_best_path_for_idx: list[int] = [],
):
    use_prior = log_prior is not None
    
    old_device = log_probs.device
    log_probs = log_probs.to(device)
    log_lm_probs = log_lm_probs.to(device)
    if use_prior:
        log_prior = log_prior.to(device)
    input_lengths = input_lengths.to(device)
    
    max_audio_time, batch_size, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs = am_scale * log_probs
    if lm_scale == 0.0:
        log_lm_probs = torch.zeros_like(log_lm_probs, device=device)
    else:
        log_lm_probs = lm_scale * log_lm_probs
    if use_prior:
        log_prior = prior_scale * log_prior
    
    # print_gradients = PrintGradients.apply
    # grad_assert = AssertGradients.apply

    # Unbind the log_probs and log_partial_empty_seq_prob for each timestep so it is faster during backprop
    log_probs = log_probs.unbind(0)

    # sum score by DP
    
    assert blank_idx == n_out - 1, "blank should be the last symbol" + str(blank_idx) + " " + str(n_out)
    assert log_prior.size(0) == n_out, "Prior shape is not correct"
    
    if print_best_path_for_idx:
        with torch.no_grad():
            best_path_print = {}
            for idx in print_best_path_for_idx:
                best_path_print[idx] = {"str": "", "am_str": "", "prior": "", "score": "", "AM": torch.full((1,), 0.0, device=device)}
    
    
    log_q = torch.zeros((batch_size,), device=device)
    for t in range(max_audio_time):
        if use_prior:
            tmp = log_probs[t] - log_prior.unsqueeze(0)
            max_val, max_idx = torch.max(tmp, dim=-1)
        else:
            max_val, max_idx = torch.max(log_probs[t], dim=-1)
        time_mask = (t < input_lengths)
        
        new_log_q = log_q + max_val
        log_q = torch.where(time_mask, new_log_q, log_q)
        
        if print_best_path_for_idx:
            with torch.no_grad():
                for idx in print_best_path_for_idx:
                    best_path_print[idx]["str"] += f" {max_idx[idx]}"
                    best_path_print[idx]["am_str"] += " {:.2f}".format(log_probs[t][idx][max_idx[idx]].tolist())
                    best_path_print[idx]["prior"] += " {:.2f}".format(log_prior[max_idx[idx]].tolist())
                    best_path_print[idx]["score"] += " {:.2f}".format(log_q[idx].tolist()) #  / (t+1)
                    best_path_print[idx]["AM"] += log_probs[t][idx][max_idx[idx]]
    
    if print_best_path_for_idx:
        with torch.no_grad():
            for idx in print_best_path_for_idx:
                print(f"Best path for {idx}: {best_path_print[idx]['str']}\nAM str: {best_path_print[idx]['am_str']}\nPrior: {best_path_print[idx]['prior']}\nScore: {best_path_print[idx]['score']}\nAM: {best_path_print[idx]['AM']}")
    
    loss = -log_q
    if old_device != device:
        loss = loss.to(old_device)
    
    return loss

# ------------------------------------------------
# Helper functions and classes

def dynamic_slice(tensor: torch.Tensor, indices_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Slices a tensor along multiple dimensions using the provided indices for each dimension.
    
    Args:
    tensor (torch.Tensor): The input tensor to be sliced.
    indices_list (list of list of int): A list containing tensors of indices for each dimension.
    
    Returns:
    torch.Tensor: The sliced tensor.
    """
    assert len(indices_list) == tensor.dim(), "Number of index lists must match the number of dimensions of the tensor"
    
    # Use advanced indexing to select the desired elements
    sliced_tensor = tensor
    for dim, indices in enumerate(indices_list):
        # Expand indices to match the dimensions of the tensor
        shape = [1] * tensor.dim()
        shape[dim] = -1
        new_sizes = list(sliced_tensor.size())
        new_sizes[dim] = -1
        indices = indices.view(shape).expand(new_sizes)
        sliced_tensor = torch.gather(sliced_tensor, dim, indices)
    
    return sliced_tensor

def safe_logsumexp(x: torch.Tensor, dim: int, *, keepdim: bool = False) -> torch.Tensor:
    """safe logsumexp, handles the case of -inf values. otherwise, when all are -inf, you get nan"""
    with torch.no_grad():
        max_x, _ = x.max(dim=dim, keepdim=True)
        max_x = max_x.detach()
        mask = max_x.isneginf()
        max_x_ = max_x if keepdim else max_x.squeeze(dim=dim)
        mask_ = mask if keepdim else mask.squeeze(dim=dim)
    sum = (x - max_x.masked_fill(mask, 0)).exp_().sum(dim=dim, keepdim=keepdim)
    sum.masked_fill_(mask_, 1).log_()
    sum += max_x_
    return sum

# def safe_logsumexp(x: torch.Tensor, dim: int, *, keepdim: bool = False) -> torch.Tensor:
#     """safe logsumexp, handles the case of -inf values. otherwise, when all are -inf, you get nan"""
#     with torch.no_grad():
#         max_x, _ = x.max(dim=dim, keepdim=True)
#         max_x = max_x.detach()
#         max_x_ = max_x if keepdim else max_x.squeeze(dim=dim)
#     diff = torch.where(max_x.isneginf(), 0.0, x - max_x)
#     return max_x_ + diff.exp().sum(dim=dim, keepdim=keepdim).log()

def safe_logaddexp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """safe logaddexp, handles the case of -inf values. otherwise, when all are -inf, you get nan"""
    with torch.no_grad():
        mask = x >= y
        inf_mask = torch.logical_and(
            torch.logical_not(torch.isfinite(x)), x == y
        )
    max_ = torch.where(mask, x, y)
    min_ = torch.where(mask, y, x)
    return max_.masked_fill_(inf_mask, float("-inf")) + torch.log1p(torch.exp(min_ - max_.masked_fill(inf_mask, 0)))

def scatter_safe_logsumexp(
    tensor: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor, *, include_self: bool = True
) -> torch.Tensor:
    """
    Like :func:`torch.scatter_reduce_` but doing safe_logsumexp as in :func:`safe_logsumpexp`.

    https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html

    As we are reducing, usually D_out < D_src.

    Note, there is also scatter_logsumexp
    (https://pytorch-scatter.readthedocs.io/en/1.4.0/_modules/torch_scatter/logsumexp.html)
    but this does not have the "safe" aspect as in :func:`safe_logsumexp`.

    :param tensor: output before we scatter. e.g. [D_out,...]
    :param dim: dim in output, e.g. dim=0
    :param index: indices in dim in output. e.g. [D_src]->D_out
    :param src: for each index, the value to scatter into output, e.g. [D_src,...]
    :param include_self: if True, also include the self value in the reduce.
    :return: tensor [D_out,...] with the scattered updates
    """
    with torch.no_grad():
        max_x = tensor.scatter_reduce(
            dim=dim, index=index, src=src, reduce="amax", include_self=include_self
        )  # [D_out,...]
        max_x_ = max_x.gather(dim=dim, index=index)  # [D_src,...]
        max_x = max_x.detach()
        max_x_ = max_x_.detach()
        mask = max_x.isneginf()
        mask_ = max_x_.isneginf()
    src_ = (src - max_x_.masked_fill_(mask_, 0)).exp_()
    tensor = (tensor - max_x.masked_fill(mask, 0)).exp_()
    scat_sum = tensor.scatter_reduce(dim=dim, index=index, src=src_, reduce="sum", include_self=include_self)
    scat_sum = scat_sum.masked_fill(mask, 1).log_()
    scat_sum += max_x
    return scat_sum

# def log_matmul_alt(A: torch.Tensor, B: torch.Tensor):
#     with torch.no_grad():
#         max_A, _ = torch.max(A, dim=1, keepdim=True)
#         max_B, _ = torch.max(B, dim=0, keepdim=True)
#         mask_A = max_A.isneginf()
#         mask_B = max_B.isneginf()
#         mask = mask_A + mask_B > 0
#     m1 = (A - max_A.masked_fill(mask_A, 0.0)).exp()
#     m2 = (B - max_B.masked_fill(mask_B, 0.0)).exp()
#     mul = m1.matmul(m2)
#     mul_masked = mul.masked_fill_(mask, 1)
#     log_mul = mul_masked.masked_fill_(mul_masked == 0.0, 0.0).log()
#     return max_A + max_B + log_mul

def log_matmul(A: torch.Tensor, B: torch.Tensor, batch_given: bool = False):
    """
    This is inefficient

    Log matrix multiplication, i.e.
    A = log X, B = log Y
    -> log_matmul(A, B) = log (X @ Y)
    https://stackoverflow.com/questions/36467022/handling-matrix-multiplication-in-log-space-in-python

    :param A: first matrix in log scale (b, v)
    :param B: second matrix in log scale (v, v2)
    :param batch_given: whether we habe the batch dimension in B given
    :returns: matrix product of the two matrices in log scale
    """
    if batch_given:
        B_expand = B # (b, v, v2)
    else:
        b = A.size(0)
        B_expand = B.unsqueeze(0).expand(b, *B.size()) # (v, v2) -> (b, v, v2)
    A_expand = A.unsqueeze(-1).expand_as(B_expand) # (b, v) -> (b, v, v2)
    
    return safe_logsumexp((A_expand + B_expand), dim=1) # (b, v, v2) -> (b, v2)

def get_bpes(tokens):
    # MONICA DREW FRESH HOPE FROM HER SON'S WRITINGS THEY WERE FULL OF NOBLE THOUGHTS AND HIGH ASPIRATIONS
    
    # path = "/u/marten.mueller/dev/ctc_baseline/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.P1DXd9G7EdsU/output/bpe.vocab"
    # d = eval(open(path, "r").read(), {"nan": float("nan"), "inf": float("inf")})
    # d = {v: k for k, v in d.items()}
    d = {0: '<s>', 1: '<unk>', 2: 'T@@', 3: 'THE', 4: 'C@@', 5: 'E@@', 6: 'M@@', 7: 'P@@', 8: 'I@@', 9: 'W@@', 10: 'S@@', 11: 'A@@', 12: 'D@@', 13: 'F@@', 14: 'G@@', 15: 'U@@', 16: 'ED', 17: 'O@@', 18: 'S', 19: 'E', 20: 'AND', 21: 'L@@', 22: 'Y', 23: 'OF', 24: 'TO', 25: 'IN@@', 26: 'RE@@', 27: 'TH@@', 28: 'B@@', 29: 'AR@@', 30: 'ING', 31: 'A', 32: 'T', 33: 'ER@@', 34: 'R@@', 35: 'AN@@', 36: 'H@@', 37: 'ST@@', 38: 'IN', 39: 'OU@@', 40: 'V@@', 41: 'D', 42: 'ON', 43: 'N@@', 44: 'K@@', 45: 'Y@@', 46: 'EN', 47: 'OR@@', 48: 'ER', 49: 'EL@@', 50: 'L', 51: 'EN@@', 52: 'ON@@', 53: 'RO@@', 54: 'ES', 55: 'IT@@', 56: 'I', 57: 'M', 58: 'R', 59: 'WAS', 60: 'HE', 61: 'ME', 62: 'AT@@', 63: 'LY', 64: 'IT', 65: 'THAT', 66: 'O', 67: 'AL@@', 68: 'AC@@', 69: 'HA@@', 70: 'BE@@', 71: 'AN', 72: 'ST', 73: 'IS', 74: 'H', 75: 'IS@@', 76: 'W', 77: 'LE', 78: 'LE@@', 79: 'K', 80: 'TI@@', 81: 'ERE', 82: 'LI@@', 83: 'HIS', 84: 'RI@@', 85: 'SI@@', 86: 'WH@@', 87: 'UR@@', 88: 'LO@@', 89: 'SE', 90: 'AT', 91: 'AS', 92: 'SA@@', 93: 'CH', 94: 'CO@@', 95: 'HAD', 96: 'THE@@', 97: 'WITH', 98: 'SE@@', 99: 'IL@@', 100: 'UN@@', 101: 'YOU', 102: 'CE', 103: 'FOR', 104: 'F', 105: 'NE@@', 106: 'AS@@', 107: 'DI@@', 108: 'HER', 109: 'DE@@', 110: 'SU@@', 111: 'N', 112: 'MA@@', 113: 'NO@@', 114: 'NOT', 115: 'LA@@', 116: 'HO@@', 117: 'BUT', 118: 'ENT', 119: 'CA@@', 120: 'OR', 121: 'OULD', 122: 'RA@@', 123: 'GHT', 124: 'WHI@@', 125: 'PO@@', 126: 'VE', 127: 'P', 128: 'J@@', 129: 'VER@@', 130: 'SHE', 131: 'SO@@', 132: 'ONE', 133: 'IR@@', 134: 'AB@@', 135: 'THER', 136: 'X@@', 137: 'BE', 138: 'OUN@@', 139: 'HE@@', 140: 'ALL', 141: 'CON@@', 142: 'HI@@', 143: 'PE@@', 144: "'S", 145: 'OUT', 146: 'HIM', 147: 'MO@@', 148: 'FOR@@', 149: 'ID', 150: 'VER', 151: 'DO@@', 152: 'TO@@', 153: 'MY', 154: "'@@", 155: 'ME@@', 156: 'THEY', 157: 'BY', 158: 'SS', 159: 'ENT@@', 160: 'KE', 161: 'G', 162: 'ATI@@', 163: 'WA@@', 164: 'HAVE', 165: 'MP@@', 166: 'AL', 167: 'SO', 168: 'Q@@', 169: 'LD', 170: 'GH@@', 171: 'Z@@', 172: 'BU@@', 173: 'C', 174: 'X', 175: 'B', 176: 'OU', 177: 'WIT@@', 178: 'U', 179: 'Z', 180: 'V', 181: 'Q', 182: 'J', 183: "'"}
    
    tokens = [t for t in tokens.split("] [")]
    tokens[0] = tokens[0][1:]
    if len(tokens) > 1:
        tokens[-1] = tokens[-1][:-1]
    if "," in tokens[0]:
        tokens = [t.split(", ")[-1] for t in tokens]
    tokens = [d[int(t)] for t in tokens]
    tokens = " ".join(tokens)
    tokens = tokens.replace("@@ ", "")
    return tokens

class PrintGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name, print_input, mean_dim = None):
        ctx.name = name
        ctx.print_input = print_input
        ctx.mean_dim = mean_dim
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        name = ctx.name
        print_input = ctx.print_input
        x, = ctx.saved_tensors
        if print_input:
            print(f"Gradients ({name}): {grad_output.mean(dim=ctx.mean_dim) if ctx.mean_dim is not None else grad_output} {grad_output.sum()}\nInput: {x}\nNaN's: {torch.isnan(grad_output).sum()}")
        else:
            print(f"Gradients ({name}): {grad_output.mean(dim=ctx.mean_dim) if ctx.mean_dim is not None else grad_output} {grad_output.sum()}\nNaN's: {torch.isnan(grad_output).sum()}")
        return grad_output, None, None, None
    
class AssertGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name, print_input):
        ctx.name = name
        ctx.print_input = print_input
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        name = ctx.name
        print_input = ctx.print_input
        x, = ctx.saved_tensors
        if print_input:
            assert not torch.isnan(grad_output).any(), f"{torch.isnan(grad_output).sum()} NaN's in gradients of {name}, see {grad_output}\nand input {x}"
        else:
            assert not torch.isnan(grad_output).any(), f"{torch.isnan(grad_output).sum()} NaN's in gradients of {name}, see {grad_output}"
        return grad_output, None, None

# ------------------------------------------------------------

def test_logsumexp():
    # torch.autograd.set_detect_anomaly(True)
    
    ag = PrintGradients.apply
    
    # t = torch.tensor([float("-inf"), float("-inf"), float("-inf")], requires_grad=True)
    # t = torch.tensor([float("-inf"), float("-0.5"), float("-1")], requires_grad=True)
    # t = ag(t, "t", False)
    
    # sum_t = torch.logsumexp(t, dim=0)
    # sum_t.backward()
    # print(sum_t)
    
    # sum_t2 = safe_logsumexp(t, dim=0)
    # sum_t2.backward()
    # print(sum_t2)
    
    t2 = torch.full((5,), float("-inf"), requires_grad=True)
    idx = torch.tensor([0, 1, 0, 1, 2])
    src = torch.tensor([0.4, 0.0, 0.2, 0.0, 0.5], requires_grad=True).log()
    t2 = ag(t2, "t2", False)
    src = ag(src, "src", False)
    
    scat = scatter_safe_logsumexp(t2, 0, idx, src)
    scat.backward(torch.ones_like(scat))
    print(scat.exp())

def test_mul():
    # torch.autograd.set_detect_anomaly(True)
    
    import time
    import gc
    torch.manual_seed(0)
    ag = PrintGradients.apply
    
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    
    r = torch.tensor([0.0], device=device)
    time_d = 0.0
    for i in range(1):
        s = time.time()
        # A = torch.randn(10, 185, device=device)
        A = torch.randn(10, 185, 185, device=device)
        # B = torch.randn(1000, 1500, device=device).log_softmax(dim=0)
        # B = torch.randn(185, 185, device=device).log_softmax(dim=1)
        B = torch.randn(185, 185, 185, device=device).log_softmax(dim=1)
        # A[0] = float("-inf")
        # B[:, 0] = float("-inf")
        A.requires_grad = True
        B.requires_grad = True
        
        A = ag(A, "A", False)
        B = ag(B, "B", False)
        
        # res = A.exp().matmul(B.exp()).log()
        # res = log_matmul_alt(A, B)
        # A, topk_idx = torch.topk(A, 20, dim=-1)
        # B = B.unsqueeze(0).expand(10, -1, -1)
        # B = B.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 185))
        # res = log_matmul(A, B, batch_given=True)
        res = log_matmul(A, B)
        res = res.exp().sum()
        
        
        res.backward(torch.ones_like(res))
        e = time.time()
        r += res.detach()
        time_d += e-s
        torch.cuda.empty_cache()
    print(f"This took {time.strftime('%H:%M:%S', time.gmtime(time_d))}: {r}")

def test_profiler():
    import time
    from torch.profiler import profile, record_function, ProfilerActivity
    
    # torch.cuda.set_sync_debug_mode(1)
    
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    
    batch_size = 12 # 14000 per GPU, 1250 stpes a 12 seqs (0.7 sec/step)
    vocab_size = 185
    frames = 100
    
    torch.manual_seed(0)
    ag = AssertGradients.apply
    
    lm = torch.randn(vocab_size - 1, vocab_size - 1, device=device)
    lm = torch.nn.functional.log_softmax(lm, dim=-1)
    
    s1 = time.time()
                
    am = torch.randn(frames, batch_size, vocab_size, requires_grad=True, device=device)
    am = torch.nn.functional.log_softmax(am, dim=-1)
    
    prior = torch.randn(vocab_size + 1, requires_grad=True, device=device)
    prior = torch.nn.functional.log_softmax(prior, dim=-1)
    
    length = torch.full((batch_size,), frames, device=device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        with record_function("loss"):
            # am = am.permute(1, 0, 2)
            # prior = _calc_log_prior(am, length)
            # am = am.permute(1, 0, 2)
            
            # am = ag(am, "AM", False)
            # prior = ag(prior, "prior", False)
            
            loss = sum_loss(
                log_probs=am,
                log_lm_probs=lm,
                log_prior=prior,
                input_lengths=length,
                LM_order=2,
                am_scale=1.0,
                lm_scale=1.0,
                blank_idx=184,
                eos_idx=0,
            )
            loss.backward(torch.ones_like(loss, device=device))
    e1 = time.time()
    print(f"Sum loss took {time.strftime('%H:%M:%S', time.gmtime(e1-s1))}: {loss}") # 5:00 mins
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

def test():
    import time
    
    # torch.autograd.set_detect_anomaly(True)
    # torch.cuda.set_sync_debug_mode(1)
    
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    
    batch_size = 12 # 14000 per GPU, 1250 stpes a 12 seqs (0.7 sec/step)
    vocab_size = 185
    frames = 100
    LM_order = 2
    
    torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    ag = AssertGradients.apply
    # ag = PrintGradients.apply
    
    lm = torch.randn((vocab_size - 1,) * LM_order, device=device)
    lm = torch.nn.functional.log_softmax(lm, dim=-1)
    
    l = torch.tensor([0.0], device=device)
    s1 = time.time()
    for i in range(1):
        s = time.time()
        am = torch.randn(frames, batch_size, vocab_size, device=device)
        am[0, 0, vocab_size - 1] = float(3)
        am[2, 0, vocab_size - 1] = float(3)
        am.requires_grad = True
        am = torch.nn.functional.log_softmax(am, dim=-1)
        
        prior = torch.randn(vocab_size + 1, requires_grad=True, device=device)
        prior = torch.nn.functional.log_softmax(prior, dim=-1)
        
        length = torch.full((batch_size,), frames, device=device)
        # length[0] -= 3

        # am = am.permute(1, 0, 2)
        # prior = _calc_log_prior(am, length, use_max=True)
        # am = am.permute(1, 0, 2)
        
        am = ag(am, "AM", False)
        prior = ag(prior, "prior", False)
        
        # TODO check gradient
        loss = sum_loss2(
            log_probs=am,
            log_lm_probs=lm,
            log_prior=prior,
            input_lengths=length,
            top_k=10,
            LM_order=LM_order,
            am_scale=1.0,
            lm_scale=1.0,
            prior_scale=1.0,
            horizontal_prior=True,
            blank_prior=True,
            blank_idx=184,
            eos_idx=0,
            print_best_path_for_idx=[0],
            alignment_topk=False,
        )
        print("OUT", loss[0].tolist())
        l += (loss / frames).mean()
        
        # del loss, am, prior
        # torch.cuda.empty_cache()
        print("Time:", time.time() - s)
        
        # targets = torch.tensor([55, 148, 178, 108, 179, 126, 110, 103, 9, 154, 84, 162, 159, 83, 153, 33, 106, 9, 131, 46, 63, 15, 162, 94, 0, 111, 121, 29, 121, 21, 151, 18, 4, 159, 118, 86, 129, 18, 13, 170, 151, 81, 77, 53, 165, 57, 134, 63, 103, 110, 47, 35, 145, 18, 34, 66, 42, 96, 139, 16, 138, 156, 1, 63, 103, 95, 149, 111, 83, 34, 113, 158, 39, 166, 34, 123, 26, 148, 134, 148, 168, 177, 18, 23, 164, 69, 145, 93, 166, 174, 162, 36, 95, 116, 123, 74, 124, 70])
        # targets = targets + 2
        targets = torch.tensor(
            [150, 110, 107, 128, 112, 105,  11, 156,  86, 164, 161,  85,
            155,  35, 108,  11, 133,  48, 133,  17, 164,  96,   2, 113, 123,  31,
            123,  23, 153,  20,   6, 161, 120,  88, 131,  20,  15,  99, 153,  58,
            119,   1,  88,  59, 136,  65, 105,  99, 122,  37, 147,  20,  36,  68,
            44,  98, 141,  18,   1, 158,   3,  65, 105,  97, 151, 113,  85,  36,
            115, 160,  83, 168,  36, 125,  28, 150, 136,  90, 170, 179,
            20,  25, 166,  71, 147,  95, 168, 176, 164,  38,  97, 118, 125,  76,
            43,  72]
        )
        # greedy_probs, greedy_idx = torch.max(am[:, 0:1], dim=-1)
        # print(greedy_idx.squeeze(-1))
        targets = targets.unsqueeze(0)
        target_lengths = torch.tensor([targets.size(1)])
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=am[:, 0:1],
            targets=targets,
            input_lengths=length[0:1],
            target_lengths=target_lengths,
            blank=184,
            reduction="none"
        )
        print(ctc_loss)
        
        
    l.backward(torch.ones_like(l, device=device))
    e1 = time.time()
    # print(f"Sum loss took {time.strftime('%H:%M:%S', time.gmtime(e1-s1))}: {l}") # 5:00 mins
    
    # s2 = time.time()
    
    # targets = torch.randint(1, vocab_size, (batch_size, 80))
    # target_lengths = torch.full((batch_size,), 80)
    # ctc_loss = torch.nn.functional.ctc_loss(
    #     log_probs=am,
    #     targets=targets,
    #     input_lengths=length,
    #     target_lengths=target_lengths,
    #     blank=0,
    #     reduction="none"
    # )
    
    # e2 = time.time()
    # print(f"CTC loss took {time.strftime('%H:%M:%S', time.gmtime(e2 - s2))}: {(ctc_loss / frames).mean()}") # 0:08 mins

def test_LM():
    # with open("/u/marten.mueller/dev/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.S9n2YtP1JzJ5/output/lm.pt", "rb") as f:
    with open("/u/marten.mueller/dev/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.wuVkNuDg8B55/output/lm.pt", "rb") as f:
        t = torch.load(f)
    print(t[3])
    
def test_get_bpes():
    tokens = "[117] [117] [117] [117] [117] [117] [117] [46] [46] [46] [46] [21] [35] [35] [55] [55] [120] [120] [120] [26] [26] [76] [13] [26] [10] [10] [74] [116] [7] [7] [19] [13] [53] [57] [57] [108] [108] [10] [10] [10] [42] [42] [24] [24] [34] [34] [34] [55] [25] [14] [14] [18] [18] [18] [18] [18] [156] [156] [9] [81] [81] [13] [15] [21] [50] [50] [23] [23] [113] [113] [113] [28] [77] [77] [27] [27] [39] [170] [2] [18] [18] [20] [20] [142] [142] [142] [170] [170] [170] [106] [106] [7] [7] [33] [33] [162] [162] [162] [52] [52] [18] [18] [18] [18] [18] [18] [18] [18] [18] [18] [18] [18] [18]"
    print(get_bpes(tokens))

if __name__ == "__main__":
    test()