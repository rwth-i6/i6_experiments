"""
Implement lattice-free MMI training for CTC
"""

import torch

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
# def sum_loss2(
#     *,
#     log_probs: torch.Tensor, # (T, B, V)
#     log_lm_probs: torch.Tensor, # (V, V)
#     log_prior: torch.Tensor | None, # (V,)
#     input_lengths: torch.Tensor, # (B,)
#     LM_order: int,
#     am_scale: float,
#     lm_scale: float,
#     prior_scale: float,
#     horizontal_prior: bool,
#     blank_idx:int = 0, # should be same as eos index
#     eos_idx: int | None = None,
#     unk_idx: int = 1,
#     log_zero: float = float("-inf"),
#     device: torch.device = torch.device("cpu"),
# ):
#     """
#     Sum criterion training for CTC, given by
#     L = sum_{all seq} q(seq)
#     where q(seq) = p_AM^alpha(seq) * p_LM^beta(seq) / p_PR(seq)
#     and p_AM = prod_n posterior.

#     The loss is calculated by sum_{u in V} [Q(T, u, blank) + Q(T, u, non-blank)],
#     where Q(t, u, {N or B}) is the sum of partial CTC alignments up to timeframe t
#     with u being the last emitted label, and the last emitted frame is non-blank or blank.

#     This Q is calculated by the two recursions (NOTE: the prior is used in all transitions):
#     Q(t, u, blank) = [p_AM(blank | x_t) / p_PR(blank)] * [Q(t-1, u, blank) + Q(t-1, u, non-blank)]
#     Q(t, u, non-blank) = [p_AM(u|x_t) / p_PR(u)] * [horizontal + diagonal + skip], where
#     horizontal = Q(t-1, u, non-blank)
#     diagonal = sum_v Q(t-1, v, blank) * p_LM(u|v)
#     skip = sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w)
    
#     Alternative calculation (NOTE: the prior is only used when the LM is used):
#     Q(t, u, blank) = p_AM(blank | x_t) * [Q(t-1, u, blank) + Q(t-1, u, non-blank)]
#     Q(t, u, non-blank) = p_AM(u|x_t) * [horizontal + diagonal + skip], where
#     horizontal = Q(t-1, u, non-blank)
#     diagonal = sum_v Q(t-1, v, blank) * p_LM(u|v) / p_PR(u)
#     skip = sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w) / p_PR(u)

#     Initialization:
#     Q(1, u, blank) = 0
#     Q(1, u, non-blank) = p_AM(u | x1) * p_LM(u | bos) / p_PR(u)

#     NOTE: In case there is EOS in the vocab, its ctc posterior should be very small,
#     because the transitions in the sum do not consider EOS.

#     :param log_probs: log CTC output probs (T, B, V)
#     :param log_lm_probs: (V, V), then log bigram LM probs of all possible context
#     :param log_prior: vocab log prior probs (V,)
#     :param input_lengths: Input lengths (B,)
#     :param LM_order: Order of the LM
#     :param am_scale: AM scale
#     :param lm_scale: LM scale
#     :param blank_idx: Blank index in V dim
#     :param eos_idx: EOS idx in the vocab. None if no EOS in log_probs vocab.
#         If None, then blank_idx in log_lm_probs should be EOS
#     :param unk_idx: Unknown index in V dim
#     :param log_zero: Value of log zero.
#     :returns: log sum loss
#     """
#     use_prior = log_prior is not None
    
#     old_device = log_probs.device
#     log_probs = log_probs.to(device)
#     log_lm_probs = log_lm_probs.to(device)
#     if use_prior:
#         log_prior = log_prior.to(device)
#     input_lengths = input_lengths.to(device)
    
#     max_audio_time, batch_size, n_out = log_probs.shape
#     # scaled log am and lm probs
#     log_probs = am_scale * log_probs
#     log_lm_probs = log_lm_probs * lm_scale
#     if use_prior:
#         log_prior = prior_scale * log_prior
    
#     # print_gradients = PrintGradients.apply
#     # grad_assert = AssertGradients.apply
    
#     if eos_idx is None or eos_idx == blank_idx: # vocab means no EOS and blank
#         vocab_size = n_out - 1
#         eos_symbol = blank_idx
#     else:
#         vocab_size = n_out - 2
#         eos_symbol = eos_idx
#         assert blank_idx == n_out - 1, "blank should be the last symbol"
#     assert unk_idx is not None, "unk_idx should be defined"
#     vocab_size -= 1 # remove unk from vocab size
#     # BOS and UNK are in the LM
#     assert log_lm_probs.size() == (vocab_size + 2,) * LM_order, f"LM shape is not correct, should be {vocab_size + 2} in all dimensions but is {log_lm_probs.size()}"
#     # Reshape higher order LMs to two dimensions, the first is the context and the second is the next vocab token
#     if i > 0:
#         log_lm_probs_list[i] = log_lm_probs_list[i].reshape(-1, vocab_size + 2)
#         assert log_lm_probs_list[i].size() == ((vocab_size + 2)**(i + 1), vocab_size + 2), f"LM shape is not correct, should be {(vocab_size + 2)**(i + 1)}x{vocab_size + 2} but is {log_lm_probs_list[i].size()}"
#     if use_prior:
#         assert log_prior.size() == (n_out + 1,), f"Prior shape is not correct, should be {n_out + 1} but is {log_prior.size()}"
    
#     # calculate empty sequence score
#     # Empty am score = sum log prob blank
#     if use_prior:
#         log_partial_empty_seq_prob = (log_probs[:, :, blank_idx] - log_prior[blank_idx]).cumsum(dim=0)
#     else:
#         log_partial_empty_seq_prob = log_probs[:, :, blank_idx].cumsum(dim=0)
#     log_empty_seq_prob = log_partial_empty_seq_prob.gather(0, (input_lengths-1).long().unsqueeze(0)).squeeze(0)
#     # Empty lm score = p_LM(eos | bos), prior score = p_PR(eos)
#     log_q_empty_seq = log_empty_seq_prob + log_lm_probs_list[0][eos_symbol, eos_symbol]

#     # Unbind the log_probs and log_partial_empty_seq_prob for each timestep so it is faster during backprop
#     log_probs = log_probs.unbind(0)
#     log_partial_empty_seq_prob = log_partial_empty_seq_prob.unbind(0)

#     # to remove blank, unk and eos from the last dim (vocab)
#     out_idx = torch.arange(n_out, device=device)
#     out_idx_vocab = out_idx[out_idx != blank_idx].long() # "vocab" means no EOS, unk and blank
#     out_idx_vocab = out_idx_vocab[out_idx_vocab != unk_idx]
#     out_idx_vocab_w_eos = out_idx[out_idx != unk_idx].long()
#     if eos_idx is not None and eos_idx != blank_idx:
#         out_idx_vocab = out_idx_vocab[out_idx_vocab != eos_idx]
#         out_idx_vocab_w_eos = out_idx_vocab_w_eos[out_idx_vocab_w_eos != blank_idx]
        
#     def _convert_idx(idx_list: torch.Tensor, N_order: int, lm_size: int) -> torch.Tensor:
#         """Converts index list into multiple dimensions for higher order LMs
#         """
#         with torch.no_grad():
#             n = idx_list.size(0)
#             ret = idx_list
#             for _ in range(N_order - 2):
#                 ret = ret * lm_size
#                 ret = ret.unsqueeze(-1).expand(-1, n)
#                 ret = ret + idx_list
#                 ret = ret.flatten()
#         return ret

#     # sum score by DP
    
#     # List in which we store the log Q values as tensors of the last N timesteps
#     # dim 2: 0 is non-blank, 1 is blank
#     # Init Q for t=1
#     # Q(1, u, blank) = 0
#     # Q(1, u, non-blank) = p_AM(u | x1) * p_LM(u | bos) / p_PR(u)
#     log_q_label = log_probs[0][:, out_idx_vocab] + log_lm_probs_list[0][eos_symbol, out_idx_vocab].unsqueeze(0)
#     if use_prior:
#         log_q_label = log_q_label - log_prior[out_idx_vocab].unsqueeze(0)
#     log_q_blank = torch.full((batch_size, vocab_size), log_zero, device=device) # (B, 2, V-1), no blank and eos in last dim
#     log_q = log_q_label
    
#     for i in range(1, LM_order - 1):
#         log_q_label = log_probs[0][:, out_idx_vocab] + log_lm_probs_list[0][eos_symbol, out_idx_vocab].unsqueeze(0)
    
#     log_lm_probs_wo_eos = log_lm_probs_list[0][out_idx_vocab][:, out_idx_vocab].fill_diagonal_(log_zero)
#     lm_idx = 0
#     for t in range(1, max_audio_time):
#         # if we have higher order LMSs, we have to change the LM used if we have enough timesteps
#         if t < LM_order - 1:
#             log_lm_probs_wo_eos = log_lm_probs_list[lm_idx][_convert_idx(out_idx_vocab, lm_idx + 2, vocab_size + 2)][:, out_idx_vocab].fill_diagonal_(log_zero, wrap=True)
#             lm_idx += 1
        
#         # case 1: emit a blank at t
#         # Q(t, u, blank) = [Q(t-1, u, blank) + Q(t-1, u, non-blank)]*p_AM(blank | x_t)
#         new_log_q_blank = log_q + log_probs[t][:, blank_idx].unsqueeze(-1)
#         if use_prior:
#             new_log_q_blank = new_log_q_blank - log_prior[blank_idx]

#         # case 2: emit a non-blank at t
#         # Q(t, u, non-blank) = p_AM(u|x_t) * [horizontal + diagonal + skip] 
        
#         # horizontal transition Q(t-1, u, non-blank)
#         log_mass_horizontal = log_q_label
#         if horizontal_prior and use_prior:
#             log_mass_horizontal = log_mass_horizontal - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
#         # diagonal transition sum_v Q(t-1, v, blank) * p_LM(u|v) / p_PR(u)
#         # take batch index b into account, this is equivalent to compute
#         # mass_diagonal[b, u] = sum_v Q(t-1, b, blank, v) * p_LM(u|v) / p_PR(u)
#         # mass_diagonal = Q(t-1, :, blank, :) @ M / p_PR(u), where M(v,u) = p_LM(u|v) = lm_probs[v][u]
#         # important: in this transition, there is a prefix empty^(t-1) that is not covered in the Q(t-1,v,blank)
#         # this is covered in log_partial_empty_seq_prob[t-1]
#         log_prev_partial_seq_probs = torch.cat([log_partial_empty_seq_prob[t-1].unsqueeze(-1), log_q_blank], dim=-1)
#         log_mass_diagonal = log_matmul(log_prev_partial_seq_probs, log_lm_probs_list[lm_idx][_convert_idx(out_idx_vocab_w_eos, lm_idx + 2, vocab_size + 2)][:, out_idx_vocab]) # (B, V) @ (V, V-1)
#         if use_prior:
#             log_mass_diagonal = log_mass_diagonal - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
#         # skip transition sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w) / p_PR(u)
#         # same consideration as diagonal transition
#         log_mass_skip = log_matmul(log_q_label, log_lm_probs_wo_eos)
#         if use_prior:
#             log_mass_skip = log_mass_skip - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
#         # multiply with p_AM(u|x_t)
#         new_log_q_label = log_probs[t][:, out_idx_vocab] + safe_logsumexp(torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1), dim=-1)
        
#         # set masked results to log_q
#         time_mask = (t < input_lengths).unsqueeze(-1).expand(-1, vocab_size)
#         log_q_blank = torch.where(time_mask, new_log_q_blank, log_q_blank)
#         log_q_label = torch.where(time_mask, new_log_q_label, log_q_label)
#         log_q = safe_logaddexp(log_q_label, log_q_blank)
        
#         torch.cuda.empty_cache()
    
#     # multiply last Q with p_LM(eos | u) and devide by prior of EOS
#     log_q = log_q + log_lm_probs[out_idx_vocab, eos_symbol].unsqueeze(0)
    
#     # sum over the last two dimensions
#     sum_score = safe_logsumexp(log_q, dim=-1)
#     # add empty sequence score
#     sum_score = safe_logaddexp(sum_score, log_q_empty_seq) # (B,) # TODO do we need to add the empty seq?
    
#     loss = -sum_score
#     if old_device != device:
#         loss = loss.to(old_device)
    
#     return loss

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

def safe_logsumexp(x: torch.Tensor, dim: int, *, keepdim: bool = False) -> torch.Tensor:
    """safe logsumexp, handles the case of -inf values. otherwise, when all are -inf, you get nan"""
    with torch.no_grad():
        max_x, _ = x.max(dim=dim, keepdim=True)
        mask = max_x.isneginf()
        max_x_ = max_x if keepdim else max_x.squeeze(dim=dim)
        mask_ = mask if keepdim else mask.squeeze(dim=dim)
    sum = (x - max_x.masked_fill(mask, 0)).exp().sum(dim=dim, keepdim=keepdim)
    return max_x_ + sum.masked_fill_(mask_, 1).log()

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

    :param A: first matrix in log scale
    :param B: second matrix in log scale
    :returns: matrix product of the two matrices in log scale
    """
    b, v = A.shape
    if batch_given:
        b, v, v2 = B.shape
        B_expand = B.transpose(1, 2) # (b, v2, v)
    else:
        v, v2 = B.shape
        B_expand = B.unsqueeze(0).expand(b, -1, -1).transpose(1, 2) # (b, v2, v)
    A_expand = A.unsqueeze(0).expand(v2, -1, -1).transpose(0, 1) # (b, v2, v)
        
    return safe_logsumexp((A_expand + B_expand), dim=-1)

class PrintGradients(torch.autograd.Function):
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
            print(f"Gradients ({name}): {grad_output} {grad_output.sum()}\nInput: {x}\nNaN's: {torch.isnan(grad_output).sum()}")
        else:
            print(f"Gradients ({name}): {grad_output} {grad_output.sum()}\nNaN's: {torch.isnan(grad_output).sum()}")
        return grad_output, None, None
    
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
        A = torch.randn(10, 185, device=device)
        # B = torch.randn(1000, 1500, device=device).log_softmax(dim=0)
        B = torch.randn(185, 185, device=device).log_softmax(dim=1)
        # A[0] = float("-inf")
        # B[:, 0] = float("-inf")
        A.requires_grad = True
        B.requires_grad = True
        
        A = ag(A, "A", False)
        B = ag(B, "B", False)
        
        # res = A.exp().matmul(B.exp()).log()
        # res = log_matmul_alt(A, B)
        A, topk_idx = torch.topk(A, 20, dim=-1)
        B = B.unsqueeze(0).expand(10, -1, -1)
        B = B.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 185))
        res = log_matmul(A, B, batch_given=True)
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
    
    # torch.cuda.set_sync_debug_mode(1)
    
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    
    batch_size = 12 # 14000 per GPU, 1250 stpes a 12 seqs (0.7 sec/step)
    vocab_size = 185
    frames = 100
    
    torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    ag = AssertGradients.apply
    
    lm = torch.randn(vocab_size - 1, vocab_size - 1, device=device)
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
        
        
        loss = sum_loss(
            log_probs=am,
            log_lm_probs=lm,
            log_prior=prior,
            input_lengths=length,
            top_k = 2,
            LM_order=2,
            am_scale=1.0,
            lm_scale=1.0,
            prior_scale=1.0,
            horizontal_prior=True,
            blank_prior=True,
            blank_idx=184,
            eos_idx=0,
            print_best_path_for_idx=[0]
        )
        print("OUT", loss[0].tolist())
        l += (loss / frames).mean()
        
        # del loss, am, prior
        # torch.cuda.empty_cache()
        # print(time.time() - s)
        
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
    with open("/u/marten.mueller/dev/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.S9n2YtP1JzJ5/output/lm.pt", "rb") as f:
    # with open("/u/marten.mueller/dev/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.wuVkNuDg8B55/output/lm.pt", "rb") as f:
        t = torch.load(f)
    print(t[0])

if __name__ == "__main__":
    test()


"""
57 150 180 110 107 128 112 105 11 156 86 164 161 85 155 35 108 11 133 48 133 17 164 96 2 113 123 31 123 23 153 20 6 161 120 88 131 20 15 99 153 58 119 1 88 59 136 65 105 99 122 37 147 20 36 68 44 98 141 18 1 158 3 65 105 97 151 113 85 36 115 160 83 168 36 125 28 150 136 90 170 179 20 25 166 71 147 95 168 176 164 38 97 118 125 76 43 72

Best path for 0: 57 150 180 110 181 128 112 105 11 156 86 164 161 85 155 35 108 11 133 48 65 17 164 96 2 113 123 31 123 23 153 20 6 161 120 88 131 20 15 172 153 83 79 55 167 59 136 65 105 112 49 37 147 20 36 68 44 98 141 18 140 158 3 65 105 97 151 113 85 36 115 160 41 168 36 125 28 150 136 150 170 170 179 179 20 25 166 71 147 95 168 176 164 38 97 118 125 76 126 72
Score: -2.91 -5.34 -8.16 -11.21 -14.50 -16.31 -18.58 -21.12 -23.79 -26.04 -28.90 -31.76 -34.22 -37.48 -40.06 -43.06 -46.01 -48.10 -51.41 -53.89 -57.03 -59.37 -62.29 -65.05 -68.47 -71.37 -74.73 -77.03 -79.78 -82.74 -86.15 -89.46 -91.50 -93.82 -96.39 -99.48 -101.92 -104.44 -107.64 -110.45 -113.45 -116.10 -119.17 -121.92 -124.92 -127.49 -130.57 -133.07 -135.62 -138.84 -142.07 -144.52 -147.23 -150.00 -152.76 -155.17 -158.34 -161.56 -163.86 -167.02 -170.22 -172.81 -174.91 -178.22 -181.32 -184.03 -187.09 -190.51 -193.15 -196.25 -199.47 -202.59 -204.92 -207.94 -210.88 -212.49 -214.79 -218.09 -221.38 -224.42 -227.13 -229.46 -232.57 -235.22 -238.76 -241.46 -244.60 -247.16 -249.67 -252.67 -254.90 -257.64 -260.54 -262.56 -265.46 -267.74 -270.98 -273.89 -277.19 -280.07
AM: 295.41814041137695
OUT 280.0710754394531
tensor([291.3092], grad_fn=<CtcLossBackward0>)


Best path for 0: 57 150 180 110 181 128 112 105 11 156 86 164 161 85 155 35 108 11 133 48 65 17 164 96 2 113 123 31 123 23 153 20 6 161 120 88 131 20 15 172 153 83 79 55 167 59 136 65 105 112 49 37 147 20 36 68 44 98 141 18 140 158 3 65 105 97 151 113 85 36 115 160 41 168 36 125 28 150 136 150 170 170 179 179 20 25 166 71 147 95 168 176 164 38 97 118 125 76 126 72
Score: -2.91 -5.35 -8.17 -11.22 -14.51 -16.32 -18.59 -21.13 -23.80 -26.05 -28.91 -31.77 -34.23 -37.49 -40.07 -43.08 -46.02 -48.11 -51.42 -53.90 -57.05 -59.38 -62.30 -65.06 -68.48 -71.38 -74.75 -77.05 -79.79 -82.75 -86.16 -89.47 -91.51 -93.83 -96.40 -99.49 -101.93 -104.45 -107.66 -110.46 -113.46 -116.12 -119.18 -121.94 -124.93 -127.51 -130.58 -133.08 -135.63 -138.85 -142.08 -144.53 -147.24 -150.01 -152.77 -155.19 -158.35 -161.57 -163.87 -167.03 -170.23 -172.82 -174.92 -178.24 -181.33 -184.05 -187.10 -190.52 -193.16 -196.26 -199.48 -202.60 -204.93 -207.95 -210.90 -212.50 -214.80 -218.10 -221.39 -224.43 -227.14 -229.47 -232.59 -235.24 -238.77 -241.47 -244.61 -247.17 -249.68 -252.68 -254.91 -257.65 -260.55 -262.57 -265.47 -267.75 -270.99 -273.90 -277.20 -280.08
AM: 295.41814041137695
OUT 280.0833740234375
tensor([291.3092], grad_fn=<CtcLossBackward0>)

Best path for 0: 57 150 180 110 181 128 112 105 11 156 86 164 161 85 155 35 108 11 133 48 65 17 164 96 2 113 123 31 123 23 153 20 6 161 120 88 131 20 15 172 153 58 79 55 167 59 136 65 105 112 49 37 147 20 36 68 44 98 141 18 140 158 3 65 105 97 151 113 85 36 115 160 41 168 36 125 28 150 136 136 170 170 179 179 20 25 166 71 147 95 168 176 164 38 97 118 125 76 126 72
Score: -2.91 -5.35 -8.17 -11.27 -14.60 -16.42 -18.70 -21.24 -23.91 -26.17 -29.03 -31.90 -34.36 -37.62 -40.23 -43.24 -46.20 -48.30 -51.61 -54.09 -57.26 -59.62 -62.54 -65.31 -68.73 -71.65 -75.01 -77.31 -80.07 -83.05 -86.46 -89.77 -91.81 -94.13 -96.71 -99.80 -102.24 -104.76 -107.97 -110.79 -113.80 -116.65 -119.72 -122.48 -125.48 -128.05 -131.13 -133.64 -136.19 -139.42 -142.72 -145.17 -147.88 -150.67 -153.43 -155.89 -159.08 -162.35 -164.65 -167.81 -171.01 -173.61 -175.71 -179.03 -182.18 -184.89 -187.95 -191.40 -194.04 -197.16 -200.38 -203.52 -205.88 -208.90 -211.85 -213.45 -215.75 -219.06 -222.37 -225.79 -228.48 -230.81 -233.95 -236.60 -240.15 -242.86 -246.00 -248.64 -251.15 -254.17 -256.40 -259.15 -262.05 -264.07 -266.97 -269.26 -272.50 -275.41 -278.74 -281.69
AM: 296.80163979530334
OUT 281.6884460449219
tensor([291.3092], grad_fn=<CtcLossBackward0>)

Best path for 0: 57 150 180 110 107 128 112 105 11 156 86 164 161 85 155 35 108 11 133 48 133 17 164 96 2 113 123 31 123 23 153 20 6 161 120 88 131 20 15 99 153 58 119 55 88 59 136 65 105 99 122 37 147 20 36 68 44 98 141 18 140 158 3 65 105 97 151 113 85 36 115 160 83 168 36 125 28 150 136 136 170 170 179 179 20 25 166 71 147 95 168 176 164 38 97 118 125 76 43 72
Score: -2.91 -5.44 -8.27 -11.46 -15.08 -17.21 -19.50 -22.07 -24.81 -27.11 -30.02 -32.94 -35.43 -38.69 -41.63 -44.67 -47.67 -50.15 -53.48 -56.09 -59.38 -61.82 -65.03 -67.86 -71.30 -74.40 -77.78 -80.17 -82.98 -86.08 -89.50 -92.89 -94.95 -97.28 -100.03 -103.33 -106.11 -108.73 -111.95 -115.11 -118.26 -121.30 -124.51 -127.30 -130.33 -132.93 -136.01 -138.65 -141.24 -144.55 -148.10 -150.59 -153.32 -156.20 -159.09 -161.73 -164.96 -168.38 -171.05 -174.24 -177.50 -180.18 -182.32 -185.68 -189.27 -192.14 -195.27 -198.77 -201.44 -204.62 -207.92 -211.18 -213.70 -216.77 -219.74 -221.67 -224.00 -227.32 -230.66 -234.09 -236.85 -239.19 -242.39 -245.04 -248.68 -251.50 -254.79 -257.75 -260.28 -263.34 -265.70 -268.50 -271.42 -273.51 -276.44 -278.74 -282.01 -284.95 -288.38 -291.61
AM: 295.8504433631897
OUT 291.6074523925781
tensor([291.3092], grad_fn=<CtcLossBackward0>)

Best path for 0: 57 150 180 110 107 128 112 105 11 156 86 164 161 85 155 35 108 11 133 48 133 17 164 96 2 113 123 31 123 23 153 20 6 161 120 88 131 20 15 99 153 58 119 55 88 59 136 65 105 99 122 37 147 20 36 68 44 98 141 18 140 158 3 65 105 97 151 113 85 36 115 160 83 168 36 125 28 150 136 136 170 170 179 179 20 25 166 71 147 95 168 176 164 38 97 118 125 76 43 72
Score: -2.91 -5.43 -8.26 -11.45 -15.07 -17.20 -19.49 -22.05 -24.80 -27.10 -30.00 -32.93 -35.42 -38.68 -41.62 -44.66 -47.66 -50.14 -53.46 -56.08 -59.36 -61.81 -65.02 -67.85 -71.29 -74.38 -77.77 -80.16 -82.97 -86.06 -89.48 -92.87 -94.94 -97.27 -100.01 -103.32 -106.09 -108.71 -111.94 -115.09 -118.25 -121.28 -124.50 -127.29 -130.32 -132.92 -136.00 -138.63 -141.23 -144.53 -148.08 -150.58 -153.30 -156.18 -159.07 -161.72 -164.95 -168.37 -171.04 -174.23 -177.48 -180.16 -182.31 -185.67 -189.26 -192.13 -195.26 -198.76 -201.43 -204.60 -207.91 -211.16 -213.69 -216.75 -219.73 -221.66 -223.99 -227.30 -230.65 -234.07 -236.84 -239.17 -242.38 -245.03 -248.67 -251.49 -254.77 -257.74 -260.27 -263.32 -265.69 -268.49 -271.41 -273.50 -276.43 -278.72 -281.99 -284.93 -288.36 -291.59
AM: 295.8504433631897
OUT 291.593994140625
tensor([291.3092], grad_fn=<CtcLossBackward0>)
"""