"""
Implement lattice-free MMI training for CTC
"""

import torch

####### Be careful about stuffs related to EOS and blank index with the BPE setup
def sum_loss(
    *,
    log_probs: torch.Tensor, # (T, B, V)
    log_lm_probs: torch.Tensor, # (V, V)
    log_prior: torch.Tensor, # (V,) # TODO
    input_lengths: torch.Tensor, # (B,)
    am_scale: float,
    lm_scale: float,
    blank_idx:int = 0, # should be same as eos index
    eos_idx: int | None = None,
    unk_idx: int = 1,
    log_zero: float = -1e15,
):
    """
    Sum criterion training for CTC, given by
    L = sum_{all seq} q(seq)
    where q(seq) = p_AM^alpha(seq) * p_LM^beta(seq) / p_PR(seq) # TODO is result scaled after prior deduction or do we need own scale for prior?
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
    :param am_scale: AM scale
    :param lm_scale: LM scale
    :param blank_idx: Blank index in V dim
    :param eos_idx: EOS idx in the vocab. None if no EOS in log_probs vocab.
        If None, then blank_idx in log_lm_probs should be EOS
    :param unk_idx: Unknown index in V dim
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :returns: log sum loss
    """
    
    # ep 1 train, step 200, aux_full_sum_4 0.283, aux_full_sum_8 0.078, full_sum 0.072, num_seqs 10, max_size:time 181296, max_size:out-spatial 105, mem_usage:cuda 7.7GB, 3.023 sec/step, elapsed 0:13:55, exp. remaining 1:08:29, complete 16.90%
    # ep 1 train, step 200, aux_full_sum_4 -1.046, aux_full_sum_8 -1.081, full_sum -1.069, num_seqs 10, max_size:time 181296, max_size:out-spatial 105, mem_usage:cuda 7.7GB, 3.596 sec/step, elapsed 0:14:43, exp. remaining 1:12:27, complete 16.90%
    
    device = log_probs.device
    log_probs = log_probs.to(device)
    log_lm_probs = log_lm_probs.to(device)
    log_prior = log_prior.to(device)
    input_lengths = input_lengths.to(device)
    
    max_audio_time, batch_size, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs_scaled = am_scale*log_probs
    log_lm_probs_scaled = lm_scale*log_lm_probs
    
    # print_gradients = PrintGradients.apply
    
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
    assert log_prior.size() == (n_out + 1,), f"Prior shape is not correct, should be {n_out + 1} but is {log_prior.size()}"
    
    # calculate empty sequence score
    # Empty am score = sum log prob blank
    log_partial_empty_seq_prob = log_probs_scaled[:, :, blank_idx].cumsum(dim=0)
    log_empty_seq_prob = log_partial_empty_seq_prob.gather(0, (input_lengths-1).long().unsqueeze(0)).squeeze(0)
    # Empty lm score = p_LM(eos | bos)
    log_q_empty_seq = log_empty_seq_prob + log_lm_probs_scaled[eos_symbol, eos_symbol]

    # to remove blank, unk and eos from the last dim (vocab)
    out_idx = torch.arange(n_out, device=device)
    out_idx_vocab = out_idx[out_idx != blank_idx].long() # "vocab" means no EOS, unk and blank
    out_idx_vocab = out_idx_vocab[out_idx_vocab != unk_idx]
    out_idx_vocab_w_eos = out_idx[out_idx != unk_idx].long()
    if eos_idx is not None and eos_idx != blank_idx:
        out_idx_vocab = out_idx_vocab[out_idx_vocab != eos_idx]
        out_idx_vocab_w_eos = out_idx_vocab_w_eos[out_idx_vocab_w_eos != blank_idx]

    # sum score by DP
    
    # Tensor in which we store the log Q values TODO: chnage this to only keep the LM context last timeteps instead of all of them
    # dim 2: 0 is non-blank, 1 is blank
    log_q = torch.full((max_audio_time, batch_size, 2, vocab_size), log_zero, device=device) # (T, B, 2, V-1), no blank and eos in last dim
    # Init Q for t=1
    # Q(1, u, blank) = 0
    # Q(1, u, non-blank) = p_AM(u | x1) * p_LM(u | bos) / p_PR(u)
    log_q[0, :, 0, :] = log_probs_scaled[0, :, out_idx_vocab] + log_lm_probs_scaled[eos_symbol, out_idx_vocab].unsqueeze(0) - log_prior[out_idx_vocab].unsqueeze(0)
    
    log_lm_probs_scaled_wo_eos = log_lm_probs_scaled[out_idx_vocab][:, out_idx_vocab]
    for t in range(1, max_audio_time):
        # case 1: emit a blank at t
        new_log_q = torch.full((batch_size, 2, vocab_size), log_zero, device=device)
        # Q(t, u, blank) = [Q(t-1, u, blank) + Q(t-1, u, non-blank)]*p_AM(blank | x_t)
        new_log_q[:, 1, :] = safe_logsumexp(log_q[t-1], dim=1) + log_probs_scaled[t, :, blank_idx].unsqueeze(-1)

        # case 2: emit a non-blank at t
        # Q(t, u, non-blank) = p_AM(u|x_t) * [horizontal + diagonal + skip] 
        
        # horizontal transition Q(t-1, u, non-blank)
        log_mass_horizontal = log_q[t-1, :, 0, :]
        
        # diagonal transition sum_v Q(t-1, v, blank) * p_LM(u|v) / p_PR(u)
        # take batch index b into account, this is equivalent to compute
        # mass_diagonal[b, u] = sum_v Q(t-1, b, blank, v) * p_LM(u|v) / p_PR(u)
        # mass_diagonal = Q(t-1, :, blank, :) @ M / p_PR(u), where M(v,u) = p_LM(u|v) = lm_probs[v][u]
        # important: in this transition, there is a prefix empty^(t-1) that is not covered in the Q(t-1,v,blank)
        # this is covered in log_partial_empty_seq_prob[t-1]
        log_prev_partial_seq_probs = torch.cat([log_partial_empty_seq_prob[t-1].unsqueeze(-1), log_q[t-1, :, 1, :]], dim=-1)
        log_mass_diagonal = log_matmul(log_prev_partial_seq_probs, log_lm_probs_scaled[out_idx_vocab_w_eos][:, out_idx_vocab]) # (B, V) @ (V, V-1)
        log_mass_diagonal = log_mass_diagonal - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
        # skip transition sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w) / p_PR(u)
        # same consideration as diagonal transition
        log_mass_skip = modified_log_matmul(log_q[t-1, :, 0, :], log_lm_probs_scaled_wo_eos)
        log_mass_skip = log_mass_skip - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
        # multiply with p_AM(u|x_t)
        new_log_q[:, 0, :] = log_probs_scaled[t, :, out_idx_vocab] + safe_logsumexp(torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1), dim=-1)
        
        # set masked results to log_q
        time_mask = (t < input_lengths).unsqueeze(-1).unsqueeze(-1).expand(-1, 2, vocab_size)
        log_q[t] = torch.where(time_mask, new_log_q, log_q[t-1])
        
        torch.cuda.empty_cache()
    
    # multiply last Q with p_LM(eos | u) and devide by prior of EOS
    log_q[-1] += log_lm_probs_scaled[out_idx_vocab, eos_symbol].unsqueeze(0).unsqueeze(0) - log_prior[-1].unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    # add empty sequence score
    sum_score = torch.logaddexp(safe_logsumexp(safe_logsumexp(log_q[-1], dim=-1), dim=-1), log_q_empty_seq) # (B,) # TODO do we need to add the empty seq? # TODO make logaddexp safe
    
    loss = -sum_score
    return loss


def sum_loss_topk(
    *,
    log_probs: torch.Tensor, # (T, B, V)
    log_bigram_probs: torch.Tensor, # (V, V)
    input_lengths: torch.Tensor, # (B,)
    am_scale: float,
    lm_scale: float,
    top_k: int,
    blank_idx: int = 10025, # should be same as eos index
    eos_idx: int | None = None,
    log_zero: float = -1e15,
):
    """
    Seems to only work for eos=0, blank=last...

    This version tries to reduce mem usage by having top K hypotheses
    at each time step.

    Sum criterion training for CTC, given by
    L = sum_{all seq} q(seq)
    where q(seq) = p_AM^alpha(seq) * p_LM^beta(seq)
    and p_AM = prod_n posterior / prior.

    This is for the case the LM is a bigram (context 1).

    The loss is calculated by sum_{u in V} [Q(T, u, blank) + Q(T, u, non-blank)],
    where Q(t, u, {N or B}) is the sum of partial CTC alignments up to timeframe t
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
    because the transitions in the denominator does not consider EOS.

    :param log_probs: log CTC output probs (T, B, V)
    :param log_lm_probs: (V, V), then log bigram LM probs of all possible context
    :param input_lengths: Input lengths (B,)
    :param am_scale: AM scale
    :param lm_scale: LM scale
    :param top_k: At each time step keeps top best K scores.
    :param blank_idx: Blank index in V dim
    :param eos_idx: EOS idx in the vocab. None if no EOS in log_probs vocab.
        If None, then blank_idx in log_lm_probs should be EOS
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :returns: log sum loss
    """
    device = log_probs.device
    max_audio_time, batch_size, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs = am_scale*log_probs
    log_bigram_probs = lm_scale*log_bigram_probs

    if eos_idx is None or eos_idx == blank_idx: # vocab means no EOS and blank
        vocab_size = n_out - 1
        eos_symbol = blank_idx
    else:
        vocab_size = n_out - 2
        eos_symbol = eos_idx
    
    # calculate empty sequence score
    # Empty am score = sum log prob blank
    log_partial_empty_seq_prob = log_probs[:, :, blank_idx].cumsum(dim=0)
    log_empty_seq_prob = log_partial_empty_seq_prob.gather(0, (input_lengths-1).to(device).long().unsqueeze(0)).squeeze(0)
    # Empty lm score = p_LM(eos | bos)
    log_q_empty_seq = log_empty_seq_prob + log_bigram_probs[eos_symbol][eos_symbol]

    # to remove blank and eos from the last dim
    out_idx = torch.arange(n_out)
    out_idx_vocab = out_idx[out_idx != blank_idx].long().to(device) # "vocab" means no EOS and blank
    if eos_idx is not None and eos_idx != blank_idx:
        out_idx_vocab = out_idx_vocab[out_idx_vocab != eos_idx]

    # sum score by DP
    # dim 2: 0 is non-blank, 1 is blank

    log_bigram_probs_no_eos = log_bigram_probs.index_select(0, out_idx_vocab).index_select(1, out_idx_vocab)
    log_bigram_probs_no_eos_masked = log_bigram_probs_no_eos.clone().fill_diagonal_(log_zero)
    # Init Q for t=1
    all_hyp_scores_label = log_probs[0, :, out_idx_vocab] + log_bigram_probs[eos_symbol:eos_symbol+1, out_idx_vocab].expand(batch_size, -1) # (B, V)
    all_hyp_scores = all_hyp_scores_label
    all_hyp_scores_blank = torch.full((batch_size, vocab_size), fill_value=log_zero, device=log_probs.device)
    
    topk_scores, topk_idx = torch.topk(all_hyp_scores, k=top_k, dim=-1) # (B, K)
    for t in range(1, max_audio_time):
        # case 1: emit a blank at t
        all_hyp_scores_blank_new = all_hyp_scores + log_probs[t, :, blank_idx].unsqueeze(1).expand(-1, vocab_size)
        
        # case 2: emit a non-blank at t

        # horizontal transition Q(t-1, u, non-blank)
        log_mass_horizontal = all_hyp_scores_label # (B, V)

        # diagonal transition sum_v Q(t-1, v, blank)*p_LM(u|v)
        # TODO why this weitd eos handling?
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
        all_hyp_scores_label_new = log_probs[t, :, out_idx_vocab] + (safe_logsumexp(torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1), dim=-1)) # (B, V)
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
    
    # add empty sequence score
    sum_score = torch.logaddexp(safe_logsumexp(topk_scores, -1), log_q_empty_seq)
    
    loss = -sum_score
    return loss


# ------------------------------------------------
# Helper functions and classes

def safe_logsumexp(x: torch.Tensor, dim: int, *, keepdim: bool = False) -> torch.Tensor:
    """safe logsumexp, handles the case of -inf values. otherwise, when all are -inf, you get nan"""
    with torch.no_grad():
        max_x, _ = x.max(dim=dim, keepdim=True)
        max_x = max_x.detach()
        max_x_ = max_x if keepdim else max_x.squeeze(dim=dim)
    return max_x_ + torch.where(max_x_.isneginf(), 0.0, (x - max_x).exp().sum(dim=dim, keepdim=keepdim).log())

class PrintGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        return x

    @staticmethod
    def backward(ctx, grad_output):
        name = ctx.name
        print(f"Gradients ({name}):", grad_output)
        print(f"NaN ({name}): {torch.isnan(grad_output).sum()}")
        print(f"Inf ({name}): {torch.isinf(grad_output).sum()}")
        return grad_output, None

def log_matmul(A: torch.Tensor, B: torch.Tensor):
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
    m, n = A.shape
    n, r = B.shape
    A_expand = A.unsqueeze(0).expand(r, -1, -1).transpose(0, 1) # (m, r, n)
    B_expand = B.unsqueeze(0).expand(m, -1, -1).transpose(1, 2) # (m, r, n)
    return safe_logsumexp((A_expand + B_expand), dim=-1)


def batch_log_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Batch case of log_matmul

    :param A: first batch of matrices in log scale
    :param B: second batch of matrices in log scale
    :returns: matrix product of the two matrices in log scale
    """
    b, m, n = A.shape
    b, n, r = B.shape
    A_expand = A.unsqueeze(1).expand(-1, r, -1, -1).transpose(1, 2) # (m, r, n)
    B_expand = B.unsqueeze(1).expand(-1, m, -1, -1).transpose(2, 3) # (m, r, n)
    return safe_logsumexp((A_expand + B_expand), dim=-1)


def modified_log_matmul(A: torch.Tensor, B: torch.Tensor, log_zero=-1e15):
    """
    This is also inefficient

    Special case of log_matmul to calculate
    Z_ij = sum{k != j} X_{ik}*Y_{kj}
    where A = log X, B = log Y,
    B is square matrix
    """
    m, n = A.shape
    A_expand = A.unsqueeze(0).expand(n, -1, -1).transpose(0, 1) # (m, n, n)
    B_expand = B.unsqueeze(0).expand(m, -1, -1).transpose(1, 2) # (m, n, n)
    C = A_expand + B_expand
    # to exclude k = j from the summation, apply some masking here
    index = torch.arange(n).unsqueeze(0).expand(m, -1).unsqueeze(-1).to(A.device)
    # write log zero to C[i][j][j] for all i, j
    C_masked = torch.scatter(C, 2, index, log_zero)
    return safe_logsumexp(C_masked, dim=-1)




def test():
    import time
    
    batch_size = 30 # 14000 per GPU, 1250 stpes a 12 seqs (0.7 sec/step)
    vocab_size = 185
    frames = 100
    
    torch.manual_seed(0)
    
    lm = torch.randn(vocab_size - 1, vocab_size - 1)
    lm = torch.nn.functional.log_softmax(lm, dim=-1)
    
    am = torch.randn(frames, batch_size, vocab_size, requires_grad=True)
    am = torch.nn.functional.log_softmax(am, dim=-1)
    
    prior = torch.randn(vocab_size + 1, requires_grad=True)
    prior = torch.nn.functional.log_softmax(prior, dim=-1)
    
    length = torch.full((batch_size,), frames)
    
    s1 = time.time()
    loss = sum_loss(
        log_probs=am,
        log_lm_probs=lm,
        log_prior=prior,
        input_lengths=length,
        am_scale=1.0,
        lm_scale=1.0,
        blank_idx=184,
        eos_idx=0,
    )
    loss = (loss / frames).mean()
    loss.backward()
    e1 = time.time()
    print(f"Sum loss took {time.strftime('%H:%M:%S', time.gmtime(e1-s1))}: {loss}") # 5:00 mins
    
    # s2 = time.time()
    # top_k = vocab_size - 1
    # loss2 = sum_loss_topk(
    #     log_probs=am,
    #     log_bigram_probs=lm,
    #     input_lengths=length,
    #     am_scale=1.0,
    #     lm_scale=1.0,
    #     top_k=top_k,
    #     blank_idx=0,
    # )
    # e2 = time.time()
    # print(f"Top {top_k} sum loss took {time.strftime('%H:%M:%S', time.gmtime(e2 - s2))}: {(loss2 / frames).mean()}") # 7:30 mins
    
    # s3 = time.time()
    
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
    
    # e3 = time.time()
    # print(f"CTC loss took {time.strftime('%H:%M:%S', time.gmtime(e3 - s3))}: {(ctc_loss / frames).mean()}") # 0:08 mins
    
if __name__ == "__main__":
    test()