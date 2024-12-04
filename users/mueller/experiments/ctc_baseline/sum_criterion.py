"""
Implement lattice-free MMI training for CTC
"""

import torch

####### Be careful about stuffs related to EOS and blank index with the BPE setup
def sum_loss(
    *,
    log_probs: torch.Tensor, # (T, B, V)
    log_lm_probs: torch.Tensor, # (V, V)
    log_prior: torch.Tensor, # (V,)
    input_lengths: torch.Tensor, # (B,)
    LM_order: int,
    am_scale: float,
    lm_scale: float,
    blank_idx:int = 0, # should be same as eos index
    eos_idx: int | None = None,
    unk_idx: int = 1,
    log_zero: float = float("-inf"),
    device: torch.device = torch.device("cpu"),
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
    
    old_device = log_probs.device
    log_probs = log_probs.to(device)
    log_lm_probs = log_lm_probs.to(device)
    log_prior = log_prior.to(device)
    input_lengths = input_lengths.to(device)
    
    max_audio_time, batch_size, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs_scaled = am_scale*log_probs
    log_lm_probs_scaled = lm_scale*log_lm_probs
    
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
    assert log_prior.size() == (n_out + 1,), f"Prior shape is not correct, should be {n_out + 1} but is {log_prior.size()}"
    
    # calculate empty sequence score
    # Empty am score = sum log prob blank
    log_partial_empty_seq_prob = (log_probs_scaled[:, :, blank_idx] - log_prior[blank_idx]).cumsum(dim=0)
    log_empty_seq_prob = log_partial_empty_seq_prob.gather(0, (input_lengths-1).long().unsqueeze(0)).squeeze(0)
    # Empty lm score = p_LM(eos | bos), prior score = p_PR(eos)
    log_q_empty_seq = log_empty_seq_prob + log_lm_probs_scaled[eos_symbol, eos_symbol]

    # Unbind the log_probs_scaled and log_partial_empty_seq_prob for each timestep so it is faster during backprop
    log_probs_scaled = log_probs_scaled.unbind(0)
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
    log_q_label = log_probs_scaled[0][:, out_idx_vocab] + log_lm_probs_scaled[eos_symbol, out_idx_vocab].unsqueeze(0) - log_prior[out_idx_vocab].unsqueeze(0)
    log_q_blank = torch.full((batch_size, vocab_size), log_zero, device=device) # (B, 2, V-1), no blank and eos in last dim
    log_q = log_q_label
    
    log_lm_probs_scaled_wo_eos = log_lm_probs_scaled[out_idx_vocab][:, out_idx_vocab].fill_diagonal_(log_zero)
    for t in range(1, max_audio_time):
        # case 1: emit a blank at t
        # Q(t, u, blank) = [Q(t-1, u, blank) + Q(t-1, u, non-blank)]*p_AM(blank | x_t)
        new_log_q_blank = log_q + log_probs_scaled[t][:, blank_idx].unsqueeze(-1) - log_prior[blank_idx]

        # case 2: emit a non-blank at t
        # Q(t, u, non-blank) = p_AM(u|x_t) * [horizontal + diagonal + skip] 
        
        # horizontal transition Q(t-1, u, non-blank)
        log_mass_horizontal = log_q_label  - log_prior[out_idx_vocab].unsqueeze(0) # TODO make this optional
        
        # diagonal transition sum_v Q(t-1, v, blank) * p_LM(u|v) / p_PR(u)
        # take batch index b into account, this is equivalent to compute
        # mass_diagonal[b, u] = sum_v Q(t-1, b, blank, v) * p_LM(u|v) / p_PR(u)
        # mass_diagonal = Q(t-1, :, blank, :) @ M / p_PR(u), where M(v,u) = p_LM(u|v) = lm_probs[v][u]
        # important: in this transition, there is a prefix empty^(t-1) that is not covered in the Q(t-1,v,blank)
        # this is covered in log_partial_empty_seq_prob[t-1]
        log_prev_partial_seq_probs = torch.cat([log_partial_empty_seq_prob[t-1].unsqueeze(-1), log_q_blank], dim=-1)
        log_mass_diagonal = log_matmul_old(log_prev_partial_seq_probs, log_lm_probs_scaled[out_idx_vocab_w_eos][:, out_idx_vocab]) # (B, V) @ (V, V-1)
        log_mass_diagonal = log_mass_diagonal - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
        # skip transition sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w) / p_PR(u)
        # same consideration as diagonal transition
        log_mass_skip = log_matmul_old(log_q_label, log_lm_probs_scaled_wo_eos)
        log_mass_skip = log_mass_skip - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
        # multiply with p_AM(u|x_t)
        new_log_q_label = log_probs_scaled[t][:, out_idx_vocab] + safe_logsumexp(torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1), dim=-1)
        
        # set masked results to log_q
        time_mask = (t < input_lengths).unsqueeze(-1).expand(-1, vocab_size)
        log_q_blank = torch.where(time_mask, new_log_q_blank, log_q_blank)
        log_q_label = torch.where(time_mask, new_log_q_label, log_q_label)
        log_q = safe_logaddexp(log_q_label, log_q_blank)
        
        torch.cuda.empty_cache()
    
    # multiply last Q with p_LM(eos | u) and devide by prior of EOS
    log_q += log_lm_probs_scaled[out_idx_vocab, eos_symbol].unsqueeze(0)
    
    # sum over the last two dimensions
    sum_score = safe_logsumexp(log_q, dim=-1)
    # add empty sequence score
    sum_score = safe_logaddexp(sum_score, log_q_empty_seq) # (B,) # TODO do we need to add the empty seq?
    
    loss = -sum_score
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

def log_matmul(A: torch.Tensor, B: torch.Tensor):
    with torch.no_grad():
        max_A, _ = torch.max(A, dim=1, keepdim=True)
        max_B, _ = torch.max(B, dim=0, keepdim=True)
        mask_A = max_A.isneginf()
        mask_B = max_B.isneginf()
        mask = mask_A + mask_B > 0
    m1 = (A - max_A.masked_fill(mask_A, 0.0)).exp()
    m2 = (B - max_B.masked_fill(mask_B, 0.0)).exp()
    mul = m1.matmul(m2)
    mul_masked = mul.masked_fill_(mask, 1)
    log_mul = mul_masked.masked_fill_(mul_masked == 0.0, 0.0).log()
    return max_A + max_B + log_mul

def log_matmul_old(A: torch.Tensor, B: torch.Tensor):
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

# def modified_log_matmul(A: torch.Tensor, B: torch.Tensor, log_zero=-1e15):
#     """
#     This is also inefficient

#     Special case of log_matmul to calculate
#     Z_ij = sum{k != j} X_{ik}*Y_{kj}
#     where A = log X, B = log Y,
#     B is square matrix
#     """
#     m, n = A.shape
#     A_expand = A.unsqueeze(0).expand(n, -1, -1).transpose(0, 1) # (m, n, n)
#     B_expand = B.unsqueeze(0).expand(m, -1, -1).transpose(1, 2) # (m, n, n)
#     C = A_expand + B_expand
#     # to exclude k = j from the summation, apply some masking here
#     index = torch.arange(n, device=A.device).unsqueeze(0).expand(m, -1).unsqueeze(-1)
#     # write log zero to C[i][j][j] for all i, j
#     C_masked = C.scatter(2, index, log_zero)
#     return safe_logsumexp(C_masked, dim=-1)

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
            print(f"Gradients ({name}): {grad_output},\nInput: {x}\nNaN's: {torch.isnan(grad_output).sum()}")
        else:
            print(f"Gradients ({name}): {grad_output}\nNaN's: {torch.isnan(grad_output).sum()}")
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

def test_mul():
    # torch.autograd.set_detect_anomaly(True)
    
    import time
    import gc
    torch.manual_seed(0)
    ag = AssertGradients.apply
    
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    
    r = torch.tensor([0.0], device=device)
    time_d = 0.0
    for i in range(5000):
        s = time.time()
        A = torch.randn(500, 185, device=device).log_softmax(dim=1)
        # B = torch.randn(1000, 1500, device=device).log_softmax(dim=0)
        B = torch.randn(185, 185, device=device).log_softmax(dim=1)
        # A[0] = float("-inf")
        # B[:, 0] = float("-inf")
        A.requires_grad = True
        B.requires_grad = True
        
        A = ag(A, "A", False)
        B = ag(B, "B", False)
        
        # res = A.exp().matmul(B.exp()).log()
        # res = log_matmul(A, B)
        res = log_matmul_old(A, B)
        # res = modified_log_matmul(A, B)
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
    
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    
    batch_size = 12 # 14000 per GPU, 1250 stpes a 12 seqs (0.7 sec/step)
    vocab_size = 185
    frames = 100
    
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    ag = PrintGradients.apply
    
    lm = torch.randn(vocab_size - 1, vocab_size - 1, device=device)
    lm = torch.nn.functional.log_softmax(lm, dim=-1)
    
    l = torch.tensor([0.0], device=device)
    s1 = time.time()
    for i in range(20):
        s = time.time()
        am = torch.randn(frames, batch_size, vocab_size, requires_grad=True, device=device)
        am = torch.nn.functional.log_softmax(am, dim=-1)
        
        prior = torch.randn(vocab_size + 1, requires_grad=True, device=device)
        prior = torch.nn.functional.log_softmax(prior, dim=-1)
        
        length = torch.full((batch_size,), frames, device=device)

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
        l += (loss / frames).mean()
        
        del loss, am, prior
        torch.cuda.empty_cache()
        print(time.time() - s)
    l.backward(torch.ones_like(l, device=device))
    e1 = time.time()
    print(f"Sum loss took {time.strftime('%H:%M:%S', time.gmtime(e1-s1))}: {l}") # 5:00 mins
    
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
    
if __name__ == "__main__":
    test()
    
    
"""
Epoch 1: Trained 1248 steps, 1:26:12 elapsed (98.4% computing time)
Epoch 1: Total train loss: aux_full_sum_4 -1.034 aux_full_sum_8 -1.084 full_sum -1.205
Epoch 1 evaluation: dev: aux_full_sum_4 -1.310 aux_full_sum_8 -1.445 full_sum -1.938
"""