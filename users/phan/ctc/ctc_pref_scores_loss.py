import torch
from i6_experiments.users.phan.utils import get_seq_mask


def log_ctc_pref_beam_scores(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    blank_idx = 0,
    log_zero = -1e15, # maybe better than float min for preventing overflowing
    eos_idx = None, # used when eos is present in vocab, mainly when an attention model is present
):
    '''
    Given log probs of times and ground truths,
    compute all ctc prefix score p_ctc(a_1^n-1, v, suffix=empty or not empty),
    denote this with p_ctc(a_1^n-1, v, ...)
    for all n from 1 to N, v in Vocab
    
    Reference: prefix search decoding in alex graves' thesis
    Beam score is then average by lm score

    ASSUMING SEQUENCES ARE PADDED WITH BLANKS

    Note that in the vocab dimension there is also blank idx.
    Blank idx can be reused as eos, consider this in specific cases.

    No masking is applied here.

    T: max input time size

    F: model output feature dimension (incl. blank),
    regardless of whether eos is included or not

    B: batch dim

    S: max target sequence length

    :param log_probs: Log probs output by the model (T, B, F)
    :param targets: Target sequences (B, S)
    :param input_lengths: Input lengths (B,)
    :param blank_idx: Blank index in F dim
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :param eos_idx: If not None, then index of eos in F dim. Used when eos is present,
    e.g. joint training with LM or AED
    :return: (log_pref_scores_beams, log_gamma)
    
    log_pref_scores_beams (B, S, F). For batch b with ground truth a_0^N-1, [b, n, v] is
    p_CTC(a_0^n-1, v, ...)

    log_gamma (T, B, 2, S+1). [t, b, 0 or 1, s]: Forward probability of a_1^s (of batch b) until
    frame t such that paths end with 0 (non blank) or 1 (blank)
    '''
    device = log_probs.device
    input_time_size, batch_size, n_out = log_probs.shape
    max_seq_len = targets.shape[1]
    # convention for all dim size 2: 0 is n, 1 is b
    prev_label_all_beams = targets.unsqueeze(-1).expand(-1, -1, n_out) # (B, S, F), S here from 1 to max_seq_len, dim 2 is prev_label of all beams
    beams = torch.arange(n_out, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, max_seq_len, -1) # (B, S, F), S here from 0 to max_seq_len-1, dim 2 is beam
    prev_label_diff = beams[:, 1:, :] != prev_label_all_beams[:, :-1, :] # (B, S-1, F)
    # max_seq_len + 1 for the empty prefix
    log_gamma = torch.full((input_time_size, batch_size, 2, max_seq_len+1), log_zero, dtype=torch.float32, device=device) # (T, B, 2, S+1)
    log_gamma[:, :, 1, 0] = torch.cumsum(log_probs[:, :, blank_idx], dim=0).nan_to_num(neginf=log_zero)
    log_gamma[0, :, 0, 1] = log_probs[0].gather(-1, targets[:, :1].long()).view(-1)
    # (B, S, F), for first beam (empty prefix), all is True
    prev_label_diff = torch.cat([torch.full((batch_size, 1, n_out), torch.tensor(True), device=device), prev_label_diff], dim=1) 
    # log_pref_scores_beams (B, S, F) S here is prefixes from empty to a_1^S-1
    log_pref_scores_beams = torch.cat(
        [log_probs[0].unsqueeze(1), torch.full((batch_size, max_seq_len-1, n_out), log_zero, device=device)],
        dim=1
    )
    for t in range(1, input_time_size):
        log_probs_t_k_ground_truth = log_probs[t].gather(-1, targets.long()) # (B, S) log prob of all symbol at time frame t
        log_new_label_prob = torch.logaddexp( # (B, S, F) S here is prefixes from empty to a_1^S-1
            log_gamma[t-1, :, 1, :-1].unsqueeze(-1).expand(-1, -1, n_out).clone(),
            log_gamma[t-1, :, 0, :-1].unsqueeze(-1).expand(-1, -1, n_out).clone().where(prev_label_diff, torch.tensor(log_zero, device=device))
        )
        # (B, S) for both below
        log_gamma[t, :, 0, 1:] = log_probs_t_k_ground_truth + torch.logaddexp(log_new_label_prob.gather(-1, targets.unsqueeze(-1)).squeeze(-1), log_gamma[t-1, :, 0, 1:].clone())
        log_gamma[t, :, 1, 1:] = log_probs[t, :, blank_idx].unsqueeze(-1).expand((-1, max_seq_len)) + torch.logsumexp(log_gamma[t-1, :, :, 1:].clone(), dim=1)
        # (B, F) to (B, S, F), probs of all possible next labels at frame t, same no matter which prefix
        log_probs_t_k_all_beams = log_probs[t, :, :].unsqueeze(-1).expand((-1, -1, max_seq_len)).transpose(1, 2)
        input_time_mask = (t < input_lengths).to(device).unsqueeze(-1).expand((-1, max_seq_len*n_out)).view(-1, max_seq_len, n_out)
        log_pref_scores_beams = log_pref_scores_beams.logaddexp(
            (log_probs_t_k_all_beams + log_new_label_prob).where(input_time_mask, torch.tensor(log_zero, device=device))
        ) # the score of a beam should not change if t > input length
    if eos_idx is not None:
        log_pref_scores_beams[:, :, eos_idx] = torch.logsumexp(log_gamma[-1, :, :, :-1], dim=1)
    return log_pref_scores_beams, log_gamma


def kldiv_lm_ctc_loss(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_lm_score, # (B, S, F)
    blank_idx = 0,
    log_zero = -1e15, # maybe better than float min for preventing overflowing
    eos_idx = None, # used when eos is present in vocab, mainly when an attention model is present
):
    '''
    Compute the KL div from p_LM to p_CTC

    ASSUMING SEQUENCES ARE PADDED WITH BLANKS

    T: max input time size

    F: model output feature dimension (incl. blank),
    regardless of whether eos is included or not

    B: batch dim

    S: max target sequence length

    :param log_probs: Log probs output by the model (T, B, F)
    :param target: Target sequences (B, S)
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,)
    :param log_lm_score: log LM score of all possible words in vocab given ground truth context (B, S, F)
    blank is included in F dim but will not be used
    :param blank_idx: Blank index in F dim of log_probs
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :param eos_idx: If not None, then index of eos in F dim. Used when eos is present,
    e.g. joint training with LM or AED
    :return: KL Div Loss sum p_LM*log p_CTC
    '''
    device = log_probs.device
    input_time_size, batch_size, n_out = log_probs.shape
    max_seq_len = targets.shape[1]
    log_pref_scores_beams, log_gamma = log_ctc_pref_beam_scores(log_probs, targets, input_lengths, blank_idx, log_zero, eos_idx)
    # seq mask in (B, S)
    seq_mask = (torch.arange(max_seq_len).unsqueeze(0).expand((batch_size, -1)) < target_lengths.unsqueeze(-1).expand((-1, max_seq_len))).float().to(device)
    # sum_n=1^N sum_{a in V} lm_score(a) log p_ctc(a_1^n-1, a ,...)
    if blank_idx != eos_idx:
        out_idx = torch.arange(n_out)
        out_idx_wo_blank = out_idx[out_idx != blank_idx].long().to(device)
        numer = (log_pref_scores_beams*torch.exp(log_lm_score))[:, :, out_idx_wo_blank].sum(dim=-1).nan_to_num(neginf=log_zero)*seq_mask
    else: # blank_idx is reused as eos_idx in LM score
        numer = (log_pref_scores_beams*torch.exp(log_lm_score)).sum(dim=-1).nan_to_num(neginf=log_zero)*seq_mask
    # sum_n=1^N log p_ctc (a_1^n-1)
    denom = log_pref_scores_beams[:, :-1, :].gather(-1, targets[:, :-1].long().unsqueeze(-1)).view(-1, max_seq_len-1).nan_to_num(neginf=log_zero)*seq_mask[:, 1:]
    loss = (denom.sum() - numer.sum()).nan_to_num(neginf=log_zero)
    # sanity check
    # sum_{v in V} p prefix(a_1^{N-1}, v) should be equal to p prefix(a_1^{N-1}) - p forward(a_1^{N-1})
    # sum_pref_beams = torch.exp(torch.logaddexp(
    #     log_pref_scores_beams[:, :-1, out_idx_wo_blank].logsumexp(dim=-1),
    #     log_gamma[-1, :, :, 1:-1].logsumexp(dim=1))
    # )[:, 1:]
    # norm_term = torch.exp(denom[:, :])
    # print(sum_pref_beams)
    # print(norm_term)
    # There should have been renormalization here if eos is not considered, but we leave it for now
    return loss


def kldiv_ctc_lm_loss(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_lm_score, # (B, S, F)
    blank_idx = 0,
    log_zero = -1e15, # maybe better than float min for preventing overflowing
    eos_idx = None,
):
    '''
    Compute the KL div from p_CTC to p_LM

    ASSUMING SEQUENCES ARE PADDED WITH BLANKS

    T: max input time size

    F: model output feature dimension (incl. blank),
    regardless of whether eos is included or not

    B: batch dim

    S: max target sequence length

    :param log_probs: Log probs output by CTC (T, B, F)
    :param target: Target sequences (B, S)
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,)
    :param log_lm_score: log LM score of all possible words in vocab given ground truth context (B, S, F)
    blank is included in F dim but will not be used
    :param blank_idx: Blank index in F dim of log_probs
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :param eos_idx: If None, then blank_idx is reused as eos_idx in log_probs
    :return: KL Div Loss sum p_CTC*log p_LM
    '''
    device = log_probs.device
    input_time_size, batch_size, n_out_ctc = log_probs.shape
    n_out_lm = log_lm_score.shape[2]
    max_seq_len = targets.shape[1]
    log_pref_scores_beams, log_gamma = log_ctc_pref_beam_scores(log_probs, targets, input_lengths, blank_idx, log_zero, eos_idx)
    if eos_idx is None and n_out_ctc == n_out_lm: # blank of CTC is reused as eos with the LM
        log_pref_scores_beams[:, :, blank_idx] = torch.logsumexp(log_gamma[-1, :, :, :-1], dim=1)
    else:
        assert n_out_ctc == n_out_lm + 1,"If blank in CTC is not reused as eos, then n_out_ctc must be equal to n_out_lm + 1"
        # In this case, the blank has to be removed from beam dim of log_pref_scores_beams
        # It is up to the users that after removing blank, indices of CTC pref score beams match indices of LM
        out_idx = torch.arange(n_out_ctc)
        out_idx_wo_blank = out_idx[out_idx != blank_idx].long().to(device)
        log_pref_scores_beams = log_pref_scores_beams[:, :, out_idx_wo_blank]
    # log_ctc_norm_term (B, S-1, F), is the denominator of
    # p_ctc(v | a_1^n) = p_ctc(a_1^n, v, ...) / p_ctc(a_1^n, ...)
    # They should be properly normalized if eos is present
    # This is mainly for the other case (no eos)
    log_ctc_norm_term = torch.logsumexp(log_pref_scores_beams, dim=-1, keepdim=True)
    log_p_ctc = (log_pref_scores_beams - log_ctc_norm_term).detach()
    kl_div = torch.nn.functional.kl_div(
        input=log_lm_score,
        target=log_p_ctc,
        log_target=True,
        reduction="none",
    )
    seq_mask = get_seq_mask(target_lengths, max_seq_len, device) # seq mask (B, S)
    seq_mask = seq_mask.unsqueeze(-1).expand(-1, -1, n_out_lm) # seq mask in (B, S, F)
    loss = (kl_div*seq_mask).sum()
    manual_kldiv = (torch.exp(log_p_ctc)*(log_p_ctc - log_lm_score)).sum(dim=-1)
    return loss


def ctc_double_softmax_loss(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_lm_score, # (B, S, F)
    blank_idx = 0,
    log_zero = -1e15, # maybe better than float min for preventing overflowing
    eos_idx = None,
):
    """
    Compute the KL div from p_CTC to p_LM

    When eos 

    ASSUMING SEQUENCES ARE PADDED WITH BLANKS

    T: max input time size

    F: model output feature dimension (incl. blank),
    regardless of whether eos is included or not

    B: batch dim

    S: max target sequence length

    :param log_probs: Log probs output by CTC (T, B, F)
    :param target: Target sequences (B, S)
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,)
    :param log_lm_score: log LM score of all possible words in vocab given ground truth context (B, S, F)
    blank is included in F dim but will not be used
    :param blank_idx: Blank index in F dim of log_probs
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :param eos_idx: If None, then blank_idx is reused as eos_idx in log_probs
    :return: KL Div Loss sum p_CTC*log p_LM
    """
    