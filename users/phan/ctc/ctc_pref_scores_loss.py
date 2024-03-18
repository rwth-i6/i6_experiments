"""
This module implements some loss related to
the CTC prefix score p_CTC(a_1^N,...)
"""
import torch
from i6_experiments.users.phan.utils import get_seq_mask


def log_ctc_pref_beam_scores(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    blank_idx = 0,
    log_zero = -1e25, # maybe better than float min for preventing overflowing
):
    '''
    Given log probs of times and ground truths,
    compute all ctc prefix score p_ctc(a_1^n-1, v, suffix=empty or not empty),
    denote this with p_ctc(a_1^n-1, v, ...)
    for all n from 1 to N, v in Vocab
    
    Reference: prefix search decoding in alex graves' thesis
    Beam score is then average by lm score

    ASSUMING SEQUENCES ARE PADDED WITH BLANKS AND NO EXPLICIT EOS IN VOCAB DIM

    Note that in the vocab dimension there is also blank idx, which will be
    used as EOS (full forward prob of current hypothesis).

    No masking is applied here. Pay attention to this.

    T: max input time size

    F: model output feature dimension (incl. blank),
    regardless of whether eos is included or not

    B: batch dim

    S: max target sequence length

    :param log_probs: Log probs output by the model (T, B, F)
    :param targets: Target sequences (B, S) WITHOUT ANY SOS EOS SYMBOL
    :param input_lengths: Input lengths (B,)
    :param blank_idx: Blank index in F dim
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :return: (log_pref_scores_beams, log_gamma)
    
    log_pref_scores_beams (B, S+1, F). For batch b with ground truth a_0^N-1, [b, n, v] is
    p_CTC(a_0^n-1, v, ...). Blank index will be reused to model full forward prob p_CTC(a_0^n-1).
    No masking is applied here yet.

    log_gamma (T, B, 2, S+1). [t, b, 0 or 1, s]: Forward probability of a_1^s (of batch b) until
    frame t such that paths end with 0 (non blank) or 1 (blank)
    '''
    device = log_probs.device
    input_time_size, batch_size, n_out = log_probs.shape
    max_seq_len = targets.shape[1]
    # convention for all dim size 2: 0 is n, 1 is b
    # max_seq_len + 1 for the empty prefix
    log_gamma = torch.full((input_time_size, batch_size, 2, max_seq_len+1), log_zero, dtype=torch.float32, device=device) # (T, B, 2, S+1)
    log_gamma[:, :, 1, 0] = torch.cumsum(log_probs[:, :, blank_idx], dim=0).nan_to_num(neginf=log_zero)
    log_gamma[0, :, 0, 1] = log_probs[0].gather(-1, targets[:, :1].long()).view(-1)
    # (B, S, F), for first beam (empty prefix), all is True
    prev_label_all_beams = targets.unsqueeze(-1).expand(-1, -1, n_out) # (B, S, F), S here from 1 to max_seq_len, dim 2 is prev_label of all beams
    beams = torch.arange(n_out, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, max_seq_len, -1) # (B, S, F), S here from 0 to max_seq_len-1, dim 2 is beam
    prev_label_diff = beams != prev_label_all_beams # (B, S, F)
    prev_label_diff = torch.cat([torch.full((batch_size, 1, n_out), torch.tensor(True), device=device), prev_label_diff], dim=1) 
    # log_pref_scores_beams (B, S+1, F) S+1 here is prefixes from empty to a_1^S
    log_pref_scores_beams = torch.cat(
        [log_probs[0].unsqueeze(1), torch.full((batch_size, max_seq_len, n_out), log_zero, device=device)],
        dim=1
    )
    for t in range(1, input_time_size):
        log_probs_t_k_ground_truth = log_probs[t].gather(-1, targets.long()) # (B, S) log prob of all symbol at time frame t
        log_new_label_prob = torch.logaddexp( # (B, S+1, F) S here is prefixes from empty to a_1^S
            log_gamma[t-1, :, 1, :].unsqueeze(-1).expand(-1, -1, n_out).clone(),
            log_gamma[t-1, :, 0, :].unsqueeze(-1).expand(-1, -1, n_out).clone().where(prev_label_diff, torch.tensor(log_zero, device=device))
        )
        # (B, S) for both below
        log_gamma[t, :, 0, 1:] = log_probs_t_k_ground_truth + torch.logaddexp(log_new_label_prob[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1), log_gamma[t-1, :, 0, 1:].clone())
        log_gamma[t, :, 1, 1:] = log_probs[t, :, blank_idx].unsqueeze(-1).expand((-1, max_seq_len)) + torch.logsumexp(log_gamma[t-1, :, :, 1:].clone(), dim=1)
        # (B, F) to (B, S+1, F), probs of all possible next labels at frame t, same no matter which prefix
        log_probs_t_k_all_beams = log_probs[t, :, :].unsqueeze(-1).expand((-1, -1, max_seq_len+1)).transpose(1, 2)
        input_time_mask = (t < input_lengths).to(device).unsqueeze(-1).expand((-1, (max_seq_len+1)*n_out)).view(-1, max_seq_len+1, n_out)
        log_pref_scores_beams = log_pref_scores_beams.logaddexp(
            (log_probs_t_k_all_beams + log_new_label_prob).where(input_time_mask, torch.tensor(log_zero, device=device))
        ) # the score of a beam should not change if t > input length
    # Reuse the blank idx as EOS, i.e. full forward prob of current hypothesis
    log_pref_scores_beams[:, :, blank_idx] = torch.logsumexp(log_gamma[-1, :, :, :], dim=1)
    return log_pref_scores_beams, log_gamma


# This is currently not maintained
def kldiv_lm_ctc_loss(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_lm_score, # (B, S, F)
    blank_idx = 0,
    log_zero = -1e25, # maybe better than float min for preventing overflowing
):
    '''
    Compute the KL div from p_LM to p_CTC

    ASSUMING SEQUENCES ARE PADDED WITH BLANKS AND NO EXPLICIT EOS IN VOCAB DIM

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
    log_pref_scores_beams, log_gamma = log_ctc_pref_beam_scores(log_probs, targets, input_lengths, blank_idx, log_zero)
    # seq mask in (B, S)
    seq_mask = (torch.arange(max_seq_len).unsqueeze(0).expand((batch_size, -1)) < target_lengths.unsqueeze(-1).expand((-1, max_seq_len))).float().to(device)
    # sum_n=1^N sum_{a in V} lm_score(a) log p_ctc(a_1^n-1, a ,...)
    # if blank_idx != eos_idx:
    #     out_idx = torch.arange(n_out)
    #     out_idx_wo_blank = out_idx[out_idx != blank_idx].long().to(device)
    #     numer = (log_pref_scores_beams*torch.exp(log_lm_score))[:, :, out_idx_wo_blank].sum(dim=-1).nan_to_num(neginf=log_zero)*seq_mask
    # else: # blank_idx is reused as eos_idx in LM score
    #     numer = (log_pref_scores_beams*torch.exp(log_lm_score)).sum(dim=-1).nan_to_num(neginf=log_zero)*seq_mask
    # # sum_n=1^N log p_ctc (a_1^n-1)
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
    log_zero = -1e25, # maybe better than float min for preventing overflowing
    eos_idx = None,
):
    '''
    Compute the KL div from p_CTC to p_LM. The blank in output dim of CTC will be
    reused as EOS. Make sure the LM match this.

    ASSUMING SEQUENCES ARE PADDED WITH BLANKS AND NO EXPLICIT EOS IN VOCAB DIM

    T: max input time size

    F: model output feature dimension (incl. blank),
    regardless of whether eos is included or not

    B: batch dim

    S: max target sequence length

    :param log_probs: Log probs output by CTC (T, B, F)
    :param targets: Target sequences (B, S) WITHOUT ANY SOS EOS SYMBOL
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,)
    :param log_lm_score: log LM score of all possible words in vocab given ground truth context (B, S, F)
    EOS of this should be blank of the CTC
    :param blank_idx: Blank index in F dim of log_probs
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :return: KL Div Loss sum p_CTC*log p_LM
    '''
    device = log_probs.device
    input_time_size, batch_size, n_out = log_probs.shape
    max_seq_len = targets.shape[1]
    log_pref_scores_beams, _ = log_ctc_pref_beam_scores(
        log_probs,
        targets,
        input_lengths,
        blank_idx,
        log_zero,
    )
    # renormalize to have p_ctc(v|hypothesis) in output dim
    log_p_ctc = log_pref_scores_beams.log_softmax(dim=-1)
    kl_div = torch.nn.functional.kl_div(
        input=log_lm_score,
        target=log_p_ctc,
        log_target=True,
        reduction="none",
    )
    seq_mask = get_seq_mask(target_lengths+1, max_seq_len+1, device) # seq mask (B, S+1)
    seq_mask = seq_mask.unsqueeze(-1).expand(-1, -1, n_out) # seq mask in (B, S+1, F)
    loss = (kl_div*seq_mask).sum()
    return loss


def ctc_double_softmax_loss(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_lm_score, # (B, S, F)
    am_scale,
    lm_scale,
    blank_idx = 0,
    log_zero = -1e25, # maybe better than float min for preventing overflowing
):
    """
    Double softmax for CTC

    ASSUMING SEQUENCES ARE PADDED WITH BLANKS AND NO EXPLICIT EOS IN VOCAB DIM

    T: max input time size

    F: model output feature dimension (incl. blank),
    regardless of whether eos is included or not

    B: batch dim

    S: max target sequence length

    :param log_probs: Log probs output by CTC (T, B, F)
    :param target: Target sequences (B, S) WITHOUT ANY EOS SOS
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,) WITHOUT ANY EOS SOS
    :param log_lm_score: if (B, S, F), then log LM score of all possible words in vocab 
    given ground truth context.
    if (B, S), then log LM score of target sequences.
    EOS idx of this should be blank idx of the CTC
    :param blank_idx: Blank index in F dim of log_probs
    :param am_scale: AM scale for CTC score
    :param lm_scale: LM scale for LM score
    :param log_zero: Value of log zero. Default to -1e25 to prevent overflow comparing to float32 min
    :return: Double softmax loss with CTC as AM and LM score as training LM
    """
    device = log_probs.device
    input_time_size, batch_size, n_out = log_probs.shape
    max_seq_len = targets.shape[1]
    log_pref_scores_beams, log_gamma = log_ctc_pref_beam_scores(
        log_probs,
        targets,
        input_lengths,
        blank_idx,
        log_zero,
    )
    # renormalize to have p_ctc(v|hypothesis) in output dim
    log_p_ctc = log_pref_scores_beams.log_softmax(dim=-1)
    # take out correct indices of target sequences
    targets_eos = torch.cat(
        [targets, torch.zeros((batch_size, 1), device=targets.device)],
        dim=1,
    ).long()
    # this is why it's called double softmax
    log_p_am_ref = log_p_ctc.gather(-1, targets_eos.unsqueeze(-1)).view(-1, max_seq_len+1)
    log_p_lm_ref = log_lm_score.gather(-1, targets_eos.unsqueeze(-1)).view(-1, max_seq_len+1)
    log_denom_ref = (am_scale*log_p_ctc + lm_scale*log_lm_score).logsumexp(dim=-1)
    double_softmax = am_scale*log_p_am_ref + lm_scale*log_p_lm_ref - log_denom_ref
    seq_mask = get_seq_mask(target_lengths+1, max_seq_len+1, device)
    loss = -(double_softmax*seq_mask).sum()
    return loss


def normalization_check(
    log_ctc_pref_beam_scores,
    targets,
    target_lengths,
):
    """
    Due to doubts of the CTC prefox score implementation,
    this checks if the prefix scores are properly "normalized"

    It checks whether sum_v p(a_1^n,v) = p(a_1^n)
    for any given prefix a_1^n of ground truth

    Note that v here must include eos, since the prefix score
    definition includes the case of a_1^n being the whole sequence.

    F: model output feature dimension (incl. eos)

    B: batch dim

    S: max target sequence length

    :param log_ctc_pref_beam_scores: The log CTC prefix scores (B, S+1, F)
    :param target: Target sequences (B, S) WITHOUT ANY SOS EOS SYMBOL
    :param target_lengths: Lengths of target sequences (B,)
    """
    batch_size, max_seq_len = targets.shape
    log_sum_p_beams = torch.logsumexp(log_ctc_pref_beam_scores, dim=-1)
    log_p_true_labels = log_ctc_pref_beam_scores[:, :-1, :].gather(-1, targets.unsqueeze(-1).long()).view(-1, max_seq_len)
    # sum_v p(v) = p(empty is prefix) = 1
    # So log p ground truth should be 0 at first
    log_p_true_labels = torch.cat([torch.zeros((batch_size, 1),device=targets.device), log_p_true_labels], dim=-1)
    seq_mask = get_seq_mask(target_lengths+1, max_seq_len+1, targets.device)
    log_sum_p_beams = log_sum_p_beams*seq_mask
    log_p_true_labels = log_p_true_labels*seq_mask
    print(log_sum_p_beams.cpu().numpy())
    print(log_p_true_labels.cpu().numpy())
    return torch.isclose(log_sum_p_beams, log_p_true_labels) # is close parameter
    