"""
Sequence-level ILM estimation for CTC
Starting point: p_{ILM}(w_1^N) = \sum_{X} Pr(X)*p_{CTC}(w_1^N | X)
Use RHS as teacher for LHS directly
"""

import torch

def kldiv_ctc_lm_sequence_level(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_lm_seq_probs, # (B)
    blank_idx = 0,
    log_zero = -1e25, # maybe better than float min for preventing overflowing
    ground_truth_weight = 0.5,
):
    '''
    Sequence-level ILM estimation for CTC
    Starting point: p_{ILM}(w_1^N) = \sum_{X} Pr(X)*p_{CTC}(w_1^N | X)
    Use RHS as teacher for LHS directly, i.e. the KL divergence is:

    \sum_{w_1^N} RHS log (RHS / p_{ILM}(w_1^N))

    Pr(X) is renomalized for the acoustic sequences in the batch.
    More weights might be given to the true X of the w_1^N.

    This is not so true but we leave it here as clue for any potential future bugs:
    ASSUMING SEQUENCES ARE PADDED WITH BLANKS AND NO EXPLICIT EOS IN VOCAB DIM

    If EOS is in the vocab, its prob should be zero.

    T: max input time size

    F: model output feature dimension (incl. blank),
    regardless of whether eos is included or not

    B: batch dim

    S: max target sequence length

    :param log_probs: Log probs output by CTC (T, B, F)
    :param targets: Target sequences (B, S) WITHOUT ANY SOS EOS SYMBOL
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,)
    :param log_lm_seq_prob: log LM score of all target sequence in the batch (B,).
        This is on sequence level, i.e. the loss is sum_{n} log p(w_n | w_1^{n-1})
    :param blank_idx: Blank index in F dim of log_probs
    :param log_zero: Value of log zero. Default to -1e25 to prevent overflow comparing to float32 min
    :param ground_truth_weight: Weight given to the true X in the approximation of Pr(X)
    :return: KL Div Loss sum p_CTC*log p_LM. No need to normalize it (?), since it is
        on sequence level anyway.
    '''
    device = log_probs.device
    input_time_size, batch_size, n_out = log_probs.shape
    log_probs_repeat = log_probs.detach().transpose(0, 1).repeat(batch_size, 1, 1).transpose(0, 1)
    input_lengths_repeat = input_lengths.repeat(batch_size)
    targets_repeat_interleave = targets.repeat_interleave(batch_size, dim=0)
    target_lengths_repeat_interleave = target_lengths.repeat_interleave(batch_size, dim=0)
    # log_probs (T, B*B, F) targets (B*B, S)
    log_p_ctc = - torch.nn.functional.ctc_loss( # (B*B), don't forget the minus
        log_probs_repeat,
        targets_repeat_interleave,
        input_lengths_repeat,
        target_lengths_repeat_interleave,
        blank=blank_idx,
        reduction="none",
    )
    log_p_ctc = log_p_ctc.view(batch_size, batch_size) # (B, B), actually (W, X) due to the repeat and repeat_interleave
    if ground_truth_weight == "average":
        ground_truth_weight = 1./batch_size
    if batch_size > 1:
        none_truth_weight = (1.-ground_truth_weight)/(batch_size-1)
    else:
        none_truth_weight = 0
    weight_diag_mat = torch.full( # square matrix to assign Pr(X) for each label sequence
        (batch_size, batch_size),
        fill_value=none_truth_weight,
        device=device
    ).fill_diagonal_(ground_truth_weight)
    log_p_ctc_weighted = (weight_diag_mat.log() + log_p_ctc) # (B, B), position (w, x) = log [Pr(x)*p_{ctc}(w|x)]
    log_probs_teacher = log_p_ctc_weighted.logsumexp(1) # (B,), position (w) = log [sum_{x} Pr(x)*p_{ctc}(w|x)]
    kl_div = torch.nn.functional.kl_div(
        input=log_lm_seq_probs,
        target=log_probs_teacher.detach(),
        log_target=True,
        reduction="none",
    )
    loss = kl_div.sum()
    return loss


def kldiv_ctc_lm_sequence_level_ground_truth_weight_1(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_lm_seq_probs, # (B)
    blank_idx = 0,
    log_zero = -1e25, # maybe better than float min for preventing overflowing
):
    '''
    Same as kldiv_ctc_lm_sequence_level but in the case the ground truth weight = 1,
    then simply do normal batch forward of ctc --> if not faster at least lighter

    Sequence-level ILM estimation for CTC
    Starting point: p_{ILM}(w_1^N) = \sum_{X} Pr(X)*p_{CTC}(w_1^N | X)
    Use RHS as teacher for LHS directly, i.e. the KL divergence is:

    \sum_{w_1^N} RHS log (RHS / p_{ILM}(w_1^N))

    Pr(X) is renomalized for the acoustic sequences in the batch.
    More weights might be given to the true X of the w_1^N.

    This is not so true but we leave it here as clue for any potential future bugs:
    ASSUMING SEQUENCES ARE PADDED WITH BLANKS AND NO EXPLICIT EOS IN VOCAB DIM

    If EOS is in the vocab, its prob should be zero.

    T: max input time size

    F: model output feature dimension (incl. blank),
    regardless of whether eos is included or not

    B: batch dim

    S: max target sequence length

    :param log_probs: Log probs output by CTC (T, B, F)
    :param targets: Target sequences (B, S) WITHOUT ANY SOS EOS SYMBOL
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,)
    :param log_lm_seq_prob: log LM score of all target sequence in the batch (B,).
        This is on sequence level, i.e. the loss is sum_{n} log p(w_n | w_1^{n-1})
    :param blank_idx: Blank index in F dim of log_probs
    :param log_zero: Value of log zero. Default to -1e25 to prevent overflow comparing to float32 min
    :param ground_truth_weight: Weight given to the true X in the approximation of Pr(X)
    :return: KL Div Loss sum p_CTC*log p_LM. No need to normalize it (?), since it is
        on sequence level anyway.
    '''
    device = log_probs.device
    input_time_size, batch_size, n_out = log_probs.shape
    # log_probs (T, B*B, F) targets (B*B, S)
    log_p_ctc = - torch.nn.functional.ctc_loss( # (B*B), don't forget the minus
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_idx,
        reduction="none",
    )
    kl_div = torch.nn.functional.kl_div(
        input=log_lm_seq_probs,
        target=log_p_ctc.detach(),
        log_target=True,
        reduction="none",
    )
    loss = kl_div.sum()
    return loss
