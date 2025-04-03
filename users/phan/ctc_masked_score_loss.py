"""
Some losses specifically for bidirectional ILM
Currently only standard KLDiv, and smoothing (sampling) loss
"""
import torch

from i6_experiments.users.phan.ctc_masked_score import ctc_masked_score

def ctc_bi_ilm_kldiv_loss(
    log_probs, # (T, B, F)
    targets, # (B, S)
    mask, # (S,), 1 = mask, 0 = no mask
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_lm_score, # (B, S, F-1)
    blank_idx = 10025,
    eos_idx = None,
    log_zero = -1e25, # maybe better than float min for preventing overflowing
):
    log_masked_probs, _, _, _, _ = ctc_masked_score( # (B, M, F-1)
        log_probs,
        targets,
        mask,
        input_lengths,
        target_lengths,
        blank_idx,
        eos_idx,
        log_zero,
    )

    log_lm_score_masks = log_lm_score[:, mask.bool(), :] # (B, M, F-1)
    kldiv = torch.nn.functional.kl_div( # (B, M, F-1)
        input=log_lm_score_masks,
        target=log_masked_probs.detach(),
        log_target=True,
        reduction="none",
    )

    # mask out loss for masked positions outside the target sequence
    batch_size = targets.shape[0]
    vocab_size = log_lm_score.shape[-1]
    mask_idxs = torch.nonzero(mask).squeeze(1).long() # (M,)
    masks_inside_max_lengths = (mask_idxs.unsqueeze(0).expand(batch_size, -1) < target_lengths.unsqueeze(-1).expand(-1, mask_idxs.shape[0])).float() # (B, M), 1 is inside, 0 is outside
    loss = kldiv * masks_inside_max_lengths.unsqueeze(-1).expand(-1, -1, vocab_size) # (B, M, F-1)
    return loss, masks_inside_max_lengths

# TODO: FINISH THIS
def ctc_bi_ilm_smoothing_kldiv_loss(
    log_probs, # (T, B, F)
    targets, # (B, S)
    mask, #(S,)
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_lm_score, # (B, S, F-1)
    blank_idx = 10025,
    eos_idx = None,
    log_zero = -1e25, # maybe better than float min for preventing overflowing
    ground_truth_weight = 0.5,
):
    '''
    Similar to sampling/smoothing ILM loss for autoregressive ILM. 
    
    Example: batch is [audio1, target1], [audio2, target2], ..., then compute loss for
    [audio 1 to n, target1], [audio 1 to n, target2]. Apply some weighting
    here, e.g. 1/2 for ground truth and 1/(2*(B-1)) for the rest

    ASSUMING SEQUENCES ARE PADDED WITH BLANKS AND NO EXPLICIT EOS IN VOCAB DIM.
    SAME TARGET MASK APPLIED TO ALL TARGET SEQUENCES.

    T: max input time size

    F: model output feature dimension (incl. blank),
    regardless of whether eos is included or not

    B: batch dim

    S: max target sequence length

    :param log_probs: Log probs output by CTC (T, B, F)
    :param targets: Target sequences (B, S) WITHOUT ANY SOS EOS SYMBOL
    :param input_lengths: Input lengths (B,)
    :param target_lengths: Target lengths (B,)
    :param log_lm_score: log LM score of all possible words in vocab given ground truth context (B, S+1, F)
    EOS of this should be blank of the CTC
    :param blank_idx: Blank index in F dim of log_probs
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :param eos_idx: If not None, this is used in the case there is EOS in the vocab.
        Instead of having one "extra" EOS in place of the blank index, the EOS score will
        be moved to the eos_idx instead. The LM score is then expected to have the same dimension
        as the vocab, not vocab + EOS.
    :param ground_truth_weight: Weight given to the loss of the ground truth
    :return: KL Div Loss sum p_CTC*log p_LM. Note that this loss is already averaged, be careful
    about this when passing to returnn
    '''
    device = log_probs.device
    input_time_size, batch_size, n_out = log_probs.shape
    log_probs = log_probs.transpose(0, 1).repeat(batch_size, 1, 1).transpose(0, 1)
    input_lengths = input_lengths.repeat(batch_size)
    targets = targets.repeat_interleave(batch_size, dim=0)
    # log_probs (T, B*B, F) targets (B*B, S)
    max_seq_len = targets.shape[1]
    log_masked_probs, _, _, _, _ = ctc_masked_score( # (B*B, M, F-1)
        log_probs,
        targets,
        mask,
        input_lengths,
        target_lengths,
        blank_idx,
        eos_idx,
        log_zero,
    )
    log_lm_score = log_lm_score.repeat_interleave(batch_size, dim=0)
    log_lm_score_masks = log_lm_score[:, mask.long(), :] # (B, M, F-1)

    kldiv = torch.nn.functional.kl_div( # (B, M, F-1)
        input=log_lm_score_masks,
        target=log_masked_probs.detach(),
        log_target=True,
        reduction="none",
    )
    if eos_idx is not None:
        n_out -= 1 # Because the EOS is moved from blank_idx to eos_idx
    seq_mask = get_seq_mask(target_lengths+1, max_seq_len+1, device) # seq mask (B, S+1)
    seq_mask_repeat = seq_mask.unsqueeze(-1).expand(-1, -1, n_out).repeat_interleave(batch_size, 0) # seq mask in (B*B, S+1, F)
    if ground_truth_weight == "average":
        ground_truth_weight = 1./batch_size
    if batch_size > 1:
        none_truth_weight = (1.-ground_truth_weight)/(batch_size-1)
    else:
        none_truth_weight = 0
    weight_diag_mat = torch.full((batch_size, batch_size), fill_value=none_truth_weight, device=device).fill_diagonal_(ground_truth_weight).flatten()
    loss = ((kl_div*seq_mask_repeat).sum(dim=-1).sum(dim=-1)*weight_diag_mat).sum() / seq_mask.sum()
    return loss