import torch
from i6_experiments.users.phan.utils.masking import get_seq_mask
from i6_experiments.users.phan.utils.math import logsubstractexp

def ctc_masked_score(
    log_probs, # (T, B, F)
    targets,
    mask,
    input_lengths,
    target_lengths,
    blank_idx=0,
    log_zero=-1e15,
):
    """
    Given target sequences, label masking and CTC outputs,
    compute p(a mask is w | the masked sequence, x_1^T)
    for a BERT-like masked LM.

    Note that in this implementation, all sequences in the
    batch will be masked by the same masking. This makes it
    feasible to implement forward-backward calculation efficiently.
    Therefore, sequences in batch should have roughly the same lengths.

    Note that in the vocab dimension of the result there is no EOS, unlike
    the CTC prefix score, because EOS score can not be computed reliably
    for the last token. Also, the middle masked tokens should never be EOS.

    M is number of masked positions.

    :param log_probs: CTC outputs in (T, B, F)
    :param targets: Target sequences (B, S) without EOS
    :param mask: A masking for a single sequence (S,). 1 = mask, 0 = no mask
    All sequences in the batch will be masked by this masking. dtype: long
    :param input_lengths: :Lengths acoustic inputs (B,)
    :param target_lengths: Lengths target sequences (B,) including EOS
    :param blank_idx: Index of blank in log_probs. Don't touch this, only works
    with blank_idx = 0
    :param log_zero: Big enough negative number to represent log(0)

    :returns:
    log_masked_probs: (B, M, F-1) conditional log probs p(<mask> is w | masked seq, acoustic)
    for each possible token for each masked position. Pay attention to "invalid" mask pos,
    i.e. the ones outside of a sequence.

    forward: (T, B, S+1, 2) forward variables. Sum of all partial alignments up to frame t,
    position s, where the last frame is non-blank (0) or blank (1). In the S+1 dim, the first
    label is virtual BOS.

    backward: (T, B, S+1, 2) Sum of all partial alignments from T down to frame t,
    from S down to position s, where the first frame is non-blank (0) or blank (1). In the S+1 dim,
    the last label is virtual EOS.

    forward_masked: (T, B, M, F-1) Auxiliary forward variables for a masked position, i.e. the forward
    as if the <mask> was replaced by a true label w.

    backward_masked: (T, B, M, F-1) Auxiliary backward variables for a masked position, i.e. the backward
    as if the <mask> was replaced by a true label w.
    """
    input_time_size, batch_size, n_out = log_probs.shape
    device = log_probs.device
    _, max_seq_len = targets.shape
    assert len(mask.shape) == 1 and mask.shape[0] == max_seq_len, "Shape of masking must be (S,)"
    batch_mask = mask.unsqueeze(0).expand(batch_size, -1)
    masked_pos = mask.nonzero().squeeze(-1)
    n_masked_pos = mask.sum() # M: number of masked pos per seq

    # Indices of non-blank labels in output dim
    out_idx = torch.arange(n_out)
    out_idx_no_blank = out_idx[out_idx != blank_idx]

    # Forward variables
    # for last dim: 0 is n, 1 is b
    # S+1 for virtual BOS [BOS, a_1^1, a_1^2, ..., a_1^{S-1}, a_1^S]
    # [t, b, s, 0 or 1] forward prob of (t, s) for batch (b) where last frame is 0 (non-blank) or 1 (blank)
    forward = torch.full((input_time_size, batch_size, max_seq_len+1, 2), log_zero).to(device) # (T, B, S+1, 2)
    # forward probs if a mask is replaced by a token
    forward_masked = torch.full((input_time_size, batch_size, n_masked_pos, n_out-1), log_zero).to(device) # (T, B, M, F-1)
    prev_mask = torch.concat([torch.tensor([0]), mask[:-1]], dim=0) # is the previous position masked?
    batch_prev_mask = prev_mask.unsqueeze(0).expand(batch_size, -1)
    prev_label_diff = torch.concat([torch.zeros(batch_size, 1).to(device), (targets[:, 1:] != targets[:, :-1])], dim=-1) # is this label same as the previous?

    # Init
    # Output empty seq prob
    forward[:, :, 0, 1] = log_probs[:, :, blank_idx].cumsum(dim=0)
    # Init differently for the first position
    if mask[0] == 0:
        forward[0, :, 1, 0] = log_probs[0].gather(-1, targets[:, :1].long()).view(-1)
    else:
        forward_masked[0, :, 0, :] = log_probs[0, :, 1:].clone()
        forward[0, :, 1, 0] = forward_masked[0, :, 0, :].logsumexp(dim=-1)

    # Indices where current position is (not) mask and previous is (not) mask
    # Use to access the S+1 dimension in forward (shift by +1)
    # previous of pos 0 is not mask (it is empty symbol)
    s_idx = torch.arange(max_seq_len).to(device)
    mask_prev_mask = mask * prev_mask
    mask_prev_mask_idx = s_idx[mask_prev_mask.bool()]
    mask_prev_not_mask = mask * (1 - prev_mask)
    mask_prev_not_mask_idx = s_idx[mask_prev_not_mask.bool()]
    not_mask_prev_mask = (1 - mask) * prev_mask
    not_mask_prev_mask_idx = s_idx[not_mask_prev_mask.bool()]
    not_mask_prev_not_mask = (1 - mask) * (1 - prev_mask)
    not_mask_prev_not_mask_idx = s_idx[not_mask_prev_not_mask.bool()]
    # Indices to access the M dimension of forward_masked array
    # These are to access the correct portion of which masks in the targets
    full_midx = torch.arange(n_masked_pos)

    # Cur is masked, prev is masked case
    mask_prev_mask_midx = full_midx[mask_prev_mask[masked_pos].bool()]
    mask_prev_mask_midx_prev = mask_prev_mask_midx - 1
    # Cur is masked, prev is not masked
    mask_prev_not_mask_midx = full_midx[mask_prev_not_mask[masked_pos].bool()]
    # Cur is not masked, prev is masked
    not_mask_prev_mask_midx = full_midx[torch.cat([not_mask_prev_mask[1:], torch.tensor([0])], dim=0)[masked_pos].bool()]

    # This is used for case a_s = <ms>, a_{s-1} != <mask>
    targets_prev = torch.concat([torch.full((batch_size, 1), -9999).to(device), targets[:, :-1]], dim=1)
    # (B, |mask_prev_not_mask_idx|, F)
    all_w_diff_prev = torch.arange(start=1, end=n_out).unsqueeze(0).unsqueeze(0).expand(batch_size, len(mask_prev_not_mask_idx), -1).to(device) != \
        targets_prev.gather(-1, mask_prev_not_mask_idx.unsqueeze(0).expand(batch_size, -1).to(device)).unsqueeze(-1).expand(-1, -1, n_out-1).to(device)

    for t in range(1, input_time_size):
        # Last frame is blank
        forward[t, :, :, 1] = forward[t-1, :, :, :].clone().logsumexp(dim=-1) + log_probs[t, :, blank_idx].unsqueeze(-1).expand(-1, max_seq_len+1)
        # Last frame is not blank, 4 sub-cases:
        # For accessing forward, idx+1 is current, idx is prev label
        # For accessing log_probs and targets, idx is current
        # a_s != <mask>, a_{s-1} != <mask>, standard CTC transitions
        idx = not_mask_prev_not_mask_idx
        log_probs_targets_t = log_probs[t, :, :].gather(-1, targets.long()) # log probs of target label at current frame
        forward[t, :, idx+1, 0] = log_probs_targets_t[:, idx] + \
            torch.stack([
                forward[t-1, :, idx+1, 0].clone(),
                forward[t-1, :, idx, 1].clone(),
                torch.where(
                    prev_label_diff[:, idx].bool(),
                    forward[t-1, :, idx, 0].clone(),
                    torch.tensor(log_zero).to(device),
                )
            ], dim=-1).logsumexp(dim=-1)
        # a_s != <mask>, a_{s-1} = <mask>
        idx = not_mask_prev_mask_idx
        midx = not_mask_prev_mask_midx
        targets_shifted = targets[:, idx] - 1 # Because blank is 0
        targets_shifted[targets_shifted < 0] = 0
        forward[t, :, idx+1, 0] = log_probs_targets_t[:, idx] + logsubstractexp(
                torch.stack([
                    forward[t-1, :, idx, 0].clone(),
                    forward[t-1, :, idx, 1].clone(),
                    forward[t-1, :, idx+1, 0].clone()
                ], dim=-1).logsumexp(dim=-1),
                forward_masked[t-1, :, midx, :].gather(-1, targets_shifted.unsqueeze(-1)).squeeze(-1),
                log_zero,
            )
        # a_s = <mask>, a_{s-1} != <mask>
        idx = mask_prev_not_mask_idx
        midx = mask_prev_not_mask_midx
        forward_masked[t, :, midx, :] = log_probs[t, :, out_idx_no_blank].unsqueeze(1).expand(-1, len(midx), -1) + \
            torch.stack([
                forward_masked[t-1, :, midx, :].clone(),
                forward[t-1, :, idx, 1].clone().unsqueeze(-1).expand(-1, -1, n_out-1),
                torch.where(
                    all_w_diff_prev,
                    forward[t-1, :, idx, 0].clone().unsqueeze(-1).expand(-1, -1, n_out-1),
                    torch.tensor(log_zero).to(device),
                )
            ], dim=-1).logsumexp(dim=-1)
        forward[t, :, idx+1, 0] = forward_masked[t, :, midx, :].logsumexp(dim=-1)
        # a_s = <mask>, a_{s-1} = <mask>
        idx = mask_prev_mask_idx
        midx = mask_prev_mask_midx
        midx_prev = mask_prev_mask_midx_prev
        forward_masked[t, :, midx, :] = log_probs[t, :, 1:].unsqueeze(1).expand(-1, len(midx), -1) + \
            logsubstractexp(
                torch.stack([
                    forward_masked[t-1, :, midx, :],
                    forward[t-1, :, idx, 1].clone().unsqueeze(-1).expand(-1, -1, n_out-1),
                    forward[t-1, :, idx, 0].unsqueeze(-1).expand(-1, -1, n_out-1),
                ], dim=-1).logsumexp(dim=-1),
                forward_masked[t-1, :, midx_prev, :],
                log_zero
            )
        forward[t, :, idx+1, 0] = forward_masked[t, :, midx, :].logsumexp(dim=-1)


    # Backward variables
    # Almost the same as above, but have to be more careful due to the padding symbol
    # for last dim: 0 is n, 1 is b
    # S+1 for virtual EOS [a_1^1, a_1^2, ..., a_1^{S-1}, a_1^S, a_1^{S+1}=EOS]
    # [t, b, s, 0 or 1] backward prob of (t, s) for batch (b) where first frame is 0 (non-blank) or 1 (blank)
    backward = torch.full((input_time_size, batch_size, max_seq_len+1, 2), log_zero).to(device) # (T, B, S+1, 2)
    # backward probs if a mask is replaced by a token
    backward_masked = torch.full((input_time_size, batch_size, n_masked_pos, n_out-1), log_zero).to(device) # (T, B, M, F-1)
    next_mask = torch.concat([mask[1:], torch.tensor([0])], dim=0) # is the next position masked?
    batch_next_mask = next_mask.unsqueeze(0).expand(batch_size, -1)
    next_label_diff = torch.concat([(targets[:, :-1] != targets[:, 1:]), torch.zeros(batch_size, 1).to(device)], dim=-1) # is this label same as the next?

    # Some indices to reverse the padded sequences
    rev_time_idx = torch.stack([torch.cat([torch.arange(T-1, -1, -1), torch.arange(T, input_time_size)], dim=0) for T in input_lengths], dim=0).to(device)
    # Init
    # Output empty seq prob
    log_probs_blank_rev = log_probs[:, :, blank_idx].gather(0, rev_time_idx.transpose(0, 1))
    log_probs_empty_seq = log_probs_blank_rev.cumsum(dim=0).gather(0, rev_time_idx.transpose(0, 1)).to(device)
    time_mask = get_seq_mask(input_lengths, input_time_size, input_lengths.device).bool()
    log_probs_empty_seq = torch.where(time_mask.transpose(0, 1), log_probs_empty_seq, torch.tensor(log_zero).to(device)).to(device)
    backward[:, :, :, 1] = torch.scatter(backward[:, :, :, 1], 2, target_lengths.unsqueeze(-1).unsqueeze(0).expand(input_time_size, -1, -1).to(device), log_probs_empty_seq.unsqueeze(-1))

    # Now init last position based on if it's mask or not. This is trickier
    # than forward because the last label positions are not the same

    last_is_masked = torch.gather(batch_mask, 1, (target_lengths-1).unsqueeze(1)).squeeze(1) # (B,)
    batch_idx = torch.arange(batch_size)
    last_label_idx = (target_lengths - 1) # (B,)

    # Init for batches where last pos is not masked
    bool_idx = ~last_is_masked.bool()
    batches = batch_idx[bool_idx] # Batch where last label is not masked
    last_frames = tuple((input_lengths-1)[bool_idx].tolist())
    last_labels_idx = tuple(last_label_idx[bool_idx].tolist())
    last_labels = tuple(targets[batches, :].gather(-1, last_label_idx[bool_idx].unsqueeze(-1).to(device)).squeeze(-1).tolist())
    backward[last_frames, batches, last_labels_idx, 0] = log_probs[last_frames, batches, last_labels]


    # Init for batches where last pos is masked
    bool_idx = last_is_masked.bool()
    batches = batch_idx[bool_idx] # Batch where last label is masked
    last_frames = tuple((input_lengths-1)[bool_idx].tolist())
    last_labels_idx = last_label_idx[bool_idx]
    last_labels = tuple(targets[batches, :].gather(-1, last_label_idx[bool_idx].unsqueeze(-1).to(device)).squeeze(-1).tolist())
    # Now we have to map last_labels_idx to the indices in M dim of backward_masked
    b_masked_pos = masked_pos.unsqueeze(0).expand(len(batches), -1)
    b_last_labels_idx = last_labels_idx.unsqueeze(-1).expand(-1, n_masked_pos)
    b_full_midx = full_midx.unsqueeze(0).expand(len(batches), -1)
    last_labels_idx = tuple(last_labels_idx.tolist())
    # where to access the corresponding variables in the (T,B,M,F-1) array
    last_labels_midx = b_full_midx[b_masked_pos.cpu() == b_last_labels_idx.cpu()] # seems alright
    last_labels_midx = tuple(last_labels_midx.tolist())
    batches = tuple(batches.tolist())
    backward_masked[last_frames, batches, last_labels_midx, :] = log_probs[last_frames, batches, 1:]
    backward[last_frames, batches, last_labels_idx, 0] = backward_masked[last_frames, batches, last_labels_midx, :].logsumexp(dim=-1)

    # Now to the recursion
    # Rules: only update if t < T and s < S ("inside the lattice")
    # Leave everything outside as it is
    # To do this, apply time and label position mask

    # Indices for the 4 cases like above
    mask_next_mask = mask * next_mask
    mask_next_mask_idx = s_idx[mask_next_mask.bool()]
    mask_next_not_mask = mask * (1 - next_mask)
    mask_next_not_mask_idx = s_idx[mask_next_not_mask.bool()]
    not_mask_next_mask = (1 - mask) * next_mask
    not_mask_next_mask_idx = s_idx[not_mask_next_mask.bool()]
    not_mask_next_not_mask = (1 - mask) * (1 - next_mask)
    not_mask_next_not_mask_idx = s_idx[not_mask_next_not_mask.bool()]

    # Index accessor for M dim of backward_masked
    # Cur is masked, next is masked
    mask_next_mask_midx = full_midx[mask_next_mask[masked_pos].bool()]
    mask_next_mask_midx_next = mask_next_mask_midx + 1
    # Cur is masked, next is not masked
    mask_next_not_mask_midx = full_midx[mask_next_not_mask[masked_pos].bool()]
    # Cur is not masked, next is masked
    not_mask_next_mask_midx = full_midx[torch.cat([torch.tensor([0]), not_mask_next_mask[:-1]], dim=0)[masked_pos].bool()]

    # This is used for case a_s = <ms>, a_{s+1} != <mask>
    targets_next = torch.concat([targets[:, 1:], torch.full((batch_size, 1), -9999).to(device)], dim=1)
    # (B, |mask_next_not_mask_idx|, F)
    all_w_diff_next = torch.arange(start=1, end=n_out).unsqueeze(0).unsqueeze(0).expand(batch_size, len(mask_next_not_mask_idx), -1).to(device) != \
        targets_next.gather(-1, mask_next_not_mask_idx.unsqueeze(0).expand(batch_size, -1)).unsqueeze(-1).expand(-1, -1, n_out-1).to(device)

    # Only update label indices "inside the lattice"
    # does not count virtual EOS
    inside_label = get_seq_mask(target_lengths, max_seq_len+1, device).long().bool()
    for t in range(input_time_size-2, -1, -1):
        inside_time = (t < input_lengths).unsqueeze(-1).expand(-1, max_seq_len+1).to(device)
        is_updated = torch.logical_and(inside_time, inside_label) # (B, S+1)
        # First frame is blank
        backward[t, :, :, 1] = torch.where(
            is_updated,
            log_probs[t, :, blank_idx].unsqueeze(-1).expand(-1, max_seq_len+1) + backward[t+1, :, :, :].logsumexp(dim=-1),
            backward[t, :, :, 1]
        )
        # First frame is not blank, 4 sub-cases:
        # For accessing backward, idx is current, idx+1 is next label
        # For accessing log_probs and targets, idx is current
        # a_s != <mask>, a_{s+1} != <mask>, standard CTC transitions
        idx = not_mask_next_not_mask_idx
        update_mask = is_updated[:, idx]
        log_probs_targets_t = log_probs[t, :, :].gather(-1, targets.long())
        res = log_probs_targets_t[:, idx] + \
            torch.stack([
                backward[t+1, :, idx, 0].clone(),
                backward[t+1, :, idx+1, 1].clone(),
                torch.where(
                    next_label_diff[:, idx].bool(),
                    backward[t+1, :, idx+1, 0].clone(),
                    torch.tensor(log_zero).to(device),
                ),
            ], dim=-1).logsumexp(dim=-1)
        backward[t, :, idx, 0] = torch.where(update_mask, res, backward[t, :, idx, 0])

        # a_s != <mask>, a_{s+1} = <mask>
        idx = not_mask_next_mask_idx
        midx = not_mask_next_mask_midx
        update_mask = is_updated[:, idx]
        targets_shifted = targets[:, idx] - 1 # Because blank is 0
        targets_shifted[targets_shifted < 0] = 0
        res = log_probs_targets_t[:, idx] + logsubstractexp(
                torch.stack([
                    backward[t+1, :, idx+1, 0],
                    backward[t+1, :, idx+1, 1],
                    backward[t+1, :, idx, 0],
                ], dim=-1).logsumexp(dim=-1),
                backward_masked[t+1, :, midx, :].gather(-1, targets_shifted.unsqueeze(-1)).squeeze(-1),
                log_zero,
            )
        backward[t, :, idx, 0] = torch.where(update_mask, res, backward[t, :, idx, 0])

        # a_s = <mask>, a_{s+1} != <mask>
        idx = mask_next_not_mask_idx
        midx = mask_next_not_mask_midx
        update_mask = is_updated[:, idx]
        update_mask_midx = update_mask.unsqueeze(-1).expand(-1, -1, n_out-1) # (B, len(midx), F-1)
        res = log_probs[t, :, out_idx_no_blank].unsqueeze(1).expand(-1, len(midx), -1) + \
            torch.stack([
                backward_masked[t+1, :, midx, :].clone(),
                backward[t+1, :, idx+1, 1].clone().unsqueeze(-1).expand(-1, -1, n_out-1),
                torch.where(
                    all_w_diff_next,
                    backward[t+1, :, idx+1, 0].clone().unsqueeze(-1).expand(-1, -1, n_out-1),
                    torch.tensor(log_zero).to(device),
                )
            ], dim=-1).logsumexp(dim=-1)
        backward_masked[t, :, midx, :] = torch.where(update_mask_midx, res, backward_masked[t, :, midx, :])
        backward[t, :, idx, 0] = torch.where(update_mask, res.logsumexp(dim=-1), backward[t, :, idx, 0])

        # a_s = <mask>, a_{s-1} = <mask>
        idx = mask_next_mask_idx
        midx = mask_next_mask_midx
        midx_next = mask_next_mask_midx_next
        update_mask = is_updated[:, idx]
        update_mask_midx = update_mask.unsqueeze(-1).expand(-1, -1, n_out-1)
        res = log_probs[t, :, 1:].unsqueeze(1).expand(-1, len(midx), -1) + \
            logsubstractexp(
                torch.stack([
                    backward_masked[t+1, :, midx, :],
                    backward[t+1, :, idx+1, 1].clone().unsqueeze(-1).expand(-1, -1, n_out-1),
                    backward[t+1, :, idx+1, 0].unsqueeze(-1).expand(-1, -1, n_out-1),
                ], dim=-1).logsumexp(dim=-1),
                backward_masked[t+1, :, midx_next, :],
                log_zero,
            )
        backward_masked[t, :, midx, :] = torch.where(update_mask_midx, res, backward_masked[t, :, midx, :])
        backward[t, :, idx, 0] = torch.where(update_mask, res.logsumexp(dim=-1), backward[t, :, idx, 0])

    # Now join the scores in forward_masked and backward_masked
    # to obtain the masked probabilies
    # Method: Sum over frames in the lattice (horizontal sum)
    # sum_t Q(t, s, n, w)[R(t, s, n, w)/p(w|x_t) - R(t+1, s, n, w)]
    # sum of paths leaving s at exactly t+1
    # pay attention to time mask
    backward_masked_1 = torch.cat( # R(t+1, s, n, w)
        [backward_masked[1:, :, :, :], torch.full((1, batch_size, n_masked_pos, n_out-1), log_zero).to(device)],
        dim=0
    )
    log_probs_all_masked = log_probs.unsqueeze(2).expand(-1, -1, n_masked_pos, -1)[:, :, :, 1:]
    next_paths = backward_masked - log_probs_all_masked
    next_paths = torch.where(next_paths < log_zero, log_zero, next_paths)

    backward_paths = logsubstractexp(
        next_paths,
        backward_masked_1,
        log_zero,
        allow_log_neg=True, # Sloppy, but to deal with masked pos outside a sequence
    )
    input_time_mask = get_seq_mask(input_lengths, input_time_size, device).long().bool().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_masked_pos, n_out-1).transpose(0, 1)
    log_fwd_bwd = torch.where(input_time_mask, forward_masked + backward_paths, torch.tensor(log_zero).to(device)) # only sums up to T of each batch
    numerator = log_fwd_bwd.logsumexp(dim=0) # (B, M, F-1) p(masked sequence, mask = w | acoustic)
    denominator_batch = backward[0, :, 0, :].logsumexp(-1)
    denominator = denominator_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_out-1) # p(masked sequence | acoustic)
    
    # try changing denom calculation by forward. hopefully more robust?
    forward_last_frame = forward.logsumexp(-1).gather(0, (input_lengths-1).unsqueeze(0).unsqueeze(-1).expand(-1, -1, max_seq_len+1)).squeeze(0)
    # denominator_batch = forward_last_frame.gather(-1, target_lengths.to(device).unsqueeze(-1)).squeeze(-1)
    # denominator = denominator_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, n_masked_pos, n_out-1)

    ctc = torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_idx,
        reduction="none",
    )
    torch.set_printoptions(precision=2, threshold=10000, linewidth=50)
    print(mask)
    print(not_mask_next_not_mask_idx)
    print(torch.stack([denominator_batch, ctc], dim=-1))

    log_masked_probs = numerator - denominator # p(mask = w | masked sequence, acoustic)
    # log_masked_probs = numerator.log_softmax(-1)

    targets_shifted = torch.where(targets > 0, targets-1, torch.tensor(0).to(device))
    # print(mask)
    # print(target_lengths)
    # print(numerator.gather(-1, targets_shifted[:, masked_pos].unsqueeze(-1)).squeeze(-1))
    # print(denominator.gather(-1, targets_shifted[:, masked_pos].unsqueeze(-1)).squeeze(-1))
    # print(log_masked_probs.gather(-1, targets_shifted[:, masked_pos].unsqueeze(-1)).squeeze(-1))

    return log_masked_probs, forward, backward, forward_masked, backward_masked
