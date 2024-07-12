"""
This module implements some loss related to
the CTC prefix score p_CTC(a_1^N,...)
"""
import torch
import torch.nn as nn
from i6_experiments.users.yang.torch.utils.masking import get_seq_mask, get_seq_mask_v2
def print_gpu_memory_usage(pos='0'):
    print("*********************************************************************************************************")
    unused = torch.cuda.memory_reserved(0) / 1e9 - torch.cuda.memory_allocated(0) / 1e9
    print("Pos: {} Total GPU Memory: {:.2f} GB".format(pos, torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("Pos: {} Allocated GPU Memory: {:.2f} GB".format(pos, torch.cuda.memory_allocated(0) / 1e9))
    print("Pos: {} Cached GPU Memory: {:.2f} GB".format(pos, torch.cuda.memory_reserved(0) / 1e9))
    print("Pos: {} Reserved but Unused GPU Memory: {:.2f} GB".format(pos, unused))
# convention for all dim size 2: 0 is n, 1 is b
def print_tensor_memory_with_gradient(tensor, name=''):
    # Calculate basic memory usage as before
    memory_bytes = tensor.nelement() * tensor.element_size()

    # Check if this tensor requires gradients
    if tensor.requires_grad:
        # Double the memory usage to account for the gradient
        memory_bytes *= 2

    memory_kb = memory_bytes / 1024
    memory_mb = memory_kb / 1024
    print("################################################")
    print(f"Tensor {name} Memory Usage: {memory_mb:.2f} MB")
    print("################################################")
def log_ctc_pref_beam_scores(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    blank_idx = 0,
    eos_idx = 0,
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
    print(f"*************input shape: ({input_time_size}, {batch_size})")
    print(f"*************target shape: ({max_seq_len}, {batch_size})")
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
    prev_gamma_1 = log_gamma[0, :, 1, :]
    prev_gamma_0 = log_gamma[0, :, 0, :]
    torch.cuda.empty_cache()
    for t in range(1, input_time_size):
        # test not using clone. has the in-place replacement error
        log_probs_t_k_ground_truth = log_probs[t].gather(-1, targets.long()) # (B, S) log prob of all symbol at time frame t
        log_new_label_prob = torch.logaddexp( # (B, S+1, F) S here is prefixes from empty to a_1^S
            prev_gamma_1.unsqueeze(-1).expand(-1, -1, n_out),
            prev_gamma_0.unsqueeze(-1).expand(-1, -1, n_out).where(prev_label_diff,torch.tensor(log_zero, device=device))
            # log_gamma[t-1, :, 1, :].unsqueeze(-1).expand(-1, -1, n_out).clone(),
            # log_gamma[t-1, :, 0, :].unsqueeze(-1).expand(-1, -1, n_out).clone().where(prev_label_diff, torch.tensor(log_zero, device=device))
        )
        # (B, S) for both below

        #log_gamma[t, :, 0, 1:] = log_probs_t_k_ground_truth + torch.logaddexp(log_new_label_prob[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1), log_gamma[t-1, :, 0, 1:].clone())
        #log_gamma[t, :, 1, 1:] = log_probs[t, :, blank_idx].unsqueeze(-1).expand((-1, max_seq_len)) + torch.logsumexp(log_gamma[t-1, :, :, 1:].clone(), dim=1)
        new_gamma_0_label = log_probs_t_k_ground_truth + torch.logaddexp(log_new_label_prob[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1), prev_gamma_0[:,1:])
        new_gamma_1_label = log_probs[t, :, blank_idx].unsqueeze(-1).expand((-1, max_seq_len)) + torch.logaddexp(prev_gamma_0[:,1:], prev_gamma_1[:, 1:]) # notice! can be optimized
        new_gamma_0 = torch.cat((log_gamma[t,:,0, 0:1], new_gamma_0_label), dim=-1)
        new_gamma_1 = torch.cat((log_gamma[t,:,1, 0:1], new_gamma_1_label), dim=-1)
        # (B, F) to (B, S+1, F), probs of all possible next labels at frame t, same no matter which prefix
        log_probs_t_k_all_beams = log_probs[t, :, :].unsqueeze(-1).expand((-1, -1, max_seq_len+1)).transpose(1, 2)
        input_time_mask = (t < input_lengths).to(device).unsqueeze(-1).expand((-1, (max_seq_len+1)*n_out)).view(-1, max_seq_len+1, n_out)
        #print(f'current time step: {t}')
        torch.cuda.empty_cache()
        print_gpu_memory_usage(f"in time loop, time: {t}")
        log_pref_scores_beams = log_pref_scores_beams.logaddexp(
            (log_probs_t_k_all_beams + log_new_label_prob).where(input_time_mask, torch.tensor(log_zero, device=device))
        ) # the score of a beam should not change if t > input length
        prev_gamma_0 = new_gamma_0
        prev_gamma_1 = new_gamma_1
        torch.cuda.empty_cache() ############# attention!!!!!!!!!!!!!!!!!!!! check if this influence the speed a lot
    # Reuse the blank idx as EOS, i.e. full forward prob of current hypothesis
    #log_pref_scores_beams[:, :, blank_idx] = torch.logsumexp(log_gamma[-1, :, :, :], dim=1)
    log_pref_scores_beams[:,:, eos_idx] = torch.logsumexp(log_gamma[-1, :, :, :], dim=1) # bug?
    return log_pref_scores_beams, log_gamma


def log_ctc_pref_beam_scores_v2(
        log_probs,  # (T, B, F)
        targets,  # (B, S)
        targets_w_bos, # (B S+1)
        targets_w_eos, # (B, S+1)
        input_lengths,  # (B,)
        blank_idx=0,
        eos_idx=0,
        bos_idx=0,
        log_zero=-1e25,  # maybe better than float min for preventing overflowing
        top_k_list=None, # (B,S+1,K)
        freeze_gamma=False,
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
    if top_k_list is not None:
        n_out = top_k_list.shape[2]
    max_seq_len = targets.shape[1]
    # convention for all dim size 2: 0 is n, 1 is b
    # max_seq_len + 1 for the empty prefix
    log_gamma = torch.full((input_time_size, batch_size, 2, max_seq_len + 1), log_zero, dtype=torch.float32,
                           device=device)  # (T, B, 2, S+1)
    log_gamma[:, :, 1, 0] = torch.cumsum(log_probs[:, :, blank_idx], dim=0).nan_to_num(neginf=log_zero)
    log_gamma[0, :, 0, 1] = log_probs[0].gather(-1, targets[:, :1].long()).view(-1)
    # (B, S, F), for first beam (empty prefix), all is True
    prev_label_all_beams = targets.unsqueeze(-1).expand(-1, -1,
                                                        n_out)  # (B, S, F), S here from 1 to max_seq_len, dim 2 is prev_label of all beams
    # beams = torch.arange(n_out, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, max_seq_len,
    #                                                                             -1)  # (B, S, F), S here from 0 to max_seq_len-1, dim 2 is beam
    # prev_label_diff = beams != prev_label_all_beams  # (B, S, F)
    # prev_label_diff = torch.cat(
    #     [torch.full((batch_size, 1, n_out), torch.tensor(True), device=device), prev_label_diff], dim=1)
    top_k_prev_label_diff = top_k_list[:, :-1, :] != prev_label_all_beams  # (B, S, F)
    top_k_prev_label_diff = torch.cat(
        [torch.full((batch_size, 1, n_out), torch.tensor(True), device=device), top_k_prev_label_diff], dim=1)


    target_label_shift_diff = torch.ne(targets_w_eos, targets_w_bos) # shape  (B, S+1)
    # log_pref_scores_beams (B, S+1, F) S+1 here is prefixes from empty to a_1^S
    if top_k_list is not None:
        selected_log_probs = log_probs.unsqueeze(2).expand(-1,-1, max_seq_len+1, -1).gather(dim=-1, index=top_k_list.unsqueeze(0).expand(input_time_size,-1,-1,-1)) # shape (T, B,S+1,K)
        log_pref_scores_beams = torch.cat(
            [selected_log_probs[0, :, :1, :], torch.full((batch_size, max_seq_len, n_out), log_zero, device=device)],
            dim=1
        ) # shape (B, S+1, K)
    else:
        log_pref_scores_beams = torch.cat(
            [log_probs[0].unsqueeze(1), torch.full((batch_size, max_seq_len, n_out), log_zero, device=device)],
            dim=1
        )
    prev_gamma_1 = log_gamma[0, :, 1, :] # (B, S+1)
    prev_gamma_0 = log_gamma[0, :, 0, :]
    #torch.cuda.empty_cache()
    for t in range(1, input_time_size):
        if freeze_gamma:
            prev_gamma_0 = prev_gamma_0.detach()
            prev_gamma_1 = prev_gamma_1.detach()
        # test not using clone. has the in-place replacement error
        log_probs_t_k_ground_truth = log_probs[t].gather(-1, targets.long())  # (B, S) log prob of all symbol at time frame t, this is used to compute gamma, so irrelevant to top-k-list

        # log_new_label_prob is not necessary, at least split into prev_gamma_1, and prev_gamma_0
        log_label_sum = torch.logaddexp(prev_gamma_1, prev_gamma_0) # (B, S+1)
        log_label_sum_target_masked = torch.where(target_label_shift_diff, log_label_sum, prev_gamma_1)
        log_label_sum_top_k_masked = torch.where(top_k_prev_label_diff, log_label_sum.unsqueeze(-1), prev_gamma_1.unsqueeze(-1)) # (B, S+1, K)

        new_gamma_0_label = log_probs_t_k_ground_truth + torch.logaddexp(log_label_sum_target_masked[:, :-1], prev_gamma_0[:, 1:]) # shape (B,S) the computation of gamma should not be influenced by top-k
        # new_gamma_1_label = log_probs[t, :, blank_idx].unsqueeze(-1).expand((-1, max_seq_len)) + torch.logaddexp(
        #     prev_gamma_0[:, 1:], prev_gamma_1[:, 1:])  # notice! can be optimized

        new_gamma_1_label = log_probs[t, :, blank_idx].unsqueeze(-1) + torch.logaddexp(
            prev_gamma_0[:, 1:], prev_gamma_1[:, 1:]) # shape (B, S)

        new_gamma_0 = torch.cat((log_gamma[t, :, 0, 0:1], new_gamma_0_label), dim=-1)
        new_gamma_1 = torch.cat((log_gamma[t, :, 1, 0:1], new_gamma_1_label), dim=-1)
        # (B, F) to (B, S+1, F), probs of all possible next labels at frame t, same no matter which prefix


        # check the difference of memory usage when selecting the log_probs here or outside the time loop
        #log_probs_extend = log_probs[t, :, :].unsqueeze(1).expand((-1,max_seq_len+1,-1))
        #log_probs_selected = log_probs_extend.gather(-1, top_k_list)


        cur_selected_log_probs = selected_log_probs[t] # shape (B,S+1,K)

        # log_probs_t_k_all_beams = log_probs[t, :, :].unsqueeze(-1).expand((-1, -1, max_seq_len + 1)).transpose(1, 2) #
        # input_time_mask = (t < input_lengths).to(device).unsqueeze(-1).expand((-1, (max_seq_len + 1) * n_out)).view(-1,
        #                                                                                                             max_seq_len + 1,
        #                                                                                                             n_out)
        input_time_mask = (t < input_lengths.to(device)) # input_time_mask in shape (B,)
        # print(f'current time step: {t}')
        #torch.cuda.empty_cache()
        #print_gpu_memory_usage(f"in time loop, time: {t}")



        # log_pref_scores_beams = log_pref_scores_beams.logaddexp(
        #     (log_probs_t_k_all_beams + log_new_label_prob).where(input_time_mask, torch.tensor(log_zero, device=device))
        # )  # the score of a beam should not change if t > input length
        cur_pref_score = torch.where(input_time_mask.unsqueeze(-1).unsqueeze(-1), (cur_selected_log_probs + log_label_sum_top_k_masked), log_zero)
        #log_pref_scores_beams = log_pref_scores_beams.logaddexp(cur_pref_score)
        log_pref_scores_beams = torch.logaddexp(log_pref_scores_beams, cur_pref_score)
        prev_gamma_0 = torch.where(input_time_mask.unsqueeze(-1), new_gamma_0, prev_gamma_0)
        prev_gamma_1 = torch.where(input_time_mask.unsqueeze(-1), new_gamma_1, prev_gamma_1)
        torch.cuda.empty_cache()  ############# attention!!!!!!!!!!!!!!!!!!!! check if this influence the speed a lot
    # Reuse the blank idx as EOS, i.e. full forward prob of current hypothesis
    ctc_eos_score = torch.logaddexp(prev_gamma_1, prev_gamma_0)
    if top_k_list is None:
        log_pref_scores_beams[:,:, eos_idx] = ctc_eos_score
    else:
        # locate eos in the top_k_list
        eos_mask = top_k_list == eos_idx # True if index is eos
        log_pref_scores_beams = torch.where(eos_mask, ctc_eos_score.unsqueeze(-1), log_pref_scores_beams)

    return log_pref_scores_beams, ctc_eos_score




def ctc_prefix_posterior(ctc_log_probs, targets, targets_w_bos, targets_w_eos, input_lengths, target_lengths, blank_index, eos_idx=0, out_no_blank=True, top_k_list=None, freeze_gamma=False):
    '''
    ctc_log_probs: ctc outputs with shape (B,T,V), normalized log probs
    targets: target seq without eos
    blank_index:
    input_lengths: length of the input
    all inputs should be pure torch tensors
    '''
    assert top_k_list is not None
    log_probs = ctc_log_probs.transpose(0,1) # to shape (T,B,V)
    #log_probs = nn.functional.log_softmax(logits.transpose(0,1), dim=-1) # to shape (T,B,V)
    # confirmed that the log prob of eos (index 0) is very low, around -50 or so
    torch_input_lengths = input_lengths
    torch_targets = targets.long()
    batch_size, max_seq_len = torch_targets.shape
    prefix_score,  ctc_eos_score = log_ctc_pref_beam_scores_v2(log_probs, torch_targets, targets_w_bos, targets_w_eos,torch_input_lengths, blank_idx=blank_index, eos_idx=eos_idx, top_k_list=top_k_list, freeze_gamma=freeze_gamma)
    indices = torch_targets.unsqueeze(-1)
    if top_k_list is None:
        prefix_score_norm = prefix_score[:, :-1, :].gather(-1, indices).squeeze(-1) # norm for given context a_1^N
        prefix_score_norm = torch.cat([torch.zeros((batch_size, 1),device=prefix_score_norm.device), prefix_score_norm], dim=-1) # first label, norm is 1
        prefix_posterior = prefix_score - prefix_score_norm.unsqueeze(-1)  # shape (B, S+1, V+1)
    else:
        prefix_posterior = nn.functional.log_softmax(prefix_score, -1)
    torch_target_lengths = target_lengths.long()
    torch_target_lengths = torch_target_lengths.to(prefix_posterior.device)
    #final_ctc_prob = prefix_score[:,:,eos_idx].gather(-1,torch_target_lengths.unsqueeze(-1)).squeeze(-1)
    final_ctc_prob = ctc_eos_score.gather(-1, torch_target_lengths.unsqueeze(-1)).squeeze(-1)
    assert not out_no_blank
    if out_no_blank:
        # code here problematic
        vocab_size = ctc_log_probs.shape[-1]
        if blank_index == vocab_size-1:
            prefix_posterior = prefix_posterior[:,:,:-1]
        else:
            prefix_posterior = torch.cat([prefix_posterior[:,:,:blank_index], prefix_posterior[:,:,blank_index+1]], dim=-1)
    return prefix_posterior, final_ctc_prob




def log_ctc_pref_beam_scores_v3(
        log_probs,  # (T, B, F)
        targets,  # (B, S)
        targets_w_bos, # (B S+1)
        targets_w_eos, # (B, S+1)
        input_lengths,  # (B,)
        target_lengths, # (B,)
        blank_idx=0,
        eos_idx=0,
        bos_idx=0,
        log_zero=-1e25,  # maybe better than float min for preventing overflowing
        top_k_list=None, # (B,S+1,K)
        no_beam_prefix=False,
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
    assert top_k_list is not None or no_beam_prefix, "general case without top-k not implemented yet"
    device = log_probs.device
    input_time_size, batch_size, n_out = log_probs.shape
    if top_k_list is not None:
        n_out = top_k_list.shape[2]
    max_seq_len = targets.shape[1]
    # convention for all dim size 2: 0 is n, 1 is b
    # max_seq_len + 1 for the empty prefix
    log_gamma = torch.full((input_time_size, batch_size, 2, max_seq_len + 1), log_zero, dtype=torch.float32,
                           device=device)  # (T, B, 2, S+1)
    log_gamma[:, :, 1, 0] = torch.cumsum(log_probs[:, :, blank_idx], dim=0).nan_to_num(neginf=log_zero)
    log_gamma[0, :, 0, 1] = log_probs[0].gather(-1, targets[:, :1].long()).view(-1)
    # (B, S, F), for first beam (empty prefix), all is True
    # the prefix probs for the target labels can be obtained almost for free, consider always compute it

    if top_k_list is not None:
        prev_label_all_beams = targets.unsqueeze(-1).expand(-1, -1, n_out)  # (B, S, F), S here from 1 to max_seq_len, dim 2 is prev_label of all beams
        top_k_prev_label_diff = top_k_list[:, :-1, :] != prev_label_all_beams  # (B, S, F)
        top_k_prev_label_diff = torch.cat(
            [torch.full((batch_size, 1, n_out), torch.tensor(True), device=device), top_k_prev_label_diff], dim=1)  # (B, S+1 F)


    target_label_shift_diff = torch.ne(targets_w_eos, targets_w_bos) # shape  (B, S+1)
    target_pos_mask = torch.range(start=0,end=max_seq_len, dtype=target_lengths.dtype, device=device).unsqueeze(0) == target_lengths.to(device).unsqueeze(-1) # (B, S+1) # locate the end of target seq position S+1
    # log_pref_scores_beams (B, S+1, F) S+1 here is prefixes from empty to a_1^S
    if top_k_list is not None:
        selected_log_probs = log_probs.unsqueeze(2).expand(-1,-1, max_seq_len+1, -1).gather(dim=-1, index=top_k_list.unsqueeze(0).expand(input_time_size,-1,-1,-1)) # shape (T, B,S+1,K)
        log_pref_scores_beams = torch.cat(
            [selected_log_probs[0, :, :1, :], torch.full((batch_size, max_seq_len, n_out), log_zero, device=device)],
            dim=1
        ) # shape (B, S+1, K)

    elif no_beam_prefix:
        log_pref_scores_beams = torch.cat(
            [log_probs[0].unsqueeze(1), torch.full((batch_size, max_seq_len, n_out), log_zero, device=device)],
            dim=1
        )
    else:
        log_pref_scores_beams = None
    prev_gamma_1 = log_gamma[0, :, 1, :] # (B, S+1)
    prev_gamma_0 = log_gamma[0, :, 0, :]

    log_target_seq_pref_scores = torch.cat(
        [log_probs[0].gather(-1, targets_w_eos[:,:1]), torch.full((batch_size, max_seq_len-1), log_zero, device=device)], dim=-1) # in shape (B,S)
    #torch.cuda.empty_cache()
    for t in range(1, input_time_size):
        # test not using clone. has the in-place replacement error
        log_probs_t_k_ground_truth = log_probs[t].gather(-1, targets.long())  # (B, S) log prob of all symbol at time frame t, this is used to compute gamma, so irrelevant to top-k-list

        # log_new_label_prob is not necessary, at least split into prev_gamma_1, and prev_gamma_0
        log_label_sum = torch.logaddexp(prev_gamma_1, prev_gamma_0) # (B, S+1)
        log_label_sum_target_masked = torch.where(target_label_shift_diff, log_label_sum, prev_gamma_1)  # (B,S+1)
        if top_k_list is not None:
            log_label_sum_top_k_masked = torch.where(top_k_prev_label_diff, log_label_sum.unsqueeze(-1), prev_gamma_1.unsqueeze(-1)) # (B, S+1, K)

        new_gamma_0_label = log_probs_t_k_ground_truth + torch.logaddexp(log_label_sum_target_masked[:, :-1], prev_gamma_0[:, 1:]) # shape (B,S) the computation of gamma should not be influenced by top-k
        # new_gamma_1_label = log_probs[t, :, blank_idx].unsqueeze(-1).expand((-1, max_seq_len)) + torch.logaddexp(
        #     prev_gamma_0[:, 1:], prev_gamma_1[:, 1:])  # notice! can be optimized

        new_gamma_1_label = log_probs[t, :, blank_idx].unsqueeze(-1) + torch.logaddexp(
            prev_gamma_0[:, 1:], prev_gamma_1[:, 1:]) # shape (B, S)

        new_gamma_0 = torch.cat((log_gamma[t, :, 0, 0:1], new_gamma_0_label), dim=-1)
        new_gamma_1 = torch.cat((log_gamma[t, :, 1, 0:1], new_gamma_1_label), dim=-1)
        # (B, F) to (B, S+1, F), probs of all possible next labels at frame t, same no matter which prefix

        # after computing new_gamma, already able to compute the log_target_seq_pref_scores
        input_time_mask = (t < input_lengths.to(device)) # (B,)
        cur_target_pref_score = torch.where(input_time_mask.unsqueeze(-1), (log_probs_t_k_ground_truth + log_label_sum_target_masked[:,:-1]), log_zero) # (B,S)
        log_target_seq_pref_scores = torch.logaddexp(log_target_seq_pref_scores, cur_target_pref_score)  #
        if top_k_list is not None:

            cur_selected_log_probs = selected_log_probs[t] # shape (B,S+1,K)
            cur_pref_score = torch.where(input_time_mask.unsqueeze(-1).unsqueeze(-1), (cur_selected_log_probs + log_label_sum_top_k_masked), log_zero)
        #log_pref_scores_beams = log_pref_scores_beams.logaddexp(cur_pref_score)
            log_pref_scores_beams = torch.logaddexp(log_pref_scores_beams, cur_pref_score)
        prev_gamma_0 = torch.where(input_time_mask.unsqueeze(-1), new_gamma_0, prev_gamma_0)
        prev_gamma_1 = torch.where(input_time_mask.unsqueeze(-1), new_gamma_1, prev_gamma_1)
        torch.cuda.empty_cache()  ############# attention!!!!!!!!!!!!!!!!!!!! check if this influence the speed a lot
    # Reuse the blank idx as EOS, i.e. full forward prob of current hypothesis
    ctc_eos_score = torch.logaddexp(prev_gamma_1, prev_gamma_0)  # (B, S+1)
    # assign the ctc seq prob to target pref scores
    log_target_seq_pref_scores = torch.cat([log_target_seq_pref_scores, torch.full((batch_size,1), log_zero, device=device)], dim=-1) # (B, S+1)
    log_target_seq_pref_scores = torch.where(target_pos_mask, ctc_eos_score, log_target_seq_pref_scores)

    if top_k_list is None:
        if not no_beam_prefix:
            log_pref_scores_beams[:,:, eos_idx] = ctc_eos_score
    else:
        # locate eos in the top_k_list
        eos_mask = top_k_list == eos_idx # True if index is eos
        log_pref_scores_beams = torch.where(eos_mask, ctc_eos_score.unsqueeze(-1), log_pref_scores_beams)

    return log_pref_scores_beams, log_target_seq_pref_scores




def ctc_prefix_posterior_v3(ctc_log_probs, targets, targets_w_bos, targets_w_eos, input_lengths, target_lengths, blank_index, eos_idx=0, out_no_blank=True, top_k_list=None, no_beam_prefix=False):
    '''
    ctc_log_probs: ctc outputs with shape (B,T,V), normalized log probs
    targets: target seq without eos
    blank_index:
    input_lengths: length of the input
    all inputs should be pure torch tensors
    '''
    assert top_k_list is not None or no_beam_prefix
    log_probs = ctc_log_probs.transpose(0,1) # to shape (T,B,V)
    #log_probs = nn.functional.log_softmax(logits.transpose(0,1), dim=-1) # to shape (T,B,V)
    # confirmed that the log prob of eos (index 0) is very low, around -50 or so
    torch_input_lengths = input_lengths
    device = ctc_log_probs.device
    torch_targets = targets.long()
    batch_size, max_seq_len = torch_targets.shape
    torch_target_lengths = target_lengths.long()
    torch_target_lengths = torch_target_lengths.to(device)
    prefix_score, target_prefix_score = log_ctc_pref_beam_scores_v3(log_probs, torch_targets, targets_w_bos, targets_w_eos,torch_input_lengths, torch_target_lengths, blank_idx=blank_index, eos_idx=eos_idx, top_k_list=top_k_list, no_beam_prefix=no_beam_prefix)
    indices = torch_targets.unsqueeze(-1)
    if top_k_list is not None:
        prefix_posterior = nn.functional.log_softmax(prefix_score, -1)
    elif not no_beam_prefix:
        prefix_score_norm = prefix_score[:, :-1, :].gather(-1, indices).squeeze(-1) # norm for given context a_1^N
        prefix_score_norm = torch.cat([torch.zeros((batch_size, 1),device=prefix_score_norm.device), prefix_score_norm], dim=-1) # first label, norm is 1
        prefix_posterior = prefix_score - prefix_score_norm.unsqueeze(-1)  # shape (B, S+1, V+1)
    else:
        prefix_posterior = None

    target_prefix_score_norm = torch.cat([torch.zeros([batch_size,1], device=device), target_prefix_score[:,:-1] ], dim=-1) # shifted target prefix scores
    target_prefix_posterior = target_prefix_score - target_prefix_score_norm
    final_ctc_prob = target_prefix_score.gather(-1, torch_target_lengths.unsqueeze(-1)).squeeze(-1)
    assert not out_no_blank
    if out_no_blank:
        # code here problematic
        vocab_size = ctc_log_probs.shape[-1]
        if blank_index == vocab_size-1:
            prefix_posterior = prefix_posterior[:,:,:-1]
        else:
            prefix_posterior = torch.cat([prefix_posterior[:,:,:blank_index], prefix_posterior[:,:,blank_index+1]], dim=-1)
    return prefix_posterior, final_ctc_prob, target_prefix_posterior





def kldiv_ctc_lm_loss(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_lm_score, # (B, S, F)
    blank_idx = 0,
    log_zero = -1e25, # maybe better than float min for preventing overflowing
    eos_idx = None,
    target_mask = None,
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
    :param log_lm_score: log LM score of all possible words in vocab given ground truth context (B, S+1, F)
    EOS of this should be blank of the CTC
    :param blank_idx: Blank index in F dim of log_probs
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
    :param target_mask: Extra label-wise masking apply to the loss (B, S+1 F)
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
    loss = kl_div*seq_mask
    if target_mask is not None:
        if len(target_mask.shape) == 2: # (B, S+1), cast to (B, S+1, F)
            target_mask = target_mask.unsqueeze(-1).expand(-1, -1, n_out)
        loss = loss*target_mask
    loss = loss.sum()
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


def kldiv_ctc_lm_sample_batch_loss(
    log_probs, # (T, B, F)
    targets, # (B, S)
    input_lengths, # (B,)
    target_lengths, # (B,)
    log_lm_score, # (B, S, F)
    blank_idx = 0,
    log_zero = -1e25, # maybe better than float min for preventing overflowing
    eos_idx = None,
    ground_truth_weight = 0.5,
):
    '''
    Same as KLDiv CTC LM above but loss computed from whole batch. Example:
    batch is [audio1, target1], [audio2, target2], ..., then compute loss for
    [audio 1 to n, target1], [audio 1 to n, target2]. Apply some weighting
    here, e.g. 1/2 for ground truth and 1/(2*(B-1)) for the rest

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
    :param log_lm_score: log LM score of all possible words in vocab given ground truth context (B, S+1, F)
    EOS of this should be blank of the CTC
    :param blank_idx: Blank index in F dim of log_probs
    :param log_zero: Value of log zero. Default to -1e15 to prevent overflow comparing to float32 min
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
    log_pref_scores_beams, _ = log_ctc_pref_beam_scores(
        log_probs,
        targets,
        input_lengths,
        blank_idx,
        log_zero,
    )
    # renormalize to have p_ctc(v|hypothesis) in output dim
    log_p_ctc = log_pref_scores_beams.log_softmax(dim=-1)
    log_lm_score = log_lm_score.repeat_interleave(batch_size, dim=0)
    #print(torch.exp(log_p_ctc.gather(-1, targets.unsqueeze(-1).long()).view(batch_size*batch_size, max_seq_len)))
    kl_div = torch.nn.functional.kl_div(
        input=log_lm_score,
        target=log_p_ctc,
        log_target=True,
        reduction="none",
    )
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
    