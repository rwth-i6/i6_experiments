import torch


def get_common_context(context_ids, vocab_size, context_length=3):
    B, k = context_ids.shape
    tail_base = vocab_size**(context_length-1)
    tails= context_ids % tail_base
    sorted_tails, perm = tails.sort(dim=-1)
    new_group = torch.ones([B,k],dtype=torch.bool,device=context_ids.device)
    new_group[:,1:] = sorted_tails[:,1:] != sorted_tails[:,:-1]
    tail_group_idx = new_group.cumsum(dim=-1) - 1 # e.g. if for one batch k hyps are [a,a,b,b,c], this will be [0,0,1,1,2]
    num_tail_groups = new_group.sum(dim=-1)
    return tail_group_idx, num_tail_groups, sorted_tails, perm


def context_recombine_allow_loop(context_ids, vocab_size, context_length=3):
    # the goal is to recombine the same contexts, for the expanded context, find common h[1:], and later when expanding next phoneme w, these context can directly be recombined
    # for loop, since there is no emitted label, take h[:-1] as "fake" context before expantion, but then store the corresponding h[-1], and then in the expantion, only h[-1] has score, and others should be masked out

    B, k = context_ids.shape
    head = context_ids//vocab_size # [B,k]
    tail_base = vocab_size**(context_length-1)
    tail = context_ids %  tail_base
    context = torch.cat([tail, head], dim=-1) # [B, 2 * k]
    sorted_tails, perm = context.sort(dim=-1)
    new_group = torch.ones([B, 2*k], dtype=torch.bool, device=context_ids.device)
    new_group[:, 1:] = sorted_tails[:,1:] != sorted_tails[:,:-1]
    tail_group_idx = new_group.cumsum(dim=-1) - 1
    num_tail_groups = new_group.sum(dim=-1)


    # to see the sorted_tails is from the tail branch or head branch
    tail_branch_mask = perm < k

    return tail_group_idx, num_tail_groups, sorted_tails, perm, tail_branch_mask

    
def _map_context_id(context_ids, vocab_size, context_length):
    # map an context integer id to a flatterned context
    current_context = None
    context_tmp = context_ids.clone().detach()
    for i in range(context_length):
        if current_context is None:
            current_context = (context_tmp % vocab_size).unsqueeze(-1)
            context_tmp = context_tmp // vocab_size
        else:
            new_context = context_tmp % vocab_size
            current_context = torch.cat([new_context.unsqueeze(-1), current_context], dim=-1)
            context_tmp = context_tmp// vocab_size

    return current_context





def one_to_one_full_sum(
    log_probs,
    seq_len,
    lm_network=None,
    lm_table=None,
    lm_vocab_size=41,
    lm_context_length=3,
    k=200,
    use_lm_silence_score=False,
    lm_scale=0.6,
    am_scale=1.0,
):
    # compute the full sum loss over all possible phoneme sequences, but for one input position there is exactly one output, with lm and am scores combined i.e. there is no alignment issue
    # this is to test if the full sum criterion can work, for an artifical decypherment task
    # log_probs is in shape [B,T,V], it is precomputed, V is the vocab size,
    # seq_len is the length of iput, shape [B,]
    # we can provide either an LM or precomputed lm_table
    # k is the beam size that at each step how many hypothesis we keep.
    # the lm table should be in shape [lm_vocab_size**lm_context_length, lm_vocab_size]
    # am_scale and lm_scale weight the AM and LM log scores before summation.
    # we have am vocab size and lm vocab size because the lm has an additional token eos/bos






    B, max_len, am_vocab_size = log_probs.shape

    if lm_table is not None:
        lm_table = lm_table.detach().clone().to(device=log_probs.device, dtype=log_probs.dtype)
        if not use_lm_silence_score:
            lm_table[:,0] = -10000
    else:
        raise NotImplementedError
    # assume the eos/bos index is lm_vocab_size-1
    lm_vocab_size_int = int(lm_vocab_size)
    lm_bos_eos_idx = lm_vocab_size_int-1
    # assume lm starting entry is [bos, bos, bos]
    lm_start_entry = torch.zeros((), dtype=torch.long, device=log_probs.device)
    lm_vocab_size_tensor = torch.tensor(lm_vocab_size_int, dtype=torch.long, device=log_probs.device)
    buffers = [lm_vocab_size_tensor**j for j in range(lm_context_length)]
    for i in range(lm_context_length):
        buffer = buffers[i]
        lm_start_entry += buffer*lm_bos_eos_idx

    assert k >= am_vocab_size, "too small n-best list"
    assert lm_vocab_size_int >= am_vocab_size, "LM vocab must cover all AM labels"

    if lm_table is not None:
        init_lm_score = lm_table[lm_start_entry] # shape[V]
    # first step score
    scores_valid = am_scale * log_probs[:,0] + lm_scale * init_lm_score[None, :am_vocab_size]
    scores_padding = torch.ones((B,k-am_vocab_size), dtype=log_probs.dtype, device=log_probs.device)* -50000
    scores = torch.cat([scores_valid, scores_padding], dim=-1) # to [B,k]
    context_ids = torch.ones([B,k],dtype=torch.long, device=log_probs.device) * lm_bos_eos_idx
    
    phon_idx = torch.arange(am_vocab_size,device=context_ids.device)

    valid_context_ids = lm_bos_eos_idx * sum(buffers[1:]) + phon_idx # e.g. for context 3, the valid context for the first step is [bos, bos, 0], [bos, bos,1],... converted to integers
    valid_context_ids = valid_context_ids[None,:].expand(B,-1)
    context_ids[:,:am_vocab_size] = valid_context_ids # here the am emission means lm emission, so this type of initialization is fine, when silence is not considered as a context, this needs to be handled differently


    # main part
    # for loop over the time dimension
    seq_len = seq_len.to(device=log_probs.device)
    seq_len_mask = torch.arange(max_len, device=log_probs.device)[None,:] < seq_len[:, None]
    for t in range(1,max_len):
        tail_group_idx, num_tail_groups, sorted_tails, perm = get_common_context(context_ids, lm_vocab_size_tensor, context_length=lm_context_length) # the am is always context 1
        sorted_context_ids = context_ids.gather(1, perm)
        if lm_table is not None:
            lm_scores = lm_table[sorted_context_ids,:am_vocab_size] # we assume that the last index is bos/eos
        else:
            raise NotImplementedError
        expand_group_idx = tail_group_idx[:,:,None].expand(-1,-1, am_vocab_size)
        perm_old_scores = scores.gather(1,perm)
        valid_tail = torch.arange(k, device=scores.device)[None,:] < num_tail_groups[:, None]

        unique_tails = torch.zeros((B,k), dtype=context_ids.dtype,device=context_ids.device)
        unique_tails.scatter_(1, tail_group_idx, sorted_tails)

        perm_combined_score = (
            perm_old_scores[:, :, None]
            + lm_scale * lm_scores
            + am_scale * log_probs[:, t, :][:, None, :]
        )


        group_max = torch.full((B,k, am_vocab_size), -50000,dtype=scores.dtype, device=scores.device)
        group_max.scatter_reduce_(dim=1, index=expand_group_idx,src=perm_combined_score,reduce="amax", include_self=True,)
        group_max_detached = group_max.detach()
        exp_scores = torch.exp(perm_combined_score - group_max_detached.gather(1,expand_group_idx))
        group_sum = torch.zeros_like(group_max)
        group_sum.scatter_add_(1, expand_group_idx, exp_scores)
        new_scores = torch.log(group_sum.clamp_min(torch.finfo(group_sum.dtype).tiny)) + group_max_detached #[B,k,V]
        # mask out padding groups
        new_scores = new_scores.masked_fill(~valid_tail[:,:,None],-50000)
        new_scores = new_scores.view(B,-1) # [B,k*V], be ready for topk

        new_context_ids = unique_tails[:,:, None] * lm_vocab_size_int + phon_idx[None, None, :]
        new_context_ids = new_context_ids.view(B,-1)
        new_scores_k, new_indices_k = torch.topk(new_scores, k=k, dim=-1)
        # only update the still valid postions
        valid_batch = seq_len_mask[:, t][:, None]
        scores = torch.where(valid_batch, new_scores_k, scores) #[B,k]
        context_ids_selected = new_context_ids.gather(1, new_indices_k)
        context_ids = torch.where(valid_batch, context_ids_selected, context_ids)

    final_scores = torch.logsumexp(scores,dim=-1) # for now we don't add lm EOS score

    return final_scores.sum()


def no_concecutive_full_sum(
    log_probs,
    seq_len,
    lm_network=None,
    lm_table=None,
    lm_vocab_size=41,
    lm_context_length=3,
    k=200,
    lm_scale=0.6,
    am_scale=1.0,
    loop_penalty=1.0
):
    # compute the full sum loss over all possible phoneme sequences, but for silence and concecutive duplicated phoneme, they are reagarded not emitting a new phoneme, but a loop-like transition, in this case, the context does not update, and the LM score is not added
    # To make the representation deterministic, on the output phoneme sequence side, there is no duplicated phonemes, like [A, A] on the phoneme level is not allowed.
    # Therefore, when doing the recominbation, if one history's last label is A, the expanded new score of hyp ...A A from this history should be -50000
    # On the otherhand, 
    # log_probs is in shape [B,T,V], it is precomputed, V is the vocab size,
    # seq_len is the length of iput, shape [B,]
    # we can provide either an LM or precomputed lm_table
    # k is the beam size that at each step how many hypothesis we keep.
    # the lm table should be in shape [lm_vocab_size**lm_context_length, lm_vocab_size]
    # am_scale and lm_scale weight the AM and LM log scores before summation.
    # we have am vocab size and lm vocab size because the lm has an additional token eos/bos




    B, max_len, am_vocab_size = log_probs.shape
    assert (lm_table is None) != (lm_network is None), "provide exactly one of lm_table or lm_network"
    assert loop_penalty >= 1.0, "loop_penalty must be >= 1.0 because it scales log-prob penalties"
    score_padding = torch.ones((B,k), dtype=log_probs.dtype, device=log_probs.device)*-50000 # used to be cat with head/tail scores later for recombination
    if lm_table is not None:
        lm_table = lm_table.detach().to(device=log_probs.device, dtype=log_probs.dtype)
    # assume the eos/bos index is lm_vocab_size-1
    lm_vocab_size_int = int(lm_vocab_size)
    lm_bos_eos_idx = lm_vocab_size_int-1
    assert lm_vocab_size == am_vocab_size + 1 # eow not handled yet

    modified_log_probs = torch.cat([log_probs, log_probs[:,:,0].unsqueeze(-1)], dim=-1)     # now it is only handling the expantion for bos context, i.e. empty seq in hypothesis, but later this can also be used when there is eow in lm, but no eow phonemes in am
    log_probs_for_loop_holder = log_probs.new_full(modified_log_probs.shape, -50000)
    log_probs_for_loop_holder[:,:,1:-1] = log_probs[:,:,0].unsqueeze(-1).expand(-1,-1, lm_vocab_size-2)
    log_probs_for_loop = torch.logaddexp(modified_log_probs * loop_penalty, log_probs_for_loop_holder * loop_penalty)
    # this is a special case here, because we don't distinguish silence and other real labels in loop transition. usually it should, like in ctc, but here we do not allow duplicated labels, so this is fine, silence is like a wildcard label, except for the empty sequence, that is a real silence

    # assume lm starting entry is [bos, bos, bos]
    lm_start_entry = torch.zeros((), dtype=torch.long, device=log_probs.device)
    lm_vocab_size_tensor = torch.tensor(lm_vocab_size_int, dtype=torch.long, device=log_probs.device)
    buffers = [lm_vocab_size_tensor**j for j in range(lm_context_length)]
    for i in range(lm_context_length):
        buffer = buffers[i]
        lm_start_entry += buffer*lm_bos_eos_idx

    assert k >= lm_vocab_size, "too small n-best list"
    assert lm_vocab_size_int >= am_vocab_size, "LM vocab must cover all AM labels"

    if lm_table is not None:
        init_lm_score = lm_table[lm_start_entry] # shape[V]


    elif lm_network is not None:
        # when the context is too long, we compute the lm scores online
        lm_history_tensor = torch.full((1, lm_context_length), lm_bos_eos_idx, dtype=torch.long, device=log_probs.device)
        with torch.no_grad():
            init_lm_score = torch.log_softmax(lm_network(lm_history_tensor), dim=-1)[:, -1, :][0]
        init_lm_score = init_lm_score.to(dtype=log_probs.dtype)

    # first step score
    scores = log_probs.new_full((B,k), -100000)
    scores_valid = am_scale * log_probs[:,0] + lm_scale * init_lm_score[None, :am_vocab_size]
    scores_empty = am_scale * log_probs[:,0,0]

    scores[:,:am_vocab_size] = scores_valid
    scores[:,am_vocab_size] = scores_empty
    # for the first step, just use the silence for empty hypothesis
    context_ids = torch.zeros([B,k],dtype=torch.long, device=log_probs.device) # in this case, silence should not appear in the context, there for we use the index of silence for padding
    
    phon_idx = torch.arange(lm_vocab_size,device=context_ids.device)

    valid_context_ids = lm_bos_eos_idx * sum(buffers[1:]) + phon_idx # e.g. for context 3, the valid context for the first step is [bos, bos, 0], [bos, bos,1],... converted to integers
    valid_context_ids = valid_context_ids[None,:].expand(B,-1) # this is regular phonemes + [bos,bos,bos]
    context_ids[:,:lm_vocab_size] = valid_context_ids

    # also make a [B,k,lm_context_length] tensor for online lm score computation


    # main part
    # for loop over the time dimension
    seq_len = seq_len.to(device=log_probs.device)
    seq_len_mask = torch.arange(max_len, device=log_probs.device)[None,:] < seq_len[:, None]
    for t in range(1,max_len):
        tail_group_idx, num_tail_groups, sorted_tails, perm, tail_pos_mask = context_recombine_allow_loop(context_ids, lm_vocab_size_tensor, context_length=lm_context_length) # the am is always context 1
        
        head_pos_mask = ~tail_pos_mask


        combined_context_ids = torch.cat([context_ids, context_ids], dim=-1) # to be consistent with the perm we got from context_recombine_allow_loop
        sorted_combined_context_ids = combined_context_ids.gather(1, perm)

        sorted_combined_context_last_phon = sorted_combined_context_ids % lm_vocab_size_tensor

        if lm_table is not None:
            lm_scores = lm_table[sorted_combined_context_ids].clone() # we assume that the last index is bos/eos, this is not used in the for-loop because no eos lm prob should be used
            lm_scores[:,:,-1] = -50000
            # we do not allow the duplicated phonemes, therefore maske them out here in the lm score
            lm_scores.scatter_(2, sorted_combined_context_last_phon.unsqueeze(-1), -50000) # the histories expanding the same last phoneme, their scores are masked as a very small value

        elif lm_network is not None:
            flat_context_ids = context_ids.reshape(-1)
            unique_context_ids, inverse_context_indices = torch.unique(
                flat_context_ids,
                sorted=False,
                return_inverse=True,
            )
            history_for_lm = _map_context_id(unique_context_ids, lm_vocab_size_int, lm_context_length)
            with torch.no_grad():
                unique_lm_scores = torch.log_softmax(lm_network(history_for_lm), dim=-1)[:,-1]
            lm_scores_base = unique_lm_scores[inverse_context_indices].reshape(B, k, lm_vocab_size_int)
            combined_lm_scores = torch.cat([lm_scores_base, lm_scores_base], dim=1)
            lm_scores = combined_lm_scores.gather(
                1,
                perm[:, :, None].expand(-1, -1, lm_vocab_size_int),
            ).to(dtype=log_probs.dtype)
            lm_scores[:,:,-1] = -50000
            lm_scores.scatter_(2, sorted_combined_context_last_phon.unsqueeze(-1), -50000)

        else:
            raise NotImplementedError
        expand_group_idx = tail_group_idx[:,:,None].expand(-1,-1, lm_vocab_size) # to allow empty hypothesis, maybe this should be expanded to lm_vocab_size
        combined_scores_for_tail = torch.cat([scores, score_padding], dim=-1)
        # compute tail expand score
        perm_old_scores_tail = combined_scores_for_tail.gather(1, perm)
        perm_tail_combined_score = perm_old_scores_tail[:,:,None] + lm_scale * lm_scores + am_scale * modified_log_probs[:,t,:][:,None,:] # shape [B,k,v_lm], to be compatible with the loop transitions scores later, the expansion of bos/eos is forbidden by the lm_score, (set -1 index score to -50000)

        # now compute the perm_head_score
        combined_scores_for_head = torch.cat([score_padding, scores], dim=-1)

        perm_old_scores_head = combined_scores_for_head.gather(1, perm)
        perm_old_scores_head_expand = perm_old_scores_head.new_full((*perm_old_scores_head.shape, lm_vocab_size),-50000.0).scatter_(2, sorted_combined_context_last_phon.unsqueeze(-1), perm_old_scores_head.unsqueeze(-1)) # also [B,k,V_lm], now this last phon could be bos, infers the empty sequence
        # we also allow silence as the loop transition for each label, except for silence, so on the label/context level, the entry for silence should always be masked out
        perm_head_combined_scores = perm_old_scores_head_expand + am_scale  * log_probs_for_loop[:,t,:][:,None,:]

        valid_tail = torch.arange(2* k, device=scores.device)[None,:] < num_tail_groups[:, None]

        unique_tails = torch.zeros((B,2* k), dtype=context_ids.dtype,device=context_ids.device)
        unique_tails.scatter_(1, tail_group_idx, sorted_tails)

        #perm_combined_score = (
        #    perm_old_scores[:, :, None]
        #    + lm_scale * lm_scores
        #    + am_scale * log_probs[:, t, :][:, None, :]
        #)

        perm_combined_score = torch.logaddexp(perm_head_combined_scores, perm_tail_combined_score) # combine the expand and "fake expand" score from tail, this is still done for each hypothesis, next, do the recombination between hypotheses


        group_max = torch.full((B,2 * k, lm_vocab_size), -50000,dtype=scores.dtype, device=scores.device)
        group_max.scatter_reduce_(dim=1, index=expand_group_idx,src=perm_combined_score,reduce="amax", include_self=True,)
        group_max_detached = group_max.detach()
        exp_scores = torch.exp(perm_combined_score - group_max_detached.gather(1,expand_group_idx))
        group_sum = torch.zeros_like(group_max)
        group_sum.scatter_add_(1, expand_group_idx, exp_scores)
        new_scores = torch.log(group_sum.clamp_min(torch.finfo(group_sum.dtype).tiny)) + group_max_detached #[B,2*k,V]
        # mask out padding groups (question, if group sum is zeros_like, then taking log clamp_min ... .tiny already gives very small values, i.e. the masked_fill below is not necessary?
        new_scores = new_scores.masked_fill(~valid_tail[:,:,None],-100000)
        
        # recombinition for new expanded phonemes finished, now do the recombination of the loop transition
        

        new_scores = new_scores.view(B,-1) # [B,2*k*V], be ready for topk

        new_context_ids = unique_tails[:,:, None] * lm_vocab_size_int + phon_idx[None, None, :] # now it can have bos in the hyp, but since any emission of bos is masked by the lm prob, in principle, the only reasonable hypotheses is [bos, bos, bos] for e.g. context 3
        new_context_ids = new_context_ids.view(B,-1)

        new_scores_k, new_indices_k = torch.topk(new_scores, k=k, dim=-1)
        # only update the still valid postions
        valid_batch = seq_len_mask[:, t][:, None]
        scores = torch.where(valid_batch, new_scores_k, scores) #[B,k]
        context_ids_selected = new_context_ids.gather(1, new_indices_k)

        context_ids = torch.where(valid_batch, context_ids_selected, context_ids)

    final_scores = torch.logsumexp(scores,dim=-1) # for now we don't add lm EOS score

    return final_scores.sum()



        
