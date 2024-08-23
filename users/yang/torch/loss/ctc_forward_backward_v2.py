import torch
from i6_experiments.users.yang.torch.utils.masking import get_seq_mask, get_seq_mask_v2

import time



def ctc_forward(
        log_probs,  # (T, B, F)
        targets,  # (B, S)
        targets_w_bos, # (B S+1)
        targets_w_eos, # (B, S+1)
        input_lengths,  # (B,)
        target_length, # (B,)
        blank_idx=10025,
        eos_idx=0,
        bos_idx=0,
        log_zero=-1e25,  # maybe better than float min for preventing overflowing
        backward=False,
        top_k_list=None, # (B,S,K), only K choices for labels
):


    device = log_probs.device
    input_time_size, batch_size, n_out = log_probs.shape
    max_label_seq_len = targets.shape[1]


    def _shift_tensor(input_tensor, input_length_tensor, target_length_tensor=None, direction=1):
        """
        :param input_tensor: shape (B,T,S) or (B,T), probs for labels or probs for blank
        :param input_length_tensor: shape (B,)
        :param target_length_tensor: shape (B,)
        """
        if len(input_tensor.shape) == 3:
            assert target_length_tensor is not None
            batch_size, max_t, max_s = input_tensor.shape
            t_indices = torch.arange(max_t).unsqueeze(0).repeat(batch_size, 1)
            t_new_indices = (t_indices + input_length_tensor.unsqueeze(1)) % max_t # shift index input_length_tensor
            t_new_indices = t_new_indices.unsqueeze(2).expand(batch_size, max_t, max_s).to(input_tensor.device)

            out_tensor = input_tensor.gather(1, t_new_indices)
            s_indices = torch.arange(max_s).unsqueeze(0).repeat(batch_size,1)
            s_new_indices = (s_indices + target_length_tensor.unsqueeze(1)) % max_s # (B,S)
            s_new_indices = s_new_indices.unsqueeze(1).expand(batch_size, max_t, max_s).to(input_tensor.device)

            out_tensor = out_tensor.gather(2, s_new_indices)



        elif len(input_tensor.shape) == 2:
            batch_size, max_t = input_tensor.shape
            t_indices = torch.arange(max_t).unsqueeze(0).repeat(batch_size, 1)
            t_new_indices = ((t_indices + input_length_tensor.unsqueeze(1)) % max_t).to(input_tensor.device) # shift index input_length_tensor
            out_tensor = input_tensor.gather(1, t_new_indices)
        else:
            assert False

        return out_tensor


    log_gamma_fw = torch.full((input_time_size, batch_size, 2, max_label_seq_len + 1), log_zero, dtype=torch.float32,
                           device=device) # (T,B, 2,S+1) 0 for label, 1 for blank


    targets_w_tdim = targets.unsqueeze(0).expand(input_time_size, -1, -1) # (T, B, S)

    log_probs_target = log_probs.gather(-1, targets_w_tdim.long()) # (T, B, S)
    log_gamma_fw[:, :, 1, 0] = torch.cumsum(log_probs[:, :, blank_idx], dim=0).nan_to_num(neginf=log_zero)
    log_gamma_fw[0, :, 0, 1] = log_probs[0].gather(-1, targets[:, :1].long()).view(-1)

    target_label_shift_diff = (torch.eq(targets_w_eos, targets_w_bos)).to(log_probs.dtype) # (B, S+1)
    # use a float mask to avoid using torch.where
    target_label_shift_diff_mask = target_label_shift_diff * log_zero # (B, S+1) 0 for unmasked pos, log_zero for masked-pos wrong, because of shift

    if backward:
        # this backward is not wrong, but not compatible with the forward, different states
        # still reverse T and S dim, but only for simplifying the masking,

        # (B,T,S) reversed tensor
        backward_log_probs_target = log_probs_target.transpose(1, 0).flip([1,2])
        # move first paddings to the end of the seq
        num_T_paddings = input_time_size - input_lengths
        num_S_paddings = max_label_seq_len - target_length
        backward_log_probs_target = _shift_tensor(backward_log_probs_target, num_T_paddings, num_S_paddings)
        backward_log_probs_target = backward_log_probs_target.transpose(1,0) # (T,B,S)
        backward_blank_prob = log_probs[:,:,blank_idx].flip(0).transpose(0,1)
        backward_blank_prob = _shift_tensor(backward_blank_prob, num_T_paddings)
        backward_blank_prob = backward_blank_prob.transpose(0,1).unsqueeze(-1) # (T,B,1)


        reversed_target_labels = targets_w_bos.flip(-1)
        reversed_target_labels = _shift_tensor(reversed_target_labels, num_S_paddings) # move S paddings to the end
        reversed_target_labels_shift = torch.cat([reversed_target_labels, torch.zeros([batch_size, 1], dtype=reversed_target_labels.dtype, device=device)], dim=-1)[:,1:] # left shifted reversed label seq,
        # because it is already reversed, the paddings does not influence the computation, so just leave it there
        reversed_target_label_mask = (torch.eq(reversed_target_labels, reversed_target_labels_shift)).to(log_probs.dtype) * log_zero # shape (B,S+1)
        backward_log_probs_target = torch.cat([backward_log_probs_target, torch.full([input_time_size, batch_size, 1], log_zero, dtype=log_probs.dtype, device=device)], dim=-1)# (T,B, S+1), s=S,S-1,...,1,<bos>, reversed
        backward_log_probs_target_label_masked = backward_log_probs_target + reversed_target_label_mask.unsqueeze(0)

        # init log_gamma_bw
        log_gamma_bw = torch.full((input_time_size, batch_size, 2, max_label_seq_len + 1), log_zero, dtype=torch.float32,
                            device=device)
        log_gamma_bw[0,:,:,0] = 0
        log_gamma_bw[1: ,:,1, 0] = torch.cumsum(backward_blank_prob.squeeze(-1)[:-1,:], dim=0).nan_to_num(neginf=log_zero)


    # move the top_k forwarding to the first for loop, then done in parallel? check if it is necessary to use .clone()
    # if clone still needed, consider concat the top-k list and the target



        # recompute label mask for bw



# shape (T,B,2,S)
    def _fw_step(t, log_probs_target, log_probs_blank, log_gamma_0, label_mask, log_gamma_1):
        # log_probs_target and label_mask with additional dim K
        # log_gamma_0 in shape (T,B,S+1,K), log_gamma_1 in shape (T,B,S+1)
        prev_gamma_0 = log_gamma_0[t-1,:,:] # try not using clone first
        prev_gamma_1 = log_gamma_1[t-1,:,:] # try not using clone first
        log_probs_t = log_probs_target[t]
        log_label_sum_target_masked = torch.logaddexp(prev_gamma_1.unsqueeze(-1),
                                                      prev_gamma_0 + label_mask)  # (B, S+1, K) repeated labels masked out
        new_gamma_0_label = log_probs_t + torch.logaddexp(log_label_sum_target_masked[:, :-1],
                                                                         prev_gamma_0[:, 1:])
        # update gamma_blank for s=1,2,...S, i.e. init state kept the same
        new_gamma_1_label = log_probs_blank[t].unsqueeze(-1) + torch.logaddexp(
            prev_gamma_0[:, 1:,0], prev_gamma_1[:, 1:])
        log_gamma_0[t, :, 0,1:] = new_gamma_0_label
        log_gamma_1[t, :, 1,1:] = new_gamma_1_label

    def _bw_step(t, log_probs_target, log_probs_target_label_masked, log_probs_blank, log_gamma):
        # log_probs_target should be in shape T,B,S+1,  pad bos to the end
        # log_probs_blank shape [T,B,1]
        # directly use masked label probs
        prev_gamma_0 = log_gamma[t-1,:,0,:].clone()
        prev_gamma_1 = log_gamma[t-1,:,1,:].clone()
        log_probs_t_loop = log_probs_target[t-1, :, 1:]
        log_probs_t_forward = log_probs_target_label_masked[t-1, :, :-1]
        log_probs_blank_t = log_probs_blank[t-1]
        new_gamma_0_forward = torch.logaddexp(log_probs_t_loop + prev_gamma_0[:,1:], log_probs_blank_t + prev_gamma_1[:,1:]) # (s=S-1, S-2, ..., <bos>), reversed
        new_gamma_0_forward = torch.logaddexp(new_gamma_0_forward, prev_gamma_0[:,:-1] + log_probs_t_forward) # next label (reversed)
        new_gamma_1_forward = torch.logaddexp(log_probs_target[t-1, :,:-1] + prev_gamma_0[:,:-1], prev_gamma_1[:,1:] + log_probs_blank_t)

        log_gamma[t, :,0,1:] = new_gamma_0_forward
        log_gamma[t, :, 1, 1:] = new_gamma_1_forward

        log_gamma[t, :, 0,0] = torch.logaddexp(prev_gamma_1[:,0] + log_probs_blank_t.squeeze(-1), prev_gamma_0[:,0] +log_probs_target[t-1,:,0])



        #return log_gamma[t-1, :, 0,:-1], log_gamma[t-1, :, 1,:-1]
    # in the for loop, pure computation

    for t in range(1, input_time_size):
        _fw_step(t, log_probs_target, log_probs[:, :, blank_idx],log_gamma_fw, target_label_shift_diff_mask)
        if backward:
            _bw_step(t, backward_log_probs_target, backward_log_probs_target_label_masked, backward_blank_prob, log_gamma_bw )



    if backward:
        # reverse the computed bw tensor


        log_gamma_bw = log_gamma_bw.transpose(0, 1) # (B,T,2,S+1)
        # before reverse back, extend to (B,T+1,2,S+2)
        log_gamma_bw = torch.cat([torch.full((batch_size, input_time_size,2,1), log_zero, dtype=torch.float32, device=device), log_gamma_bw], dim=-1) # (B,T,2,S+1) to (B,T,2,S+2)
        log_gamma_bw = torch.cat([torch.full((batch_size, 1,2,max_label_seq_len+2), log_zero, dtype=torch.float32, device=device), log_gamma_bw], dim=1)


        log_gamma_bw = log_gamma_bw.flip([1,-1])
        log_gamma_bw_0 = _shift_tensor(log_gamma_bw[:,:,0,:], num_T_paddings, num_S_paddings)
        log_gamma_bw_1 = _shift_tensor(log_gamma_bw[:,:,1,:], num_T_paddings, num_S_paddings)
        log_gamma_bw = torch.stack([log_gamma_bw_0, log_gamma_bw_1], dim=2).transpose(0,1)




    # replace each target by top-k list, compute the prob:

    # check if all the fw-bw scores are correct

    log_gamma_bw_cut = log_gamma_bw[:-1,:,:,:-1] # (T,B,2,S+1)
    combine = log_gamma_bw_cut + log_gamma_fw
    combine = combine.logsumexp(dim=-1)
    combine = combine.logsumexp(dim=-1)



    if top_k_list is not None:
        top_k = top_k_list.shape[2]
        top_k_list_flat = top_k_list.view(batch_size, max_label_seq_len*top_k)
        top_k_list_idx = top_k_list_flat.unsqueeze(0).expand(input_time_size,-1,-1).long()
        log_probs_select_k = log_probs.gather(-1, top_k_list_idx).view(input_time_size, batch_size,max_label_seq_len, top_k) # (T,B,S,K)
        # at true label position s: label: fw s-1, bw s+1, blank: fw s-1, bw s+1 namely fw [:-1], bw [1:]
        # at time frame t: fw t-1, prob t, prob t+1, bw t+2
        # compute the forward only on (B, S, K)?
        fw_top_k = torch.full((batch_size, max_label_seq_len, top_k), log_zero, dtype=torch.float32,
                            device=device) # (B,S,K)
        # prepare for the backward part
        # log_gamma_bw in shape (T+1,B,2,S+2)
        log_probs_next_target = torch.cat([log_probs_target, torch.full((input_time_size, batch_size, 1), log_zero, dtype=torch.float32, device=device)], dim=-1) #( T,B, S+1) s=1, 2,3,... ,<eos>
        log_probs_next_target = torch.cat([log_probs_next_target, torch.zeros([1,batch_size, max_label_seq_len+1], dtype=log_probs_target.dtype, device=device)], dim=0) # (T, B, S) to (T+1, B, S)

        bw_combine_label = log_probs_next_target[:,:,1:] + log_gamma_bw[:,:,0,2:] # (T+1,B, S), s =2, ..., <eos>
        log_probs_next_blank = torch.cat([log_probs[:,:,blank_idx], torch.zeros([1,batch_size], dtype=log_probs_target.dtype, device=device)], dim=0) # (T+1,B)

        bw_combine_blank = log_probs_next_blank.unsqueeze(-1) + log_gamma_bw[:,:,0, 1:-1] # s = 1,..., S (T+1, B,S)

        bw_label_mask = (targets_w_eos[:,1:].unsqueeze(-1) == top_k_list).float() * log_zero # shape (B,S, K)

        bw_combine = torch.logaddexp(bw_combine_blank.unsqueeze(-1), bw_combine_label.unsqueeze(-1)+bw_label_mask.unsqueeze(0)) # (T+1, B, S, K)

        # for bw_combine, when t = T, only S =0, all the others = log_zero
        bw_combine[input_lengths, torch.arange(batch_size, device=device), target_length-1] = 0


        bw_T1_mask = get_seq_mask(input_lengths+1, input_time_size+1, device=device).transpose(0,1) # mask used for sum over T dim, after T+1 the sum shall not be updated

        bw_combine = bw_combine + (1-bw_T1_mask).unsqueeze(-1).unsqueeze(-1) * log_zero #shape (T+1,B,S, K)





        # init fw_top_k and fw_bw_top_k for t=0
        fw_top_k[:,0,:] = log_probs_select_k[0,:,0,:]
        fw_label_mask = (targets_w_bos[:,:-1].unsqueeze(-1) == top_k_list).float() * log_zero #( B,S,K)

        fw_bw_top_k = fw_top_k + bw_combine[1]

        # debug, check the last one, s = S
        #print("####!!! check bw_combine", bw_combine[-5:,0,-4:].detach().cpu().numpy())


        for t in range(1, input_time_size):
            assert t+1 < bw_combine.shape[0]
            fw_pre = torch.logaddexp(log_gamma_fw[t-1,:,0,:-1].unsqueeze(-1) + fw_label_mask, log_gamma_fw[t-1,:,1,:-1].unsqueeze(-1)) # (B,S, K)#

            new_fw_top_k = torch.logaddexp(fw_top_k, fw_pre) + log_probs_select_k[t] # (B,S,K) # only forward needed to be computed here



            cur_fw_bw_top_k = new_fw_top_k + bw_combine[t+1]
            fw_top_k = new_fw_top_k

            fw_bw_top_k = torch.logaddexp(fw_bw_top_k, cur_fw_bw_top_k) # add t-th node, if larger than length, cur_fw_bw_top_k should be log-zero
        # last time frame



    if top_k_list is not None:
        return log_gamma_fw, (log_gamma_bw, fw_bw_top_k)
    if backward:
        return log_gamma_fw, log_gamma_bw



    return log_gamma_fw




# compute forward backward separately?