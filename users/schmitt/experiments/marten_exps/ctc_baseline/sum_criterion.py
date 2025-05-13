"""
Implement lattice-free MMI training for CTC
"""

import os
import json
import torch
import warnings
import numpy as np

from i6_experiments.users.mueller.experiments.language_models.ffnn import FeedForwardLm
from i6_experiments.users.mueller.experiments.ctc_baseline.model import Model
from i6_experiments.users.mueller.experiments.ctc_baseline.recombination import safe_logsumexp, safe_logaddexp, scatter_safe_logsumexp
from i6_experiments.users.mueller.experiments.ctc_baseline import recombination

import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.tensor import batch_dim
from returnn.torch.util import diagnose_gpu

####### Be careful about stuffs related to EOS and blank index with the BPE setup
def sum_loss_bigram(
    *,
    log_probs: torch.Tensor, # (T, B, V)
    log_lm_probs: torch.Tensor, # (V, V)
    log_prior: torch.Tensor | None, # (V,)
    input_lengths: torch.Tensor, # (B,)
    top_k: int = 0,
    LM_order: int,
    am_scale: float,
    lm_scale: float,
    prior_scale: float,
    horizontal_prior: bool,
    blank_prior: bool = True,
    blank_idx:int = 0, # should be same as eos index
    eos_idx: int | None = None,
    unk_idx: int = 1,
    log_zero: float = float("-inf"),
    device: torch.device = torch.device("cpu"),
    print_best_path_for_idx: list[int] = [],
    alignment_topk: bool = False,
):
    """
    Sum criterion training for CTC, given by
    L = sum_{all seq} q(seq)
    where q(seq) = p_AM^alpha(seq) * p_LM^beta(seq) / p_PR(seq)
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
    
    use_prior = log_prior is not None
    
    old_device = log_probs.device
    log_probs = log_probs.to(device)
    log_lm_probs = log_lm_probs.to(device)
    if use_prior:
        log_prior = log_prior.to(device)
    input_lengths = input_lengths.to(device)
    
    max_audio_time, batch_size, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs = am_scale * log_probs
    if lm_scale == 0.0:
        log_lm_probs = torch.zeros_like(log_lm_probs, device=device)
    else:
        log_lm_probs = lm_scale * log_lm_probs
    if use_prior:
        log_prior = prior_scale * log_prior
    
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
    if use_prior:
        assert log_prior.size() == (n_out + 1,) or log_prior.size() == (n_out,), f"Prior shape is not correct, should be {n_out} or {n_out + 1} but is {log_prior.size()}"
    
    # calculate empty sequence score
    # Empty am score = sum log prob blank
    if use_prior and blank_prior:
        log_partial_empty_seq_prob = (log_probs[:, :, blank_idx] - log_prior[blank_idx]).cumsum(dim=0)
    else:
        log_partial_empty_seq_prob = log_probs[:, :, blank_idx].cumsum(dim=0)
    log_empty_seq_prob = log_partial_empty_seq_prob.gather(0, (input_lengths-1).long().unsqueeze(0)).squeeze(0)
    # Empty lm score = p_LM(eos | bos), prior score = p_PR(eos)
    log_q_empty_seq = log_empty_seq_prob + log_lm_probs[eos_symbol, eos_symbol]

    # Unbind the log_probs and log_partial_empty_seq_prob for each timestep so it is faster during backprop
    log_probs = log_probs.unbind(0)
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
    log_q_label = log_probs[0][:, out_idx_vocab] + log_lm_probs[eos_symbol, out_idx_vocab].unsqueeze(0)
    if use_prior:
        log_q_label = log_q_label - log_prior[out_idx_vocab].unsqueeze(0)
    log_q_blank = torch.full((batch_size, vocab_size), log_zero, device=device) # (B, 2, V-1), no blank and eos in last dim
    log_q = log_q_label
    
    # Calculate initial top k if needed
    if top_k > 0:
        if alignment_topk:
            tmp_log_q = torch.cat([log_q_label, log_q_blank, log_partial_empty_seq_prob[0].unsqueeze(-1)], dim=-1)
        else:
            tmp_log_q = torch.cat([log_q, log_partial_empty_seq_prob[0].unsqueeze(-1)], dim=-1)
        topk_scores, topk_idx = torch.topk(tmp_log_q, top_k, dim=-1, sorted=False)
        # p_idx = topk_idx.clone()
        # p_idx[p_idx == vocab_size] = 0
        # print(p_idx[print_best_path_for_idx[0]], topk_scores[print_best_path_for_idx[0]], log_q[print_best_path_for_idx[0], p_idx[print_best_path_for_idx[0]]])
    
    # Set up the best path print
    if print_best_path_for_idx:
        with torch.no_grad():
            best_path_print = {}
            max_val, max_idx = torch.max(log_q, dim=-1)
            for idx in print_best_path_for_idx:
                best_path_print[idx] = {"str": f"{max_idx[idx] + 2}", "am_str": "{:.2f}".format(log_probs[0][idx][max_idx[idx] + 2].tolist()), "prior": "{:.2f}".format(log_prior[max_idx[idx] + 2].tolist() if use_prior else 0.0), "LM": "{:.2f}".format(log_lm_probs[eos_symbol, max_idx[idx] + 2].tolist()), "score": "{:.2f}".format(max_val[idx].tolist()), "AM": log_probs[0][idx][max_idx[idx] + 2].tolist()}
    
    log_lm_probs_wo_eos = log_lm_probs[out_idx_vocab][:, out_idx_vocab].fill_diagonal_(log_zero)
    for t in range(1, max_audio_time):
        # case 1: emit a blank at t
        # Q(t, u, blank) = [Q(t-1, u, blank) + Q(t-1, u, non-blank)]*p_AM(blank | x_t)
        if top_k > 0:
            # Only consider blanks from top k indexes (excluding the blank sequence)
            mask_label = torch.zeros((batch_size, vocab_size), device=device, dtype=torch.bool)
            if alignment_topk:
                mask_blank = torch.zeros((batch_size, vocab_size), device=device, dtype=torch.bool)
            for b in range(batch_size):
                mask_label[b, topk_idx[b][topk_idx[b] < vocab_size]] = True
                if alignment_topk:
                    mask_blank[b, topk_idx[b][(topk_idx[b] >= vocab_size) & (topk_idx[b] < 2 * vocab_size)] - vocab_size] = True
            if alignment_topk:
                new_log_q_blank_l = log_q_label + log_probs[t][:, blank_idx].unsqueeze(-1)
                new_log_q_blank_l = torch.where(mask_label, new_log_q_blank_l, log_zero)
                new_log_q_blank_b = log_q_blank + log_probs[t][:, blank_idx].unsqueeze(-1)
                new_log_q_blank_b = torch.where(mask_blank, new_log_q_blank_b, log_zero)
                new_log_q_blank = safe_logaddexp(new_log_q_blank_l, new_log_q_blank_b)
            else:
                new_log_q_blank = log_q + log_probs[t][:, blank_idx].unsqueeze(-1)
                new_log_q_blank = torch.where(mask_label, new_log_q_blank, log_zero)
        else:
            new_log_q_blank = log_q + log_probs[t][:, blank_idx].unsqueeze(-1)
        if use_prior and blank_prior:
            new_log_q_blank = new_log_q_blank - log_prior[blank_idx]

        # case 2: emit a non-blank at t
        # Q(t, u, non-blank) = p_AM(u|x_t) * [horizontal + diagonal + skip] 
        
        # horizontal transition Q(t-1, u, non-blank)
        if top_k > 0:
            log_mass_horizontal = log_q_label
            log_mass_horizontal = torch.where(mask_label, log_mass_horizontal, log_zero)
        else:
            log_mass_horizontal = log_q_label
        if horizontal_prior and use_prior:
            log_mass_horizontal = log_mass_horizontal - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
        # diagonal transition sum_v Q(t-1, v, blank) * p_LM(u|v) / p_PR(u)
        # take batch index b into account, this is equivalent to compute
        # mass_diagonal[b, u] = sum_v Q(t-1, b, blank, v) * p_LM(u|v) / p_PR(u)
        # mass_diagonal = Q(t-1, :, blank, :) @ M / p_PR(u), where M(v,u) = p_LM(u|v) = lm_probs[v][u]
        # important: in this transition, there is a prefix empty^(t-1) that is not covered in the Q(t-1,v,blank)
        # this is covered in log_partial_empty_seq_prob[t-1]
        if top_k > 0:
            if alignment_topk:
                empty = torch.full((batch_size, vocab_size), log_zero, device=device)
                log_q_blank_topk = torch.cat([empty, log_q_blank, log_partial_empty_seq_prob[t-1].unsqueeze(-1)], dim=-1).gather(-1, topk_idx) # (B, K)
            else:
                log_q_blank_topk = torch.cat([log_q_blank, log_partial_empty_seq_prob[t-1].unsqueeze(-1)], dim=-1).gather(-1, topk_idx) # (B, K)
            log_lm_probs_topk = log_lm_probs[out_idx_vocab][:, out_idx_vocab].unsqueeze(0).expand(batch_size, -1, -1) # (B, V-1, V-1)
            if alignment_topk:
                log_lm_probs_topk = torch.cat([log_lm_probs_topk, log_lm_probs_topk, log_lm_probs[eos_symbol, out_idx_vocab].unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)], dim=1) # (B, V, V-1)
            else:
                log_lm_probs_topk = torch.cat([log_lm_probs_topk, log_lm_probs[eos_symbol, out_idx_vocab].unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)], dim=1) # (B, V, V-1)
            log_lm_probs_topk = log_lm_probs_topk.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size)) # (B, K, V-1)
            log_mass_diagonal = log_matmul(log_q_blank_topk, log_lm_probs_topk, batch_given=True) # (B, K+1) @ (B, K+1, V-1) -> (B, V-1)
        else:
            log_prev_partial_seq_probs = torch.cat([log_partial_empty_seq_prob[t-1].unsqueeze(-1), log_q_blank], dim=-1) # (B, V)
            log_mass_diagonal = log_matmul(log_prev_partial_seq_probs, log_lm_probs[out_idx_vocab_w_eos][:, out_idx_vocab]) # (B, V) @ (V, V-1) -> (B, V-1)
        if use_prior:
            log_mass_diagonal = log_mass_diagonal - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
        # skip transition sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w) / p_PR(u)
        # same consideration as diagonal transition
        if top_k > 0:
            if alignment_topk:
                empty = torch.full((batch_size, vocab_size), log_zero, device=device)
                log_q_label_topk = torch.cat([log_q_label, empty, torch.full((batch_size, 1), log_zero, device=device)], dim=-1).gather(-1, topk_idx)
                log_lm_probs_wo_eos_topk = torch.cat([log_lm_probs_wo_eos, log_lm_probs_wo_eos, torch.full((1, vocab_size), log_zero, device=device)], dim=0).unsqueeze(0).expand(batch_size, -1, -1).gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size)) # (B, K, V-1)
            else:
                log_q_label_topk = torch.cat([log_q_label, torch.full((batch_size, 1), log_zero, device=device)], dim=-1).gather(-1, topk_idx)
                log_lm_probs_wo_eos_topk = torch.cat([log_lm_probs_wo_eos, torch.full((1, vocab_size), log_zero, device=device)], dim=0).unsqueeze(0).expand(batch_size, -1, -1).gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size)) # (B, K, V-1)
            log_mass_skip = log_matmul(log_q_label_topk, log_lm_probs_wo_eos_topk, batch_given=True) # (B, K) @ (B, K, V-1) -> (B, V-1)
        else:
            log_mass_skip = log_matmul(log_q_label, log_lm_probs_wo_eos) # (B, V-1) @ (V-1, V-1) -> (B, V-1)
        if use_prior:
            log_mass_skip = log_mass_skip - log_prior[out_idx_vocab].unsqueeze(0) # divide by prior
        
        # multiply with p_AM(u|x_t)
        new_log_q_label = log_probs[t][:, out_idx_vocab] + safe_logsumexp(torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1), dim=-1)
        
        # set masked results to log_q
        time_mask = (t < input_lengths).unsqueeze(-1).expand(-1, vocab_size)
        log_q_blank = torch.where(time_mask, new_log_q_blank, log_q_blank)
        log_q_label = torch.where(time_mask, new_log_q_label, log_q_label)
        log_q = safe_logaddexp(log_q_label, log_q_blank)
        
        if print_best_path_for_idx:
            with torch.no_grad():
                max_val, max_idx = torch.max(log_q, dim=-1)
                for idx in print_best_path_for_idx:
                    best_path_print[idx]["str"] += f" {max_idx[idx] + 2}"
                    best_path_print[idx]["am_str"] += " {:.2f}".format(log_probs[t][idx][max_idx[idx] + 2].tolist())
                    best_path_print[idx]["prior"] += " {:.2f}".format(log_prior[max_idx[idx] + 2].tolist() if use_prior else 0.0)
                    best_path_print[idx]["LM"] += " {:.2f}".format(safe_logsumexp(log_lm_probs_wo_eos[:, max_idx[idx]], dim=-1).tolist() if lm_scale > 0.0 else 0.0)
                    best_path_print[idx]["score"] += " {:.2f}".format(max_val[idx].tolist()) #  / (t+1)
                    best_path_print[idx]["AM"] += log_probs[t][idx][max_idx[idx] + 2].tolist()
        
        if top_k > 0:
            tmp_log_partial_empty_seq_prob = torch.where((t < input_lengths), log_partial_empty_seq_prob[t], log_empty_seq_prob).unsqueeze(-1)
            if alignment_topk:
                tmp_log_q = torch.cat([log_q_label, log_q_blank, tmp_log_partial_empty_seq_prob], dim=-1) # TODO: we should only add it if it also was in top k up until now
                # If we are in the last timestep, we also have to add the EOS LM probabaility
                last_mask = (t == input_lengths - 1).unsqueeze(-1).expand(-1, vocab_size * 2 + 1)
                tmp_log_q = torch.where(last_mask, tmp_log_q + torch.cat([log_lm_probs[out_idx_vocab, eos_symbol], log_lm_probs[out_idx_vocab, eos_symbol], log_lm_probs[eos_symbol, eos_symbol].unsqueeze(0)], dim=0).unsqueeze(0), tmp_log_q)
            else:
                tmp_log_q = torch.cat([log_q, tmp_log_partial_empty_seq_prob], dim=-1) # TODO: we should only add it if it also was in top k up until now
                # If we are in the last timestep, we also have to add the EOS LM probabaility
                last_mask = (t == input_lengths - 1).unsqueeze(-1).expand(-1, vocab_size + 1)
                tmp_log_q = torch.where(last_mask, tmp_log_q + torch.cat([log_lm_probs[out_idx_vocab, eos_symbol], log_lm_probs[eos_symbol, eos_symbol].unsqueeze(0)], dim=0).unsqueeze(0), tmp_log_q)
            # Calculate top k and apply time mask
            time_mask_k = (t < input_lengths).unsqueeze(-1).expand(-1, top_k)
            new_topk_scores, new_topk_idx = torch.topk(tmp_log_q, top_k, dim=-1, sorted=False)
            topk_scores = torch.where(time_mask_k, new_topk_scores, topk_scores)
            topk_idx = torch.where(time_mask_k, new_topk_idx, topk_idx)
            # p_idx = topk_idx.clone()
            # p_idx[p_idx == vocab_size] = 0
            # print(p_idx[print_best_path_for_idx[0]], topk_scores[print_best_path_for_idx[0]], log_q[print_best_path_for_idx[0], p_idx[print_best_path_for_idx[0]]])
        
        torch.cuda.empty_cache()
    
    # multiply last Q with p_LM(eos | u) and devide by prior of EOS
    if top_k > 0:
        log_q = topk_scores
    else:
        log_q = log_q + log_lm_probs[out_idx_vocab, eos_symbol].unsqueeze(0)
    if print_best_path_for_idx:
        with torch.no_grad():
            for idx in print_best_path_for_idx:
                print(f"Best path for {idx}: {best_path_print[idx]['str']}\nAM str: {best_path_print[idx]['am_str']}\nPrior: {best_path_print[idx]['prior']}\nLM: {best_path_print[idx]['LM']}\nScore: {best_path_print[idx]['score']}\nAM: {best_path_print[idx]['AM']}")
    
    # sum over the vocab dimension
    sum_score = safe_logsumexp(log_q, dim=-1)
    if top_k <= 0:
        # add empty sequence score
        sum_score = safe_logaddexp(sum_score, log_q_empty_seq) # (B,)
    
    loss = -sum_score
    if old_device != device:
        loss = loss.to(old_device)
    
    return loss

def sum_loss_ngram(
    *,
    log_probs: torch.Tensor, # (T, B, V)
    log_lm_probs: torch.Tensor, # (V, V)
    log_prior: torch.Tensor | None, # (V,)
    input_lengths: torch.Tensor, # (B,)
    top_k: int = 0,
    LM_order: int,
    am_scale: float,
    lm_scale: float,
    prior_scale: float,
    horizontal_prior: bool,
    blank_prior: bool = True,
    blank_idx:int = 0,
    eos_idx: int | None = None,
    unk_idx: int = 1,
    log_zero: float = float("-inf"),
    device: torch.device = torch.device("cpu"),
    print_best_path_for_idx: list[int] = [],
    alignment_topk: bool = False,
    blank_correction_version: int = 0,
    correction_in_final_score: bool = False
):
    """
    Sum criterion training for CTC, given by
    L = sum_{all seq} q(seq)
    where q(seq) = p_AM^alpha(seq) * p_LM^beta(seq) / p_PR(seq)
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
    use_prior = log_prior is not None
    
    old_device = log_probs.device
    log_probs = log_probs.to(device)
    log_lm_probs = log_lm_probs.to(device)
    if use_prior:
        log_prior = log_prior.to(device)
    input_lengths = input_lengths.to(device)
    
    max_audio_time, batch_size, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs = am_scale * log_probs
    if lm_scale == 0.0:
        log_lm_probs = torch.zeros_like(log_lm_probs, device=device)
    else:
        log_lm_probs = lm_scale * log_lm_probs
    if use_prior:
        log_prior = prior_scale * log_prior
    
    # print_gradients = PrintGradients.apply
    # grad_assert = AssertGradients.apply
    
    if eos_idx is None or eos_idx == blank_idx: # vocab means no EOS and blank
        vocab_size = n_out - 1
        eos_symbol = blank_idx
        assert eos_idx == 0, "EOS should be the first symbol"
    else:
        vocab_size = n_out - 2
        eos_symbol = eos_idx
        assert eos_idx == 0, "EOS should be the first symbol"
        assert blank_idx == n_out - 1, "blank should be the last symbol"
    assert unk_idx is not None, "unk_idx should be defined"
    vocab_size -= 1 # remove unk from vocab size
    # BoS / EoS and UNK are in the LM
    assert log_lm_probs.size() == (vocab_size + 2,) * LM_order, f"LM shape is not correct, should be {vocab_size + 2} in all dimensions but is {log_lm_probs.size()}"
    if use_prior:
        assert log_prior.size() == (n_out + 1,) or log_prior.size() == (n_out,), f"Prior shape is not correct, should be {n_out} or {n_out + 1} but is {log_prior.size()}"
        if top_k > 0:
            assert horizontal_prior, "Not using the horizontal prior is not implemented for top_k > 0"

    # Unbind the log_probs for each timestep so it is faster during backprop
    log_probs = log_probs.unbind(0)

    # Used to remove blank, unk and eos from the last dim (vocab)
    out_idx = torch.arange(n_out, device=device)
    out_idx_vocab = out_idx[out_idx != blank_idx].long() # "vocab" means no EOS, unk and blank
    out_idx_vocab = out_idx_vocab[out_idx_vocab != unk_idx]
    out_idx_vocab_w_eos = out_idx[out_idx != unk_idx].long()
    if eos_idx is not None and eos_idx != blank_idx:
        out_idx_vocab = out_idx_vocab[out_idx_vocab != eos_idx]
        out_idx_vocab_w_eos = out_idx_vocab_w_eos[out_idx_vocab_w_eos != blank_idx]

    # sum score by DP
    # log_q is the list in which we store the log Q values as tensors of the last N timesteps
    # It is split into ending on a blank and ending on a non-blank
    
    # Init Q for t=0
    # Q(0, u, blank) = 0
    # Q(0, u, non-blank) = p_AM(u | x1) * p_LM(u | bos) / p_PR(u)
    log_q_label_init = log_probs[0][:, out_idx_vocab] + log_lm_probs[*[eos_symbol] * (LM_order - 1), out_idx_vocab].unsqueeze(0)
    if use_prior:
        log_q_label_init = log_q_label_init - log_prior[out_idx_vocab].unsqueeze(0)
    # We have to prepend the BoS symbol even though it is not used in the q_label calculation, but it is need for higher order LMs
    log_q_label_init = torch.cat([torch.full((batch_size, 1), log_zero, device=device), log_q_label_init], dim=1)
    log_q_label = torch.full((batch_size, *(vocab_size + 1,) * (LM_order - 1)), log_zero, device=device) # (B, V, ..., V), no blank in vocab dims
    log_q_label[:, *(0,) * (LM_order - 2)] = log_q_label_init
    log_q_blank = torch.full_like(log_q_label, log_zero, device=device) # (B, V, ..., V)
    # Calculate partial empty sequence score
    # Empty am score = sum log prob blank, lm score = p_LM(eos | bos), prior score = p_PR(eos)
    log_partial_empty_seq_prob = log_probs[0][:, blank_idx]
    if use_prior and blank_prior:
        log_partial_empty_seq_prob = log_partial_empty_seq_prob - log_prior[blank_idx]
    log_q_blank[:, *(0,) * (LM_order - 1)] = log_partial_empty_seq_prob
    log_q = safe_logaddexp(log_q_label, log_q_blank)
    original_shape = log_q.shape
    
    # Calculate initial top k if needed
    if top_k > 0:
        log_q_label = log_q_label.view(batch_size, -1)
        log_q_blank = log_q_blank.view(batch_size, -1)
        log_q = log_q.view(batch_size, -1)
        
        if alignment_topk:
            raise NotImplementedError("Alignment topk is not implemented yet")
            tmp_log_q = torch.stack([log_q_label, log_q_blank], dim=-1).view(batch_size, -1)
        else:
            tmp_log_q = log_q
        
        if blank_correction_version > 0:
            tmp_log_q_blank = log_q_blank.clone() if not correction_in_final_score else log_q_blank
            first_lm_probs = log_lm_probs[*[eos_symbol] * (LM_order - 1), out_idx_vocab_w_eos]
            if blank_correction_version in [1, 2]: # mean next lm prob
                vocab_size_log = torch.log(torch.tensor(vocab_size + 1, device=device, dtype=log_probs[0].dtype))
                tmp_log_q_blank[:, 0] += safe_logsumexp(first_lm_probs, dim=-1) - vocab_size_log
            elif blank_correction_version in [3, 4]: # median next lm prob
                tmp_log_q_blank[:, 0] += torch.median(first_lm_probs, dim=-1).values
            elif blank_correction_version in [5, 6]: # 90% quantile next lm prob
                tmp_log_q_blank[:, 0] += torch.quantile(first_lm_probs, 0.9, dim=-1, interpolation="higher")
            elif blank_correction_version in [7, 8]: # average over top 10% next lm prob
                top_10 = torch.topk(first_lm_probs, int(0.1 * (vocab_size + 1)), dim=-1, sorted=False).values
                top_10_size_log = torch.log(torch.tensor(top_10.size(0), device=device, dtype=log_probs[0].dtype))
                tmp_log_q_blank[:, 0] += safe_logsumexp(top_10, dim=-1) - top_10_size_log
            elif blank_correction_version in [9, 10]: # average over top 5% next lm prob
                top_5 = torch.topk(first_lm_probs, int(0.05 * (vocab_size + 1)), dim=-1, sorted=False).values
                top_5_size_log = torch.log(torch.tensor(top_5.size(0), device=device, dtype=log_probs[0].dtype))
                tmp_log_q_blank[:, 0] += safe_logsumexp(top_5, dim=-1) - top_5_size_log
            elif blank_correction_version in [11, 12]: # 70% quantile next lm prob
                tmp_log_q_blank[:, 0] += torch.quantile(first_lm_probs, 0.7, dim=-1, interpolation="higher")
            elif blank_correction_version in [13, 14]: # 80% quantile next lm prob
                tmp_log_q_blank[:, 0] += torch.quantile(first_lm_probs, 0.8, dim=-1, interpolation="higher")
            elif blank_correction_version in [15, 16]: # 87% quantile next lm prob
                tmp_log_q_blank[:, 0] += torch.quantile(first_lm_probs, 0.87, dim=-1, interpolation="higher")
            else:
                raise NotImplementedError(f"Blank correction version {blank_correction_version} is not implemented")
            tmp_log_q = safe_logaddexp(log_q_label, tmp_log_q_blank)
            if correction_in_final_score:
                log_q = tmp_log_q
        
        topk_scores, topk_idx = torch.topk(tmp_log_q, top_k, dim=1) #, sorted=False) # TODO replace log_q with topk_scores
        topk_idx = torch.tensor(np.array([np.unravel_index(topk_idx[b].cpu().numpy(), (log_q.size(1), 2)) for b in range(batch_size)]), device=device).transpose(1,2) if alignment_topk else topk_idx
        # print(topk_idx.shape)
        
        new_last_idx = torch.arange(vocab_size + 1, device=device)[None, None, None, :].expand(batch_size, top_k, 1, vocab_size + 1)
        new_idx = torch.tensor(np.array([np.unravel_index(topk_idx[b].cpu().numpy(), original_shape[1:]) for b in range(batch_size)]), device=device).transpose(1,2)[:, :, 1:].unsqueeze(-1).expand(-1, -1, -1, vocab_size + 1)
        new_idx = torch.cat([new_idx, new_last_idx], dim=2)
        new_idx = torch.tensor(np.array([[np.ravel_multi_index(new_idx[b, k].cpu().numpy(), original_shape[1:]) for k in range(top_k)] for b in range(batch_size)]), device=device)
        new_idx = new_idx.view(batch_size, -1)
        # print(topk_idx[print_best_path_for_idx[0]], topk_scores[print_best_path_for_idx[0]], log_q[print_best_path_for_idx[0], topk_idx[print_best_path_for_idx[0]]])
    
    # Set up the best path print
    if print_best_path_for_idx:
        with torch.no_grad():
            best_path_print = {}
            max_val, max_idx = torch.max(log_q.view(batch_size, -1), dim=-1)
            max_idx = torch.tensor([np.unravel_index(max_idx[b].cpu().numpy(), (vocab_size + 1,) * (LM_order - 1)) for b in range(batch_size)], device=device)
            if top_k > 0:
                tmp_topk_idx = torch.tensor(np.array([np.unravel_index(topk_idx[b].cpu().numpy(), original_shape[1:]) for b in range(batch_size)]), device=device).transpose(1,2)
            for idx in print_best_path_for_idx:
                m_idx = max_idx[idx].clone()
                m_idx[m_idx > 0] += 1
                a_idx = m_idx.clone()
                a_idx[a_idx == 0] = blank_idx
                best_path_print[idx] = {"str": f"{m_idx.tolist()}", "am_str": "{:.2f}".format(log_probs[0][idx][a_idx[-1]].tolist()), "prior": "{:.2f}".format(log_prior[a_idx[-1]].tolist() if use_prior else 0.0), "LM": "{:.2f}".format(log_lm_probs[*[eos_symbol] * (LM_order - 1), m_idx[-1]].tolist()), "score": "{:.2f}".format(max_val[idx].tolist()), "AM": log_probs[0][idx][a_idx[-1]].tolist()} # TODO AM greedy score
                for k in range(top_k):
                    if k == 5:
                        break
                    k_idx = tmp_topk_idx[idx, k].clone()
                    k_idx[k_idx > 0] += 1
                    best_path_print[idx][f"top{k}_str"] = f"{k_idx.tolist()}"
    
    if blank_correction_version > 0 and top_k > 0:
        # Prepare lm tensor for blank transition
        log_lm_probs_w_eos = dynamic_slice(log_lm_probs, [out_idx_vocab_w_eos] * (LM_order)).view(-1, vocab_size + 1).unsqueeze(0)
        if blank_correction_version in [1, 2]: # mean next lm prob
            log_lm_probs_w_eos = safe_logsumexp(log_lm_probs_w_eos, dim=-1) - vocab_size_log
        elif blank_correction_version in [3, 4]: # median next lm prob
            log_lm_probs_w_eos = torch.median(log_lm_probs_w_eos, dim=-1).values
        elif blank_correction_version in [5, 6]: # 90% quantile next lm prob
            log_lm_probs_w_eos = torch.quantile(log_lm_probs_w_eos, 0.9, dim=-1, interpolation="higher")
        elif blank_correction_version in [7, 8]: # average over top 10% next lm prob
            top_10 = torch.topk(log_lm_probs_w_eos, int(0.1 * (vocab_size + 1)), dim=-1, sorted=False).values
            log_lm_probs_w_eos = safe_logsumexp(top_10, dim=-1) - top_10_size_log
        elif blank_correction_version in [9, 10]: # average over top 5% next lm prob
            top_5 = torch.topk(log_lm_probs_w_eos, int(0.05 * (vocab_size + 1)), dim=-1, sorted=False).values
            log_lm_probs_w_eos = safe_logsumexp(top_5, dim=-1) - top_5_size_log
        elif blank_correction_version in [11, 12]: # 70% quantile next lm prob
            log_lm_probs_w_eos = torch.quantile(log_lm_probs_w_eos, 0.7, dim=-1, interpolation="higher")
        elif blank_correction_version in [13, 14]: # 80% quantile next lm prob
            log_lm_probs_w_eos = torch.quantile(log_lm_probs_w_eos, 0.8, dim=-1, interpolation="higher")
        elif blank_correction_version in [15, 16]: # 87% quantile next lm prob
            log_lm_probs_w_eos = torch.quantile(log_lm_probs_w_eos, 0.87, dim=-1, interpolation="higher")
    # Prepare lm tensor for the diagonal transition
    log_lm_probs_wo_last_eos = dynamic_slice(log_lm_probs, [out_idx_vocab_w_eos] * (LM_order - 1) + [out_idx_vocab])
    log_lm_probs_wo_last_eos = torch.cat([torch.full((*log_lm_probs_wo_last_eos.size()[:-1], 1), log_zero, device=device), log_lm_probs_wo_last_eos], dim=-1) # EoS in last dimension is set to log_zero
    # Prepare lm tensor for the skip transition
    log_lm_probs_wo_diag = dynamic_slice(log_lm_probs, [out_idx_vocab_w_eos] * (LM_order - 2) + [out_idx_vocab] * 2)
    log_lm_probs_wo_diag = torch.cat([torch.full((*log_lm_probs_wo_diag.size()[:-2], 1, vocab_size), log_zero, device=device), log_lm_probs_wo_diag], dim=-2) # EoS in penultimate dimension is set to log_zero
    log_lm_probs_wo_diag = torch.cat([torch.full((*log_lm_probs_wo_diag.size()[:-1], 1), log_zero, device=device), log_lm_probs_wo_diag], dim=-1) # EoS in last dimension is set to log_zero
    log_lm_probs_wo_diag = (1 - torch.eye(vocab_size + 1, device=device)).log() + log_lm_probs_wo_diag # Fill diagonal in last two dimensions with log_zero as we don't allow repetitions of the same label here
    # Prepare lm tensor for EoS transition
    log_lm_probs_eos = dynamic_slice(log_lm_probs, [out_idx_vocab_w_eos] * (LM_order - 1) + [torch.tensor([eos_symbol], device=device)]).squeeze(-1).unsqueeze(0)
    if top_k > 0 and lm_scale > 0.0:
        log_lm_probs_eos[log_lm_probs_eos == float("-inf")] = -1000000.0 # Set to a very low value to avoid having -inf scores in the top k
    # Prepare prior
    if use_prior:
        log_prior_wo_bos = torch.cat([torch.full((1,), 0.0, device=device), log_prior[out_idx_vocab]], dim=0).unsqueeze(0)[:, *(None,) * (LM_order - 2), :].expand(original_shape)
        if top_k > 0:
            log_prior_wo_bos = log_prior_wo_bos.reshape(batch_size, -1)
    
    # print(safe_logsumexp(log_q[0], dim=-1), safe_logsumexp(log_q[0], dim=-1).exp())
    
    for t in range(1, max_audio_time):
        # case 1: emit a blank at t
        # Q(t, u, blank) = [Q(t-1, u, blank) + Q(t-1, u, non-blank)]*p_AM(blank | x_t)
        if top_k > 0:
            # Only consider blanks following top k sequences
            new_log_q_blank = torch.full_like(log_q, log_zero, device=device)
            if alignment_topk:
                label_topk_idx = topk_idx[topk_idx[:, :, -1] == 0][:-1]
                new_log_q_blank[label_topk_idx] = log_q_label[label_topk_idx] + log_probs[t][:, blank_idx][:, *(None,) * (LM_order - 1)]
                blank_topk_idx = topk_idx[topk_idx[-1] == 1][:-1]
                # We could already have entries from the topk labels, so we have to add
                new_log_q_blank[blank_topk_idx] = safe_logaddexp(new_log_q_blank[blank_topk_idx], log_q_blank[blank_topk_idx] + log_probs[t][:, blank_idx][:, *(None,) * (LM_order - 1)])
            else:
                new_log_q_blank.scatter_(1, topk_idx, log_q.gather(1, topk_idx) + log_probs[t][:, blank_idx].unsqueeze(-1))
        else:
            new_log_q_blank = log_q + log_probs[t][:, blank_idx][:, *(None,) * (LM_order - 1)]
        if use_prior and blank_prior:
            new_log_q_blank = new_log_q_blank - log_prior[blank_idx]

        # case 2: emit a non-blank at t
        # Q(t, u, non-blank) = p_AM(u|x_t) * [horizontal + diagonal + skip]
        
        # horizontal transition Q(t-1, u, non-blank)
        if top_k > 0:
            new_log_q_label = torch.full_like(log_q, log_zero, device=device)
            if blank_correction_version > 0 and blank_correction_version % 2 == 0 and not correction_in_final_score:
                topk_log_q_label = torch.full_like(log_q, log_zero, device=device)
            if alignment_topk:
                new_log_q_label[label_topk_idx] = log_q_label[label_topk_idx] # TODO maybe we need gather instead
            else:
                if blank_correction_version > 0 and blank_correction_version % 2 == 0 and correction_in_final_score:
                    new_log_q_label.scatter_(1, topk_idx, (log_q_label + log_lm_probs_w_eos).gather(1, topk_idx))
                else:
                    new_log_q_label.scatter_(1, topk_idx, log_q_label.gather(1, topk_idx))
                    if blank_correction_version > 0 and blank_correction_version % 2 == 0:
                        topk_log_q_label.scatter_(1, topk_idx, (log_q_label + log_lm_probs_w_eos).gather(1, topk_idx))
        else:
            log_mass_horizontal = log_q_label
            if use_prior and horizontal_prior:
                log_mass_horizontal = log_mass_horizontal - log_prior_wo_bos
        
        # diagonal transition sum_v Q(t-1, v, blank) * p_LM(u|v) / p_PR(u)
        # take batch index b into account, this is equivalent to compute
        # mass_diagonal[b, u] = sum_v Q(t-1, b, blank, v) * p_LM(u|v) / p_PR(u)
        # mass_diagonal = Q(t-1, :, blank, :) @ M / p_PR(u), where M(v,u) = p_LM(u|v) = lm_probs[v][u]
        if top_k > 0:
            log_lm_probs_topk = log_lm_probs_wo_last_eos.view(-1, vocab_size + 1).unsqueeze(0).expand(batch_size, -1, -1) # (B, V, V)
            if alignment_topk:
                log_q_blank_topk = log_q_blank.gather(1, blank_topk_idx) # (B, K)
                log_lm_probs_topk = log_lm_probs_topk.gather(1, blank_topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size + 1)) # (B, K, V-1)
            else:
                log_q_blank_topk = log_q_blank.gather(1, topk_idx) # (B, K)
                log_lm_probs_topk = log_lm_probs_topk.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size + 1)) # (B, K, V)
            log_q_blank_topk = log_q_blank_topk.unsqueeze(-1).expand_as(log_lm_probs_topk)
            log_mass_diagonal_add = log_q_blank_topk + log_lm_probs_topk # (B, K, V)
            log_mass_diagonal_add = log_mass_diagonal_add.view(batch_size, -1)
            new_log_q_label = scatter_safe_logsumexp(new_log_q_label, 1, new_idx, log_mass_diagonal_add, include_self=True)
            if blank_correction_version > 0 and blank_correction_version % 2 == 0 and not correction_in_final_score:
                topk_log_q_label = scatter_safe_logsumexp(topk_log_q_label, 1, new_idx, log_mass_diagonal_add, include_self=True)
        else:
            log_mass_diagonal = log_matmul(log_q_blank, log_lm_probs_wo_last_eos) # (B, V) @ (V, V) -> (B, V)
            if use_prior:
                log_mass_diagonal = log_mass_diagonal - log_prior_wo_bos
        
        # skip transition sum{w!=u} Q(t-1, w, non-blank) * p_LM(u|w) / p_PR(u)
        if top_k > 0:
            log_lm_probs_topk = log_lm_probs_wo_diag.view(-1, vocab_size + 1).unsqueeze(0).expand(batch_size, -1, -1) # (B, V1, ..., Vm)
            if alignment_topk:
                log_q_label_topk = log_q_label.gather(1, label_topk_idx) # (B, K)
                log_lm_probs_topk = log_lm_probs_topk.gather(1, label_topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size + 1)) # (B, K, V-1)
            else:
                log_q_label_topk = log_q_label.gather(1, topk_idx) # (B, K)
                log_lm_probs_topk = log_lm_probs_topk.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, vocab_size + 1)) # (B, K, V)
            log_q_label_topk = log_q_label_topk.unsqueeze(-1).expand_as(log_lm_probs_topk)
            log_mass_skip_add = log_q_label_topk + log_lm_probs_topk # (B, K, V)
            log_mass_skip_add = log_mass_skip_add.view(batch_size, -1)
            new_log_q_label = scatter_safe_logsumexp(new_log_q_label, 1, new_idx, log_mass_skip_add, include_self=True)
            if blank_correction_version > 0 and blank_correction_version % 2 == 0 and not correction_in_final_score:
                topk_log_q_label = scatter_safe_logsumexp(topk_log_q_label, 1, new_idx, log_mass_skip_add, include_self=True)
        else:
            log_mass_skip = log_matmul(log_q_label, log_lm_probs_wo_diag) # (B, V) @ (V, V) -> (B, V)
            if use_prior:
                log_mass_skip = log_mass_skip - log_prior_wo_bos
        
        # add up the three transition types
        if top_k <= 0:
            new_log_q_label = safe_logsumexp(torch.stack([log_mass_horizontal, log_mass_diagonal, log_mass_skip], dim=-1), dim=-1)
        # correct the prior for topK
        elif use_prior:
            new_log_q_label -= log_prior_wo_bos
        
        # multiply with p_AM(u|x_t)
        if top_k > 0:
            label_am = torch.cat([torch.full((batch_size, 1), log_zero, device=device), log_probs[t][:, out_idx_vocab]], dim=-1)[:, *(None,) * (LM_order - 2), :].expand(original_shape).reshape(batch_size, -1)
            new_log_q_label += label_am
            if blank_correction_version > 0 and blank_correction_version % 2 == 0 and not correction_in_final_score:
                topk_log_q_label += label_am
        else:
            new_log_q_label += torch.cat([torch.full((batch_size, 1), log_zero, device=device), log_probs[t][:, out_idx_vocab]], dim=-1)[:, *(None,) * (LM_order - 2), :]
        
        # set masked results to log_q
        time_mask = (t < input_lengths)[:, *(None,) * (LM_order - 1)] if top_k <= 0 else (t < input_lengths).unsqueeze(-1)
        log_q_blank = torch.where(time_mask.expand_as(log_q), new_log_q_blank, log_q_blank)
        log_q_label = torch.where(time_mask.expand_as(log_q), new_log_q_label, log_q_label)
        log_q = safe_logaddexp(log_q_label, log_q_blank)
        # print(safe_logsumexp(log_q[0], dim=-1), safe_logsumexp(log_q[0], dim=-1).exp())
        
        assert torch.all(torch.isneginf(log_q_label[..., 0])), "There should be no probability for the BoS symbol in log_q_label"
        
        if top_k > 0:
            if alignment_topk:
                tmp_log_q = torch.stack([log_q_label, log_q_blank], dim=-1)
            else:
                tmp_log_q = log_q
            
            if blank_correction_version > 0:
                # if print_best_path_for_idx:
                #     with torch.no_grad():
                #         for idx in print_best_path_for_idx:
                #             print(f"Blank correction for {idx} in {t}: {log_lm_probs_w_eos[0].gather(0, topk_idx[idx]).tolist()}")
                            
                tmp_log_q_blank = log_q_blank + log_lm_probs_w_eos
                if blank_correction_version % 2 == 0 and not correction_in_final_score:
                    tmp_log_q_2 = safe_logaddexp(topk_log_q_label, tmp_log_q_blank)
                else:
                    tmp_log_q_2 = safe_logaddexp(log_q_label, tmp_log_q_blank)
                
                if correction_in_final_score:
                    log_q_blank = tmp_log_q_blank
                    tmp_log_q = tmp_log_q_2
                    log_q = tmp_log_q_2
                    
            else:
                tmp_log_q_2 = tmp_log_q
            
            # If we are in the last timestep, we also have to add the EOS LM probability
            last_mask = (t == input_lengths - 1).unsqueeze(-1).expand_as(tmp_log_q)
            tmp_log_q = torch.where(last_mask, tmp_log_q + log_lm_probs_eos.view(1, -1).expand_as(tmp_log_q), tmp_log_q_2)
            # Calculate top k and apply time mask
            new_topk_scores, new_topk_idx = torch.topk(tmp_log_q, top_k, dim=1) #, sorted=False)
            new_topk_idx = torch.tensor(np.array([np.unravel_index(new_topk_idx[b].cpu().numpy(), (log_q.size(1), 2)) for b in range(batch_size)]), device=device).transpose(1,2) if alignment_topk else new_topk_idx
            
            # if blank_correction_version > 0 and print_best_path_for_idx:
            #     with torch.no_grad():
            #         for idx in print_best_path_for_idx:
            #             tmp_topk_idx = torch.tensor(np.array([np.unravel_index(new_topk_idx[b].cpu().numpy(), original_shape[1:]) for b in range(batch_size)]), device=device).transpose(1,2)[idx, :, -1]
            #             print(f"Top-K correction for {idx} in {t}: {log_lm_probs_wo_last_eos.view(-1, vocab_size + 1).gather(0, topk_idx[idx].unsqueeze(-1).expand(-1, vocab_size + 1))[:, tmp_topk_idx].tolist()}")
            
            topk_scores = torch.where(time_mask.expand_as(topk_scores), new_topk_scores, topk_scores)
            topk_idx = torch.where(time_mask.expand_as(topk_idx), new_topk_idx, topk_idx)

            new_last_idx = torch.arange(vocab_size + 1, device=device)[None, None, None, :].expand(batch_size, top_k, 1, vocab_size + 1)
            new_idx = torch.tensor(np.array([np.unravel_index(topk_idx[b].cpu().numpy(), original_shape[1:]) for b in range(batch_size)]), device=device).transpose(1,2)[:, :, 1:].unsqueeze(-1).expand(-1, -1, -1, vocab_size + 1)
            new_idx = torch.cat([new_idx, new_last_idx], dim=2)
            new_idx = torch.tensor(np.array([[np.ravel_multi_index(new_idx[b, k].cpu().numpy(), original_shape[1:]) for k in range(top_k)] for b in range(batch_size)]), device=device)
            new_idx = new_idx.view(batch_size, -1)
            # print(topk_idx[print_best_path_for_idx[0]], topk_scores[print_best_path_for_idx[0]], log_q[print_best_path_for_idx[0], topk_idx[print_best_path_for_idx[0]]])
            
        if print_best_path_for_idx:
            with torch.no_grad():
                max_val, max_idx = torch.max(log_q.view(batch_size, -1), dim=-1)
                max_idx = torch.tensor([np.unravel_index(max_idx[b].cpu().numpy(), (vocab_size + 1,) * (LM_order - 1)) for b in range(batch_size)], device=device)
                if top_k > 0:
                    tmp_topk_idx = torch.tensor(np.array([np.unravel_index(topk_idx[b].cpu().numpy(), original_shape[1:]) for b in range(batch_size)]), device=device).transpose(1,2)
                for idx in print_best_path_for_idx:
                    m_idx = max_idx[idx].clone()
                    m_idx[m_idx > 0] += 1
                    a_idx = m_idx.clone()
                    a_idx[a_idx == 0] = blank_idx
                    
                    best_path_print[idx]["str"] += f" {m_idx.tolist()}"
                    best_path_print[idx]["am_str"] += " {:.2f}".format(log_probs[t][idx][a_idx[-1]].tolist())
                    best_path_print[idx]["prior"] += " {:.2f}".format(log_prior[a_idx[-1]].tolist() if use_prior else 0.0)
                    best_path_print[idx]["LM"] += " {:.2f}".format(safe_logsumexp(log_lm_probs_wo_last_eos[:, *max_idx[idx]], dim=-1).tolist() if lm_scale > 0.0 else 0.0)
                    best_path_print[idx]["score"] += " {:.2f}".format(max_val[idx].tolist()) #  / (t+1)
                    best_path_print[idx]["AM"] += log_probs[t][idx][a_idx[-1]].tolist()
                    for k in range(top_k):
                        if k == 5:
                            break
                        k_idx = tmp_topk_idx[idx, k].clone()
                        k_idx[k_idx > 0] += 1
                        best_path_print[idx][f"top{k}_str"] += f" {k_idx.tolist()}"
        
        torch.cuda.empty_cache()
    
    if top_k > 0:
        sum_score = safe_logsumexp(topk_scores, dim=-1)
    else:
        # multiply last Q with p_LM(eos | u)
        log_q += log_lm_probs_eos
        # sum over the vocab dimensions
        sum_score = log_q
        for _ in range(LM_order - 1):
            sum_score = safe_logsumexp(sum_score, dim=-1)
    
    if print_best_path_for_idx:
        with torch.no_grad():
            for idx in print_best_path_for_idx:
                print(f"\n\nBest path for {idx}: \n{get_bpes(best_path_print[idx]['str'])}\nAM str: {best_path_print[idx]['am_str']}\nPrior: {best_path_print[idx]['prior']}\nLM: {best_path_print[idx]['LM']}\nScore: {best_path_print[idx]['score']}\nAM: {best_path_print[idx]['AM']}")
                for k in range(top_k):
                    if k == 5:
                        break
                    print(f"Top {k + 1} path: \n{get_bpes(best_path_print[idx][f'top{k}_str'])}")
                print("\n\n")
    
    loss = -sum_score
    if old_device != device:
        loss = loss.to(old_device)
    
    return loss

def sum_loss_ngram_rf(
    *,
    model: Model,
    log_probs: rf.Tensor, # (T, B, V)
    log_lm_probs: torch.Tensor | None, # (V, ..., V)
    context_size: int,
    log_prior: rf.Tensor | None, # (V,)
    input_lengths: rf.Dim, # (B,)
    top_k: int = 10,
    am_scale: float,
    lm_scale: float,
    prior_scale: float,
    horizontal_prior: bool,
    blank_prior: bool,
    log_zero: float = float("-inf"),
    device: str = "cpu",
    use_recombination: bool = True,
    recomb_blank: bool = True,
    recomb_after_topk: bool = True,
    recomb_with_sum: bool = False,
    blank_correction_version: int = 0,
    print_best_path_for_idx: list[int] = [],
):
    assert top_k > 0, "Top k should be greater than 0 as exact sum is not implemented for FFNN"
    assert blank_correction_version == 0, "Blank correction is not implemented for FFNN"
    
    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    import returnn
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__
    
    def _update_context(context: rf.Tensor, new_label: rf.Tensor, context_dim: rf.Dim) -> rf.Tensor:
        new_dim = rf.Dim(1, name="new_label")
        new_label = rf.expand_dim(new_label, dim=new_dim)
        old_context, old_context_dim = rf.slice(context, axis=context_dim, start=1)
        new_context, new_context_dim = rf.concat((old_context, old_context_dim), (new_label, new_dim), out_dim=context_dim)
        assert new_context_dim == context_dim
        return new_context
    
    def _target_remove_blank(target: rf.Tensor, *, target_dim: rf.Dim, wb_target_dim: rf.Dim, blank_idx: int) -> rf.Tensor:
        assert target.sparse_dim == wb_target_dim
        assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
        return rf.set_sparse_dim(target, target_dim)

    def _target_dense_extend_blank(
        target: rf.Tensor, *, target_dim: rf.Dim, wb_target_dim: rf.Dim, blank_idx: int, value: float
    ) -> rf.Tensor:
        assert target_dim in target.dims
        assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
        res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
        return res
    
    use_prior = log_prior is not None
    use_lm = log_lm_probs is not None
    
    old_device = log_probs.device
    log_probs = rf.copy_to_device(log_probs, device)
    if use_prior:
        if not blank_prior and model.target_dim in log_prior.dims:
            new_dim = rf.Dim(1)
            log_prior = rf.concat(
                [(log_prior, model.target_dim),(rf.zeros(dims = [new_dim],  dtype="float32", device=log_prior.device), new_dim)],
                out_dim=model.wb_target_dim
            )
        assert model.wb_target_dim in log_prior.dims
        log_prior = rf.copy_to_device(log_prior, device)
    
    batch_dims = [batch_dim]
    batch_size = int(batch_dim.get_dim_value())
    beam_dim = rf.Dim(1, name="initial-beam")
    context_dim = rf.Dim(context_size, name="context")
    batch_dims_ = batch_dims + [beam_dim]
    seq_log_prob = rf.constant(0.0, dims=batch_dims_) # Batch, Beam
    
    # scaled log am and prior probs
    log_probs = am_scale * log_probs
    if use_prior:
        log_prior = prior_scale * log_prior
        
        # If not blank prior this is still applied as the log prior for blank is just 0
        if horizontal_prior:
            log_probs -= log_prior
        
    log_probs = rf.where(
        input_lengths.get_mask(),
        log_probs,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    log_probs_ta = TensorArray.unstack(log_probs, axis=input_lengths)  # t -> Batch, VocabWB
    
    target = rf.constant(model.bos_idx, dims=batch_dims_ + [context_dim], sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB
    
    # Prepare LM
    if use_lm:
        with torch.no_grad():
            indices = []
            for i in range(context_size):
                indices.append(target.raw_tensor[..., i])
            lm_logits = log_lm_probs[*indices]
            assert lm_logits.size(-1) == int(model.target_dim.get_dim_value())
            lm_logits = rf.convert_to_tensor(lm_logits, dims=batch_dims_ + [model.target_dim], dtype="float32", device=device, name="lm_logits")
            assert lm_logits.dims == (*batch_dims_, model.target_dim)
            # lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
            lm_log_probs *= lm_scale
    
    max_seq_len = int(input_lengths.get_dim_value())
    backrefs = None
    if use_recombination:
        assert len(batch_dims) == 1
        if recomb_after_topk:
            seq_hash = rf.constant(0, dims=batch_dims_, dtype="int64")
        else:
            seq_hash = rf.constant(0, dims=batch_dims_ + [model.wb_target_dim], dtype="int64")
    
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        seq_log_prob = seq_log_prob + log_probs_ta[t]  # Batch, InBeam, VocabWB

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if use_lm:
                # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
                seq_log_prob += rf.where(
                    (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
                    _target_dense_extend_blank(
                        lm_log_probs,
                        target_dim=model.target_dim,
                        wb_target_dim=model.wb_target_dim,
                        blank_idx=model.blank_idx,
                        value=0.0,
                    ),
                    0.0,
                )  # Batch, InBeam, VocabWB
            if use_prior and not horizontal_prior:
                # Subtract prior score. If prev align label (target_wb) is blank or != cur, add prior score, otherwise 0.
                seq_log_prob -= rf.where(
                    (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
                    log_prior,
                    0.0,
                )  # Batch, InBeam, VocabWB
            
        if use_recombination and not recomb_after_topk:
            seq_hash = recombination.update_seq_hash(seq_hash, rf.range_over_dim(model.wb_target_dim), backrefs, target_wb, model.blank_idx)
            if t > 0:
                seq_log_prob = recombination.recombine_seqs(
                    seq_log_prob,
                    seq_hash,
                    beam_dim,
                    batch_dims[0],
                    model.wb_target_dim,
                    model.blank_idx,
                    recomb_blank=recomb_blank,
                    use_sum=recomb_with_sum,
                )
            
        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=rf.Dim(top_k, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        target_wb = rf.cast(target_wb, "int32")

        if use_lm:
            lm_log_probs = rf.gather(lm_log_probs, indices=backrefs)  # Batch, Beam, Vocab
        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB
        got_new_label = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _update_context(
                prev_target,
                _target_remove_blank(
                    target_wb, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
                ),
                context_dim
            ),
            prev_target,
        )  # Batch, Beam -> Vocab
        
        if use_recombination and recomb_after_topk:
            seq_hash = recombination.update_seq_hash(seq_hash, target_wb, backrefs, prev_target_wb, model.blank_idx, gather_old_target=False)
            if t > 0:
                seq_log_prob = recombination.recombine_seqs(
                    seq_log_prob,
                    seq_hash,
                    beam_dim,
                    batch_dims[0],
                    None,
                    model.blank_idx,
                    recomb_blank=recomb_blank,
                    use_sum=recomb_with_sum,
                    is_blank=(target_wb == model.blank_idx),
                )

        if use_lm:
            with torch.no_grad():
                got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
                if got_new_label_cpu.raw_tensor.sum().item() > 0:
                    target_, packed_new_label_dim, packed_new_label_dim_map = rf.nested.masked_select_nested(
                        target,
                        mask=got_new_label,
                        mask_cpu=got_new_label_cpu,
                        dims=batch_dims + [beam_dim],
                    )
                    # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
                    assert packed_new_label_dim.get_dim_value() > 0
                    
                    indices_ = []
                    for i in range(context_size):
                        indices_.append(target_.raw_tensor[..., i])
                    lm_logits_ = log_lm_probs[*indices_]
                    assert lm_logits_.size(-1) == int(model.target_dim.get_dim_value())
                    lm_logits_ = rf.convert_to_tensor(lm_logits_, dims=[packed_new_label_dim, model.target_dim], dtype="float32", device=device, name="lm_logits_")
                    assert lm_logits_.dims == (packed_new_label_dim, model.target_dim)
                    lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
                    lm_log_probs_ *= lm_scale

                    lm_log_probs = rf.nested.masked_scatter_nested(
                        lm_log_probs_,
                        lm_log_probs,
                        mask=got_new_label,
                        mask_cpu=got_new_label_cpu,
                        dims=batch_dims + [beam_dim],
                        in_dim=packed_new_label_dim,
                        masked_select_dim_map=packed_new_label_dim_map,
                    )  # Batch, Beam, Vocab / ...

        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    
    if use_lm:
        # seq_log_prob, lm_log_probs: Batch, Beam
        # Add LM EOS score at the end.
        lm_eos_score = rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)
        seq_log_prob += lm_eos_score  # Batch, Beam -> VocabWB
    seq_log_prob = seq_log_prob.raw_tensor
    
    sum_score = safe_logsumexp(seq_log_prob, dim=-1)
    
    loss = -sum_score
    if old_device != device:
        loss = loss.to(old_device)
    
    return loss

def sum_loss_ffnn(
    *,
    model: Model,
    log_probs: rf.Tensor, # (T, B, V)
    lm: FeedForwardLm | None,
    context_size: int,
    log_prior: rf.Tensor | None, # (V,)
    input_lengths: rf.Dim, # (B,)
    top_k: int = 10,
    am_scale: float,
    lm_scale: float,
    prior_scale: float,
    horizontal_prior: bool,
    blank_prior: bool,
    log_zero: float = float("-inf"),
    device: str = "cpu",
    use_recombination: bool = True,
    recomb_blank: bool = True,
    recomb_after_topk: bool = True,
    recomb_with_sum: bool = False,
    blank_correction_version: int = 0,
    print_best_path_for_idx: list[int] = [],
):
    if top_k == 0:
        return sum_loss_ffnn_exact(
            model=model,
            log_probs=log_probs,
            lm=lm,
            context_size=context_size,
            log_prior=log_prior,
            input_lengths=input_lengths,
            am_scale=am_scale,
            lm_scale=lm_scale,
            prior_scale=prior_scale,
            horizontal_prior=horizontal_prior,
            blank_prior=blank_prior,
            log_zero=log_zero,
            device=device,
        )
    
    assert top_k > 0, "Top k should be greater than 0 as exact sum is not implemented for FFNN"
    assert blank_correction_version == 0, "Blank correction is not implemented for FFNN"
    
    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    import returnn
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__
    
    def _update_context(context: rf.Tensor, new_label: rf.Tensor, context_dim: rf.Dim) -> rf.Tensor:
        new_dim = rf.Dim(1, name="new_label")
        new_label = rf.expand_dim(new_label, dim=new_dim)
        old_context, old_context_dim = rf.slice(context, axis=context_dim, start=1)
        new_context, new_context_dim = rf.concat((old_context, old_context_dim), (new_label, new_dim), out_dim=context_dim)
        assert new_context_dim == context_dim
        return new_context
    
    def _target_remove_blank(target: rf.Tensor, *, target_dim: rf.Dim, wb_target_dim: rf.Dim, blank_idx: int) -> rf.Tensor:
        assert target.sparse_dim == wb_target_dim
        assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
        return rf.set_sparse_dim(target, target_dim)

    def _target_dense_extend_blank(
        target: rf.Tensor, *, target_dim: rf.Dim, wb_target_dim: rf.Dim, blank_idx: int, value: float
    ) -> rf.Tensor:
        assert target_dim in target.dims
        assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
        res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
        return res
    
    use_prior = log_prior is not None
    use_lm = lm is not None
    
    old_device = log_probs.device
    log_probs = rf.copy_to_device(log_probs, device)
    if use_prior:
        if not blank_prior and model.target_dim in log_prior.dims:
            new_dim = rf.Dim(1)
            log_prior = rf.concat(
                [(log_prior, model.target_dim),(rf.zeros(dims = [new_dim],  dtype="float32", device=log_prior.device), new_dim)],
                out_dim=model.wb_target_dim
            )
        assert model.wb_target_dim in log_prior.dims
        log_prior = rf.copy_to_device(log_prior, device)
    
    batch_dims = [batch_dim]
    beam_dim = rf.Dim(1, name="initial-beam")
    context_dim = rf.Dim(context_size, name="context")
    lm_out_dim = rf.Dim(context_size + 1, name="context+1")
    batch_dims_ = batch_dims + [beam_dim]
    seq_log_prob = rf.constant(0.0, dims=batch_dims_) # Batch, Beam
    
    # scaled log am and prior probs
    log_probs = am_scale * log_probs
    if use_prior:
        log_prior = prior_scale * log_prior
        
        # If not blank prior this is still applied as the log prior for blank is just 0
        if horizontal_prior:
            log_probs -= log_prior
        
    log_probs = rf.where(
        input_lengths.get_mask(),
        log_probs,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    log_probs_ta = TensorArray.unstack(log_probs, axis=input_lengths)  # t -> Batch, VocabWB
    
    target = rf.constant(model.bos_idx, dims=batch_dims_ + [context_dim], sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB
    
    # Prepare LM
    if use_lm:
        with torch.no_grad():
            lm_state = lm.default_initial_state(batch_dims=[])
            lm_logits, lm_state = get_lm_logits(batch_dims, target, lm, context_dim, lm_out_dim, lm_state)
            lm_logits = rf.gather(lm_logits, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
            assert lm_logits.dims == (*batch_dims_, model.target_dim)
            lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
            lm_log_probs *= lm_scale
    
    max_seq_len = int(input_lengths.get_dim_value())
    backrefs = None
    if use_recombination:
        assert len(batch_dims) == 1
        if recomb_after_topk:
            seq_hash = rf.constant(0, dims=batch_dims_, dtype="int64")
        else:
            seq_hash = rf.constant(0, dims=batch_dims_ + [model.wb_target_dim], dtype="int64")
    
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        seq_log_prob = seq_log_prob + log_probs_ta[t]  # Batch, InBeam, VocabWB

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if use_lm:
                # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
                seq_log_prob += rf.where(
                    (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
                    _target_dense_extend_blank(
                        lm_log_probs,
                        target_dim=model.target_dim,
                        wb_target_dim=model.wb_target_dim,
                        blank_idx=model.blank_idx,
                        value=0.0,
                    ),
                    0.0,
                )  # Batch, InBeam, VocabWB
            if use_prior and not horizontal_prior:
                # Subtract prior score. If prev align label (target_wb) is blank or != cur, add prior score, otherwise 0.
                seq_log_prob -= rf.where(
                    (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
                    log_prior,
                    0.0,
                )  # Batch, InBeam, VocabWB
            
        if use_recombination and not recomb_after_topk:
            seq_hash = recombination.update_seq_hash(seq_hash, rf.range_over_dim(model.wb_target_dim), backrefs, target_wb, model.blank_idx)
            if t > 0:
                seq_log_prob = recombination.recombine_seqs(
                    seq_log_prob,
                    seq_hash,
                    beam_dim,
                    batch_dims[0],
                    model.wb_target_dim,
                    model.blank_idx,
                    recomb_blank=recomb_blank,
                    use_sum=recomb_with_sum,
                )
            
        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=rf.Dim(top_k, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        target_wb = rf.cast(target_wb, "int32")

        if use_lm:
            lm_log_probs = rf.gather(lm_log_probs, indices=backrefs)  # Batch, Beam, Vocab
            lm_state = rf.nested.gather_nested(lm_state, indices=backrefs)
        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB
        got_new_label = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _update_context(
                prev_target,
                _target_remove_blank(
                    target_wb, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
                ),
                context_dim
            ),
            prev_target,
        )  # Batch, Beam -> Vocab
        
        if use_recombination and recomb_after_topk:
            seq_hash = recombination.update_seq_hash(seq_hash, target_wb, backrefs, prev_target_wb, model.blank_idx, gather_old_target=False)
            if t > 0:
                seq_log_prob = recombination.recombine_seqs(
                    seq_log_prob,
                    seq_hash,
                    beam_dim,
                    batch_dims[0],
                    None,
                    model.blank_idx,
                    recomb_blank=recomb_blank,
                    use_sum=recomb_with_sum,
                    is_blank=(target_wb == model.blank_idx),
                )

        if use_lm:
            with torch.no_grad():
                got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
                if got_new_label_cpu.raw_tensor.sum().item() > 0:
                    (target_, lm_state_), packed_new_label_dim, packed_new_label_dim_map = rf.nested.masked_select_nested(
                        (target, lm_state),
                        mask=got_new_label,
                        mask_cpu=got_new_label_cpu,
                        dims=batch_dims + [beam_dim],
                    )
                    # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
                    assert packed_new_label_dim.get_dim_value() > 0
                    
                    lm_logits_, lm_state_ = get_lm_logits([packed_new_label_dim], target_, lm, context_dim, lm_out_dim, lm_state_)
                    lm_logits_ = rf.gather(lm_logits_, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
                    assert lm_logits_.dims == (packed_new_label_dim, model.target_dim)
                    lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
                    lm_log_probs_ *= lm_scale

                    lm_log_probs, lm_state = rf.nested.masked_scatter_nested(
                        (lm_log_probs_, lm_state_),
                        (lm_log_probs, lm_state),
                        mask=got_new_label,
                        mask_cpu=got_new_label_cpu,
                        dims=batch_dims + [beam_dim],
                        in_dim=packed_new_label_dim,
                        masked_select_dim_map=packed_new_label_dim_map,
                    )  # Batch, Beam, Vocab / ...

        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    
    if use_lm:
        # seq_log_prob, lm_log_probs: Batch, Beam
        # Add LM EOS score at the end.
        lm_eos_score = rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)
        seq_log_prob += lm_eos_score  # Batch, Beam -> VocabWB
    seq_log_prob = seq_log_prob.raw_tensor
    
    sum_score = safe_logsumexp(seq_log_prob, dim=-1)
    
    loss = -sum_score
    if old_device != device:
        loss = loss.to(old_device)
    
    return loss

def sum_loss_ffnn_exact(
    *,
    model: Model,
    log_probs: rf.Tensor, # (T, B, V)
    lm: FeedForwardLm | None,
    context_size: int,
    log_prior: rf.Tensor | None, # (V,)
    input_lengths: rf.Dim, # (B,)
    am_scale: float,
    lm_scale: float,
    prior_scale: float,
    horizontal_prior: bool,
    blank_prior: bool,
    log_zero: float = float("-inf"),
    device: str = "cpu",
):
    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    import returnn
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__
    
    def _update_context(context: rf.Tensor, new_label: rf.Tensor, context_dim: rf.Dim) -> rf.Tensor:
        new_dim = rf.Dim(1, name="new_label")
        new_label = rf.expand_dim(new_label, dim=new_dim)
        old_context, old_context_dim = rf.slice(context, axis=context_dim, start=1)
        new_context, new_context_dim = rf.concat((old_context, old_context_dim), (new_label, new_dim), allow_broadcast=True, out_dim=context_dim)
        assert new_context_dim == context_dim
        return new_context
    
    def _target_remove_blank(target: rf.Tensor, *, target_dim: rf.Dim, wb_target_dim: rf.Dim, blank_idx: int) -> rf.Tensor:
        assert target.sparse_dim == wb_target_dim
        assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
        return rf.set_sparse_dim(target, target_dim)

    def _target_dense_extend_blank(
        target: rf.Tensor, *, target_dim: rf.Dim, wb_target_dim: rf.Dim, blank_idx: int, value: float
    ) -> rf.Tensor:
        assert target_dim in target.dims
        assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
        res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
        return res
    
    use_prior = log_prior is not None
    use_lm = lm is not None
    
    old_device = log_probs.device
    log_probs = rf.copy_to_device(log_probs, device)
    if use_prior:
        if not blank_prior and model.target_dim in log_prior.dims:
            new_dim = rf.Dim(1)
            log_prior = rf.concat(
                [(log_prior, model.target_dim),(rf.zeros(dims = [new_dim],  dtype="float32", device=log_prior.device), new_dim)],
                out_dim=model.wb_target_dim
            )
        assert model.wb_target_dim in log_prior.dims
        log_prior = rf.copy_to_device(log_prior, device)
    
    batch_dims = [batch_dim]
    beam_dim = rf.Dim(1, name=f"initial-beam")
    context_dim = rf.Dim(context_size, name="context")
    lm_out_dim = rf.Dim(context_size + 1, name="context+1")
    batch_dims_ = batch_dims + [beam_dim]
    seq_log_prob = rf.constant(0.0, dims=batch_dims_) # Batch, Beam
    
    # scaled log am and prior probs
    log_probs = am_scale * log_probs
    if use_prior:
        log_prior = prior_scale * log_prior
        
        # If not blank prior this is still applied as the log prior for blank is just 0
        if horizontal_prior:
            log_probs -= log_prior
        
    log_probs = rf.where(
        input_lengths.get_mask(),
        log_probs,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    log_probs_ta = TensorArray.unstack(log_probs, axis=input_lengths)  # t -> Batch, VocabWB
    
    target = rf.constant(model.bos_idx, dims=batch_dims_ + [context_dim], sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB
    
    # Prepare LM
    if use_lm:
        with torch.no_grad():
            lm_state = lm.default_initial_state(batch_dims=[])
            lm_logits, lm_state = get_lm_logits(batch_dims, target, lm, context_dim, lm_out_dim, lm_state)
            lm_logits = rf.gather(lm_logits, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
            assert lm_logits.dims == (*batch_dims_, model.target_dim)
            lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
            lm_log_probs *= lm_scale
    
    max_seq_len = int(input_lengths.get_dim_value())
    assert len(batch_dims) == 1
    
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb
        prev_beam = beam_dim

        seq_log_prob = seq_log_prob + log_probs_ta[t]  # Batch, InBeam, VocabWB

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if use_lm:
                # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
                seq_log_prob += rf.where(
                    (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
                    _target_dense_extend_blank(
                        lm_log_probs,
                        target_dim=model.target_dim,
                        wb_target_dim=model.wb_target_dim,
                        blank_idx=model.blank_idx,
                        value=0.0,
                    ),
                    0.0,
                )  # Batch, InBeam, VocabWB
            if use_prior and not horizontal_prior:
                # Subtract prior score. If prev align label (target_wb) is blank or != cur, add prior score, otherwise 0.
                seq_log_prob -= rf.where(
                    (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
                    log_prior,
                    0.0,
                )  # Batch, InBeam, VocabWB
            
        beam_dim = rf.Dim(int(model.wb_target_dim.get_dim_value()) ** (t + 1 if t < context_size else context_size), name=f"dec-step{t}-beam")
        if t <= context_size:
            target_wb_tmp = rf.range_over_dim(model.wb_target_dim)
            target_wb_tmp = rf.expand_dims(target_wb_tmp, dims=batch_dims + [prev_beam])
            target_wb_tmp = rf.cast(target_wb_tmp, "int32")
            got_new_label = (target_wb_tmp != model.blank_idx) & (target_wb_tmp != prev_target_wb)  # Batch, Beam -> 0|1
            target = rf.where(
                got_new_label,
                _update_context(
                    prev_target,
                    _target_remove_blank(
                        target_wb_tmp, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
                    ),
                    context_dim
                ),
                prev_target,
            )  # Batch, Beam -> Vocab
            target_raveled = torch.tensor(
                np.array(
                    [
                        [
                            [
                                np.ravel_multi_index(target.raw_tensor[b, k, :, v].cpu().numpy(), (model.target_dim.dimension,) * context_size)
                                for v in range(target.dims[-1].get_dim_value())
                            ]
                            for k in range(prev_beam.get_dim_value())
                        ]
                        for b in range(batch_dim.get_dim_value())
                    ]
                ),
                device=device
            )
            print(target_raveled.shape)
            target_raveled = target_raveled.view(batch_dim.get_dim_value(), -1)
            
            if t < context_size:
                target_wb, _ = rf.merge_dims(target_wb_tmp, dims=[prev_beam, model.wb_target_dim], out_dim=beam_dim)
            print(target_wb)
        
            target, _ = rf.merge_dims(target, dims = [prev_beam, model.wb_target_dim], out_dim = beam_dim)
            got_new_label, _ = rf.merge_dims(got_new_label, dims = [prev_beam, model.wb_target_dim], out_dim = beam_dim)

        print(target.raw_tensor.shape, seq_log_prob.raw_tensor.shape)
        
        seq_log_prob = seq_log_prob.raw_tensor.view(batch_dim.get_dim_value(), -1)
        
        # do a scatter logsumexp here together with the new target labels
        new_Log_probs = torch.zeros((batch_dim.get_dim_value(), beam_dim.get_dim_value()), dtype=torch.float32, device=device)
        print(new_Log_probs.shape)
        
        seq_log_prob = scatter_safe_logsumexp(new_Log_probs, -1, target_raveled, seq_log_prob, include_self=False)
        seq_log_prob = rf.convert_to_tensor(seq_log_prob, dims=batch_dims + [beam_dim], dtype="float32", device=device, name="seq_log_prob")
        print(seq_log_prob)
        
        print(got_new_label)
        print(target)

        if use_lm and t < context_size:
            print(lm_log_probs)
            lm_log_probs = rf.expand_dim(lm_log_probs, dim=model.wb_target_dim)
            lm_log_probs, _ = rf.merge_dims(lm_log_probs, dims = [prev_beam, model.wb_target_dim], out_dim = beam_dim)
            print(lm_log_probs, lm_log_probs.raw_tensor.shape)
            
            with torch.no_grad():
                got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
                if got_new_label_cpu.raw_tensor.sum().item() > 0:
                    (target_, lm_state_), packed_new_label_dim, packed_new_label_dim_map = rf.nested.masked_select_nested(
                        (target, lm_state),
                        mask=got_new_label,
                        mask_cpu=got_new_label_cpu,
                        dims=got_new_label.dims,
                    )
                    # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
                    assert packed_new_label_dim.get_dim_value() > 0
                    
                    lm_logits_, lm_state_ = get_lm_logits([packed_new_label_dim], target_, lm, context_dim, lm_out_dim, lm_state_)
                    lm_logits_ = rf.gather(lm_logits_, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
                    assert lm_logits_.dims == (packed_new_label_dim, model.target_dim)
                    lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
                    lm_log_probs_ *= lm_scale

                    lm_log_probs, lm_state = rf.nested.masked_scatter_nested(
                        (lm_log_probs_, lm_state_),
                        (lm_log_probs, lm_state),
                        mask=got_new_label,
                        mask_cpu=got_new_label_cpu,
                        dims=got_new_label.dims,
                        in_dim=packed_new_label_dim,
                        masked_select_dim_map=packed_new_label_dim_map,
                    )  # Batch, Beam, Vocab / ...

        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    
    if use_lm:
        # seq_log_prob, lm_log_probs: Batch, Beam
        # Add LM EOS score at the end.
        lm_eos_score = rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)
        seq_log_prob += lm_eos_score  # Batch, Beam -> VocabWB
    seq_log_prob = seq_log_prob.raw_tensor
    
    sum_score = safe_logsumexp(seq_log_prob, dim=-1)
    
    loss = -sum_score
    if old_device != device:
        loss = loss.to(old_device)
    
    return loss


def sum_loss_approx(
    *,
    log_probs: torch.Tensor, # (T, B, V)
    log_lm_probs: torch.Tensor, # (V, V)
    log_prior: torch.Tensor | None, # (V,)
    input_lengths: torch.Tensor, # (B,)
    top_k: int = 0,
    LM_order: int,
    am_scale: float,
    lm_scale: float,
    prior_scale: float,
    horizontal_prior: bool,
    blank_prior: bool = True,
    blank_idx:int = 0, # should be same as eos index
    eos_idx: int | None = None,
    unk_idx: int = 1,
    log_zero: float = float("-inf"),
    device: torch.device = torch.device("cpu"),
    print_best_path_for_idx: list[int] = [],
):
    """Testing purposes only
    """
    use_prior = log_prior is not None
    
    old_device = log_probs.device
    log_probs = log_probs.to(device)
    log_lm_probs = log_lm_probs.to(device)
    if use_prior:
        log_prior = log_prior.to(device)
    input_lengths = input_lengths.to(device)
    
    max_audio_time, batch_size, n_out = log_probs.shape
    # scaled log am and lm probs
    log_probs = am_scale * log_probs
    if lm_scale == 0.0:
        log_lm_probs = torch.zeros_like(log_lm_probs, device=device)
    else:
        log_lm_probs = lm_scale * log_lm_probs
    if use_prior:
        log_prior = prior_scale * log_prior
    
    # print_gradients = PrintGradients.apply
    # grad_assert = AssertGradients.apply

    # Unbind the log_probs and log_partial_empty_seq_prob for each timestep so it is faster during backprop
    log_probs = log_probs.unbind(0)

    # sum score by DP
    
    assert blank_idx == n_out - 1, "blank should be the last symbol" + str(blank_idx) + " " + str(n_out)
    assert log_prior.size(0) == n_out, "Prior shape is not correct"
    
    if print_best_path_for_idx:
        with torch.no_grad():
            best_path_print = {}
            for idx in print_best_path_for_idx:
                best_path_print[idx] = {"str": "", "am_str": "", "prior": "", "score": "", "AM": torch.full((1,), 0.0, device=device)}
    
    
    log_q = torch.zeros((batch_size,), device=device)
    for t in range(max_audio_time):
        if use_prior:
            tmp = log_probs[t] - log_prior.unsqueeze(0)
            max_val, max_idx = torch.max(tmp, dim=-1)
        else:
            max_val, max_idx = torch.max(log_probs[t], dim=-1)
        time_mask = (t < input_lengths)
        
        new_log_q = log_q + max_val
        log_q = torch.where(time_mask, new_log_q, log_q)
        
        if print_best_path_for_idx:
            with torch.no_grad():
                for idx in print_best_path_for_idx:
                    best_path_print[idx]["str"] += f" {max_idx[idx]}"
                    best_path_print[idx]["am_str"] += " {:.2f}".format(log_probs[t][idx][max_idx[idx]].tolist())
                    best_path_print[idx]["prior"] += " {:.2f}".format(log_prior[max_idx[idx]].tolist())
                    best_path_print[idx]["score"] += " {:.2f}".format(log_q[idx].tolist()) #  / (t+1)
                    best_path_print[idx]["AM"] += log_probs[t][idx][max_idx[idx]]
    
    if print_best_path_for_idx:
        with torch.no_grad():
            for idx in print_best_path_for_idx:
                print(f"Best path for {idx}: {best_path_print[idx]['str']}\nAM str: {best_path_print[idx]['am_str']}\nPrior: {best_path_print[idx]['prior']}\nScore: {best_path_print[idx]['score']}\nAM: {best_path_print[idx]['AM']}")
    
    loss = -log_q
    if old_device != device:
        loss = loss.to(old_device)
    
    return loss

# ------------------------------------------------
# Helper functions and classes

def get_lm_logits(batch_dims: list[rf.Dim], target: rf.Tensor, lm: FeedForwardLm, context_dim: rf.Dim, lm_out_dim: rf.Dim, lm_state):
    lm_logits = None
    done = False
    splits = 1
    while not done:
        try:
            if splits > 1:
                batch_size = batch_dims[0].dyn_size_ext.raw_tensor.item()
                n_seqs = int(np.ceil(batch_size / splits))
                new_dims = []
                for i in range(splits):
                    if (i + 1) * n_seqs <= batch_size:
                        new_dims.append(rf.Dim(n_seqs, name=f"split-{i}"))
                    else:
                        new_dims.append(rf.Dim(batch_size - i * n_seqs, name=f"split-{i}"))
                target_split = rf.split(target, axis=batch_dims[0], out_dims=new_dims)
                lm_logits_split = []
                for i in range(splits):
                    lm_logits_i, _ = lm(
                        target_split[i],
                        spatial_dim=context_dim,
                        out_spatial_dim=lm_out_dim,
                        state=lm_state,
                    )
                    lm_logits_split.append((lm_logits_i, new_dims[i]))
                lm_logits = rf.concat(lm_logits_split, out_dim=batch_dims[0])
            else:
                lm_logits, lm_state = lm(
                    target,
                    spatial_dim=context_dim,
                    out_spatial_dim=lm_out_dim,
                    state=lm_state,
                )
            done = True
        except Exception as exc:
            print(f"OOM with {splits} splits")
            diagnose_gpu.garbage_collect()
            splits *= 2
            if splits <= batch_dims[0].dyn_size_ext.raw_tensor.item():
                continue
            else:
                raise
    return lm_logits, lm_state

def dynamic_slice(tensor: torch.Tensor, indices_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Slices a tensor along multiple dimensions using the provided indices for each dimension.
    
    Args:
    tensor (torch.Tensor): The input tensor to be sliced.
    indices_list (list of list of int): A list containing tensors of indices for each dimension.
    
    Returns:
    torch.Tensor: The sliced tensor.
    """
    assert len(indices_list) == tensor.dim(), "Number of index lists must match the number of dimensions of the tensor"
    
    # Use advanced indexing to select the desired elements
    sliced_tensor = tensor
    for dim, indices in enumerate(indices_list):
        # Expand indices to match the dimensions of the tensor
        shape = [1] * tensor.dim()
        shape[dim] = -1
        new_sizes = list(sliced_tensor.size())
        new_sizes[dim] = -1
        indices = indices.view(shape).expand(new_sizes)
        sliced_tensor = torch.gather(sliced_tensor, dim, indices)
    
    return sliced_tensor

# def log_matmul_alt(A: torch.Tensor, B: torch.Tensor):
#     with torch.no_grad():
#         max_A, _ = torch.max(A, dim=1, keepdim=True)
#         max_B, _ = torch.max(B, dim=0, keepdim=True)
#         mask_A = max_A.isneginf()
#         mask_B = max_B.isneginf()
#         mask = mask_A + mask_B > 0
#     m1 = (A - max_A.masked_fill(mask_A, 0.0)).exp()
#     m2 = (B - max_B.masked_fill(mask_B, 0.0)).exp()
#     mul = m1.matmul(m2)
#     mul_masked = mul.masked_fill_(mask, 1)
#     log_mul = mul_masked.masked_fill_(mul_masked == 0.0, 0.0).log()
#     return max_A + max_B + log_mul

def log_matmul(A: torch.Tensor, B: torch.Tensor, batch_given: bool = False):
    """
    This is inefficient

    Log matrix multiplication, i.e.
    A = log X, B = log Y
    -> log_matmul(A, B) = log (X @ Y)
    https://stackoverflow.com/questions/36467022/handling-matrix-multiplication-in-log-space-in-python

    :param A: first matrix in log scale (b, v)
    :param B: second matrix in log scale (v, v2)
    :param batch_given: whether we habe the batch dimension in B given
    :returns: matrix product of the two matrices in log scale
    """
    if batch_given:
        B_expand = B # (b, v, v2)
    else:
        b = A.size(0)
        B_expand = B.unsqueeze(0).expand(b, *B.size()) # (v, v2) -> (b, v, v2)
    A_expand = A.unsqueeze(-1).expand_as(B_expand) # (b, v) -> (b, v, v2)
    
    return safe_logsumexp((A_expand + B_expand), dim=1) # (b, v, v2) -> (b, v2)

def get_bpe_from_dict(idx: int):
    d = {0: '<s>', 1: '<unk>', 2: 'T@@', 3: 'THE', 4: 'C@@', 5: 'E@@', 6: 'M@@', 7: 'P@@', 8: 'I@@', 9: 'W@@', 10: 'S@@', 11: 'A@@', 12: 'D@@', 13: 'F@@', 14: 'G@@', 15: 'U@@', 16: 'ED', 17: 'O@@', 18: 'S', 19: 'E', 20: 'AND', 21: 'L@@', 22: 'Y', 23: 'OF', 24: 'TO', 25: 'IN@@', 26: 'RE@@', 27: 'TH@@', 28: 'B@@', 29: 'AR@@', 30: 'ING', 31: 'A', 32: 'T', 33: 'ER@@', 34: 'R@@', 35: 'AN@@', 36: 'H@@', 37: 'ST@@', 38: 'IN', 39: 'OU@@', 40: 'V@@', 41: 'D', 42: 'ON', 43: 'N@@', 44: 'K@@', 45: 'Y@@', 46: 'EN', 47: 'OR@@', 48: 'ER', 49: 'EL@@', 50: 'L', 51: 'EN@@', 52: 'ON@@', 53: 'RO@@', 54: 'ES', 55: 'IT@@', 56: 'I', 57: 'M', 58: 'R', 59: 'WAS', 60: 'HE', 61: 'ME', 62: 'AT@@', 63: 'LY', 64: 'IT', 65: 'THAT', 66: 'O', 67: 'AL@@', 68: 'AC@@', 69: 'HA@@', 70: 'BE@@', 71: 'AN', 72: 'ST', 73: 'IS', 74: 'H', 75: 'IS@@', 76: 'W', 77: 'LE', 78: 'LE@@', 79: 'K', 80: 'TI@@', 81: 'ERE', 82: 'LI@@', 83: 'HIS', 84: 'RI@@', 85: 'SI@@', 86: 'WH@@', 87: 'UR@@', 88: 'LO@@', 89: 'SE', 90: 'AT', 91: 'AS', 92: 'SA@@', 93: 'CH', 94: 'CO@@', 95: 'HAD', 96: 'THE@@', 97: 'WITH', 98: 'SE@@', 99: 'IL@@', 100: 'UN@@', 101: 'YOU', 102: 'CE', 103: 'FOR', 104: 'F', 105: 'NE@@', 106: 'AS@@', 107: 'DI@@', 108: 'HER', 109: 'DE@@', 110: 'SU@@', 111: 'N', 112: 'MA@@', 113: 'NO@@', 114: 'NOT', 115: 'LA@@', 116: 'HO@@', 117: 'BUT', 118: 'ENT', 119: 'CA@@', 120: 'OR', 121: 'OULD', 122: 'RA@@', 123: 'GHT', 124: 'WHI@@', 125: 'PO@@', 126: 'VE', 127: 'P', 128: 'J@@', 129: 'VER@@', 130: 'SHE', 131: 'SO@@', 132: 'ONE', 133: 'IR@@', 134: 'AB@@', 135: 'THER', 136: 'X@@', 137: 'BE', 138: 'OUN@@', 139: 'HE@@', 140: 'ALL', 141: 'CON@@', 142: 'HI@@', 143: 'PE@@', 144: "'S", 145: 'OUT', 146: 'HIM', 147: 'MO@@', 148: 'FOR@@', 149: 'ID', 150: 'VER', 151: 'DO@@', 152: 'TO@@', 153: 'MY', 154: "'@@", 155: 'ME@@', 156: 'THEY', 157: 'BY', 158: 'SS', 159: 'ENT@@', 160: 'KE', 161: 'G', 162: 'ATI@@', 163: 'WA@@', 164: 'HAVE', 165: 'MP@@', 166: 'AL', 167: 'SO', 168: 'Q@@', 169: 'LD', 170: 'GH@@', 171: 'Z@@', 172: 'BU@@', 173: 'C', 174: 'X', 175: 'B', 176: 'OU', 177: 'WIT@@', 178: 'U', 179: 'Z', 180: 'V', 181: 'Q', 182: 'J', 183: "'", 184: "<blank>"}
    return d[idx]

def get_bpes(tokens):
    # MONICA DREW FRESH HOPE FROM HER SON'S WRITINGS THEY WERE FULL OF NOBLE THOUGHTS AND HIGH ASPIRATIONS
    
    # path = "/u/marten.mueller/dev/ctc_baseline/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.P1DXd9G7EdsU/output/bpe.vocab"
    # d = eval(open(path, "r").read(), {"nan": float("nan"), "inf": float("inf")})
    # d = {v: k for k, v in d.items()}
    d = {0: '<s>', 1: '<unk>', 2: 'T@@', 3: 'THE', 4: 'C@@', 5: 'E@@', 6: 'M@@', 7: 'P@@', 8: 'I@@', 9: 'W@@', 10: 'S@@', 11: 'A@@', 12: 'D@@', 13: 'F@@', 14: 'G@@', 15: 'U@@', 16: 'ED', 17: 'O@@', 18: 'S', 19: 'E', 20: 'AND', 21: 'L@@', 22: 'Y', 23: 'OF', 24: 'TO', 25: 'IN@@', 26: 'RE@@', 27: 'TH@@', 28: 'B@@', 29: 'AR@@', 30: 'ING', 31: 'A', 32: 'T', 33: 'ER@@', 34: 'R@@', 35: 'AN@@', 36: 'H@@', 37: 'ST@@', 38: 'IN', 39: 'OU@@', 40: 'V@@', 41: 'D', 42: 'ON', 43: 'N@@', 44: 'K@@', 45: 'Y@@', 46: 'EN', 47: 'OR@@', 48: 'ER', 49: 'EL@@', 50: 'L', 51: 'EN@@', 52: 'ON@@', 53: 'RO@@', 54: 'ES', 55: 'IT@@', 56: 'I', 57: 'M', 58: 'R', 59: 'WAS', 60: 'HE', 61: 'ME', 62: 'AT@@', 63: 'LY', 64: 'IT', 65: 'THAT', 66: 'O', 67: 'AL@@', 68: 'AC@@', 69: 'HA@@', 70: 'BE@@', 71: 'AN', 72: 'ST', 73: 'IS', 74: 'H', 75: 'IS@@', 76: 'W', 77: 'LE', 78: 'LE@@', 79: 'K', 80: 'TI@@', 81: 'ERE', 82: 'LI@@', 83: 'HIS', 84: 'RI@@', 85: 'SI@@', 86: 'WH@@', 87: 'UR@@', 88: 'LO@@', 89: 'SE', 90: 'AT', 91: 'AS', 92: 'SA@@', 93: 'CH', 94: 'CO@@', 95: 'HAD', 96: 'THE@@', 97: 'WITH', 98: 'SE@@', 99: 'IL@@', 100: 'UN@@', 101: 'YOU', 102: 'CE', 103: 'FOR', 104: 'F', 105: 'NE@@', 106: 'AS@@', 107: 'DI@@', 108: 'HER', 109: 'DE@@', 110: 'SU@@', 111: 'N', 112: 'MA@@', 113: 'NO@@', 114: 'NOT', 115: 'LA@@', 116: 'HO@@', 117: 'BUT', 118: 'ENT', 119: 'CA@@', 120: 'OR', 121: 'OULD', 122: 'RA@@', 123: 'GHT', 124: 'WHI@@', 125: 'PO@@', 126: 'VE', 127: 'P', 128: 'J@@', 129: 'VER@@', 130: 'SHE', 131: 'SO@@', 132: 'ONE', 133: 'IR@@', 134: 'AB@@', 135: 'THER', 136: 'X@@', 137: 'BE', 138: 'OUN@@', 139: 'HE@@', 140: 'ALL', 141: 'CON@@', 142: 'HI@@', 143: 'PE@@', 144: "'S", 145: 'OUT', 146: 'HIM', 147: 'MO@@', 148: 'FOR@@', 149: 'ID', 150: 'VER', 151: 'DO@@', 152: 'TO@@', 153: 'MY', 154: "'@@", 155: 'ME@@', 156: 'THEY', 157: 'BY', 158: 'SS', 159: 'ENT@@', 160: 'KE', 161: 'G', 162: 'ATI@@', 163: 'WA@@', 164: 'HAVE', 165: 'MP@@', 166: 'AL', 167: 'SO', 168: 'Q@@', 169: 'LD', 170: 'GH@@', 171: 'Z@@', 172: 'BU@@', 173: 'C', 174: 'X', 175: 'B', 176: 'OU', 177: 'WIT@@', 178: 'U', 179: 'Z', 180: 'V', 181: 'Q', 182: 'J', 183: "'"}
    
    tokens = [t for t in tokens.split("] [")]
    tokens[0] = tokens[0][1:]
    if len(tokens) > 1:
        tokens[-1] = tokens[-1][:-1]
    if "," in tokens[0]:
        tokens = [t.split(", ")[-1] for t in tokens]
    tokens = [d[int(t)] for t in tokens]
    tokens = " ".join(tokens)
    tokens = tokens.replace("@@ ", "")
    return tokens

def plot_gradients(gradients: np.ndarray, savename: str, title: str = "Text", all_timesteps=False, forwards: np.ndarray = None, norm: str = None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from datetime import datetime
    if all_timesteps:
        if forwards is not None:
            if "logits" in savename:
                fig, ax = plt.subplots(figsize=(15, 15))
                fig.supylabel("Vocab")
                fig.supxlabel("Timestep")
                ax.imshow(gradients.T, origin="lower", cmap=cm.gray)
                ax.set_yticks(np.arange(0, 185, 10))
                ax.set_title("Gradients " + title[0])
                g_min = -1 * gradients.min()
                g_max = -1 * gradients.max()
                ax.text(2, -10, f'black: {g_min}, white: {g_max}', bbox={'facecolor': 'white', 'pad': 10})
            else:
                fig, axs = plt.subplots(1, 5, figsize=(15 + gradients.shape[0] / 4, 15))
                fig.supylabel("Vocab")
                fig.supxlabel("Timestep")
                axs[0].imshow(gradients.T, origin="lower", cmap=cm.gray)
                axs[0].set_yticks(np.arange(0, 185, 10))
                axs[0].set_title("Gradients " + title[0])
                g_min = -1 * gradients.min()
                g_max = gradients.max()
                if g_max != 0.0:
                    g_max = -g_max
                axs[0].text(2, -10, f'black: {g_min}, white: {g_max}', bbox={'facecolor': 'white', 'pad': 10})
                log_gr = np.log((-gradients))
                axs[1].imshow(-log_gr.T, origin="lower", cmap=cm.gray)
                axs[1].set_yticks(np.arange(0, 185, 10))
                axs[1].set_title("Log Gradients " + title[0])
                axs[1].text(2, -10, f'black: {log_gr.max()}, white: {log_gr.min()}', bbox={'facecolor': 'white', 'pad': 10})
                # argmax = np.argmax(forwards, axis=1)
                # one_hot = np.eye(forwards.shape[1])[argmax]
                # one_hot = -one_hot
                # axs[2].imshow(one_hot.T, origin="lower", cmap=cm.gray, norm=norm)
                # axs[2].set_yticks(np.arange(0, 185, 10))
                # axs[2].set_title("Argmax Inputs " + title[0])
                forwards_exp = np.exp(forwards)
                f_min = forwards_exp.min()
                f_max = forwards_exp.max()
                axs[2].imshow(-forwards_exp.T, origin="lower", cmap=cm.gray, norm=norm)
                axs[2].set_yticks(np.arange(0, 185, 10))
                axs[2].set_title("Probs " + title[0])
                axs[2].text(2, -10, f'black: {f_max}, white: {f_min}', bbox={'facecolor': 'white', 'pad': 10})
                fw = -1 * forwards
                axs[3].imshow(fw.T, origin="lower", cmap=cm.gray, norm=norm)
                axs[3].set_yticks(np.arange(0, 185, 10))
                axs[3].set_title("Inputs " + title[0])
                axs[3].text(2, -10, f'black: {forwards.max()}, white: {forwards.min()}', bbox={'facecolor': 'white', 'pad': 10})
                diff = forwards_exp + gradients
                axs[4].imshow(diff.T, origin="lower", cmap=cm.gray, norm=norm)
                axs[4].set_yticks(np.arange(0, 185, 10))
                axs[4].set_title("Diff Inputs - Gradients " + title[0])
                axs[4].text(2, -10, f'black: {-diff.min()}, white: {-diff.max()}', bbox={'facecolor': 'white', 'pad': 10})
        else:
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.imshow(gradients, origin="lower", cmap=cm.gray)
            fig.supxlabel("Vocab")
            fig.supylabel("Timestep")
            ax.set_xticks(np.arange(0, 185, 10))
            ax.set_title("Gradients " + title[0])
    elif gradients.ndim == 1:
        notzero = np.where(gradients != 0)[0]
        if len(notzero) > 10:
            notzero = np.argpartition(np.abs(gradients), -10)[-10:]
        fig = plt.figure(figsize=(10, 5))
        plt.plot(gradients, ".-", linewidth=1, markersize=3)
        plt.xticks(np.arange(0, 185, 10))
        plt.title(title)
        plt.xlabel("Vocab")
        plt.ylabel("Gradients")
        for idx in notzero:
            plt.text(idx - 3, gradients[idx], get_bpe_from_dict(idx), rotation = "vertical", fontsize = "x-small")
    elif gradients.ndim == 2:
        fig, axs = plt.subplots(gradients.shape[0], 1, figsize=(10, 5 * gradients.shape[0]))
        fig.supxlabel("Vocab")
        fig.supylabel("Gradients")
        for t in range(gradients.shape[0]):
            notzero = np.where(gradients[t] != 0)[0]
            if len(notzero) > 10:
                notzero = np.argpartition(np.abs(gradients[t]), -10)[-10:]
            axs[t].plot(gradients[t], ".-", linewidth=1, markersize=3)
            axs[t].set_xticks(np.arange(0, 185, 10))
            axs[t].set_title(title[t])
            for idx in notzero:
                axs[t].text(idx - 3, gradients[t][idx], get_bpe_from_dict(idx), rotation = "vertical", fontsize = "x-small")
    else:
        raise NotImplementedError("Only 2D gradients are supported")
    now = datetime.now()
    fig.savefig(savename + now.strftime("_%H:%M:%S_%d-%m") + ".png")

class PrintGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name, prefix, print_input, mean_dim = None, all_steps = True, batch_idx = None, timesteps = [], title: list = [], length: int = None, norm: str = None):
        ctx.name = name
        ctx.prefix = prefix
        ctx.print_input = print_input
        ctx.mean_dim = mean_dim
        ctx.all_steps = all_steps
        ctx.length = length
        ctx.norm = norm
        if batch_idx is not None:
            if type(batch_idx) == int:
                ctx.batch_idx = [batch_idx]
            else:
                ctx.batch_idx = batch_idx
        ctx.timesteps = timesteps
        ctx.title = title
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        name = ctx.name
        prefix = ctx.prefix
        print_input = ctx.print_input
        x, = ctx.saved_tensors
        if ctx.batch_idx is not None:
            prefix += "/"
            prefix = "/u/marten.mueller/dev/ctc_baseline/output/" + prefix
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            if ctx.mean_dim is not None:
                g = grad_output[ctx.batch_idx]
                plot_gradients(g.mean(dim=ctx.mean_dim).squeeze(0).cpu().numpy(), prefix + name, "")
                # if print_input:
                #     print(f"Gradients ({name}): {g.mean(dim=ctx.mean_dim).cpu().numpy()} Sum: {g.sum()} Mean: {g.mean()}\nInput: {x[ctx.batch_idx].mean(dim=ctx.mean_dim).cpu().numpy()}\nNaN's: {torch.isnan(g).sum()}")
                # else:
                #     print(f"Gradients ({name}): {g.mean(dim=ctx.mean_dim).cpu().numpy()} Sum: {g.sum()} Mean: {g.mean()}\nNaN's: {torch.isnan(g).sum()}")
            elif ctx.all_steps:
                if ctx.length is not None:
                    l = ctx.length
                    if l > 50:
                        l = 50
                    g = grad_output[ctx.batch_idx, :l].detach().squeeze(0).cpu().numpy()
                    f = x[ctx.batch_idx, :l].detach().squeeze(0).cpu().numpy()
                else:
                    g = grad_output[ctx.batch_idx].detach().squeeze(0).cpu().numpy()
                    f = x[ctx.batch_idx].detach().squeeze(0).cpu().numpy()
                plot_gradients(g, prefix + name + "_all", ctx.title, True, f, ctx.norm)
            else:
                np.set_printoptions(threshold=10000)
                g = grad_output[ctx.batch_idx, ctx.timesteps].squeeze(0).cpu().numpy()
                plot_gradients(g, prefix + name + "_ts", ctx.title)
                # if print_input:
                #     x_b = x[ctx.batch_idx, ctx.timesteps].squeeze(0).cpu().numpy()
                #     print(f"Gradients ({name}): NaN's: {torch.isnan(grad_output).sum()}")
                #     for j, i in zip(ctx.timesteps, range(len(ctx.timesteps))):
                #         print(f"{j}: {g[i]} Max: {g[i].max()} Sum: {g[i].sum()}\nInput: {x_b[i]} Max: {x_b[i].max()} Sum: {x_b[i].sum()}")
                # else:
                #     print(f"Gradients ({name}): NaN's: {torch.isnan(grad_output).sum()}")
                #     for j, i in zip(ctx.timesteps, range(len(ctx.timesteps))):
                #         print(f"{j}: {g[i]} Max: {g[i].max()} Sum: {g[i].sum()}")
        else:
            if print_input:
                print(f"Gradients ({name}): {grad_output.mean(dim=ctx.mean_dim).cpu().numpy() if ctx.mean_dim is not None else grad_output} Sum: {grad_output.sum()} Mean: {grad_output.mean()}\nInput: {x.mean(dim=ctx.mean_dim).cpu().numpy() if ctx.mean_dim else (x[ctx.batch_idx] if ctx.batch_idx else x)}\nNaN's: {torch.isnan(grad_output).sum()}")
            else:
                print(f"Gradients ({name}): {grad_output.mean(dim=ctx.mean_dim).cpu().numpy() if ctx.mean_dim is not None else grad_output} Sum: {grad_output.sum()} Mean: {grad_output.mean()}\nNaN's: {torch.isnan(grad_output).sum()}")
        return grad_output, None, None, None, None, None, None, None, None, None, None
    
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
    
class NormGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        print("Before Norm", grad_output[0].sum(dim=-1), grad_output.shape)
        # grad = torch.softmax(grad_output, dim=-1)
        # grad_output = grad_output / grad_output.norm(dim=-1, keepdim=True)
        grad_output = torch.nn.functional.normalize(grad_output, p=1, dim=-1)
        print("After Norm", grad_output[0].sum(dim=-1), grad_output.shape)
        return grad_output

# ------------------------------------------------------------

def test_logsumexp():
    # torch.autograd.set_detect_anomaly(True)
    
    ag = PrintGradients.apply
    
    # t = torch.tensor([float("-inf"), float("-inf"), float("-inf")], requires_grad=True)
    # t = torch.tensor([float("-inf"), float("-0.5"), float("-1")], requires_grad=True)
    # t = ag(t, "t", False)
    
    # sum_t = torch.logsumexp(t, dim=0)
    # sum_t.backward()
    # print(sum_t)
    
    # sum_t2 = safe_logsumexp(t, dim=0)
    # sum_t2.backward()
    # print(sum_t2)
    
    t2 = torch.full((5,), float("-inf"), requires_grad=True)
    idx = torch.tensor([0, 1, 0, 1, 2])
    src = torch.tensor([0.4, 0.0, 0.2, 0.0, 0.5], requires_grad=True).log()
    t2 = ag(t2, "t2", False)
    src = ag(src, "src", False)
    
    scat = scatter_safe_logsumexp(t2, 0, idx, src)
    scat.backward(torch.ones_like(scat))
    print(scat.exp())

def test_mul():
    # torch.autograd.set_detect_anomaly(True)
    
    import time
    import gc
    torch.manual_seed(0)
    ag = PrintGradients.apply
    
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    
    r = torch.tensor([0.0], device=device)
    time_d = 0.0
    for i in range(1):
        s = time.time()
        # A = torch.randn(10, 185, device=device)
        A = torch.randn(10, 185, 185, device=device)
        # B = torch.randn(1000, 1500, device=device).log_softmax(dim=0)
        # B = torch.randn(185, 185, device=device).log_softmax(dim=1)
        B = torch.randn(185, 185, 185, device=device).log_softmax(dim=1)
        # A[0] = float("-inf")
        # B[:, 0] = float("-inf")
        A.requires_grad = True
        B.requires_grad = True
        
        A = ag(A, "A", False)
        B = ag(B, "B", False)
        
        # res = A.exp().matmul(B.exp()).log()
        # res = log_matmul_alt(A, B)
        # A, topk_idx = torch.topk(A, 20, dim=-1)
        # B = B.unsqueeze(0).expand(10, -1, -1)
        # B = B.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 185))
        # res = log_matmul(A, B, batch_given=True)
        res = log_matmul(A, B)
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
            
            loss = sum_loss_ngram(
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
    
    # torch.autograd.set_detect_anomaly(True)
    # torch.cuda.set_sync_debug_mode(1)
    
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    
    batch_size = 12 # 14000 per GPU, 1250 stpes a 12 seqs (0.7 sec/step)
    vocab_size = 185
    frames = 100
    LM_order = 2
    
    torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # ag = AssertGradients.apply
    ag = PrintGradients.apply
    ng = NormGradients.apply
    
    # lm = torch.randn((vocab_size - 1,) * LM_order, device=device)
    # lm = torch.nn.functional.log_softmax(lm, dim=-1)
    if LM_order == 2:
        lm_path = "/u/marten.mueller/dev/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.wuVkNuDg8B55/output/lm.pt"
    elif LM_order == 3:
        lm_path = "/u/marten.mueller/dev/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.S9n2YtP1JzJ5/output/lm.pt"
    elif LM_order == 4:
        lm_path = "/u/marten.mueller/dev/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.XxvP7yk50Q8u/output/lm.pt"
    with open(lm_path, "rb") as f:
        lm = torch.load(f)
        
    lm = torch.log_softmax(lm, dim=-1)
    
    l = torch.tensor([0.0], device=device)
    s1 = time.time()
    for i in range(1):
        s = time.time()
        am = torch.randn(frames, batch_size, vocab_size, device=device)
        # am[0, 0, vocab_size - 1] = float(3)
        # am[2, 0, vocab_size - 1] = float(3)
        am.requires_grad = True
        
        am = am.permute(1, 0, 2)
        # am = ag(am, "logits", "/u/marten.mueller/dev/ctc_baseline/recipe/i6_experiments/users/mueller/experiments/ctc_baseline", False, None, True, 0, [],  ["Logits"], frames)
        am = am.permute(1, 0, 2)
        
        am = torch.cat([torch.full((1,1,2), float("-inf"), device=device).expand(frames, batch_size, 2), torch.nn.functional.log_softmax(am[:, :, 2:], dim=-1)], dim = -1)
        
        # res = 0.0
        # res = am[0, 0, :].unsqueeze(0)
        # for t in range(1, frames):
        #     res = log_matmul(res, am[t, 0, :].unsqueeze(0).expand(vocab_size, vocab_size))
        # print(safe_logsumexp(res, dim=-1), safe_logsumexp(res, dim=-1).exp())
        
        prior = torch.randn(vocab_size + 1, requires_grad=True, device=device)
        prior = torch.nn.functional.log_softmax(prior, dim=-1)
        
        length = torch.full((batch_size,), frames, device=device)
        # length[0] -= 3

        # am = am.permute(1, 0, 2)
        # prior = _calc_log_prior(am, length, use_max=True)
        # am = am.permute(1, 0, 2)
        
        am = am.permute(1, 0, 2)
        # x, name, prefix, print_input, mean_dim = None, all_steps = True, batch_idx = None, timesteps = [], title: list = []
        # am = ag(am, "log_probs", "/u/marten.mueller/dev/ctc_baseline/recipe/i6_experiments/users/mueller/experiments/ctc_baseline", False, None, True, 0, [],  ["Log Probs"], frames)
        # am = ng(am)
        # am = ag(am, "AM", False)
        am = am.permute(1, 0, 2)
        # prior = ag(prior, "prior", False)
        
        loss = sum_loss_ngram(
            log_probs=am,
            log_lm_probs=lm,
            log_prior=prior,
            input_lengths=length,
            top_k=0,
            LM_order=LM_order,
            am_scale=1.0,
            lm_scale=0.0,
            prior_scale=0.3,
            horizontal_prior=True,
            blank_prior=True,
            blank_idx=184,
            eos_idx=0,
            print_best_path_for_idx=[0],
            alignment_topk=False,
            blank_correction_version = 0,
            correction_in_final_score = False
        )
        print("OUT", (-loss[0]).tolist(), (-loss[0]).exp().tolist())
        # l += (loss / frames).mean()
        l = loss
        
        # del loss, am, prior
        # torch.cuda.empty_cache()
        print("Time:", time.time() - s)
        
        # targets = torch.tensor(
        #     [ 34,  34, 117,  31,  67,  12, 146, 107, 154,  45,  45, 123,  17,  53,
        #     35,  97,  97, 120,  48, 135, 103,  20,  75, 117,  96, 120,  58,   9,
        #     30,  27,  13,  28, 162, 175, 123, 151,  65,  70, 145, 109, 103,  76,
        #     99, 156, 141, 157, 173,  62,  44,  89, 155, 159,  58,   6,  19, 126,
        #     138, 152, 152,  44, 111, 122,  14,  50, 146, 122, 179,   4,  36,  55,
        #     117,  99,  42, 171, 157, 137,  21, 128,   9,  78,  19,  31,  37, 140,
        #     16,  30, 109, 102,  32,  72, 146,  37, 126, 130, 147,  83,  19, 137,
        #     13,  14]
        # )
        # targets = targets - 2
        # greedy_probs, greedy_idx = torch.max(am[:, 0:1], dim=-1)
        # # print(greedy_idx.squeeze(-1))
        # # print(greedy_probs.sum())
        # targets = targets.unsqueeze(0)
        # target_lengths = torch.tensor([targets.size(1)])
        # ctc_loss = torch.nn.functional.ctc_loss(
        #     log_probs=am[:, 0:1],
        #     targets=targets,
        #     input_lengths=length[0:1],
        #     target_lengths=target_lengths,
        #     blank=184,
        #     reduction="none"
        # )
        # print(ctc_loss)
        
        
    l.backward(torch.ones_like(l, device=device))
    e1 = time.time()
    # print(f"Sum loss took {time.strftime('%H:%M:%S', time.gmtime(e1-s1))}: {l}") # 5:00 mins
    
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

def test_LM():
    with open("/u/marten.mueller/dev/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.wuVkNuDg8B55/output/lm.pt", "rb") as f: # 2-gram
    # with open("/u/marten.mueller/dev/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.S9n2YtP1JzJ5/output/lm.pt", "rb") as f: # 3-gram
    # with open("/u/marten.mueller/dev/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.XxvP7yk50Q8u/output/lm.pt", "rb") as f: # 4-gram
        t = torch.load(f)
    # print(safe_logsumexp(safe_logsumexp(safe_logsumexp(t, dim=-1), dim=-1), dim=-1))
    print(t.shape)
    t = torch.log_softmax(t, dim=-1)
    print(safe_logsumexp(t, dim=-1))
    print(t.isnan().sum())
    
def test_get_bpes():
    tokens = "[117] [117] [117] [117] [117] [117] [117] [46] [46] [46] [46] [21] [35] [35] [55] [55] [120] [120] [120] [26] [26] [76] [13] [26] [10] [10] [74] [116] [7] [7] [19] [13] [53] [57] [57] [108] [108] [10] [10] [10] [42] [42] [24] [24] [34] [34] [34] [55] [25] [14] [14] [18] [18] [18] [18] [18] [156] [156] [9] [81] [81] [13] [15] [21] [50] [50] [23] [23] [113] [113] [113] [28] [77] [77] [27] [27] [39] [170] [2] [18] [18] [20] [20] [142] [142] [142] [170] [170] [170] [106] [106] [7] [7] [33] [33] [162] [162] [162] [52] [52] [18] [18] [18] [18] [18] [18] [18] [18] [18] [18] [18] [18] [18]"
    print(get_bpes(tokens))

if __name__ == "__main__":
    test()