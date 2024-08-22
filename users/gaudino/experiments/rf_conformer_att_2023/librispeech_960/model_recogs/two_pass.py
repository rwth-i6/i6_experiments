import torch
import returnn.frontend as rf


def rescore_w_ctc(
    search_args, seq_targets, seq_log_prob, ctc_logits, batch_size, beam_size, blank_idx=10025
):
    """rescore hyps with ctc"""

    ctc_logits = ctc_logits.to("cpu")
    ctc_scores = torch.zeros_like(seq_log_prob.raw_tensor)
    seq_targets_raw = torch.clone(seq_targets.raw_tensor).to("cpu")  # [T, Batch, Beam]
    for i in range(batch_size):
        for j in range(beam_size):
            seq = seq_targets_raw[:, i, j]
            seq = seq[seq != 0]
            ctc_scores[i, j] = ctc_forward_algorithm(ctc_logits[i], seq, blank_idx)

    ctc_scores = rf.Tensor('ctc_re_scores', seq_log_prob.dims, dtype=seq_log_prob.dtype, raw_tensor=ctc_scores)
    seq_log_prob = (
        search_args["rescore_att_scale"] * seq_log_prob
        + search_args["rescore_ctc_scale"] * ctc_scores
    )

    return seq_targets, seq_log_prob


def ctc_forward_algorithm(ctc_log_probs, seq, blank_idx=10025, rescale=False):
    """ctc forward score for all possible paths."""

    mod_seq = torch.stack([seq, torch.fill(seq, blank_idx)], dim=1).flatten()
    mod_seq = torch.cat((torch.tensor([blank_idx], device=mod_seq.device), mod_seq))
    mod_seq_len = mod_seq.size(0)

    # ctc_log_probs [T, O]
    T = ctc_log_probs.size(0)
    A = torch.full((T, mod_seq_len), float("-inf"), device="cpu")  # [T, S]
    C = torch.full((T,), float("-inf"), device="cpu")  # [T]

    # Initialize the first row of the forward variable
    A[0, 0] = ctc_log_probs[0, blank_idx]
    A[0, 1] = ctc_log_probs[0, seq[0]]

    # About rescaling: in the orig CTC paper they suggest to rescale at each step to avoid underflow.
    # However in his dissertation, Alex Graves says working in log space is even more stable
    # At least they give the same result I think.

    # rescale first row

    if rescale:
        C[0] = torch.logsumexp(A[0], 0)
        A[0] = A[0] - C[0]

    # Iteration
    for i in range(1, T):

        prev_A_shift = torch.roll(A[i - 1], 1, dims=0)
        prev_A_shift[0] = float("-inf")
        prev_A_comb_1 = torch.logsumexp(torch.stack((A[i - 1], prev_A_shift)), dim=0)

        prev_A_shift_2 = torch.roll(A[i - 1], 2, dims=0)
        prev_A_shift_2[0] = float("-inf")
        prev_A_shift_2[1] = float("-inf")
        prev_A_comb_2 = torch.logsumexp(torch.stack((prev_A_comb_1, prev_A_shift_2)), dim=0)

        mask = torch.logical_or(mod_seq == blank_idx, mod_seq == torch.roll(mod_seq, -2, dims=0))
        prev_A_comb = torch.where(mask, prev_A_comb_1, prev_A_comb_2)

        A[i] = prev_A_comb + ctc_log_probs[i, mod_seq]

        if rescale:
            C[i] = torch.logsumexp(A[i], 0)
            A[i] = A[i] - C[i]

    if rescale:
        res = torch.sum(C)
    else:
        res = torch.logsumexp(torch.stack((A[T-1, mod_seq_len-1], A[T-1, mod_seq_len-2])), dim=0)

    return res


# def ctc_viterbi_score(ctc_log_probs, seqs, blank_idx=10025):
#     """ctc score for the best path."""
#
#     breakpoint()
#     ctc_raw = ctc_log_probs
#     batch_n = seqs.raw_tensor.shape[0]
#     seq_len = seqs.raw_tensor.shape[1]
#
#     seqs = seqs.raw_tensor
#     # seq_lens = seqs.dims[1].dyn_size_ext.raw_tensor
#     ext_seq_len = 2 * seq_len +1
#     ext_seqs = torch.stack([seqs, torch.fill(seqs, blank_idx)], dim=2).flatten(start_dim=1)
#     ext_seqs = torch.cat((torch.tensor([blank_idx]), ext_seqs))
#
#     # Initialization
#     # Transition matrix A, Path variables V
#
#     V = torch.full([batch_n, ctc_raw.shape[1], ext_seq_len], float("-inf"), dtype=ctc_log_probs.dtype) # [B, T, S]
#     backrefs = torch.full([batch_n, ctc_raw.shape[1], ext_seq_len], -1, dtype="int32") # [B, T, S]
#
#     V[:, 0, 0] = ctc_log_probs[:, 0, blank_idx]
#     V[:, 0, 1] = ctc_log_probs[:, 0, seqs[0]]
#
#     # Iteration
#     for i in range(1, ctc_log_probs.dims[0].get_dim_value()): # T
#         for j in range(ext_seq_len.dimension): # S
#             prev_paths = []
#             prev_paths.append(V[i-1, j])
#             if j > 0:
#                 prev_paths.append(V[i-1, j-1])
#             if j > 1 and seqs[j//2] != seqs[j//2-1]:
#                 prev_paths.append(V[i-1, j-2])
#             prev_contrib = rf.max(prev_paths, axis=0)
#             backrefs[i, j] = rf.argmax(prev_paths, axis=0)
#
#             V[i, j] = prev_contrib + ctc_log_probs[i, seq[j]]
#
#     # Backtracking
#     best_path = []
#     for i in range(ext_seq_len-1, -1, -1):
#         best_path.append(seqs[i//2])
#         i = backrefs[i, i]
#     score = rf.max(V[-1])
#
#     return score, best_path[::-1]

def ctc_viterbi_one_seq(ctc_log_probs, seq, t_max, blank_idx=10025):
    mod_len = 2 * seq.shape[0] + 1
    mod_seq = torch.stack([seq, torch.full(seq.shape, blank_idx,device=seq.device)], dim=1).flatten()
    mod_seq = torch.cat((torch.tensor([blank_idx], device=mod_seq.device), mod_seq))
    V = torch.full((t_max, mod_len), float("-inf"))  # [T, 2S+1]

    V[0, 0] = ctc_log_probs[0, blank_idx]
    V[0, 1] = ctc_log_probs[0, seq[0]]

    backref = torch.full((t_max, mod_len), -1, dtype=torch.int64, device="cuda")

    for t in range(1, t_max):
        for s in range(mod_len):
            if s > 2 * t + 1:
                continue
            skip = False
            if s % 2 != 0 and s >= 3:
                idx = (s - 1) // 2
                prev_idx = (s - 3) // 2
                if seq[idx] != seq[prev_idx]:
                    skip = True

            if skip:
                V[t, s] = max(V[t - 1, s], V[t - 1, s - 1], V[t - 1, s - 2]) + ctc_log_probs[t, mod_seq[s]]
                backref[t, s] = torch.argmax(torch.tensor([V[t - 1, s], V[t - 1, s - 1], V[t - 1, s - 2]]))
            else:
                V[t, s] = max(V[t - 1, s], V[t - 1, s - 1]) + ctc_log_probs[t, mod_seq[s]]
                backref[t, s] = torch.argmax(torch.tensor([V[t - 1, s], V[t - 1, s - 1]]))

    score = torch.max(V[t_max - 1, :])
    idx = torch.argmax(V[t_max - 1, :])
    res = [mod_seq[idx]]

    for t in range(t_max - 1, 0, -1):
        next_idx = idx - backref[t, idx]
        res.append(mod_seq[next_idx])
        idx = next_idx

    res = torch.tensor(res).flip(0)
    return res, score

def scale_hyp_wo_blank(ctc_log_probs, seq, ctc_scale, blank_idx=10025):
    blank_mask = (seq == blank_idx).to("cuda")

    ctc_scores = torch.gather(ctc_log_probs, 1, seq.unsqueeze(1).to("cuda")).squeeze()

    scores_blank = torch.masked_select(ctc_scores, blank_mask)
    scores_no_blank = torch.masked_select(ctc_scores, ~blank_mask)

    score_blank = torch.sum(scores_blank)
    score_no_blank = torch.sum(scores_no_blank) * ctc_scale

    score = score_blank + score_no_blank

    return score



