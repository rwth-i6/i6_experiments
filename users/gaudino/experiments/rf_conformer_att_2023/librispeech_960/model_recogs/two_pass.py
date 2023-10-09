import torch
import returnn.frontend as rf


def rescore_w_ctc(
    model, seq_targets, seq_log_prob, ctc_logits, batch_size, beam_size, blank_idx=10025
):
    """rescore hyps with ctc"""

    ctc_scores = torch.zeros_like(seq_log_prob.raw_tensor)
    seq_targets_raw = torch.clone(seq_targets.raw_tensor)  # [T, Batch, Beam]
    for i in range(batch_size):
        for j in range(beam_size):
            seq = seq_targets_raw[:, i, j]
            seq = seq[seq != 0]
            ctc_scores[i, j] = ctc_forward_algorithm(ctc_logits[i], seq, blank_idx)

    ctc_scores = rf.Tensor('ctc_re_scores', seq_log_prob.dims, dtype=seq_log_prob.dtype, raw_tensor=ctc_scores)
    seq_log_prob = (
        model.search_args["att_scale"] * seq_log_prob
        + model.search_args["ctc_scale"] * ctc_scores
    )

    return seq_targets, seq_log_prob


def ctc_forward_algorithm(ctc_log_probs, seq, blank_idx=10025):
    mod_seq = torch.stack([seq, torch.fill(seq, blank_idx)], dim=1).flatten()
    mod_seq = torch.cat((torch.tensor([blank_idx], device="cuda"), mod_seq))
    mod_seq_len = mod_seq.size(0)

    # ctc_log_probs [T, O]
    T = ctc_log_probs.size(0)
    A = torch.full((T, mod_seq_len), float("-inf"), device="cuda")  # [T, S]

    # Initialize the first row of the forward variable
    A[0, 0] = ctc_log_probs[0, blank_idx]
    A[0, 1] = ctc_log_probs[0, seq[0]]

    # rescale first row ?

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

        # rescale A[i]
        A[i] = A[i] - torch.logsumexp(A[i], 0)

    return A[T - 1, mod_seq_len - 1] + A[T - 1, mod_seq_len - 2]


def viterbi_decode(
    tag_sequence: torch.Tensor, transition_matrix: torch.Tensor, top_k: int = 5
):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    top_k : int, required.
        Integer defining the top number of paths to decode.
    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : float
        The score of the viterbi path.
    """
    sequence_length, num_tags = list(tag_sequence.size())

    path_scores = []
    path_indices = []
    # At the beginning, the maximum number of permutations is 1; therefore, we unsqueeze(0)
    # to allow for 1 permutation.
    path_scores.append(tag_sequence[0, :].unsqueeze(0))
    # assert path_scores[0].size() == (n_permutations, num_tags)

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        # assert path_scores[timestep - 1].size() == (n_permutations, num_tags)
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)

        # Best pairwise potential path score from the previous timestep.
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)
        # assert scores.size() == (n_permutations, num_tags)
        # assert paths.size() == (n_permutations, num_tags)

        scores = tag_sequence[timestep, :] + scores
        # assert scores.size() == (n_permutations, num_tags)
        path_scores.append(scores)
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    path_scores = path_scores[-1].view(-1)
    max_k = min(path_scores.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()
        # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)
    return viterbi_paths, viterbi_scores
