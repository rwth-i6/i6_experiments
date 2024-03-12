import torch


def espnet_end_detect(
    ended_hyps_log_prob: torch.Tensor,
    ended_hyps_seq_len: torch.Tensor,
    *,
    i: int,
    m: int = 3,
    d_end: float = -10.0,
    bad_score: float = -1e30,
) -> torch.Tensor:
    """
    End detection.
    described in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

    Adapted from here: https://github.com/espnet/espnet/blob/master/espnet/nets/e2e_asr_common.py#L17

    :param ended_hyps_log_prob: [Batch,EndBeam].
        this assumes it is sorted in descending order, i.e. [:, 0] are the best hyps.
        values in ended_hyps_log_prob <= bad_score are ignored.
    :param ended_hyps_seq_len: [Batch,EndBeam]
    :param i: current decoder step
    :param m: parameter for end detection, how many recent hyp lengths to check
    :param d_end: parameter for end detection
    :param bad_score: lower threshold. values in ended_hyps_log_prob <= bad_score are ignored
    :return: [Batch]. whether to end the search for this batch entry (i.e. whether to prune all active hyps away)
    """
    batch_size = ended_hyps_log_prob.shape[0]
    if ended_hyps_log_prob.shape[1] == 0:
        return torch.full([batch_size], False, device=ended_hyps_log_prob.device)
    count = torch.zeros([batch_size], dtype=torch.int32, device=ended_hyps_log_prob.device)  # [Batch]
    best_hyp = ended_hyps_log_prob.max(dim=1).values  # [Batch]
    for m_ in range(m):
        # Get ended_hyps with their length is i - m - 2.
        # The offset -2 is because in ESPnet, the hyps seq len includes SOS and EOS,
        # while we do not include this here.
        # This is actually a bit strange, as e.g. in the first step (i=0),
        # all ended hyps in this step have length 2 in ESPnet,
        # or in general, in step i, all hyps ended in this step have length i+2 in ESPnet.
        hyps_this_length_log_prob = torch.where(
            (ended_hyps_seq_len == i - m_ - 2) & (ended_hyps_log_prob > bad_score),
            ended_hyps_log_prob,
            torch.full((), bad_score, device=ended_hyps_log_prob.device),
        )  # [Batch,Beam]
        best_hyp_this_length = torch.max(hyps_this_length_log_prob, dim=1).values  # [Batch]
        count = torch.where(
            (best_hyp_this_length < d_end + best_hyp) & (best_hyp_this_length > bad_score), count + 1, count
        )

    return count == m
