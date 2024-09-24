import torch

def convert_alignments_to_target_sequences(
    alignments: torch.Tensor,
    sil_index: int = 0,
    pad_value: int = 0,
):
    """
    Convert alignments to the target sequences they represent

    :param alignments: Alignments in (B, T)
    :param sil_index: Silence index in alignments
    :param pad_value: Padding value for the target sequences
    :returns:
    targets: (B, S) Target sequences
    targets_len: (B,) Lengths of the target sequences
    """
    batch_size = alignments.shape[0]
    device = alignments.device
    targets_align = []
    targets_len_align = []
    for b in range(batch_size):
        cur_align = alignments[b]
        seq = cur_align.unique_consecutive()
        seq_no_sil = seq[seq != sil_index]
        targets_align.append(seq_no_sil)
        targets_len_align.append(len(seq_no_sil))
    targets = torch.nn.utils.rnn.pad_sequence(targets_align, batch_first=True, padding_value=pad_value).long()
    targets_len = torch.tensor(data=targets_len_align, dtype=torch.int64).cpu()
    return targets, targets_len
