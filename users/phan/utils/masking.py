import torch
import numpy as np

def get_seq_mask(seq_lens: torch.Tensor, max_seq_len, device):
    """
    Convert seq_lens to a sequence mask.
    1 is in sequence, 0 is not in sequence.

    :param seq_lens: Sequence lengths (B,)
    :param max_seq_len: Maximum sequence length
    :return: Sequence mask in float32 (B, max_seq_len)
    """
    assert (seq_lens <= max_seq_len).all(), "One of the sequence length is larger than max seq len"
    batch_size = seq_lens.shape[0]
    seq_lens = seq_lens.to(device)
    seq_mask = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1) < seq_lens.unsqueeze(-1).expand(-1, max_seq_len)
    seq_mask = seq_mask.float().to(device)
    return seq_mask

def mask_audio_features_with_alignments_single_seq(
    alignment: torch.Tensor,
    mask_ratio=0.15,
    sil_index=0,
):
    """
    Single sequence version.

    Given alignment of a single sequence, randomly mask out
    audio input of labels according to the masking ratio.

    This assumes that the target sequence is the same as the alignment.
    (Important in some cases like phoneme-based models)

    :param alignments: (audio features len,), usually frame-wise alignments,
    not to be confused with raw audio input
    :param mask_ratio: How many labels should be masked
    :sil_index: Index of silence, in CTC should be same as blank
    :returns: 0-1 maskings for audio features and target sequence
    audio features: 1 = no mask, 0 = mask
    target sequence: 1 = mask, 0 = no mask
    """
    device = alignment.device
    unique, counts = alignment.unique_consecutive(return_counts=True)
    rand = torch.rand(unique.shape, device=device)
    not_sil = (unique != sil_index)
    mask = (rand < mask_ratio)
    unique_mask = torch.logical_and(mask, not_sil).logical_not().float() # only mask if it's not sil
    feature_mask = torch.repeat_interleave(unique_mask, repeats=counts).float().to(device) # 0 1 masking
    target_mask = 1. - unique_mask[unique != sil_index] # for target masking 0-1 is reversed
    return feature_mask, target_mask

def mask_audio_features_with_alignments(
    alignments: torch.Tensor,
    mask_ratio=0.15,
    sil_index=0,
):
    """
    Batched version.

    Given alignments, randomly mask out
    audio input of labels according to the masking ratio.

    This assumes that the target sequence is the same as the alignment.
    (Important in some cases like phoneme-based models)

    :param alignments: (B, max audio features len), usually frame-wise alignments,
    not to be confused with raw audio input
    :param mask_ratio: How many labels should be masked
    :sil_index: Index of silence, in CTC should be same as blank
    :returns: 0-1 maskings for audio features and target sequence
    audio features: 1 = no mask, 0 = mask (B, T)
    target sequence: 1 = mask, 0 = no mask (B, S)
    """
    # pad start and end with sil_index, reshape to 1-dim
    # then we can apply the single seq version
    # assert len(alignments.shape) == 2, "Alignments shape must be (B, T)"
    device = alignments.device
    batch_size, _ = alignments.shape
    # alignments_pad = torch.cat(
    #     [torch.full((batch_size, 1), sil_index, device=device), alignments, torch.full((batch_size, 1), sil_index, device=device)],
    #     dim=1,
    # )
    # alignments_one_seq = alignments_pad.reshape(-1)
    # feature_mask_one_seq = mask_audio_features_with_alignments_single_seq(alignments_one_seq, mask_ratio=mask_ratio, sil_index=sil_index)
    # feature_mask = feature_mask_one_seq.reshape(batch_size, -1)[:, 1:-1]
    # return feature_mask
    feature_masks = []
    target_masks = []
    for b in range(batch_size):
        cur_align = alignments[b]
        feature_mask, target_mask = mask_audio_features_with_alignments_single_seq(cur_align, mask_ratio=mask_ratio, sil_index=sil_index)
        feature_masks.append(feature_mask)
        target_masks.append(target_mask)
    batch_feature_mask = torch.stack(feature_masks)
    batch_target_mask = torch.nn.utils.rnn.pad_sequence(target_masks, batch_first=True)
    return batch_feature_mask, batch_target_mask
