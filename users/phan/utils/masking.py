"""
Generally:
For masking audio features: 1 = no mask, 0 = mask
For masking target labels: 1 = mask, 0 = no mask
Be careful about this
"""

import torch
import numpy as np
from i6_experiments.users.phan.utils.alignments import convert_alignments_to_target_sequences

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

def mask_target_sequences(
    targets: torch.Tensor,
    mask_ratio=0.2,
    eos_idx=0,
    mask_idx=79,
):
    """
    Mask some label of target sequences according to a masking rate.
    Do not mask eos (maybe this can change)

    :param targets: Target sequences in (B, S), preferrably long. ALREADY WITH EOS.
    :param mask_ratio: How many labels to mask
    :param eos_idx: Index of end-of-sentence AND also pad tokens.
    Actually whether the pad tokens are masked or not may not be important.
    :param mask_idx: Replace masked token with this mask index.
    :returns: The masked target sequences, masked tokens replaced by mask index.

    Target masking: 1 = mask, 0 = no mask (B, S). Need this to compute the loss
    only on masked targets (For not masked token it's trivial, the token has seen itself)
    """
    device = targets.device
    rand = torch.rand(targets.shape, device=device)
    is_masked = (rand < mask_ratio)
    not_eos = (targets != eos_idx)
    # Only mask if it is not eos
    target_masking = torch.logical_and(is_masked, not_eos)
    targets_masked = torch.where(target_masking, mask_idx, targets)
    target_masking = target_masking.float()
    return targets_masked, target_masking

def mask_audio_features_exact_label_pos_single_seq(
    alignment: torch.Tensor,
    mask: torch.Tensor,
    sil_index=0,
):
    """
    For a single alignment, mask acoutic input of labels
    according to positions given by a mask.

    :param alignment: A single alignment (T,)
    :param mask: 1 = mask, 0 = no mask (S,)
    :param sil_index: Silence in the alignment
    :returns:
    alignment_mask Masking for audio features. 1 = no mask, 0 = mask
    """
    n_pos = len(mask)
    s_idx = torch.arange(n_pos)
    device = alignment.device
    masked_pos = s_idx[mask.bool()] # index of masked positions
    unique, counts = alignment.unique_consecutive(return_counts=True)
    alignment_mask_unique = []
    last_label_idx = -1
    masked_pos_idx = 0
    for i in range(len(unique)):
        frame_mask = 1
        if unique[i] != sil_index:
            last_label_idx += 1
            if masked_pos_idx < len(masked_pos):
                if last_label_idx == masked_pos[masked_pos_idx]:
                    frame_mask = 0
                    masked_pos_idx += 1
        alignment_mask_unique.append(frame_mask)
    alignment_mask = torch.repeat_interleave(torch.tensor(alignment_mask_unique).to(device), counts.to(device))
    return alignment_mask

def mask_audio_features_exact_label_pos(
    alignments: torch.Tensor,
    label_mask: torch.Tensor,
    sil_index=0,
    pad_value=0,
):
    """
    Basically the same as mask_audio_features_with_alignments,
    but all target sequences are masked at the same positions
    along the S axis. The mask is computed beforehand

    :param alignments: (B, max audio features len), usually frame-wise alignments,
    not to be confused with raw audio input
    :param label_mask: A masking of size (S,). Must guarantee that S is
    max sequence length of the targets represented by the alignments.
    :sil_index: Index of silence, in CTC should be same as blank
    :returns: 0-1 maskings for audio features, 1 = no mask, 0 = mask (B, T)
    """
    feature_masks = []
    device = alignments.device
    batch_size = alignments.shape[0]
    for b in range(batch_size):
        feature_masks_b = mask_audio_features_exact_label_pos_single_seq(alignments[b], label_mask, sil_index)
        feature_masks.append(feature_masks_b)
    feature_masks = torch.nn.utils.rnn.pad_sequence(feature_masks, batch_first=True).to(device).float()
    return feature_masks
