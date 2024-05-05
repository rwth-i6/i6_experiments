"This train step uses alignments to randomly mask out acoustic inputs of some labels."

from returnn.tensor import batch_dim
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch
from i6_experiments.users.phan.ctc.ctc_pref_scores_loss import kldiv_ctc_lm_loss
from i6_experiments.users.phan.utils import get_seq_mask, mask_audio_features_with_alignments


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, mask_ratio: float, sil_index: 0, **kwargs):
    """
    Note: The student LM should have an attribute self.cfg.vocab_dim
    to tell the one-hot encoding the vocab size
    """
    audio_features = extern_data["data"].raw_tensor
    assert extern_data["data"].dims[1].dyn_size_ext is not None

    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_len is not None

    assert extern_data["targets"].raw_tensor is not None
    targets = extern_data["targets"].raw_tensor.long()

    targets_len_rf = extern_data["targets"].dims[1].dyn_size_ext
    assert targets_len_rf is not None
    

    if extern_data["align"] is not None:
        # if there is alignment, take the target sequence from the alignments
        # due to pronunciation variances
        # change targets and targets_len_rf
        alignments = extern_data["align"].raw_tensor
        batch_size = alignments.shape[0]
        targets_align = []
        targets_len_align = []
        for b in range(batch_size):
            cur_align = alignments[b]
            seq = cur_align.unique_consecutive()
            seq_no_sil = seq[seq != sil_index]
            targets_align.append(seq_no_sil)
            targets_len_align.append(len(seq_no_sil))
        new_targets = torch.nn.utils.rnn.pad_sequence(targets_align, batch_first=True)
        new_targets_len = torch.tensor(data=targets_len_align, dtype=torch.int32, device=audio_features.device)
        targets = new_targets.long()
        targets_len_rf.raw_tensor = new_targets_len

    targets_len = targets_len_rf.raw_tensor
    assert targets_len is not None
    # print(targets)
    # print(targets_len)

    model.train()
    model.module_dict["teacher_ctc"].eval()

    
    device = audio_features.device
    batch_size, max_seq_len = targets.shape

    forward_kwargs = {
        "audio_features": audio_features,
        "audio_features_len": audio_features_len.to("cuda"),
    }
    if extern_data["align"] is not None: # some returnn code was changed to allow this
        # train data
        alignments = extern_data["align"].raw_tensor
        audio_features_mask, target_mask = mask_audio_features_with_alignments(alignments, mask_ratio, sil_index)
        forward_kwargs["audio_features_mask"] = audio_features_mask
        target_mask = torch.cat([target_mask, torch.zeros((batch_size, 1), device=device, dtype=torch.float32)], dim=1)
        # for b in range(batch_size):
        # #     print(b)
        # #     t0 = targets[b]
        # #     print(t0[t0.nonzero(as_tuple=True)])
        # #     a0 = alignments[b].unique_consecutive()
        # #     print(a0[a0.nonzero(as_tuple=True)])
        # #     try:
        # #         print(t0[t0.nonzero(as_tuple=True)] == a0[a0.nonzero(as_tuple=True)])
        # #     except:
        # #         print("shape mismatch")
        #     print(b)
        #     print(alignments[b])
        #     print(audio_features_mask[b])
        #     print(targets[b])
        #     print(target_mask[b])
        #     mask_alignment = alignments[b]*audio_features_mask[b]
        #     mask_alignment_reduced = mask_alignment.unique_consecutive()
        #     mask_alignment_reduced_nosil = mask_alignment_reduced[mask_alignment_reduced != 0.]
        #     print(mask_alignment_reduced_nosil)
    else: # dev data
        target_mask = None

    
    log_probs, sequence_lengths = model(
        args = [],
        kwargs = forward_kwargs,
        module="teacher_ctc",
        inference=True,
    )
    sequence_lengths = sequence_lengths.long()

    # pad 0 at the beginning a,b,c -> <eos>,a,b,c
    eos_targets = torch.cat(
        [torch.zeros((batch_size, 1), device=targets.device), targets],
        dim=1,
    ).long()
    eos_targets_one_hot = torch.nn.functional.one_hot(
        eos_targets,
        num_classes=model.module_dict["student_lm"].cfg.vocab_dim
    ).to(device).float()

    log_lm_score = model(
        args=[eos_targets_one_hot],
        kwargs={},
        module="student_lm",
        inference=False,
    )

    log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, F]
    loss = kldiv_ctc_lm_loss(
        log_probs=log_probs.detach(),
        targets=targets,
        input_lengths=sequence_lengths,
        target_lengths=targets_len,
        log_lm_score=log_lm_score,
        blank_idx=0,
        log_zero=-1e25,
        eos_idx=None,
        target_mask=target_mask,
    )
    if target_mask is not None:
        targets_len_rf.raw_tensor = target_mask.sum(dim=1).int()
    else:
        targets_len_rf.raw_tensor += 1
    rf.get_run_ctx().mark_as_loss(
        name="kldiv_ctc_lm", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )
    # Also report PPL of the LM
    targets_eos = torch.cat(
        [targets, torch.zeros((batch_size, 1), device=targets.device)],
        dim=1,
    ).long()
    ce = torch.nn.functional.cross_entropy(log_lm_score.transpose(1, 2), targets_eos, reduction='none')
    seq_mask = get_seq_mask(targets_len, max_seq_len+1, device)
    log_ppl = (ce*seq_mask).sum()/(targets_len.sum())
    ppl = torch.exp(log_ppl)
    rf.get_run_ctx().mark_as_loss(
        name="student_lm_ppl", loss=ppl, as_error=True,
    )
    if target_mask is not None: # also report PPL on masked tokens
        log_ppl_masked = (ce*seq_mask*target_mask).sum()/(target_mask.sum())
        ppl_masked = torch.exp(log_ppl_masked)
        rf.get_run_ctx().mark_as_loss(
            name="student_lm_ppl_masked", loss=ppl_masked, as_error=True,
        )
