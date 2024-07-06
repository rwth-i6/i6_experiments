from returnn.tensor import batch_dim
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch
from i6_experiments.users.phan.ctc.ctc_pref_scores_loss import kldiv_ctc_lm_loss
from i6_experiments.users.phan.utils import get_seq_mask

def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
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
    targets_len = targets_len_rf.raw_tensor
    assert targets_len is not None

    model.train()
    model.module_dict["teacher_ctc"].eval()

    log_probs, sequence_mask, _ = model(
        args = [],
        kwargs = {
            "audio_features": audio_features,
            "audio_features_len": audio_features_len.to("cuda"),
        },
        module="teacher_ctc",
        inference=True,
    )
    sequence_lengths = sequence_mask.sum(-1).long()
    device = log_probs.device
    batch_size, max_seq_len = targets.shape
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
    )
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
