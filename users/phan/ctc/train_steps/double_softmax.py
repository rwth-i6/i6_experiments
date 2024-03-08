from returnn.tensor import batch_dim
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch
from i6_experiments.users.phan.ctc.ctc_pref_scores_loss import ctc_double_softmax_loss
from i6_experiments.users.phan.utils import get_seq_mask

def train_step(
    *,
    model: torch.nn.Module,
    extern_data: TensorDict,
    am_scale: float,
    lm_scale: float, 
    **kwargs
):
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
    model.module_dict["train_lm"].eval()

    log_probs, sequence_mask = model(
        args = [],
        kwargs = {
            "audio_features": audio_features,
            "audio_features_len": audio_features_len.to("cuda"),
        },
        module="conformer_ctc",
        inference=False,
    )
    input_lengths = torch.sum(sequence_mask.long(), dim=1)
    device = log_probs.device
    batch_size, max_seq_len = targets.shape
    # pad 0 at the beginning a,b,c -> <eos>,a,b,c
    eos_targets = torch.cat(
        [torch.zeros((batch_size, 1), device=targets.device), targets],
        dim=1,
    ).long()
    eos_targets_one_hot = torch.nn.functional.one_hot(
        eos_targets,
        num_classes=model.module_dict["train_lm"].cfg.vocab_dim
    ).to(device).float()

    log_lm_score = model(
        args=[eos_targets_one_hot],
        kwargs={},
        module="train_lm",
        inference=True,
    )

    log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, F]
    loss = ctc_double_softmax_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=targets_len,
        log_lm_score=log_lm_score.detach(),
        am_scale=am_scale,
        lm_scale=lm_scale,
        blank_idx=0,
        log_zero=-1e25,
    )
    targets_len_rf.raw_tensor += 1 # due to SOS
    rf.get_run_ctx().mark_as_loss(
        name="double_softmax", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )
    targets_len_rf.raw_tensor -= 1
    # also report CTC loss of Conformer CTC
    ctc_loss = torch.nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=targets_len,
        blank=0,
        reduction="sum",
        zero_infinity=True,
    )
    rf.get_run_ctx().mark_as_loss(
        name="ctc", loss=ctc_loss, as_error=True,
        custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim),
    )
