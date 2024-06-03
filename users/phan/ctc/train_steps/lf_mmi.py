from returnn.tensor import batch_dim
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch
from i6_experiments.users.phan.ctc.ctc_lf_mmi import ctc_lf_mmi_context_1
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
    n_out = model.module_dict["train_lm"].cfg.vocab_dim
    vocab = torch.arange(n_out)
    vocab_one_hot = torch.nn.functional.one_hot(vocab, num_classes=n_out).to(device).float()

    # All possible bigram scores (V, V)
    log_lm_probs = model(
        args=[vocab_one_hot],
        kwargs={},
        module="train_lm",
        inference=True,
    )

    log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, F]
    loss = ctc_lf_mmi_context_1(
        log_probs=log_probs,
        log_lm_probs=log_lm_probs.detach(),
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=targets_len,
        am_scale=am_scale,
        lm_scale=lm_scale,
        blank_idx=0,
        log_zero=-1e15,
    )
    rf.get_run_ctx().mark_as_loss(
        name="lf_mmi", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )
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
