from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    targets = extern_data["targets"].raw_tensor.long()
    targets_len = extern_data["targets"].dims[1].dyn_size_ext.raw_tensor

    model.train()

    log_probs, sequence_mask = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, F]
    sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)

    loss = torch.nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=sequence_lengths,
        target_lengths=targets_len,
        blank=0,
        reduction="sum",
        zero_infinity=True,
    )

    loss /= torch.sum(sequence_lengths)

    rf.get_run_ctx().mark_as_loss(name="CTC", loss=loss)
