from typing import TYPE_CHECKING
import torch
if TYPE_CHECKING:
    from returnn.tensor.tensor_dict import TensorDict


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_len is not None

    targets = extern_data["targets"].raw_tensor.long()
    targets_len = extern_data["targets"].dims[1].dyn_size_ext.raw_tensor
    assert targets_len is not None

    model.train()

    log_probs, sequence_mask = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    log_probs = torch.transpose(log_probs, 1, 2)  # [B, F, T]
    sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)  # [B]

    targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
        targets, targets_len, batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = torch.nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = torch.nn.functional.cross_entropy(input=log_probs, target=targets_masked, ignore_index=-100, reduction="sum")

    loss /= torch.sum(sequence_lengths)

    import returnn.frontend as rf
    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)
