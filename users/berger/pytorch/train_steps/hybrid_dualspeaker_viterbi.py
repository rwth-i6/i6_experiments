from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features_0 = extern_data["data_0"].raw_tensor
    audio_features_0_len = extern_data["data_0"].dims[1].dyn_size_ext.raw_tensor
    audio_features_1 = extern_data["data_1"].raw_tensor
    audio_features_1_len = extern_data["data_1"].dims[1].dyn_size_ext.raw_tensor
    audio_features_mix = extern_data["data_mix"].raw_tensor
    audio_features_mix_len = extern_data["data_mix"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_0_len is not None
    assert audio_features_1_len is not None
    assert audio_features_mix_len is not None
    assert audio_features_0_len == audio_features_1_len == audio_features_mix_len

    audio_features_len, indices = torch.sort(audio_features_0_len, descending=True)
    audio_features_0 = audio_features_0[indices, :, :]
    audio_features_1 = audio_features_1[indices, :, :]
    audio_features_mix = audio_features_mix[indices, :, :]

    targets_0 = extern_data["targets_0"].raw_tensor.long()
    targets_0_len = extern_data["targets_0"].dims[1].dyn_size_ext.raw_tensor
    targets_1 = extern_data["targets_1"].raw_tensor.long()
    targets_1_len = extern_data["targets_1"].dims[1].dyn_size_ext.raw_tensor
    assert targets_0_len is not None
    assert targets_1_len is not None

    targets_0_len = targets_0_len[indices, :, :]
    targets_1_len = targets_1_len[indices, :, :]
    assert targets_0_len == targets_1_len == audio_features_len

    targets_0 = targets_0[indices, :, :]
    targets_1 = targets_1[indices, :, :]

    model.train()

    log_probs_0, log_probs_1, sequence_mask = model(
        sep_0_audio_features=audio_features_0,
        sep_1_audio_features=audio_features_1,
        mix_audio_features=audio_features_mix,
        audio_features_len=audio_features_len.to("cuda"),
    )

    log_probs_0 = torch.transpose(log_probs_0, 1, 2)  # [B, F, T]
    log_probs_1 = torch.transpose(log_probs_1, 1, 2)  # [B, F, T]

    sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)  # [B]

    targets_packed_0 = torch.nn.utils.rnn.pack_padded_sequence(
        targets_0, targets_0_len, batch_first=True, enforce_sorted=False
    )
    targets_packed_1 = torch.nn.utils.rnn.pack_padded_sequence(
        targets_1, targets_1_len, batch_first=True, enforce_sorted=False
    )

    targets_masked_0, _ = torch.nn.utils.rnn.pad_packed_sequence(
        targets_packed_0, batch_first=True, padding_value=-100
    )
    targets_masked_1, _ = torch.nn.utils.rnn.pad_packed_sequence(
        targets_packed_1, batch_first=True, padding_value=-100
    )

    loss_0 = torch.nn.functional.cross_entropy(
        input=log_probs_0, target=targets_masked_0, ignore_index=-100, reduction="sum"
    )

    loss_1 = torch.nn.functional.cross_entropy(
        input=log_probs_1, target=targets_masked_1, ignore_index=-100, reduction="sum"
    )

    loss_0 /= torch.sum(sequence_lengths)
    loss_1 /= torch.sum(sequence_lengths)

    rf.get_run_ctx().mark_as_loss(name="CE_0", loss=loss_0)
    rf.get_run_ctx().mark_as_loss(name="CE_1", loss=loss_1)
