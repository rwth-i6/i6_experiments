import torch
from returnn.tensor.tensor_dict import TensorDict


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features_prim = extern_data["features_primary"].raw_tensor
    audio_features_prim_len = extern_data["features_primary"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_prim is not None
    assert audio_features_prim_len is not None

    audio_features_len, indices = torch.sort(audio_features_prim_len, descending=True)
    audio_features_prim = audio_features_prim[indices, :, :]

    try:
        audio_features_sec = extern_data["features_secondary"].raw_tensor
        audio_features_sec_len = extern_data["features_secondary"].dims[1].dyn_size_ext.raw_tensor
        assert audio_features_sec is not None
        assert audio_features_sec_len is not None
        audio_features_sec = audio_features_sec[indices, :, :]
    except KeyError:
        audio_features_sec = None

    try:
        audio_features_mix = extern_data["features_mix"].raw_tensor
        audio_features_mix_len = extern_data["features_mix"].dims[1].dyn_size_ext.raw_tensor
        assert audio_features_mix is not None
        assert audio_features_mix_len is not None
        audio_features_mix = audio_features_mix[indices, :, :]
    except KeyError:
        audio_features_mix = None

    classes = extern_data["classes"].raw_tensor.long()
    classes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor
    assert classes_len is not None

    classes_len = classes_len[indices]
    classes = classes[indices, :]
    classes = torch.squeeze(classes, 2)

    model.train()

    log_probs, sequence_lengths = model(
        primary_audio_features=audio_features_prim,
        audio_features_len=audio_features_len.to("cuda"),
        secondary_audio_features=audio_features_sec,
        mix_audio_features=audio_features_mix,
    )

    predicted_classes = torch.argmax(log_probs, dim=2)
    frame_error = 1 - torch.sum(torch.eq(classes, predicted_classes)) / sum(classes_len)

    classes_packed = torch.nn.utils.rnn.pack_padded_sequence(
        classes, classes_len, batch_first=True, enforce_sorted=False
    )
    classes_masked, _ = torch.nn.utils.rnn.pad_packed_sequence(classes_packed, batch_first=True, padding_value=-100)

    log_probs = torch.transpose(log_probs, 1, 2)  # [B, F, T]

    loss = torch.nn.functional.cross_entropy(input=log_probs, target=classes_masked, ignore_index=-100, reduction="sum")
    loss /= torch.sum(sequence_lengths)

    import returnn.frontend as rf

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)
    rf.get_run_ctx().mark_as_loss(name="FER", loss=frame_error, as_error=True)
