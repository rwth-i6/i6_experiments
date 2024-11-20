from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch
from torch import nn


def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    log_probs, _ = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to(audio_features.device),
    )  # [B, T, F]
    features_len = audio_features_len.to(dtype=torch.int32)
    rf.get_run_ctx().expected_outputs["log_probs"].dims[1].dyn_size_ext.raw_tensor = features_len

    rf.get_run_ctx().mark_as_output(log_probs, name="log_probs")


def loss_forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )  # [B, T, F]
    phonemes = extern_data["classes"].raw_tensor[indices, :].long()
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor[indices]
    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)
    loss = torch.unsqueeze(torch.unsqueeze(loss, dim=0), dim=0)
    rf.get_run_ctx().mark_as_output(loss, "ce_score")
