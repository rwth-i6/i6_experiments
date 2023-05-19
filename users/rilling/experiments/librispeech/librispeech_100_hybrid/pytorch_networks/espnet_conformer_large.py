import os
from random import random
import torch
import time
from typing import Dict, Optional
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis

from returnn.torch.engine import Engine as TorchEngine
from returnn.util import NumbersDict
from returnn.log import log
from returnn.torch.context import get_run_ctx

from espnet.nets.pytorch_backend.e2e_asr_conformer import Encoder


def lengths_to_mask(lengths, max_len):
    return torch.arange(max_len).expand(lengths.size()[0], max_len) < lengths.unsqueeze(1)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        target_size=12001
        attention_dim = 512
        self.espnet_conformer = Encoder(idim=50, num_blocks=12, attention_dim=attention_dim, attention_heads=8, input_layer="linear")
        self.final_linear = nn.Linear(attention_dim, target_size)

    def forward(
            self,
            audio_features: torch.Tensor,  # [B T D]
            audio_features_len: torch.Tensor,  # [B]
    ):

        if self.training:
            audio_features_time_masked = mask_along_axis(audio_features, mask_param=20, mask_value=0.0, axis=1)
            audio_features_time_masked_2 = mask_along_axis(audio_features_time_masked, mask_param=20, mask_value=0.0, axis=1)
            audio_features_masked = mask_along_axis(audio_features_time_masked_2, mask_param=10, mask_value=0.0, axis=2)
            audio_features_masked_2 = mask_along_axis(audio_features_masked, mask_param=10, mask_value=0.0, axis=2)
        else:
            audio_features_masked_2 = audio_features

        run_ctx = get_run_ctx()
        mask = lengths_to_mask(audio_features_len, max_len=audio_features.size()[1])
        mask = torch.unsqueeze(mask, dim=1).to(run_ctx.device)


        conformer_out, mask = self.espnet_conformer(audio_features_masked_2, mask)

        logits = self.final_linear(conformer_out)  # [B, T, F]
        logits_ce_order  = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


scripted_model = None

def train_step(*, model: Model, data, run_ctx, **_kwargs):
    global scripted_model
    audio_features = data["data"]
    audio_features_len = data["data:seq_len"]

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    phonemes = data["classes"][indices, :]
    phonemes_len = data["classes:seq_len"][indices]

    #if scripted_model is None:
    #    # check exportability
    #    tmp_filename = "/var/tmp/onnx_export/export.onnx"
    #    os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)
    #    scripted_model = export(model=model, model_filename=tmp_filename)
    #    os.unlink(tmp_filename)

    # distributed_model = DataParallel(model)
    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len,
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(phonemes, phonemes_len, batch_first=True, enforce_sorted=False)
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)

    run_ctx.mark_as_loss(name="ce", loss=loss)


def export(*, model: Model, model_filename: str):
    from returnn.torch import context
    context._run_ctx = context.RunCtx(stage="forward_step", device="cpu")
    dummy_data = torch.randn(1, 30, 50, device="cpu") * 10
    #dummy_data_len, _ = torch.sort(torch.randint(low=10, high=30, size=(1,), device="cpu", dtype=torch.int32), descending=True)
    dummy_data_len = torch.ones((1,)) * 25
    dummy_data_len2 = torch.ones((1,)) * 25
    scripted_model = torch.jit.optimize_for_inference(torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len)))
    l1 = scripted_model(dummy_data, dummy_data_len)
    l2 = scripted_model(dummy_data, dummy_data_len2)
    print(l1)
    print(l2)
    assert False
    onnx_export(
        scripted_model,
        (dummy_data, dummy_data_len),
        f=model_filename,
        verbose=True,
        input_names=["data", "data_len"],
        output_names=["classes"],
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data_len": {0: "batch"},
            "classes": {0: "batch", 1: "time"}
        }
    )
    return scripted_model

