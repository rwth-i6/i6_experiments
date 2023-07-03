from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis

from dataclasses import dataclass
import torch
from torch import nn

import returnn.frontend as rf
from i6_models.config import ModelConfiguration


@dataclass()
class BlstmEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: number of bi-directional LSTM layers, minimum 2
        input_dim: input dimension size
        hidden_dim: hidden dimension of one direction of LSTM, the total output size is twice of this
        dropout: nn.LSTM supports internal Dropout applied between each layer of BLSTM (but not on input/output)
        enforce_sorted: keep activated for ONNX-Export, requires that the lengths are sorted decreasing from longest

            Sorting can for example be performed independent of the ONNX export in e.g. train_step:

                audio_features_len, indices = torch.sort(audio_features_len, descending=True)
                audio_features = audio_features[indices, :, :]
                labels = labels[indices, :]
                labels_len = labels_len[indices]
    """

    num_layers: int
    input_dim: int
    hidden_dim: int
    dropout: float
    enforce_sorted: bool = True


class BlstmEncoderV1(torch.nn.Module):
    """
    Simple multi-layer BLSTM model including dropout, batch-first variant,
    hardcoded to use B,T,F input

    supports: TorchScript, ONNX-export
    """

    def __init__(self, config: BlstmEncoderV1Config):
        """
        :param config: configuration object
        """
        super().__init__()
        self.dropout = config.dropout
        self.enforce_sorted = config.enforce_sorted
        self.blstm_stack = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            bidirectional=True,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, T, input_dim]
        :param seq_len: [B], should be on CPU for Script/Trace mode
        :return [B, T, 2 * hidden_dim]
        """
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            # during graph mode we have to assume all Tensors are on the correct device,
            # otherwise move lengths to the CPU if they are on GPU
            if seq_len.get_device() >= 0:
                seq_len = seq_len.cpu()

        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=seq_len,
            enforce_sorted=self.enforce_sorted,
            batch_first=True,
        )
        blstm_out, _ = self.blstm_stack(blstm_packed_in)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(blstm_out, padding_value=0.0, batch_first=True)

        return blstm_out


class Model(torch.nn.Module):
    def __init__(self, epoch, step, **kwargs):
        super().__init__()
        lstm_size = 512
        target_size = 9001
        blstm_cfg = BlstmEncoderV1Config(num_layers=6, input_dim=80, hidden_dim=lstm_size, dropout=0.0)
        self.blstm_stack = BlstmEncoderV1(config=blstm_cfg)
        self.final_linear = nn.Linear(2 * lstm_size, target_size)
        self.lstm_size = lstm_size

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_len: torch.Tensor,
    ):
        if self.training:
            audio_features_time_masked = mask_along_axis(audio_features, mask_param=20, mask_value=0.0, axis=1)
            audio_features_time_masked_2 = mask_along_axis(
                audio_features_time_masked, mask_param=20, mask_value=0.0, axis=1
            )
            audio_features_time_masked_3 = mask_along_axis(
                audio_features_time_masked_2, mask_param=20, mask_value=0.0, axis=1
            )
            audio_features_masked = mask_along_axis(audio_features_time_masked_3, mask_param=10, mask_value=0.0, axis=2)
            audio_features_masked_2 = mask_along_axis(audio_features_masked, mask_param=10, mask_value=0.0, axis=2)
        else:
            audio_features_masked_2 = audio_features

        # blstm_in = torch.swapaxes(audio_features_masked_2, 0, 1)  # [B, T, F] -> [T, B, F]
        blstm_in = audio_features_masked_2
        blstm_out = self.blstm_stack(blstm_in, audio_features_len)

        logits = self.final_linear(blstm_out)  # [B, T, F]
        logits_rasr_order = logits  # RASR expects [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits_rasr_order, dim=2)

        return log_probs, logits_ce_order


scripted_model = None


def train_step(*, model: Model, extern_data, **_kwargs):
    global scripted_model
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    phonemes = extern_data["classes"].raw_tensor[indices, :].long()
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor[indices]

    if scripted_model is None:
        scripted_model = torch.jit.script(model)

    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len,
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)


def export(*, model: Model, model_filename: str):
    model.export_mode = True
    dummy_data = torch.randn(1, 30, 80, device="cpu")
    dummy_data_len = torch.ones((1,), dtype=torch.int32) * 30
    scripted_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len))
    onnx_export(
        scripted_model,
        (dummy_data, dummy_data_len),
        f=model_filename,
        verbose=True,
        input_names=["data", "data_len"],
        output_names=["classes"],
        opset_version=14,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data_len": {0: "batch"},
            "classes": {0: "batch", 1: "time"},
        },
    )
