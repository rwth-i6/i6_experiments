from dataclasses import dataclass
import time
import torch
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis

from i6_models.config import ModelConfiguration

@dataclass()
class BlstmEncoderConfig(ModelConfiguration):
    num_layers: int
    """number of bi-directional LSTM layers, minimum 2"""
    input_dimension: int
    """input dimension size"""
    hidden_dimension: int
    """hidden dimension of one direction of LSTM, the total output size is twice of this"""
    dropout: float
    """nn.LSTM supports an internal Dropout layer"""
    enforce_sorted: bool = True
    """
        keep activated for ONNX-Export, requires that the lengths are sorted decreasing from longest

        Sorting can performed using something like:    

            audio_features_len, indices = torch.sort(audio_features_len, descending=True)
            audio_features = audio_features[indices, :, :]
            labels = labels[indices, :]
            labels_len = labels_len[indices]
    """


class BlstmEncoder(torch.nn.Module):
    """
        Simple multi-layer BLSTM model including dropout, batch-first variant

        Is hardcoded to use B,T,F input

        supports: TorchScript, ONNX-export
    """

    def __init__(self, config: BlstmEncoderConfig, onnx_export: bool = False):
        """
        :param config:
        :param onnx_export: The BlstmEncoder has specific onnx-behavior
        """
        super().__init__()
        self.dropout = config.dropout
        self.enforce_sorted = config.enforce_sorted
        self.onnx_export = onnx_export
        self.blstm_stack = nn.LSTM(
            input_size=config.input_dimension,
            hidden_size=config.hidden_dimension,
            bidirectional=True,
            num_layers=config.num_layers,
            batch_first=False,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, T, input_dimension]
        :param seq_len: [B], should be on CPU for Script/Trace mode
        :return [B, T, 2 * hidden_dimension]
        """
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            # during graph mode we assume all Tensors are on the correct device,
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
        lstm_size = 1024
        target_size = 12001
        config = BlstmEncoderConfig(
            num_layers=8,
            input_dimension=50,
            hidden_dimension=1024,
            dropout=0.0,
            enforce_sorted=True
        )
        self.blstm = BlstmEncoder(config=config)
        self.final_linear = nn.Linear(2*lstm_size, target_size)

    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: torch.Tensor,
    ):
        if self.training:
            audio_features_time_masked = mask_along_axis(audio_features, mask_param=20, mask_value=0.0, axis=1)
            audio_features_time_masked_2 = mask_along_axis(audio_features_time_masked, mask_param=20, mask_value=0.0, axis=1)
            audio_features_masked = mask_along_axis(audio_features_time_masked_2, mask_param=10, mask_value=0.0, axis=2)
            audio_features_masked_2 = mask_along_axis(audio_features_masked, mask_param=10, mask_value=0.0, axis=2)
        else:
            audio_features_masked_2 = audio_features

        blstm_out = self.blstm(audio_features_masked_2, audio_features_len)
        
        logits = self.final_linear(blstm_out)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


scripted_model = None

def train_step(*, model: Model, data, run_ctx, **_kwargs):
    global scripted_model
    audio_features = data["data"]
    audio_features_len = data["data:size1"]

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    phonemes = data["classes"][indices, :]
    phonemes_len = data["classes:size1"][indices]

    # confirm scripting
    if scripted_model is None:
        scripted_model = torch.jit.script(model)

    # distributed_model = DataParallel(model)
    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len,
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(phonemes, phonemes_len.cpu(), batch_first=True, enforce_sorted=False)
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)

    run_ctx.mark_as_loss(name="CE", loss=loss)


def export(*, model: Model, model_filename: str):
    scripted_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
    dummy_data = torch.randn(1, 30, 50, device="cpu")
    dummy_data_len, _ = torch.sort(torch.randint(low=10, high=30, size=(2,), device="cpu", dtype=torch.int32), descending=True)
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
