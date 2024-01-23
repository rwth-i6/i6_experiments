import torch
import torchaudio
from torch import nn
from torch.onnx import export as onnx_export

import returnn.frontend as rf


class Model(torch.nn.Module):
    def __init__(self, out_dim=12001, **kwargs):
        super().__init__()
        from fairseq.models import wav2vec
        cfg = wav2vec.Wav2Vec2Config()
        self.wav2vec_model = wav2vec.Wav2Vec2Model.build_model(cfg, task=None)
        self.wav2vec_model.remove_pretraining_modules()
        inner_dim = self.wav2vec_model.encoder.embedding_dim
        # for exactly twice the length: padding = (kernel_size - 2) / 2
        self.upsampling = torch.nn.ConvTranspose1d(inner_dim, inner_dim, kernel_size=6, stride=2, padding=2)
        self.out_proj = torch.nn.Linear(inner_dim, out_dim)
        self._onnx_export = False

    @torch.jit.ignore
    def forward_wav2vec_model(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wav2vec_model(x, features_only=True, mask=False)["x"]  # (B, T, F)
        return x

    def forward_wav2vec_model_onnx_export(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.wav2vec_model(x)  # (B, T, F)  # for torchaudio
        return x

    def forward(self, audio_features: torch.Tensor):
        x = torch.squeeze(audio_features, dim=-1)  # squeeze feature dim, result is (B, T)
        x = nn.functional.pad(x, (80, 80))  # pad to match alignment length
        if self._onnx_export:
            x = self.forward_wav2vec_model_onnx_export(x)
        else:
            x = self.forward_wav2vec_model(x)
        x = torch.swapaxes(x, 1, 2)  # (B, F, T)
        x = self.upsampling(x)  # (B, F, T')
        x = torch.swapaxes(x, 1, 2)  # (B, T', F)
        x = self.out_proj(x)  # (B, T', F')
        logits_rasr_order = x  # RASR expects [B, T, F]
        logits_ce_order = torch.permute(x, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits_rasr_order, dim=2)
        return log_probs, logits_ce_order

    def prepare_for_export(self):
        self.eval()
        self._onnx_export = True
        self.wav2vec_model = torchaudio.models.wav2vec2.utils.import_fairseq_model(self.wav2vec_model)


scripted_model = None


def train_step(*, model: Model, extern_data, **_kwargs):
    global scripted_model

    data = extern_data["data"]
    audio_features = data.raw_tensor
    audio_features_len = data.dims[1].dyn_size_ext.raw_tensor

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    targets = extern_data["classes"]
    phonemes = targets.raw_tensor[indices, :]
    phonemes_len = targets.dims[1].dyn_size_ext.raw_tensor[indices]

    # if scripted_model is None:
    #     scripted_model = torch.jit.script(model)

    log_probs, logits = model(audio_features)

    targets_packed = nn.utils.rnn.pack_padded_sequence(phonemes, phonemes_len, batch_first=True, enforce_sorted=False)
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)
    targets_masked = targets_masked.squeeze(dim=-1)

    loss = nn.functional.cross_entropy(logits, targets_masked.long())

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)


def export_script(*, model: Model, model_filename: str):
    model.prepare_for_export()
    dummy_data = torch.randn(1, 32 * 160, 1, device="cpu")
    # import ipdb
    # ipdb.set_trace()
    scripted_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
    onnx_export(
        scripted_model,
        (dummy_data,),
        f=model_filename,
        verbose=True,
        input_names=["data"],
        output_names=["classes"],
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "classes": {0: "batch", 1: "time"}
        }
    )


def export_trace(*, model: Model, model_filename: str):
    # with the hack for multi_head_attention_forward, this runs without error but does not output an exported file.
    # maybe related to: https://github.com/pyg-team/pytorch_geometric/issues/5656 (python list instead of ModuleList
    dummy_data = torch.randn(1, 30 * 160, 1, device="cpu")  # (B, T, F)
    dummy_data_len, _ = torch.sort(
        torch.randint(low=10 * 160, high=30 * 160, size=(1,), device="cpu", dtype=torch.int32), descending=True)
    scripted_model = torch.jit.optimize_for_inference(
        torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len)))
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


def export(*args, **kwargs):
    return export_script(*args, **kwargs)
