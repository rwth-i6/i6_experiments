__all__ = ["Wav2Vec2HybridModel", "train_step", "export"]

import sys
import torch
import torchaudio
from torch import nn
from torch.onnx import export as onnx_export
from returnn.tensor import TensorDict

from i6_experiments.users.jxu.experiments.fairseq_multilingual.finetuning.pytorch_networks.import_fairseq_conformer_wav2vec2 import _import_wav2vec2_pretraining
import returnn.frontend as rf

class Wav2Vec2HybridModel(torch.nn.Module):
    def __init__(self, epoch=1, step=0, out_dim=9001, **kwargs):
        super().__init__()
        from fairseq.models import wav2vec

        cfg = wav2vec.Wav2Vec2Config(**kwargs["wav2vec2_args"])
        self.wav2vec_model = wav2vec.Wav2Vec2Model.build_model(cfg, task=None)
        self.wav2vec_model.remove_pretraining_modules()
        inner_dim = self.wav2vec_model.encoder.embedding_dim
        # for exactly twice the length: padding = (kernel_size - 2) / 2
        self.upsampling = torch.nn.ConvTranspose1d(
            inner_dim, inner_dim, kernel_size=6, stride=2, padding=2
        )
        self.out_proj = torch.nn.Linear(inner_dim, out_dim)
        self._onnx_export = False

    @torch.jit.unused
    def forward_wav2vec_model(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wav2vec_model(x, features_only=True)["x"]  # (B, T, F)
        return x

    def forward_wav2vec_model_onnx_export(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.wav2vec_model(x)  # (B, T, F)  # for torchaudio
        return x

    def forward(self, audio_features: torch.Tensor):
        x = torch.squeeze(
            audio_features, dim=-1
        )  # squeeze feature dim, result is (B, T)
        x = nn.functional.pad(x, (0, 240))  # pad to match alignment length
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
        self.wav2vec_model = _import_wav2vec2_pretraining(
            self.wav2vec_model
        )
        self.wav2vec_model.eval()

def train_step(*, model: Wav2Vec2HybridModel, extern_data, **_kwargs):
    global scripted_model

    data = extern_data["data"]
    audio_features = data.raw_tensor
    audio_features = audio_features.type(torch.float32)

    audio_features_len = data.dims[1].dyn_size_ext.raw_tensor

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    targets = extern_data["classes"]
    phonemes = targets.raw_tensor[indices, :]
    phonemes_len = targets.dims[1].dyn_size_ext.raw_tensor[indices]

    log_probs, logits = model(audio_features)

    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len, batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(
        targets_packed, batch_first=True, padding_value=-100
    )
    targets_masked = targets_masked.squeeze(dim=-1)

    logits = logits[:, :, 0 : targets_masked.size()[1]]
    loss = nn.functional.cross_entropy(
        logits,
        targets_masked.long(),
        reduction="sum",
    )
    num_phonemes = torch.sum(phonemes_len).item()
    rf.get_run_ctx().mark_as_loss(
        name="CE", loss=loss, custom_inv_norm_factor=num_phonemes
    )


def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    #model.prepare_for_export()
    model.cuda()
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    log_probs, logits_ce_order = model(
        audio_features=audio_features.type(torch.float32),
        #audio_features_len=audio_features_len.to("cuda"),
    )  # [B, T, F]
    rf.get_run_ctx().mark_as_output(torch.tensor(log_probs), name="log_probs")
    
def export(*, model: Wav2Vec2HybridModel, model_filename: str):
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
            "classes": {0: "batch", 1: "time"},
        },
    )
