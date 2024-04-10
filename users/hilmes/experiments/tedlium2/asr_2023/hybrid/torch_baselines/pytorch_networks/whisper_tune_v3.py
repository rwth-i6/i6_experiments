"""
Like v2 but changes to grad layer choices.
"""
import torch
from torch import nn
import whisper
import returnn.frontend as rf
import torch.nn.functional as F
from typing import Optional, Iterable
from torch.onnx import export as onnx_export
from whisper.audio import N_FRAMES
from whisper.model import MultiHeadAttention, LayerNorm, Linear, Tensor, Conv1d, sinusoids


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, dropout: float = 0.0):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.dropout = nn.Dropout(p=dropout) if dropout != 0.0 else None
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        y = self.mlp_ln(x)
        y = self.mlp[0](y)
        y = self.mlp[1](y)
        if self.dropout:
            y = self.dropout(y)
        y = self.mlp[2](y)
        x = x + y
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, dropout: float
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, dropout=dropout) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

class Whisper(nn.Module):
    def __init__(self, dims: whisper.ModelDimensions, dropout: float):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dropout=dropout,
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel)

class Model(nn.Module):

    def __init__(self, epoch, step, **kwargs):
        super().__init__()

        self.whisper_name = kwargs.pop("whisper_model", "base")
        self.just_encoder = kwargs.pop("just_encoder", False)
        self.finetune_whisper = kwargs.pop("finetune_whisper", None)  # select specific layers to be finetuned
        self.split_seq = kwargs.pop("split_seq", True)
        self.dropout = kwargs.pop("dropout", 0.0)
        self.keep_layers = kwargs.pop("keep_layers", None)  # if int the last layer to be kept, if list selects right layers

        if self.just_encoder:
            with open(f"/work/asr4/hilmes/debug/whisper/{self.whisper_name}.pt", "rb") as f:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                checkpoint = torch.load(f, map_location=device)
                dims = whisper.ModelDimensions(**checkpoint["dims"])
            self.whisper = Whisper(dims, self.dropout)
            if epoch == 1 and step == 0:
                model_dict = self.whisper.state_dict()
                print(model_dict.keys())
                pretrained_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if k in model_dict}
                print(pretrained_dict.keys())
                self.whisper.load_state_dict(pretrained_dict)
        else:
            raise NotImplementedError
        if self.keep_layers is not None:
            if isinstance(self.keep_layers, list):
                layer_ls = nn.ModuleList()
                for num in self.keep_layers:
                    layer_ls.append(self.whisper.encoder.blocks[num])
                for layer, num in zip(layer_ls, self.keep_layers):
                    assert layer == self.whisper.encoder.blocks[num], "Wrong layers were picked"
                self.whisper.encoder.blocks = layer_ls
            elif isinstance(self.keep_layers, int):
                self.whisper.encoder.blocks = self.whisper.encoder.blocks[:self.keep_layers+1]
                print(self.whisper.encoder.blocks)
            else:
                raise NotImplementedError
        elif self.finetune_whisper is not None:
            for param in self.whisper.parameters():
                param.requires_grad_(False)
            if isinstance(self.finetune_whisper, list):
                for layer_num in self.finetune_whisper:
                    for name, param in self.whisper.encoder.blocks[layer_num].named_parameters():
                        param.requires_grad_(True)
            elif isinstance(self.finetune_whisper, int):
                for layer_num in range(1, self.finetune_whisper + 1):
                    for name, param in self.whisper.encoder.blocks[-layer_num].named_parameters():
                        param.requires_grad_(True)
            else:
                raise NotImplementedError
        for name, param in self.whisper.encoder.blocks.named_parameters():
            if param.requires_grad:
                print(name)

        self.upsampling_layer = torch.nn.ConvTranspose1d(
            in_channels=self.whisper.dims.n_audio_state, out_channels=512, kernel_size=5, stride=2, padding=1
        )
        self.final_layer = nn.Linear(512, 9001)

    def forward(self, audio_mel_features: torch.Tensor):
        # TODO: SpecAugment
        trans_audio_mel_features = torch.transpose(audio_mel_features, 1, 2)
        assert any(
            param.requires_grad for param in self.whisper.encoder.parameters()) or not self.finetune_whisper, "No layers are being fine-tuned"
        if self.split_seq:
            pad_widths = [(0, 0)] * trans_audio_mel_features.ndim
            pad_widths[-1] = (0, 2 * N_FRAMES - trans_audio_mel_features.shape[-1])
            padded_features = nn.functional.pad(trans_audio_mel_features, [pad for sizes in pad_widths[::-1] for pad in sizes], value=0)
            trans_audio_mel_features_1 = padded_features.index_select(
                dim=-1, index=torch.arange(end=N_FRAMES, device=trans_audio_mel_features.device)
            )
            trans_audio_mel_features_2 = padded_features.index_select(
                dim=-1, index=torch.arange(
                    start=N_FRAMES,
                    end=2*N_FRAMES,
                    device=trans_audio_mel_features.device
                )
            )
            if self.just_encoder:
                x_1: torch.Tensor = self.whisper.encoder(trans_audio_mel_features_1)
                if torch.is_nonzero(torch.count_nonzero(trans_audio_mel_features_2)):
                    x_2: torch.Tensor = self.whisper.encoder(trans_audio_mel_features_2)
            else:
                x_1: torch.Tensor = self.whisper(trans_audio_mel_features_1)
                if torch.is_nonzero(torch.count_nonzero(trans_audio_mel_features_2)):
                    x_2: torch.Tensor = self.whisper(trans_audio_mel_features_2)
            x_1 = self.upsampling_layer(x_1.transpose(1, 2)).transpose(1, 2)
            if torch.is_nonzero(torch.count_nonzero(trans_audio_mel_features_2)):
                x_2 = self.upsampling_layer(x_2.transpose(1, 2)).transpose(1, 2)
                x = torch.cat((x_1, x_2), dim=1)
            else:
                x = x_1
        else:
            pad_widths = [(0, 0)] * trans_audio_mel_features.ndim
            pad_widths[-1] = (0, 2 * N_FRAMES - trans_audio_mel_features.shape[-1])
            padded_features = nn.functional.pad(trans_audio_mel_features, [pad for sizes in pad_widths[::-1] for pad in sizes], value=0)
            if self.just_encoder:
                x: torch.Tensor = self.whisper.encoder(padded_features)
            else:
                x: torch.Tensor = self.whisper(padded_features)
            x = self.upsampling_layer(x.transpose(1, 2)).transpose(1, 2)
            
        x = x[:, :audio_mel_features.size()[1], :]
        logits = self.final_layer(x)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


def train_step(*, model: Model, extern_data, **_kwargs):
    global scripted_model
    audio_features = extern_data["data"].raw_tensor
    phonemes = extern_data["classes"].raw_tensor
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor
    log_probs, logits = model(
        audio_mel_features=audio_features,
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)
    targets_masked = targets_masked.long()

    loss = nn.functional.cross_entropy(logits, targets_masked)

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)


def export(*, model: Model, model_filename: str):
    model.export_mode = True
    dummy_data = torch.randn(1, 3500, 80, device="cpu")
    scripted_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data,))
    onnx_export(
        scripted_model,
        (dummy_data,),
        f=model_filename,
        verbose=True,
        input_names=["data"],
        output_names=["classes"],
        opset_version=14,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "classes": {0: "batch", 1: "time"},
        },
    )