"""
V1 for distillation
"""
from dataclasses import dataclass
import torch
from torch import nn
import whisper
import returnn.frontend as rf
import torch.nn.functional as F
from typing import Optional, Iterable, Union, Tuple
from torch.onnx import export as onnx_export
from whisper.audio import N_FRAMES
from whisper.model import MultiHeadAttention, LayerNorm, Linear, Tensor, Conv1d, sinusoids
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1
from i6_models.assemblies.conformer.conformer_v1 import (
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
    ConformerPositionwiseFeedForwardV1Config,
    ConformerConvolutionV1Config,
    ConformerMHSAV1Config,
)
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config

def _lengths_to_padding_mask(lengths: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to a pytorch MHSA compatible key mask

    :param lengths: [B]
    :return: B x T, where 0 means within sequence and 1 means outside sequence
    """
    i_ = torch.arange(x.shape[1], device=lengths.device)  # [T]
    return i_[None, :] < lengths[:, None]  # [B, T],

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

@dataclass
class WhisperDistillConfig:
     distill_scale: float
     model_name: str
     just_encoder: bool

@dataclass
class ConformerStudentConfig:
    hidden_d: int
    conv_kernel_size: int
    att_heads: int
    ff_dim: int
    spec_num_time: int
    spec_max_time: int
    spec_num_feat: int
    spec_max_feat: int
    pool_1_stride: Union[int, Tuple[int, int]]
    pool_1_kernel_size: Union[int, Tuple[int, int]]
    pool_1_padding: Optional[int]
    pool_2_stride: Optional[int]
    pool_2_kernel_size: Union[int, Tuple[int, int]]
    pool_2_padding: Optional[int]
    num_layers: int
    upsample_kernel: int
    upsample_stride: int
    upsample_padding: int
    upsample_out_padding: int

class Model(nn.Module):

    def __init__(self, epoch, step, whisper_dict, conformer_dict, **kwargs):
        super().__init__()

        self.whisper_cfg = WhisperDistillConfig(**whisper_dict)
        self.conformer_cfg = ConformerStudentConfig(**conformer_dict)
        self.distill_scale = self.whisper_cfg.distill_scale
        target_size = 9001

        conv_cfg = ConformerConvolutionV1Config(
            channels=self.conformer_cfg.hidden_d,
            kernel_size=self.conformer_cfg.conv_kernel_size,
            dropout=0.2,
            activation=nn.SiLU(),
            norm=LayerNormNC(self.conformer_cfg.hidden_d),
        )
        mhsa_cfg = ConformerMHSAV1Config(
            input_dim=self.conformer_cfg.hidden_d,
            num_att_heads=self.conformer_cfg.att_heads,
            att_weights_dropout=0.2,
            dropout=0.2
        )
        ff_cfg = ConformerPositionwiseFeedForwardV1Config(
            input_dim=self.conformer_cfg.hidden_d,
            hidden_dim=self.conformer_cfg.ff_dim,
            activation=nn.SiLU(),
            dropout=0.2,
        )
        block_cfg = ConformerBlockV1Config(ff_cfg=ff_cfg, mhsa_cfg=mhsa_cfg, conv_cfg=conv_cfg)
        frontend_cfg = VGG4LayerActFrontendV1Config(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 1),
            pool1_kernel_size=self.conformer_cfg.pool_1_kernel_size,
            pool1_stride=self.conformer_cfg.pool_1_stride,
            activation=nn.ReLU(),
            conv_padding=None,
            pool1_padding=self.conformer_cfg.pool_1_padding,
            out_features=self.conformer_cfg.hidden_d,
            pool2_kernel_size=self.conformer_cfg.pool_2_kernel_size,
            pool2_stride=self.conformer_cfg.pool_2_stride,
            pool2_padding=self.conformer_cfg.pool_2_padding,
        )

        frontend = ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_cfg)
        conformer_cfg = ConformerEncoderV1Config(num_layers=self.conformer_cfg.num_layers, frontend=frontend, block_cfg=block_cfg)
        self.conformer = ConformerEncoderV1(cfg=conformer_cfg)

        self.upsample_conv = torch.nn.ConvTranspose1d(
            in_channels=self.conformer_cfg.hidden_d,
            out_channels=self.conformer_cfg.hidden_d,
            kernel_size=self.conformer_cfg.upsample_kernel,
            stride=self.conformer_cfg.upsample_stride,
            padding=self.conformer_cfg.upsample_padding,
            output_padding=self.conformer_cfg.upsample_out_padding
        )
        # self.initial_linear = nn.Linear(80, conformer_size)
        self.final_linear = nn.Linear(self.conformer_cfg.hidden_d, target_size)
        self.export_mode = False
        self.prior_comp = False
        assert len(kwargs) in [0, 1]  # for some reason there is some random arg always here
        run_ctx = rf.get_run_ctx()
        if not run_ctx.stage == "train_step" and not run_ctx.stage == "init" or kwargs.pop("export", None) is True:
            import logging
            logging.warning("Whisper not loaded")
        elif self.training:
            self.hubert = None
            if self.whisper_cfg.just_encoder:
                with open(f"/work/asr4/hilmes/debug/whisper/{self.whisper_cfg.model_name}.pt", "rb") as f:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    checkpoint = torch.load(f, map_location=device)
                    dims = whisper.ModelDimensions(**checkpoint["dims"])
                self.whisper = Whisper(dims, 0)
                model_dict = self.whisper.state_dict()
                print(model_dict.keys())
                pretrained_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if k in model_dict}
                print(pretrained_dict.keys())
                self.whisper.load_state_dict(pretrained_dict)
            else:
                raise NotImplementedError
            for param in self.whisper.parameters():
                param.requires_grad_(False)
        else:
            print("Whisper not loaded")
            self.hubert = None
        self.upsampling_layer = torch.nn.ConvTranspose1d(
            in_channels=self.whisper.dims.n_audio_state, out_channels=512, kernel_size=5, stride=2, padding=1
        )


    def forward(self, audio_features: torch.Tensor, audio_features_len: torch.Tensor):

        run_ctx = rf.get_run_ctx()
        # Whisper teacher:
        if self.training or run_ctx.stage == "train_step":
            trans_audio_mel_features = torch.transpose(audio_features, 1, 2)
            pad_widths = [(0, 0)] * trans_audio_mel_features.ndim
            if not self.training:
                pad_widths[-1] = (0, 2 * N_FRAMES - trans_audio_mel_features.shape[-1])
            else:
                pad_widths[-1] = (0, N_FRAMES - trans_audio_mel_features.shape[-1])
            padded_features = nn.functional.pad(trans_audio_mel_features,
                                                [pad for sizes in pad_widths[::-1] for pad in sizes], value=0)
            if not self.training:
                trans_audio_mel_features_1 = padded_features.index_select(
                    dim=-1, index=torch.arange(end=N_FRAMES, device=trans_audio_mel_features.device)
                )
                trans_audio_mel_features_2 = padded_features.index_select(
                    dim=-1, index=torch.arange(
                        start=N_FRAMES,
                        end=2 * N_FRAMES,
                        device=trans_audio_mel_features.device
                    )
                )
            else:
                trans_audio_mel_features_1 = padded_features
            x_1: torch.Tensor = self.whisper.encoder(trans_audio_mel_features_1)
            if not self.training:
                x_2: torch.Tensor = self.whisper.encoder(trans_audio_mel_features_2)
                x = torch.cat((x_1, x_2), dim=1)
            else:
                x = x_1
            x = self.upsampling_layer(x.transpose(1, 2)).transpose(1, 2)
            teacher_features = x[:, :audio_features.size()[1], :]
        else:
            teacher_features = None

        mask = _lengths_to_padding_mask(audio_features_len, audio_features)
        conformer_out, _ = self.conformer(audio_features, mask)

        upsampled = self.upsample_conv(conformer_out.transpose(1, 2)).transpose(1, 2)  # final upsampled [B, T, F]
        upsampled = upsampled[:, 0: audio_features.size()[1], :]
        student_features = upsampled
        upsampled_dropped = nn.functional.dropout(upsampled, p=0.2, training=self.training)

        final_out = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(final_out, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(final_out, dim=2)

        return log_probs, logits_ce_order, teacher_features, student_features


def train_step(*, model: Model, extern_data, **_kwargs):
    global scripted_model
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    phonemes = extern_data["classes"].raw_tensor
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor

    log_probs, logits_ce_order, teacher_features, student_features = model(
        audio_features=audio_features, audio_features_len=audio_features_len.to(device="cuda")
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)
    targets_masked = targets_masked.long()

    loss_ce = nn.functional.cross_entropy(logits_ce_order, targets_masked)
    loss_features = nn.functional.l1_loss(student_features, teacher_features, reduction="mean")

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss_ce)
    rf.get_run_ctx().mark_as_loss(name="L1 Dist", loss=loss_features, scale=model.distill_scale)


def export(*, model: Model, model_filename: str):
    model.export_mode = True
    dummy_data = torch.randn(1, 30, 80, device="cpu")
    dummy_data_len = torch.ones((1,), dtype=torch.int32) * 30
    #scripted_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len))
    onnx_export(
        model.eval(),
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
