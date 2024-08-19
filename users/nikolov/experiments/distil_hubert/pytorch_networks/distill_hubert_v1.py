import torch
from torch import nn
import returnn.frontend as rf
from torch.onnx import export as onnx_export
from transformers import HubertConfig, HubertModel
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1
from i6_models.assemblies.conformer.conformer_v1 import (
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
    ConformerPositionwiseFeedForwardV1Config,
    ConformerConvolutionV1Config,
    ConformerMHSAV1Config,
)
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config, RasrCompatibleLogMelFeatureExtractionV1Config, RasrCompatibleLogMelFeatureExtractionV1

def _lengths_to_padding_mask(lengths: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to a pytorch MHSA compatible key mask

    :param lengths: [B]
    :return: B x T, where 0 means within sequence and 1 means outside sequence
    """
    i_ = torch.arange(x.shape[1], device=lengths.device)  # [T]
    return i_[None, :] < lengths[:, None]  # [B, T],

@dataclass
class HubertTeacherConfig:
    model_name: str
    distill_scale: float

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
    dropout: float
    feat_extr: bool

class Model(nn.Module):

    def __init__(self, epoch, step, hubert_dict, conformer_dict, **kwargs):
        super().__init__()
        run_ctx = rf.get_run_ctx()
        self.hubert_cfg = HubertTeacherConfig(**hubert_dict)
        self.conformer_cfg = ConformerStudentConfig(**conformer_dict)
        self.distill_scale = self.hubert_cfg.distill_scale
        self.config = HubertConfig.from_pretrained(f"facebook/hubert-{self.hubert_cfg.model_name}",  cache_dir='/work/asr4/hilmes/debug/whisper/hubert_for_ctc')
        if not run_ctx.stage == "train_step" and not run_ctx.stage == "init" or True:
            import logging
            logging.warning("Hubert not loaded")
            self.hubert = None
        elif self.training:
            # self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft",
            #     cache_dir='/work/asr4/hilmes/debug/whisper/hubert_processor')
            print("Loading Hubert Model Weights")
            self.hubert = HubertModel.from_pretrained(f"facebook/hubert-{self.hubert_cfg.model_name}",
                cache_dir='/work/asr4/hilmes/debug/whisper/hubert_for_ctc')
        # elif self.training:
        #     self.hubert: HubertModel = HubertModel(
        #         HubertConfig.from_pretrained(f"facebook/hubert-{self.hubert_cfg.model_name}",
        #                                      cache_dir="/work/asr4/hilmes/debug/whisper/transformers/"))
        else:
            print("Hubert not loaded")
            self.hubert = None
        if self.hubert:
            for param in self.hubert.parameters():
                param.requires_grad_(False)

        self.final_linear = nn.Linear(self.conformer_cfg.hidden_d, 9001)
        if len(kwargs) >= 2:
            assert False, f"You did not use all kwargs: {kwargs}"
        elif len(kwargs) == 1:
            assert "random" in list(kwargs.keys())[0], "This must only be RETURNN random arg"
        # else len == 0

        if self.conformer_cfg.feat_extr is True:
            self.fe_cfg = RasrCompatibleLogMelFeatureExtractionV1Config(
                            sample_rate=16000,
                            win_size=0.025,
                            hop_size=0.01,
                            min_amp=1e-10,
                            num_filters=80,
                        )
            self.feature_extraction = RasrCompatibleLogMelFeatureExtractionV1(self.fe_cfg)

        conv_cfg = ConformerConvolutionV1Config(
            channels=self.conformer_cfg.hidden_d,
            kernel_size=self.conformer_cfg.conv_kernel_size,
            dropout=self.conformer_cfg.dropout,
            activation=nn.SiLU(),
            norm=LayerNormNC(self.conformer_cfg.hidden_d),
        )
        mhsa_cfg = ConformerMHSAV1Config(
            input_dim=self.conformer_cfg.hidden_d,
            num_att_heads=self.conformer_cfg.att_heads,
            att_weights_dropout=self.conformer_cfg.dropout,
            dropout=self.conformer_cfg.dropout
        )
        ff_cfg = ConformerPositionwiseFeedForwardV1Config(
            input_dim=self.conformer_cfg.hidden_d,
            hidden_dim=self.conformer_cfg.ff_dim,
            activation=nn.SiLU(),
            dropout=self.conformer_cfg.dropout,
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
        conformer_cfg = ConformerEncoderV1Config(num_layers=self.conformer_cfg.num_layers, frontend=frontend,
                                                 block_cfg=block_cfg)
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

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):

        run_ctx = rf.get_run_ctx()
        if self.conformer_cfg.feat_extr is True and (self.training or run_ctx.stage == "train_step"):
            squeezed_features = torch.squeeze(raw_audio, dim=-1)
            audio_features, audio_features_len = self.feature_extraction(raw_audio=squeezed_features, length=raw_audio_len)
        else:
            # for export / forward we take external features
            audio_features = raw_audio
            audio_features_len = raw_audio_len

        # Hubert teacher:
        if (self.training or run_ctx.stage == "train_step") and raw_audio is not None and raw_audio_len is not None and False:
            assert False, "Since this is just testing this should not happen, if entering here is correct please just delete this line"
            hubert_outputs = self.hubert(input_values=squeezed_features)
            audio_features_size = self.hubert._get_feat_extract_output_lengths(raw_audio_len).to(dtype=torch.int64)
            encoder_output = hubert_outputs.last_hidden_state
            upsampled = self.upsample_conv(encoder_output.transpose(1, 2)).transpose(1, 2)  # final upsampled [B, T, F]
            teacher_features = upsampled[:, :audio_features.size()[1], :]
        else:
            teacher_features = None
        if self.training:
            audio_features_masked = specaugment_v1_by_length(
                audio_features,
                time_min_num_masks=2,
                time_max_mask_per_n_frames=self.conformer_cfg.spec_num_time,
                time_mask_max_size=self.conformer_cfg.spec_max_time,
                freq_min_num_masks=2,
                freq_max_num_masks=self.conformer_cfg.spec_num_feat,
                freq_mask_max_size=self.conformer_cfg.spec_max_feat
            )
        else:
            audio_features_masked = audio_features

        mask = _lengths_to_padding_mask(audio_features_len, audio_features_masked)
        conformer_out, _ = self.conformer(audio_features_masked, mask)

        upsampled = self.upsample_conv(conformer_out.transpose(1, 2)).transpose(1, 2)  # final upsampled [B, T, F]
        print(upsampled.shape, audio_features.shape, audio_features_len, conformer_out.shape)
        upsampled = upsampled[:, 0: audio_features.size()[1]+1, :]
        student_features = upsampled
        upsampled_dropped = nn.functional.dropout(student_features, p=self.conformer_cfg.dropout, training=self.training)

        final_out = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(final_out, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(final_out, dim=2)

        if teacher_features is not None:
            return log_probs, logits_ce_order, teacher_features, student_features, audio_features_len
        elif self.training or run_ctx.stage == "train_step":
            return log_probs, logits_ce_order, None, None, audio_features_len
        else:
            return log_probs, logits_ce_order, audio_features_len


def train_step(*, model: Model, extern_data, **_kwargs):
    audio_raw = extern_data["data_raw"].raw_tensor
    audio_raw_len = extern_data["data_raw"].dims[1].dyn_size_ext.raw_tensor

    phonemes = extern_data["classes"].raw_tensor
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor
    log_probs, logits_ce_order, teacher_features, student_features, audio_features_len = model(
        raw_audio=audio_raw,
        raw_audio_len=audio_raw_len.to("cuda"),
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)
    targets_masked = targets_masked.long()
    print(f"{logits_ce_order.shape=}, {targets_masked.shape=}")
    loss_ce = nn.functional.cross_entropy(logits_ce_order, targets_masked)
    #loss_features = nn.functional.l1_loss(student_features, teacher_features, reduction="mean")

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss_ce)
    # TODO: KL div loss
    # TODO: look if Hubert model has a softmax somewhere
    #rf.get_run_ctx().mark_as_loss(name="L1 Dist", loss=loss_features, scale=model.distill_scale)
