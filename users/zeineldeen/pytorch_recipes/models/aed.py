from typing import Tuple

import torch
from torch import nn

from i6_models.config import ModuleFactoryV1
from i6_models.assemblies.conformer import (
    ConformerEncoderV1,
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
)
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import (
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.decoder.attention import (
    AttentionLSTMDecoderV1,
    AttentionLSTMDecoderV1Config,
    AdditiveAttentionConfig,
)
from i6_models.primitives.feature_extraction import (
    LogMelFeatureExtractionV1,
    LogMelFeatureExtractionV1Config,
)
from i6_experiments.users.zeineldeen.pytorch_recipes.backend.conformer.frontend import (
    ConformerConv2dFrontend,
    ConformerConv2dFrontendConfig,
)


# I get an error when using dataclasses so using standard class here
class ConformerAEDModelConfig:
    def __init__(
        self,
        feat_extraction_cfg: LogMelFeatureExtractionV1Config,
        encoder_cfg: ConformerEncoderV1Config,
        decoder_cfg: AttentionLSTMDecoderV1Config,
    ):
        self.feat_extraction_cfg = feat_extraction_cfg
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg


class ConformerAEDModel(nn.Module):
    """
    Conformer Encoder-Decoder Attention model for ASR
    """

    def __init__(self, cfg: ConformerAEDModelConfig):
        super().__init__()

        self.feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feat_extraction_cfg)
        self.encoder = ConformerEncoderV1(cfg=cfg.encoder_cfg)
        self.decoder = AttentionLSTMDecoderV1(cfg=cfg.decoder_cfg)

    def encode(
        self, raw_audio_features: torch.Tensor, raw_audio_features_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run encoder
        """
        audio_features = raw_audio_features.squeeze(-1)  # [B,T]
        with torch.no_grad():
            audio_features, audio_features_lens = self.feat_extraction(
                audio_features, raw_audio_features_lens.cuda()
            )  # [B,T,F]

        time_arange = torch.arange(audio_features.size(1), device="cuda")  # [0, ..., T-1]
        time_mask = torch.less(time_arange[None, :], audio_features_lens[:, None])  # [B,T]

        encoder_outputs, _ = self.encoder(audio_features, time_mask)
        return encoder_outputs, audio_features_lens

    def decode(
        self, encoder_outputs: torch.Tensor, bpe_labels: torch.Tensor, audio_features_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        Run decoder
        """
        decoder_logits = self.decoder(encoder_outputs, bpe_labels, audio_features_lens)
        return decoder_logits

    def forward(
        self,
        raw_audio_features: torch.Tensor,
        raw_audio_features_lens: torch.Tensor,
        bpe_labels: torch.Tensor,
    ):
        """
        :param raw_audio_features: audio raw samples of shape [B,T,F=1]
        :param raw_audio_features_lens: audio sequence length of shape [B]
        :param bpe_labels: bpe targets of shape [B,N]
        :return:
        """
        encoder_outputs, enc_lens = self.encode(raw_audio_features, raw_audio_features_lens)
        decoder_logits = self.decode(encoder_outputs, bpe_labels, enc_lens)
        return decoder_logits


def create_model():
    feat_extraction_cfg = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.05,
        hop_size=0.0125,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=True,
    )
    conformer_conv2d_frontend_cfg = ConformerConv2dFrontendConfig(strides=[3, 2], input_dim=80, output_dim=512)
    conformer_encoder_cfg = ConformerEncoderV1Config(
        num_layers=12,
        frontend=ModuleFactoryV1(module_class=ConformerConv2dFrontend, cfg=conformer_conv2d_frontend_cfg),
        block_cfg=ConformerBlockV1Config(
            ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                input_dim=512, hidden_dim=2048, dropout=0.1, activation=nn.SiLU()
            ),
            mhsa_cfg=ConformerMHSAV1Config(input_dim=512, num_att_heads=8, att_weights_dropout=0.1, dropout=0.1),
            conv_cfg=ConformerConvolutionV1Config(
                channels=512,
                kernel_size=32,
                dropout=0.1,
                activation=nn.SiLU(),
                norm=nn.BatchNorm1d(num_features=512),
            ),
        ),
    )
    decoder_attention_cfg = AdditiveAttentionConfig(attention_dim=1024, att_weights_dropout=0.0)
    decoder_cfg = AttentionLSTMDecoderV1Config(
        encoder_dim=512,
        vocab_size=10025,
        target_embed_dim=640,
        target_embed_dropout=0.1,
        lstm_hidden_size=1024,
        zoneout_drop_h=0.05,
        zoneout_drop_c=0.15,
        attention_cfg=decoder_attention_cfg,
        output_proj_dim=1024,
        output_dropout=0.3,
    )
    model_cfg = ConformerAEDModelConfig(
        feat_extraction_cfg=feat_extraction_cfg,
        encoder_cfg=conformer_encoder_cfg,
        decoder_cfg=decoder_cfg,
    )
    aed_model = ConformerAEDModel(cfg=model_cfg)
    return aed_model
