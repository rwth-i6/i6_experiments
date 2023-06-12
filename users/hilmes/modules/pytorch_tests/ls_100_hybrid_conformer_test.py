import torch
from torch import nn
from i6_models.assemblies.conformer import (
    ConformerEncoderV1,
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
)
from i6_models.parts.conformer import (
    ConformerFrontendV1Config,
    ConformerPositionwiseFeedForwardV1Config,
    ConformerConvolutionV1Config,
    ConformerMHSAV1Config,
    LayerNormNC
)

conv_cfg = ConformerConvolutionV1Config(channels=256, kernel_size=5, dropout=0.1, activation=nn.SiLU(), norm=LayerNormNC(250))
mhsa_cfg = ConformerMHSAV1Config(input_dim=256, num_att_heads=8, att_weights_dropout=0.1, dropout=0.1)
ff_cfg = ConformerPositionwiseFeedForwardV1Config(input_dim=256, hidden_dim=4 * 256, activation=nn.SiLU(), dropout=0.1)
block_cfg = ConformerBlockV1Config(ff_cfg=ff_cfg, mhsa_cfg=mhsa_cfg, conv_cfg=conv_cfg)
front_cfg = ConformerFrontendV1Config(
    feature_dim=80, hidden_dim=256, dropout=0.1, conv_stride=2, conv_kernel=5, conv_padding=0, spec_aug_cfg=None
)
conformer_cfg = ConformerEncoderV1Config(num_layers=12, front_cfg=front_cfg, block_cfg=block_cfg)
conf = ConformerEncoderV1(cfg=conformer_cfg)

print(conf)
