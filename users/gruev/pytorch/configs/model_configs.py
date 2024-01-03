import copy
from typing import List

import torch

from i6_models.config import ModuleFactoryV1
from i6_experiments.users.gruev.pytorch.models.i6_base_model import ConformerCTCConfig

from i6_models.primitives.feature_extraction import (
    LogMelFeatureExtractionV1Config,
    LogMelFeatureExtractionV1,
)
from i6_experiments.users.berger.pytorch.custom_parts.specaugment import (
    SpecaugmentByLengthConfigV1,
    SpecaugmentByLengthModuleV1,
)
from i6_models.parts.frontend.vgg_act import (
    VGG4LayerActFrontendV1Config,
    VGG4LayerActFrontendV1,
)

from i6_experiments.users.berger.pytorch.custom_parts import (
    ConformerMHSARelposV1Config,
    SelfAttRelPosEncodingV1Config,
)
from i6_experiments.users.gruev.pytorch.custom_parts.conformer import (
    ConformerPositionwiseFeedForwardV1Config,
    ConformerConvolutionV1Config,
    ConformerBlockConvFirstV1Config,
    ConformerEncoderConvFirstV1Config,
    ConformerEncoderConvFirstV1,
)
from i6_models.parts.conformer.norm import LayerNormNC

from i6_experiments.users.gruev.pytorch.serializers.basic import (
    get_basic_pt_network_train_serializer,
    get_basic_pt_network_recog_serializer,
)


def get_model_config(
    num_inputs: int,
    num_outputs: int,
    aux_layers: List[int] = [12],
    aux_scales: List[float] = [1.0],
    share_aux_parameters: bool = True,
    fairseq_weight_init: bool = False,
) -> ConformerCTCConfig:
    # TODO: effect of center, n_fft
    feature_extraction_cfg = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.050,
        hop_size=0.0125,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=num_inputs,
        center=True,
    )
    feature_extraction = ModuleFactoryV1(LogMelFeatureExtractionV1, cfg=feature_extraction_cfg)
    # TODO: tune
    specaugment_cfg = SpecaugmentByLengthConfigV1(
        time_min_num_masks=2,
        time_max_mask_per_n_frames=25,
        time_mask_max_size=20,
        freq_min_num_masks=2,
        freq_max_num_masks=5,
        freq_mask_max_size=8,
    )
    specaugment = ModuleFactoryV1(SpecaugmentByLengthModuleV1, cfg=specaugment_cfg)
    # TODO: change with generic_frontend
    frontend_cfg = VGG4LayerActFrontendV1Config(
        in_features=num_inputs,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation=torch.nn.ReLU(),
        out_features=512,
    )

    # frontend_cfg = VGGFrontendConfigV1(
    #     num_inputs=num_inputs, # number of DbMel filters
    #     conv1_channels=32,
    #     conv2_channels=64,
    #     conv3_channels=64,
    #     conv_kernel_size=3, # (3,3)-kernel
    #     conv1_stride=1, # (1,1), i.e. no stride
    #     conv2_stride=3, # (3,1)-subsampling in time
    #     conv3_stride=2, # (2,1)-subsampling in time
    #     pool_size=2, # (2,1)-subsampling in features
    #     linear_size=512,
    #     dropout=0.1,
    # )

    # ModuleFactoryV1 instead of SubassemblyWithOptions
    frontend = ModuleFactoryV1(VGG4LayerActFrontendV1, frontend_cfg)

    ff_cfg = ConformerPositionwiseFeedForwardV1Config(
        input_dim=512,
        hidden_dim=2048,
        dropout=0.1,
        activation=torch.nn.SiLU(),
    )
    # TODO: change
    rel_pos_enc_cfg = SelfAttRelPosEncodingV1Config(out_dim=64, clipping=32, dropout=0.1)
    # TODO: custom weight initialization
    mhsa_cfg = ConformerMHSARelposV1Config(
        input_dim=512,
        num_att_heads=8,
        att_weights_dropout=0.1,
        dropout=0.1,
    )

    def _serialize_layer_norm(layer_norm_instance):
        return f"{layer_norm_instance.__class__.__name__}({layer_norm_instance.normalized_shape[0]})"

    layer_norm = LayerNormNC(512)
    layer_norm.__repr__ = lambda: _serialize_layer_norm(layer_norm)

    conv_cfg = ConformerConvolutionV1Config(
        channels=512,
        kernel_size=31,
        dropout=0.1,
        activation=torch.nn.SiLU(),
        norm=layer_norm,
    )
    block_cfg = ConformerBlockConvFirstV1Config(
        rel_pos_enc_cfg=rel_pos_enc_cfg,
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
    )
    conformer_cfg = ConformerEncoderConvFirstV1Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )
    conformer = ModuleFactoryV1(ConformerEncoderConvFirstV1, cfg=conformer_cfg)

    # TODO: more generic dim, i.e. conformer_dim in model_config
    return ConformerCTCConfig(
        feature_extraction=feature_extraction,
        specaugment=specaugment,
        conformer=conformer,
        dim=512,
        target_size=num_outputs,
        aux_layers=aux_layers,
        aux_scales=aux_scales,
        share_aux_parameters=share_aux_parameters,
        fairseq_weight_init=fairseq_weight_init,
    )
