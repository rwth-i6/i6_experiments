from dataclasses import dataclass
from typing import Optional
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant

from returnn.tensor import batch_dim
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf


import torch
from torch import nn
from typing import Tuple
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import Import
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
# from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_experiments.users.berger.pytorch.custom_parts.vgg_frontend import VGGFrontendConfigV1, VGGFrontendV1
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config, ConformerConvolutionV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config, ConformerMHSAV1
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config, ConformerPositionwiseFeedForwardV1
from i6_models.assemblies.conformer import (
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
)
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_experiments.users.berger.pytorch.models.util import lengths_to_padding_mask


from i6_experiments.users.berger.pytorch.custom_parts import specaugment


class ConformerBlockV1(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.ff1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAV1(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV1(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outsideq, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        x = 0.5 * self.ff1(x) + x  #  [B, T, F]
        x = self.conv(x) + x  #  [B, T, F]
        x = self.mhsa(x, sequence_mask) + x  #  [B, T, F]
        x = 0.5 * self.ff2(x) + x  #  [B, T, F]
        x = self.final_layer_norm(x)  #  [B, T, F]
        return x



class ConformerEncoderV1(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV1(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']
        for module in self.module_list:
            x = module(x, sequence_mask)  # [B, T, F']

        return x, sequence_mask

@dataclass
class ConformerCTCConfig(ModelConfiguration):
    feature_extraction_cfg: LogMelFeatureExtractionV1Config
    specaugment_cfg: specaugment.SpecaugmentByLengthConfigV1
    conformer_cfg: ConformerEncoderV1Config
    target_size: int


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerCTCConfig, **kwargs):
        super().__init__()
        self.logmel_feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_cfg)
        self.specaugment = specaugment.SpecaugmentByLengthModuleV1(cfg=cfg.specaugment_cfg)
        self.conformer = ConformerEncoderV1(cfg.conformer_cfg)
        self.final_linear = torch.nn.Linear(cfg.conformer_cfg.block_cfg.ff_cfg.input_dim, cfg.target_size)
        self.export_mode = False

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: Optional[torch.Tensor] = None,
    ):
        with torch.no_grad():
            squeezed_features = torch.squeeze(raw_audio)
            if self.export_mode:
                queezed_features = squeezed_features.type(torch.FloatTensor)
            else:
                squeezed_features = squeezed_features.type(torch.cuda.FloatTensor)

            audio_features, audio_features_len = self.logmel_feat_extraction(squeezed_features, raw_audio_len)
            x = self.specaugment(audio_features)  # [B, T, F]

        sequence_mask = None if self.export_mode else lengths_to_padding_mask(audio_features_len)
        # sequence_mask = lengths_to_padding_mask((audio_features_len + 2) // 3)
        x, sequence_mask = self.conformer(x, sequence_mask)  # [B, T, F]
        logits = self.final_linear(x)  # [B, T, F]
        log_probs = torch.log_softmax(logits, dim=2)

        if self.training:
            return log_probs, sequence_mask
        return log_probs, audio_features_len

#
#
# def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
#     audio_features = extern_data["data"].raw_tensor
#     assert extern_data["data"].dims[1].dyn_size_ext is not None
#
#     audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
#     assert audio_features_len is not None
#
#     assert extern_data["targets"].raw_tensor is not None
#     targets = extern_data["targets"].raw_tensor.long()
#
#     targets_len_rf = extern_data["targets"].dims[1].dyn_size_ext
#     assert targets_len_rf is not None
#     targets_len = targets_len_rf.raw_tensor
#     assert targets_len is not None
#
#     model.train()
#
#     log_probs, sequence_mask = model(
#         raw_audio=audio_features,
#         raw_audio_len=audio_features_len.to("cuda"),
#     )
#
#     log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, F]
#     sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)
#
#     loss = torch.nn.functional.ctc_loss(
#         log_probs=log_probs,
#         targets=targets,
#         input_lengths=sequence_lengths,
#         target_lengths=targets_len,
#         blank=78,
#         reduction="sum",
#         zero_infinity=True,
#     )
#
#     rf.get_run_ctx().mark_as_loss(
#         name="CTC", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
#     )
#
#
def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    log_probs, sequence_mask = model(
        raw_audio=audio_features,
        raw_audio_len=audio_features_len.to("cuda"),
    )  # [B, T, F]

    rf.get_run_ctx().mark_as_output(log_probs, name="log_probs")


def get_train_serializer(
    model_config: ConformerCTCConfig,
) -> Collection:
    # pytorch_package = __package__.rpartition(".")[0]
    pytorch_package = "i6_experiments.users.berger.pytorch"
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.train_step"),
            # Import(f"{pytorch_package}.train_steps.ctc.train_step"),
        ],
    )


def get_prior_serializer(
    model_config: ConformerCTCConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    pytorch_package = "i6_experiments.users.berger.pytorch"
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            # Import(f"{pytorch_package}.forward.basic.forward_step"),
            Import(f"{__name__}.forward_step"),
            Import(f"{pytorch_package}.forward.prior_callback.ComputePriorCallback", import_as="forward_callback"),
        ],
    )


def get_recog_serializer(
    model_config: ConformerCTCConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.export"),
        ],
    )


def get_serializer(model_config: ConformerCTCConfig, variant: ConfigVariant) -> Collection:
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.ALIGN:
        return get_recog_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def get_default_config_v1(num_inputs: int, num_outputs: int) -> ConformerCTCConfig:
    feature_extraction_cfg = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=False,
    )

    specaugment_cfg = specaugment.SpecaugmentByLengthConfigV1(
        time_min_num_masks=2,
        time_max_mask_per_n_frames=25,
        time_mask_max_size=20,
        freq_min_num_masks=2,
        freq_max_num_masks=5,
        freq_mask_max_size=10
    )
    d_model = 512
    frontend_cfg = VGGFrontendConfigV1(
        num_inputs=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv_kernel_size=3,
        conv1_stride=1,
        conv2_stride=2,
        conv3_stride=2,
        pool_size=2,
        linear_size=512,
        dropout=0.1,
    )

    frontend = ModuleFactoryV1(VGGFrontendV1, frontend_cfg)

    ff_cfg = ConformerPositionwiseFeedForwardV1Config(
        input_dim=d_model,
        hidden_dim=d_model*4,
        dropout=0.2,
        activation=torch.nn.SiLU(),
    )

    mhsa_cfg = ConformerMHSAV1Config(
        input_dim=d_model,
        num_att_heads=d_model//64,
        att_weights_dropout=0.1,
        dropout=0.2,
    )

    conv_cfg = ConformerConvolutionV1Config(
        channels=d_model,
        kernel_size=31,
        dropout=0.2,
        activation=torch.nn.SiLU(),
        norm=LayerNormNC(d_model),
    )

    block_cfg = ConformerBlockV1Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
    )

    conformer_cfg = ConformerEncoderV1Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    return ConformerCTCConfig(
        feature_extraction_cfg=feature_extraction_cfg,
        specaugment_cfg=specaugment_cfg,
        conformer_cfg=conformer_cfg,
        target_size=num_outputs,
    )



def export(*, model: torch.nn.Module, model_filename: str):
    dummy_data = torch.randn(1, 30*160, 1, device="cpu")
    dummy_data_len = torch.ones((1,), dtype=torch.int32)*30*160

    model.export_mode = True
    torch.onnx.export(
        model=model.eval(),
        args=(dummy_data, dummy_data_len),
        f=model_filename,
        verbose=True,
        input_names=["data", "data_len"],
        output_names=["classes"],
        opset_version=17,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data_len": {0: "batch"},
            "targets": {0: "batch", 1: "time"},
        },
    )
