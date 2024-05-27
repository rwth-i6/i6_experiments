from dataclasses import dataclass
from typing import Optional
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant

import torch
from torch import nn

from returnn.tensor.tensor_dict import TensorDict
from i6_experiments.users.berger.pytorch.helper_functions import map_tensor_to_minus1_plus1_interval
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config, ConformerConvolutionV1
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config, ConformerMHSAV1
from i6_models_repo.i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config, \
    ConformerPositionwiseFeedForwardV1
from i6_experiments.users.berger.pytorch.models.util import lengths_to_padding_mask
from i6_experiments.common.setups.serialization import Import
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.assemblies.conformer.conformer_v2 import ConformerBlockV2Config, ConformerEncoderV2Config, ConformerBlockV2, ConformerEncoderV2
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.config import ModelConfiguration, ModuleFactoryV1


@dataclass
class ConformerCTCConfig(ModelConfiguration):
    feature_extraction_cfg: LogMelFeatureExtractionV1Config
    specaug_args: dict
    conformer_cfg: ConformerEncoderV2Config
    final_dropout: float
    target_size: int


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerCTCConfig, **kwargs):
        super().__init__()
        self.logmel_feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_cfg)
        self.specaug_args = cfg.specaug_args
        self.conformer = ConformerEncoderV2(cfg.conformer_cfg)
        self.final_linear = torch.nn.Linear(cfg.conformer_cfg.block_cfg.ff_cfg.input_dim, cfg.target_size)
        self.final_dropout = nn.Dropout(p=cfg.final_dropout)
        self.export_mode = False

    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: Optional[torch.Tensor] = None,
    ):
        with torch.no_grad():
            squeezed_features = torch.squeeze(audio_features)
            if self.export_mode:
                squeezed_features = squeezed_features.type(torch.FloatTensor)
            else:
                squeezed_features = squeezed_features.type(torch.cuda.FloatTensor)

            audio_features, audio_features_len = self.logmel_feat_extraction(squeezed_features, audio_features_len)

        if self.training:
            x = specaugment_v1_by_length(audio_features,**self.specaug_args)  # [B, T, F]
        else:
            x = audio_features
        # sequence_mask = None if self.export_mode else lengths_to_padding_mask(audio_features_len)
        sequence_mask = lengths_to_padding_mask(audio_features_len)
        # sequence_mask = lengths_to_padding_mask((audio_features_len + 2) // 3)
        outputs, sequence_mask = self.conformer(x, sequence_mask)  # [B, T, F]
        x = self.final_dropout(outputs[0])
        logits = self.final_linear(x)  # [B, T, F]
        log_probs = torch.log_softmax(logits, dim=2)

        if self.export_mode:
            return log_probs
        return log_probs, sequence_mask


def get_default_config_v1(num_inputs: int, num_outputs: int, network_args: dict) -> ConformerCTCConfig:
    dropout = 0.2 if "dropout" not in network_args else network_args["dropout"]
    num_att_heads = 6 if "num_att_heads" not in network_args else network_args["num_att_heads"]
    att_weights_dropout = 0.1 if "att_weights_dropout" not in network_args else network_args["att_weights_dropout"]
    num_layers = 12 if "num_layers" not in network_args else network_args["num_layers"]
    kernel_size = 31 if "kernel_size" not in network_args else network_args["kernel_size"]
    specaug_args = {"time_min_num_masks": 2,
                    "time_max_mask_per_n_frames": 25,
                    "time_mask_max_size": 20,
                    "freq_min_num_masks": 2,
                    "freq_mask_max_size": 5,
                    "freq_max_num_masks": 10} if "specaug_args" not in network_args else network_args["specaug_args"]
    final_dropout = 0 if "final_dropout" not in network_args else network_args["final_dropout"]

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
        out_features=384,
    )

    frontend = ModuleFactoryV1(VGG4LayerActFrontendV1, frontend_cfg)

    ff_cfg = ConformerPositionwiseFeedForwardV1Config(
        input_dim=384,
        hidden_dim=1536,
        dropout=dropout,
        activation=torch.nn.SiLU(),
    )

    mhsa_cfg = ConformerMHSAV1Config(
        input_dim=384,
        num_att_heads=num_att_heads,
        att_weights_dropout=att_weights_dropout,
        dropout=dropout,
    )

    conv_cfg = ConformerConvolutionV1Config(
        channels=384,
        kernel_size=kernel_size,
        dropout=dropout,
        activation=torch.nn.SiLU(),
        norm=LayerNormNC(384),
    )

    block_cfg = ConformerBlockV2Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
        modules=["ff", "conv", "mhsa", "ff"],
    )

    conformer_cfg = ConformerEncoderV2Config(
        num_layers=num_layers,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    return ConformerCTCConfig(
        feature_extraction_cfg=feature_extraction_cfg,
        specaug_args=specaug_args,
        conformer_cfg=conformer_cfg,
        target_size=num_outputs,
        final_dropout=final_dropout
    )


def export(*, model: torch.nn.Module, model_filename: str):
    dummy_data = torch.randn(1, 30 * 160, 1, device="cpu")
    dummy_data_len = torch.ones((1,), dtype=torch.int32) * 30 * 160

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


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **_):
    audio_features = extern_data["data"].raw_tensor
    audio_features = audio_features.squeeze(-1)
    audio_features = map_tensor_to_minus1_plus1_interval(audio_features)
    assert extern_data["data"].dims[1].dyn_size_ext is not None

    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_len is not None

    assert extern_data["targets"].raw_tensor is not None
    targets = extern_data["targets"].raw_tensor.long()

    targets_len_rf = extern_data["targets"].dims[1].dyn_size_ext
    assert targets_len_rf is not None
    targets_len = targets_len_rf.raw_tensor
    assert targets_len is not None

    log_probs, sequence_mask = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)

    log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, F]

    loss = torch.nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=sequence_lengths,
        target_lengths=targets_len,
        blank=0,
        reduction="sum",
        zero_infinity=True,
    )

    from returnn.tensor import batch_dim
    import returnn.frontend as rf

    rf.get_run_ctx().mark_as_loss(
        name="CTC", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )

def get_recog_serializer(
        model_config: ConformerCTCConfig,
) -> Collection:
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.ConformerCTCModel",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.export"),
        ],
    )


def get_prior_serializer(
        model_config: ConformerCTCConfig,
) -> Collection:
    berger_pytorch_package = "i6_experiments.users.berger.pytorch"
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.ConformerCTCModel",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{berger_pytorch_package}.forward.basic.forward_step"),
            Import(f"{berger_pytorch_package}.forward.prior_callback.ComputePriorCallback",
                   import_as="forward_callback"),
        ],
    )


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
        ],
    )