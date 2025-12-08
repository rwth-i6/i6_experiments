from dataclasses import dataclass
from typing import Optional
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant

import torch
from torch import nn

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV2Config
from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV2Config
from i6_models_repo.i6_models.parts.conformer.norm import LayerNormNC
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_experiments.users.berger.pytorch.models.util import lengths_to_padding_mask
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.jxu.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosBlockV1Config, ConformerRelPosEncoderV1Config, ConformerRelPosBlockV1,ConformerRelPosEncoderV1
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_experiments.common.setups.serialization import Import, NonhashedCode


from returnn.tensor.tensor_dict import TensorDict


@dataclass
class ConformerCTCConfig(ModelConfiguration):
    feature_extraction_cfg: LogMelFeatureExtractionV1Config
    specaug_args: dict
    conformer_cfg: ConformerRelPosEncoderV1Config
    final_dropout: float
    target_size: int
    aux_losses: dict


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerCTCConfig, **kwargs):
        super().__init__()
        self.logmel_feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_cfg)
        self.specaug_args = cfg.specaug_args
        self.conformer = ConformerRelPosEncoderV1(cfg.conformer_cfg)
        self.final_linear_list = torch.nn.ModuleList(
            [nn.Linear(cfg.conformer_cfg.block_cfg.ff_cfg.input_dim, cfg.target_size) for _ in range(len(cfg.aux_losses))])
        self.aux_losses = cfg.aux_losses
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
                x = specaugment_v1_by_length(audio_features, **self.specaug_args)  # [B, T, F]
            else:
                x = audio_features

        sequence_mask = lengths_to_padding_mask(audio_features_len)
        returnn_layers = [int(i)-1 for i in self.aux_losses.keys()]
        conformer_out_list, sequence_mask = self.conformer(x, sequence_mask, return_layers=returnn_layers)  # [B, T, F]
        log_probs_list = []
        for i in range(len(self.aux_losses)):
            logits = self.final_linear_list[i](conformer_out_list[i])  # [B, T, F]
            log_probs = torch.nn.functional.log_softmax(logits.to(torch.float32), dim=-1)
            log_probs_list.append(log_probs)

        if self.training:
            return log_probs_list, sequence_mask

        return log_probs_list[-1], audio_features_len


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


def get_default_config_v1(num_inputs: int, num_outputs: int, network_args) -> ConformerCTCConfig:
    aux_losses = dict(sorted(network_args["aux_losses"].items(), key=lambda x:int(x[0])))
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
    specaug_args = {"time_min_num_masks": 2,
                    "time_max_mask_per_n_frames": 25,
                    "time_mask_max_size": 20,
                    "freq_min_num_masks": 2,
                    "freq_mask_max_size": 5,
                    "freq_max_num_masks": 8} if "specaug_args" not in network_args else network_args["specaug_args"]
    att_weights_dropout = 0.1 if "att_weights_dropout" not in network_args else network_args["att_weights_dropout"]
    kernel_size = 31 if "kernel_size" not in network_args else network_args["kernel_size"]
    num_layers = 12 if "num_layers" not in network_args else network_args["num_layers"]
    final_dropout = 0.1 if "final_dropout" not in network_args else network_args["final_dropout"]
    dropout = 0.1 if "dropout" not in network_args else network_args["dropout"]
    d_model = 512 if "d_model" not in network_args else network_args["d_model"]
    vgg_act = "relu"

    frontend_cfg = VGG4LayerActFrontendV1Config(
        in_features=num_inputs,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(1, 2),
        pool1_stride=None,
        pool1_padding=None,
        pool2_kernel_size=(1, 2),
        pool2_stride=(4, 1),
        pool2_padding=None,
        activation=torch.nn.ReLU() if vgg_act=="relu" else torch.nn.SiLU(),
        out_features=d_model,
    )

    frontend = ModuleFactoryV1(VGG4LayerActFrontendV1, frontend_cfg)

    ff_cfg = ConformerPositionwiseFeedForwardV2Config(
        input_dim=d_model,
        hidden_dim=d_model*4,
        dropout=dropout,
        activation=torch.nn.SiLU(),
        dropout_broadcast_axes=None
    )

    mhsa_cfg = ConformerMHSARelPosV1Config(
        input_dim=d_model,
        num_att_heads=d_model//64,
        with_bias=True,
        att_weights_dropout=att_weights_dropout,
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
        dropout=dropout,
        dropout_broadcast_axes = None
    )

    conv_cfg = ConformerConvolutionV2Config(
        channels=d_model,
        kernel_size=kernel_size,
        dropout=dropout,
        activation=torch.nn.SiLU(),
        norm=LayerNormNC(d_model),
        dropout_broadcast_axes=None,
    )

    block_cfg = ConformerRelPosBlockV1Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
        modules=["ff", "mhsa", "conv", "ff"],
    )

    conformer_cfg = ConformerRelPosEncoderV1Config(
        num_layers=num_layers,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    return ConformerCTCConfig(
        feature_extraction_cfg=feature_extraction_cfg,
        specaug_args=specaug_args,
        conformer_cfg=conformer_cfg,
        target_size=num_outputs,
        final_dropout=final_dropout,
        aux_losses=aux_losses
    )


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    assert extern_data["classes"].raw_tensor is not None
    targets = extern_data["classes"].raw_tensor.long()

    targets_len_rf = extern_data["classes"].dims[1].dyn_size_ext
    assert targets_len_rf is not None
    targets_len = targets_len_rf.raw_tensor
    assert targets_len is not None

    model.train()

    log_probs_list, sequence_mask = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )
    sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)

    # make sure the layers ordering is right
    loss_layers = list(model.aux_losses.keys())
    loss_scales = list(model.aux_losses.values())

    import returnn.frontend as rf
    from returnn.tensor import batch_dim

    for i in range(len(log_probs_list)):
        log_probs = torch.transpose(log_probs_list[i], 0, 1)  # [T, B, F]

        loss = torch.nn.functional.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=sequence_lengths,
            target_lengths=targets_len,
            blank=0,
            reduction="sum",
            zero_infinity=True,
        )

        rf.get_run_ctx().mark_as_loss(
            name=f"CTC_{loss_layers[i]}",
            loss=loss,
            custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim),
            scale=loss_scales[i],
            use_normalized_loss=True,
        )


def get_train_serializer(
    model_config: ConformerCTCConfig,
) -> Collection:
    pytorch_package = "i6_experiments.users.berger.pytorch"
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.train_step"),
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
            NonhashedCode("import returnn.frontend as rf\n"),
            Import(f"{pytorch_package}.forward.basic.forward_step"),
            Import(f"{pytorch_package}.forward.prior_callback.ComputePriorCallback", import_as="forward_callback"),
        ],
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
        input_names=["data", "data:size1"],
        output_names=["log_probs"],
        opset_version=17,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data:size1": {0: "batch"},
            "log_probs": {0: "batch", 1: "time"},
        },
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
