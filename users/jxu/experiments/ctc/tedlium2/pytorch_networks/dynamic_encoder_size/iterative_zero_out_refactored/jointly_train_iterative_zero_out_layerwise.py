from dataclasses import dataclass
from typing import Optional
from collections import OrderedDict

import torch
from torch import nn
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import numpy as np

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.assemblies.conformer_with_dynamic_model_size.selection_with_iterative_zero_out import (
    ConformerBlockConfig,
    ConformerEncoderConfig,
    ConformerEncoder
)
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.config import ModelConfiguration, ModuleFactoryV1

from i6_experiments.users.berger.pytorch.models.util import lengths_to_padding_mask
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)

from i6_experiments.common.setups.serialization import Import


@dataclass
class ConformerCTCConfig(ModelConfiguration):
    conformer_cfg: ConformerEncoderConfig
    target_size: int
    recog_num_layers: int
    zero_out_kwargs: dict


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerCTCConfig, **kwargs):
        super().__init__()
        self.conformer = ConformerEncoder(cfg.conformer_cfg)
        self.final_linear_list = torch.nn.ModuleList(
            [nn.Linear(cfg.conformer_cfg.block_cfg.ff_cfg.input_dim, cfg.target_size) for _ in
             range(len(self.conformer.num_layers_set))])
        self.recog_num_layers = cfg.recog_num_layers
        self.conformer.recog_num_layers = cfg.recog_num_layers
        self.zero_out_kwargs = cfg.zero_out_kwargs
        self.export_mode = False

    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: Optional[torch.Tensor] = None,
            global_train_step: int = 0
    ):
        if self.training:
            x = specaugment_v1_by_length(audio_features,
                                         time_min_num_masks=2,
                                         time_max_mask_per_n_frames=25,
                                         time_mask_max_size=20,
                                         freq_min_num_masks=2,
                                         freq_mask_max_size=5,
                                         freq_max_num_masks=10)  # [B, T, F]
        else:
            x = audio_features
        # sequence_mask = None if self.export_mode else lengths_to_padding_mask(audio_features_len)
        sequence_mask = lengths_to_padding_mask(audio_features_len)
        # sequence_mask = lengths_to_padding_mask((audio_features_len + 2) // 3)
        conformer_out_list, total_utilised_layers, sequence_mask = self.conformer(x, sequence_mask, global_train_step,
                                                           self.zero_out_kwargs)  # [B, T, F]

        log_probs_list = []
        for i in range(len(conformer_out_list)):
            logits = self.final_linear_list[i](conformer_out_list[i])  # [B, T, F]
            log_probs = torch.log_softmax(logits, dim=2)
            log_probs_list.append(log_probs)

        if self.training:
            return log_probs_list, total_utilised_layers, sequence_mask

        idx = self.conformer.num_layers_set.index(self.recog_num_layers)
        return log_probs_list[idx]


def get_default_config_v1(num_inputs: int, num_outputs: int, network_args: dict) -> ConformerCTCConfig:
    dropout = 0.2 if "dropout" not in network_args else network_args["dropout"]
    num_att_heads = 6 if "num_att_heads" not in network_args else network_args["num_att_heads"]
    att_weights_dropout = 0.1 if "att_weights_dropout" not in network_args else network_args["att_weights_dropout"]
    num_layers = 12 if "num_layers" not in network_args else network_args["num_layers"]
    kernel_size = 31 if "kernel_size" not in network_args else network_args["kernel_size"]
    num_layers_set = network_args["num_layers_set"]
    layer_dropout_kwargs = network_args["layer_dropout_kwargs"]
    recog_num_layers = network_args["recog_num_layers"]
    zero_out_kwargs = network_args["zero_out_kwargs"]
    layer_gate_activation = network_args["layer_gate_activation"]

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

    block_cfg = ConformerBlockConfig(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
        layer_dropout=layer_dropout_kwargs["layer_dropout_stage_1"],
        modules=["ff", "conv", "mhsa", "ff"],
        scales=[0.5, 1.0, 1.0, 0.5]
    )


    conformer_cfg = ConformerEncoderConfig(
        num_layers=num_layers,
        frontend=frontend,
        block_cfg=block_cfg,
        num_layers_set=num_layers_set,
        layer_dropout_kwargs=layer_dropout_kwargs,
        layer_gate_activation=torch.nn.Sigmoid() if layer_gate_activation=="sigmoid" else torch.nn.Identity()
    )

    return ConformerCTCConfig(
        conformer_cfg=conformer_cfg,
        target_size=num_outputs,
        recog_num_layers=recog_num_layers,
        zero_out_kwargs=zero_out_kwargs,
    )


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, global_train_step, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    targets = extern_data["targets"].raw_tensor.long()
    targets_len = extern_data["targets"].dims[1].dyn_size_ext.raw_tensor

    model.train()

    log_probs_list, total_utilised_layers, sequence_mask = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
        global_train_step=global_train_step
    )
    sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)

    num_steps_per_iter = model.zero_out_kwargs["num_steps_per_iter"]
    cum_steps_per_iter = np.cumsum(num_steps_per_iter)
    num_zeroout_elements_per_iter = model.zero_out_kwargs["num_zeroout_elements_per_iter"]
    stage_1_expected_sparsity_per_iter = [n / (4*len(model.conformer.module_list)) for n in num_zeroout_elements_per_iter]

    assert len(num_steps_per_iter) == len(num_zeroout_elements_per_iter)

    loss_layers = model.conformer.num_layers_set

    # stage 1 : jointly train the largest and smallest model
    if global_train_step <= cum_steps_per_iter[-1]:
        loss_scales = [0.3, 1]
        iter_idx = int(model.conformer.iter_idx)
        sparsity_loss = torch.abs((48-total_utilised_layers)/48 - stage_1_expected_sparsity_per_iter[iter_idx])
        rf.get_run_ctx().mark_as_loss(name=f"sparsity_loss", loss=sparsity_loss, scale=5)
        if global_train_step == cum_steps_per_iter[iter_idx] and global_train_step<cum_steps_per_iter[-1]:
            # apply zero at the end of the iteration
            print("iter_idx zeroout", iter_idx)
            print("num_zeroout_elements_per_iter", num_zeroout_elements_per_iter[iter_idx])
            _, zeroout_mods_indices = torch.topk(model.conformer.layer_gates, k=num_zeroout_elements_per_iter[iter_idx], largest=False)
            zeroout_val = model.zero_out_kwargs["zeroout_val"] if isinstance(model.conformer.layer_gate_activation, torch.nn.Sigmoid) else 0
            model.conformer.layer_gates.data[zeroout_mods_indices] = zeroout_val

        for i in [0, -1]:
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

            loss /= torch.sum(sequence_lengths)
            rf.get_run_ctx().mark_as_loss(name=f"CTC_{loss_layers[i]}", loss=loss, scale=loss_scales[i])

    # stage 2 : jointly train all models efficiently with sandwich rules
    else:
        if len(model.conformer.num_layers_set) <= 3:
            loss_scales = [0.3]*(len(model.conformer.num_layers_set)-1) + [1]
        else:
            loss_scales = [0.3] + [0]*(len(model.conformer.num_layers_set)-2) + [1]
            loss_scales[model.conformer.random_idx] = 0.3
            print(f"random_idx {model.conformer.random_idx}")

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

            loss /= torch.sum(sequence_lengths)
            rf.get_run_ctx().mark_as_loss(name=f"CTC_{loss_layers[i]}", loss=loss, scale=loss_scales[i])


def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    log_probs = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )  # [B, T, F]
    rf.get_run_ctx().mark_as_output(log_probs, name="log_probs")


def get_prior_serializer(
        model_config: ConformerCTCConfig,
) -> Collection:
    pytorch_package = "i6_experiments.users.berger.pytorch"

    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.forward_step"),
            Import(f"{pytorch_package}.forward.prior_callback.ComputePriorCallback", import_as="forward_callback"),
        ],
    )


def export(*, model: torch.nn.Module, model_filename: str):
    model.export_mode = True
    dummy_data = torch.randn(1, 30, 50, device="cpu")
    dummy_data_len = torch.tensor([30], dtype=torch.int32)
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


def get_recog_serializer(
        model_config: ConformerCTCConfig,
) -> Collection:
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.export"),
        ],
    )


def get_train_serializer(
        model_config: ConformerCTCConfig,
) -> Collection:
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.train_step"),
        ],
    )
