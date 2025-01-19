from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import numpy as np

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.config import ModelConfiguration, ModuleFactoryV1

from i6_experiments.users.berger.pytorch.helper_functions import map_tensor_to_minus1_plus1_interval
from i6_experiments.users.berger.pytorch.models.util import lengths_to_padding_mask
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models_repo.i6_models.parts.conformer.norm import LayerNormNC

from i6_experiments.common.setups.serialization import Import
from i6_experiments.users.berger.pytorch.custom_parts import specaugment
from i6_models.assemblies.conformer_with_dynamic_model_size.selection_with_independent_softmax_layerwise_num_params_aware_random_pct import (
    ConformerBlockConfig,
    ConformerEncoderConfig,
    ConformerEncoder
)

EPSILON = np.finfo(np.float32).tiny


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

@dataclass
class ConformerCTCConfig(ModelConfiguration):
    feature_extraction_cfg: LogMelFeatureExtractionV1Config
    specaugment_cfg: specaugment.SpecaugmentByLengthConfigV1
    conformer_cfg: ConformerEncoderConfig
    target_size: int
    recog_param_pct: float
    stage_1_global_steps: int
    params_kwargs: dict
    aux_loss_scales: dict
    final_dropout: float


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerCTCConfig, **kwargs):
        super().__init__()
        self.logmel_feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_cfg)
        self.specaugment = specaugment.SpecaugmentByLengthModuleV1(cfg=cfg.specaugment_cfg)
        self.conformer = ConformerEncoder(cfg.conformer_cfg)
        self.final_linear_list = torch.nn.ModuleList(
            [nn.Linear(cfg.conformer_cfg.block_cfg.ff_cfg.input_dim, cfg.target_size) for _ in
             range(len(self.conformer.pct_params_set))])
        self.small_model_layers = None
        self.recog_param_pct = cfg.recog_param_pct
        self.conformer.recog_param_pct = self.recog_param_pct
        self.stage_1_global_steps = cfg.stage_1_global_steps
        self.params_kwargs = cfg.params_kwargs
        self.aux_loss_scales = cfg.aux_loss_scales
        self.softmax_lambda = nn.Parameter(torch.tensor(EPSILON))
        self.final_dropout = nn.Dropout(p=cfg.final_dropout)
        self.export_mode = False

    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: Optional[torch.Tensor] = None,
            global_train_step: int = 0
    ):
        with torch.no_grad():
            squeezed_features = torch.squeeze(audio_features)

            audio_features, audio_features_len = self.logmel_feat_extraction(squeezed_features, audio_features_len)
            x = self.specaugment(audio_features)  # [B, T, F]

        sequence_mask = lengths_to_padding_mask(audio_features_len)
        conformer_out_list, softmax_constraint, sequence_mask = self.conformer(x, sequence_mask, global_train_step,
                                                                               self.stage_1_global_steps,
                                                                               self.params_kwargs)  # [B, T, F]

        if self.conformer.softmax_kwargs["softmax_constraint_loss_scale"] == "lagrange_adaptive":
            softmax_lambda = GradMultiply.apply(self.softmax_lambda, -1)
            print("softmax_lambda", self.softmax_lambda)
            softmax_constraint = softmax_constraint * softmax_lambda

        if self.training:
            log_probs_list = []
            for i in range(len(conformer_out_list)):
                output = self.final_dropout(conformer_out_list[i])
                logits = self.final_linear_list[i](output)  # [B, T, F]
                log_probs = torch.log_softmax(logits, dim=2)
                log_probs_list.append(log_probs)
            return log_probs_list, softmax_constraint, sequence_mask

        idx = self.conformer.pct_params_set.index(self.recog_param_pct)
        output = self.final_dropout(conformer_out_list[idx])
        logits = self.final_linear_list[idx](output)  # [B, T, F]
        log_probs = torch.log_softmax(logits, dim=2)
        return log_probs


def get_default_config_v1(num_inputs: int, num_outputs: int, network_args: dict) -> ConformerCTCConfig:
    time_max_mask_per_n_frames = network_args["time_max_mask_per_n_frames"]
    freq_max_num_masks = network_args["freq_max_num_masks"]
    dropout = network_args["dropout"]
    recog_param_pct = network_args["recog_param_pct"]
    stage_1_global_steps = network_args["stage_1_global_steps"]
    pct_params_set = network_args["pct_params_set"]
    softmax_kwargs = network_args["softmax_kwargs"]
    layer_dropout_kwargs = network_args["layer_dropout_kwargs"]
    params_kwargs = network_args["params_kwargs"]
    final_dropout = network_args["final_dropout"]
    aux_loss_scales = network_args["aux_loss_scales"]
    freq_mask_max_size = 8 if "freq_mask_max_size" not in network_args else network_args["freq_mask_max_size"]

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
        time_max_mask_per_n_frames=time_max_mask_per_n_frames,
        time_mask_max_size=20,
        freq_min_num_masks=2,
        freq_max_num_masks=freq_max_num_masks,
        freq_mask_max_size=freq_mask_max_size
    )
    d_model = 512

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
        num_att_heads=384 // 64,
        att_weights_dropout=0.1,
        dropout=dropout,
    )

    conv_cfg = ConformerConvolutionV1Config(
        channels=384,
        kernel_size=31,
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
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
        pct_params_set=pct_params_set,
        layer_dropout_kwargs=layer_dropout_kwargs,
        softmax_kwargs=softmax_kwargs
    )

    return ConformerCTCConfig(
        feature_extraction_cfg=feature_extraction_cfg,
        specaugment_cfg=specaugment_cfg,
        conformer_cfg=conformer_cfg,
        target_size=num_outputs,
        recog_param_pct=recog_param_pct,
        stage_1_global_steps=stage_1_global_steps,
        params_kwargs=params_kwargs,
        final_dropout=final_dropout,
        aux_loss_scales=aux_loss_scales
    )


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, global_train_step, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features = audio_features.squeeze(-1)
    audio_features = map_tensor_to_minus1_plus1_interval(audio_features)
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    targets = extern_data["targets"].raw_tensor.long()
    targets_len = extern_data["targets"].dims[1].dyn_size_ext.raw_tensor

    model.train()

    log_probs_list, softmax_constraint, sequence_mask = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
        global_train_step=global_train_step
    )

    if softmax_constraint != 0:
        softmax_constraint_scale = 1
        if model.conformer.softmax_kwargs["softmax_constraint_loss_scale"] == "linear_increase":
            softmax_constraint_scale = min(model.conformer.softmax_kwargs["max_softmax_constraint_loss_scale"],
                                           model.conformer.softmax_kwargs["max_softmax_constraint_loss_scale"] /
                                           model.conformer.softmax_kwargs[
                                               "softmax_constraint_warmup_steps"] * global_train_step)
            print("softmax_constraint_scale", softmax_constraint_scale)
        elif model.conformer.softmax_kwargs["softmax_constraint_loss_scale"] != "lagrange_adaptive":
            softmax_constraint_scale = model.conformer.softmax_kwargs["softmax_constraint_loss_scale"]
        rf.get_run_ctx().mark_as_loss(name="softmax_constraint", loss=softmax_constraint,
                                      scale=softmax_constraint_scale)

    sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)

    loss_layers = model.conformer.pct_params_set
    # stage 1 : jointly train the largest and smallest model
    if global_train_step <= model.stage_1_global_steps:
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

            if isinstance(model.aux_loss_scales[str(loss_layers[i])], float) or isinstance(
                    model.aux_loss_scales[str(loss_layers[i])], int):
                loss_scale = model.aux_loss_scales[str(loss_layers[i])]
            else:
                if model.aux_loss_scales[str(loss_layers[i])] == "focal_loss":
                    loss_scale = 1 - torch.exp(-loss)
                    print(f"{loss_layers[i]} loss scale", loss_scale)

            rf.get_run_ctx().mark_as_loss(name=f"CTC_{loss_layers[i]}", loss=loss, scale=loss_scale)

    # stage 2 : jointly train all models efficiently with sandwich rules
    else:
        if len(model.conformer.pct_params_set) > 3:
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

            loss_scale = 0
            if i in [0, len(log_probs_list) - 1, model.conformer.random_idx]:
                if isinstance(model.aux_loss_scales[str(loss_layers[i])], float) or isinstance(
                        model.aux_loss_scales[str(loss_layers[i])], int):
                    loss_scale = model.aux_loss_scales[str(loss_layers[i])]
                else:
                    if model.aux_loss_scales[str(loss_layers[i])] == "focal_loss":
                        loss_scale = 1 - torch.exp(-loss)

            print(f"{loss_layers[i]} loss scale", loss_scale)

            rf.get_run_ctx().mark_as_loss(name=f"CTC_{loss_layers[i]}", loss=loss, scale=loss_scale)


def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    assert audio_features is not None
    audio_features = map_tensor_to_minus1_plus1_interval(audio_features)

    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_len is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.export_mode = True
    log_probs = model(
        audio_features=audio_features.to(device),
        audio_features_len=audio_features_len.to(device),
    )  # [B, T, F]

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    run_ctx.mark_as_output(log_probs, name="log_probs")


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_data = torch.randn(1, 30 * 160, 1, device=device)
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
