from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.dynamic_adaptable_conformer import (
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.dynamic_adaptable_e_branchformer import ConvolutionalGatingMLPV1Config, MergerV1Config
from i6_experiments.users.berger.pytorch.models.util import lengths_to_padding_mask
from i6_experiments.common.setups.serialization import Import
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.jxu.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.assemblies.dynamic_adaptable_e_branchformer import (
    EbranchformerBlockV1Config,
    EbranchformerEncoderV1Config,
    EbranchformerEncoderV1,
)
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.config import ModelConfiguration, ModuleFactoryV1

from returnn.tensor.tensor_dict import TensorDict
from i6_experiments.users.berger.pytorch.helper_functions import map_tensor_to_minus1_plus1_interval
from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.neural_block.dynamic_adaptable_e_branchformer.gradient_based_score import *


@dataclass
class EbranchformerCTCConfig(ModelConfiguration):
    feature_extraction_cfg: LogMelFeatureExtractionV1Config
    specaug_args: dict
    e_branchformer_cfg: EbranchformerEncoderV1Config
    final_dropout: float
    target_size: int
    grad_score_opts: dict
    adaptation_opts: dict
    component_dist: dict
    total_num_components: int
    lst_cmp_cost: list


class EbranchformerCTCModel(torch.nn.Module):
    def __init__(self, step: int, cfg: EbranchformerCTCConfig, **kwargs):
        super().__init__()
        self.logmel_feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_cfg)
        self.specaug_args = cfg.specaug_args
        self.e_branchformer = EbranchformerEncoderV1(cfg.e_branchformer_cfg)
        self.total_num_components = cfg.total_num_components
        self.score_per_component = torch.nn.Parameter(
            torch.FloatTensor(torch.zeros((cfg.total_num_components))), requires_grad=False
        )
        self.grad_score_opts = cfg.grad_score_opts
        self.adaptation_opts = cfg.adaptation_opts
        self.component_dist = cfg.component_dist
        self.register_parameter("lst_cmp_cost", nn.Parameter(torch.tensor(cfg.lst_cmp_cost, dtype=torch.float)))
        # assert self.total_num_components == self.lst_cmp_cost.size()[0]
        # TODO: should save the
        self.final_linear = torch.nn.Linear(cfg.e_branchformer_cfg.block_cfgs[-1].ff2_cfg.input_dim, cfg.target_size)
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
            x = specaugment_v1_by_length(audio_features, **self.specaug_args)  # [B, T, F]
        else:
            x = audio_features
        # sequence_mask = None if self.export_mode else lengths_to_padding_mask(audio_features_len)
        sequence_mask = lengths_to_padding_mask(audio_features_len)
        # sequence_mask = lengths_to_padding_mask((audio_features_len + 2) // 3)
        out, sequence_mask = self.e_branchformer(x, sequence_mask)  # [B, T, F]
        x = self.final_dropout(out)
        logits = self.final_linear(x)  # [B, T, F]
        log_probs = torch.log_softmax(logits, dim=2)

        if not self.export_mode:
            return log_probs, sequence_mask
        return log_probs


def get_default_config_v1(num_inputs: int, num_outputs: int, network_args: dict) -> EbranchformerCTCConfig:
    dropout = 0.2 if "dropout" not in network_args else network_args["dropout"]
    num_layers = 12 if "num_layers" not in network_args else network_args["num_layers"]
    model_dim = 384 if "model_dim" not in network_args else network_args["model_dim"]
    total_num_components = 216 if "total_num_components" not in network_args else network_args["total_num_components"]
    lst_ff1_h_dims = (
        [model_dim * 4] * num_layers if "lst_ff1_h_dims" not in network_args else network_args["lst_ff1_h_dims"]
    )
    lst_ff2_h_dims = (
        [model_dim * 4] * num_layers if "lst_ff2_h_dims" not in network_args else network_args["lst_ff2_h_dims"]
    )
    lst_att_heads = (
        [model_dim // 64] * num_layers if "lst_att_heads" not in network_args else network_args["lst_att_heads"]
    )
    lst_channels = [model_dim * 6] * num_layers if "lst_channels" not in network_args else network_args["lst_channels"]
    lst_ff1_dropouts = (
        [dropout] * num_layers if "lst_ff1_dropouts" not in network_args else network_args["lst_ff1_dropouts"]
    )
    lst_ff2_dropouts = (
        [dropout] * num_layers if "lst_ff1_dropouts" not in network_args else network_args["lst_ff1_dropouts"]
    )

    att_weights_dropout = 0.1 if "att_weights_dropout" not in network_args else network_args["att_weights_dropout"]
    kernel_size = 31 if "kernel_size" not in network_args else network_args["kernel_size"]
    merge_kernel_size = 31 if "merge_kernel_size" not in network_args else network_args["merge_kernel_size"]

    specaug_args = (
        {
            "time_min_num_masks": 2,
            "time_max_mask_per_n_frames": 25,
            "time_mask_max_size": 20,
            "freq_min_num_masks": 2,
            "freq_mask_max_size": 5,
            "freq_max_num_masks": 10,
        }
        if "specaug_args" not in network_args
        else network_args["specaug_args"]
    )
    final_dropout = 0 if "final_dropout" not in network_args else network_args["final_dropout"]
    grad_score_opts = network_args["grad_score_opts"]
    adaptation_opts = network_args["adaptation_opts"]
    component_dist = network_args["component_dist"]

    adjust_dropout = network_args["adjust_dropout"]
    lst_cmp_cost = network_args["lst_cmp_cost"]

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
        out_features=model_dim,
    )

    frontend = ModuleFactoryV1(VGG4LayerActFrontendV1, frontend_cfg)

    block_cfgs = []
    for i in range(num_layers):
        ff1_cfg = ConformerPositionwiseFeedForwardV1Config(
            input_dim=model_dim,
            hidden_dim=lst_ff1_h_dims[i],
            dropout=lst_ff1_dropouts[i],
            activation=nn.SiLU(),
        )

        cgmlp_cfg = ConvolutionalGatingMLPV1Config(
            input_dim=model_dim,
            hidden_dim=lst_channels[i],
            kernel_size=kernel_size,
            dropout=dropout,
            activation=nn.SiLU(),
        )

        mhsa_cfg = ConformerMHSAV1Config(
            att_head_dim=64,
            input_dim=model_dim,
            num_att_heads=lst_att_heads[i],
            att_weights_dropout=att_weights_dropout,
            dropout=dropout,
        )

        ff2_cfg = ConformerPositionwiseFeedForwardV1Config(
            input_dim=model_dim,
            hidden_dim=lst_ff2_h_dims[i],
            dropout=lst_ff2_dropouts[i],
            activation=nn.SiLU(),
        )

        merger_cfg = MergerV1Config(input_dim=model_dim, kernel_size=merge_kernel_size, dropout=dropout)

        block_cfg = EbranchformerBlockV1Config(
            ff1_cfg=ff1_cfg,
            mhsa_cfg=mhsa_cfg,
            cgmlp_cfg=cgmlp_cfg,
            ff2_cfg=ff2_cfg,
            merger_cfg=merger_cfg,
            adjust_dropout=adjust_dropout,
        )

        block_cfgs.append(block_cfg)

    e_branchformer_cfg = EbranchformerEncoderV1Config(
        num_layers=num_layers,
        frontend=frontend,
        block_cfgs=block_cfgs,
    )

    return EbranchformerCTCConfig(
        feature_extraction_cfg=feature_extraction_cfg,
        specaug_args=specaug_args,
        e_branchformer_cfg=e_branchformer_cfg,
        target_size=num_outputs,
        final_dropout=final_dropout,
        grad_score_opts=grad_score_opts,
        adaptation_opts=adaptation_opts,
        component_dist=component_dist,
        total_num_components=total_num_components,
        lst_cmp_cost=lst_cmp_cost,
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
        input_names=["data", "data:size1"],
        output_names=["log_probs"],
        opset_version=17,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data:size1": {0: "batch"},
            "targets": {0: "batch", 1: "time"},
        },
    )


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, global_train_step, **_):
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

    if global_train_step % model.grad_score_opts["grad_score_update_steps"] == 0:
        if model.grad_score_opts["grad_score_metric"] == "first_taylor":
            update_gradient_score = grad_score_taylor(loss, model)

        model.score_per_component[:] = model.grad_score_opts["grad_update_beta"] * model.score_per_component + (
            1 - model.grad_score_opts["grad_update_beta"]
        ) * update_gradient_score.to(model.score_per_component.device)

        for param in model.parameters():
            param.grad = None  # This sets the gradient to zero

    # dynamically adapt model
    if global_train_step in model.adaptation_opts["adaptation_global_step"]:
        idx = model.adaptation_opts["adaptation_global_step"].index(global_train_step)
        replace_pct = model.adaptation_opts["lst_replace_pct"][idx]
        total_cost = model.adaptation_opts["total_cost"]
        _, sorted_indices = torch.sort(model.score_per_component, descending=True)

        tmp_rm_cost = 0
        i = len(sorted_indices) - 1
        remove_component_indices = []
        while tmp_rm_cost <= (total_cost - model.adaptation_opts["rest_cost"]) * replace_pct:
            remove_component_indices.append(sorted_indices[i])
            tmp_rm_cost += model.lst_cmp_cost[sorted_indices[i]]
            i -= 1

        tmp_add_cost = 0
        i = 0
        duplicate_component_indices = []
        while (
            tmp_add_cost <= (total_cost - model.adaptation_opts["rest_cost"]) * replace_pct
            or tmp_add_cost <= tmp_rm_cost
        ):
            duplicate_component_indices.append(sorted_indices[i])
            tmp_add_cost += model.lst_cmp_cost[sorted_indices[i]]
            i += 1

        selected_indices = list(range(model.total_num_components))
        selected_indices = [i for i in selected_indices if i not in remove_component_indices]
        selected_indices += duplicate_component_indices
        selected_indices = sorted(int(x) for x in selected_indices)

        new_component_dist = {"ff1": [], "cgmlp": [], "mhsa": [], "ff2": []}
        new_lst_cmp_cost = []
        block_modules = ["ff1", "cgmlp", "mhsa", "ff2"]
        cur_indx = 0
        selected_idx = 0
        for block_idx in range(len(model.e_branchformer.module_list)):
            for mod_idx in range(len(block_modules)):
                module = block_modules[mod_idx]
                gates = []
                temp_idx = cur_indx + model.component_dist[module][block_idx]
                num_chunks = model.component_dist[module][block_idx]
                for cmp_idx in range(cur_indx, temp_idx):
                    if cmp_idx in selected_indices:
                        gates.append(True)
                        num_chunks += selected_indices.count(cmp_idx) - 1
                    else:
                        gates.append(False)
                        num_chunks -= 1

                new_component_dist[module].append(num_chunks)
                if num_chunks > 0:
                    new_lst_cmp_cost += [model.adaptation_opts["dict_module_cost"][module]] * num_chunks

                new_weights_noise_var = (
                    0
                    if "new_weights_noise_var" not in model.adaptation_opts
                    else model.adaptation_opts["new_weights_noise_var"]
                )

                if module == "ff1":
                    if len(gates) > 0:
                        dim_per_chunk = model.e_branchformer.module_list[block_idx].ff_1.linear_ff_weight.size()[
                            0
                        ] // len(gates)
                        model.e_branchformer.module_list[block_idx].ff_1.double_and_prune_params(
                            dim_per_chunk * num_chunks,
                            [e - cur_indx for e in selected_indices[selected_idx : selected_idx + num_chunks]],
                            new_weights_noise_var,
                        )
                elif module == "ff2":
                    if len(gates) > 0:
                        dim_per_chunk = model.e_branchformer.module_list[block_idx].ff_2.linear_ff_weight.size()[
                            0
                        ] // len(gates)
                        model.e_branchformer.module_list[block_idx].ff_2.double_and_prune_params(
                            dim_per_chunk * num_chunks,
                            [e - cur_indx for e in selected_indices[selected_idx : selected_idx + num_chunks]],
                            new_weights_noise_var,
                        )
                elif module in ["cgmlp"]:
                    if len(gates) > 0:
                        chunk_dim = model.e_branchformer.module_list[block_idx].cgmlp.linear_out_weights.size(1) // len(
                            gates
                        )
                        model.e_branchformer.module_list[block_idx].cgmlp.double_and_prune_params(
                            chunk_dim * num_chunks,
                            [e - cur_indx for e in selected_indices[selected_idx : selected_idx + num_chunks]],
                            new_weights_noise_var,
                        )
                elif module in ["mhsa"]:
                    model.e_branchformer.module_list[block_idx].mhsa.mhsa.double_and_prune_params(
                        num_chunks,
                        [e - cur_indx for e in selected_indices[selected_idx : selected_idx + num_chunks]],
                        new_weights_noise_var,
                    )

                cur_indx = temp_idx
                selected_idx += num_chunks

        model.to("cuda")

        model.component_dist = new_component_dist
        model.score_per_component = torch.nn.Parameter(
            torch.FloatTensor(torch.zeros((len(new_lst_cmp_cost)))), requires_grad=False
        )
        model.lst_cmp_cost = torch.nn.Parameter(torch.FloatTensor(new_lst_cmp_cost), requires_grad=False)
        model.total_num_components = len(new_lst_cmp_cost)

    from returnn.tensor import batch_dim
    import returnn.frontend as rf

    rf.get_run_ctx().mark_as_loss(
        name="CTC", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )


def get_recog_serializer(
    model_config: EbranchformerCTCConfig,
) -> Collection:
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{EbranchformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.export"),
            Import(
                "i6_models.assemblies.dynamic_adaptable_e_branchformer.e_branchformer_v1.EbranchformerBlockV1Config"
            ),
            Import("torch.nn.SiLU"),
            Import("i6_models.parts.dynamic_adaptable_conformer.ConformerMHSAV1Config"),
            Import("i6_models.parts.dynamic_adaptable_conformer.ConformerPositionwiseFeedForwardV1Config"),
            Import("i6_models.parts.dynamic_adaptable_e_branchformer.ConvolutionalGatingMLPV1Config"),
            Import("i6_models.parts.dynamic_adaptable_e_branchformer.MergerV1Config"),
        ],
    )


def get_prior_serializer(
    model_config: EbranchformerCTCConfig,
) -> Collection:
    berger_pytorch_package = "i6_experiments.users.berger.pytorch"
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{EbranchformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{berger_pytorch_package}.forward.basic.forward_step"),
            Import(
                f"{berger_pytorch_package}.forward.prior_callback.ComputePriorCallback", import_as="forward_callback"
            ),
            Import(
                "i6_models.assemblies.dynamic_adaptable_e_branchformer.e_branchformer_v1.EbranchformerBlockV1Config"
            ),
            Import("torch.nn.SiLU"),
            Import("i6_models.parts.dynamic_adaptable_conformer.ConformerMHSAV1Config"),
            Import("i6_models.parts.dynamic_adaptable_conformer.ConformerPositionwiseFeedForwardV1Config"),
            Import("i6_models.parts.dynamic_adaptable_e_branchformer.ConvolutionalGatingMLPV1Config"),
            Import("i6_models.parts.dynamic_adaptable_e_branchformer.MergerV1Config"),
        ],
    )


def get_train_serializer(
    model_config: EbranchformerCTCConfig,
) -> Collection:
    # pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{EbranchformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.train_step"),
            Import(
                "i6_models.assemblies.dynamic_adaptable_e_branchformer.e_branchformer_v1.EbranchformerBlockV1Config"
            ),
            Import("torch.nn.SiLU"),
            Import("i6_models.parts.dynamic_adaptable_conformer.ConformerMHSAV1Config"),
            Import("i6_models.parts.dynamic_adaptable_conformer.ConformerPositionwiseFeedForwardV1Config"),
            Import("i6_models.parts.dynamic_adaptable_e_branchformer.ConvolutionalGatingMLPV1Config"),
            Import("i6_models.parts.dynamic_adaptable_e_branchformer.MergerV1Config"),
        ],
    )
