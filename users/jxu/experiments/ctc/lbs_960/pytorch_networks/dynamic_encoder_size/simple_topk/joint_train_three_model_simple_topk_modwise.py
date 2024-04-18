from dataclasses import dataclass
from typing import Optional
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant

import torch
from torch import nn
import numpy as np
from typing import Tuple
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf

from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import Import
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
# from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config, ConformerConvolutionV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config, ConformerMHSAV1
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config, ConformerPositionwiseFeedForwardV1
from i6_models.assemblies.conformer import (
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
)
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks.components.stochastic_depth import \
    StochasticDepth
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_experiments.users.berger.pytorch.models.util import lengths_to_padding_mask

from i6_experiments.users.berger.pytorch.custom_parts import specaugment


EPSILON = np.finfo(np.float32).tiny

class SubsetOperator(torch.nn.Module):
    def __init__(self, k):
        super(SubsetOperator, self).__init__()
        self.k = k

    def forward(self, scores):
        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).cuda())
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores)
            khot = khot + onehot_approx

        # straight through
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, self.k)
        khot_hard = khot_hard.scatter_(0, ind, 1)
        res = khot_hard - khot.detach() + khot

        return res


class ConformerBlockV1(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockV1Config, layer_dropout):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.ff1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAV1(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV1(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.stochastic_depth_list = torch.nn.ModuleList(
            [StochasticDepth(p=layer_dropout, mode="row") for _ in range(4)])
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor, module_gates: torch.Tensor,
                dropout_indices: torch.Tensor, direct_jump=False) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :param module_gates:  module gates
        :param dropout_indices: indicies to apply dropout
        :return: torch.Tensor of shape [B, T, F]
        """
        if 0 not in dropout_indices:
            x = 0.5 * self.ff1(x) * module_gates[0] + x  # [B, T, F]
        else:
            if direct_jump:
                x = x
            else:
                x = self.stochastic_depth_list[0](0.5 * self.ff1(x) * module_gates[0]) + x  # [B, T, F]

        if 1 not in dropout_indices:
            x = self.conv(x) * module_gates[1] + x  # [B, T, F]
        else:
            if direct_jump:
                x = x
            else:
                x = self.stochastic_depth_list[1](self.conv(x) * module_gates[1]) + x

        if 2 not in dropout_indices:
            x = self.mhsa(x, sequence_mask) * module_gates[2] + x
        else:
            if direct_jump:
                x = x
            else:
                x = self.stochastic_depth_list[2](self.mhsa(x, sequence_mask) * module_gates[2]) + x

        if 3 not in dropout_indices:
            x = 0.5 * self.ff2(x) * module_gates[3] + x  # [B, T, F]
        else:
            if direct_jump:
                x = x
            else:
                x = self.stochastic_depth_list[3](0.5 * self.ff2(x) * module_gates[3]) + x

        x = self.final_layer_norm(x)  # [B, T, F]
        return x



class ConformerEncoderV1(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderV1Config, small_model_num_mods: int, medium_model_num_mods: int,
                 tau_args: dict,
                 layer_dropout: dict):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList(
            [ConformerBlockV1(cfg.block_cfg, layer_dropout["layer_dropout_mod_select"]) for _ in range(cfg.num_layers)])
        self.sampler = SubsetOperator(k=small_model_num_mods)
        self.small_model_num_mods = small_model_num_mods
        self.medium_model_num_mods = medium_model_num_mods
        self.tau_args = tau_args
        self.layer_dropout = layer_dropout
        self.gates = torch.nn.Parameter(torch.FloatTensor([0.5] * cfg.num_layers*4))

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, start_select_step: int,
                global_train_step: int, k_anneal_args: dict, recog_num_mods=48) -> Tuple[
        list[torch.Tensor], torch.Tensor]:
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
        y = x
        z = x

        k_anneal_num_steps_per_iter = k_anneal_args["k_anneal_num_steps_per_iter"]
        k_reduction_per_iter = k_anneal_args["k_reduction_per_iter"]
        k_anneal_num_iters = (48-self.small_model_num_mods) / k_reduction_per_iter

        if self.training:
            # train the large model only
            if global_train_step < start_select_step:
                for i in range(len(self.module_list)):
                    x = self.module_list[i](x, sequence_mask, module_gates=torch.tensor([1, 1, 1, 1]),
                                            dropout_indices=[])  # [B, T, F']
            # small model selection part
            elif global_train_step <= start_select_step + k_anneal_num_steps_per_iter * k_anneal_num_iters:
                if global_train_step == start_select_step:
                    print("dropout in mod selection: {}".format(self.layer_dropout["layer_dropout_mod_select"]))

                k = int(max((48 - k_reduction_per_iter) - ((
                                                               global_train_step - start_select_step) // k_anneal_num_steps_per_iter * k_reduction_per_iter),
                        self.small_model_num_mods))
                self.sampler.k = k
                gumbel_softmax = self.sampler(self.gates)

                _, remain_mods_indices = torch.topk(gumbel_softmax, k=k)
                _, remove_mods_indices = torch.topk(gumbel_softmax, k=48 - k, largest=False)

                # large model
                for i in range(len(self.module_list)):
                    dropout_indices = []
                    for j in range(4):
                        if 4 * i + j in remove_mods_indices:
                            dropout_indices.append(j)
                    x = self.module_list[i](x, sequence_mask, module_gates=torch.tensor([1, 1, 1, 1]),
                                            dropout_indices=dropout_indices)

                # small model
                for i in range(len(self.module_list)):
                    y = self.module_list[i](y, sequence_mask, module_gates=gumbel_softmax[4 * i:4 * i + 4],
                                            dropout_indices=[])

                if global_train_step % 200 == 0:
                    print(f"small model num_layers: {k}")
                    print("gates: {}".format(self.gates))
                    print("gumbel_softmax: {}".format(gumbel_softmax))
                    print("remain_mods_indices: {}".format(sorted([int(i + 1) for i in remain_mods_indices])))
                    print("remove_mods_indices: {}".format(sorted([int(i + 1) for i in remove_mods_indices])))

            # fix the selection and jointly train small, medium, large models
            else:
                if global_train_step == start_select_step + k_anneal_num_steps_per_iter * k_anneal_num_iters + 1:
                    # change the layer dropout to layer_dropout_fix_mod
                    for i in range(len(self.module_list)):
                        for j in range(len(self.module_list[i].stochastic_depth_list)):
                            self.module_list[i].stochastic_depth_list[j].p = self.layer_dropout["layer_dropout_fix_mod"]
                            print("changed layer dropout in layer_dropout_fix_mod to {}".format(
                                self.layer_dropout["layer_dropout_fix_mod"]))

                _, small_remove_mods_indices = torch.topk(self.gates, k=48 - self.small_model_num_mods, largest=False)
                _, medium_remove_mods_indices = torch.topk(self.gates, k=48 - self.medium_model_num_mods, largest=False)

                # large model
                for i in range(len(self.module_list)):
                    dropout_indices = []
                    for j in range(4):
                        if 4 * i + j in small_remove_mods_indices:
                            dropout_indices.append(j)
                    x = self.module_list[i](x, sequence_mask, module_gates=torch.tensor([1, 1, 1, 1]),
                                            dropout_indices=dropout_indices)

                # small model
                for i in range(len(self.module_list)):
                    dropout_indices = []
                    for j in range(4):
                        if 4 * i + j in small_remove_mods_indices:
                            dropout_indices.append(j)
                    y = self.module_list[i](y, sequence_mask, module_gates=torch.tensor([1, 1, 1, 1]),
                                            dropout_indices=dropout_indices, direct_jump=True)

                # medium model
                for i in range(len(self.module_list)):
                    dropout_indices = []
                    for j in range(4):
                        if 4 * i + j in medium_remove_mods_indices:
                            dropout_indices.append(j)
                    z = self.module_list[i](z, sequence_mask, module_gates=torch.tensor([1, 1, 1, 1]),
                                            dropout_indices=dropout_indices, direct_jump=True)

                if global_train_step % 200 == 0:
                    print(
                        "small remove_mods_indices: {}".format(sorted([int(i + 1) for i in small_remove_mods_indices])))
                    print("medium remove_mods_indices: {}".format(
                        sorted([int(i + 1) for i in medium_remove_mods_indices])))
                    print("gates: {}".format(self.gates))


        else:
            _, small_zeroout_mods_indices = torch.topk(self.gates, k=48 - self.small_model_num_mods, largest=False)
            _, medium_zeroout_mods_indices = torch.topk(self.gates, k=48 - self.medium_model_num_mods, largest=False)

            # large model
            if recog_num_mods == 48:
                for i in range(len(self.module_list)):
                    x = self.module_list[i](x, sequence_mask, torch.tensor([1, 1, 1, 1]), [])

            # small model
            if recog_num_mods == self.small_model_num_mods:
                for i in range(len(self.module_list)):
                    dropout_indices = []
                    for j in range(4):
                        if 4 * i + j in small_zeroout_mods_indices:
                            dropout_indices.append(j)
                    y = self.module_list[i](y, sequence_mask, torch.tensor([1, 1, 1, 1]),
                                            dropout_indices=dropout_indices, direct_jump=True)

            # medium model
            if recog_num_mods == self.medium_model_num_mods:
                for i in range(len(self.module_list)):
                    dropout_indices = []
                    for j in range(4):
                        if 4 * i + j in medium_zeroout_mods_indices:
                            dropout_indices.append(j)
                    z = self.module_list[i](z, sequence_mask, module_gates=torch.tensor([1, 1, 1, 1]),
                                            dropout_indices=dropout_indices, direct_jump=True)
        return [y,x,z], sequence_mask


@dataclass
class ConformerCTCConfig(ModelConfiguration):
    feature_extraction_cfg: LogMelFeatureExtractionV1Config
    specaugment_cfg: specaugment.SpecaugmentByLengthConfigV1
    conformer_cfg: ConformerEncoderV1Config
    target_size: int
    start_select_step: int
    small_model_num_mods: int
    medium_model_num_mods: int
    tau_args: dict
    layer_dropout: dict
    recog_num_mods: int
    k_anneal_args: dict


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerCTCConfig, **kwargs):
        super().__init__()
        self.logmel_feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_cfg)
        self.specaugment = specaugment.SpecaugmentByLengthModuleV1(cfg=cfg.specaugment_cfg)
        self.conformer = ConformerEncoderV1(cfg.conformer_cfg, small_model_num_mods=cfg.small_model_num_mods,
                                            medium_model_num_mods=cfg.medium_model_num_mods,
                                            tau_args=cfg.tau_args,
                                            layer_dropout=cfg.layer_dropout)
        self.final_linear_list = torch.nn.ModuleList(
            [nn.Linear(cfg.conformer_cfg.block_cfg.ff_cfg.input_dim, cfg.target_size) for _ in
             range(3)])
        self.start_select_step = cfg.start_select_step
        self.small_model_num_mods = cfg.small_model_num_mods
        self.medium_model_num_mods = cfg.medium_model_num_mods
        self.small_model_layers = None
        self.recog_num_mods = cfg.recog_num_mods
        self.k_anneal_args = cfg.k_anneal_args
        self.export_mode = False


    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: Optional[torch.Tensor] = None,
            global_train_step: int = 0
    ):
        with torch.no_grad():
            squeezed_features = torch.squeeze(audio_features)
            if self.export_mode:
                squeezed_features = squeezed_features.type(torch.FloatTensor)
            else:
                squeezed_features = squeezed_features.type(torch.cuda.FloatTensor)

            audio_features, audio_features_len = self.logmel_feat_extraction(squeezed_features, audio_features_len)
            x = self.specaugment(audio_features)  # [B, T, F]

        sequence_mask = lengths_to_padding_mask(audio_features_len)
        conformer_out_list, sequence_mask = self.conformer(x, sequence_mask, self.start_select_step, global_train_step,
                                                           self.k_anneal_args, recog_num_mods=self.recog_num_mods)  # [B, T, F]
        if self.training:
            log_probs_list = []
            for i in range(3):
                logits = self.final_linear_list[i](conformer_out_list[i])  # [B, T, F]
                log_probs = torch.log_softmax(logits, dim=2)
                log_probs_list.append(log_probs)
            return log_probs_list, sequence_mask

        i = None
        if self.recog_num_mods == self.small_model_num_mods:
            i = 0
        elif self.recog_num_mods == self.medium_model_num_mods:
            i = 2
        elif self.recog_num_mods == 48:
            i = 1
        logits = self.final_linear_list[i](conformer_out_list[i])
        log_probs = torch.log_softmax(logits, dim=2)
        return log_probs

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
    pytorch_package = __package__.rpartition(".")[0]
    pytorch_package = "i6_experiments.users.berger.pytorch"
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
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


def get_default_config_v1(num_inputs: int, num_outputs: int, network_args: dict) -> ConformerCTCConfig:
    time_max_mask_per_n_frames = network_args["time_max_mask_per_n_frames"]
    freq_max_num_masks = network_args["freq_max_num_masks"]
    vgg_act = network_args["vgg_act"]
    dropout = network_args["dropout"]
    num_layers = network_args["num_layers"]
    small_model_num_mods = network_args["small_model_num_mods"]
    medium_model_num_mods = network_args["medium_model_num_mods"]
    start_select_step = network_args["start_select_step"]
    tau_args = network_args["tau_args"]
    layer_dropout = network_args["layer_dropout"]
    recog_num_mods = network_args["recog_num_mods"]
    k_anneal_args = network_args["k_anneal_args"]

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
        freq_mask_max_size=8
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
        activation=torch.nn.ReLU() if vgg_act=="relu" else torch.nn.SiLU(),
        out_features=d_model,
    )

    frontend = ModuleFactoryV1(VGG4LayerActFrontendV1, frontend_cfg)

    ff_cfg = ConformerPositionwiseFeedForwardV1Config(
        input_dim=d_model,
        hidden_dim=d_model*4,
        dropout=dropout,
        activation=torch.nn.SiLU(),
    )

    mhsa_cfg = ConformerMHSAV1Config(
        input_dim=d_model,
        num_att_heads=d_model//64,
        att_weights_dropout=0.1,
        dropout=0.1,
    )

    conv_cfg = ConformerConvolutionV1Config(
        channels=d_model,
        kernel_size=31,
        dropout=dropout,
        activation=torch.nn.SiLU(),
        norm=LayerNormNC(d_model),
    )

    block_cfg = ConformerBlockV1Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
    )

    conformer_cfg = ConformerEncoderV1Config(
        num_layers=num_layers,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    return ConformerCTCConfig(
        feature_extraction_cfg=feature_extraction_cfg,
        specaugment_cfg=specaugment_cfg,
        conformer_cfg=conformer_cfg,
        target_size=num_outputs,
        start_select_step=start_select_step,
        small_model_num_mods=small_model_num_mods,
        medium_model_num_mods=medium_model_num_mods,
        tau_args=tau_args,
        layer_dropout=layer_dropout,
        recog_num_mods=recog_num_mods,
        k_anneal_args=k_anneal_args
    )


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, global_train_step, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    targets = extern_data["targets"].raw_tensor.long()
    targets_len = extern_data["targets"].dims[1].dyn_size_ext.raw_tensor

    model.train()

    log_probs_list, sequence_mask = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
        global_train_step=global_train_step
    )
    sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)

    # make sure the layers ordering is right
    # TODO it can be set!
    loss_layers = [model.small_model_num_mods, 48, model.medium_model_num_mods]
    loss_scales = [0.3, 1, 0.3]

    k_anneal_num_steps_per_iter = model.k_anneal_args["k_anneal_num_steps_per_iter"]
    k_reduction_per_iter = model.k_anneal_args["k_reduction_per_iter"]
    k_anneal_num_iters = (48-model.small_model_num_mods) / k_reduction_per_iter

    # mod selection stage
    if global_train_step <= model.start_select_step + k_anneal_num_steps_per_iter * k_anneal_num_iters:
        output_indices = [0, 1]
    else:
        # mod fixed stage
        output_indices = [0, 1, 2]

    for i in output_indices:
        log_probs = torch.transpose(log_probs_list[i], 0, 1)  # [T, B, F]

        if not (global_train_step < model.start_select_step and loss_layers[i] != 48):
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

