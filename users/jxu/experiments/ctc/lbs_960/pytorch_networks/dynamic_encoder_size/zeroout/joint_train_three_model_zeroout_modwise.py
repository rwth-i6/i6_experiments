from dataclasses import dataclass
from typing import Optional
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant

import torch
from torch import nn
from typing import Tuple
import numpy as np
import copy

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
from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks.components.stochastic_depth import \
    StochasticDepth
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_experiments.users.berger.pytorch.models.util import lengths_to_padding_mask

from i6_experiments.users.berger.pytorch.custom_parts import specaugment


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
        self.stochastic_depth_list = torch.nn.ModuleList([StochasticDepth(p=layer_dropout, mode="row") for _ in range(4)])
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
            x = self.conv(x)*module_gates[1] + x  # [B, T, F]
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
            x = 0.5 * self.ff2(x)*module_gates[3] + x  # [B, T, F]
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

    def __init__(self, cfg: ConformerEncoderV1Config, small_model_num_mods: int, medium_model_num_mods: int, layer_dropout: float, gate_activation:str):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList(
            [ConformerBlockV1(cfg.block_cfg, layer_dropout) for _ in range(cfg.num_layers)])
        self.small_model_num_mods = small_model_num_mods
        self.medium_model_num_mods = medium_model_num_mods

        self.register_buffer("medium_model_indicies", torch.tensor([-1]*medium_model_num_mods))
        # one gate for each module in the block
        self.gate_activation = gate_activation
        if self.gate_activation == "sigmoid":
            self.gates = torch.nn.Parameter(torch.FloatTensor([10.0] * (cfg.num_layers * 4)))
        elif self.gate_activation == "identity":
            self.gates = torch.nn.Parameter(torch.FloatTensor([1.0] * (cfg.num_layers * 4)))
        self.register_buffer("idx_iter", torch.tensor(0))
        # self.gates.data.normal_(1.0, 0.01)

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, stage_args: dict, global_train_step: int) -> Tuple[
        list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        stage_1_num_steps_per_iter = stage_args["stage_1_num_steps_per_iter"]
        stage_1_cum_steps_per_iter = np.cumsum(stage_1_num_steps_per_iter)
        stage_1_num_dropout_per_iter = copy.deepcopy(stage_args["stage_1_num_zero_per_iter"])
        assert len(stage_1_num_steps_per_iter) == len(stage_args["stage_1_num_zero_per_iter"])
        # since the zeroout is always applied at the end of each iter
        stage_1_num_dropout_per_iter.insert(0, 0)
        stage_1_layer_dropout_on_large = True if "stage_1_layer_dropout_on_large" not in stage_args else stage_args["stage_1_layer_dropout_on_large"]

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']
        y = x
        z = x

        total_utilised_mods = torch.tensor(0.0).to('cuda')

        gates_after_act = self.gates
        if self.gate_activation == "sigmoid":
            gates_after_act = torch.sigmoid(self.gates)

        if self.training:
            # stage 1: iteratively increase the sparsity and zero out the smallest weights at the end of the iteration
            if global_train_step <= stage_1_cum_steps_per_iter[-1]:
                idx_iter = 0
                while global_train_step > stage_1_cum_steps_per_iter[idx_iter]:
                    idx_iter += 1
                self.idx_iter = torch.tensor(idx_iter)

                _, dropout_mods_indices = torch.topk(self.gates, k=stage_1_num_dropout_per_iter[idx_iter], largest=False)

                # for the large model
                # TODO whether to apply layer dropout
                for i in range(len(self.module_list)):
                    dropout_indices = []
                    if stage_1_layer_dropout_on_large:
                        for j in range(4):
                            if 4*i + j in dropout_mods_indices:
                                dropout_indices.append(j)
                    x = self.module_list[i](x, sequence_mask, torch.tensor([1, 1, 1, 1]), dropout_indices)

                # for the small model
                for i in range(len(self.module_list)):
                    y = self.module_list[i](y, sequence_mask, module_gates=gates_after_act[4 * i:4 * i + 4],
                                            dropout_indices=[])

                total_utilised_mods += torch.sum(gates_after_act)


            # stage 2: joint train the large model with a fixed small model
            else:
                _, dropout_mods_indices = torch.topk(self.gates, k=48-self.small_model_num_mods, largest=False)
                _, small_model_indices = torch.topk(self.gates, k=self.small_model_num_mods)

                # large model
                for i in range(len(self.module_list)):
                    dropout_indices = []
                    for j in range(4):
                        if 4*i + j in dropout_mods_indices:
                            dropout_indices.append(j)

                    x = self.module_list[i](x, sequence_mask, torch.tensor([1, 1, 1, 1]),
                                            dropout_indices)

                # small model
                for i in range(len(self.module_list)):
                    dropout_indices = []
                    for j in range(4):
                        if 4*i + j in dropout_mods_indices:
                            dropout_indices.append(j)
                    y = self.module_list[i](y, sequence_mask, module_gates=torch.tensor([1, 1, 1, 1]),
                                            dropout_indices=dropout_indices, direct_jump=True)

                # for medium model
                for i in range(len(self.module_list)):
                    dropout_indices = []
                    for j in range(4):
                        if 4 * i + j not in self.medium_model_indicies:
                            dropout_indices.append(j)
                    z = self.module_list[i](z, sequence_mask, module_gates=torch.tensor([1, 1, 1, 1]),
                                            dropout_indices=dropout_indices, direct_jump=True)

            if global_train_step % 200 == 0:
                print(f"iter idx: {self.idx_iter}")
                print("zeroout_mods_indices: {}".format(sorted([int(i + 1) for i in dropout_mods_indices])))
                num_FFN_1, num_CONV, num_MHSA, num_FFN_2 = self.cal_num_mods(dropout_mods_indices)
                print(f"num_FFN_1:{num_FFN_1}, num_CONV:{num_CONV}, num_MHSA:{num_MHSA}, num_FFN_2:{num_FFN_2}")
                print(f"medium_model_indicies: {self.medium_model_indicies}")
                print("gates: {}".format(self.gates))
                print("gates after activation: {}".format(gates_after_act))
                print("sum sigmoid gates: {}".format(torch.sum(gates_after_act)))

        else:
            _, small_model_zeroout_mods_indices = torch.topk(self.gates, k=48-self.small_model_num_mods, largest=False)
            # large model
            for i in range(len(self.module_list)):
                dropout_indices = []
                for j in range(4):
                    if 4 * i + j in small_model_zeroout_mods_indices:
                        dropout_indices.append(j)
                x = self.module_list[i](x, sequence_mask, torch.tensor([1, 1, 1, 1]),
                                        dropout_indices)

            # small model
            for i in range(len(self.module_list)):
                dropout_indices = []
                for j in range(4):
                    if 4 * i + j in small_model_zeroout_mods_indices:
                        dropout_indices.append(j)
                y = self.module_list[i](y, sequence_mask, module_gates=torch.tensor([1, 1, 1, 1]),
                                        dropout_indices=dropout_indices, direct_jump=True)


            # for medium model
            for i in range(len(self.module_list)):
                dropout_indices = []
                for j in range(4):
                    if 4 * i + j not in self.medium_model_indicies:
                        dropout_indices.append(j)
                z = self.module_list[i](z, sequence_mask, module_gates=torch.tensor([1, 1, 1, 1]),
                                        dropout_indices=dropout_indices, direct_jump=True)

        return [y, x, z], total_utilised_mods, sequence_mask

    def cal_num_mods(selfs, indices):
        num_FFN_1, num_CONV, num_MHSA, num_FFN_2 = 0, 0, 0, 0
        for idx in indices:
            if idx%4 == 0:
                num_FFN_1 += 1
            elif idx%4 == 1:
                num_CONV += 1
            elif idx%4 == 2:
                num_MHSA += 1
            elif idx%4 == 3:
                num_FFN_2 += 1
        return num_FFN_1, num_CONV, num_MHSA, num_FFN_2


@dataclass
class ConformerCTCConfig(ModelConfiguration):
    feature_extraction_cfg: LogMelFeatureExtractionV1Config
    specaugment_cfg: specaugment.SpecaugmentByLengthConfigV1
    conformer_cfg: ConformerEncoderV1Config
    target_size: int
    stage_args: dict
    small_model_num_mods: int
    medium_model_num_mods: int
    layer_dropout: float
    sparsity_loss_scale: float
    recog_num_mods: int


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerCTCConfig, **kwargs):
        super().__init__()
        self.logmel_feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_cfg)
        self.specaugment = specaugment.SpecaugmentByLengthModuleV1(cfg=cfg.specaugment_cfg)
        self.conformer = ConformerEncoderV1(cfg.conformer_cfg, small_model_num_mods=cfg.small_model_num_mods, medium_model_num_mods=cfg.medium_model_num_mods,
                                            layer_dropout=cfg.layer_dropout, gate_activation=cfg.stage_args["gate_activation"])
        self.final_linear_list = torch.nn.ModuleList(
            [nn.Linear(cfg.conformer_cfg.block_cfg.ff_cfg.input_dim, cfg.target_size) for _ in
             range(3)])
        self.stage_args = cfg.stage_args
        self.small_model_num_mods = cfg.small_model_num_mods
        self.medium_model_num_mods = cfg.medium_model_num_mods
        self.small_model_layers = None
        self.sparsity_loss_scale = cfg.sparsity_loss_scale
        self.recog_num_mods = cfg.recog_num_mods
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

        # sequence_mask = lengths_to_padding_mask((audio_features_len + 2) // 3)
        conformer_out_list, total_utilised_mods, sequence_mask = self.conformer(x, sequence_mask, self.stage_args,
                                                                                global_train_step)  # [B, T, F]

        log_probs_list = []
        for i in range(3):
            logits = self.final_linear_list[i](conformer_out_list[i])  # [B, T, F]
            log_probs = torch.log_softmax(logits, dim=2)
            log_probs_list.append(log_probs)

        if self.training:
            return log_probs_list, total_utilised_mods, sequence_mask

        if self.recog_num_mods == self.small_model_num_mods:
            return log_probs_list[0]
        elif self.recog_num_mods == 48:
            return log_probs_list[1]
        elif self.recog_num_mods == self.medium_model_num_mods:
            return log_probs_list[2]


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
    num_layers =  12 if "num_layers" not in network_args else network_args["num_layers"]
    stage_args = network_args["stage_args"]
    small_model_num_mods = network_args["small_model_num_mods"]
    medium_model_num_mods = network_args["medium_model_num_mods"]
    layer_dropout = network_args["layer_dropout"]
    sparsity_loss_scale = network_args["sparsity_loss_scale"]
    recog_num_mods = network_args["recog_num_mods"]

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
        stage_args=stage_args,
        small_model_num_mods=small_model_num_mods,
        medium_model_num_mods=medium_model_num_mods,
        layer_dropout=layer_dropout,
        sparsity_loss_scale=sparsity_loss_scale,
        recog_num_mods=recog_num_mods
    )


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, global_train_step, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    targets = extern_data["targets"].raw_tensor.long()
    targets_len = extern_data["targets"].dims[1].dyn_size_ext.raw_tensor

    model.train()

    log_probs_list, total_utilised_mods, sequence_mask = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
        global_train_step=global_train_step
    )
    sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)

    # make sure the layers ordering is right

    # TODO it can be set!
    loss_layers = [model.small_model_num_mods, 48, model.medium_model_num_mods]
    loss_scales = [0.3, 1, 0.3]

    # add the utilisation loss
    stage_1_num_steps_per_iter = model.stage_args["stage_1_num_steps_per_iter"]
    stage_1_cum_steps_per_iter = np.cumsum(stage_1_num_steps_per_iter)
    stage_1_num_zero_per_iter = model.stage_args["stage_1_num_zero_per_iter"]
    stage_1_expected_sparsity_per_iter = model.stage_args["stage_1_expected_sparsity_per_iter"]
    stage_1_small_model_loss_scale = 0.3 if "stage_1_small_model_loss_scale" not in model.stage_args else model.stage_args["stage_1_small_model_loss_scale"]
    stage_1_large_model_loss_scale = 1 if "stage_1_large_model_loss_scale" not in model.stage_args else model.stage_args["stage_1_large_model_loss_scale"]
    stage_2_small_model_loss_scale = 0.3 if "stage_2_small_model_loss_scale" not in model.stage_args else model.stage_args["stage_2_small_model_loss_scale"]
    stage_2_large_model_loss_scale = 1 if "stage_2_large_model_loss_scale" not in model.stage_args else model.stage_args["stage_2_large_model_loss_scale"]
    zeroout_val = model.stage_args["zeroout_val"]

    assert len(stage_1_num_steps_per_iter) == len(stage_1_num_zero_per_iter)

    # -------------------------------------- stage 1 -------------------------------------------------------

    if global_train_step <= stage_1_cum_steps_per_iter[-1]:
        idx_iter = int(model.conformer.idx_iter)
        sparsity_loss = torch.abs((48-total_utilised_mods)/48 - stage_1_expected_sparsity_per_iter[idx_iter])
        rf.get_run_ctx().mark_as_loss(name=f"sparsity_loss", loss=sparsity_loss, scale=model.sparsity_loss_scale)
        if global_train_step == stage_1_cum_steps_per_iter[idx_iter]:
            # should get the indices before zeroout for the final step
            if global_train_step == stage_1_cum_steps_per_iter[-1]:
                # model.conformer.medium_model_indicies = torch.tensor([x for x in range(0,48) if x not in zeroout_mods_indices])
                _, model.conformer.medium_model_indicies = torch.topk(model.conformer.gates, k=model.medium_model_num_mods)

            # apply zero at the end of the iteration
            num_zeroout_mods =  stage_1_num_zero_per_iter[idx_iter]
            _, zeroout_mods_indices = torch.topk(model.conformer.gates, k=num_zeroout_mods, largest=False)
            model.conformer.gates.data[zeroout_mods_indices] = zeroout_val

    else:
        loss_scales = [stage_2_small_model_loss_scale, stage_2_large_model_loss_scale, stage_2_small_model_loss_scale]

    if global_train_step <= stage_1_cum_steps_per_iter[-1]:
        output_indices = [0, 1]
    else:
        output_indices = [0, 1, 2]

    for i in output_indices:
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
        rf.get_run_ctx().mark_as_loss(name=f"CTC_{loss_layers[i]}_mods", loss=loss, scale=loss_scales[i])


def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    log_probs = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )  # [B, T, F]
    rf.get_run_ctx().mark_as_output(log_probs, name="log_probs")


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

