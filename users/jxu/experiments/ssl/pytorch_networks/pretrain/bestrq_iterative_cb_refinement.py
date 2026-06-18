from dataclasses import dataclass
from typing import Optional
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant

import torch
from torch import nn
import torch.nn.functional as F
# import torchaudio

from scipy.optimize import linear_sum_assignment

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV2Config
from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV2Config
from i6_models_repo.i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.best_rq import RandomMask, RandomProjectionPCAQuantizer
from i6_models.parts.frontend.common import mask_pool

from i6_experiments.users.berger.pytorch.models.util import lengths_to_padding_mask
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.jxu.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosBlockV1Config, \
    ConformerRelPosEncoderV1Config, ConformerRelPosEncoderV1
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_experiments.common.setups.serialization import Import, NonhashedCode
from i6_experiments.users.jxu.experiments.ssl.pytorch_networks.input_norm import InputNormalization

from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf


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
class BestRQConformerConfig(ModelConfiguration):
    feature_extraction_cfg: LogMelFeatureExtractionV1Config
    conformer_cfg: ConformerRelPosEncoderV1Config
    final_dropout: float
    aux_losses: dict
    internal_subsampling_rate: int
    input_codebook_dim: int
    input_codebook_num_vars: int
    mask_replace_val: str
    mask_prob: float
    mask_length: int
    cb_distance_measure: str
    normalise_after_PCA: bool
    PCA_update_steps: int
    input_quantizer_update_kwargs: dict


class BestRQConformerModel(torch.nn.Module):
    def __init__(self, step: int, cfg: BestRQConformerConfig, **kwargs):
        super().__init__()
        self.logmel_feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_cfg)
        self.input_quantizer = RandomProjectionPCAQuantizer(
            input_dim=cfg.feature_extraction_cfg.num_filters, codebook_dim=cfg.input_codebook_dim,
            codebook_num_vars=cfg.input_codebook_num_vars, distance_meature=cfg.cb_distance_measure,
        )
        self.compute_mask = RandomMask(cfg.feature_extraction_cfg.num_filters, cfg.mask_replace_val, cfg.mask_prob,
                                       cfg.mask_length)
        self.conformer = ConformerRelPosEncoderV1(cfg.conformer_cfg)
        self.final_linear = nn.Linear(cfg.conformer_cfg.block_cfg.ff_cfg.input_dim, cfg.input_codebook_num_vars)
        self.aux_losses = cfg.aux_losses
        self.internal_subsampling_rate = cfg.internal_subsampling_rate
        self.input_norm = InputNormalization()
        self.intermediate_norm = InputNormalization(feat_dim=cfg.conformer_cfg.block_cfg.ff_cfg.input_dim)
        self.new_cluster_sum = nn.Parameter(F.normalize(self.input_quantizer.CB), requires_grad=False)
        self.new_cluster_amount = nn.Parameter(torch.ones(cfg.input_codebook_num_vars), requires_grad=False)
        self.conf_matrix = nn.Parameter(torch.zeros(cfg.input_codebook_num_vars, cfg.input_codebook_num_vars), requires_grad=False)
        self.export_mode = False
        self.forward_step = False
        self.dump_logmel = False
        self.get_predict_label = False
        self.normalise_after_PCA = cfg.normalise_after_PCA
        self.PCA_update_steps = cfg.PCA_update_steps
        self.input_quantizer_update_kwargs = cfg.input_quantizer_update_kwargs
        from returnn.config import get_global_config
        config = get_global_config(return_empty_if_none=True)
        self.forward_output_layer = config.typed_value("forward_output_layer")
        self.get_intermediate_repr = False


    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: Optional[torch.Tensor] = None,
            global_train_steps: Optional[int] = None
    ):
        with torch.no_grad():
            squeezed_features = torch.squeeze(audio_features)
            if self.export_mode:
                squeezed_features = squeezed_features.type(torch.FloatTensor)
            else:
                squeezed_features = squeezed_features.type(torch.cuda.FloatTensor)

            audio_features, audio_features_len = self.logmel_feat_extraction(squeezed_features, audio_features_len)

        sequence_mask = lengths_to_padding_mask(audio_features_len)

        # apply masking and get targets
        inp_len = sequence_mask.sum(dim=1)
        normalised_audio_features = self.input_norm(audio_features, inp_len)

        targets = self.input_quantizer(normalised_audio_features, sequence_mask, self.normalise_after_PCA,
                                       self.PCA_update_steps, global_train_steps)
        masked_audio_features, mask = self.compute_mask.forward(audio_features, ~sequence_mask)

        if self.forward_step:
            return targets, sequence_mask

        out = []
        # forward Conformer
        x, sequence_mask = self.conformer.frontend(masked_audio_features, sequence_mask)  # [B, T, F']

        # since the input features are downsampled in the frond-end, we need to downsample the masks accordingly
        downsampled_mask = mask_pool(
            mask,
            kernel_size=self.conformer.frontend.pool1.kernel_size[0],
            stride=self.conformer.frontend.pool1.stride[0],
            padding=self.conformer.frontend.pool1.padding[0],
        )

        downsampled_mask = mask_pool(
            downsampled_mask,
            kernel_size=self.conformer.frontend.pool2.kernel_size[0],
            stride=self.conformer.frontend.pool2.stride[0],
            padding=self.conformer.frontend.pool2.padding[0],
        )

        # downsample the targets
        downsampled_targets = targets[:, :: self.internal_subsampling_rate]
        masked_targets = downsampled_targets[downsampled_mask]
        print("input mask", len(masked_targets) / torch.sum(sequence_mask))

        if self.get_predict_label:
            return downsampled_targets, sequence_mask

        return_layers = [int(i) - 1 for i in self.aux_losses.keys()]

        assert (
                max(return_layers) < len(self.conformer.module_list) and min(return_layers) >= 0
        ), f"invalid layer index, should be between 0 and {len(self.module_list) - 1}"

        for i in range(len(self.conformer.module_list)):
            x = self.conformer.module_list[i](x, sequence_mask)  # [B, T, F']
            if self.get_intermediate_repr and i+1 == self.forward_output_layer:
                print("x.size", x.size())
                return x

        conformer_out = x

        targets_list = [masked_targets]

        logits = self.final_linear(conformer_out)  # [B, T, F]
        log_probs = torch.log_softmax(logits, dim=2)
        # if self.get_predict_label and i == 0:
        #     return torch.argmax(log_probs, -1), sequence_mask
        masked_log_probs = log_probs[downsampled_mask]
        out.append(masked_log_probs)

        # update the new cluster statistics
        input_quantizer_update_steps = torch.tensor(self.input_quantizer_update_kwargs["input_quantizer_update_steps"])
        compute_statistics_steps =  torch.tensor(self.input_quantizer_update_kwargs["compute_statistics_steps"])

        if self.training and ((global_train_steps >= (input_quantizer_update_steps - compute_statistics_steps)) & (global_train_steps <= input_quantizer_update_steps)).any():
            print("update cluster")

            if self.input_quantizer_update_kwargs["update_mask"] == "sequence_mask":
                update_mask = sequence_mask
            elif self.input_quantizer_update_kwargs["update_mask"] == "unmask":
                update_mask = sequence_mask & (~downsampled_mask)
            elif self.input_quantizer_update_kwargs["update_mask"] == "mask":
                update_mask = sequence_mask & downsampled_mask

            update_from_labels = downsampled_targets

            with torch.no_grad():
                for a, b in zip(downsampled_targets[update_mask], update_from_labels[update_mask]):
                    self.conf_matrix[a, b] += 1

                tmp_clsuter_sum = torch.zeros(self.new_cluster_sum.size()).to(x.device)
                tmp_clsuter_sum = tmp_clsuter_sum.index_add(0, update_from_labels[update_mask],
                                                            normalised_audio_features[:, :: self.internal_subsampling_rate][update_mask]@ self.input_quantizer.PCA_components.T)
                tmp_cluster_amount = torch.bincount(update_from_labels[update_mask], minlength=len(self.new_cluster_amount)).to(self.new_cluster_amount.device)
                # update_decay = 0.9 * tmp_cluster_amount.clamp(max=1)
                # update_decay = update_decay.masked_fill(update_decay == 0, 1)
                self.new_cluster_sum += tmp_clsuter_sum
                self.new_cluster_amount += tmp_cluster_amount

                print("new_cluster_sum", self.new_cluster_sum)
                print("new_cluster_amount", self.new_cluster_amount)

                if global_train_steps in input_quantizer_update_steps:
                    cost_matrix = -self.conf_matrix.cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    print("linear assignment row ind", row_ind)
                    print("linear assignment col ind", col_ind)
                    new_cb = self.new_cluster_sum.to("cuda")/self.new_cluster_amount.to("cuda").unsqueeze(1)
                    self.input_quantizer.CB = new_cb[col_ind]

                    # re-initialize
                    self.new_cluster_sum.data = F.normalize(self.input_quantizer.CB)
                    self.new_cluster_amount.data = torch.ones(len(row_ind))
                    self.conf_matrix.data = torch.zeros(len(row_ind), len(row_ind))

        return out, targets_list


def get_serializer(model_config: BestRQConformerConfig, variant: ConfigVariant) -> Collection:
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.ALIGN:
        return get_recog_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def get_default_config_v1(num_inputs: int, num_outputs: int, network_args) -> BestRQConformerConfig:
    aux_losses = dict(sorted(network_args["aux_losses"].items(), key=lambda x: int(x[0])))
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
    att_weights_dropout = 0.1 if "att_weights_dropout" not in network_args else network_args["att_weights_dropout"]
    kernel_size = 31 if "kernel_size" not in network_args else network_args["kernel_size"]
    num_layers = 12 if "num_layers" not in network_args else network_args["num_layers"]
    final_dropout = 0.1 if "final_dropout" not in network_args else network_args["final_dropout"]
    dropout = 0.1 if "dropout" not in network_args else network_args["dropout"]
    d_model = 512 if "d_model" not in network_args else network_args["d_model"]
    vgg_act = "relu"
    mask_replace_val = network_args["mask_replace_val"]
    mask_prob = network_args["mask_prob"]
    mask_length = network_args["mask_length"]
    internal_subsampling_rate = network_args["internal_subsampling_rate"]
    cb_distance_measure = network_args["cb_distance_measure"]
    normalise_after_PCA = network_args["normalise_after_PCA"]
    PCA_update_steps = network_args["PCA_update_steps"]
    input_codebook_dim = network_args["input_codebook_dim"]
    input_codebook_num_vars = network_args["input_codebook_num_vars"]
    input_quantizer_update_kwargs = network_args["input_quantizer_update_kwargs"]
    pool2_stride = 4 if "pool2_stride" not in network_args else network_args["pool2_stride"]

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
        pool2_stride=(pool2_stride, 1),
        pool2_padding=None,
        activation=torch.nn.ReLU() if vgg_act == "relu" else torch.nn.SiLU(),
        out_features=d_model,
    )

    frontend = ModuleFactoryV1(VGG4LayerActFrontendV1, frontend_cfg)

    ff_cfg = ConformerPositionwiseFeedForwardV2Config(
        input_dim=d_model,
        hidden_dim=d_model * 4,
        dropout=dropout,
        activation=torch.nn.SiLU(),
        dropout_broadcast_axes=None
    )

    mhsa_cfg = ConformerMHSARelPosV1Config(
        input_dim=d_model,
        num_att_heads=d_model // 64,
        with_bias=True,
        att_weights_dropout=att_weights_dropout,
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
        dropout=dropout,
        dropout_broadcast_axes=None
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

    return BestRQConformerConfig(
        feature_extraction_cfg=feature_extraction_cfg,
        conformer_cfg=conformer_cfg,
        final_dropout=final_dropout,
        aux_losses=aux_losses,
        input_codebook_dim=input_codebook_dim,
        input_codebook_num_vars=input_codebook_num_vars,
        mask_replace_val=mask_replace_val,
        mask_prob=mask_prob,
        mask_length=mask_length,
        internal_subsampling_rate=internal_subsampling_rate,
        cb_distance_measure=cb_distance_measure,
        normalise_after_PCA=normalise_after_PCA,
        PCA_update_steps=PCA_update_steps,
        input_quantizer_update_kwargs=input_quantizer_update_kwargs
    )


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, global_train_step: int, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    logits_list, targets_list = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
        global_train_steps=global_train_step
    )


    for i in range(len(targets_list)):
        print(f"targets {i}", targets_list[i])
        print(f"targets {i} size", targets_list[i].size())
        print(f"targets {i} uniq vals", len(torch.unique(targets_list[i])))

    loss = torch.nn.CrossEntropyLoss(reduction="none")(logits_list[0], targets_list[0].long())
    rf.get_run_ctx().mark_as_loss(
        name="ce_logmel_12", loss=loss, scale=1
    )

    frame_error = torch.argmax(logits_list[0].data, dim=-1).not_equal(targets_list[0].data)
    rf.get_run_ctx().mark_as_loss(name="fer_logmel_12", loss=frame_error,
                                  as_error=True)


def get_train_serializer(
        model_config: BestRQConformerConfig,
) -> Collection:
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{BestRQConformerModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.train_step"),
        ],
    )


def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    # model.forward_step = True
    # model.get_predict_label = True
    # model.dump_logmel = True
    model.get_intermediate_repr = True

    targets = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )
    print("targets", targets)
    rf.get_run_ctx().mark_as_output(targets, name="targets")


def get_prior_serializer(
        model_config: BestRQConformerConfig,
) -> Collection:
    pytorch_package = "i6_experiments.users.jxu.experiments.pretrain"

    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{BestRQConformerModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            NonhashedCode("import returnn.frontend as rf\n"),
            Import(f"{__name__}.forward_step"),
            # Import(f"{pytorch_package}.forward.get_intermediate_layer_repr_callback.GetIntermediateLayerReprCallback", import_as="forward_callback"),
            Import(f"{pytorch_package}.forward.get_cluster_id.GetClusterIDCallback", import_as="forward_callback"),
        ],
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


def get_recog_serializer(
        model_config: BestRQConformerConfig,
) -> Collection:
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{BestRQConformerModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.export"),
        ],
    )
