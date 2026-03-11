from dataclasses import dataclass, field
from typing import Optional, Tuple
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant

import torch
from torch import nn
import abc

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV2Config
from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV2Config
from i6_models_repo.i6_models.parts.conformer.norm import LayerNormNC
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_experiments.users.berger.pytorch.models.util import lengths_to_padding_mask
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.jxu.pytorch.serializers.basic import (
    get_basic_pt_network_serializer_v2,
)
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosBlockV1Config, ConformerRelPosEncoderV1Config, ConformerRelPosBlockV1,ConformerRelPosEncoderV1
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_experiments.common.setups.serialization import Import, NonhashedCode


from returnn.tensor.tensor_dict import TensorDict


class AbstractComputeBias(torch.nn.Module, abc.ABC):
    def __init__(self, start_step: int = 0):
        super().__init__()
        self.start_step = start_step

    @abc.abstractmethod
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: Optional[torch.Tensor] = None, 
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        step: Optional[int] = None
    ) -> torch.Tensor:
        pass

class SoftCTCComputeBias(AbstractComputeBias):
    def __init__(
        self,
        scale: float = 1.0,
        method: str = "tanh",
        start_step: int = 0,
        # for later tuning/ablation
        temperature: float = 1.0,
        use_query_gating: bool = True,
        use_diagonal_protection: bool = False, # more for without query gating
    ):
        super().__init__(start_step=start_step)
        self.scale = scale
        self.method = method 
        self.temperature = temperature
        self.use_query_gating = use_query_gating
        self.use_diagonal_protection = use_diagonal_protection

    def forward(self, logits: torch.Tensor, step: Optional[int]=None, **kwargs) -> torch.Tensor:
        # in inference, step will be None, masking will be used(for consistency)
        if step is not None and step < self.start_step:
            return None
        blank_logits = logits[:, :, 0]
        if self.temperature != 1.0:
            blank_logits = blank_logits / self.temperature 

        if self.method == "sigmoid":
            key_term = -self.scale * torch.sigmoid(blank_logits)
        elif self.method == "logit":
            key_term = -self.scale * blank_logits
        elif self.method == "tanh":
            key_term = -self.scale * torch.tanh(blank_logits)
        elif self.method == "prob":
            probs = torch.softmax(logits, dim=-1)
            prob_blank = probs[:, :, 0]
            mapped_non_blank = 2 * (1.0 - prob_blank) - 1 # map to [-1, 1]
            key_term = -self.scale * mapped_non_blank
        else:
            return None

        key_term = key_term.unsqueeze(1) # [B, 1, T]

        # apply query gating to avoid the biasing for blank positions
        if self.use_query_gating:
            probs = torch.softmax(logits, dim=-1)
            prob_blank = probs[:, :, 0]
            query_gate = (1.0 - prob_blank).unsqueeze(2) # [B, T, 1]
            bias = query_gate * key_term # [B, T, T]
        else:
            bias = key_term # [B, 1, T]

        # diagonal protection to skip biasing at
        if self.use_diagonal_protection:
            T = blank_logits.shape[1]
            
            if bias.dim() == 3:
                bias = bias.expand(-1, T, -1)
            
            # 0 on diagonal, 1 off-diagonal
            eye = torch.eye(T, device=bias.device, dtype=bias.dtype)
            off_diag_mask = 1.0 - eye 
            
            bias = bias * off_diag_mask

        # add head dimension -> [B, 1, T, T] or [B, 1, 1, T]
        bias = bias.unsqueeze(1)
        
        return bias 

class LearnableEmbeddingComputeBias(AbstractComputeBias):
    def __init__(self, embed_dim: int, start_step: int = 0):
        super().__init__(start_step=start_step)
        
        # map 1-d logit to embed_dim of Q vector (d_model // num_heads)
        self.projection = nn.Sequential(
            nn.Linear(1, embed_dim * 2),
            nn.Tanh(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, logits: torch.Tensor, step: Optional[int] = None, **kwargs) -> Optional[torch.Tensor]:
        if step is not None and step < self.start_step:
            return None
        
        blank_logits = logits[:, :, 0:1] 
        k_bias = self.projection(blank_logits)  # [B, T, D]
        k_bias = k_bias.unsqueeze(2)  # [B, T, 1(H), D]
        
        return k_bias

# maybe other variants in the future like using forced alignment

@dataclass
class ConformerCTCConfig(ModelConfiguration):
    feature_extraction_cfg: LogMelFeatureExtractionV1Config
    specaug_args: dict
    conformer_cfg: ConformerRelPosEncoderV1Config
    final_dropout: float
    target_size: int
    aux_losses: dict
    
    bias_layer_index: Optional[int] = None
    compute_bias_type: str = "soft_ctc"
    compute_bias_args: dict = field(default_factory=dict)

class ConformerCTCModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerCTCConfig, **kwargs):
        super().__init__()
        self.logmel_feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_cfg)
        self.specaug_args = cfg.specaug_args
        
        self.aux_losses = cfg.aux_losses
        self.final_linear_list = torch.nn.ModuleList(
            [nn.Linear(cfg.conformer_cfg.block_cfg.ff_cfg.input_dim, cfg.target_size) for _ in range(len(cfg.aux_losses))])
        
        self.bias_idx = cfg.bias_layer_index
        if self.bias_idx is None:
            self.conformer = ConformerRelPosEncoderV1(cfg.conformer_cfg)
            self.encoder_bottom = None
            self.encoder_top = None
            self.compute_bias = None
        else:
            import copy
            self.conformer = None
            
            # Setup Bottom Encoder
            cfg_bottom = copy.deepcopy(cfg.conformer_cfg)
            cfg_bottom.num_layers = self.bias_idx
            self.encoder_bottom = ConformerRelPosEncoderV1(cfg_bottom)
            
            # Setup Top Encoder
            cfg_top = copy.deepcopy(cfg.conformer_cfg)
            cfg_top.num_layers = cfg.conformer_cfg.num_layers - self.bias_idx
            cfg_top.frontend = None # Top has no frontend
            self.encoder_top = ConformerRelPosEncoderV1(cfg_top)
            
            # Setup Bias Strategy
            if cfg.compute_bias_type == "soft_ctc":
                self.compute_bias = SoftCTCComputeBias(**cfg.compute_bias_args)
            elif cfg.compute_bias_type == "learnable_embedding":
                self.compute_bias = LearnableEmbeddingComputeBias(**cfg.compute_bias_args)
            else:
                raise ValueError(f"Unknown compute bias type: {cfg.compute_bias_type}")

        self.export_mode = False

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_len: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
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
        
        log_probs_list = []
        
        if self.bias_idx is None:
            conformer_out_list, sequence_mask = self.conformer(x, sequence_mask, return_layers=returnn_layers)
            for i, feat in enumerate(conformer_out_list):
                logits = self.final_linear_list[i](feat)
                log_probs_list.append(torch.log_softmax(logits.to(torch.float32), dim=-1))
        else:
            # first run bottom encoder
            last_bottom_idx = self.bias_idx - 1 # return_layers is 0-based, bias_idx 1-based
            ret_layers_bottom = [l for l in returnn_layers if l < self.bias_idx]
            if last_bottom_idx not in ret_layers_bottom:
                ret_layers_bottom.append(last_bottom_idx)
            ret_layers_bottom.sort()
            
            out_bottom, sequence_mask = self.encoder_bottom(x, sequence_mask, return_layers=ret_layers_bottom)
            
            # compute bias
            guide_feat = out_bottom[-1]
            sorted_keys = sorted(self.aux_losses.keys(), key=lambda x: int(x))
            guide_key = str(last_bottom_idx + 1)  # covert back to the aux_losses key, which is 1-based int in str
            guide_lin_idx = sorted_keys.index(guide_key)
            guide_logits = self.final_linear_list[guide_lin_idx](guide_feat)
            
            # biasing should not affect gradients of bottom encoder
            bias = self.compute_bias(
                logits=guide_logits.detach(),
                targets=targets,
                input_lengths=audio_features_len,
                target_lengths=target_lengths,
                step=step,
            )
            
            for i, idx in enumerate(ret_layers_bottom):
                 key = str(idx + 1)
                 if key in sorted_keys: 
                     lin_idx = sorted_keys.index(key)
                     logits = self.final_linear_list[lin_idx](out_bottom[i])
                     log_probs_list.append(torch.log_softmax(logits.to(torch.float32), dim=-1))  # convert for amp

            # forward top encoder with attention bias
            ret_layers_top = [l - self.bias_idx for l in returnn_layers if l >= self.bias_idx]
            out_top, _ = self.encoder_top(
                guide_feat, 
                sequence_mask, 
                return_layers=ret_layers_top, 
                attention_bias=bias # pass bias
            )
            for i, idx in enumerate(ret_layers_top):
                real_idx = idx + self.bias_idx
                key = str(real_idx + 1)
                lin_idx = sorted_keys.index(key)
                logits = self.final_linear_list[lin_idx](out_top[i])
                log_probs_list.append(torch.log_softmax(logits.to(torch.float32), dim=-1))

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

    bias_layer_index = network_args.get("bias_layer_index", 8)
    compute_bias_type = network_args.get("compute_bias_type", "soft_ctc")
    compute_bias_args = network_args.get("compute_bias_args", {})

    return ConformerCTCConfig(
        feature_extraction_cfg=feature_extraction_cfg,
        specaug_args=specaug_args,
        conformer_cfg=conformer_cfg,
        target_size=num_outputs,
        final_dropout=final_dropout,
        aux_losses=aux_losses,
        bias_layer_index=bias_layer_index,
        compute_bias_type=compute_bias_type,
        compute_bias_args=compute_bias_args,
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
    
    import returnn.frontend as rf
    from returnn.tensor import batch_dim
    step = rf.get_run_ctx().step

    log_probs_list, sequence_mask = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
        targets=targets,
        target_lengths=targets_len,
        step=step,
    )
    sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)

    # make sure the layers ordering is right
    loss_layers = list(model.aux_losses.keys())
    loss_scales = list(model.aux_losses.values())

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
    return get_basic_pt_network_serializer_v2(
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

    return get_basic_pt_network_serializer_v2(
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
    return get_basic_pt_network_serializer_v2(
        module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{__name__}.export"),
        ],
    )
