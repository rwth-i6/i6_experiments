__all__ = ["Model"]

import math
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union, Any

import torch
from torch import Tensor, nn

from i6_models.assemblies.conformer.conformer_rel_pos_v1 import (
    ConformerConvolutionV2Config,
    ConformerMHSARelPosV1,
    ConformerMHSARelPosV1Config,
    ConformerPositionwiseFeedForwardV2Config,
    ConformerRelPosBlockV1Config,
    ConformerRelPosEncoderV1,
    ConformerRelPosEncoderV1Config,
)
from i6_models.assemblies.transformer.transformer_decoder_v1 import (
    CausalSelfAttentionV1Config,
    CrossAttentionV1Config,
    TransformerDecoderBlockV1Config,
    TransformerDecoderV1,
    TransformerDecoderV1Config,
    TransformerDecoderV1State,
)
from i6_models.config import ModuleFactoryV1
from i6_models.parts.decoder import CrossAttentionV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.masked_norm import MaskedBatchNorm1dV1
from i6_models.primitives.feature_extraction import (
    LogMelFeatureExtractionV1,
    LogMelFeatureExtractionV1Config,
    RasrCompatibleLogMelFeatureExtractionV1,
    RasrCompatibleLogMelFeatureExtractionV1Config,
)
from i6_models.primitives.specaugment import specaugment_v1_by_length

from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms.recognition.ctc.forward_step import CtcModel

from .. import PACKAGE


def _relu_sq(x):
    """Squared ReLU."""
    return nn.functional.relu(x) ** 2.0


def _apply_filter(mod: nn.Module, fn: Callable[[nn.Module], bool]) -> nn.Module:
    """
    Applies the given function to `mod` and to its children, if the function
    returned `True` on the invocation on the parent module.
    """
    res = fn(mod)
    if res:
        for ch in mod.children():
            _apply_filter(ch, fn)
    return mod


def _init_rf(mod: nn.Module) -> bool:
    """
    Applies RETURNN frontend-like initialization to `mod`.

    Use with `_apply_filter`.
    """

    if isinstance(mod, CrossAttentionV1):
        # See https://github.com/rwth-i6/returnn/blob/d209de55c468afc230529659777fb00c2dc3599f/returnn/frontend/attention.py#L700-L712
        nn.init.xavier_uniform_(mod.kv.weight, gain=math.sqrt(3 / 4))  # RF "scale" = PT gain^2
        if mod.kv.bias is not None:
            nn.init.zeros_(mod.kv.bias)
        nn.init.xavier_uniform_(mod.q.weight, gain=math.sqrt(1 / 2))  # RF "scale" = PT gain^2
        if mod.q.bias is not None:
            nn.init.zeros_(mod.q.bias)
        return False
    elif isinstance(mod, ConformerMHSARelPosV1):
        if mod.rel_pos_embeddings is not None:
            nn.init.zeros_(mod.rel_pos_embeddings)
        # continue normally on the child modules
    elif isinstance(mod, nn.Embedding):
        # See https://github.com/rwth-i6/returnn/blob/d209de55c468afc230529659777fb00c2dc3599f/returnn/frontend/linear.py#L53-L54
        nn.init.xavier_uniform_(mod.weight)
    elif isinstance(
        mod,
        (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    ):
        # See
        #   - Linear: https://github.com/rwth-i6/returnn/blob/d209de55c468afc230529659777fb00c2dc3599f/returnn/frontend/linear.py#L23-L29
        #   - Conv: https://github.com/rwth-i6/returnn/blob/d209de55c468afc230529659777fb00c2dc3599f/returnn/frontend/conv.py#L83-L92
        nn.init.xavier_uniform_(mod.weight)
        if mod.bias is not None:
            nn.init.zeros_(mod.bias)
    return True


class _SpecAugArgs(TypedDict):
    time_min_num_masks: int
    time_max_mask_per_n_frames: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int


class Model(nn.Module, CtcModel):
    """
    Conformer encoder + Transformer decoder AED + CTC model
    similar to the RETURNN frontend implementation but using primitives from i6_models.

    Uses:
        - `RasrCompatibleLogMelFeatureExtractionV1` for feature extraction,
        - `VGG4LayerActFrontendV1` as convolutional frontend,
        - `ConformerRelPosEncoderV1` as encoder and
        - `TransformerDecoderV1` as decoder.
    """

    def __init__(
        self,
        # RETURNN get_model parameters
        epoch: int,
        step: int,
        *,
        sampling_rate: int,
        # Model Size/Structure
        n_mels: int = 80,
        model_dim: int,
        out_dim_wo_blank: int,
        num_heads: int,
        num_enc_layers: int,
        aux_loss_layers: Sequence[int],
        # RF Defaults
        dropout: float = 0.1,
        dropout_broadcast_axes: Optional[Literal["B", "BT", "T"]] = "BT",
        specaug_start: Optional[Union[int, Tuple[int, int, int]]] = 10,
        specaug_args: Optional[Dict[str, int]] = None,
        use_rf_init: bool = True,
        logits_bias: bool = False,
        aux_logits_bias: bool = False,
        feature_extraction_config: Optional[Dict[str, Any]] = None,
        lm_opts: Optional[Dict[str, Any]] = None,
        **_kwargs_unused,
    ):
        super().__init__()

        assert model_dim > 0
        assert out_dim_wo_blank > 0
        assert len(aux_loss_layers) == len(set(aux_loss_layers))
        assert list(aux_loss_layers) == sorted(aux_loss_layers)

        # positional embedding like RF
        rel_pos_clip = 16
        pos_emb_dropout = 0.1
        learnable_pos_emb = True
        with_linear_pos = False
        with_pos_bias = False
        separate_pos_emb_per_head = False

        # RF does not broadcast attention dropout masks
        attn_dropout_broadcast = None

        self.num_labels = out_dim_wo_blank

        # blank index is not part of the vocabulary, make space for it
        self.out_dim_w_blank = out_dim_wo_blank + 1
        self.blank_idx = out_dim_wo_blank

        if feature_extraction_config is None:
            mel_cfg = RasrCompatibleLogMelFeatureExtractionV1Config(
                sample_rate=sampling_rate, win_size=25 / 1000, hop_size=10 / 1000, min_amp=1e-4, num_filters=n_mels
            )
            self.mel_frontend = RasrCompatibleLogMelFeatureExtractionV1(mel_cfg)
        else:
            assert "class" in feature_extraction_config
            feature_extraction_class_str = feature_extraction_config.pop("class")
            feature_extraction_class = eval(feature_extraction_class_str)
            feature_extraction_config_class = eval(feature_extraction_class_str + "Config")
            mel_cfg = feature_extraction_config_class(
                sample_rate=sampling_rate,
                **feature_extraction_config,
            )
            self.mel_frontend = feature_extraction_class(mel_cfg)

        frontend = ModuleFactoryV1(
            module_class=VGG4LayerActFrontendV1,
            cfg=VGG4LayerActFrontendV1Config(
                in_features=n_mels,
                conv1_channels=32,
                conv2_channels=64,
                conv3_channels=64,
                conv4_channels=32,
                conv_kernel_size=(3, 3),
                conv_padding=None,
                pool1_kernel_size=(1, 2),
                pool1_padding=None,
                pool1_stride=(3, 1),
                pool2_kernel_size=(1, 2),
                pool2_padding=None,
                pool2_stride=(2, 1),
                activation=nn.functional.relu,
                out_features=model_dim,
            ),
        )
        block_cfg = ConformerRelPosBlockV1Config(
            ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                input_dim=model_dim,
                hidden_dim=model_dim * 4,
                dropout=dropout,
                activation=_relu_sq,
                dropout_broadcast_axes=dropout_broadcast_axes,
            ),
            mhsa_cfg=ConformerMHSARelPosV1Config(
                input_dim=model_dim,
                num_att_heads=num_heads,
                att_weights_dropout=dropout,
                with_bias=False,
                dropout=dropout,
                # this is not applied to attention weights, whose dropout is not broadcast
                dropout_broadcast_axes=dropout_broadcast_axes,
                rel_pos_clip=rel_pos_clip,
                pos_emb_dropout=pos_emb_dropout,
                learnable_pos_emb=learnable_pos_emb,
                with_linear_pos=with_linear_pos,
                with_pos_bias=with_pos_bias,
                separate_pos_emb_per_head=separate_pos_emb_per_head,
            ),
            conv_cfg=ConformerConvolutionV2Config(
                channels=model_dim,
                kernel_size=33,
                dropout=dropout,
                dropout_broadcast_axes=dropout_broadcast_axes,
                activation=nn.functional.silu,
                norm=MaskedBatchNorm1dV1(model_dim, eps=1e-3, momentum=0.1),
            ),
            modules=["ff", "mhsa", "conv", "ff"],
            scales=[0.5, 1.0, 1.0, 0.5],
        )
        enc_cfg = ConformerRelPosEncoderV1Config(num_layers=num_enc_layers, frontend=frontend, block_cfg=block_cfg)
        self.encoder = ConformerRelPosEncoderV1(enc_cfg)

        self.specaug_args: _SpecAugArgs = {
            "time_min_num_masks": 1,
            "time_max_mask_per_n_frames": 100,
            "time_mask_max_size": 20,
            "freq_min_num_masks": 1,
            "freq_max_num_masks": 2,
            "freq_mask_max_size": n_mels // 5,
        }
        if specaug_args is not None:
            self.specaug_args.update(specaug_args)
        self.specaug_start = specaug_start

        self.out_aux_logits = nn.ModuleList(
            [nn.Linear(model_dim, self.out_dim_w_blank, bias=aux_logits_bias) for _ in range(len(aux_loss_layers))]
        )
        self.out_logits = nn.Linear(model_dim, self.out_dim_w_blank, bias=logits_bias)
        self._out_fetch_layers = sorted(v - 1 for v in {*aux_loss_layers, enc_cfg.num_layers})

        if use_rf_init:
            _apply_filter(self.encoder, _init_rf)
            _apply_filter(self.out_aux_logits, _init_rf)

        self.external_lm = None
        if lm_opts is not None:
            self.external_lm = __import__(
                f"{PACKAGE}.pytorch_networks.{lm_opts['lm_module']}",
                fromlist=["Model"],
            ).Model(
                lm_opts["lm_network_args"],
            )

    def _apply_specaug(self, data: Tensor, data_len: Tensor) -> Tensor:
        if not self.training:
            return data

        import returnn.frontend as rf
        from returnn.tensor import Dim

        # `self.specaug_start` configures whether we use the standard i6_models
        # SpecAugment-by-length or the stepwise-scheduled RF implementation.
        #
        # To use the i6_models implemenentation, `self.specaug_start` must be an
        # integer. It then configures the first epoch that SpecAug will be applied
        # in. Starting SpecAug right in the first epoch leads to convergence
        # problems.
        #
        # When using RF SpecAug, set `self.specaug_start` to a tuple of three train
        # step indices. These represent the points at which the RF implementation
        # will increase the amount of regularization.

        if isinstance(self.specaug_start, int):
            if rf.get_run_ctx().epoch < self.specaug_start:
                return data
            return specaugment_v1_by_length(data, **self.specaug_args)

        batch_dim = Dim(int(data.shape[0]), name="batch")
        data_len = rf.convert_to_tensor(data_len.cpu(), dims=[batch_dim])
        time_dim = Dim(data_len, name="time")
        feature_dim = Dim(int(data.shape[-1]), name="feature")
        data = rf.convert_to_tensor(data, dims=[batch_dim, time_dim, feature_dim], feature_dim=feature_dim)
        data_w_specaug = rf.audio.specaugment(
            data,
            feature_dim=feature_dim,
            spatial_dim=time_dim,
            global_train_step_dependent=True,
            steps=self.specaug_start,
            max_consecutive_feature_dims=self.specaug_args["freq_mask_max_size"],
            max_consecutive_spatial_dims=self.specaug_args["time_mask_max_size"],
            num_spatial_mask_factor=self.specaug_args["time_max_mask_per_n_frames"],
        )
        return data_w_specaug.raw_tensor

    def forward(self, raw_audio: Tensor, raw_audio_lens: Tensor) -> Tuple[Tensor, List[Tensor], Tensor, Tensor]:
        raw_audio = raw_audio.squeeze(dim=2).float()
        data, seq_lens = self.mel_frontend(raw_audio, raw_audio_lens)
        if self.specaug_start is not None:
            data = self._apply_specaug(data, seq_lens)

        data_mask = torch.less(torch.arange(data.shape[-2], device=data.device)[None, :], seq_lens[:, None])
        encoder_outputs, out_mask = self.encoder.forward(data, data_mask, return_layers=self._out_fetch_layers)
        assert len(self.out_aux_logits) <= len(encoder_outputs)
        out_aux_logits = [aux_linear(aux_out) for aux_linear, aux_out in zip(self.out_aux_logits, encoder_outputs)]
        out_seq_lens = out_mask.sum(dim=-1)

        out_logits = self.out_logits(encoder_outputs[-1])
        return out_logits, out_aux_logits, out_seq_lens, out_mask
