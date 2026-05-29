"""
Like v2, but with i6_models specaugment (v3)
and now controllable start time for when specaugment is applied (v4)
and with the proper feature extraction from i6-models
"""

import numpy as np
import torch
from torch import nn

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosEncoderV1Config
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosBlockV1Config, ConformerRelPosEncoderV1
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1

from i6_models.parts.conformer.convolution import ConformerConvolutionV2Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV2Config
from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1Config
from i6_models.parts.dropout import BroadcastDropout
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.parts.rasr_fsa import RasrFsaBuilderV2

import returnn.frontend as rf
from returnn.forward_iface import ForwardCallbackIface

from .phmm_zhou_cfg import ModelConfig

class RasrFsaBuilderOrth(RasrFsaBuilderV2):
    def build_single(self, orth: str):
        """
        Build the FSA for the given sequence tag in the corpus.

        :param seq_tag: sequence tag
        :return: FSA as a tuple containing
            * number of states S
            * number of edges E
            * integer edge array of shape [E, 3] where each row is an edge
                consisting of from-state, to-state and the emission idx
            * float weight array of shape [E,]
        """
        raw_fsa = self.builder.build_by_orthography(orth)
        return raw_fsa 

def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        frontend_config = self.cfg.frontend_config
        conformer_size = self.cfg.conformer_size
        conformer_config = ConformerRelPosEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerRelPosBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                    input_dim=conformer_size,
                    hidden_dim=self.cfg.ff_dim,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes,
                ),
                mhsa_cfg=ConformerMHSARelPosV1Config(
                    input_dim=conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    with_bias=self.cfg.mhsa_with_bias,
                    dropout=self.cfg.mhsa_dropout,
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes,
                    learnable_pos_emb=self.cfg.pos_emb_config.learnable_pos_emb,
                    rel_pos_clip=self.cfg.pos_emb_config.rel_pos_clip,
                    with_linear_pos=self.cfg.pos_emb_config.with_linear_pos,
                    with_pos_bias=self.cfg.pos_emb_config.with_pos_bias,
                    separate_pos_emb_per_head=self.cfg.pos_emb_config.separate_pos_emb_per_head,
                    pos_emb_dropout=self.cfg.pos_emb_config.pos_emb_dropout,
                ),
                conv_cfg=ConformerConvolutionV2Config(
                    channels=conformer_size,
                    kernel_size=self.cfg.conv_kernel_size,
                    dropout=self.cfg.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size),
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes,
                ),
                modules=self.cfg.module_list,
                scales=self.cfg.module_scales,
            ),
        )

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerRelPosEncoderV1(cfg=conformer_config)
        self.num_output_linears = 1 if self.cfg.aux_ctc_loss_layers is None else len(self.cfg.aux_ctc_loss_layers)
        self.output_linears = nn.ModuleList(
            [
                nn.Linear(conformer_size, self.cfg.label_target_size)
                for _ in range(self.num_output_linears)
            ]
        )
        self.output_dropout = BroadcastDropout(
            p=self.cfg.final_dropout, dropout_broadcast_axes=self.cfg.dropout_broadcast_axes
        )

        self.return_layers = self.cfg.aux_ctc_loss_layers or [self.cfg.num_layers - 1]
        self.scales = self.cfg.aux_ctc_loss_scales or [1.0]

        self.specaug_start_epoch = self.cfg.specauc_start_epoch

        # No particular weight init!

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T, #labels]
        """

        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = rf.get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,  # TODO: make configurable
                    time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)

        conformer_out_layers, out_mask = self.conformer(conformer_in, mask, return_layers=self.return_layers)
        log_probs_list = []
        for i, (out_layer, scale) in enumerate(zip(conformer_out_layers, self.scales)):
            if scale == 0.0:
                continue
            conformer_out = self.output_dropout(out_layer)
            logits = self.output_linears[i](conformer_out)
            log_probs = torch.log_softmax(logits, dim=2)
            log_probs_list.append(log_probs)

        return log_probs_list, torch.sum(out_mask, dim=1)


class PhmmTrainStep:
    """
    Callable train step for posterior HMM training using fbw2_loss.

    Usage in RETURNN config:
        train_step = PhmmTrainStep(fsa_exporter_config_path="/path/to/rasr.config")

    Requires librasr (for FSA building) and i6_native_ops (for fbw2_loss) in the training venv.
    Orthography text must be available as data["labels"] (list of strings per batch).
    """

    def __init__(
        self,
        fsa_exporter_config_path: str,
        transition_scale: float = 1.0,
        zero_infinity: bool = True,
        label_smoothing_scale: float = 0.0,
    ):
        """
        :param fsa_exporter_config_path: path to the RASR FSA exporter config file
        :param transition_scale: scale for FSA transition weights (TDPs)
        :param zero_infinity: replace inf losses with 0 (can occur when logits are shorter than label sequence)
        :param label_smoothing_scale: interpolation weight toward uniform distribution (0 = no smoothing)
        """

        self.fsa_builder = RasrFsaBuilderOrth(fsa_exporter_config_path, transition_scale)
        self.zero_infinity = zero_infinity
        self.label_smoothing_scale = label_smoothing_scale

    def __call__(self, *, model: Model, extern_data, **kwargs):
        from i6_native_ops.fbw2 import fbw2_loss

        run_ctx = rf.get_run_ctx()

        raw_audio = extern_data["raw_audio"].raw_tensor  # [B, T', F]
        raw_audio_len = extern_data["raw_audio"].dims[1].dyn_size_ext.raw_tensor.to("cpu")  # [B]

        # labels are UTF-8 bytes (uint8 tensor [B, T]) from OggZipDataset "orth", decode to strings.
        # Move the whole tensor to CPU in a single transfer instead of slicing per sequence,
        # which previously triggered one device->host sync per sequence.
        labels_raw = extern_data["labels"].raw_tensor.cpu()  # [B, T] uint8
        labels_len = extern_data["labels"].dims[1].dyn_size_ext.raw_tensor.cpu()  # [B]
        labels = [
            bytes(labels_raw[i, :labels_len[i]].tolist()).decode("utf8") + " "
            for i in range(labels_raw.shape[0])
        ]

        logprobs_list, audio_features_len = model(
            raw_audio=raw_audio,
            raw_audio_len=raw_audio_len,
        )
        audio_features_len_for_loss = audio_features_len.to(dtype=torch.int32).contiguous()

        # The target FSA depends only on the labels, not on the encoder layer, so build it once
        # and reuse it for every auxiliary loss. Building runs on the CPU and overlaps the
        # (asynchronously queued) forward pass. Previously it was rebuilt once per aux layer.
        target_fsa = self.fsa_builder.build_batch(labels).to(logprobs_list[0].device)

        for logprobs, layer_index, scale in zip(logprobs_list, model.return_layers, model.scales):
            # Apply label smoothing in probability space before FBW loss
            if self.label_smoothing_scale > 0:
                V = logprobs.shape[-1]
                probs = torch.exp(logprobs)
                smoothed_probs = (1 - self.label_smoothing_scale) * probs + self.label_smoothing_scale / V
                logprobs = torch.log(smoothed_probs)

            ml_loss = fbw2_loss(logprobs, target_fsa, audio_features_len_for_loss)  # [B]
            norm_frames = audio_features_len  # [B]

            if self.zero_infinity:
                # Replace inf losses (e.g. when the subsampled logits are shorter than the label
                # sequence) with 0. Done entirely on-device via torch.where so that, unlike the
                # previous `if torch.any(inf_mask): ...` diagnostic, it does not force a per-step
                # CPU<->GPU synchronization.
                valid_mask = ~torch.isinf(ml_loss)
                ml_loss = torch.where(valid_mask, ml_loss, torch.zeros_like(ml_loss))
                norm_frames = torch.where(valid_mask, norm_frames, torch.zeros_like(norm_frames))

            # RETURNN's mark_as_loss does not divide by a frame count, so normalize here to
            # reproduce MiniRETURNN's optimizer loss (sum(loss) * scale / sum(frames)). The loss is
            # passed as a scalar (dims=[]); use_normalized_loss stays False so the optimizer uses
            # exactly this value times the scale.
            inv_norm_factor = torch.clamp(torch.sum(norm_frames).to(ml_loss.dtype), min=1.0)
            normalized_loss = torch.sum(ml_loss) / inv_norm_factor

            run_ctx.mark_as_loss(
                loss=normalized_loss,
                name=f"phmm_loss_layer{layer_index + 1}",
                dims=[],
                scale=scale,
            )


_phmm_train_step = None


def train_step(*, model: Model, extern_data, **kwargs):
    """
    Standard RETURNN (torch) train_step interface. Pass 'fsa_exporter_config_path' and optionally
    'transition_scale', 'zero_infinity', 'label_smoothing_scale' via RETURNN config.
    """
    global _phmm_train_step
    if _phmm_train_step is None:
        _phmm_train_step = PhmmTrainStep(
            fsa_exporter_config_path=kwargs["fsa_exporter_config_path"],
            transition_scale=kwargs.get("transition_scale", 1.0),
            zero_infinity=kwargs.get("zero_infinity", True),
            label_smoothing_scale=kwargs.get("label_smoothing_scale", 0.0),
        )
    _phmm_train_step(model=model, extern_data=extern_data, **kwargs)


def prior_step(*, model: Model, extern_data, **kwargs):
    """
    Real-RETURNN prior forward step: run the AM and mark the per-frame probabilities
    as output (with a dynamic time dim, so RETURNN trims the padding per sequence).
    The actual accumulation happens in :class:`PriorCallback`.
    """
    from returnn.tensor import Dim

    audio = extern_data["raw_audio"]
    batch_dim = audio.dims[0]
    raw_audio = audio.raw_tensor  # [B, T', F]
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len.to(raw_audio.device),
    )
    if isinstance(logprobs, list):
        logprobs = logprobs[-1]
    probs = torch.exp(logprobs)

    feat_len = audio_features_len.to(device="cpu", dtype=torch.int32)
    time_dim = Dim(None, name="prior_out_time")
    time_dim.dyn_size_ext = rf.convert_to_tensor(feat_len, dims=[batch_dim], dtype="int32")
    feat_dim = Dim(int(probs.shape[-1]), name="prior_labels")
    rf.convert_to_tensor(probs, dims=[batch_dim, time_dim, feat_dim], name="probs").mark_as_output(
        "probs", shape=[batch_dim, time_dim, feat_dim]
    )


class PriorCallback(ForwardCallbackIface):
    """Accumulate the average AM posterior over the dataset and write ``prior.txt``."""

    def init(self, *, model):
        self.sum_probs = None
        self.sum_frames = 0

    def process_seq(self, *, seq_tag, outputs):
        probs = outputs["probs"].raw_tensor  # [T, V] numpy, padding already removed
        self.sum_frames += probs.shape[0]
        seq_sum = probs.sum(axis=0)
        if self.sum_probs is None:
            self.sum_probs = seq_sum
        else:
            self.sum_probs += seq_sum

    def finish(self):
        average_probs = self.sum_probs / self.sum_frames
        log_average_probs = np.log(average_probs)
        print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
        with open("prior.txt", "w") as f:
            np.savetxt(f, log_average_probs, delimiter=" ")
        print("Saved prior in prior.txt in +log space.")
