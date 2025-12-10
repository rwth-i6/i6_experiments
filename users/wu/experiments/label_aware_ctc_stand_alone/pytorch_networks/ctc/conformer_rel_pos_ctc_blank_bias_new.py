"""
Modified version with LearnableEmbeddingBias support.
"""

import numpy as np
import torch
from torch import nn
from typing import Optional

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

from returnn.torch.context import get_run_ctx

from .conformer_rel_pos_ctc_blank_bias_new_cfg import ModelConfig


# --- Bias Components ---
class LearnableEmbeddingBias(torch.nn.Module):
    """
    Project CTC logits into 1D vectors [B, T, 1, D].
    These act as 'Bias Keys' for the Transformer-XL style interaction.
    """
    def __init__(self, embed_dim: int, start_step: int = 0):
        super().__init__()
        self.start_step = start_step
        self.register_buffer("stored_step", torch.tensor(0, dtype=torch.long))
        
        # Scalar (1) -> Vector (D) projection
        self.projection = nn.Sequential(
            nn.Linear(1, embed_dim * 2),
            nn.Tanh(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def get_effective_step(self, step: Optional[int]) -> int:
        if self.training and step is not None:
             self.stored_step.fill_(step)
        if step is not None:
            return step
        else:
            # memorized in training
            return self.stored_step.item()

    def forward(self, logits: torch.Tensor, step: Optional[int] = None) -> Optional[torch.Tensor]:
        eff_step = self.get_effective_step(step)
        if eff_step < self.start_step:
            return None
        
        # Get Blank Logits [B, T, 1]
        # Assumes blank is the last index
        blank_logits = logits[:, :, -1:]
        
        # Project to Embedding [B, T, D]
        k_bias = self.projection(blank_logits)
        
        # Reshape for Broadcast [B, T, 1, D]
        # Insert 1 at Heads dim (dim 2) for broadcasting
        k_bias = k_bias.unsqueeze(2)
        
        return k_bias


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)
    seq_mask = torch.less(r[None, :], seq_len[:, None])
    return seq_mask


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        
        # Extract bias args before creating config
        self.bias_idx = model_config_dict.pop("bias_layer_index", None)
        bias_start_step = model_config_dict.pop("bias_start_step", 0)
        
        self.cfg = ModelConfig.from_dict(model_config_dict)
        frontend_config = self.cfg.frontend_config
        conformer_size = self.cfg.conformer_size
        
        # Prepare Conformer Config
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
        
        self.num_output_linears = 1 if self.cfg.aux_ctc_loss_layers is None else len(self.cfg.aux_ctc_loss_layers)
        self.output_linears = nn.ModuleList(
            [
                nn.Linear(conformer_size, self.cfg.label_target_size + 1)
                for _ in range(self.num_output_linears)
            ]
        )
        self.output_dropout = BroadcastDropout(
            p=self.cfg.final_dropout, dropout_broadcast_axes=self.cfg.dropout_broadcast_axes
        )
        self.return_layers = self.cfg.aux_ctc_loss_layers or [self.cfg.num_layers - 1]
        self.scales = self.cfg.aux_ctc_loss_scales or [1.0]
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

        # --- Bias Initialization ---
        if self.bias_idx is None:
            self.conformer = ConformerRelPosEncoderV1(cfg=conformer_config)
            self.encoder_bottom = None
            self.encoder_top = None
            self.compute_bias = None
        else:
            import copy
            self.conformer = None
            
            cfg_bottom = copy.deepcopy(conformer_config)
            cfg_bottom.num_layers = self.bias_idx
            self.encoder_bottom = ConformerRelPosEncoderV1(cfg_bottom)
            
            cfg_top = copy.deepcopy(conformer_config)
            cfg_top.num_layers = self.cfg.num_layers - self.bias_idx
            cfg_top.frontend = None
            self.encoder_top = ConformerRelPosEncoderV1(cfg_top)
            
            # Auto-infer head_dim
            head_dim = conformer_size // self.cfg.num_heads
            self.compute_bias = LearnableEmbeddingBias(embed_dim=head_dim, start_step=bias_start_step)

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ):
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        mask = mask_tensor(conformer_in, audio_features_len)
        log_probs_list = []

        if self.bias_idx is None:
            # Standard Path
            conformer_out_layers, out_mask = self.conformer(conformer_in, mask, return_layers=self.return_layers)
            for i, (out_layer, scale) in enumerate(zip(conformer_out_layers, self.scales)):
                if scale == 0.0:
                    continue
                conformer_out = self.output_dropout(out_layer)
                logits = self.output_linears[i](conformer_out)
                log_probs = torch.log_softmax(logits, dim=2)
                log_probs_list.append(log_probs)
        else:
            # Biased Path
            # return_layers are 0-based indices from config
            last_bottom_idx = self.bias_idx - 1
            ret_layers_bottom = [l for l in self.return_layers if l < self.bias_idx]
            
            # Ensure we get the guide layer output
            guide_needed_for_calc = last_bottom_idx not in ret_layers_bottom
            if guide_needed_for_calc:
                ret_layers_bottom.append(last_bottom_idx)
            ret_layers_bottom.sort()

            # 1. Run Bottom
            out_bottom, out_mask = self.encoder_bottom(conformer_in, mask, return_layers=ret_layers_bottom)

            # 2. Compute Bias
            guide_feat = out_bottom[-1]
            
            # Map guide layer (last_bottom_idx) to output linear
            # self.return_layers corresponds to self.output_linears indices
            # Find which linear layer corresponds to last_bottom_idx
            try:
                guide_lin_idx = self.return_layers.index(last_bottom_idx)
            except ValueError:
                raise ValueError(f"Bias layer {self.bias_idx} (idx {last_bottom_idx}) must be in aux_ctc_loss_layers {self.return_layers}")

            # Get logits, apply dropout first to match standard path? 
            # Standard path: dropout -> linear.
            guide_out_drop = self.output_dropout(guide_feat)
            guide_logits = self.output_linears[guide_lin_idx](guide_out_drop)

            step = get_run_ctx().step if self.training else None
            bias = self.compute_bias(logits=guide_logits.detach(), step=step)

            # Collect outputs from bottom
            # We need to map `out_bottom` entries back to `self.return_layers` order/indices
            # self.return_layers is not necessarily sorted, but `ret_layers_bottom` IS sorted.
            # We map by value.
            
            # Store bottom outputs in a map for easy retrieval
            feat_map = {layer_idx: feat for layer_idx, feat in zip(ret_layers_bottom, out_bottom)}

            # 3. Run Top
            ret_layers_top = [l - self.bias_idx for l in self.return_layers if l >= self.bias_idx]
            if ret_layers_top:
                out_top, _ = self.encoder_top(
                    guide_feat, 
                    out_mask, 
                    return_layers=ret_layers_top, 
                    attention_bias=bias
                )
                for i, layer_idx in enumerate(ret_layers_top):
                    real_idx = layer_idx + self.bias_idx
                    feat_map[real_idx] = out_top[i]

            # 4. Generate Log Probs in original order
            for i, layer_idx in enumerate(self.return_layers):
                scale = self.scales[i]
                if scale == 0.0:
                    continue
                
                feat = feat_map[layer_idx]
                feat_drop = self.output_dropout(feat)
                logits = self.output_linears[i](feat_drop)
                log_probs = torch.log_softmax(logits, dim=2)
                log_probs_list.append(log_probs)

        return log_probs_list, torch.sum(mask, dim=1)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to("cpu")
    labels = data["labels"]
    labels_len = data["labels:size1"]

    logprobs_list, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    for logprobs, layer_index, scale in zip(logprobs_list, model.return_layers, model.scales):
        transposed_logprobs = torch.permute(logprobs, (1, 0, 2))
        ctc_loss = nn.functional.ctc_loss(
            transposed_logprobs,
            labels,
            input_lengths=audio_features_len,
            target_lengths=labels_len,
            blank=model.cfg.label_target_size,
            reduction="sum",
            zero_infinity=True,
        )
        num_phonemes = torch.sum(labels_len)
        run_ctx.mark_as_loss(
            name=f"ctc_loss_layer{layer_index + 1}", loss=ctc_loss, scale=scale, inv_norm_factor=num_phonemes
        )


def prior_init_hook(run_ctx, **kwargs):
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0

def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
    with open("prior.txt", "w") as f:
        np.savetxt(f, log_average_probs, delimiter=" ")
    print("Saved prior in prior.txt in +log space.")

def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    logprobs = logprobs[-1]

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))
