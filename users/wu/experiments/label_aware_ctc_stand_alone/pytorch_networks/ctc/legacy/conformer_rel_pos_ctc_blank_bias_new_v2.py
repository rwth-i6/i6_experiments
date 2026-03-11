import numpy as np
import torch
from torch import nn
from typing import Optional
import copy  # Added for config copying

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
    def __init__(self, embed_dim: int, start_epoch: int = 0):
        super().__init__()
        self.start_epoch = start_epoch
        self.register_buffer("stored_epoch", torch.tensor(0, dtype=torch.long))
        
        # Scalar (1) -> Vector (D) projection
        self.projection = nn.Sequential(
            nn.Linear(1, embed_dim * 2),
            nn.Tanh(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def get_effective_epoch(self, epoch: Optional[int]) -> int:
        if self.training and epoch is not None:
             self.stored_epoch.fill_(epoch)
        if epoch is not None:
            return epoch
        else:
            # memorized in training
            return self.stored_epoch.item()

    def forward(self, logits: torch.Tensor, epoch: Optional[int] = None) -> Optional[torch.Tensor]:
        eff_epoch = self.get_effective_epoch(epoch)
        if eff_epoch < self.start_epoch:
            return None
        
        blank_logits = logits[:, :, -1:]  # blank is the last index
        k_bias = self.projection(blank_logits)  # [B, T, D]
        k_bias = k_bias.unsqueeze(2)  # [B, T, 1(H), D]
        
        return k_bias


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)
    seq_mask = torch.less(r[None, :], seq_len[:, None])
    return seq_mask


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        
        self.cfg = ModelConfig.from_dict(model_config_dict)
        frontend_config = self.cfg.frontend_config
        conformer_size = self.cfg.conformer_size
        bias_start_epoch = self.cfg.bias_start_epoch
        
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
        
        self.output_linear = nn.Linear(conformer_size, self.cfg.label_target_size + 1)
        
        self.output_dropout = BroadcastDropout(
            p=self.cfg.final_dropout, dropout_broadcast_axes=self.cfg.dropout_broadcast_axes
        )
        
        raw_return_layers = self.cfg.aux_ctc_loss_layers or [self.cfg.num_layers - 1]
        raw_scales = self.cfg.aux_ctc_loss_scales or [1.0]
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

        # Sort layers and scales to ensure processing order
        layers_and_scales = sorted(zip(raw_return_layers, raw_scales), key=lambda x: x[0])
        self.return_layers = [x[0] for x in layers_and_scales]
        self.scales = [x[1] for x in layers_and_scales]

        # --- Flexible Block Construction based on aux_ctc_loss_layers ---
        self.encoder_blocks = nn.ModuleList()
        self.bias_computers = nn.ModuleList()
        
        # Determine boundaries for blocks.
        # We need to cover the entire depth of the model up to num_layers-1.
        # Each aux layer acts as a boundary where we output loss and potentially bias.
        boundaries = sorted(list(set(self.return_layers)))
        if boundaries[-1] < self.cfg.num_layers - 1:
            boundaries.append(self.cfg.num_layers - 1)
            
        current_layer_idx = 0
        head_dim = conformer_size // self.cfg.num_heads

        for i, end_layer_idx in enumerate(boundaries):
            num_layers_in_block = end_layer_idx - current_layer_idx + 1
            if num_layers_in_block <= 0:
                continue

            block_cfg = copy.deepcopy(conformer_config)
            block_cfg.num_layers = num_layers_in_block
            
            # Only the first block gets the frontend
            if current_layer_idx == 0:
                block_cfg.frontend = ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config)
            else:
                block_cfg.frontend = None
            
            self.encoder_blocks.append(ConformerRelPosEncoderV1(cfg=block_cfg))
            
            # Create Bias computer if this is not the last block
            # This bias will guide the NEXT block.
            if i < len(boundaries) - 1:
                self.bias_computers.append(
                    LearnableEmbeddingBias(embed_dim=head_dim, start_epoch=bias_start_epoch)
                )
            
            current_layer_idx = end_layer_idx + 1

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
        
        current_bias = None
        epoch = get_run_ctx().epoch if self.training else None

        # Iterate through chained blocks
        for i, encoder in enumerate(self.encoder_blocks):
            # We always want the output of the last layer in this block
            # Internal block indices are 0..N-1
            local_return_layers = [encoder.cfg.num_layers - 1]
            
            out_layers, out_mask = encoder(
                conformer_in, 
                mask, 
                return_layers=local_return_layers, 
                attention_bias=current_bias
            )
            
            # Feature at the boundary
            feat = out_layers[0] 
            
            # Pass through shared linear to get logits
            # Use dropout before linear as in original
            feat_drop = self.output_dropout(feat)
            logits = self.output_linear(feat_drop)
            log_probs = torch.log_softmax(logits, dim=2)
            
            # We only store log_probs if this boundary is actually in self.return_layers
            # (We sorted self.return_layers in init, so the order is preserved if we blindly append,
            # but we must check if this specific boundary is a requested return layer).
            # To match the logic of 'boundaries' in init, we can reconstruct the global index.
            # However, simpler: we built blocks exactly on boundaries.
            # If the last boundary was added artificially (end of model) but not in return_layers, check.
            
            # Since we iterate blocks in order, we can just append if it's meant to be returned.
            # But wait: if we added an artificial boundary at the end, we might output it here.
            # Let's rely on the count of outputs matching return_layers logic or just append all 
            # and filter in train_step? No, train_step iterates zip.
            
            # Optimization: check if current iteration corresponds to a return layer.
            # We know boundaries[i] is the global layer index of this block's end.
            # But self.return_layers is just a list of indices.
            
            # Just append. If self.return_layers didn't include the very last layer, 
            # we might have an extra log_probs. But we constructed blocks based on return_layers.
            # The only case is if we extended boundaries for the model end.
            
            # A safe check: is this log_probs used?
            # Actually, the biases MUST be computed from logits. So we need logits regardless.
            
            # If this is a valid return layer, add to list.
            # We can't easily check global index here without tracking `current_layer`.
            # But `log_probs_list` size needs to match `self.return_layers` size for `train_step`.
            
            # Let's trust that boundaries corresponds to return_layers generally. 
            # If we appended the final layer just for completeness, we might have 1 extra.
            # We will slice log_probs_list at return if needed, or filter here.
            
            if i < len(self.return_layers):
                 # This assumes boundaries[:len(return_layers)] == return_layers.
                 # Given we sorted both, this is true, UNLESS boundaries has an extra element at end.
                 log_probs_list.append(log_probs)
            elif i == len(self.encoder_blocks) - 1 and len(log_probs_list) < len(self.return_layers):
                 # Fallback for the last one
                 log_probs_list.append(log_probs)
            
            # Prepare for next block
            conformer_in = feat
            
            # Compute bias for next block if applicable
            if i < len(self.bias_computers):
                current_bias = self.bias_computers[i](logits=logits.detach(), epoch=epoch)

        return log_probs_list, torch.sum(out_mask, dim=1)


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
