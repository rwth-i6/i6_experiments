import numpy as np
import torch
from torch import nn
from typing import Optional, List, Dict
import copy

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

from .conformer_rel_pos_ctc_layer_refinement_v2_cfg import ModelConfig


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
        
        torch.manual_seed(self.cfg.seed)
        
        # text embedding
        self.vocab_size = self.cfg.label_target_size
        self.sep_token_idx = self.vocab_size + 1
        self.mask_token_idx = self.vocab_size + 2
        self.text_embedding = nn.Embedding(self.vocab_size + 3, conformer_size)
        
        # MLM head(linear layer for MLM loss)
        self.mlm_head = nn.Linear(conformer_size, self.vocab_size)
        
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
        assert self.cfg.num_layers - 1 in raw_return_layers
        raw_scales = self.cfg.aux_ctc_loss_scales or [1.0]
        assert len(raw_scales)==len(raw_return_layers)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

        layers_and_scales = sorted(zip(raw_return_layers, raw_scales), key=lambda x: x[0])
        self.return_layers = [x[0] for x in layers_and_scales]
        self.scales = [x[1] for x in layers_and_scales]

        self.encoder_blocks = nn.ModuleList()

        boundaries = sorted(list(set(self.return_layers)))
        if boundaries[-1] < self.cfg.num_layers - 1:
            boundaries.append(self.cfg.num_layers - 1)
            
        current_layer_idx = 0

        # create encoder blocks
        for i, end_layer_idx in enumerate(boundaries):
            num_layers_in_block = end_layer_idx - current_layer_idx + 1
            if num_layers_in_block <= 0:
                continue

            block_cfg = copy.deepcopy(conformer_config)
            block_cfg.num_layers = num_layers_in_block
            
            # frontend only
            if current_layer_idx == 0:
                block_cfg.frontend = ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config)
            else:
                block_cfg.frontend = None
            
            self.encoder_blocks.append(ConformerRelPosEncoderV1(cfg=block_cfg))
            current_layer_idx = end_layer_idx + 1

    def prepare_masked_gt(self, labels: torch.Tensor, labels_len: torch.Tensor):
        B, L = labels.shape
        device = labels.device
        labels_len = labels_len.to(device=device)
        
        loss_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        for b in range(B):
            valid_len = labels_len[b].item()
            if valid_len == 0: continue
            
            num_to_mask = int(valid_len * self.cfg.mlm_mask_rate)
            span_len = self.cfg.mlm_span_length
            masked_count = 0
            
            while masked_count < num_to_mask:
                # Use standard torch.randint for GPU safety
                start_idx = torch.randint(0, valid_len, (1,), device=device).item()
                end_idx = min(start_idx + span_len, valid_len)
                for k in range(start_idx, end_idx):
                    if not loss_mask[b, k]:
                        loss_mask[b, k] = True
                        masked_count += 1
                        
        input_ids = labels.clone()
        input_ids[loss_mask] = self.mask_token_idx
        
        embeddings = self.text_embedding(input_ids)
        
        # avoid leakage of pad-index embedding due to convolution
        valid_len_mask = torch.arange(L, device=device)[None, :] < labels_len[:, None]
        embeddings = embeddings * valid_len_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
        
        return embeddings, loss_mask

    def prepare_greedy_input(self, logits: torch.Tensor, audio_len: torch.Tensor):
        logits = logits.detach()
        B = logits.size(0)
        preds = torch.argmax(logits, dim=-1) # [B, T]
        
        blank_idx = self.cfg.label_target_size
        
        batch_tokens = []
        batch_lens = []
        
        for b in range(B):
            valid_len = audio_len[b].item()
            p = preds[b, :valid_len]
            
            # collapse identical consecutive labels
            if p.numel() > 0:
                p_collapsed = torch.unique_consecutive(p)
            else:
                p_collapsed = p
            # remove blanks
            p_noblank = p_collapsed[p_collapsed != blank_idx]
            # TODO: maybe separator makes more sense? Not sure
            if len(p_noblank) == 0:
                p_noblank = torch.tensor([self.mask_token_idx], device=logits.device)
                
            batch_tokens.append(p_noblank)
            batch_lens.append(len(p_noblank))
            
        max_len = max(batch_lens)
        padded_emb = torch.zeros(B, max_len, self.cfg.conformer_size, device=logits.device)
        
        for b in range(B):
            toks = batch_tokens[b]
            l = batch_lens[b]
            padded_emb[b, :l] = self.text_embedding(toks)
            
        return padded_emb, torch.tensor(batch_lens, device=logits.device)

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
        labels: Optional[torch.Tensor] = None, 
        labels_len: Optional[torch.Tensor] = None,
    ):
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            epoch = run_ctx.epoch if self.training else 0
            
            if self.training and epoch >= self.specaug_start_epoch:
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
        current_lens = audio_features_len
        
        audio_portion_len = audio_features_len.clone()
        max_audio_len = conformer_in.shape[1]
        
        log_probs_list = []
        next_block_mlm_target = None
        mlm_outputs_list = [] 

        for i, encoder in enumerate(self.encoder_blocks):
            mask = mask_tensor(conformer_in, current_lens)
            
            attention_bias = None
            if self.cfg.mask_t2a_attn and (current_lens > audio_portion_len).any():
                B, T = conformer_in.shape[:2]
                device = conformer_in.device
                
                idx = torch.arange(T, device=device)
                row_idx = idx.view(1, T, 1)  # query positions
                col_idx = idx.view(1, 1, T)  # key positions
                
                a_len = audio_portion_len.view(B, 1, 1)
                
                is_text_query = (row_idx >= a_len) # [B, T, 1], also mask potential sep tokens
                is_audio_key = (col_idx < a_len)        # [B, 1, T]
                mask_condition = is_text_query & is_audio_key
                
                attention_bias = torch.zeros(B, 1, T, T, device=device)
                attention_bias.masked_fill_(mask_condition.unsqueeze(1), -float('inf'))

            out_layers, out_mask = encoder(conformer_in, mask, attention_bias=attention_bias) 
            feat = out_layers[0] 
            
            if i == 0:
                audio_portion_len = torch.sum(out_mask, dim=1).long()
                max_audio_len = feat.shape[1]
            
            # take first T features as audio-only features
            audio_feat = feat[:, :max_audio_len, :]
            
            feat_drop = self.output_dropout(audio_feat)
            logits = self.output_linear(feat_drop)
            log_probs = torch.log_softmax(logits, dim=2)
            log_probs_list.append(log_probs)
            
            # resolve MLM output from last block
            if next_block_mlm_target is not None:
                text_start, targets_dict = next_block_mlm_target
                if text_start < feat.shape[1]:
                    text_feat = feat[:, text_start:, :]
                    mlm_logits = self.mlm_head(text_feat)
                    
                    mlm_outputs_list.append({
                        "logits": mlm_logits,
                        "targets": targets_dict["labels"], 
                        "mask": targets_dict["mask"],      
                        "active_mask": targets_dict["active_mask"] 
                    })
                next_block_mlm_target = None

            # prepare input for the next block
            if i < len(self.encoder_blocks) - 1:
                gt_emb, gt_mask_target = None, None
                gt_lens = labels_len if labels_len is not None else torch.zeros(feat.size(0), device=feat.device)
                
                if self.training and labels is not None:
                    gt_emb, gt_mask_target = self.prepare_masked_gt(labels, labels_len)
                
                greedy_emb, greedy_lens = self.prepare_greedy_input(logits, audio_portion_len)
                
                B = feat.size(0)
                use_gt_mask = torch.zeros(B, dtype=torch.bool, device=feat.device)
                
                if self.training and labels is not None:
                    progress = min(1.0, max(0.0, (epoch / self.cfg.gt_decay_epochs)))
                    gt_prob = self.cfg.gt_prob_start + progress * (self.cfg.gt_prob_end - self.cfg.gt_prob_start)
                    # random decision on using GT for this block
                    use_gt_mask = torch.rand(B, device=feat.device) < gt_prob

                max_gt_len = gt_emb.size(1) if gt_emb is not None else 0
                max_gr_len = greedy_emb.size(1)
                final_len = max(max_gt_len, max_gr_len)
                
                final_gt_emb = torch.zeros(B, final_len, self.cfg.conformer_size, device=feat.device)
                if gt_emb is not None:
                    final_gt_emb[:, :max_gt_len, :] = gt_emb
                
                final_gr_emb = torch.zeros(B, final_len, self.cfg.conformer_size, device=feat.device)
                final_gr_emb[:, :max_gr_len, :] = greedy_emb
                
                mask_bc = use_gt_mask.view(B, 1, 1)
                final_emb = torch.where(mask_bc, final_gt_emb, final_gr_emb)
                
                if gt_emb is not None:
                    final_lens = torch.where(use_gt_mask, gt_lens, greedy_lens)
                else:
                    final_lens = greedy_lens
                
                # if gt needed, we pass the current masking decision to next block
                if gt_emb is not None and use_gt_mask.any():
                    final_loss_mask = torch.zeros(B, final_len, dtype=torch.bool, device=feat.device)
                    final_loss_mask[:, :max_gt_len] = gt_mask_target
                    
                    final_labels = torch.zeros(B, final_len, dtype=torch.long, device=feat.device) 
                    final_labels[:, :labels.size(1)] = labels
                    
                    next_block_mlm_target = (
                        max_audio_len + self.cfg.num_sep_tokens,
                        {
                            "labels": final_labels,
                            "mask": final_loss_mask,  # within seq
                            "active_mask": use_gt_mask,  # batch level 
                        }
                    )
                
                # use separator
                sep_token = torch.tensor([self.sep_token_idx], device=feat.device)
                sep_emb = self.text_embedding(sep_token).view(1, 1, -1).expand(B, self.cfg.num_sep_tokens, -1)
                
                # audio masking for modality matching
                next_audio_feat = audio_feat
                
                if self.training:
                    mask_decay_ep = self.cfg.audio_mask_decay_epochs
                    progress_mask = min(1.0, max(0.0, (epoch / mask_decay_ep)))
                    
                    p_start = self.cfg.audio_mask_prob_start
                    p_end = self.cfg.audio_mask_prob_end
                    
                    # Probability of applying mask to a candidate sample
                    selection_prob = p_start + progress_mask * (p_end - p_start)
                    
                    do_audio_mask = torch.rand(B, device=feat.device) < selection_prob
                    
                    if do_audio_mask.any():
                        span_len = self.cfg.audio_mask_span
                        
                        max_a = next_audio_feat.shape[1]
                        keep_mask = torch.ones(B, max_a, dtype=torch.bool, device=feat.device)
                        
                        # span masking to mask out some non-blank positions
                        for b_idx in range(B):
                            if do_audio_mask[b_idx]:
                                cur_len = audio_portion_len[b_idx].item()
                                if cur_len > span_len:
                                    num_tokens_to_mask = int(cur_len * self.cfg.audio_mask_rate)
                                    num_spans = max(1, num_tokens_to_mask // span_len)
                                    
                                    for _ in range(num_spans):
                                        start = torch.randint(0, cur_len - span_len, (1,), device=feat.device).item()
                                        keep_mask[b_idx, start:start+span_len] = False
                        
                        next_audio_feat = next_audio_feat * keep_mask.unsqueeze(-1).type_as(next_audio_feat)

                next_in = torch.cat([next_audio_feat, sep_emb, final_emb], dim=1)
                conformer_in = next_in
                current_lens = audio_portion_len + self.cfg.num_sep_tokens + final_lens

        return log_probs_list, audio_portion_len, mlm_outputs_list


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to("cpu")
    labels = data["labels"]
    labels_len = data["labels:size1"]

    logprobs_list, audio_features_len, mlm_outputs_list = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
        labels=labels,
        labels_len=labels_len
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

    # 2. MLM Losses
    for i, mlm_data in enumerate(mlm_outputs_list):
        logits = mlm_data["logits"]
        targets = mlm_data["targets"]
        mask = mlm_data["mask"]
        active_mask = mlm_data["active_mask"]
        
        # Only compute loss at masked position in GT sequence
        combined_mask = mask & active_mask.unsqueeze(1)
        
        if combined_mask.any():
            active_logits = logits.reshape(-1, logits.size(-1))[combined_mask.view(-1)]
            active_targets = targets.reshape(-1)[combined_mask.view(-1)].long()
            
            mlm_loss = nn.functional.cross_entropy(active_logits, active_targets, reduction='mean')  # mean reduction is same as using inv_norm_factor
            
            run_ctx.mark_as_loss(
                name=f"mlm_loss_block_{i}", 
                loss=mlm_loss, 
                scale=model.cfg.mlm_loss_scale
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

    logprobs, audio_features_len, _ = model(
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
