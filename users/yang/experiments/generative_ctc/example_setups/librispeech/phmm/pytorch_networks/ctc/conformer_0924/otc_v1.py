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
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.loss.fixed_ctc_loss import torch_ctc_fixed_grad


from returnn.torch.context import get_run_ctx

#from .i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import ModelConfig
from .otc_cfg_v1 import ModelConfig


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
                nn.Linear(conformer_size, self.cfg.label_target_size + 1)  # + CTC blank
                for _ in range(self.num_output_linears)
            ]
        )
        self.output_dropout = BroadcastDropout(
            p=self.cfg.final_dropout, dropout_broadcast_axes=self.cfg.dropout_broadcast_axes
        )

        self.return_layers = self.cfg.aux_ctc_loss_layers or [self.cfg.num_layers - 1]
        self.scales = self.cfg.aux_ctc_loss_scales or [1.0]

        self.specaug_start_epoch = self.cfg.specauc_start_epoch
        self.beta_1 = self.cfg.beta_1
        self.pi_1 = self.cfg.pi_1
        self.beta_2 = self.cfg.beta_2
        self.pi_2 = self.cfg.pi_2
        self.use_otc = self.cfg.use_otc

        # No particular weight init!

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T, #labels + blank]
        """

        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
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


def perturb_sequences(
    x: torch.Tensor,
    lengths: torch.Tensor,
    vocab_size: int,
    pad_id: int,
    p_sub: float = 0.0,
    p_del: float = 0.1,
    p_ins: float = 0.0,
    max_ins_len: int = 2,
):

    device = x.device
    B,  T, = x.shape
    min_length = lengths.min()
    sub_mask = torch.rand((B,T),device=device) < p_sub
    #
    del_mask = torch.rand((B, T), device=device) > p_del # keep these frames
    old_length_mask = torch.arange(T, device=device, dtype=lengths.dtype)[None, :] < lengths[:, None]
    del_mask = old_length_mask & del_mask



    sub_shuf = torch.randint(1, vocab_size, (B,T), device=device, dtype=x.dtype)
    sub_shuf = torch.where(sub_mask, sub_shuf, 0)
    sub_x = torch.remainder(x + sub_shuf, vocab_size)

    ins_matrix = torch.randint(0, vocab_size, (B, T, max_ins_len), device=device, dtype=x.dtype)

    ins_mask = (torch.rand((B,T), device=device) < p_ins) & del_mask # let ins only happens when there is no del
    ins_mask = ins_mask & old_length_mask # only ins for original posistions
    ins_length_vector = torch.arange(max_ins_len, device=device, dtype=x.dtype)
    ins_length_mask = ins_length_vector[None, None,:] < torch.randint(1, max_ins_len+1, (B,T), device=device, dtype=x.dtype)[:,:, None]
    final_ins_mask = ins_length_mask & ins_mask.unsqueeze(-1) # (B,T,max_ins_len)
    del_and_ins_mask = torch.cat([del_mask.unsqueeze(-1), final_ins_mask], dim=-1)
    final_mask = del_and_ins_mask.view(B,-1)
    sub_ins_x = torch.cat([sub_x.unsqueeze(-1), ins_matrix], dim=-1) # (B,T,1+max_ins_len)
    sub_ins_x = sub_ins_x.view(B, -1)
    idx = final_mask.cumsum(dim=-1)-1
    kept_token_lengths = final_mask.sum(dim=-1)
    out = x.new_full((B,max_ins_len*T), pad_id)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T*(max_ins_len+1))

    # select only valid (kept) positions
    flat_rows = batch_idx[final_mask]
    flat_cols = idx[final_mask]
    flat_vals = sub_ins_x[final_mask]

    # scatter into output
    out[flat_rows, flat_cols] = flat_vals
    final_lengths = final_mask.to(lengths.dtype).sum(dim=-1)
    new_max_length = final_lengths.max()
    out = out[:,:new_max_length]


    return out, final_lengths

import math
def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"] # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    logprobs_list, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    for logprobs, layer_index, scale in zip(logprobs_list, model.return_layers, model.scales):
        transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
        #smoothing_prob = transposed_logprobs[:,:,:-1].logsumexp(dim=-1)- torch.log(model.cfg.label_target_size) # log space


        if model.use_otc:

            smoothing_prob =torch.log(1-torch.exp(transposed_logprobs[:, :, -1])) - math.log(
                model.cfg.label_target_size)  # log space
            cur_epoch = run_ctx.epoch
            lambda_1 = model.beta_1 * model.pi_1**cur_epoch # lambda_1 for blank
            lambda_2 = model.beta_2 * model.pi_2**cur_epoch # lambda_2 for labels
            weighted_smoothing_1 = lambda_1+ smoothing_prob
            weighted_smoothing_2 = lambda_2+ smoothing_prob

            smoothed_blank_prob = torch.logaddexp(transposed_logprobs[:,:,-1], weighted_smoothing_1)
            smoothed_label_prob = torch.logaddexp(transposed_logprobs[:,:, :-1], weighted_smoothing_2[:,:,None])
            smoothed_ctc_output_prob = torch.cat([smoothed_label_prob, smoothed_blank_prob[:,:,None]], dim=-1)
            ctc_prob = smoothed_ctc_output_prob
        else:
            ctc_prob = transposed_logprobs
        noisy_data, noisy_length = perturb_sequences(labels, labels_len, vocab_size=model.cfg.label_target_size, pad_id=0, p_sub=model.cfg.p_sub,
                                                     p_del=model.cfg.p_del, p_ins=model.cfg.p_ins)




        ctc_loss = torch_ctc_fixed_grad(
            ctc_prob,
            noisy_data,
            input_lengths=audio_features_len,
            target_lengths=noisy_length,
            blank=model.cfg.label_target_size,
            reduction="sum",
            zero_infinity=True,
        )
        num_phonemes = torch.sum(labels_len)
        run_ctx.mark_as_loss(
            name=f"ctc_loss_layer{layer_index + 1}", loss=ctc_loss, scale=scale, inv_norm_factor=num_phonemes
        )



















def prior_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
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
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

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
