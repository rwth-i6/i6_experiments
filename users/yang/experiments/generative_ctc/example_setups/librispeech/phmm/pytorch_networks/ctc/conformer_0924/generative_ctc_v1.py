"""

copied from i6models_relposV1_VGG4LayerActFrontendV1_v1.py
apply the same topology as CTC for gaussian model, i.e. it is a generative model with blank

only the output is modified
"""

import numpy as np
import torch
from torch import nn
import torch.nn.utils.parametrize as P

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
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.loss.fixed_ctc_loss import torch_ctc_fixed_grad, ctc_loss_forward_batch


from returnn.torch.context import get_run_ctx

from .generative_ctc_v1_cfg import ModelConfig


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

class UnitNorm(nn.Module):
    """Enforce L2=1 along `dim` (1=row-wise, 0=column-wise)."""
    def __init__(self, dim=1, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
    def forward(self, W):
        # W: [out_features, in_features]
        vector_norm = W.norm(p=2, dim=self.dim, keepdim=True).clamp_min(self.eps)
        return W / vector_norm


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
        self.vector_regularization = False
        if self.cfg.freeze_encoder:

            for param in self.parameters():
                param.requires_grad = False
            # for name, param in self.named_parameters():
            #     print(name, param.shape, param.requires_grad)
        else:
            self.vector_regularization = True #force to have regularization loss to avoid trivial solution

        self.num_output_linears = 1 if self.cfg.aux_ctc_loss_layers is None else len(self.cfg.aux_ctc_loss_layers)
        assert self.num_output_linears == 1, "do not support auxiliary loss yet"
        self.output_linears = nn.ModuleList(
            [
                nn.Linear(conformer_size, self.cfg.label_target_size + 1, bias=False)  # + CTC blank
                for _ in range(self.num_output_linears) # gaussian distribution mean, no bias
            ]
        )
        # another gaussian mixture to model silence hopefully?
        self.num_mixtures = getattr(self.cfg, "num_mixtures", 1)
        self.gaussian_mixture_weight = nn.Parameter(torch.randn(self.num_mixtures))
        self.extra_gaussian = nn.ModuleList([nn.Linear(conformer_size, self.cfg.label_target_size + 1, bias=False)  # + CTC blank
                for _ in range(self.num_mixtures-1)])
        self.norm_vector = self.cfg.norm_vector
        if self.norm_vector:
            for layer in self.output_linears:
                # avoid double-registration
                already = hasattr(layer, "parametrizations") and hasattr(layer.parametrizations, "weight")
                if not already:
                    P.register_parametrization(layer, "weight", UnitNorm(dim=1))  # rows L2=1


        # compute the gaussian distribution with linear and norm operation

        self.output_dropout = BroadcastDropout(
            p=self.cfg.final_dropout, dropout_broadcast_axes=self.cfg.dropout_broadcast_axes
        )

        self.return_layers = self.cfg.aux_ctc_loss_layers or [self.cfg.num_layers - 1]
        self.scales = self.cfg.aux_ctc_loss_scales or [1.0]

        self.specaug_start_epoch = self.cfg.specauc_start_epoch

        if self.vector_regularization:
            self.encoder_output_list =[]

        # No particular weight init!

    def get_enc_output(self):
        return self.encoder_output_list

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
        gaussian_log_probs_list = []
        conformer_outputs = []
        for i, (out_layer, scale) in enumerate(zip(conformer_out_layers, self.scales)):
            if scale == 0.0:
                continue
            conformer_out = self.output_dropout(out_layer)
            if self.cfg.freeze_encoder:
                conformer_out = conformer_out.detach()
            if self.norm_vector:
                conformer_out = conformer_out / conformer_out.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            conformer_outputs.append(conformer_out)

            # for gaussian models, we don't need logits, but log-gaussian probs
            prod_term = self.output_linears[i](conformer_out) # (B,T,W)
            debug = False
            if debug:
                print("NOTE!!!!!!!!!! This is debugging!!!!!!!!!!!!!!!!!!!!")

                gaussian_log_prob = torch.nn.functional.log_softmax(prod_term, dim=-1)
            elif not self.norm_vector:
                log_mixture_list = []
                h_l2 = torch.pow(conformer_out,2).sum(-1, keepdim=True)
                mean_vec_extend = torch.pow(self.output_linears[i].weight,2).sum(1).unsqueeze(0).unsqueeze(0) #(1,1,W)
                gaussian_log_prob_tmp = prod_term - 0.5 * (h_l2 + mean_vec_extend)
                log_mixture_list.append(gaussian_log_prob_tmp)
                for i, extra_linear in enumerate(self.extra_gaussian):
                    prod_term = extra_linear(conformer_out)
                    mean_vec_extend = torch.pow(extra_linear.weight,2).sum(1).unsqueeze(0).unsqueeze(0)
                    gaussian_log_prob_tmp = prod_term - 0.5 * (h_l2 + mean_vec_extend) # (B,T,W)
                    log_mixture_list.append(gaussian_log_prob_tmp)
                log_mixture_weight = torch.nn.functional.log_softmax(self.gaussian_mixture_weight, dim=-1) # i
                log_mixture_weight = log_mixture_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                log_gaussian_mixtures = torch.stack(log_mixture_list, dim=0)
                gaussian_log_prob = torch.logsumexp(log_gaussian_mixtures + log_mixture_weight, dim=0)

                #print('guassian probs', gaussian_log_prob[0,:5])

            else:
                # print("check conformer normalized output,", torch.pow(conformer_out,2).sum(-1)[:4,:4])
                # print("check weight matrix normalized,", torch.pow(self.output_linears[i].weight,2).sum(1))
                # print("check weight matrix shape,", self.output_linears[i].weight.shape)
                gaussian_log_prob = prod_term - 1.0 # check if it is always <=0
                #print("check < 0**********:", torch.all(gaussian_log_prob<0))

            gaussian_log_probs_list.append(gaussian_log_prob)

            #logits = self.output_linears[i](conformer_out)
            #log_probs = torch.log_softmax(logits, dim=2)
            #log_probs_list.append(log_probs)
            #self.encoder_output_list.append(conformer_out)



        return gaussian_log_probs_list, torch.sum(out_mask, dim=1), conformer_outputs


def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"].to(torch.long)  # [B, N] # avoid using CuDNN, to make sure fixed ctc loss computation works

    logprobs_list, audio_features_len, conformer_outputs = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    for logprobs, layer_index, scale in zip(logprobs_list, model.return_layers, model.scales):
        transposed_logprobs = torch.permute(logprobs, (1, 0, 2)).to(torch.float32)  # CTC needs [T, B, F]
        # ctc_loss = torch.nn.functional.ctc_loss(
        #     transposed_logprobs,
        #     labels,
        #     input_lengths=audio_features_len,
        #     target_lengths=labels_len,
        #     blank=model.cfg.label_target_size,
        #     reduction="sum",  # "sum",
        #     zero_infinity=True,
        # )
        ctc_loss = torch_ctc_fixed_grad(
            transposed_logprobs,
            labels,
            input_lengths=audio_features_len,
            target_lengths=labels_len,
            blank=model.cfg.label_target_size,
            reduction="sum",#"sum",
            zero_infinity=True,
        )
        # ctc_loss_debug = ctc_loss_forward_batch(
        #     transposed_logprobs,
        #     labels,
        #     input_lengths=audio_features_len,
        #     target_lengths=labels_len,
        #     blank=model.cfg.label_target_size,
        #     zero_infinity=True,
        # )
        
        # from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.loss.fixed_ctc_loss import compare_grads_log_probs
        # print("source length ****", audio_features_len)
        # print("target length ****", labels_len)
        
        # check = compare_grads_log_probs(
        #     logits=transposed_logprobs,
        #     targets=labels,
        #     input_lengths=audio_features_len,
        #     target_lengths=labels_len,
        #     blank=model.cfg.label_target_size,
        #     zero_infinity=True,
        # )
        # print("**********ctc_loss*********",ctc_loss)
        # print("**********ctc_loss_debug*****", ctc_loss_debug)
        # print("ok batches", check['ok'])
        # print("max diff******************", check['max_abs_diff'])
        # print("max rel diff*************", check['max_rel_diff'])
        # print("max value abs diff ********", check['max_value_abs_diff'])
        # ctc_loss = ctc_loss.sum()
        num_phonemes = torch.sum(labels_len)
        run_ctx.mark_as_loss(
            name=f"ctc_loss_layer{layer_index + 1}", loss=ctc_loss, scale=scale, inv_norm_factor=num_phonemes
        )

        if model.vector_regularization:
            # encoder_output = model.get_enc_output()[-1] # (B,T,F)
            # normed_encoder = torch.nn.functional.normalize(encoder_output, dim=-1)
            # covariance = normed_encoder @ normed_encoder.transpose(-1,-2)
            # max_len = normed_encoder.shape[1]
            # batch_size = normed_encoder.shape[0]
            # diagonal = torch.eye(max_len, dtype=covariance.dtype, device= covariance.device)
            # orth = (covariance - diagonal.unsqueeze(0))**2
            # length_mask = mask_tensor(encoder_output, audio_features_len)
            # length_mask_2d = torch.logical_and(length_mask.unsqueeze(-1), length_mask.unsqueeze(1))

            # orth_active = orth[length_mask_2d]

            # regularization_loss = orth_active.sum()
            # num_frames = audio_features_len.sum()

            # scale = max(0.1, 1 -0.1 * run_ctx.epoch)
            last_linear = model.output_linears[-1]
            # weights_norm = last_linear.weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            # normed_weight = last_linear.weight / weights_norm
            normed_weight = torch.nn.functional.normalize(last_linear.weight, dim=1)
            weight_regularization = normed_weight @ normed_weight.T
            num_outputs, hid_dim = normed_weight.shape
            identity_matrix = torch.eye(num_outputs, dtype = normed_weight.dtype, device = normed_weight.device)
            regularization_loss = (weight_regularization - identity_matrix)**2
            regularization_loss = regularization_loss.sum()
            num_elements = torch.ones_like(normed_weight).to(torch.int32).sum()
            num_elements = num_elements / num_elements


            run_ctx.mark_as_loss(
            name="regularization_loss", loss=regularization_loss, scale=20, inv_norm_factor=num_elements
            )# try larger scale, to force them to be different

            # encoder norm loss:
            # TODO: also apply orthogonal loss to the encoder outputs
            encoder_output = conformer_outputs[-1]
            encoder_output_norm = encoder_output.norm(p=2, dim=-1).clamp_min(1e-12)
            encoder_norm_loss = (encoder_output_norm - 1)**2
            encoder_norm_loss = encoder_norm_loss.sum()
            #num_elements = torch.ones_like(encoder_output_norm).to(torch.int32).sum()
            run_ctx.mark_as_loss(name='encoder_norm_loss', loss=encoder_norm_loss, scale=10, inv_norm_factor=num_elements)






# priors are not needed

def prior_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.zeros_like(average_probs) # do not need prior
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
