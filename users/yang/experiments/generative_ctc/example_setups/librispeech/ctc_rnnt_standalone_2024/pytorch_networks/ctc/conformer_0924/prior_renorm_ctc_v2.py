"""
use empirical prior rather than
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
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.ctc.model.conformer_rel_pos_v2 import ConformerRelPosEncoderV2

from returnn.torch.context import get_run_ctx

from .prior_renorm_ctc_v2_cfg import ModelConfig


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
        self.conformer = ConformerRelPosEncoderV2(cfg=conformer_config)
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
        self.train_prior_scale = self.cfg.train_prior_scale
        self.prior_match_loss = self.cfg.prior_match_loss
        self.empirical_blank_ratio = self.cfg.empirical_blank_ratio
        self.prior_match_loss_scale = self.cfg.prior_match_loss_scale
        #hard coded prior for now
        self.label_empirical_prior = torch.tensor([0.01924980157835924, 0.03697661920007039, 0.06919701317605428, 0.015352396298161566, 0.004870774241212486, 0.011223243591164079, 0.01766233914689441, 0.0030749326205891858, 0.017481466010704855, 0.03069124794902783, 0.028553456078995478, 0.014465921459025834, 0.011825042689453745, 0.015814879024270995, 0.008513331773069347, 0.02721637612753235, 0.057743510580870086, 0.012648289328907677, 0.003482105556204839, 0.02225715980474989, 0.029137929936129396, 0.02190490966970529, 0.04715260852343519, 0.002531807148275427, 0.008880942431200484, 0.0008394406792971049, 0.01660228444318967, 0.03080847612715727, 0.0344361196627615, 0.007548417053443249, 0.036265361490561344, 0.003350618287845843, 0.00443877441363138, 0.005165005656304247, 0.008404528067873031, 0.02328744605286056, 0.006794560683979088, 0.0037809213134580868, 0.0004406326797033847, 1e-09, 0.00015342662013182305, 4.167583783169427e-07, 0.03224915030408775, 0.00026348060046309, 0.0010722597704983055, 0.007128652061111305, 0.00017256773707880835, 0.0026234642230495604, 0.03246401901656573, 0.0023127708520142795, 4.822489806238909e-05, 0.012380968597672952, 0.003476122096630431, 0.0028234784761860987, 0.000343944735933854, 3.274530115347407e-07, 4.762952895050774e-07, 0.020577712845499394, 0.0009745596992385765, 0.004680970568344713, 0.010476918640965217, 0.007930167727981568, 0.02450396375916772, 0.008151109205400736, 0.004415465712901226, 0.000189654830589803, 0.001938492059830071, 0.010944610846803609, 0.01288369827574556, 0.0005352070631257366, 0.031823074399169864, 0.0012299730482356742, 1.6074966020796362e-06, 0.005569856652383564, 0.011473060470509493, 5.9536911188134674e-08, 7.144429342576161e-07, 0.024110275933936175, 4.405731427921966e-06])
        empirical_prior = torch.cat((self.label_empirical_prior*(1-self.empirical_blank_ratio), torch.tensor([self.empirical_blank_ratio])), dim=0)
        # smooth the empirical prior to avoid 0 or very small values
        self.prior_smoothing_factor = self.cfg.prior_smoothing_factor
        self.empirical_prior = empirical_prior * (1-self.prior_smoothing_factor) + self.prior_smoothing_factor/(self.cfg.label_target_size+1)
        self.empirical_prior = torch.log(self.empirical_prior)
        assert self.empirical_prior.shape[0] == self.cfg.label_target_size + 1

        # optionally disable mhsa dropout, to be more consistent with decoding
        self.disable_dropout_rate = self.cfg.disable_dropout_rate

        # No particular weight init!

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
        if_dropout: bool=True
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

        conformer_out_layers, out_mask = self.conformer(conformer_in, mask, return_layers=self.return_layers, if_dropout=if_dropout)
        log_probs_list = []
        # estimate the prior from the current batch
        log_prior_list = []
        self.empirical_prior = self.empirical_prior.to(mask.device)
        for i, (out_layer, scale) in enumerate(zip(conformer_out_layers, self.scales)):
            if scale == 0.0:
                continue
            conformer_out = self.output_dropout(out_layer)
            logits = self.output_linears[i](conformer_out)
            log_probs = torch.log_softmax(logits, dim=2)
            bool_out_mask = out_mask.bool()
            valid_log_probs = log_probs[bool_out_mask]
            prior = torch.logsumexp(valid_log_probs,dim=0) - torch.log(out_mask.sum())
            log_prior_list.append(prior)

            log_probs_list.append(log_probs)

        return log_probs_list, log_prior_list, torch.sum(out_mask, dim=1)


def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B,]?
    if_dropout = torch.rand(()).item() > model.disable_dropout_rate

    logprobs_list, log_prior_list, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
        if_dropout=if_dropout
    )
    for logprobs, log_prior, layer_index, scale in zip(logprobs_list, log_prior_list, model.return_layers, model.scales):
        transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]

        # torch.set_printoptions(profile="full")
        print("model prior", torch.exp(log_prior))
        # print("blank_prob", torch.exp(transposed_logprobs[:10,:500,-1]).detach().cpu())
        transposed_logprobs = transposed_logprobs - model.empirical_prior[None, None,:] * model.train_prior_scale # use empirical prior rather than batch model prior

        ctc_loss = torch_ctc_fixed_grad(
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

        #print(labels_len)
        if model.prior_match_loss == "batch":
            vocab_size = logprobs.shape[-1]
            target_one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=vocab_size)
            max_L = labels.shape[1]
            mask = torch.arange(max_L, device=logprobs.device)[None,:] < labels_len[:, None]
            target_frequency = target_one_hot[mask].sum(0)/labels_len.sum() #[V]
            # prob_mass for blank, depending on the pre-designed factor
            target_frequency = target_frequency * (1-model.empirical_blank_ratio)
            target_frequency[-1] = model.empirical_blank_ratio
            target_frequency = target_frequency.detach()
            smoothing_factor = model.prior_smoothing_factor
            smooth_vector = target_frequency.new_ones(target_frequency.shape) * smoothing_factor/vocab_size
            target_frequency = target_frequency * (1-smoothing_factor) + smooth_vector
            # compute KL divergence instead of CE

            #loss_prior = - target_frequency * log_prior
            loss_prior = target_frequency* (torch.log(target_frequency)- log_prior)
            # add smoothing?
            loss_prior = loss_prior.sum()
            # print("target prior", target_frequency)
            # print("model prior", torch.exp(log_prior))
            # print("prior KLD:", loss_prior)
            #
            run_ctx.mark_as_loss(name=f"prior_match_loss{layer_index + 1}", loss=loss_prior, scale=model.prior_match_loss_scale)




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

    logprobs, logpriors, audio_features_len, = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    logprobs = logprobs[-1]

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    # remove padding
    # double check with the pre-computed log priors:
    max_T = probs.shape[1]
    vocab_size = probs.shape[-1]
    mask = torch.arange(max_T, device=probs.device)[None, :] < audio_features_len[:, None]
    marginal_priors = probs[mask].sum(dim=0)



    if run_ctx.sum_probs is None:
        #run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
        run_ctx.sum_probs = marginal_priors
    else:
        run_ctx.sum_probs += marginal_priors
