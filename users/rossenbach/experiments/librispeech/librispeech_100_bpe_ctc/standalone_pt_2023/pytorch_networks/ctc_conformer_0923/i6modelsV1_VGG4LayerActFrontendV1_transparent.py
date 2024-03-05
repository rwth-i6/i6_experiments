"""
Trying to make the aligner more AppTek-Like

Extended weight init code
"""

import numpy as np
import torch
from torch import nn
from typing import Tuple
import math

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerBlockV1, ConformerEncoderV1
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config, VGG4LayerActFrontendV1

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config

from .i6modelsV1_VGG4LayerActFrontendV1_cfg import ModelConfig
from .specaugment_fixed import returnn_specaugment_by_length
from .legacy_feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config


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


class TransparentConformerEncoderV1(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.

    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV1(cfg.block_cfg) for _ in range(cfg.num_layers)])
        self.transparent_scales = nn.Parameter(torch.empty((cfg.num_layers + 1,)))

        torch.nn.init.constant_(self.transparent_scales, 1 / (cfg.num_layers + 1))

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']

        transparent_weights = torch.softmax(self.transparent_scales + 0.001, dim=0)
        print(transparent_weights)

        final = transparent_weights[0] * x
        for i, module in enumerate(self.module_list):
            x = module(x, sequence_mask)  # [B, T, F']
            final = final + (transparent_weights[i + 1] * x)
        return final, sequence_mask


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        fe_config = LogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            f_min=60,
            f_max=7600,
            min_amp=1e-10,
            num_filters=80,
            center=False,
        )
        frontend_config = self.cfg.frontend_config
        conformer_size = self.cfg.conformer_size
        self.label_target_size = self.cfg.label_target_size
        conformer_config = ConformerEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=conformer_size,
                    hidden_dim=conformer_size,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    dropout=self.cfg.mhsa_dropout,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=conformer_size, kernel_size=self.cfg.conv_kernel_size, dropout=self.cfg.conv_dropout, activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size)
                ),
            ),
        )

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=fe_config)
        self.conformer = TransparentConformerEncoderV1(cfg=conformer_config)
        self.final_linear = nn.Linear(conformer_size, self.label_target_size + 1)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)

        self.export_mode = False

        # No particular weight init!

    def forward(
            self,
            raw_audio: torch.Tensor,
            raw_audio_len: torch.Tensor,
    ):
        """

        :param raw_audio:
        :param raw_audio_len:
        :return: logprobs [B, T, #labels + blank]
        """
        
        squeezed_features = torch.squeeze(raw_audio)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            if self.training:
                audio_features_masked_2 = returnn_specaugment_by_length(
                    audio_features,
                    repeat_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    max_dim_time=self.cfg.specaug_config.max_dim_time,
                    num_repeat_feat=self.cfg.specaug_config.num_repeat_feat,
                    max_dim_feat=self.cfg.specaug_config.max_dim_feat)
            else:
                audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)

        conformer_out, out_mask = self.conformer(conformer_in, mask)
        conformer_out = self.final_dropout(conformer_out)
        logits = self.final_linear(conformer_out)

        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)


def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    bpe_labels = data["bpe_labels"]  # [B, N] (sparse)
    bpe_labels_len = data["bpe_labels:size1"]  # [B, N]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        bpe_labels,
        input_lengths=audio_features_len,
        target_lengths=bpe_labels_len,
        blank=model.label_target_size,
        reduction="sum",
    )
    num_phonemes = torch.sum(bpe_labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes)


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    from torchaudio.models.decoder import ctc_decoder
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")
    import subprocess
    if kwargs["arpa_lm"] is not None:
        lm = subprocess.check_output(["cf", kwargs["arpa_lm"]]).decode().strip()
    else:
        lm = None
    from returnn.datasets.util.vocabulary import Vocabulary
    vocab = Vocabulary.create_vocab(
        vocab_file=kwargs["returnn_vocab"],
        unknown_label=None)
    labels = vocab.labels
    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=kwargs["lexicon"],
        lm=lm,
        lm_weight=kwargs["lm_weight"],
        tokens=labels + ["[blank]", "[SILENCE]"],
        blank_token="[blank]",
        sil_token="[SILENCE]",
        unk_word="[UNKNOWN]",
        nbest=1,
        beam_size=kwargs["beam_size"],
        beam_threshold=kwargs["beam_threshold"],
        sil_score=kwargs.get("sil_score", 0.0),
        word_score=kwargs.get("word_score", 0.0),
    )
    run_ctx.labels = labels
    run_ctx.blank_log_penalty = kwargs.get("blank_log_penalty", None)

    if kwargs.get("prior_file", None):
        run_ctx.prior = np.loadtxt(kwargs["prior_file"], dtype="float32")
        run_ctx.prior_scale = kwargs["prior_scale"]
    else:
        run_ctx.prior = None


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    # write empty HDF until new ForwardJob exists
    f = open("output.hdf", "wt")
    f.write(" ")
    f.close()


def search_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    tags = data["seq_tag"]

    logprobs_cpu = logprobs.cpu()
    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last
        logprobs_cpu[:, :, -1] -= run_ctx.blank_log_penalty
    if run_ctx.prior is not None:
        logprobs_cpu -= run_ctx.prior_scale * run_ctx.prior

    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len.cpu())

    for hyp, tag in zip(hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))


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
    with open("prior.txt", 'w') as f:
        np.savetxt(f, log_average_probs, delimiter=' ')
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))

