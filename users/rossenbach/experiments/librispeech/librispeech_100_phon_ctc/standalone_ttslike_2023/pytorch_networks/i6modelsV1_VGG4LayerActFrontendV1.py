"""
Trying to make the aligner more AppTek-Like

Extended weight init code
"""

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from typing import Tuple
import math
import time

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerBlockV1, ConformerEncoderV1
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config, VGG4LayerActFrontendV1

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config

from .shared.specaugment_fixed import returnn_specaugment_by_length
from .shared.feature_extraction import DbMelFeatureExtraction, DbMelFeatureExtractionConfig


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


@dataclass
class SpecaugConfig(ModelConfiguration):
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int


@dataclass
class ModelConfig():
    frontend_config: VGG4LayerActFrontendV1Config
    conformer_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    att_weights_dropout: float
    conv_dropout: float
    ff_dropout: float
    mhsa_dropout: float
    conv_kernel_size: int
    final_dropout: float


class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        fe_config = DbMelFeatureExtractionConfig.from_dict(kwargs["fe_config"])
        self.specaug_config = SpecaugConfig(
            repeat_per_n_frames=25,
            max_dim_time=20,
            max_dim_feat=16,
            num_repeat_feat=5,
        )
        frontend_config = VGG4LayerActFrontendV1Config(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(2, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            out_features=384,
            activation=nn.ReLU(),
        )
        self.cfg = ModelConfig(
            frontend_config=frontend_config,
            conformer_size=384,
            num_layers=12,
            num_heads=4,
            ff_dim=2048,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
        )

        conformer_size = self.cfg.conformer_size
        self.label_target_size = 44
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

        self.feature_extraction = DbMelFeatureExtraction(config=fe_config)
        self.conformer = ConformerEncoderV1(cfg=conformer_config)
        self.final_linear = nn.Linear(conformer_size, self.label_target_size + 1)  # + CTC blank is last
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
                    repeat_per_n_frames=self.specaug_config.repeat_per_n_frames,
                    max_dim_time=self.specaug_config.max_dim_time,
                    num_repeat_feat=self.specaug_config.num_repeat_feat,
                    max_dim_feat=self.specaug_config.max_dim_feat)
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

    phon_labels = data["phon_labels"]  # [B, N] (sparse)
    phon_labels_len = data["phon_labels:size1"]  # [B, N]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        phon_labels,
        input_lengths=audio_features_len,
        target_lengths=phon_labels_len,
        blank=model.label_target_size,
        reduction="sum",
        zero_infinity=True,
    )
    num_phonemes = torch.sum(phon_labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes)


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    from torchaudio.models.decoder import ctc_decoder
    lexicon = kwargs["lexicon"]
    import subprocess
    lm = subprocess.check_output(["cf", kwargs["arpa_lm"]]).decode().strip()
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")
    # labels = run_ctx.engine.forward_dataset.datasets["zip_dataset"].targets.labels
    from returnn.datasets.util.vocabulary import Vocabulary
    vocab = Vocabulary.create_vocab(
        vocab_file=kwargs["returnn_vocab"], unknown_label="[UNKNOWN]")
    labels = vocab.labels
    run_ctx.ctc_decoder = ctc_decoder(
            lexicon=lexicon,
            lm=lm,
            lm_weight=kwargs["lm_weight"],
            tokens=labels,
            blank_token="[blank]",
            sil_token="[space]",
            unk_word="[UNKNOWN]",
            nbest=1,
            beam_size=kwargs["beam_size"],
            beam_threshold=kwargs.get("beam_threshold", 50),
            sil_score=kwargs.get("sil_score", 0)
    )
    run_ctx.labels = labels


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

def search_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    tags = data["seq_tag"]

    start = time.time()
    hypothesis = run_ctx.ctc_decoder(logprobs.cpu(), audio_features_len.cpu())
    print("decoding_time: %.2f" % (time.time() - start))
    for hyp, tag in zip (hypothesis, tags):
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

