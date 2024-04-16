"""
Trying to make the aligner more AppTek-Like

Extended weight init code
"""

from dataclasses import dataclass
import torch
import numpy as np
from torch import nn
import multiprocessing
from librosa import filters
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union
import math

from torchaudio.functional import mask_along_axis

from i6_models.parts.blstm import BlstmEncoderV1, BlstmEncoderV1Config


from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerEncoderV1
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config

from ..shared.configs import SpecaugConfig, ModelConfigV2, FlowDecoderConfig, ConformerASRConfig


from returnn.torch.context import get_run_ctx

from ..shared.configs import DbMelFeatureExtractionConfig
from ..shared.feature_extraction import DbMelFeatureExtraction
from ..shared.spec_augment import apply_spec_aug
from ..shared.mask import mask_tensor

from ..shared import modules
from ..shared import commons
from ..shared import attentions
from ..monotonic_align import maximum_path

class XVector(nn.Module):
    def __init__(self, input_dim=40, num_classes=8, **kwargs):
        super(XVector, self).__init__()
        self.tdnn1 = modules.TDNN(
            input_dim=input_dim,
            output_dim=512,
            context_size=5,
            dilation=1,
            dropout_p=0.5,
            batch_norm=True
        )
        self.tdnn2 = modules.TDNN(
            input_dim=512, output_dim=512, context_size=3, dilation=2, dropout_p=0.5, batch_norm=True
        )
        self.tdnn3 = modules.TDNN(
            input_dim=512, output_dim=512, context_size=2, dilation=3, dropout_p=0.5, batch_norm=True
        )
        self.tdnn4 = modules.TDNN(
            input_dim=512, output_dim=512, context_size=1, dilation=1, dropout_p=0.5, batch_norm=True
        )
        self.tdnn5 = modules.TDNN(
            input_dim=512, output_dim=512, context_size=1, dilation=1, dropout_p=0.5, batch_norm=True
        )
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # fe_config = DbMelFeatureExtractionConfig.from_dict(kwargs["fe_config"])
        # self.feature_extraction = DbMelFeatureExtraction(config=fe_config)

    def forward(self, x, x_lengths):
        # with torch.no_grad():
        #     squeezed_audio = torch.squeeze(raw_audio)
        #     x, x_lengths = self.feature_extraction(squeezed_audio, raw_audio_lengths)  # [B, T, F]

        # x = x.transpose(1, 2)
        tdnn1_out = self.tdnn1(x)
        # return tdnn1_out
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        mean = torch.mean(tdnn5_out, 2)
        std = torch.std(tdnn5_out, 2)
        stat_pooling = torch.cat((mean, std), 1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        output = self.output(x_vec)
        predictions = self.softmax(output)
        return output, predictions, x_vec

class FlowDecoder(nn.Module):
    def __init__(self, cfg: FlowDecoderConfig, in_channels, gin_channels):
        """Flow-based decoder model

        Args:
            in_channels (int): Number of incoming channels
            hidden_channels (int): Number of hidden channels
            kernel_size (int): Kernel Size for convolutions in coupling blocks
            dilation_rate (float): Dilation Rate to define dilation in convolutions of coupling block
            n_blocks (int): Number of coupling blocks
            n_layers (int): Number of layers in CNN of the coupling blocks
            p_dropout (float, optional): Dropout probability for CNN in coupling blocks. Defaults to 0..
            n_split (int, optional): Number of splits for the 1x1 convolution for flows in the decoder. Defaults to 4.
            n_sqz (int, optional): Squeeze. Defaults to 1.
            sigmoid_scale (bool, optional): Boolean to define if log probs in coupling layers should be rescaled using sigmoid. Defaults to False.
            gin_channels (int, optional): Number of speaker embedding channels. Defaults to 0.
        """
        super().__init__()
        self.cfg = cfg

        self.flows = nn.ModuleList()

        for _ in range(self.cfg.n_blocks):
            self.flows.append(modules.ActNorm(channels=in_channels * self.cfg.n_sqz))
            self.flows.append(modules.InvConvNear(channels=in_channels * self.cfg.n_sqz, n_split=self.cfg.n_split))
            self.flows.append(
                attentions.CouplingBlock(
                    in_channels * self.cfg.n_sqz,
                    self.cfg.hidden_channels,
                    kernel_size=self.cfg.kernel_size,
                    dilation_rate=self.cfg.dilation_rate,
                    n_layers=self.cfg.n_layers,
                    gin_channels=gin_channels,
                    p_dropout=self.cfg.p_dropout,
                    sigmoid_scale=self.cfg.sigmoid_scale,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if g is not None:
            g = g.unsqueeze(-1)

        if self.cfg.n_sqz > 1:
            x, x_mask = commons.channel_squeeze(x, x_mask, self.cfg.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
        if self.cfg.n_sqz > 1:
            x, x_mask = commons.channel_unsqueeze(x, x_mask, self.cfg.n_sqz)
        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()

class Model(nn.Module):
    """
    Flow-based ASR model based on GlowTTS Structure using a pre-trained flow-based decoder
    trained to generate spectrograms from given statistics coming from an encoder

    Model was pretrained using the architecture in
    users/rilling/experiments/librispeech/librispeech_glowtts/pytorch_networks/glowTTS.py
    """

    def __init__(
        self,
        model_config: ModelConfigV2,
        # n_vocab: int,
        # hidden_channels: int = 192,
        # out_channels: int = 80,
        # n_blocks_dec: int = 12,
        # kernel_size_dec: int = 5,
        # dilation_rate: int = 1,
        # n_block_layers: int = 4,
        # p_dropout: float = 0.1,
        # p_dropout_flow: float = 0.05,
        # gin_channels: int = 0,
        # n_split: int = 4,
        # n_sqz: int = 2,
        # sigmoid_scale: bool = False,
        # window_size: int = 4,
        # block_length: int = None,
        # hidden_channels_dec: int = None,
        # label_target_size=None,
        # spec_augment = False,
        # layer_norm = False,
        # batch_norm = False,
        # n_speakers = 1,
        **kwargs,
    ):
        """_summary_

        Args:
            n_vocab (int): vocabulary size
            hidden_channels (int): Number of hidden channels in encoder
            out_channels (int): Number of channels in the output
            n_blocks_dec (int, optional): Number of coupling blocks in the decoder. Defaults to 12.
            kernel_size_dec (int, optional): Kernel size in the decoder. Defaults to 5.
            dilation_rate (int, optional): Dilation rate for CNNs of coupling blocks in decoder. Defaults to 5.
            n_block_layers (int, optional): Number of layers in the CNN of the coupling blocks in decoder. Defaults to 4.
            p_dropout_dec (_type_, optional): Dropout probability in the decoder. Defaults to 0..
            n_speakers (int, optional): Number of speakers. Defaults to 0.
            gin_channels (int, optional): Number of speaker embedding channels. Defaults to 0.
            n_split (int, optional): Number of splits for the 1x1 convolution for flows in the decoder. Defaults to 4.
            n_sqz (int, optional): Squeeze. Defaults to 1.
            sigmoid_scale (bool, optional): Boolean to define if log probs in coupling layers should be rescaled using sigmoid. Defaults to False.
            window_size (int, optional): Window size  in Multi-Head Self-Attention for encoder. Defaults to None.
            block_length (_type_, optional): Block length for optional block masking in Multi-Head Attention for encoder. Defaults to None.
            hidden_channels_dec (_type_, optional): Number of hidden channels in decodder. Defaults to hidden_channels.
            final_hidden_channels: Number of hidden channels in the final network
            final_n_layers: Number of layers in the final network
            label_target_size: Target size of target vocabulary, target size for final network
        """
        super().__init__()
        # self.n_vocab = n_vocab
        # self.hidden_channels = hidden_channels
        # self.out_channels = out_channels
        # self.n_blocks_dec = n_blocks_dec
        # self.kernel_size_dec = kernel_size_dec
        # self.dilation_rate = dilation_rate
        # self.n_block_layers = n_block_layers
        # self.p_dropout = p_dropout
        # self.p_dropout_flow = p_dropout_flow
        # self.n_split = n_split
        # self.n_sqz = n_sqz
        # self.sigmoid_scale = sigmoid_scale
        # self.window_size = window_size
        # self.block_length = block_length
        # self.hidden_channels_dec = hidden_channels_dec
        # self.spec_augment = spec_augment
        # self.layer_norm = layer_norm
        # self.batch_norm = batch_norm

        fe_config = DbMelFeatureExtractionConfig.from_dict(kwargs["fe_config"])
        self.feature_extraction = DbMelFeatureExtraction(config=fe_config)

        self.cfg = ModelConfigV2.from_dict(model_config)
        text_encoder_config = self.cfg.text_encoder_config
        decoder_config = self.cfg.decoder_config

        if self.cfg.n_speakers > 1:
            self.x_vector = XVector(self.cfg.out_channels, self.cfg.n_speakers)
            self.x_vector_bottleneck = nn.Sequential(nn.Linear(512, self.cfg.gin_channels), nn.ReLU())

        # self.encoder = TextEncoder(
        #     text_encoder_config, out_channels=self.cfg.out_channels, gin_channels=self.cfg.gin_channels
        # )

        self.decoder = FlowDecoder(
            decoder_config, in_channels=self.cfg.out_channels, gin_channels=self.cfg.gin_channels
        )

        if self.cfg.n_speakers > 1:
            self.x_vector = XVector(self.cfg.out_channels, self.cfg.n_speakers)
            self.x_vector_bottleneck = nn.Sequential(nn.Linear(512, self.cfg.gin_channels), nn.ReLU())


        # specaug_config = SpecaugConfig(
        #     repeat_per_n_frames=25,
        #     max_dim_time=20,
        #     max_dim_feat=16,
        #     num_repeat_feat=5,
        # )
        
        self.conf_cfg = self.cfg.phoneme_prediction_config
        frontend_config = self.conf_cfg.frontend_config
        conformer_size = self.conf_cfg.conformer_size
        conformer_config = ConformerEncoderV1Config(
            num_layers=self.conf_cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=conformer_size,
                    hidden_dim=self.conf_cfg.ff_dim,
                    dropout=self.conf_cfg.ff_dropout,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=conformer_size,
                    num_att_heads=self.conf_cfg.num_heads,
                    att_weights_dropout=self.conf_cfg.att_weights_dropout,
                    dropout=self.conf_cfg.mhsa_dropout,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=conformer_size, kernel_size=self.conf_cfg.conv_kernel_size, dropout=self.conf_cfg.conv_dropout, activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size)
                ),
            ),
        )

        self.conformer = ConformerEncoderV1(cfg=conformer_config)
        self.final_linear = nn.Linear(conformer_size, self.conf_cfg.label_target_size + 1)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.conf_cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch
        self.specaug_cfg = self.cfg.specaug_config

    def forward(self, raw_audio, raw_audio_len):
        with torch.no_grad():
            self.x_vector.eval()
            self.x_vector_bottleneck.eval()
            self.decoder.eval()
            squeezed_audio = torch.squeeze(raw_audio)
            log_mel_features, log_mel_features_len = self.feature_extraction(squeezed_audio, raw_audio_len)  # [B, T, F]

            audio_max_length = log_mel_features.size(1)

            flow_in = log_mel_features.transpose(1,2) # [B, F, T]
            flow_in, flow_in_length, flow_in_max_length = self.preprocess(flow_in, log_mel_features_len, audio_max_length)
            mask = torch.unsqueeze(commons.sequence_mask(log_mel_features_len, flow_in.size(2)), 1).to(flow_in.dtype)

            _, _, g = self.x_vector(log_mel_features.transpose(1,2), log_mel_features_len)
            g = self.x_vector_bottleneck(g)

            flow_out, _ = self.decoder(flow_in, mask, g=g, reverse=False) # [B, F, T]

            spec_augment_in = flow_out.transpose(1,2) # [B, T, F]
            mask = mask_tensor(spec_augment_in, flow_in_length)

            run_ctx = get_run_ctx()
            if self.training and self.specaug_start_epoch is not None and run_ctx.epoch > self.specaug_start_epoch:
                audio_features_masked_2 = apply_spec_aug(
                    spec_augment_in,
                    num_repeat_time=torch.max(log_mel_features_len).detach().cpu().numpy()
                    // self.cfg.specaug_config.repeat_per_n_frames,
                    max_dim_time=self.cfg.specaug_config.max_dim_time,
                    num_repeat_feat=self.cfg.specaug_config.num_repeat_feat,
                    max_dim_feat=self.cfg.specaug_config.max_dim_feat,
                )
            else:
                audio_features_masked_2 = spec_augment_in

        conformer_in = audio_features_masked_2

        conformer_out, out_mask = self.conformer(conformer_in, mask)
        conformer_out = self.final_dropout(conformer_out)
        logits = self.final_linear(conformer_out)

        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.cfg.decoder_config.n_sqz) * self.cfg.decoder_config.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = (y_lengths // self.cfg.decoder_config.n_sqz) * self.cfg.decoder_config.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()


def train_step(*, model: nn.Module, data, run_ctx, **kwargs):
    raw_audio = data["audio_features"]  # [B, T', F]
    raw_audio_len = data["audio_features:size1"]  # [B]

    phon_labels = data["phonemes_eow"]  # [B, N] (sparse)
    phon_labels_len = data["phonemes_eow:size1"]  # [B, N]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        phon_labels,
        input_lengths=audio_features_len,
        target_lengths=phon_labels_len,
        blank=model.cfg.label_target_size,
        reduction="sum",
        zero_infinity=True
    )
    num_phonemes = torch.sum(phon_labels_len)
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
        vocab_file=kwargs["returnn_vocab"], unknown_label=None)
    labels = vocab.labels
    print(f"labels from vocab:{labels}")
    if "asr_data" in kwargs.keys() and kwargs["asr_data"]:
        print(f"Using ctc_decoder for ASR data...")
        run_ctx.ctc_decoder = ctc_decoder(
            lexicon=kwargs["lexicon"],
            lm=lm,
            lm_weight=kwargs["lm_weight"],
            tokens=labels + ["[blank]", "[SILENCE]", "[UNK]"],
            # "[SILENCE]" and "[UNK]" are not actually part of the vocab,
            # but the decoder is happy as long they are defined in the token list
            # even if they do not exist as label index in the softmax output,
            blank_token="[blank]",
            sil_token="[SILENCE]",
            unk_word="[unknown]",
            nbest=1,
            beam_size=kwargs["beam_size"],
            beam_size_token=kwargs.get("beam_size_token", None),
            beam_threshold=kwargs["beam_threshold"],
            sil_score=kwargs.get("sil_score", 0.0),
            word_score=kwargs.get("word_score", 0.0),
        )
    else:
        print(f"Using ctc_decoder for TTS data...")

        run_ctx.ctc_decoder = ctc_decoder(
            lexicon=kwargs["lexicon"],
            lm=lm,
            lm_weight=kwargs["lm_weight"],
            tokens=labels,
            blank_token="[blank]",
            sil_token="[space]",  # [space] is our actual silence
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


def forward_step(*, model, data, run_ctx, **kwargs):
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


