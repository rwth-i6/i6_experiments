"""
Trying to make the aligner more AppTek-Like

Extended weight init code
"""

from dataclasses import dataclass
import torch
from torch import nn
from typing import Any, Dict, Optional, Tuple, Union
import math

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config, ConformerEncoderV1
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerBlockV1
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.parts.frontend.common import mask_pool

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config, ConformerConvolutionV1
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config, ConformerPositionwiseFeedForwardV1
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config, ConformerMHSAV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length

from returnn.torch.context import get_run_ctx

from .i6modelsV1_VGG4LayerActFrontendV1_cfg import ModelConfig, VGG4LayerActFrontendV1Config_mod, SpecaugConfig


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


class ESPNetPositionalEncoding(torch.nn.Module):
    """
    Absolute positional encoding taken from ESPNet, reformulated in i6-style
    https://github.com/espnet/espnet/blob/5d0758e2a7063b82d1f10a8ac2de98eb6cf8a352/espnet/nets/pytorch_backend/transformer/embedding.py#L35

    :param d_model: Embedding dimension.
    :param dropout_rate: Dropout rate.
    :param max_len: Maximum input length.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super(ESPNetPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """
        Reset the positional encodings.

        :param x:
        """
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        :param x: Input tensor [B, T, *]
        :returns: Masked tensor [B, T, *]
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class VGG4LayerActFrontendV1PosEnc(VGG4LayerActFrontendV1):

    def __init__(self, cfg: VGG4LayerActFrontendV1Config):
        super().__init__(cfg)
        self.posenc = ESPNetPositionalEncoding(self.cfg.out_features, 0.1)

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor, sequence_mask = super().forward(tensor, sequence_mask)
        tensor = self.posenc(tensor)
        return tensor, sequence_mask


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
       
        torch.nn.init.constant_(self.transparent_scales, 1/(cfg.num_layers + 1))


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


class Model(nn.Module):
    """
    Conformer Encoder-Decoder Attention model for ASR
    """

    def __init__(self, cfg: ConformerAEDModelConfig):
        super().__init__()

        self.feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feat_extraction_cfg)
        # TODO: add specaugment
        self.encoder = ConformerEncoderV1ConvFirst(cfg=cfg.encoder_cfg)
        self.vocab_size = cfg.decoder_cfg.vocab_size
        self.ctc_linear = nn.Linear(in_features=cfg.encoder_cfg.block_cfg.ff_cfg.input_dim,
                                    out_features=self.vocab_size + 1)  # +1 for CTC
        self.decoder = AttentionLSTMDecoderV1(cfg=cfg.decoder_cfg)

        self.apply(self._weight_init)

    @staticmethod
    def _weight_init(module: torch.nn.Module):
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            print("apply xavier weight init for %s" % str(module))
            nn.init.xavier_uniform_(module.weight)

    def forward(
            self,
            raw_audio: torch.Tensor,
            raw_audio_len: torch.Tensor,
            bpe_labels: torch.Tensor,
    ):
        """
        :param raw_audio: audio raw samples of shape [B,T,F=1]
        :param raw_audio_len: audio sequence length of shape [B]
        :param bpe_labels: bpe targets of shape [B,N]
        :return:
        """

        squeezed_features = torch.squeeze(raw_audio)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            if self.training:
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
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)

        time_arange = torch.arange(audio_features.size(1), device=audio_features.device)  # [0, ..., T-1]
        time_mask = torch.less(time_arange[None, :], audio_features_len[:, None])  # [B,T]

        encoder_outputs, encoder_seq_mask = self.encoder(audio_features, time_mask)
        decoder_logits, state = self.decoder(
            encoder_outputs, bpe_labels, audio_features_len
        )

        encoder_seq_len = torch.sum(encoder_seq_mask, dim=1)  # [B]
        ctc_logits = self.ctc_linear(encoder_outputs)  # [B, T, #vocab + 1]
        ctc_logprobs = nn.functional.log_softmax(ctc_logits, dim=2)

        return decoder_logits, state, encoder_outputs, encoder_seq_len, ctc_logprobs


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
        conformer_config = ConformerEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1PosEnc, cfg=frontend_config),
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=conformer_size,
                    hidden_dim=self.cfg.ff_dim,
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

        run_ctx = get_run_ctx()
        dataset = run_ctx.engine.train_dataset or run_ctx.engine.forward_dataset
        self.label_target_size = len(dataset.datasets["zip_dataset"].targets.labels)

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=fe_config)
        self.conformer = TransparentConformerEncoderV1(cfg=conformer_config)
        self.final_linear = nn.Linear(conformer_size, self.label_target_size + 1)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)

        # initialize weights
        self.apply(self._weight_init)

    @staticmethod
    def _weight_init(module: torch.nn.Module):
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            print("apply weight init for %s" % str(module))
            nn.init.xavier_uniform_(module.weight)

    def forward(
            self,
            raw_audio: torch.Tensor,
            raw_audio_len: torch.Tensor,
    ): 
        
        squeezed_features = torch.squeeze(raw_audio)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            if self.training:
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
    labels = run_ctx.engine.forward_dataset.datasets["zip_dataset"].targets.labels
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
    )
    run_ctx.labels = labels
    run_ctx.blank_log_penalty = kwargs.get("blank_log_penalty", None)


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

    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len.cpu())

    for hyp, tag in zip(hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))

