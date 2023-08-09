import copy
import os
import numpy as np
from librosa import filters
from dataclasses import asdict, dataclass
from typing import Union, Dict, Any, Optional, Tuple

import torch
from torch import nn
from torchaudio.models import Conformer

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from ..custom_parts.feature_extraction import DbMelFeatureExtractionConfigV1, DbMelFeatureExtractionV1
from ..custom_parts.specaugment import SpecaugmentConfigV1, SpecaugmentModuleV1
from ..custom_parts.vgg_frontend import VGGFrontendConfigV1, VGGFrontendV1


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to a pytorch MHSA compatible key mask

    :param lengths: [B]
    :return: B x T, where 0 means within sequence and 1 means outside sequence
    """
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(batch_size, max_length)
    padding_mask = torch.gt(padding_mask, lengths.unsqueeze(1))
    return padding_mask


@dataclass
class ConformerCTCConfig(ModelConfiguration):
    feature_extraction_cfg: DbMelFeatureExtractionConfigV1
    specaugment_cfg: SpecaugmentConfigV1
    frontend_cfg: ModuleFactoryV1
    subsampling_factor: int = 4
    input_dim: int = 512
    target_size: int = 5048
    num_heads: int = 8
    ffn_dim: int = 2048
    num_layers: int = 12
    depthwise_conv_kernel_size: int = 31
    dropout: float = 0.1
    use_group_norm: bool = False
    convolution_first: bool = True


class ConformerCTCModel(nn.Module):
    """
    Conformer PyTorch Model

    Implemented via torchaudio.models.Conformer,
    Supports feature extraction via DbMelFeatureExtraction,
    Supports a custom VGG frontend,
    Supports a custom Specaugment
    """

    def __init__(self, step: int, cfg: ConformerCTCConfig, **kwargs):
        super().__init__()
        self.target_size = cfg.target_size
        self.dropout = cfg.dropout
        self.conformer = Conformer(
            input_dim=cfg.input_dim,
            num_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim,
            num_layers=cfg.num_layers,
            depthwise_conv_kernel_size=cfg.depthwise_conv_kernel_size,
            dropout=cfg.dropout,
            convolution_first=cfg.convolution_first,
        )
        self.final_linear = nn.Linear(cfg.input_dim, cfg.target_size + 1)
        self.subsampling_factor = cfg.subsampling_factor

        self.feature_extraction = DbMelFeatureExtractionV1(cfg=cfg.feature_extraction_cfg)
        self.specaugment = SpecaugmentModuleV1(step=step, cfg=cfg.specaugment_cfg)
        # self.vgg_frontend = vgg_frontend.VGGFrontendV1(cfg=cfg.vgg_config)
        self.vgg_frontend = cfg.frontend_cfg.construct()


    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: torch.Tensor,
    ):
        """
        :param audio_features: [B, T, F=1]
        :param audio_features_len: length of T as [B]
        """

        # Raw audio, i.e. F=1
        squeezed_features = torch.squeeze(audio_features, dim=-1) # [B, T]

        with torch.no_grad():
            # Feature extraction, [B, T, F]
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, audio_features_len)

        # Specaugment, [B, T, F]
        audio_features = self.specaugment(audio_features)

        # Conv subsampling , [B, T', F]
        audio_features = self.vgg_frontend(audio_features)
        audio_features_len = torch.ceil(audio_features_len / self.subsampling_factor)
        audio_features_len = audio_features_len.type(torch.int)

        conformer_out, _ = self.conformer(audio_features, audio_features_len)
        conformer_out_dropped = nn.functional.dropout(conformer_out, p=self.dropout)

        logits = self.final_linear(conformer_out_dropped) # [B, T, target_size]
        log_probs = torch.log_softmax(logits, dim=2) # [B, T, target_size]
        return log_probs, audio_features_len


def train_step(*, model: ConformerCTCModel, data, run_ctx, **_kwargs):
    audio_features = data["audio_features"] # [B, T, F]
    audio_features_len = data["audio_features:size1"] # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    bpes = data["bpe_labels"][indices, :] # [B, T] (sparse)
    bpes_len = data["bpe_labels:size1"][indices] # [B]

    logprobs, audio_features_len = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, target_size]
    ctc_loss = nn.functional.ctc_loss(
        log_probs=transposed_logprobs,
        targets=bpes,
        input_lengths=audio_features_len,
        target_lengths=bpes_len,
        blank=model.target_size,
        reduction="sum",
    )

    num_frames = torch.sum(bpes_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_frames)

###### SEARCH STUFF ######
def search_init_hook(run_ctx, text_lexicon, **kwargs):
    from torchaudio.models.decoder import ctc_decoder
    # Prevent forward job from crashing due to missing "output.hdf" file
    run_ctx.hdf_output = open("output.hdf", "wt")
    # Dump recognition dictionary to "recognition.txt" and use that instead
    run_ctx.recognition_file = open("recognition.txt", "wt")
    run_ctx.recognition_dict = {}

    bpe_vocab = run_ctx.engine.forward_dataset.datasets["zip_dataset"].targets.vocab_file
    bpe_tokens = os.path.join(os.path.dirname(bpe_vocab), "bpe.tokens")

    # Produce an appropriate file for ctc_decoder containing all the labels
    if not os.path.exists(bpe_tokens):
        bpe_tokens_file = open(bpe_tokens, "wt")

        bpe_dict = eval(open(bpe_vocab, "rt").read())
        sorted_bpe_dict = sorted(bpe_dict.items(), key=lambda e: e[1])  # [(label, idx), ...]

        for label, _ in sorted_bpe_dict:
            bpe_tokens_file.write(label + '\n')

        # Addition of other special tokens if required
        bpe_tokens_file.write('<sil>\n')
        bpe_tokens_file.write('<blank>')
        bpe_tokens_file.close()

    # import multiprocessing as mp
    # run_ctx.pool = mp.get_context("fork").Pool(8)
    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=text_lexicon,
        tokens=bpe_tokens,
        # lm_dict="/u/corpora/speech/LibriSpeech/lexicon/librispeech-lexicon.txt",
        lm="/work/asr3/zeineldeen/hiwis/atanas.gruev/lm/4-gram.arpa", # expects .arpa or .bin
        beam_threshold=20,
        blank_token="<blank>",
        sil_token='<sil>',
        unk_word="<unk>"
    )


def search_finish_hook(run_ctx, **kwargs):
    # run_ctx.pool.close()
    # run_ctx.pool.join()

    import json
    run_ctx.recognition_file.write(json.dumps(run_ctx.recognition_dict))
    run_ctx.recognition_file.close()
    run_ctx.hdf_output.close()


def search_step(*, model: ConformerCTCModel, data, run_ctx, **kwargs):
    audio_features = data["audio_features"]  # [B, T, F]
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    tags = [data["seq_tag"][i] for i in list(indices.cpu().numpy())]
    audio_features = audio_features[indices, :, :]

    logprobs, audio_features_len = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len,
    )

    logprobs = logprobs.detach().cpu()
    audio_features_len = audio_features_len.detach().cpu()

    # TODO: Multiprocessing is difficult to deploy due to Flashlight Dictionary
    # global decoder_worker
    # def decoder_worker(input):
    #     tag, emission, length = input
    #     hypothesis = run_ctx.ctc_decoder(emission, length)
    #     run_ctx.recognition_file.write(tag + ": " + " ".join(hypothesis[0][0].words + ',\n'))
    #
    # run_ctx.pool.map(decoder_worker, zip(tags, logprobs, audio_features_len))

    # run_ctx.pool.map(run_ctx.ctc_decoder, zip(logprobs, audio_features_len))

    hypotheses = run_ctx.ctc_decoder(logprobs, audio_features_len)

    for tag, hypothesis in zip(tags, hypotheses):
        run_ctx.recognition_dict[tag] = " ".join(hypothesis[0].words)


