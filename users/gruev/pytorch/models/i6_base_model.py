import os
import time
import copy
import math
import numpy as np

from dataclasses import dataclass
from typing import List, Dict, Optional

import torch

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_experiments.users.gruev.implementations.pytorch.blank_collapse import blank_collapse_batched as blank_collapse


def lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to a pytorch MHSA compatible key mask

    :param lengths: [B]
    :return: B x T, where 1 means within sequence and 0 means outside sequence
    """
    batch_size = lengths.shape[0]
    max_length = torch.max(lengths)
    sequence_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) < lengths.unsqueeze(1)
    return sequence_mask


@dataclass
class ConformerCTCConfig(ModelConfiguration):
    feature_extraction: Optional[ModuleFactoryV1]
    specaugment: ModuleFactoryV1
    conformer: ModuleFactoryV1
    dim: int
    target_size: int

    # Auxiliary losses
    aux_layers: List[int]
    aux_scales: List[float]
    share_aux_parameters: bool

    # Weight initialization
    fairseq_weight_init: bool


class ConformerCTCModel(torch.nn.Module):
    def __init__(self, step: int, cfg: ConformerCTCConfig, **kwargs):
        super().__init__()
        self.feature_extraction = cfg.feature_extraction() if cfg.feature_extraction else None
        self.specaugment = cfg.specaugment()
        self.conformer = cfg.conformer()
        self.target_size = cfg.target_size

        # Auxiliary losses
        self.aux_layers = cfg.aux_layers
        self.aux_scales = cfg.aux_scales
        self.share_aux_parameters = cfg.share_aux_parameters

        if self.share_aux_parameters:
            self.output_aux_linear = torch.nn.Linear(cfg.dim, cfg.target_size + 1)
        else:
            self.output_aux_linear = torch.nn.ModuleList(
                [
                    torch.nn.Linear(cfg.dim, self.target_size, cfg.target_size + 1)
                    for _ in range(len(self.aux_layers) - 1)
                ]
            )

        self.final_linear = torch.nn.Linear(cfg.dim, cfg.target_size + 1)

        if cfg.fairseq_weight_init:
            self.apply(self._weight_init)

    def forward(self, audio_features: torch.Tensor, audio_features_len: torch.Tensor):
        """
        :param audio_features: [B, T, F=1]
        :param audio_features_len: length of T as [B,]
        """
        with torch.no_grad():
            audio_features = torch.squeeze(audio_features, dim=-1)  # [B, T]
            audio_features, audio_features_len = self.feature_extraction(
                audio_features, audio_features_len
            )  # [B, T, F]

            # TODO: better specaug
            if self.training:
                audio_features_aug = self.specaugment(audio_features)  # [B, T, F]
            else:
                audio_features_aug = audio_features  # [B, T, F]

        sequence_mask = lengths_to_padding_mask(audio_features_len)

        conformer_out_list, sequence_mask = self.conformer(
            audio_features_aug,
            sequence_mask,
            self.aux_layers,
        )  # List of [B, T, F]

        log_probs_list = []
        if self.training:
            for i in range(len(self.aux_layers) - 1):
                if self.share_aux_parameters:
                    logits = self.output_aux_linear(conformer_out_list[i])
                else:
                    logits = self.output_aux_linear[i](conformer_out_list[i])
                log_probs = torch.log_softmax(logits, dim=2)
                log_probs_list.append(log_probs)

        # Important: final linear layer is kept separate
        logits = self.final_linear(conformer_out_list[-1])
        log_probs = torch.log_softmax(logits, dim=2)
        log_probs_list.append(log_probs)

        return log_probs_list, torch.sum(sequence_mask, dim=1)

    @staticmethod
    def _weight_init(module):
        for param_name, param in module.named_parameters():
            # Only detects for "conformer.module_list.i.mhsa.mhsa"
            if param_name in ["in_proj_weight", "out_proj.weight"]:
                torch.nn.init.xavier_uniform_(param, gain=1 / math.sqrt(2))


def train_step(*, model: ConformerCTCModel, data, run_ctx, **_kwargs):
    audio_features = data["audio_features"]  # [B, T, F]
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    bpes = data["bpe_labels"][indices, :]  # [B, T] (sparse)
    bpes_len = data["bpe_labels:size1"][indices]  # [B]
    num_frames = torch.sum(bpes_len)

    log_probs_list, sequence_mask = model(audio_features=audio_features, audio_features_len=audio_features_len)

    for i in range(len(log_probs_list)):
        log_probs = torch.permute(log_probs_list[i], (1, 0, 2))  # [T, B, F]
        loss = torch.nn.functional.ctc_loss(
            log_probs,
            bpes,
            input_lengths=sequence_mask,
            target_lengths=bpes_len,
            blank=model.target_size,
            reduction="sum",
            zero_infinity=True,
        )
        run_ctx.mark_as_loss(
            name=f"CTC_interLoss_{model.aux_layers[i]}",
            loss=loss,
            scale=model.aux_scales[i],
            inv_norm_factor=num_frames,
        )


###### SEARCH STUFF ######
def search_init_hook(run_ctx, **kwargs):
    from torchaudio.models.decoder import ctc_decoder

    # Dump recognition dictionary to "recognition.txt" and use that instead
    run_ctx.recognition_file = open("recognition.txt", "wt")
    run_ctx.recognition_file.write("{\n")

    # Use context manager via cf to accelerate LM loading
    import subprocess

    arpa_lm = kwargs.get("arpa_lm", None)
    lm = subprocess.check_output(["cf", arpa_lm]).decode().strip() if arpa_lm else None

    # Get labels directly, no need to load the vocab file
    labels = run_ctx.engine.forward_dataset.datasets["zip_dataset"].targets.labels

    # Use blank_collapse for faster decoding
    run_ctx.blank_collapse = kwargs.get("blank_collapse", False)

    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=kwargs.get("text_lexicon", None),
        lm=lm,
        lm_weight=kwargs.get("lm_weight", 1),
        tokens=labels + ["[SILENCE]", "[blank]"],
        blank_token="[blank]",
        sil_token="[SILENCE]",
        unk_word="[UNKNOWN]",
        beam_size=kwargs.get("beam_size", 50),
        beam_size_token=kwargs.get("beam_size_token", len(labels)),
        beam_threshold=kwargs.get("beam_threshold", 50),
    )


def search_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    f = open("output.hdf", "wt")
    f.write("")
    f.close()


def search_step(*, model: ConformerCTCModel, data, run_ctx, **kwargs):
    audio_features = data["audio_features"]  # [B, T, F]
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    tags = [data["seq_tag"][i] for i in list(indices.cpu().numpy())]
    audio_features = audio_features[indices, :, :]

    log_probs_list, audio_features_len = model(audio_features, audio_features_len)

    from IPython import embed
    embed()

    # see also model.forward()
    log_probs = log_probs_list[0]

    # TODO
    if run_ctx.blank_collapse:
        log_probs, audio_features_len = blank_collapse(
            log_probs,
            audio_features_len,
            blank_threshold=torch.log(torch.Tensor([0.995])),
            blank_idx=model.target_size,
        )

    hypotheses = run_ctx.ctc_decoder(log_probs, audio_features_len)

    for tag, hypothesis in zip(tags, hypotheses):
        words = hypothesis[0].words
        sequence = " ".join(words)
        print(sequence)
        run_ctx.recognition_file.write(f"{repr(tag)}: {repr(sequence)},\n")
