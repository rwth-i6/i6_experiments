from abc import abstractmethod
from typing import List, Protocol, Sequence, Tuple, Union, Optional, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpad_sequence
import numpy as np

import returnn.frontend as rf
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import TensorDict

from .util import get_random_mask, mask_sequence
from .aed_denoising_discrete import DenoisingAedModel
from . import aed_denoising_discrete


class SharedDenoisingAedModel(DenoisingAedModel):
    mask_idx: int
    bos_idx: int
    eos_idx: int
    embedding: nn.Embedding
    decoder: nn.Module

    audio_mask_idx: int
    audio_bos_idx: int
    audio_eos_idx: int
    audio_embedding: nn.Embedding
    audio_decoder: nn.Module

    text_mask_idx: int
    text_bos_idx: int
    text_eos_idx: int
    text_embedding: nn.Embedding
    text_decoder: nn.Module

    @abstractmethod
    def forward_text(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward the data through the encoder.

        :return:
        """

    @abstractmethod
    def forward_audio(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward the data through the encoder.

        :return:
        """

    @abstractmethod
    def decode_audio_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        """
        Forward the decoder for the entire sequence in `x`, discarding any intermediate state afterwards.

        :param x: current sequence to be decoded
        :param x_lens: length of the seqs in x
        :param encoder_output: output of the encoder
        :param encoder_output_mask: padding mask of the encoder output
        """

    @abstractmethod
    def decode_text_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        """
        Forward the decoder for the entire sequence in `x`, discarding any intermediate state afterwards.

        :param x: current sequence to be decoded
        :param x_lens: length of the seqs in x
        :param encoder_output: output of the encoder
        :param encoder_output_mask: padding mask of the encoder output
        """

    @abstractmethod
    def decode_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        """
        Forward the decoder for the entire sequence in `x`, discarding any intermediate state afterwards.

        :param x: current sequence to be decoded
        :param x_lens: length of the seqs in x
        :param encoder_output: output of the encoder
        :param encoder_output_mask: padding mask of the encoder output
        """

def train_step(
        *,
        model: SharedDenoisingAedModel,
        extern_data: TensorDict,
        audio_ce_loss_scale: Optional[float] = None,
        audio_masked_ce_loss_scale: Optional[float] = None,
        text_ce_loss_scale: Optional[float] = None,
        text_masked_ce_loss_scale: Optional[float] = None,
        label_smoothing: float = 0.0,
        label_smoothing_start_epoch: int = 0,
        text_masking_opts: Optional[Dict] = None,
        audio_masking_opts: Optional[Dict] = None,
        aux_loss_scales: Optional[Sequence[float]] = None,
        **_kwargs,
):
    assert set(extern_data.data.keys()) == {"data", "phon_indices", "seq_tag"}
    audio_indices_: ReturnnTensor = extern_data["data"]
    audio_indices_lens: Tensor = audio_indices_.dims[1].dyn_size_ext.raw_tensor
    phon_indices_: ReturnnTensor = extern_data["phon_indices"]
    phon_indices_lens: Tensor = phon_indices_.dims[1].dyn_size_ext.raw_tensor

    # if torch.all(phon_indices_lens > 0).item():
    #     assert torch.all(audio_indices_lens == 0).item()
    model.decode_seq = model.decode_text_seq
    model.forward = model.forward_text
    model.mask_idx = model.text_mask_idx
    model.bos_idx = model.text_bos_idx
    model.eos_idx = model.text_eos_idx
    model.embedding = model.text_embedding
    model.decoder = model.text_decoder
    aed_denoising_discrete.train_step(
        model=model,
        extern_data=TensorDict({"data": phon_indices_, "seq_tag": extern_data["seq_tag"]}),
        ce_loss_scale=text_ce_loss_scale,
        masked_ce_loss_scale=text_masked_ce_loss_scale,
        label_smoothing=label_smoothing,
        label_smoothing_start_epoch=label_smoothing_start_epoch,
        masking_opts=text_masking_opts,
        aux_loss_scales=aux_loss_scales,
        loss_name="text"
    )

    # elif torch.all(audio_indices_lens > 0).item():
    #     assert torch.all(phon_indices_lens == 0).item()
    model.decode_seq = model.decode_audio_seq
    model.forward = model.forward_audio
    model.mask_idx = model.audio_mask_idx
    model.bos_idx = model.audio_bos_idx
    model.eos_idx = model.audio_eos_idx
    model.embedding = model.audio_embedding
    model.decoder = model.audio_decoder
    aed_denoising_discrete.train_step(
        model=model,
        extern_data=TensorDict({"data": audio_indices_, "seq_tag": extern_data["seq_tag"]}),
        ce_loss_scale=audio_ce_loss_scale,
        masked_ce_loss_scale=audio_masked_ce_loss_scale,
        label_smoothing=label_smoothing,
        label_smoothing_start_epoch=label_smoothing_start_epoch,
        masking_opts=audio_masking_opts,
        aux_loss_scales=aux_loss_scales,
        loss_name="audio"
    )
    # else:
    #     raise ValueError("In each batch, either audio or phoneme indices must be present exclusively.")

def train_step_v2(
        *,
        model: SharedDenoisingAedModel,
        extern_data: TensorDict,
        audio_ce_loss_scale: Optional[float] = None,
        audio_masked_ce_loss_scale: Optional[float] = None,
        text_ce_loss_scale: Optional[float] = None,
        text_masked_ce_loss_scale: Optional[float] = None,
        label_smoothing: float = 0.0,
        label_smoothing_start_epoch: int = 0,
        text_masking_opts: Optional[Dict] = None,
        audio_masking_opts: Optional[Dict] = None,
        aux_loss_scales: Optional[Sequence[float]] = None,
        **_kwargs,
):
    assert set(extern_data.data.keys()) == {"data", "phon_indices", "seq_tag"}
    audio_indices_: ReturnnTensor = extern_data["data"]
    audio_indices_lens: Tensor = audio_indices_.dims[1].dyn_size_ext.raw_tensor
    phon_indices_: ReturnnTensor = extern_data["phon_indices"]
    phon_indices_lens: Tensor = phon_indices_.dims[1].dyn_size_ext.raw_tensor

    # if torch.all(phon_indices_lens > 0).item():
    #     assert torch.all(audio_indices_lens == 0).item()
    model.decode_seq = model.decode_text_seq
    model.forward = model.forward_text
    model.mask_idx = model.text_mask_idx
    model.bos_idx = model.text_bos_idx
    model.eos_idx = model.text_eos_idx
    model.embedding = model.text_embedding
    model.decoder = model.text_decoder
    aed_denoising_discrete.train_step_v2(
        model=model,
        extern_data=TensorDict({"data": phon_indices_, "seq_tag": extern_data["seq_tag"]}),
        ce_loss_scale=text_ce_loss_scale,
        masked_ce_loss_scale=text_masked_ce_loss_scale,
        label_smoothing=label_smoothing,
        label_smoothing_start_epoch=label_smoothing_start_epoch,
        masking_opts=text_masking_opts,
        aux_loss_scales=aux_loss_scales,
        loss_name="text"
    )

    # elif torch.all(audio_indices_lens > 0).item():
    #     assert torch.all(phon_indices_lens == 0).item()
    model.decode_seq = model.decode_audio_seq
    model.forward = model.forward_audio
    model.mask_idx = model.audio_mask_idx
    model.bos_idx = model.audio_bos_idx
    model.eos_idx = model.audio_eos_idx
    model.embedding = model.audio_embedding
    model.decoder = model.audio_decoder
    aed_denoising_discrete.train_step_v2(
        model=model,
        extern_data=TensorDict({"data": audio_indices_, "seq_tag": extern_data["seq_tag"]}),
        ce_loss_scale=audio_ce_loss_scale,
        masked_ce_loss_scale=audio_masked_ce_loss_scale,
        label_smoothing=label_smoothing,
        label_smoothing_start_epoch=label_smoothing_start_epoch,
        masking_opts=audio_masking_opts,
        aux_loss_scales=aux_loss_scales,
        loss_name="audio"
    )
