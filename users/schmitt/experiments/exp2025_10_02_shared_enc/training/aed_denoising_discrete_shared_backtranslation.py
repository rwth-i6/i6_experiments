from abc import abstractmethod
from typing import List, Protocol, Sequence, Tuple, Union, Optional, Dict, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpad_sequence
import numpy as np

import returnn.frontend as rf
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import Dim as ReturnnDim
from returnn.tensor import TensorDict

from .util import get_random_mask, mask_sequence
from .aed_denoising_discrete import DenoisingAedModel
from . import aed_denoising_discrete
from ..recognition.aed.beam_search import beam_search_v1


class SharedDenoisingAedModel(DenoisingAedModel):
    mask_idx: int
    bos_idx: int
    eos_idx: int
    embedding: nn.Embedding
    decoder: nn.Module

    audio_mask_idx: int
    audio_bos_idx: int
    audio_eos_idx: int
    audio_blank_idx: int
    audio_out_dim: int
    audio_embedding: nn.Embedding
    audio_decoder: nn.Module
    audio_aux_loss_layers: List[int]

    text_mask_idx: int
    text_bos_idx: int
    text_eos_idx: int
    text_blank_idx: int
    text_out_dim: int
    text_embedding: nn.Embedding
    text_decoder: nn.Module
    text_aux_loss_layers: List[int]

    discriminator: Optional[nn.Module]

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


def backtranslation_step(
    model: SharedDenoisingAedModel,
    extern_data: TensorDict,
    ce_loss_scale: float,
    label_smoothing: float,
    label_smoothing_start_epoch: int,
    aux_loss_scales: Optional[Sequence[float]],
    src_indices_: ReturnnTensor,
    forward_target: Callable,
    target_decoder: nn.Module,
    target_bos_idx: int,
    target_eos_idx: int,
    target_out_dim: int,
    forward_src: Callable,
    src_decoder: nn.Module,
    decode_src_seq: Callable,
    src_bos_idx: int,
    src_eos_idx: int,
    src_blank_idx: int,
    src_out_dim: int,
    loss_name: str,
):
    src_indices: Tensor = src_indices_.raw_tensor
    src_indices_lens: Tensor = src_indices_.dims[1].dyn_size_ext.raw_tensor
    if torch.any(src_indices_lens != 0).item():
        src_indices = src_indices[src_indices_lens > 0]
        src_indices_lens = src_indices_lens[src_indices_lens > 0]

        with torch.no_grad():
            seq_len = src_indices_lens.to(src_indices.device)
            target_decoder_state = model.forward_encoder(
                src_indices,
                seq_len,
                decoder=target_decoder,
                forward_func=forward_src
            )
            pseudo_target_indices, _, _, pseudo_target_indices_lens = beam_search_v1(
                model=model,
                beam_size=1,
                batch_size=src_indices.shape[0],
                decoder_state=target_decoder_state,
                device=src_indices.device,
                max_seq_len=seq_len * 2,
                # decoder=target_decoder,
                bos_idx=target_bos_idx,
                eos_idx=target_eos_idx,
                out_dim=target_out_dim,
            )
            # get rid of beam dim and EOS idx
            pseudo_target_indices = pseudo_target_indices[:, 0, :-1]  # B, T
            pseudo_target_indices_lens = pseudo_target_indices_lens[:, 0]  # B
            batch_dim = ReturnnDim(pseudo_target_indices.shape[0], name="batch")
            target_vocab_dim = ReturnnDim(target_out_dim, name="vocab")
            pseudo_target_lens_data = rf.convert_to_tensor(pseudo_target_indices_lens, dims=[batch_dim])
            pseudo_target_lens_dim = ReturnnDim(pseudo_target_lens_data, name="seq_len")
            pseudo_target_indices_ = rf.convert_to_tensor(
                pseudo_target_indices, dims=[batch_dim, pseudo_target_lens_dim], sparse_dim=target_vocab_dim)

            src_vocab_dim = ReturnnDim(src_out_dim, name="vocab")
            src_indices_lens_data = rf.convert_to_tensor(src_indices_lens, dims=[batch_dim])
            src_indices_lens_dim = ReturnnDim(src_indices_lens_data, name="seq_len")
            src_indices_ = rf.convert_to_tensor(
                src_indices, dims=[batch_dim, src_indices_lens_dim], sparse_dim=src_vocab_dim)

        model.decode_seq = decode_src_seq
        model.forward = forward_target
        model.mask_idx = None # model.audio_mask_idx  # not used
        model.bos_idx = src_bos_idx
        model.eos_idx = src_eos_idx
        model.blank_idx = src_blank_idx
        model.decoder = src_decoder
        aed_denoising_discrete.train_step(
            model=model,
            extern_data=TensorDict({
                "data": pseudo_target_indices_,
                "target": src_indices_,
                "seq_tag": extern_data["seq_tag"]
            }),
            ce_loss_scale=ce_loss_scale,
            masked_ce_loss_scale=0.0,
            label_smoothing=label_smoothing,
            label_smoothing_start_epoch=label_smoothing_start_epoch,
            masking_opts={
                "mask_prob": 0.0,
                "min_span": 0,
                "max_span": 0,
            },
            aux_loss_scales=aux_loss_scales,
            loss_name=loss_name,
        )


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
        text_aux_loss_scales: Optional[Sequence[float]] = None,
        audio_aux_loss_scales: Optional[Sequence[float]] = None,
        pseudo_audio_text_ce_loss_scale: float = 1.0,
        pseudo_text_audio_ce_loss_scale: float = 0.0,
        adv_loss_scale: float = 0.0,
        **_kwargs,
):
    assert set(extern_data.data.keys()) == {"data", "phon_indices", "seq_tag"}
    if "data" in extern_data:
        audio_indices_: ReturnnTensor = extern_data["data"]
    else:
        audio_indices_ = None
    if "phon_indices" in extern_data:
        phon_indices_: ReturnnTensor = extern_data["phon_indices"]
    else:
        phon_indices_ = None

    if text_ce_loss_scale > 0.0 or text_masked_ce_loss_scale > 0.0:
        model.decode_seq = model.decode_text_seq
        model.forward = model.forward_text
        model.mask_idx = model.text_mask_idx
        model.bos_idx = model.text_bos_idx
        model.eos_idx = model.text_eos_idx
        model.decoder = model.text_decoder
        aed_denoising_discrete.train_step(
            model=model,
            extern_data=TensorDict({"data": phon_indices_, "seq_tag": extern_data["seq_tag"]}),
            ce_loss_scale=text_ce_loss_scale,
            masked_ce_loss_scale=text_masked_ce_loss_scale,
            label_smoothing=label_smoothing,
            label_smoothing_start_epoch=label_smoothing_start_epoch,
            masking_opts=text_masking_opts,
            aux_loss_scales=None,
            loss_name="text",
            adv_loss_scale=adv_loss_scale,
            true_adv_target=1,  # real text
        )

    if audio_ce_loss_scale > 0.0 or audio_masked_ce_loss_scale > 0.0:
        model.decode_seq = model.decode_audio_seq
        model.forward = model.forward_audio
        model.mask_idx = model.audio_mask_idx
        model.bos_idx = model.audio_bos_idx
        model.eos_idx = model.audio_eos_idx
        model.decoder = model.audio_decoder
        aed_denoising_discrete.train_step(
            model=model,
            extern_data=TensorDict({"data": audio_indices_, "seq_tag": extern_data["seq_tag"]}),
            ce_loss_scale=audio_ce_loss_scale,
            masked_ce_loss_scale=audio_masked_ce_loss_scale,
            label_smoothing=label_smoothing,
            label_smoothing_start_epoch=label_smoothing_start_epoch,
            masking_opts=audio_masking_opts,
            aux_loss_scales=None,
            loss_name="audio",
            adv_loss_scale=adv_loss_scale,
            true_adv_target=0,  # real audio
        )

    if pseudo_audio_text_ce_loss_scale > 0.0:
        model.step_decoder = model.step_audio_decoder
        backtranslation_step(
            model=model,
            extern_data=extern_data,
            ce_loss_scale=pseudo_audio_text_ce_loss_scale,
            label_smoothing=label_smoothing,
            label_smoothing_start_epoch=label_smoothing_start_epoch,
            aux_loss_scales=text_aux_loss_scales,
            src_indices_=phon_indices_,
            forward_target=model.forward_audio,
            target_decoder=model.audio_decoder,
            target_bos_idx=model.audio_bos_idx,
            target_eos_idx=model.audio_eos_idx,
            target_out_dim=model.audio_out_dim,
            forward_src=model.forward_text,
            src_decoder=model.text_decoder,
            decode_src_seq=model.decode_text_seq,
            src_bos_idx=model.text_bos_idx,
            src_eos_idx=model.text_eos_idx,
            src_blank_idx=model.text_blank_idx,
            src_out_dim=model.text_out_dim,
            loss_name="pseudo",
        )

    if pseudo_text_audio_ce_loss_scale > 0.0:
        model.step_decoder = model.step_text_decoder
        backtranslation_step(
            model=model,
            extern_data=extern_data,
            ce_loss_scale=pseudo_text_audio_ce_loss_scale,
            label_smoothing=label_smoothing,
            label_smoothing_start_epoch=label_smoothing_start_epoch,
            aux_loss_scales=audio_aux_loss_scales,
            src_indices_=audio_indices_,
            forward_target=model.forward_text,
            target_decoder=model.text_decoder,
            target_bos_idx=model.text_bos_idx,
            target_eos_idx=model.text_eos_idx,
            target_out_dim=model.text_out_dim,
            forward_src=model.forward_audio,
            src_decoder=model.audio_decoder,
            decode_src_seq=model.decode_audio_seq,
            src_bos_idx=model.audio_bos_idx,
            src_eos_idx=model.audio_eos_idx,
            src_blank_idx=model.audio_blank_idx,
            src_out_dim=model.audio_out_dim,
            loss_name="pseudo_text-audio"
        )
