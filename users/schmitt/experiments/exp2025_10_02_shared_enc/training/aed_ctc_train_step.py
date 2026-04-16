__all__ = ["SharedEncoderAedModel", "train_step"]

from abc import abstractmethod
from typing import List, Protocol, Sequence, Tuple, Union, Optional, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpad_sequence

import returnn.frontend as rf
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import TensorDict

from .util import get_random_mask, mask_sequence


class SharedEncoderAedModel(Protocol):
    mask_idx: int
    bos_idx: int
    eos_idx: int

    embedding: nn.Embedding

    @abstractmethod
    def forward_text(self, text_indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward the data through the encoder.

        :return:
        """

    @abstractmethod
    def forward_features(self, features: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward the data through the encoder.

        :return:
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
    def decode_audio_seq(
            self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward the decoder for the entire sequence in `x`, discarding any intermediate state afterwards.

        :param x: current sequence to be decoded
        :param x_lens: length of the seqs in x
        :param encoder_output: output of the encoder
        :param encoder_output_mask: padding mask of the encoder output
        """


def train_step(
        *,
        model: SharedEncoderAedModel,
        extern_data: TensorDict,
        eos_ce_loss_scale: Optional[float] = None,
        ce_loss_scale: Optional[float] = None,
        masked_ce_loss_scale: Optional[float] = None,
        masked_mse_loss_scale: Optional[float] = None,
        mse_loss_scale: Optional[float] = None,
        label_smoothing: float = 0.0,
        label_smoothing_start_epoch: int = 0,
        label_masking_opts: Optional[Dict] = None,
        feature_masking_opts: Optional[Dict] = None,
        aux_loss_scales: Optional[Sequence[float]] = None,
        audio_loss: str = "mse",
        **_kwargs,
):
    ctx = rf.get_run_ctx()
    num_eos_symbols = 1

    if set(extern_data.data.keys()) == {"data", "seq_tag"}:
        if extern_data["data"].sparse_dim is not None:
            phon_indices_: ReturnnTensor = extern_data["data"]
            phon_indices: Tensor = phon_indices_.raw_tensor
            phon_indices_lens: Tensor = phon_indices_.dims[1].dyn_size_ext.raw_tensor
            audio_features = None
            audio_features_lens = None
        else:
            audio_features_: ReturnnTensor = extern_data["data"]
            audio_features: Tensor = audio_features_.raw_tensor
            audio_features_lens: Tensor = audio_features_.dims[1].dyn_size_ext.raw_tensor
            phon_indices = None
            phon_indices_lens = None
    else:
        assert set(extern_data.data.keys()) == {"data", "phon_indices", "seq_tag"}
        audio_features_: ReturnnTensor = extern_data["data"]
        audio_features: Tensor = audio_features_.raw_tensor
        audio_features_lens: Tensor = audio_features_.dims[1].dyn_size_ext.raw_tensor
        phon_indices_: ReturnnTensor = extern_data["phon_indices"]
        phon_indices: Tensor = phon_indices_.raw_tensor
        phon_indices_lens: Tensor = phon_indices_.dims[1].dyn_size_ext.raw_tensor
        phon_indices = phon_indices[phon_indices_lens > 0]
        phon_indices_lens = phon_indices_lens[phon_indices_lens > 0]
        audio_features = audio_features[audio_features_lens > 0]
        audio_features_lens = audio_features_lens[audio_features_lens > 0]

    if phon_indices is not None and phon_indices_lens.size(0) != 0:
        assert ce_loss_scale is not None and masked_ce_loss_scale is not None

        # mask is True, if original value should be kept
        phon_mask = get_random_mask(phon_indices_lens.to(phon_indices.device), **label_masking_opts)
        phon_indices_masked, phon_indices_masked_lens = mask_sequence(
            phon_indices, phon_indices_lens.to(phon_indices.device), phon_mask, mask_value=model.mask_idx)

        encoder_output, encoder_lens, _ = model.forward_text(phon_indices_masked, phon_indices_masked_lens)

        input_labels = F.pad(phon_indices, (1, 0), "constant", value=model.bos_idx)
        input_labels_len = phon_indices_lens + 1
        logits = model.decode_text_seq(
            input_labels, input_labels_len.to(device=input_labels.device), encoder_output, encoder_lens
        )
        logits_packed = pack_padded_sequence(logits, input_labels_len, batch_first=True, enforce_sorted=False)

        single_seqs = unpad_sequence(phon_indices, phon_indices_lens, batch_first=True)
        eos_tensor = torch.tensor(num_eos_symbols * [model.eos_idx], device=phon_indices.device, dtype=torch.int32)
        targets_w_eos_packed = pack_sequence(
            [torch.concat((seq, eos_tensor), dim=-1) for seq in single_seqs],
            enforce_sorted=False,
        ).data
        single_masks = unpad_sequence(phon_mask, phon_indices_lens, batch_first=True)
        phon_mask_packed = pack_sequence(
            # append True for each EOS symbol
            [torch.concat((mask, torch.ones(num_eos_symbols, dtype=torch.bool, device=phon_indices.device)), dim=-1) for
             mask in single_masks],
            enforce_sorted=False,
        ).data

        ce_loss = F.cross_entropy(
            logits_packed.data,
            targets_w_eos_packed.long(),
            label_smoothing=label_smoothing if ctx.epoch >= label_smoothing_start_epoch else 0.0,
            reduction="none",
        )

        num_masked_rf = phon_indices_.dims[1].get_size_tensor().copy()
        num_masked_rf.raw_tensor = (~phon_mask).sum(dim=1).to(num_masked_rf.raw_tensor.dtype)

        # loss is only calculated on masked positions
        ctx.mark_as_loss(
            # mask=False corresponds to masked positions in the original sequence
            ce_loss[~phon_mask_packed],
            "masked_ce",
            scale=masked_ce_loss_scale,
            custom_inv_norm_factor=num_masked_rf,
            use_normalized_loss=True,
            # as_error=True
        )

        ctx.mark_as_loss(
            ce_loss,
            "ce",
            scale=ce_loss_scale,
            custom_inv_norm_factor=phon_indices_.dims[1].get_size_tensor() + num_eos_symbols,
            use_normalized_loss=True,
            # as_error=True
        )
        error = torch.argmax(logits_packed.data, dim=-1).not_equal(targets_w_eos_packed)
        ctx.mark_as_loss(error, "fer", as_error=True)
        ctx.mark_as_loss(error[~phon_mask_packed], "masked_fer", as_error=True)

    if audio_features is not None and audio_features_lens.size(0) != 0:
        assert mse_loss_scale is not None and masked_mse_loss_scale is not None and eos_ce_loss_scale is not None

        audio_mask = get_random_mask(audio_features_lens.to(audio_features.device), **feature_masking_opts)
        mask_embedding = model.embedding(
            torch.full(audio_features.shape[:2], model.mask_idx, device=audio_features.device))
        audio_features_masked, audio_features_masked_lens = mask_sequence(
            audio_features, audio_features_lens.to(audio_features.device), audio_mask, mask_value=mask_embedding)

        # AUDIO STEP
        encoder_output, encoder_lens, _ = model.forward_features(audio_features_masked, audio_features_masked_lens)

        bos_tensor = torch.tensor([model.bos_idx], device=audio_features.device, dtype=torch.int32)
        bos_embeddings = model.embedding(bos_tensor)
        input_features = torch.cat(
            [bos_embeddings.unsqueeze(0).expand(audio_features.size(0), -1, -1), audio_features],
            dim=1,
        )

        input_features_len = audio_features_lens + 1
        out_features, out_eos_logits = model.decode_audio_seq(
            input_features, input_features_len.to(device=input_features.device), encoder_output, encoder_lens
        )

        # use `audio_features_lens` here, because there is no eos symbol for audio features
        out_features_packed = pack_padded_sequence(out_features, audio_features_lens, batch_first=True,
                                                   enforce_sorted=False)
        out_eos_logits_packed = pack_padded_sequence(out_eos_logits, audio_features_lens, batch_first=True,
                                                     enforce_sorted=False)

        is_eos = torch.arange(out_eos_logits.size(1), device=audio_features.device).unsqueeze(0) == (
                    audio_features_lens.unsqueeze(1).to(audio_features.device) - 1)
        eos_targets = torch.where(
            is_eos,
            torch.ones_like(out_eos_logits),
            torch.zeros_like(out_eos_logits),
        )
        single_eos_seqs = unpad_sequence(eos_targets, audio_features_lens, batch_first=True)
        eos_targets_packed = pack_sequence(single_eos_seqs, enforce_sorted=False).data
        eos_loss = F.binary_cross_entropy_with_logits(
            out_eos_logits_packed.data,
            eos_targets_packed.data,
            reduction="none",
        )

        single_seqs = unpad_sequence(audio_features, audio_features_lens, batch_first=True)
        targets_packed = pack_sequence(single_seqs, enforce_sorted=False).data

        single_masks = unpad_sequence(audio_mask, audio_features_lens, batch_first=True)
        audio_mask_packed = pack_sequence(single_masks, enforce_sorted=False).data

        if audio_loss == "mse":
            mse_loss = F.mse_loss(
                out_features_packed.data,
                targets_packed,
                reduction="none",
            )
        else:
            assert audio_loss == "l1"
            mse_loss = F.l1_loss(
                out_features_packed.data,
                targets_packed,
                reduction="none",
            )

        num_masked_rf = audio_features_.dims[1].get_size_tensor().copy()
        num_masked_rf.raw_tensor = (~audio_mask).sum(dim=1).to(num_masked_rf.raw_tensor.dtype)

        ctx.mark_as_loss(
            # mask=False corresponds to masked positions in the original sequence
            mse_loss[~audio_mask_packed],
            f"masked_{audio_loss}",
            scale=masked_mse_loss_scale,
            custom_inv_norm_factor=num_masked_rf,
            use_normalized_loss=True,
            # as_error=True
        )
        ctx.mark_as_loss(
            mse_loss,
            audio_loss,
            scale=mse_loss_scale,
            custom_inv_norm_factor=audio_features_.dims[1].get_size_tensor(),
            use_normalized_loss=True,
            # as_error=True,
        )
        ctx.mark_as_loss(
            eos_loss,
            "eos_ce",
            scale=eos_ce_loss_scale,
            custom_inv_norm_factor=audio_features_.dims[1].get_size_tensor(),
            use_normalized_loss=True,
        )
