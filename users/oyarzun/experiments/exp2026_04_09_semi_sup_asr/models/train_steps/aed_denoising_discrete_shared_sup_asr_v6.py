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

    @abstractmethod
    def forward_text(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def forward_audio(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def decode_text_seq(
        self,
        indices: Tensor,
        seq_lens: Tensor,
        encoder_output: Tensor,
        encoder_output_mask: Tensor,
    ) -> Tensor:
        pass

    @abstractmethod
    def decode_audio_seq(
        self,
        indices: Tensor,
        seq_lens: Tensor,
        encoder_output: Tensor,
        encoder_output_mask: Tensor,
    ) -> Tensor:
        pass


def _single_modality_step(
    *,
    model: SharedDenoisingAedModel,
    extern_data: TensorDict,
    ce_loss_scale: float = 1.0,
    masked_ce_loss_scale: float = 0.0,
    label_smoothing: float = 0.0,
    label_smoothing_start_epoch: int = 0,
    masking_opts: Optional[Dict] = None,
    aux_loss_scales: Optional[Sequence[float]] = None,
    codebook_diversity_loss_scale: float = 0.0,
    loss_name: str = "",
):
    ctx = rf.get_run_ctx()
    loss_suffix = f"_{loss_name}" if loss_name else ""

    label_indices_: ReturnnTensor = extern_data["data"]
    label_indices = label_indices_.raw_tensor
    label_indices_lens = label_indices_.dims[1].dyn_size_ext.raw_tensor
    valid_mask = label_indices_lens != 0
    num_eos_symbols = 1

    if not torch.all(valid_mask).item():
        label_indices = label_indices[valid_mask.to(label_indices.device)]
        label_indices_lens = label_indices_lens[valid_mask.to(label_indices_lens.device)]

    if "target" in extern_data:
        target_indices_: ReturnnTensor = extern_data["target"]
        _target_indices = target_indices_.raw_tensor
        _target_lens = target_indices_.dims[1].dyn_size_ext.raw_tensor
        target_indices: Tensor = _target_indices[valid_mask.to(_target_indices.device)]
        target_indices_lens: Tensor = _target_lens[valid_mask.to(_target_lens.device)]
    else:
        target_indices_ = label_indices_
        target_indices = label_indices
        target_indices_lens = label_indices_lens

    # Masking for denoising
    if not masking_opts or masking_opts.get("mask_prob", 0.0) == 0.0:
        label_indices_masked = label_indices
        label_indices_masked_lens = label_indices_lens.to(label_indices.device)
        B = label_indices_masked_lens.size(0)
        T = label_indices_masked_lens.max().item()
        phon_mask = torch.ones(B, T, device=label_indices.device).bool()
    else:
        if masking_opts.get("mask", None) is not None:
            phon_mask = masking_opts["mask"]
        else:
            phon_mask = get_random_mask(label_indices_lens.to(label_indices.device), **masking_opts)
        label_indices_masked, label_indices_masked_lens = mask_sequence(
            label_indices, label_indices_lens.to(label_indices.device), phon_mask, mask_value=model.mask_idx
        )

    encoder_output, aux_logits, encoder_lens, _ = model.forward(label_indices_masked, label_indices_masked_lens)

    # Codebook diversity loss and codebook perplexity logging
    if getattr(model, "quantizer_out", None) is not None:
        q_out = model.quantizer_out
        num_vars = q_out.get("num_vars", 1)
        prob_ppl = q_out.get("prob_perplexity", None)
        code_ppl = q_out.get("code_perplexity", None)

        if codebook_diversity_loss_scale > 0.0 and prob_ppl is not None:
            diversity_loss = (num_vars - prob_ppl) / num_vars
            ctx.mark_as_loss(
                diversity_loss, f"codebook_diversity{loss_suffix}", dims=[], scale=codebook_diversity_loss_scale
            )

        if ctx.stage == "train_step":
            if prob_ppl is not None:
                ctx.mark_as_loss(prob_ppl, f"codebook_prob_ppl{loss_suffix}", dims=[], as_error=True)
            if code_ppl is not None:
                ctx.mark_as_loss(code_ppl, f"codebook_hard_ppl{loss_suffix}", dims=[], as_error=True)

    # Reconstruction / ASR loss
    if ce_loss_scale > 0.0 or masked_ce_loss_scale > 0.0:
        input_labels = F.pad(target_indices, (1, 0), "constant", value=model.bos_idx)
        input_labels_len = target_indices_lens + 1
        logits = model.decode_seq(
            input_labels, input_labels_len.to(device=input_labels.device), encoder_output, encoder_lens
        )
        logits_packed = pack_padded_sequence(logits, input_labels_len, batch_first=True, enforce_sorted=False)

        single_seqs = unpad_sequence(target_indices, target_indices_lens, batch_first=True)
        eos_tensor = torch.tensor(num_eos_symbols * [model.eos_idx], device=label_indices.device, dtype=torch.int32)
        targets_w_eos_packed = pack_sequence(
            [torch.concat((seq, eos_tensor), dim=-1) for seq in single_seqs],
            enforce_sorted=False,
        ).data
        single_masks = unpad_sequence(phon_mask, label_indices_lens, batch_first=True)
        phon_mask_packed = pack_sequence(
            [
                torch.concat((mask, torch.ones(num_eos_symbols, dtype=torch.bool, device=label_indices.device)), dim=-1)
                for mask in single_masks
            ],
            enforce_sorted=False,
        ).data

        ce_loss = F.cross_entropy(
            logits_packed.data,
            targets_w_eos_packed.long(),
            label_smoothing=label_smoothing if ctx.epoch >= label_smoothing_start_epoch else 0.0,
            reduction="none",
        )

        num_masked_rf = label_indices_.dims[1].get_size_tensor().copy()
        if label_indices.size(0) < num_masked_rf.raw_tensor.size(0):
            original_lens = label_indices_.dims[1].dyn_size_ext.raw_tensor
            num_masked_rf_raw = torch.zeros_like(num_masked_rf.raw_tensor)
            num_masked_rf_raw[original_lens > 0] = torch.clamp((~phon_mask).sum(dim=1), min=1).to(
                device=num_masked_rf.raw_tensor.device, dtype=num_masked_rf.raw_tensor.dtype
            )
            num_masked_rf.raw_tensor = num_masked_rf_raw
        else:
            num_masked_rf.raw_tensor = torch.clamp((~phon_mask).sum(dim=1), min=1).to(
                device=num_masked_rf.raw_tensor.device, dtype=num_masked_rf.raw_tensor.dtype
            )

        error = torch.argmax(logits_packed.data, dim=-1).not_equal(targets_w_eos_packed)

        if masked_ce_loss_scale > 0.0:
            if ctx.stage == "train_step":
                ctx.mark_as_loss(
                    ce_loss[~phon_mask_packed],
                    f"masked_ce{loss_suffix}",
                    scale=masked_ce_loss_scale,
                    custom_inv_norm_factor=num_masked_rf,
                    use_normalized_loss=True,
                )
                ctx.mark_as_loss(error[~phon_mask_packed], f"masked_fer{loss_suffix}", as_error=True)

        if ctx.stage == "train_step":
            ctx.mark_as_loss(
                ce_loss,
                f"ce{loss_suffix}",
                scale=ce_loss_scale,
                custom_inv_norm_factor=label_indices_.dims[1].get_size_tensor() + num_eos_symbols,
                use_normalized_loss=True,
            )
            ctx.mark_as_loss(error, f"fer{loss_suffix}", as_error=True)

    # CTC Auxiliary Losses
    if aux_loss_scales and len(aux_loss_scales) > 0 and not all(scale == 0 for scale in aux_loss_scales):
        assert len(aux_loss_scales) == len(aux_logits)
        target_lens_ = target_indices_lens.to(device=target_indices.device)
        for i, (aux_logits_layer_i, scale) in enumerate(zip(aux_logits, aux_loss_scales)):
            aux_log_probs = F.log_softmax(aux_logits_layer_i, dim=-1)
            aux_loss = F.ctc_loss(
                log_probs=aux_log_probs.transpose(0, 1).to(torch.float32),
                targets=target_indices,
                input_lengths=encoder_lens,
                target_lengths=target_lens_,
                blank=model.blank_idx,
                reduction="none",
                zero_infinity=True,
            )
            ctx.mark_as_loss(
                aux_loss,
                name=f"ctc-{i}{loss_suffix}",
                custom_inv_norm_factor=target_indices_.dims[1].get_size_tensor(),
                scale=scale,
                use_normalized_loss=True,
            )


def train_step(
    *,
    model: SharedDenoisingAedModel,
    extern_data: TensorDict,
    audio_ce_loss_scale: Optional[float] = 0.2,
    audio_masked_ce_loss_scale: Optional[float] = 1.0,
    text_ce_loss_scale: Optional[float] = 0.0,
    text_masked_ce_loss_scale: Optional[float] = 0.0,
    label_smoothing: float = 0.0,
    label_smoothing_start_epoch: int = 0,
    text_masking_opts: Optional[Dict] = None,
    audio_masking_opts: Optional[Dict] = None,
    supervised_asr_ce_loss_scale: float = 1.0,
    codebook_diversity_loss_scale: float = 0.0,
    denoise_pretrain_epochs: int = 0,
    pretrain_codebook_prob: Optional[float] = None,
    pretrain_codebook_diversity_loss_scale: Optional[float] = None,
    asr_loss_warmup_steps: int = 0,
    **_kwargs,
):
    assert {"data", "target", "seq_tag"}.issubset(extern_data.data.keys())
    audio_indices_: ReturnnTensor = extern_data["data"]
    phon_indices_: ReturnnTensor = extern_data["target"]

    ctx = rf.get_run_ctx()

    if not hasattr(model, "_asr_start_step"):
        model._asr_start_step = None

    is_pretraining = ctx.epoch <= denoise_pretrain_epochs

    if not is_pretraining and model._asr_start_step is None:
        model._asr_start_step = ctx.step

    if is_pretraining and pretrain_codebook_prob is not None:
        if getattr(model, "_orig_codebook_prob", None) is None:
            model._orig_codebook_prob = model.codebook_prob
        model.codebook_prob = pretrain_codebook_prob
    elif getattr(model, "_orig_codebook_prob", None) is not None:
        model.codebook_prob = model._orig_codebook_prob
        model._orig_codebook_prob = None

    if is_pretraining and pretrain_codebook_diversity_loss_scale is not None:
        codebook_diversity_loss_scale = pretrain_codebook_diversity_loss_scale

    # 1. Pretraining: Audio Denoising Reconstruction
    if is_pretraining and (audio_ce_loss_scale > 0.0 or audio_masked_ce_loss_scale > 0.0):
        model.decode_seq = model.decode_audio_seq
        model.forward = model.forward_audio
        model.mask_idx = model.audio_mask_idx
        model.bos_idx = model.audio_bos_idx
        model.eos_idx = model.audio_eos_idx
        model.decoder = model.audio_decoder
        _single_modality_step(
            model=model,
            extern_data=TensorDict({"data": audio_indices_, "seq_tag": extern_data["seq_tag"]}),
            ce_loss_scale=audio_ce_loss_scale,
            masked_ce_loss_scale=audio_masked_ce_loss_scale,
            label_smoothing=label_smoothing,
            label_smoothing_start_epoch=label_smoothing_start_epoch,
            masking_opts=audio_masking_opts,
            aux_loss_scales=None,
            codebook_diversity_loss_scale=codebook_diversity_loss_scale,
            loss_name="audio",
        )

    # 2. Supervised ASR Training
    if not is_pretraining:
        asr_step = ctx.step - model._asr_start_step

        if asr_loss_warmup_steps > 0 and asr_step < asr_loss_warmup_steps:
            warmup_factor = max(0.0, float(asr_step) / float(asr_loss_warmup_steps))
            supervised_asr_ce_loss_scale *= warmup_factor

        if supervised_asr_ce_loss_scale > 0.0:
            model.decode_seq = model.decode_text_seq
            model.forward = model.forward_audio
            model.mask_idx = None
            model.bos_idx = model.text_bos_idx
            model.eos_idx = model.text_eos_idx
            model.decoder = model.text_decoder
            _single_modality_step(
                model=model,
                extern_data=extern_data,
                ce_loss_scale=supervised_asr_ce_loss_scale,
                masked_ce_loss_scale=0.0,
                label_smoothing=label_smoothing,
                label_smoothing_start_epoch=label_smoothing_start_epoch,
                masking_opts={"mask_prob": 0.0},
                aux_loss_scales=None,
                codebook_diversity_loss_scale=codebook_diversity_loss_scale,
                loss_name="sup_asr",
            )
