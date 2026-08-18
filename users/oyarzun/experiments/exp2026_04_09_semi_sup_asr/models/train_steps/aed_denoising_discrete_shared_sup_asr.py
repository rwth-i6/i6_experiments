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
    def decode_audio_seq(
        self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor
    ) -> Tensor:
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
    text_aux_loss_scales: Optional[Sequence[float]] = None,
    audio_aux_loss_scales: Optional[Sequence[float]] = None,
    supervised_asr_ce_loss_scale: float = 1.0,
    adv_loss_scale: float = 0.0,
    codebook_diversity_loss_scale: float = 0.0,  
    denoise_pretrain_steps: int = 0,
    denoise_pretrain_epochs: int = 0,
    pretrain_codebook_prob: Optional[float] = None,
    pretrain_codebook_diversity_loss_scale: Optional[float] = None,
    pretrain_adv_loss_scale: Optional[float] = None,
    gradual_unfreeze: bool = False,
    gradual_unfreeze_proportion: float = 0.8,
    gradual_unfreeze_start_iter: int = 0,
    gradual_unfreeze_end_iter: int = 0,
    asr_loss_warmup_steps: int = 0,
    use_lm_for_asr_adv: bool = False,
    **_kwargs,
):
    assert {"data", "target", "seq_tag"}.issubset(extern_data.data.keys())
    if "data" in extern_data:
        audio_indices_: ReturnnTensor = extern_data["data"]
    else:
        audio_indices_ = None
    if "target" in extern_data:
        phon_indices_: ReturnnTensor = extern_data["target"]
    else:
        phon_indices_ = None

    ctx = rf.get_run_ctx()
    # Check if we are in the MLM pretraining phase where backtranslation is skipped
    if denoise_pretrain_epochs > 0:
        is_pretraining = ctx.epoch <= denoise_pretrain_epochs
    else:
        is_pretraining = ctx.step < denoise_pretrain_steps
        
    if not is_pretraining and not hasattr(model, "_asr_start_step"):
        model._asr_start_step = ctx.step
        print(f"========== PRETRAINING OVER, TRANSLATION STARTED at global step {ctx.step} (epoch {ctx.epoch}) ==========", flush=True)
    
    if gradual_unfreeze and not is_pretraining:
        bt_step = ctx.step - model._asr_start_step
        encoder_obj = getattr(model.encoder, "encoder", model.encoder)
        if hasattr(encoder_obj, "module_list"):
            num_layers = len(encoder_obj.module_list)
            num_frozen_layers_initial = int(num_layers * gradual_unfreeze_proportion)
            
            if bt_step < gradual_unfreeze_start_iter:
                num_frozen_layers = num_frozen_layers_initial
            elif bt_step >= gradual_unfreeze_end_iter:
                num_frozen_layers = 0
            else:
                ratio = (bt_step - gradual_unfreeze_start_iter) / (gradual_unfreeze_end_iter - gradual_unfreeze_start_iter)
                num_frozen_layers = int(num_frozen_layers_initial * (1.0 - ratio))
                
            for i, layer in enumerate(encoder_obj.module_list):
                for param in layer.parameters():
                    param.requires_grad = (i >= num_frozen_layers)
            
            if hasattr(encoder_obj, "frontend") and encoder_obj.frontend is not None:
                for param in encoder_obj.frontend.parameters():
                    param.requires_grad = (num_frozen_layers == 0)

    if is_pretraining and pretrain_codebook_prob is not None:
        if hasattr(model, "codebook_prob"):
            if getattr(model, "_orig_codebook_prob", None) is None:
                model._orig_codebook_prob = model.codebook_prob
            model.codebook_prob = pretrain_codebook_prob
    elif getattr(model, "_orig_codebook_prob", None) is not None:
        model.codebook_prob = model._orig_codebook_prob
        model._orig_codebook_prob = None

    if is_pretraining and pretrain_codebook_diversity_loss_scale is not None:
        codebook_diversity_loss_scale = pretrain_codebook_diversity_loss_scale

    if is_pretraining and pretrain_adv_loss_scale is not None:
        adv_loss_scale = pretrain_adv_loss_scale

    if is_pretraining and (text_ce_loss_scale > 0.0 or text_masked_ce_loss_scale > 0.0):
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
            codebook_diversity_loss_scale=codebook_diversity_loss_scale,  
            loss_name="text",
            adv_loss_scale=adv_loss_scale,
            true_adv_target=1,  # real text
        )

    if is_pretraining and (audio_ce_loss_scale > 0.0 or audio_masked_ce_loss_scale > 0.0):
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
            codebook_diversity_loss_scale=codebook_diversity_loss_scale,  
            loss_name="audio",
            adv_loss_scale=adv_loss_scale,
            true_adv_target=0,  # real audio
        )

    if not is_pretraining:
        asr_step = ctx.step - model._asr_start_step
        
        # Lower initial ASR LR: scale the loss down during warmup
        if asr_loss_warmup_steps > 0:
            if asr_step < asr_loss_warmup_steps:
                warmup_factor = max(0.0, float(asr_step) / float(asr_loss_warmup_steps))
                supervised_asr_ce_loss_scale *= warmup_factor

        if supervised_asr_ce_loss_scale > 0.0:
            model.decode_seq = model.decode_text_seq
            model.forward = model.forward_audio
            model.mask_idx = None
            model.bos_idx = model.text_bos_idx
            model.eos_idx = model.text_eos_idx
            model.decoder = model.text_decoder
            aed_denoising_discrete.train_step(
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
                adv_loss_scale=adv_loss_scale if use_lm_for_asr_adv else 0.0,
                true_adv_target=0 if use_lm_for_asr_adv else None,
            )

        if use_lm_for_asr_adv and adv_loss_scale > 0.0 and "lm_text" in extern_data:
            lm_text_indices_ = extern_data["lm_text"]
            model.decode_seq = model.decode_text_seq
            model.forward = model.forward_text
            model.mask_idx = model.text_mask_idx
            model.bos_idx = model.text_bos_idx
            model.eos_idx = model.text_eos_idx
            model.decoder = model.text_decoder
            aed_denoising_discrete.train_step(
                model=model,
                extern_data=TensorDict({"data": lm_text_indices_, "seq_tag": extern_data["seq_tag"]}),
                ce_loss_scale=0.0,
                masked_ce_loss_scale=0.0,
                label_smoothing=label_smoothing,
                label_smoothing_start_epoch=label_smoothing_start_epoch,
                masking_opts={"mask_prob": 0.0},
                aux_loss_scales=None,
                codebook_diversity_loss_scale=codebook_diversity_loss_scale,  
                loss_name="lm_text",
                adv_loss_scale=adv_loss_scale,
                true_adv_target=1,  # real text
            )

    dummy_tensor = next((p for p in model.parameters() if p.requires_grad), next(model.parameters(), None))
    if dummy_tensor is not None:
        dummy_loss = (dummy_tensor * 0.0).sum()
        ctx.mark_as_loss(dummy_loss, "dummy_accum_fallback", scale=1.0)
        
    # Mark the global step as a metric so it appears in the training logs
    step_tensor = torch.tensor(ctx.step, dtype=torch.float32, device=dummy_tensor.device if dummy_tensor is not None else "cpu")
    ctx.mark_as_loss(step_tensor, "global_step", scale=0.0)

