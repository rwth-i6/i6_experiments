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


def greedy_decode_discrete(
    *,
    model: SharedDenoisingAedModel,
    decoder_state,
    bos_idx: int,
    eos_idx: int,
    max_seq_len: Tensor,
    step_decoder_func: Callable,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    Highly optimized greedy decoder (beam_size=1) for generating backtranslations.
    """
    batch_size = max_seq_len.size(0)
    # Use shape (B, 1, 1) to match the expected (B, beam, T) input format of the decoder,
    # which avoids triggering unintentional batch dimension broadcasting in cross attention.
    target = torch.full([batch_size, 1, 1], bos_idx, dtype=torch.int32, device=device)
    ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    seq_targets = []
    max_steps = int(max_seq_len.max().item())
    
    for step in range(max_steps):
        logits, decoder_state = step_decoder_func(target, decoder_state)
        # Squeeze the two dimensions of size 1 corresponding to beam and time steps
        logits = logits.squeeze(-2).squeeze(-2) # B, Vocab
        
        next_token = logits.argmax(dim=-1) # B
        next_token = torch.where(ended, eos_idx, next_token)
        
        ended = ended | (next_token == eos_idx)
        ended = ended | (step >= max_seq_len)
        
        if ended.all():
            break
            
        lengths = lengths + (~ended).int()
        seq_targets.append(next_token)
        target = next_token.unsqueeze(-1).unsqueeze(-1).int()
        
    if not seq_targets:
        return torch.full([batch_size, 1], eos_idx, dtype=torch.int32, device=device), torch.zeros(batch_size, dtype=torch.int32, device=device)
        
    seq_targets = torch.stack(seq_targets, dim=1) # [B, T]
    return seq_targets, lengths


def generate_pseudo_labels_for_batch(
    model: SharedDenoisingAedModel,
    src_indices_: ReturnnTensor,
    forward_src: Callable,
    target_decoder: nn.Module,
    target_bos_idx: int,
    target_eos_idx: int,
    target_out_dim: int,
    step_decoder_func: Callable,
) -> Optional[ReturnnTensor]:
    src_indices: Tensor = src_indices_.raw_tensor
    src_indices_lens: Tensor = src_indices_.dims[1].dyn_size_ext.raw_tensor
    if torch.any(src_indices_lens != 0).item():
        src_indices = src_indices[src_indices_lens > 0]
        src_indices_lens = src_indices_lens[src_indices_lens > 0]

        with torch.no_grad():
            seq_len = src_indices_lens.to(src_indices.device)
            target_decoder_state = model.forward_encoder(
                src_indices, seq_len, decoder=target_decoder, forward_func=forward_src
            )
            max_seq_len = (seq_len * 1.5).int()
            pseudo_target_indices, pseudo_target_indices_lens = greedy_decode_discrete(
                model=model,
                decoder_state=target_decoder_state,
                device=src_indices.device,
                max_seq_len=max_seq_len,
                step_decoder_func=step_decoder_func,
                bos_idx=target_bos_idx,
                eos_idx=target_eos_idx,
            )
            pseudo_target_indices = pseudo_target_indices[:, :-1]
            
            batch_dim = ReturnnDim(pseudo_target_indices.shape[0], name="batch")
            target_vocab_dim = ReturnnDim(target_out_dim, name="vocab")
            pseudo_target_lens_data = rf.convert_to_tensor(pseudo_target_indices_lens, dims=[batch_dim])
            pseudo_target_lens_dim = ReturnnDim(pseudo_target_lens_data, name="seq_len")
            pseudo_target_indices_ = rf.convert_to_tensor(
                pseudo_target_indices, dims=[batch_dim, pseudo_target_lens_dim], sparse_dim=target_vocab_dim
            )
            return pseudo_target_indices_
    return None

def compute_bt_loss_for_batch(
    model: SharedDenoisingAedModel,
    extern_data: TensorDict,
    ce_loss_scale: float,
    label_smoothing: float,
    label_smoothing_start_epoch: int,
    aux_loss_scales: Optional[Sequence[float]],
    codebook_diversity_loss_scale: float,
    pseudo_target_indices_: Optional[ReturnnTensor],
    src_indices_: ReturnnTensor,
    forward_target: Callable,
    src_decoder: nn.Module,
    decode_src_seq: Callable,
    src_bos_idx: int,
    src_eos_idx: int,
    src_blank_idx: int,
    src_out_dim: int,
    loss_name: str,
):
    if pseudo_target_indices_ is None:
        return
        
    src_indices = src_indices_.raw_tensor
    src_indices_lens = src_indices_.dims[1].dyn_size_ext.raw_tensor
    if not torch.any(src_indices_lens != 0).item():
        return
        
    src_indices = src_indices[src_indices_lens > 0]
    src_indices_lens = src_indices_lens[src_indices_lens > 0]
    batch_dim = pseudo_target_indices_.dims[0]
    src_vocab_dim = ReturnnDim(src_out_dim, name="vocab")
    src_indices_lens_data = rf.convert_to_tensor(src_indices_lens, dims=[batch_dim])
    src_indices_lens_dim = ReturnnDim(src_indices_lens_data, name="seq_len")
    src_indices_filtered_ = rf.convert_to_tensor(
        src_indices, dims=[batch_dim, src_indices_lens_dim], sparse_dim=src_vocab_dim
    )
    
    model.decode_seq = decode_src_seq
    model.forward = forward_target
    model.mask_idx = None
    model.bos_idx = src_bos_idx
    model.eos_idx = src_eos_idx
    model.blank_idx = src_blank_idx
    model.decoder = src_decoder
    aed_denoising_discrete.train_step(
        model=model,
        extern_data=TensorDict(
            {"data": pseudo_target_indices_, "target": src_indices_filtered_, "seq_tag": extern_data["seq_tag"]}
        ),
        ce_loss_scale=ce_loss_scale,
        masked_ce_loss_scale=0.0,
        label_smoothing=label_smoothing,
        label_smoothing_start_epoch=label_smoothing_start_epoch,
        masking_opts={"mask_prob": 0.0, "min_span": 0, "max_span": 0},
        aux_loss_scales=aux_loss_scales,
        codebook_diversity_loss_scale=codebook_diversity_loss_scale,  
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
    codebook_diversity_loss_scale: float = 0.0,  
    denoise_pretrain_steps: int = 0,
    pretrain_codebook_prob: Optional[float] = None,
    pretrain_codebook_diversity_loss_scale: Optional[float] = None,
    pretrain_adv_loss_scale: Optional[float] = None,
    gradual_unfreeze: bool = False,
    gradual_unfreeze_proportion: float = 0.8,
    gradual_unfreeze_start_iter: int = 0,
    gradual_unfreeze_end_iter: int = 0,
    bt_buffer_size_steps: int = 10,
    bt_train_iterations: int = 10,
    **_kwargs,
):
    assert set(extern_data.data.keys()) == {"data", "target", "seq_tag"}
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
    is_pretraining = ctx.step < denoise_pretrain_steps
    if not is_pretraining and (ctx.step - denoise_pretrain_steps) == 0:
        print(f"========== PRETRAINING OVER, TRANSLATION STARTED at global step {ctx.step} ==========", flush=True)
    
    if not hasattr(model, "_bt_phon_sparse_dim") and phon_indices_ is not None:
        model._bt_phon_sparse_dim = phon_indices_.sparse_dim
    if not hasattr(model, "_bt_audio_sparse_dim") and audio_indices_ is not None:
        model._bt_audio_sparse_dim = audio_indices_.sparse_dim

    if gradual_unfreeze and not is_pretraining:
        bt_step = ctx.step - denoise_pretrain_steps
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
    else:
        import random
        if not hasattr(model, "_bt_v2_state"):
            model._bt_v2_state = "ACCUMULATE"
            model._bt_v2_buffer = []
            model._bt_v2_accum_steps = 0
            model._bt_v2_train_steps = 0
            
        if model._bt_v2_state == "ACCUMULATE":
            pseudo_audio_from_phon = None
            if phon_indices_ is not None:
                pseudo_audio_from_phon = generate_pseudo_labels_for_batch(
                    model=model,
                    src_indices_=phon_indices_,
                    forward_src=model.forward_text,
                    target_decoder=model.audio_decoder,
                    target_bos_idx=model.audio_bos_idx,
                    target_eos_idx=model.audio_eos_idx,
                    target_out_dim=model.audio_out_dim,
                    step_decoder_func=model.step_audio_decoder,
                )
            pseudo_text_from_audio = None
            if audio_indices_ is not None:
                pseudo_text_from_audio = generate_pseudo_labels_for_batch(
                    model=model,
                    src_indices_=audio_indices_,
                    forward_src=model.forward_audio,
                    target_decoder=model.text_decoder,
                    target_bos_idx=model.text_bos_idx,
                    target_eos_idx=model.text_eos_idx,
                    target_out_dim=model.text_out_dim,
                    step_decoder_func=model.step_text_decoder,
                )

            b_dict = {
                "seq_tag": extern_data.data["seq_tag"] if "seq_tag" in extern_data.data else None,
            }
            if phon_indices_ is not None:
                b_dict["phon_indices"] = phon_indices_.raw_tensor.detach().clone()
                b_dict["phon_lens"] = phon_indices_.dims[1].dyn_size_ext.raw_tensor.detach().clone()
            if audio_indices_ is not None:
                b_dict["audio_indices"] = audio_indices_.raw_tensor.detach().clone()
                b_dict["audio_lens"] = audio_indices_.dims[1].dyn_size_ext.raw_tensor.detach().clone()
            
            if pseudo_audio_from_phon is not None:
                b_dict["pseudo_audio_from_phon"] = pseudo_audio_from_phon.raw_tensor.detach().clone()
                b_dict["pseudo_audio_lens"] = pseudo_audio_from_phon.dims[1].dyn_size_ext.raw_tensor.detach().clone()
            
            if pseudo_text_from_audio is not None:
                b_dict["pseudo_text_from_audio"] = pseudo_text_from_audio.raw_tensor.detach().clone()
                b_dict["pseudo_text_lens"] = pseudo_text_from_audio.dims[1].dyn_size_ext.raw_tensor.detach().clone()

            model._bt_v2_buffer.append(b_dict)
            model._bt_v2_accum_steps += 1
            
            if model._bt_v2_accum_steps >= bt_buffer_size_steps:
                model._bt_v2_state = "TRAIN"
                model._bt_v2_train_steps = 0
                
            
        elif model._bt_v2_state == "TRAIN":
            batch = random.choice(model._bt_v2_buffer)
            b_extern_data = {"seq_tag": batch["seq_tag"]} if batch.get("seq_tag") is not None else {}
            
            b_phon_indices_ = None
            if "phon_indices" in batch:
                batch_dim_p = ReturnnDim(batch["phon_indices"].shape[0], name="batch")
                time_dim_p = ReturnnDim(batch["phon_indices"].shape[1], name="time")
                phon_lens_data = rf.convert_to_tensor(batch["phon_lens"], dims=[batch_dim_p])
                time_dim_p.dyn_size_ext = phon_lens_data
                b_phon_indices_ = rf.convert_to_tensor(batch["phon_indices"], dims=[batch_dim_p, time_dim_p], sparse_dim=getattr(model, "_bt_phon_sparse_dim", None))
                
            b_audio_indices_ = None
            if "audio_indices" in batch:
                batch_dim_a = ReturnnDim(batch["audio_indices"].shape[0], name="batch")
                time_dim_a = ReturnnDim(batch["audio_indices"].shape[1], name="time")
                audio_lens_data = rf.convert_to_tensor(batch["audio_lens"], dims=[batch_dim_a])
                time_dim_a.dyn_size_ext = audio_lens_data
                b_audio_indices_ = rf.convert_to_tensor(batch["audio_indices"], dims=[batch_dim_a, time_dim_a], sparse_dim=getattr(model, "_bt_audio_sparse_dim", None))
                
            b_pseudo_audio_from_phon_ = None
            if batch.get("pseudo_audio_from_phon") is not None:
                batch_dim_pa = ReturnnDim(batch["pseudo_audio_from_phon"].shape[0], name="batch")
                time_dim_pa = ReturnnDim(batch["pseudo_audio_from_phon"].shape[1], name="time")
                pa_lens_data = rf.convert_to_tensor(batch["pseudo_audio_lens"], dims=[batch_dim_pa])
                time_dim_pa.dyn_size_ext = pa_lens_data
                b_pseudo_audio_from_phon_ = rf.convert_to_tensor(batch["pseudo_audio_from_phon"], dims=[batch_dim_pa, time_dim_pa], sparse_dim=ReturnnDim(model.audio_out_dim, name="vocab"))
                
            b_pseudo_text_from_audio_ = None
            if batch.get("pseudo_text_from_audio") is not None:
                batch_dim_pt = ReturnnDim(batch["pseudo_text_from_audio"].shape[0], name="batch")
                time_dim_pt = ReturnnDim(batch["pseudo_text_from_audio"].shape[1], name="time")
                pt_lens_data = rf.convert_to_tensor(batch["pseudo_text_lens"], dims=[batch_dim_pt])
                time_dim_pt.dyn_size_ext = pt_lens_data
                b_pseudo_text_from_audio_ = rf.convert_to_tensor(batch["pseudo_text_from_audio"], dims=[batch_dim_pt, time_dim_pt], sparse_dim=ReturnnDim(model.text_out_dim, name="vocab"))
            
            if text_ce_loss_scale > 0.0 or text_masked_ce_loss_scale > 0.0:
                model.decode_seq = model.decode_text_seq
                model.forward = model.forward_text
                model.mask_idx = model.text_mask_idx
                model.bos_idx = model.text_bos_idx
                model.eos_idx = model.text_eos_idx
                model.decoder = model.text_decoder
                aed_denoising_discrete.train_step(
                    model=model,
                    extern_data=TensorDict({"data": b_phon_indices_, "seq_tag": b_extern_data["seq_tag"]}),
                    ce_loss_scale=text_ce_loss_scale,
                    masked_ce_loss_scale=text_masked_ce_loss_scale,
                    label_smoothing=label_smoothing,
                    label_smoothing_start_epoch=label_smoothing_start_epoch,
                    masking_opts=text_masking_opts,
                    aux_loss_scales=None,
                    codebook_diversity_loss_scale=codebook_diversity_loss_scale,  
                    loss_name="text",
                    adv_loss_scale=adv_loss_scale,
                    true_adv_target=1,
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
                    extern_data=TensorDict({"data": b_audio_indices_, "seq_tag": b_extern_data["seq_tag"]}),
                    ce_loss_scale=audio_ce_loss_scale,
                    masked_ce_loss_scale=audio_masked_ce_loss_scale,
                    label_smoothing=label_smoothing,
                    label_smoothing_start_epoch=label_smoothing_start_epoch,
                    masking_opts=audio_masking_opts,
                    aux_loss_scales=None,
                    codebook_diversity_loss_scale=codebook_diversity_loss_scale,  
                    loss_name="audio",
                    adv_loss_scale=adv_loss_scale,
                    true_adv_target=0,
                )

            if pseudo_audio_text_ce_loss_scale > 0.0 and b_phon_indices_ is not None:
                model.step_decoder = model.step_audio_decoder
                compute_bt_loss_for_batch(
                    model=model,
                    extern_data=b_extern_data,
                    ce_loss_scale=pseudo_audio_text_ce_loss_scale,
                    label_smoothing=label_smoothing,
                    label_smoothing_start_epoch=label_smoothing_start_epoch,
                    aux_loss_scales=text_aux_loss_scales,
                    codebook_diversity_loss_scale=codebook_diversity_loss_scale,
                    pseudo_target_indices_=b_pseudo_audio_from_phon_,
                    src_indices_=b_phon_indices_,
                    forward_target=model.forward_audio,
                    src_decoder=model.text_decoder,
                    decode_src_seq=model.decode_text_seq,
                    src_bos_idx=model.text_bos_idx,
                    src_eos_idx=model.text_eos_idx,
                    src_blank_idx=model.text_blank_idx,
                    src_out_dim=model.text_out_dim,
                    loss_name="pseudo",
                )

            if pseudo_text_audio_ce_loss_scale > 0.0 and b_audio_indices_ is not None:
                model.step_decoder = model.step_text_decoder
                compute_bt_loss_for_batch(
                    model=model,
                    extern_data=b_extern_data,
                    ce_loss_scale=pseudo_text_audio_ce_loss_scale,
                    label_smoothing=label_smoothing,
                    label_smoothing_start_epoch=label_smoothing_start_epoch,
                    aux_loss_scales=audio_aux_loss_scales,
                    codebook_diversity_loss_scale=codebook_diversity_loss_scale,
                    pseudo_target_indices_=b_pseudo_text_from_audio_,
                    src_indices_=b_audio_indices_,
                    forward_target=model.forward_text,
                    src_decoder=model.audio_decoder,
                    decode_src_seq=model.decode_audio_seq,
                    src_bos_idx=model.audio_bos_idx,
                    src_eos_idx=model.audio_eos_idx,
                    src_blank_idx=model.audio_blank_idx,
                    src_out_dim=model.audio_out_dim,
                    loss_name="pseudo_reverse",
                )

            model._bt_v2_train_steps += 1
            if model._bt_v2_train_steps >= bt_train_iterations:
                model._bt_v2_state = "ACCUMULATE"
                model._bt_v2_buffer = []
                model._bt_v2_accum_steps = 0

    dummy_tensor = next((p for p in model.parameters() if p.requires_grad), next(model.parameters(), None))
    if dummy_tensor is not None:
        dummy_loss = (dummy_tensor * 0.0).sum()
        ctx.mark_as_loss(dummy_loss, "dummy_accum_fallback", scale=1.0)

    # Mark the global step as a metric so it appears in the training logs
    step_tensor = torch.tensor(ctx.step, dtype=torch.float32, device=dummy_tensor.device if dummy_tensor is not None else "cpu")
    ctx.mark_as_loss(step_tensor, "global_step", scale=0.0)
