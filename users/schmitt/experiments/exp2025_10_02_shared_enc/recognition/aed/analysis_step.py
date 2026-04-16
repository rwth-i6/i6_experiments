__all__ = ["forward_step"]

import copy
from abc import abstractmethod
from typing import Generic, Optional, Callable, Tuple
import sys
import functools

import torch
from torch import Tensor
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

from .beam_search import LabelScorer, State, beam_search_v1
import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim

# from i6_models.assemblies.transformer.transformer_decoder_v1 import TransformerDecoderV1, TransformerDecoderV1State
from i6_models.assemblies.transformer.transformer_decoder_v1 import (
    CausalSelfAttentionV1Config,
    CrossAttentionV1Config,
    CrossAttentionV1State,
    TransformerDecoderBlockV1Config,
    TransformerDecoderV1,
    TransformerDecoderV1Config,
    TransformerDecoderV1State,
    TransformerDecoderBlockV1
)
from i6_models.parts.decoder.cross_att import CrossAttentionV1

from ...pytorch_networks.conformer_aed_discrete_shared_v1 import Model
from ...training.aed_denoising_discrete_shared_backtranslation import train_step, backtranslation_step
from ...training import aed_denoising_discrete


# taken from https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False
) -> torch.Tensor:
    import math
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def cross_att_forward(cross_att_module: CrossAttentionV1, x: Tensor, x_lens: Tensor, state: CrossAttentionV1State) -> Tuple[Tensor, CrossAttentionV1State]:
    """
    Apply cross attention.

    :param x: input of shape (B..., T, F)
    :param x_lens: unused
    :param state: recurrent state of the cross attention module
    """

    # Ev: attention value dim
    # L: length of query (i.e. length of token sequence)

    x = cross_att_module.norm(x)
    q = cross_att_module.q(x)  # B... L Ev
    q = torch.unflatten(q, -1, (cross_att_module.num_heads, -1)).transpose(-3, -2)  # B... H L Ev

    att_out = scaled_dot_product_attention(
        q,
        key=state["k"],
        value=state["v"],
        attn_mask=state["mask"],
        dropout_p=cross_att_module.att_dropout if cross_att_module.training else 0.0,
    )  # B... H L E
    out = att_out.transpose(-3, -2).flatten(-2)  # B... L E
    out = cross_att_module.out_proj(out)  # B... L F
    out = cross_att_module.dropout(out)

    return out, state


def analyze_audio_denoising_step(
    *,
    model: Model,
    extern_data: TensorDict,
    beam_size: int,
    max_tokens_per_sec: Optional[int] = None,
    sample_rate: Optional[int] = None,
    **kwargs,
):
    for block in model.audio_decoder.module_list:
        for module in block.module_list:
            if isinstance(module, CrossAttentionV1):
                module.forward = functools.partial(cross_att_forward, module)

    with torch.enable_grad():
        code_obj_to_func = {}
        captured_tensors = {}  # func -> (list of calls) -> tensor local name -> (list of versions) -> tensor

        def _trace_func(frame, event, arg):
            """
            Trace func to get intermediate outputs.
            """
            func = code_obj_to_func.get(frame.f_code)
            if func:
                if event == "call":
                    captured_tensors.setdefault(func, []).append({})
                else:
                    for k, v in frame.f_locals.items():
                        if not isinstance(v, torch.Tensor):
                            continue
                        prev = captured_tensors[func][-1].get(k, None)
                        if prev is None or prev[-1] is not v:
                            print(f"{func.__qualname__} tensor var changed: {k} = {v}")
                            captured_tensors[func][-1].setdefault(k, []).append(v)
                return _trace_func

        assert beam_size > 0
        for param in model.parameters():
            param.requires_grad = True

        seq_tags = extern_data["seq_tag"].raw_tensor.tolist()
        data = extern_data["data"].raw_tensor
        seq_lens = extern_data["data"].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)
        phon_indices = extern_data["data"].copy_template_replace_dim_tag(axis=1, new_dim_tag=extern_data["data"].dims[1].copy())
        phon_indices.dims[1].dyn_size_ext.raw_tensor = torch.zeros_like(seq_lens)

        funcs_to_trace_list = [
            train_step,
            backtranslation_step,
            aed_denoising_discrete.train_step,
            Model.decode_audio_seq,
            TransformerDecoderV1.forward,
            TransformerDecoderBlockV1.forward,
            CrossAttentionV1.forward,
            scaled_dot_product_attention
        ]
        code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
        _layer_mapping = {
            "att_weights": (scaled_dot_product_attention, 2, "attn_weight", -1),
            "logits": (TransformerDecoderV1.forward, 0, "output_logits", -1),
            "enc_out": (aed_denoising_discrete.train_step, 0, "encoder_output", 0),
            "ce_loss": (aed_denoising_discrete.train_step, 0, "ce_loss", 0),
            "phon_mask": (aed_denoising_discrete.train_step, 0, "phon_mask", -1),
            "label_indices_masked": (aed_denoising_discrete.train_step, 0, "label_indices_masked", -1),
            "label_indices_masked_lens": (aed_denoising_discrete.train_step, 0, "label_indices_masked_lens", -1),
        }

        sys.settrace(_trace_func)

        # use determinstic mask for analysis
        min_span = 4
        max_span = 20
        mask = torch.ones_like(data).bool()
        for b in range(data.size(0)):
            seq_len = seq_lens[b].item()
            start1 = int(seq_len * 0.2)
            start2 = int(seq_len * 0.7)
            mask[b, start1:start1 + 15] = False
            mask[b, start2:start2 + 5] = False
        train_step(
            model=model,
            extern_data=TensorDict(
                {
                    "data": extern_data["data"],
                    "phon_indices": phon_indices,
                    "seq_tag": extern_data["seq_tag"]}
            ),
            audio_ce_loss_scale=1.0,
            audio_masked_ce_loss_scale=0.0,
            text_ce_loss_scale=0.0,
            text_masked_ce_loss_scale=0.0,
            pseudo_audio_text_ce_loss_scale=0.0,
            pseudo_text_audio_ce_loss_scale=0.0,
            text_masking_opts={"mask_prob": 0.0},
            audio_masking_opts={"mask_prob": 0.3, "min_span": min_span, "max_span": max_span, "mask": mask},
            # audio_masking_opts={"mask_prob": 0.0, "min_span": 4, "max_span": 20},
        )
        sys.settrace(None)

        tensors = {}
        for tensor_name, var_path in _layer_mapping.items():
            tensors[tensor_name] = captured_tensors
            for k in var_path:
                tensors[tensor_name] = tensors[tensor_name][k]  # (B, H, L, S)

        ce_loss = tensors["ce_loss"].sum() / (seq_lens + 1).float().sum()
        enc_out = tensors["enc_out"]
        logits = tensors["logits"]
        enc_out.retain_grad()
        logits.retain_grad()
        ce_loss.backward()

        enc_out_grads = enc_out.grad.abs().mean()
        logits_grads = logits.grad.abs().mean()

        print("CE loss:", ce_loss.item())
        print("Encoder output grads abs mean:", enc_out_grads.item())
        print("Logits grads abs mean:", logits_grads.item())

        att_weights = tensors["att_weights"].mean(dim=1).detach().cpu().numpy()  # average over heads
        for b in range(att_weights.shape[0]):
            seq_len = seq_lens[b].item()
            masked_len = tensors["label_indices_masked_lens"][b].item()
            label_indices_masked = tensors["label_indices_masked"][b, :masked_len].detach().cpu().numpy()
            label_indices = data[b, :seq_len].detach().cpu().numpy()
            att_weights_b = att_weights[b, :seq_len, :masked_len]  # target len x source len
            fig, ax = plt.subplots(figsize=(masked_len * 0.2, seq_len * 0.2))
            mat = ax.matshow(att_weights_b)
            plt.title("Attention weights", fontsize=25)
            plt.xlabel("Source sequence position", fontsize=25)
            plt.ylabel("Target sequence position", fontsize=25)
            plt.xticks(ticks=np.arange(0, masked_len), labels=label_indices_masked, rotation=90)
            plt.yticks(ticks=np.arange(0, seq_len), labels=label_indices)
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes((
                0.91,
                0.1,
                0.02,
                0.8
            ))
            cbar = fig.colorbar(mat, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=20)

            phon_mask = tensors["phon_mask"][b, :seq_len].detach().cpu().numpy()
            masked_pos = np.where(~phon_mask)[0]  # positions where mask is False
            masked_pos_collapsed = np.where(label_indices_masked == model.audio_mask_idx)[0]
            for pos in masked_pos:
                ax.axhline(y=pos, xmin=0, xmax=1, color="k", linewidth=.5, alpha=0.7)
            for pos in masked_pos_collapsed:
                ax.axvline(x=pos, ymin=0, ymax=1, color="r", linewidth=.5, alpha=0.7)

            seq_tag = seq_tags[b].replace('/', '_')
            plt.savefig(f"attention_weights_{seq_tag}.png")
            plt.close()

            if b >= 10:
                break

        exit()


def analyze_pseudo_audio_text_ce_step(
    *,
    model: Model,
    extern_data: TensorDict,
    beam_size: int,
    max_tokens_per_sec: Optional[int] = None,
    sample_rate: Optional[int] = None,
    pseudo_audio_text_ce_loss_scale: float = 1.0,
    **kwargs,
):
    for block in model.text_decoder.module_list:
        for module in block.module_list:
            if isinstance(module, CrossAttentionV1):
                module.forward = functools.partial(cross_att_forward, module)

    for block in model.audio_decoder.module_list:
        for module in block.module_list:
            if isinstance(module, CrossAttentionV1):
                module.forward = functools.partial(cross_att_forward, module)

    with torch.enable_grad():
        code_obj_to_func = {}
        captured_tensors = {}  # func -> (list of calls) -> tensor local name -> (list of versions) -> tensor

        def _trace_func(frame, event, arg):
            """
            Trace func to get intermediate outputs.
            """
            func = code_obj_to_func.get(frame.f_code)
            if func:
                if event == "call":
                    captured_tensors.setdefault(func, []).append({})
                else:
                    for k, v in frame.f_locals.items():
                        if not isinstance(v, torch.Tensor):
                            continue
                        prev = captured_tensors[func][-1].get(k, None)
                        if prev is None or prev[-1] is not v:
                            print(f"{func.__qualname__} tensor var changed: {k} = {v}")
                            captured_tensors[func][-1].setdefault(k, []).append(v)
                return _trace_func

        assert beam_size > 0
        for param in model.parameters():
            param.requires_grad = True

        seq_tags = extern_data["seq_tag"].raw_tensor.tolist()
        data = extern_data["data"].raw_tensor
        seq_lens = extern_data["data"].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)
        audio_indices = extern_data["data"].copy_template_replace_dim_tag(axis=1, new_dim_tag=extern_data["data"].dims[1].copy())
        audio_indices.dims[1].dyn_size_ext.raw_tensor = torch.zeros_like(seq_lens)

        # print("data.shape:", data.shape)
        # print("seq_len:", seq_len.detach().cpu().numpy())
        # print(extern_data["data"].dims[1].dyn_size_ext.raw_tensor)
        # exit()

        funcs_to_trace_list = [
            train_step,
            backtranslation_step,
            aed_denoising_discrete.train_step,
            Model.decode_audio_seq,
            TransformerDecoderV1.forward,
            TransformerDecoderBlockV1.forward,
            CrossAttentionV1.forward,
            scaled_dot_product_attention
        ]
        code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
        _layer_mapping = {
            "att_weights": (scaled_dot_product_attention, 2, "attn_weight", -1),
            "logits": (TransformerDecoderV1.forward, 0, "output_logits", -1),
            "enc_out": (aed_denoising_discrete.train_step, 0, "encoder_output", 0),
            "ce_loss": (aed_denoising_discrete.train_step, 0, "ce_loss", 0),
            "phon_mask": (aed_denoising_discrete.train_step, 0, "phon_mask", -1),
            "label_indices_masked": (aed_denoising_discrete.train_step, 0, "label_indices_masked", -1),
            "label_indices_masked_lens": (aed_denoising_discrete.train_step, 0, "label_indices_masked_lens", -1),
        }
        if pseudo_audio_text_ce_loss_scale > 0.0:
            _layer_mapping.update({
                "att_weights_backtranslate": (scaled_dot_product_attention, -1, "attn_weight", -1),
                "pseudo_targets": (backtranslation_step, 0, "pseudo_target_indices", -1),
                "pseudo_target_lens": (backtranslation_step, 0, "pseudo_target_indices_lens", -1),
            })

        sys.settrace(_trace_func)

        # use determinstic mask for analysis
        min_span = 4
        max_span = 20
        mask = torch.ones_like(data).bool()
        for b in range(data.size(0)):
            seq_len = seq_lens[b].item()
            start1 = int(seq_len * 0.2)
            start2 = int(seq_len * 0.7)
            mask[b, start1:start1 + 5] = False
            mask[b, start2:start2 + 3] = False
        train_step(
            model=model,
            extern_data=TensorDict(
                {
                    "data": audio_indices,
                    "phon_indices": extern_data["data"],
                    "seq_tag": extern_data["seq_tag"]}
            ),
            audio_ce_loss_scale=0.0,
            audio_masked_ce_loss_scale=0.0,
            text_ce_loss_scale=1.0,
            text_masked_ce_loss_scale=0.0,
            pseudo_audio_text_ce_loss_scale=pseudo_audio_text_ce_loss_scale,
            pseudo_text_audio_ce_loss_scale=0.0,
            audio_masking_opts={"mask_prob": 0.0},
            text_masking_opts={"mask_prob": 0.3, "min_span": min_span, "max_span": max_span, "mask": mask},
            # audio_masking_opts={"mask_prob": 0.0, "min_span": 4, "max_span": 20},
        )
        sys.settrace(None)

        tensors = {}
        for tensor_name, var_path in _layer_mapping.items():
            tensors[tensor_name] = captured_tensors
            for k in var_path:
                tensors[tensor_name] = tensors[tensor_name][k]  # (B, H, L, S)

        ce_loss = tensors["ce_loss"].sum() / (seq_lens + 1).float().sum()
        enc_out = tensors["enc_out"]
        logits = tensors["logits"]
        enc_out.retain_grad()
        logits.retain_grad()
        ce_loss.backward()

        enc_out_grads = enc_out.grad.abs().mean()
        logits_grads = logits.grad.abs().mean()

        print("CE loss:", ce_loss.item())
        print("Encoder output grads abs mean:", enc_out_grads.item())
        print("Logits grads abs mean:", logits_grads.item())

        att_weights = tensors["att_weights"].mean(dim=1).detach().cpu().numpy()  # average over heads
        print("att_weights.shape:", att_weights.shape)
        for b in range(att_weights.shape[0]):
            seq_len = seq_lens[b].item()
            masked_len = tensors["label_indices_masked_lens"][b].item()
            label_indices_masked = tensors["label_indices_masked"][b, :masked_len].detach().cpu().numpy()
            label_indices = data[b, :seq_len].detach().cpu().numpy()
            att_weights_b = att_weights[b, :seq_len, :masked_len]  # target len x source len
            fig, ax = plt.subplots(figsize=(masked_len * 0.2, seq_len * 0.2))
            mat = ax.matshow(att_weights_b)
            plt.title("Attention weights", fontsize=25)
            plt.xlabel("Source sequence position", fontsize=25)
            plt.ylabel("Target sequence position", fontsize=25)
            plt.xticks(ticks=np.arange(0, masked_len), labels=label_indices_masked, rotation=90)
            plt.yticks(ticks=np.arange(0, seq_len), labels=label_indices)
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes((
                0.91,
                0.1,
                0.02,
                0.8
            ))
            cbar = fig.colorbar(mat, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=20)

            phon_mask = tensors["phon_mask"][b, :seq_len].detach().cpu().numpy()
            masked_pos = np.where(~phon_mask)[0]  # positions where mask is False
            masked_pos_collapsed = np.where(label_indices_masked == model.text_mask_idx)[0]
            for pos in masked_pos:
                ax.axhline(y=pos, xmin=0, xmax=1, color="k", linewidth=.5, alpha=0.7)
            for pos in masked_pos_collapsed:
                ax.axvline(x=pos, ymin=0, ymax=1, color="r", linewidth=.5, alpha=0.7)

            seq_tag = seq_tags[b].replace('/', '_')
            plt.savefig(f"attention_weights_text_{seq_tag}.png")
            plt.close()

            if b >= 10:
                break

        if "att_weights_backtranslate" in tensors:
            att_weights_bt = tensors["att_weights_backtranslate"].mean(dim=1).detach().cpu().numpy()  # average over heads
            for b in range(att_weights.shape[0]):
                seq_len = seq_lens[b].item()
                pseudo_target_len = tensors["pseudo_target_lens"][b].item()
                pseudo_targets = tensors["pseudo_targets"][b, :masked_len].detach().cpu().numpy()
                label_indices = data[b, :seq_len].detach().cpu().numpy()
                att_weights_bt_b = att_weights_bt[b, :seq_len, :pseudo_target_len]  # target len x source len
                fig, ax = plt.subplots(figsize=(pseudo_target_len * 0.2, seq_len * 0.2))
                mat = ax.matshow(att_weights_bt_b)
                plt.title("Attention weights", fontsize=25)
                plt.xlabel("Source sequence position", fontsize=25)
                plt.ylabel("Target sequence position", fontsize=25)
                plt.xticks(ticks=np.arange(0, pseudo_target_len), labels=pseudo_targets, rotation=90)
                plt.yticks(ticks=np.arange(0, seq_len), labels=label_indices)
                fig.subplots_adjust(right=0.9)
                cbar_ax = fig.add_axes((
                    0.91,
                    0.1,
                    0.02,
                    0.8
                ))
                cbar = fig.colorbar(mat, cax=cbar_ax)
                cbar.ax.tick_params(labelsize=20)

                seq_tag = seq_tags[b].replace('/', '_')
                plt.savefig(f"attention_weights_text_bt_{seq_tag}.png")
                plt.close()

                if b >= 10:
                    break

        exit()

        # beam_dim = Dim(beam_size, name="beam")
        # vocab_dim = Dim(model.text_out_dim, name="vocab")
        # lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim])
        # lens_dim = Dim(lens_data, name="seq_len")
        #
        # ctx = rf.get_run_ctx()
        # seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)
        # ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
        # ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])
