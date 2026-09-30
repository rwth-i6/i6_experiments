__all__ = ["Model"]

import math
import os
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union, Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from i6_models.assemblies.conformer.conformer_rel_pos_v1 import (
    ConformerConvolutionV2Config,
    ConformerMHSARelPosV1,
    ConformerMHSARelPosV1Config,
    ConformerPositionwiseFeedForwardV2Config,
    ConformerRelPosBlockV1Config,
    ConformerRelPosEncoderV1,
    ConformerRelPosEncoderV1Config,
)
from i6_models.assemblies.transformer.transformer_decoder_v1 import (
    CausalSelfAttentionV1Config,
    CrossAttentionV1Config,
    TransformerDecoderBlockV1Config,
    TransformerDecoderV1,
    TransformerDecoderV1Config,
    TransformerDecoderV1State,
)
from i6_models.parts.masked_norm import MaskedBatchNorm1dV1

from i6_experiments.users.schmitt.experiments.exp2025_10_02_shared_enc.recognition.aed import (
    EncoderDecoderModel,
)
from i6_experiments.users.schmitt.experiments.exp2025_10_02_shared_enc.training.aed_denoising_discrete_shared import (
    SharedDenoisingAedModel,
)


def _relu_sq(x):
    """Squared ReLU."""
    return nn.functional.relu(x) ** 2.0


class ConformerEncoderWithBottleneck(nn.Module):
    def __init__(self, encoder: ConformerRelPosEncoderV1, enc_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = nn.Linear(enc_dim, bottleneck_dim)

    def forward(self, *args, **kwargs):
        encoder_outputs, out_mask = self.encoder.forward(*args, **kwargs)
        encoder_outputs = [self.bottleneck(enc_out) for enc_out in encoder_outputs]
        return encoder_outputs, out_mask


def _valid_frame_mask(seq_lens: Tensor, max_len: int) -> Tensor:
    """[B, max_len] bool mask, True for non-padding frames."""
    return torch.arange(max_len, device=seq_lens.device)[None, :] < seq_lens[:, None]


def _ngram_windows(encoder_output: Tensor, encoder_lens: Tensor, num_frames: int, right_pad: bool = False) -> Tensor:
    B, T, F = encoder_output.shape
    encoder_lens = encoder_lens.to(encoder_output.device)
    if num_frames == 1:
        return encoder_output[_valid_frame_mask(encoder_lens, T)]
    if right_pad:
        encoder_output = encoder_output * _valid_frame_mask(encoder_lens, T).unsqueeze(-1)
        encoder_output = torch.cat([encoder_output, encoder_output.new_zeros((B, num_frames - 1, F))], dim=1)
    elif T < num_frames:
        return encoder_output.new_zeros((0, F * num_frames))
    windows = encoder_output.unfold(dimension=1, size=num_frames, step=1)
    W = windows.shape[1]
    windows = windows.permute(0, 1, 3, 2).reshape(B, W, num_frames * F)
    win_pos = torch.arange(W, device=encoder_output.device)[None, :]
    if right_pad:
        valid = win_pos < encoder_lens[:, None]
    else:
        valid = win_pos <= (encoder_lens[:, None] - num_frames)
    return windows[valid]


class _Discriminator(nn.Module):
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def gradient_penalty(self, encoder_output: Tensor, encoder_lens: Tensor) -> Tensor:
        x = encoder_output.detach().requires_grad_(True)
        grads = self._critic_input_grad(x, encoder_lens)
        grad_norm = grads.norm(2, dim=-1)
        valid = _valid_frame_mask(encoder_lens.to(grad_norm.device), grad_norm.shape[1])
        return ((grad_norm[valid] - 1.0) ** 2).mean()

    def _critic_input_grad(self, inp: Tensor, lens: Tensor) -> Tensor:
        with torch.backends.cudnn.flags(enabled=False):
            disc_out = self.forward(inp, lens)
            (grads,) = torch.autograd.grad(
                outputs=disc_out,
                inputs=inp,
                grad_outputs=torch.ones_like(disc_out),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
        return grads


class MlpDiscriminator(_Discriminator):
    def __init__(
        self, in_dim: int, num_frames: int = 1, num_layers: int = 3, hidden_dim: int = 512, pad_windows: bool = False
    ):
        super().__init__()
        self.num_frames = num_frames
        self.pad_windows = pad_windows
        self.layers = nn.ModuleList(
            nn.Linear((in_dim * num_frames) if i == 0 else hidden_dim, hidden_dim if i < num_layers - 1 else 1)
            for i in range(num_layers)
        )

    def forward(self, encoder_output: Tensor, encoder_lens: Tensor) -> Tensor:
        x = _ngram_windows(
            encoder_output, encoder_lens, self.num_frames, right_pad=self.pad_windows
        )
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))
        x = self.layers[-1](x)
        return x.squeeze(-1)


class LstmDiscriminator(_Discriminator):
    def __init__(self, in_dim: int, hidden_dim: int = 512, num_layers: int = 2, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.out = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, encoder_output: Tensor, encoder_lens: Tensor) -> Tensor:
        packed = pack_padded_sequence(encoder_output, encoder_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out_padded, _ = pad_packed_sequence(out, batch_first=True)
        logits = self.out(out_padded).squeeze(-1)
        mask = _valid_frame_mask(encoder_lens.to(logits.device), out_padded.shape[1])
        return logits[mask]


_MLP_DISCRIMINATOR_NGRAM = {"mlp": 1, "mlp_2gram": 2, "mlp_3gram": 3, "mlp_4gram": 4}


class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_vars: int,
        temp: Tuple[float, float, float],
        groups: int,
        combine_groups: bool,
        vq_dim: int,
        time_first: bool,
        activation: nn.Module = None,
        weight_proj_depth: int = 1,
        weight_proj_factor: int = 1,
    ):
        super().__init__()
        if activation is None:
            activation = nn.GELU()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert vq_dim % groups == 0, f"dim {vq_dim} must be divisible by groups {groups} for concatenation"
        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)

        if weight_proj_depth > 1:
            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[block(self.input_dim if i == 0 else inner_dim, inner_dim) for i in range(weight_proj_depth - 1)],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        assert len(temp) == 3, f"{temp}, {len(temp)}"
        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp

    def set_num_updates(self, num_updates: int):
        self.curr_temp = max(self.max_temp * self.temp_decay**num_updates, self.min_temp)

    def compute_orthogonality_loss(self) -> Tensor:
        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)
        vars = vars.reshape(self.groups, self.num_vars, -1)
        normed_vars = nn.functional.normalize(vars, p=2.0, dim=-1)
        sim_matrix = torch.bmm(normed_vars, normed_vars.transpose(1, 2))
        eye = torch.eye(self.num_vars, device=vars.device, dtype=vars.dtype).unsqueeze(0)
        diff = sim_matrix - eye
        if self.num_vars > 1:
            orth_loss = (diff ** 2).sum(dim=(-1, -2)) / (self.num_vars * (self.num_vars - 1))
        else:
            orth_loss = (diff ** 2).mean(dim=(-1, -2))
        return orth_loss.mean()

    def forward(self, x: Tensor) -> Dict[str, Any]:
        result = {"num_vars": self.num_vars * self.groups}
        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)

        _, k = x.max(-1)
        hard_x = x.new_zeros(*x.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(-torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)).sum()

        avg_probs = torch.softmax(x.view(bsz * tsz, self.groups, -1).float(), dim=-1).mean(dim=0)
        result["prob_perplexity"] = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)).sum()

        result["temp"] = self.curr_temp
        result["code_orthogonality_loss"] = self.compute_orthogonality_loss()

        if self.training:
            x = nn.functional.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.reshape(bsz * tsz, self.groups, self.num_vars)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)
        vars = vars.reshape(self.groups, self.num_vars, -1)

        x = torch.einsum("bgv,gvd->bgd", x, vars)
        x = x.reshape(bsz, tsz, -1)

        if not self.time_first:
            x = x.transpose(1, 2)

        result["x"] = x
        return result


_DEFAULT_CODEBOOK_OPTS = {
    "codebook_prob": 0.5,
    "latent_vars": 320,
    "latent_groups": 2,
    "latent_dim": 0,
    "latent_temp": (2.0, 0.5, 0.999995),
    "quantizer_depth": 1,
    "quantizer_factor": 3,
}


class Model(nn.Module, SharedDenoisingAedModel, EncoderDecoderModel):
    """
    Conformer encoder + Transformer decoder AED model with Frozen Cross-Modal Embeddings
    (Artetxe et al., UNMT architecture).
    """

    def __init__(
        self,
        *,
        model_dim: int,
        text_out_dim: Optional[int],
        audio_out_dim: Optional[int],
        num_heads: int,
        num_enc_layers: int,
        num_text_dec_layers: int,
        num_audio_dec_layers: int,
        text_aux_loss_layers: Sequence[int] = (),
        audio_aux_loss_layers: Sequence[int] = (),
        dropout: float = 0.1,
        dropout_broadcast_axes: Optional[Literal["B", "BT", "T"]] = "BT",
        logits_bias: bool = False,
        aux_logits_bias: bool = False,
        enc_bottleneck_dim: Optional[int] = None,
        share_decoder: bool = False,
        discriminator_type: Optional[str] = None,
        discriminator_pad_ngram_windows: bool = False,
        codebook_opts: Optional[Dict[str, Any]] = None,
        fix_decode_text_seq_for_shared_dec: bool = False,
        # Cross-modal learned embeddings configuration
        audio_embeddings_path: Optional[str] = None,
        text_embeddings_path: Optional[str] = None,
        cross_modal_emb_dim: int = 300,
        freeze_cross_modal_embeddings: bool = True,
        **_kwargs_unused,
    ):
        super().__init__()

        assert model_dim > 0
        assert len(text_aux_loss_layers) == len(set(text_aux_loss_layers))
        assert list(text_aux_loss_layers) == sorted(text_aux_loss_layers)
        assert len(audio_aux_loss_layers) == len(set(audio_aux_loss_layers))
        assert list(audio_aux_loss_layers) == sorted(audio_aux_loss_layers)
        assert not share_decoder or (num_text_dec_layers == num_audio_dec_layers), (
            "when sharing decoder, number of text and audio decoder layers must be equal"
        )

        rel_pos_clip = 16
        pos_emb_dropout = 0.1
        learnable_pos_emb = True
        with_linear_pos = False
        with_pos_bias = False
        separate_pos_emb_per_head = False
        attn_dropout_broadcast = None

        self.audio_mask_idx = audio_out_dim
        self.audio_bos_idx = audio_out_dim + 1
        self.audio_eos_idx = audio_out_dim + 2
        self.audio_out_dim = audio_out_dim + 3

        self.text_mask_idx = text_out_dim
        self.text_bos_idx = text_out_dim + 1
        self.text_eos_idx = text_out_dim + 2
        self.text_out_dim = text_out_dim + 3

        self.mask_idx = self.text_mask_idx
        self.bos_idx = self.text_bos_idx
        self.eos_idx = self.text_eos_idx
        self.out_dim = self.text_out_dim

        block_cfg = ConformerRelPosBlockV1Config(
            ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                input_dim=model_dim,
                hidden_dim=model_dim * 4,
                dropout=dropout,
                activation=_relu_sq,
                dropout_broadcast_axes=dropout_broadcast_axes,
            ),
            mhsa_cfg=ConformerMHSARelPosV1Config(
                input_dim=model_dim,
                num_att_heads=num_heads,
                att_weights_dropout=dropout,
                with_bias=False,
                dropout=dropout,
                dropout_broadcast_axes=dropout_broadcast_axes,
                rel_pos_clip=rel_pos_clip,
                pos_emb_dropout=pos_emb_dropout,
                learnable_pos_emb=learnable_pos_emb,
                with_linear_pos=with_linear_pos,
                with_pos_bias=with_pos_bias,
                separate_pos_emb_per_head=separate_pos_emb_per_head,
            ),
            conv_cfg=ConformerConvolutionV2Config(
                channels=model_dim,
                kernel_size=33,
                dropout=dropout,
                dropout_broadcast_axes=dropout_broadcast_axes,
                activation=nn.functional.silu,
                norm=MaskedBatchNorm1dV1(model_dim, eps=1e-3, momentum=0.1),
            ),
            modules=["ff", "mhsa", "conv", "ff"],
            scales=[0.5, 1.0, 1.0, 0.5],
        )
        enc_cfg = ConformerRelPosEncoderV1Config(num_layers=num_enc_layers, frontend=None, block_cfg=block_cfg)
        self.encoder = ConformerRelPosEncoderV1(enc_cfg)
        if enc_bottleneck_dim is not None:
            self.encoder = ConformerEncoderWithBottleneck(self.encoder, model_dim, enc_bottleneck_dim)

        # -------------------------------------------------------------
        # Cross-Modal Frozen Input Embeddings Setup (UNMT Paradigm)
        # -------------------------------------------------------------
        self.cross_modal_emb_dim = cross_modal_emb_dim
        self.audio_embeddings_path = audio_embeddings_path
        self.text_embeddings_path = text_embeddings_path
        self.freeze_cross_modal_embeddings = freeze_cross_modal_embeddings

        emb_dim = cross_modal_emb_dim

        # Load audio embeddings from file if available
        audio_emb_data = None
        if audio_embeddings_path is not None:
            path_str = str(audio_embeddings_path)
            if os.path.exists(path_str):
                try:
                    audio_emb_data = np.load(path_str)
                    emb_dim = audio_emb_data.shape[1]
                    print(f"Loaded audio cross-modal embeddings from {path_str} with shape {audio_emb_data.shape}", flush=True)
                except Exception as e:
                    print(f"Warning: could not load audio embeddings from {path_str}: {e}", flush=True)
            else:
                print(f"Notice: Audio embeddings path '{path_str}' not on disk yet (will be loaded when rsync'd). Initializing with dim {emb_dim}.", flush=True)

        # Load text embeddings from file if available
        text_emb_data = None
        if text_embeddings_path is not None:
            path_str = str(text_embeddings_path)
            if os.path.exists(path_str):
                try:
                    text_emb_data = np.load(path_str)
                    emb_dim = text_emb_data.shape[1]
                    print(f"Loaded text cross-modal embeddings from {path_str} with shape {text_emb_data.shape}", flush=True)
                except Exception as e:
                    print(f"Warning: could not load text embeddings from {path_str}: {e}", flush=True)
            else:
                print(f"Notice: Text embeddings path '{path_str}' not on disk yet (will be loaded when rsync'd). Initializing with dim {emb_dim}.", flush=True)

        self.audio_embedding = nn.Embedding(self.audio_out_dim, emb_dim)
        self.text_embedding = nn.Embedding(self.text_out_dim, emb_dim)

        if audio_emb_data is not None:
            k_audio = min(audio_emb_data.shape[0], self.audio_out_dim)
            self.audio_embedding.weight.data[:k_audio] = torch.from_numpy(audio_emb_data[:k_audio]).float()
        if text_emb_data is not None:
            k_text = min(text_emb_data.shape[0], self.text_out_dim)
            self.text_embedding.weight.data[:k_text] = torch.from_numpy(text_emb_data[:k_text]).float()

        if freeze_cross_modal_embeddings:
            self.audio_embedding.weight.requires_grad = False
            self.text_embedding.weight.requires_grad = False

        # Input projection bridging cross-modal embedding space with Conformer model_dim
        if emb_dim != model_dim:
            self.enc_input_proj = nn.Linear(emb_dim, model_dim)
        else:
            self.enc_input_proj = nn.Identity()

        self.share_decoder = share_decoder
        encoder_dim = model_dim if enc_bottleneck_dim is None else enc_bottleneck_dim
        dec_cfgs = {
            name: TransformerDecoderV1Config(
                block_cfg=TransformerDecoderBlockV1Config(
                    ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                        input_dim=model_dim,
                        hidden_dim=model_dim * 4,
                        dropout=dropout,
                        activation=nn.functional.relu,
                        dropout_broadcast_axes=dropout_broadcast_axes,
                    ),
                    mhsa_cfg=CausalSelfAttentionV1Config(
                        att_dropout=dropout,
                        att_dropout_broadcast_axes=attn_dropout_broadcast,
                        dropout=dropout,
                        dropout_broadcast_axes=dropout_broadcast_axes,
                        model_dim=model_dim,
                        key_dim_total=model_dim,
                        value_dim_total=model_dim,
                        num_heads=num_heads,
                        with_bias=True,
                    ),
                    cross_cfg=CrossAttentionV1Config(
                        att_dropout=dropout,
                        att_dropout_broadcast_axes=attn_dropout_broadcast,
                        dropout=dropout,
                        dropout_broadcast_axes=dropout_broadcast_axes,
                        encoder_dim=encoder_dim,
                        model_dim=model_dim,
                        key_dim_total=model_dim,
                        value_dim_total=model_dim,
                        num_heads=num_heads,
                        with_bias=True,
                    ),
                ),
                input_dropout=dropout,
                input_embedding_scale=None,
                num_blocks=num_text_dec_layers
                if share_decoder
                else (num_text_dec_layers if name == "text" else num_audio_dec_layers),
                num_output=(self.text_out_dim + self.audio_out_dim)
                if share_decoder
                else (self.text_out_dim if name == "text" else self.audio_out_dim),
                logits_bias=logits_bias,
                share_embedding=False,
            )
            for name in (["shared"] if share_decoder else ["text", "audio"])
        }
        if share_decoder:
            self.decoder = TransformerDecoderV1(dec_cfgs["shared"])
            self.text_decoder = self.decoder
            self.audio_decoder = self.decoder
        else:
            self.text_decoder = TransformerDecoderV1(dec_cfgs["text"])
            self.audio_decoder = TransformerDecoderV1(dec_cfgs["audio"])

        if discriminator_type is None:
            self.discriminator = None
        elif discriminator_type == "lstm":
            self.discriminator = LstmDiscriminator(in_dim=encoder_dim)
        else:
            assert discriminator_type in _MLP_DISCRIMINATOR_NGRAM, f"unknown discriminator_type {discriminator_type!r}"
            self.discriminator = MlpDiscriminator(
                in_dim=encoder_dim,
                num_frames=_MLP_DISCRIMINATOR_NGRAM[discriminator_type],
                pad_windows=discriminator_pad_ngram_windows,
            )

        if codebook_opts is not None and codebook_opts.get("codebook_prob", 0.0) > 0.0:
            codebook_opts = {**_DEFAULT_CODEBOOK_OPTS, **codebook_opts}
            self.codebook_prob = codebook_opts["codebook_prob"]
            vq_dim = codebook_opts["latent_dim"] if codebook_opts["latent_dim"] > 0 else encoder_dim
            assert vq_dim == encoder_dim, (
                f"quantized dim ({vq_dim}) must match the encoder output dim ({encoder_dim})"
            )
            self.quantizer = GumbelVectorQuantizer(
                dim=encoder_dim,
                num_vars=codebook_opts["latent_vars"],
                temp=codebook_opts["latent_temp"],
                groups=codebook_opts["latent_groups"],
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=codebook_opts["quantizer_depth"],
                weight_proj_factor=codebook_opts["quantizer_factor"],
            )
        else:
            self.codebook_prob = None
            self.quantizer = None
        self.quantizer_out = None

        self.text_aux_loss_layers = text_aux_loss_layers
        self.out_text_aux_logits = nn.ModuleList(
            [nn.Linear(encoder_dim, text_out_dim + 1, bias=aux_logits_bias) for _ in range(len(text_aux_loss_layers))]
        )
        self._out_fetch_layers_text = sorted(v - 1 for v in {*text_aux_loss_layers, enc_cfg.num_layers})
        self.text_blank_idx = text_out_dim

        self.audio_aux_loss_layers = audio_aux_loss_layers
        self.out_audio_aux_logits = nn.ModuleList(
            [nn.Linear(encoder_dim, audio_out_dim + 1, bias=aux_logits_bias) for _ in range(len(audio_aux_loss_layers))]
        )
        self._out_fetch_layers_audio = sorted(v - 1 for v in {*audio_aux_loss_layers, enc_cfg.num_layers})
        self.audio_blank_idx = audio_out_dim

        self.fix_decode_text_seq_for_shared_dec = fix_decode_text_seq_for_shared_dec

    def load_pretrained_embeddings(self, audio_path: Optional[str] = None, text_path: Optional[str] = None):
        """Loads and updates the frozen cross-modal embedding weights from disk."""
        if audio_path and os.path.exists(audio_path):
            audio_emb = np.load(audio_path)
            k = min(audio_emb.shape[0], self.audio_out_dim)
            self.audio_embedding.weight.data[:k] = torch.from_numpy(audio_emb[:k]).float()
            print(f"Reloaded audio cross-modal embeddings from {audio_path}", flush=True)

        if text_path and os.path.exists(text_path):
            text_emb = np.load(text_path)
            p = min(text_emb.shape[0], self.text_out_dim)
            self.text_embedding.weight.data[:p] = torch.from_numpy(text_emb[:p]).float()
            print(f"Reloaded text cross-modal embeddings from {text_path}", flush=True)

        if self.freeze_cross_modal_embeddings:
            self.audio_embedding.weight.requires_grad = False
            self.text_embedding.weight.requires_grad = False

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        if hasattr(self, "enc_input_proj") and isinstance(self.enc_input_proj, nn.Linear):
            for param in self.enc_input_proj.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        if hasattr(self, "enc_input_proj") and isinstance(self.enc_input_proj, nn.Linear):
            for param in self.enc_input_proj.parameters():
                param.requires_grad = True

    def _maybe_quantize(self, encoder_output: Tensor) -> Tensor:
        if self.quantizer is None:
            return encoder_output
        q = self.quantizer(encoder_output)
        self.quantizer_out = {k: v for k, v in q.items() if k != "x"}
        quantized = q["x"]
        time_size = quantized.shape[1]
        num_replace = int(time_size * self.codebook_prob)
        q_w = quantized.new_zeros(time_size)
        if num_replace > 0:
            replace_idx = torch.randperm(time_size, device=quantized.device)[:num_replace]
            q_w[replace_idx] = 1.0
        q_w = q_w.view(1, time_size, 1)
        return q_w * quantized + (1.0 - q_w) * encoder_output

    def forward(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.forward_audio(indices, seq_lens)

    def forward_text(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if indices.dtype in (torch.float32, torch.float16, torch.bfloat16):
            data = indices
        else:
            data = self.text_embedding(indices)  # (B, T, emb_dim)
        data = self.enc_input_proj(data)  # (B, T, model_dim)
        data_mask = torch.less(torch.arange(data.shape[-2], device=data.device)[None, :], seq_lens[:, None])
        encoder_outputs, out_mask = self.encoder.forward(data, data_mask, return_layers=self._out_fetch_layers_text)
        assert len(self.out_text_aux_logits) <= len(encoder_outputs)
        out_aux_logits = [
            aux_linear(aux_out) for aux_linear, aux_out in zip(self.out_audio_aux_logits, encoder_outputs)
        ]
        out_seq_lens = out_mask.sum(dim=-1)
        return self._maybe_quantize(encoder_outputs[-1]), out_aux_logits, out_seq_lens, out_mask

    def forward_audio(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if indices.dtype in (torch.float32, torch.float16, torch.bfloat16):
            data = indices
        else:
            data = self.audio_embedding(indices)  # (B, T, emb_dim)
        data = self.enc_input_proj(data)  # (B, T, model_dim)
        data_mask = torch.less(torch.arange(data.shape[-2], device=data.device)[None, :], seq_lens[:, None])
        encoder_outputs, out_mask = self.encoder.forward(data, data_mask, return_layers=self._out_fetch_layers_audio)
        assert len(self.out_audio_aux_logits) <= len(encoder_outputs)
        out_aux_logits = [aux_linear(aux_out) for aux_linear, aux_out in zip(self.out_text_aux_logits, encoder_outputs)]
        out_seq_lens = out_mask.sum(dim=-1)
        return self._maybe_quantize(encoder_outputs[-1]), out_aux_logits, out_seq_lens, out_mask

    def decode_text_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        state = self.decoder.transform_encoder_output(
            encoder_output, encoder_output_lens, self.text_decoder.get_initial_state()
        )
        dec_out, _ = self.text_decoder.forward(x, x_lens, state)
        if self.share_decoder and self.fix_decode_text_seq_for_shared_dec:
            dec_out = dec_out[..., : self.text_out_dim]
        return dec_out

    def decode_audio_seq(
        self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor
    ) -> Tensor:
        state = self.decoder.transform_encoder_output(
            encoder_output, encoder_output_lens, self.audio_decoder.get_initial_state()
        )
        if self.share_decoder:
            x = x + self.text_out_dim
        dec_out, _ = self.audio_decoder.forward(x, x_lens, state)
        if self.share_decoder:
            dec_out = dec_out[:, :, self.text_out_dim :]
        return dec_out

    def decode_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        return self.decode_text_seq(x, x_lens, encoder_output, encoder_output_lens)

    def forward_encoder(
        self, indices: Tensor, indices_lens: Tensor, decoder: TransformerDecoderV1, forward_func: Callable
    ) -> TransformerDecoderV1State:
        encoder_output, aux_logits, encoder_lens, _ = forward_func(indices, indices_lens)
        state = decoder.get_initial_state()
        state = decoder.transform_encoder_output(encoder_output.unsqueeze(1), encoder_lens.unsqueeze(1), state)
        return state

    def step_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State, decoder: Optional[TransformerDecoderV1] = None
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        return self.step_text_decoder(labels, state)

    def step_audio_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        if self.share_decoder:
            labels = labels + self.text_out_dim
        dec_out, dec_state = self.audio_decoder.forward(
            labels,
            torch.full(labels.shape[:-1], 1, device=labels.device, dtype=torch.int32),
            state,
        )
        if self.share_decoder:
            dec_out = dec_out[..., self.text_out_dim :]
        return dec_out, dec_state

    def step_text_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        dec_out, dec_state = self.text_decoder.forward(
            labels,
            torch.full(labels.shape[:-1], 1, device=labels.device, dtype=torch.int32),
            state,
        )
        if self.share_decoder:
            dec_out = dec_out[..., : self.text_out_dim]
        return dec_out, dec_state
