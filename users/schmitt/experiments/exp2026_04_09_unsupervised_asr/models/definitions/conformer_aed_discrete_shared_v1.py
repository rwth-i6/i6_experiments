__all__ = ["Model"]

import math
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union, Any

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

# from i6_models.config import ModuleFactoryV1
# from i6_models.parts.decoder import CrossAttentionV1
# from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
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


class _SpecAugArgs(TypedDict):
    time_min_num_masks: int
    time_max_mask_per_n_frames: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int


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
    """Concatenate ``num_frames`` consecutive frames in the feature dim (sliding window / n-gram).

    Returns a flat ``[M, F * num_frames]`` tensor over the windows in the batch. For
    ``num_frames == 1`` this is just the flat set of valid frames ``[sum(lens), F]``.

    ``right_pad`` controls the boundary handling for ``num_frames > 1``:

    - ``False`` (default): only windows that fit fully inside a sequence are kept, so a sequence of
      valid length ``L`` yields ``max(L - num_frames + 1, 0)`` windows. Windows never span into
      padding, but the last ``num_frames - 1`` frames start no window (they are under-represented)
      and sequences shorter than ``num_frames`` yield no windows at all.
    - ``True``: every sequence is right-padded by ``num_frames - 1`` zero frames so each real frame
      starts its own window -> exactly ``L`` windows per sequence (every frame equally represented;
      short sequences still yield windows). The trailing frames' windows see zeros for the padded
      tail (a zero-padding analog; the encoder has no mask-token state at its output).
    """
    B, T, F = encoder_output.shape
    encoder_lens = encoder_lens.to(encoder_output.device)
    if num_frames == 1:
        return encoder_output[_valid_frame_mask(encoder_lens, T)]  # [sum(lens), F]
    if right_pad:
        # zero the padded (non-valid) positions first so windows extending past a sequence's real
        # length see zeros rather than the encoder's states for batch-padding frames, then append
        # num_frames-1 zero frames so each real frame can start a window.
        encoder_output = encoder_output * _valid_frame_mask(encoder_lens, T).unsqueeze(-1)
        encoder_output = torch.cat([encoder_output, encoder_output.new_zeros((B, num_frames - 1, F))], dim=1)
    elif T < num_frames:
        return encoder_output.new_zeros((0, F * num_frames))
    # unfold over time -> [B, W, F, num_frames]; permute+reshape so the num_frames consecutive frames
    # are concatenated along the feature dim (f_0, f_1, ..., f_{n-1}).
    windows = encoder_output.unfold(dimension=1, size=num_frames, step=1)
    W = windows.shape[1]
    windows = windows.permute(0, 1, 3, 2).reshape(B, W, num_frames * F)  # [B, W, num_frames * F]
    win_pos = torch.arange(W, device=encoder_output.device)[None, :]
    if right_pad:
        valid = win_pos < encoder_lens[:, None]  # each real frame starts a window
    else:
        # a window starting at position w is valid iff w + num_frames <= len (w <= len - num_frames)
        valid = win_pos <= (encoder_lens[:, None] - num_frames)
    return windows[valid]  # [M, num_frames * F]


class _Discriminator(nn.Module):
    """Base class for the domain-adversarial discriminators, providing freeze/unfreeze helpers.

    Subclasses implement ``forward(encoder_output [B, T, F], encoder_lens [B]) -> [M]``, returning a
    flat vector of per-frame (or per-window) logits over the valid positions in the batch.
    """

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def gradient_penalty(self, encoder_output: Tensor, encoder_lens: Tensor) -> Tensor:
        """1-centered gradient penalty (WGAN-GP) on the critic input; returns a scalar.

        Textbook WGAN-GP penalizes the critic's input-gradient norm at points interpolated between
        real and fake samples. Under ``alternate_batching`` a batch is single-modality, so real
        (text) and fake (audio) encoder states never co-occur in one step and cannot be
        interpolated. We instead penalize the gradient norm at the *real* encoder states of the
        current modality (a "real-sample" gradient penalty); accumulated over the alternating
        text/audio sub-batches this constrains the critic to be ~1-Lipschitz on both domains'
        state manifolds.

        The input is detached (the encoder is frozen on the discriminator step anyway) so only the
        discriminator parameters receive the penalty gradient. ``create_graph=True`` makes the
        penalty differentiable w.r.t. those parameters.
        """
        x = encoder_output.detach().requires_grad_(True)
        grads = self._critic_input_grad(x, encoder_lens)  # [B, T, F]
        grad_norm = grads.norm(2, dim=-1)  # per-frame gradient norm over the feature dim -> [B, T]
        valid = _valid_frame_mask(encoder_lens.to(grad_norm.device), grad_norm.shape[1])
        return ((grad_norm[valid] - 1.0) ** 2).mean()

    def interpolated_gradient_penalty(
        self, real_states: Tensor, real_lens: Tensor, fake_states: Tensor, fake_lens: Tensor
    ) -> Tensor:
        """Textbook WGAN-GP gradient penalty at points interpolated between real and fake; scalar.

        This is the faithful WGAN-GP (Gulrajani et al.): sample ``x_hat = alpha*real + (1-alpha)*fake``
        per sequence and penalize ``(||d D(x_hat)/d x_hat||_2 - 1)^2``. It needs both modalities'
        encoder states in the same forward, which only happens with **mixed batches** (i.e. with
        ``alternate_batching`` turned off). ``real`` = text states, ``fake`` = audio states.

        Real/fake generally differ in batch size and length, so (following fairseq's wav2vec-U) both
        are truncated to the common ``min`` shape before interpolating; the interpolated frames are
        all treated as valid (no padding). ``create_graph=True`` keeps the penalty differentiable
        w.r.t. the discriminator parameters (the inputs are detached, so the encoder gets no grad).
        """
        b = min(real_states.size(0), fake_states.size(0))
        t = min(real_states.size(1), fake_states.size(1))
        real = real_states[:b, :t].detach()
        fake = fake_states[:b, :t].detach()
        alpha = torch.rand(b, 1, 1, device=real.device, dtype=real.dtype)
        interpolates = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        lens = torch.full((b,), t, device=real.device, dtype=torch.long)
        grads = self._critic_input_grad(interpolates, lens)  # [b, t, F]
        grad_norm = grads.norm(2, dim=-1)  # per-frame gradient norm over the feature dim -> [b, t]
        return ((grad_norm - 1.0) ** 2).mean()

    def _critic_input_grad(self, inp: Tensor, lens: Tensor) -> Tensor:
        """``d(sum_i D(inp)_i) / d inp`` with ``create_graph=True`` (for the WGAN-GP double backward).

        cuDNN's fused RNN has no double-backward implementation, so an LSTM critic crashes when
        RETURNN later backpropagates the penalty (``NotImplementedError: ... _cudnn_rnn_backward``).
        Running the discriminator forward with the cuDNN backend disabled makes autograd record the
        native (double-backward-capable) RNN backward instead. This is scoped to the *penalty* forward
        only -- once per discriminator step, over the small discriminator -- while the main critic
        forwards for the Wasserstein loss keep cuDNN. It is a no-op for the MLP discriminators (they
        have no cuDNN RNN/conv), so it is safe to apply unconditionally.
        """
        with torch.backends.cudnn.flags(enabled=False):
            disc_out = self.forward(inp, lens)  # [M]
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
    """Frame-wise / n-gram MLP discriminator over the (shared) encoder states.

    With ``num_frames == 1`` each encoder frame is classified independently. With ``num_frames > 1``,
    ``num_frames`` consecutive frames are concatenated in the feature dim (a sliding window, i.e. a
    bigram/3-gram/4-gram) before the MLP, giving the discriminator local temporal context.

    ``pad_windows`` chooses the boundary handling (see ``_ngram_windows``): ``False`` (default) keeps
    only windows fully inside a sequence; ``True`` right-pads so every real frame starts a window.
    """

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
        )  # [M, in_dim * num_frames]
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))
        x = self.layers[-1](x)  # [M, 1]
        return x.squeeze(-1)  # [M]


class LstmDiscriminator(_Discriminator):
    """LSTM discriminator that consumes the whole encoder output sequence.

    A (by default bidirectional) LSTM runs over the entire sequence, so every frame's decision is
    informed by the full temporal context; a per-frame linear head then produces one logit per valid
    frame. Unlike the (n-gram) MLP this is not limited to a fixed local window.
    """

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
        out_padded, _ = pad_packed_sequence(out, batch_first=True)  # [B, T', hidden_dim * num_directions]
        logits = self.out(out_padded).squeeze(-1)  # [B, T']
        mask = _valid_frame_mask(encoder_lens.to(logits.device), out_padded.shape[1])
        return logits[mask]  # [M]


# maps the model's ``discriminator_type`` string to the n-gram window size of the MLP discriminator.
_MLP_DISCRIMINATOR_NGRAM = {"mlp": 1, "mlp_2gram": 2, "mlp_3gram": 3, "mlp_4gram": 4}


class GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax, as used by wav2vec 2.0 / SpeechT5.

    Self-contained reimplementation of ``fairseq.modules.GumbelVectorQuantizer`` (fairseq is not
    available in the RETURNN runtime env). See
    https://github.com/microsoft/SpeechT5/blob/main/SpeechT5/speecht5/models/speecht5.py

    Args:
        dim: input dimension (channels) of the states to quantize.
        num_vars: number of codebook entries V per group.
        temp: gumbel-softmax temperature schedule, a tuple ``(start, end, decay)``.
        groups: number of groups G of latent variables.
        combine_groups: whether to share the codebook across groups.
        vq_dim: dimensionality of the resulting quantized vector (must be divisible by ``groups``).
        time_first: if True, expect/return input as ``[B, T, C]`` (otherwise ``[B, C, T]``).
        activation: activation between projection layers (only used if ``weight_proj_depth > 1``).
        weight_proj_depth: number of layers projecting the input before computing the logits.
        weight_proj_factor: scales the inner dim of the projection (only if depth > 1).
    """

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

        if self.training:
            x = nn.functional.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.reshape(bsz * tsz, self.groups, self.num_vars)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)
        vars = vars.reshape(self.groups, self.num_vars, -1)  # [groups, num_vars, var_dim]

        # combine the (one-hot in eval / gumbel-soft in train) selection with the codebook vectors.
        # Equivalent to fairseq's `x.unsqueeze(-1) * vars` broadcast + sum over num_vars, but done
        # as a per-group matmul to avoid materializing the [B*T, groups*num_vars, var_dim]
        # intermediate -- that intermediate is what OOMs for long, unsubsampled token sequences.
        x = torch.einsum("bgv,gvd->bgd", x, vars)  # [B*T, groups, var_dim]
        x = x.reshape(bsz, tsz, -1)  # [B, T, groups*var_dim = vq_dim]

        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        result["x"] = x

        return result


# defaults for `Model(codebook_opts=...)`; a passed dict overrides individual keys.
_DEFAULT_CODEBOOK_OPTS = {
    "codebook_prob": 0.5,  # fraction of time steps replaced by their quantized code
    "latent_vars": 320,  # codebook entries V per group
    "latent_groups": 2,  # number of groups G
    "latent_dim": 0,  # quantized dim; 0 -> use the encoder dim
    "latent_temp": (2.0, 0.5, 0.999995),  # gumbel temperature (start, end, decay)
    "quantizer_depth": 1,
    "quantizer_factor": 3,
}


class Model(nn.Module, SharedDenoisingAedModel, EncoderDecoderModel):
    """
    Conformer encoder + Transformer decoder AED + CTC model
    similar to the RETURNN frontend implementation but using primitives from i6_models.

    Uses:
        - `RasrCompatibleLogMelFeatureExtractionV1` for feature extraction,
        - `VGG4LayerActFrontendV1` as convolutional frontend,
        - `ConformerRelPosEncoderV1` as encoder and
        - `TransformerDecoderV1` as decoder.
    """

    def __init__(
        self,
        *,
        # Model Size/Structure
        model_dim: int,
        text_out_dim: Optional[int],
        audio_out_dim: Optional[int],
        num_heads: int,
        num_enc_layers: int,
        num_text_dec_layers: int,
        num_audio_dec_layers: int,
        text_aux_loss_layers: Sequence[int] = (),
        audio_aux_loss_layers: Sequence[int] = (),
        # RF Defaults
        dropout: float = 0.1,
        dropout_broadcast_axes: Optional[Literal["B", "BT", "T"]] = "BT",
        logits_bias: bool = False,
        aux_logits_bias: bool = False,
        enc_bottleneck_dim: Optional[int] = None,
        share_decoder: bool = False,
        discriminator_type: Optional[str] = None,
        # for the n-gram MLP discriminators ("mlp_2gram"/etc.): if True, right-pad each sequence so
        # every real frame starts a window (equal per-frame coverage) instead of dropping the windows
        # that would overlap padding. Off by default; ignored for the "mlp" (1-gram) and "lstm" types.
        discriminator_pad_ngram_windows: bool = False,
        # If given, add a GumbelVectorQuantizer codebook on top of the (shared) encoder, à la
        # SpeechT5, to push the audio and text encoder states into a shared discrete space. The
        # dict may override any of the keys in `_DEFAULT_CODEBOOK_OPTS` (an empty dict uses all
        # defaults); None disables the codebook.
        codebook_opts: Optional[Dict[str, Any]] = None,
        fix_decode_text_seq_for_shared_dec: bool = False,
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

        # positional embedding like RF
        rel_pos_clip = 16
        pos_emb_dropout = 0.1
        learnable_pos_emb = True
        with_linear_pos = False
        with_pos_bias = False
        separate_pos_emb_per_head = False

        # RF does not broadcast attention dropout masks
        attn_dropout_broadcast = None

        self.audio_mask_idx = audio_out_dim
        self.audio_bos_idx = audio_out_dim + 1
        self.audio_eos_idx = audio_out_dim + 2
        self.audio_out_dim = audio_out_dim + 3

        self.text_mask_idx = text_out_dim
        self.text_bos_idx = text_out_dim + 1
        self.text_eos_idx = text_out_dim + 2
        self.text_out_dim = text_out_dim + 3

        # in the end, we want to use the model for ASR, so the default indices are text
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
                # this is not applied to attention weights, whose dropout is not broadcast
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
        self.text_embedding = nn.Embedding(self.text_out_dim, model_dim)
        self.audio_embedding = nn.Embedding(self.audio_out_dim, model_dim)
        self.share_decoder = share_decoder
        # self.audio_eos_predictor = nn.Linear(model_dim, 1)  # binary classifier for audio eos

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

        # domain-adversarial discriminator on the shared encoder output. Options:
        #   None                  -> no discriminator
        #   "mlp"                 -> frame-wise MLP (1-gram)
        #   "mlp_2gram/3gram/4gram" -> MLP over 2/3/4 concatenated consecutive frames
        #   "lstm"                -> LSTM over the whole encoder output sequence
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

        # codebook (GumbelVectorQuantizer) on top of the shared encoder output
        if codebook_opts is not None:
            codebook_opts = {**_DEFAULT_CODEBOOK_OPTS, **codebook_opts}
            self.codebook_prob = codebook_opts["codebook_prob"]
            vq_dim = codebook_opts["latent_dim"] if codebook_opts["latent_dim"] > 0 else encoder_dim
            assert vq_dim == encoder_dim, (
                f"quantized dim ({vq_dim}) must match the encoder output dim ({encoder_dim}) so the"
                " quantized states can replace the encoder output"
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
        # result dict (prob_perplexity, num_vars, ...) of the most recent quantizer call, read by
        # the train_step to add the codebook diversity loss.
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

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def _maybe_quantize(self, encoder_output: Tensor) -> Tensor:
        """
        Apply the GumbelVectorQuantizer codebook on top of the encoder output (if enabled).

        Following SpeechT5, a random fraction (``codebook_prob``) of time steps is replaced by its
        quantized code; the rest keep the original encoder state. The quantizer metrics (used for
        the diversity loss) are stashed in ``self.quantizer_out`` for the train_step to pick up.
        """
        if self.quantizer is None:
            return encoder_output
        q = self.quantizer(encoder_output)  # [B, T, F] -> dict, q["x"]: [B, T, F]
        self.quantizer_out = {k: v for k, v in q.items() if k != "x"}
        quantized = q["x"]
        time_size = quantized.shape[1]
        num_replace = int(time_size * self.codebook_prob)
        # same time positions are replaced across the whole batch (as in SpeechT5)
        q_w = quantized.new_zeros(time_size)
        if num_replace > 0:
            replace_idx = torch.randperm(time_size, device=quantized.device)[:num_replace]
            q_w[replace_idx] = 1.0
        q_w = q_w.view(1, time_size, 1)
        return q_w * quantized + (1.0 - q_w) * encoder_output

    def forward(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # by default, forward as audio, since this is the final task for the model (ASR)
        return self.forward_audio(indices, seq_lens)

    def forward_text(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        Args:
            text_indices: (B, T)
            seq_lens: (B,)

        Returns:

        """
        data = self.text_embedding(indices)  # (B, T, F)
        data_mask = torch.less(torch.arange(data.shape[-2], device=data.device)[None, :], seq_lens[:, None])
        encoder_outputs, out_mask = self.encoder.forward(data, data_mask, return_layers=self._out_fetch_layers_text)
        assert len(self.out_text_aux_logits) <= len(encoder_outputs)
        out_aux_logits = [
            aux_linear(aux_out) for aux_linear, aux_out in zip(self.out_audio_aux_logits, encoder_outputs)
        ]
        out_seq_lens = out_mask.sum(dim=-1)
        return self._maybe_quantize(encoder_outputs[-1]), out_aux_logits, out_seq_lens, out_mask

    def forward_audio(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        Args:
            text_indices: (B, T)
            seq_lens: (B,)

        Returns:

        """
        data = self.audio_embedding(indices)  # (B, T, F)
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
            dec_out = dec_out[..., : self.text_out_dim]  # take only text part

        return dec_out

    def decode_audio_seq(
        self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor
    ) -> Tensor:
        state = self.decoder.transform_encoder_output(
            encoder_output, encoder_output_lens, self.audio_decoder.get_initial_state()
        )
        if self.share_decoder:
            x += self.text_out_dim  # shift to audio output indices
        dec_out, _ = self.audio_decoder.forward(x, x_lens, state)
        if self.share_decoder:
            dec_out = dec_out[:, :, self.text_out_dim :]  # take only audio part

        return dec_out

    def decode_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        # by default, decode as text, since this is the final task for the model (ASR)
        return self.decode_text_seq(x, x_lens, encoder_output, encoder_output_lens)

    def forward_encoder(
        self, indices: Tensor, indices_lens: Tensor, decoder: TransformerDecoderV1, forward_func: Callable
    ) -> TransformerDecoderV1State:
        encoder_output, aux_logits, encoder_lens, _ = forward_func(indices, indices_lens)
        state = decoder.get_initial_state()
        state = decoder.transform_encoder_output(encoder_output.unsqueeze(1), encoder_lens.unsqueeze(1), state)
        return state

    def step_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State, decoder: TransformerDecoderV1
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        raise NotImplementedError
        # return decoder.forward(
        #     labels,
        #     torch.full(labels.shape[:-1], 1, device=labels.device, dtype=torch.int32),
        #     state,
        # )

    def step_audio_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        if self.share_decoder:
            labels = labels + self.text_out_dim  # shift to audio output indices
        dec_out, dec_state = self.audio_decoder.forward(
            labels,
            torch.full(labels.shape[:-1], 1, device=labels.device, dtype=torch.int32),
            state,
        )
        if self.share_decoder:
            dec_out = dec_out[..., self.text_out_dim :]  # take only audio part (B, beam, T, audio_out_dim)
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
            dec_out = dec_out[..., : self.text_out_dim]  # take only text part (B, beam, T, text_out_dim)
        return dec_out, dec_state
