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
from i6_models.parts.masked_norm import MaskedBatchNorm1dV1

from i6_experiments.users.schmitt.experiments.exp2025_10_02_shared_enc.recognition.aed import (
    EncoderDecoderModel,
)
from i6_experiments.users.schmitt.experiments.exp2025_10_02_shared_enc.training.aed_denoising_discrete_shared import (
    SharedDenoisingAedModel,
)

from .conformer_aed_discrete_shared_v1 import (
    _relu_sq,
    _SpecAugArgs,
    ConformerEncoderWithBottleneck,
    _valid_frame_mask,
    _ngram_windows,
    _Discriminator,
    MlpDiscriminator,
    LstmDiscriminator,
    _MLP_DISCRIMINATOR_NGRAM,
    GumbelVectorQuantizer,
    _DEFAULT_CODEBOOK_OPTS,
)


class Model(nn.Module, SharedDenoisingAedModel, EncoderDecoderModel):
    """
    Conformer encoder + Transformer decoder AED model with an auxiliary self-supervised
    reconstruction head placed immediately before the last linear layer (`linear_out`)
    of the final Conformer encoder block, following the auxiliary objective idea of
    wav2vec-U 2.0 (https://arxiv.org/pdf/2204.02492).
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
        audio_aux_rec_layers: Optional[Sequence[int]] = None,
        # RF Defaults
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
        **_kwargs_unused,
    ):
        super().__init__()

        if audio_aux_rec_layers is None:
            audio_aux_rec_layers = (num_enc_layers,)
        self.audio_aux_rec_layers = tuple(audio_aux_rec_layers)

        assert model_dim > 0
        assert len(text_aux_loss_layers) == len(set(text_aux_loss_layers))
        assert list(text_aux_loss_layers) == sorted(text_aux_loss_layers)
        assert len(audio_aux_loss_layers) == len(set(audio_aux_loss_layers))
        assert list(audio_aux_loss_layers) == sorted(audio_aux_loss_layers)
        assert len(self.audio_aux_rec_layers) > 0
        assert len(self.audio_aux_rec_layers) == len(set(self.audio_aux_rec_layers))
        assert list(self.audio_aux_rec_layers) == sorted(self.audio_aux_rec_layers)
        assert all(1 <= layer_idx <= num_enc_layers for layer_idx in self.audio_aux_rec_layers), (
            f"audio_aux_rec_layers {self.audio_aux_rec_layers} must be in 1..{num_enc_layers}"
        )
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

        self.audio_vocab_size = audio_out_dim
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

        ff_hidden_dim = model_dim * 4
        block_cfg = ConformerRelPosBlockV1Config(
            ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                input_dim=model_dim,
                hidden_dim=ff_hidden_dim,
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
        self.text_embedding = nn.Embedding(self.text_out_dim, model_dim)
        self.audio_embedding = nn.Embedding(self.audio_out_dim, model_dim)
        self.share_decoder = share_decoder

        # Auxiliary linear transformation(s) placed immediately before the last linear layer (`linear_out`)
        # of each configured Conformer encoder block in `self.audio_aux_rec_layers`
        # (where hidden dim is `ff_hidden_dim = model_dim * 4`).
        self.audio_aux_rec_projs = nn.ModuleList(
            [nn.Linear(ff_hidden_dim, self.audio_vocab_size) for _ in self.audio_aux_rec_layers]
        )
        self._captured_ff_pre_linear_outs: Dict[int, Tensor] = {}

        raw_encoder = self.encoder.encoder if isinstance(self.encoder, ConformerEncoderWithBottleneck) else self.encoder
        for layer_idx in self.audio_aux_rec_layers:
            block = raw_encoder.module_list[layer_idx - 1]
            ff2 = block.module_list[-1]

            def _make_capture_hook(idx: int):
                def _capture_pre_linear_out(_module: nn.Module, args: Tuple[Tensor, ...]) -> None:
                    self._captured_ff_pre_linear_outs[idx] = args[0]

                return _capture_pre_linear_out

            ff2.linear_out.register_forward_pre_hook(_make_capture_hook(layer_idx))

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

    def compute_audio_aux_rec_logits(self) -> Tensor:
        """
        Project the hidden representation(s) captured immediately before `linear_out` of the
        configured Conformer encoder block(s) (`[B, T, 4 * model_dim]`) to `[B, T, audio_vocab_size]`.
        When multiple layers are configured in `audio_aux_rec_layers`, their logits are averaged
        before computing the softmax.
        """
        assert len(self._captured_ff_pre_linear_outs) == len(self.audio_aux_rec_layers), (
            f"Expected {len(self.audio_aux_rec_layers)} captured encoder layers, "
            f"got {sorted(self._captured_ff_pre_linear_outs.keys())}"
        )
        layer_logits = [
            proj(self._captured_ff_pre_linear_outs[layer_idx])
            for proj, layer_idx in zip(self.audio_aux_rec_projs, self.audio_aux_rec_layers)
        ]
        self._captured_ff_pre_linear_outs.clear()
        if len(layer_logits) == 1:
            return layer_logits[0]
        return torch.stack(layer_logits, dim=0).mean(dim=0)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
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
        self._captured_ff_pre_linear_outs.clear()
        data = self.text_embedding(indices)  # (B, T, F)
        data_mask = torch.less(torch.arange(data.shape[-2], device=data.device)[None, :], seq_lens[:, None])
        encoder_outputs, out_mask = self.encoder.forward(data, data_mask, return_layers=self._out_fetch_layers_text)
        self._captured_ff_pre_linear_outs.clear()  # Clear after text forward since aux rec is only for real audio
        assert len(self.out_text_aux_logits) <= len(encoder_outputs)
        out_aux_logits = [
            aux_linear(aux_out) for aux_linear, aux_out in zip(self.out_audio_aux_logits, encoder_outputs)
        ]
        out_seq_lens = out_mask.sum(dim=-1)
        return self._maybe_quantize(encoder_outputs[-1]), out_aux_logits, out_seq_lens, out_mask

    def forward_audio(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        self._captured_ff_pre_linear_outs.clear()
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
        return self.decode_text_seq(x, x_lens, encoder_output, encoder_output_lens)

    def forward_encoder(
        self, indices: Tensor, indices_lens: Tensor, decoder: TransformerDecoderV1, forward_func: Callable
    ) -> TransformerDecoderV1State:
        encoder_output, aux_logits, encoder_lens, _ = forward_func(indices, indices_lens)
        self._captured_ff_pre_linear_outs.clear()
        state = decoder.get_initial_state()
        state = decoder.transform_encoder_output(encoder_output.unsqueeze(1), encoder_lens.unsqueeze(1), state)
        return state

    def step_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State, decoder: TransformerDecoderV1
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        raise NotImplementedError

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
