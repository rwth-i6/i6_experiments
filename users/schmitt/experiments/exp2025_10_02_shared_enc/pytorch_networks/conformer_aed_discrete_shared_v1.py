__all__ = ["Model"]

import math
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union, Any

import torch
from torch import Tensor, nn

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

from i6_experiments.users.schmitt.experiments.exp2025_10_02_shared_enc.recognition.aed import EncoderDecoderModel
from i6_experiments.users.schmitt.experiments.exp2025_10_02_shared_enc.training.aed_denoising_discrete_shared import \
    SharedDenoisingAedModel


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


class MlpDiscriminator(nn.Module):
    def __init__(self, in_dim: int, num_layers: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim if i < num_layers - 1 else 1)
            for i in range(num_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


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
            # RETURNN get_model parameters
            epoch: int,
            step: int,
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
            **_kwargs_unused,
    ):
        super().__init__()

        assert model_dim > 0
        assert len(text_aux_loss_layers) == len(set(text_aux_loss_layers))
        assert list(text_aux_loss_layers) == sorted(text_aux_loss_layers)
        assert len(audio_aux_loss_layers) == len(set(audio_aux_loss_layers))
        assert list(audio_aux_loss_layers) == sorted(audio_aux_loss_layers)
        assert not share_decoder or (num_text_dec_layers == num_audio_dec_layers), \
            "when sharing decoder, number of text and audio decoder layers must be equal"

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
                num_blocks=num_text_dec_layers if share_decoder else (num_text_dec_layers if name == "text" else num_audio_dec_layers),
                num_output=(self.text_out_dim + self.audio_out_dim) if share_decoder else (self.text_out_dim if name == "text" else self.audio_out_dim),
                logits_bias=logits_bias,
                share_embedding=False,
            ) for name in (["shared"] if share_decoder else ["text", "audio"])
        }
        if share_decoder:
            self.decoder = TransformerDecoderV1(dec_cfgs["shared"])
            self.text_decoder = self.decoder
            self.audio_decoder = self.decoder
        else:
            self.text_decoder = TransformerDecoderV1(dec_cfgs["text"])
            self.audio_decoder = TransformerDecoderV1(dec_cfgs["audio"])

        assert discriminator_type == "mlp" or discriminator_type is None
        if discriminator_type == "mlp":
            self.discriminator = MlpDiscriminator(in_dim=encoder_dim)
        else:
            self.discriminator = None

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

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

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
        out_aux_logits = [aux_linear(aux_out) for aux_linear, aux_out in zip(self.out_audio_aux_logits, encoder_outputs)]
        out_seq_lens = out_mask.sum(dim=-1)
        return encoder_outputs[-1], out_aux_logits, out_seq_lens, out_mask

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
        return encoder_outputs[-1], out_aux_logits, out_seq_lens, out_mask

    def decode_text_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        state = self.decoder.transform_encoder_output(
            encoder_output, encoder_output_lens, self.text_decoder.get_initial_state()
        )
        dec_out, _ = self.text_decoder.forward(x, x_lens, state)

        return dec_out

    def decode_audio_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        state = self.decoder.transform_encoder_output(
            encoder_output, encoder_output_lens, self.audio_decoder.get_initial_state()
        )
        if self.share_decoder:
            x += self.text_out_dim  # shift to audio output indices
        dec_out, _ = self.audio_decoder.forward(x, x_lens, state)
        if self.share_decoder:
            dec_out = dec_out[:, :, self.text_out_dim:]  # take only audio part

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
            dec_out = dec_out[:, :, :, self.text_out_dim:]  # take only audio part (B, beam, T, audio_out_dim)
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
            dec_out = dec_out[:, :, :, :self.text_out_dim]  # take only text part (B, beam, T, text_out_dim)
        return dec_out, dec_state
