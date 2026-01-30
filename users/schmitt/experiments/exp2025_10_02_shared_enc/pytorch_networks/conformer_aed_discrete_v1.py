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
from i6_experiments.users.schmitt.experiments.exp2025_10_02_shared_enc.training.aed_denoising_discrete import DenoisingAedModel


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


class Model(nn.Module, DenoisingAedModel, EncoderDecoderModel):
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
        out_dim: Optional[int],
        num_heads: int,
        num_enc_layers: int,
        num_dec_layers: int,
        aux_loss_layers: Sequence[int],
        # RF Defaults
        dropout: float = 0.1,
        dropout_broadcast_axes: Optional[Literal["B", "BT", "T"]] = "BT",
        logits_bias: bool = False,
        aux_logits_bias: bool = False,
        **_kwargs_unused,
    ):
        super().__init__()

        assert model_dim > 0
        assert len(aux_loss_layers) == len(set(aux_loss_layers))
        assert list(aux_loss_layers) == sorted(aux_loss_layers)

        # positional embedding like RF
        rel_pos_clip = 16
        pos_emb_dropout = 0.1
        learnable_pos_emb = True
        with_linear_pos = False
        with_pos_bias = False
        separate_pos_emb_per_head = False

        # RF does not broadcast attention dropout masks
        attn_dropout_broadcast = None

        self.mask_idx = out_dim
        self.bos_idx = self.mask_idx + 1
        self.eos_idx = out_dim + 2
        out_dim = out_dim + 3

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
        self.embedding = nn.Embedding(out_dim, model_dim)
        self.audio_eos_predictor = nn.Linear(model_dim, 1)  # binary classifier for audio eos

        dec_cfg = TransformerDecoderV1Config(
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
                    encoder_dim=model_dim,
                    model_dim=model_dim,
                    key_dim_total=model_dim,
                    value_dim_total=model_dim,
                    num_heads=num_heads,
                    with_bias=True,
                ),
            ),
            input_dropout=dropout,
            input_embedding_scale=None,
            num_blocks=num_dec_layers,
            num_output=out_dim,
            logits_bias=logits_bias,
            share_embedding=False,
        )
        self.decoder = TransformerDecoderV1(dec_cfg)

        self.out_aux_logits = nn.ModuleList(
            [nn.Linear(model_dim, out_dim, bias=aux_logits_bias) for _ in range(len(aux_loss_layers))]
        )
        self._out_fetch_layers = sorted(v - 1 for v in {enc_cfg.num_layers})
        self.blank_idx = out_dim

    def forward(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        Args:
            indices: (B, T)
            seq_lens: (B,)

        Returns:

        """
        data = self.embedding(indices)  # (B, T, F)
        data_mask = torch.less(torch.arange(data.shape[-2], device=data.device)[None, :], seq_lens[:, None])
        encoder_outputs, out_mask = self.encoder.forward(data, data_mask, return_layers=self._out_fetch_layers)
        assert len(self.out_aux_logits) <= len(encoder_outputs)
        out_aux_logits = [aux_linear(aux_out) for aux_linear, aux_out in zip(self.out_aux_logits, encoder_outputs)]
        out_seq_lens = out_mask.sum(dim=-1)
        return encoder_outputs[-1], out_aux_logits, out_seq_lens, out_mask

    def decode_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        state = self.decoder.transform_encoder_output(
            encoder_output, encoder_output_lens, self.decoder.get_initial_state()
        )
        dec_out, _ = self.decoder.forward(x, x_lens, state)
        return dec_out

    def forward_ctc(self, raw_audio: Tensor, raw_audio_lens: Tensor) -> Tuple[List[Tensor], Tensor]:
        _, ctc_logits, ctc_len, _ = self.forward(raw_audio, raw_audio_lens)
        return ctc_logits, ctc_len

    def forward_encoder(self, raw_audio: Tensor, raw_audio_lens: Tensor) -> TransformerDecoderV1State:
        state, *_ = self.forward_encoder_with_ctc(raw_audio, raw_audio_lens)
        return state

    def forward_encoder_with_ctc(
        self, raw_audio: Tensor, raw_audio_lens: Tensor
    ) -> Tuple[TransformerDecoderV1State, Tensor, Tensor]:
        encoder_out, ctc_logits, lens, _ = self.forward(raw_audio, raw_audio_lens)
        state = self.decoder.get_initial_state()
        state = self.decoder.transform_encoder_output(encoder_out.unsqueeze(1), lens.unsqueeze(1), state)
        return state, ctc_logits[-1], lens

    def step_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        return self.decoder.forward(
            labels,
            torch.full(labels.shape[:-1], 1, device=labels.device, dtype=torch.int32),
            state,
        )
