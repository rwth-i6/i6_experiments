__all__ = ["Model"]

from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from i6_models.assemblies.conformer.conformer_rel_pos_v1 import (
    ConformerPositionwiseFeedForwardV2Config,
)
from i6_models.assemblies.transformer.transformer_decoder_v1 import (
    CausalSelfAttentionV1Config,
    TransformerDecoderBlockV1Config,
    TransformerDecoderV1,
    TransformerDecoderV1Config,
    TransformerDecoderV1State,
)


class Model(nn.Module):
    """
    Decoder-only phoneme language model.

    A causal Transformer decoder *without cross attention* over phoneme label sequences, trained
    with plain next-token cross entropy. This mirrors the (text) decoder of
    ``conformer_aed_discrete_shared_v1.Model`` but drops the encoder and the cross-attention module
    (``cross_cfg=None`` / ``modules=["mhcsa", "ff"]``), so it is a standalone LM.

    Each vocab gets ``+3`` reserved ids for ``[mask, bos, eos]`` to match the AED index convention
    (``mask`` is unused here, kept only so the special-symbol arithmetic stays identical).
    """

    def __init__(
        self,
        *,
        model_dim: int,
        out_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        dropout_broadcast_axes: Optional[Literal["B", "BT", "T"]] = "BT",
        logits_bias: bool = False,
        **_kwargs_unused,
    ):
        super().__init__()

        assert model_dim > 0

        self.mask_idx = out_dim
        self.bos_idx = out_dim + 1
        self.eos_idx = out_dim + 2
        self.out_dim = out_dim + 3

        # RF does not broadcast attention dropout masks
        attn_dropout_broadcast = None

        block_cfg = TransformerDecoderBlockV1Config(
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
            # decoder-only LM: no cross attention block
            cross_cfg=None,
            modules=["mhcsa", "ff"],
            scales=[1.0, 1.0],
        )
        dec_cfg = TransformerDecoderV1Config(
            block_cfg=block_cfg,
            input_dropout=dropout,
            input_embedding_scale=None,
            num_blocks=num_layers,
            num_output=self.out_dim,
            logits_bias=logits_bias,
            share_embedding=False,
        )
        self.decoder = TransformerDecoderV1(dec_cfg)

    def score_text(self, labels: Tensor, labels_lens: Tensor) -> Tensor:
        """
        Teacher-forced logits over the whole label sequence, ``[B, T, out_dim]``.

        As there is no encoder / cross attention, we start from the decoder's initial state and run
        the sequence in one shot (``transform_encoder_output`` is a no-op for causal self attention).
        """
        state = self.decoder.get_initial_state()
        logits, _ = self.decoder.forward(labels, labels_lens, state)
        return logits

    def step_text_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        """Single-/multi-step decode, provided for symmetry with the AED model's decoder interface."""
        return self.decoder.forward(
            labels,
            torch.full(labels.shape[:-1], 1, device=labels.device, dtype=torch.int32),
            state,
        )
