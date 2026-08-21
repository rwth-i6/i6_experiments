__all__ = ["Model"]

import math
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union, Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from i6_models.assemblies.transformer.transformer_decoder_v1 import (
    TransformerDecoderV1,
    TransformerDecoderV1State,
)

from .conformer_aed_discrete_shared_v1 import Model as AEDModel


class Model(AEDModel):
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.decoder = None
        self.text_decoder = None
        self.audio_decoder = None

    def print_param_summary(self):
        num_enc_params = 0
        num_train_enc_params = 0
        for param in self.encoder.parameters():
            num_enc_params += param.numel()
            if param.requires_grad:
                num_train_enc_params += param.numel()

        num_text_emb_params = 0
        num_train_text_emb_params = 0
        for param in self.text_embedding.parameters():
            num_text_emb_params += param.numel()
            if param.requires_grad:
                num_train_text_emb_params += param.numel()

        num_audio_emb_params = 0
        num_train_audio_emb_params = 0
        for param in self.audio_embedding.parameters():
            num_audio_emb_params += param.numel()
            if param.requires_grad:
                num_train_audio_emb_params += param.numel()

        num_total_params = 0
        num_train_params = 0
        for param in self.parameters():
            num_total_params += param.numel()
            if param.requires_grad:
                num_train_params += param.numel()

        print(f"#enc_params: {num_enc_params} ({num_train_enc_params} trainable)")
        print(f"#text_emb_params: {num_text_emb_params} ({num_train_text_emb_params} trainable)")
        print(f"#audio_emb_params: {num_audio_emb_params} ({num_train_audio_emb_params} trainable)")
        print(f"#total_params: {num_total_params} ({num_train_params} trainable)")

    def freeze_params(
        self,
        freeze_list: List[str],
    ):
        import re

        for name, param in self.named_parameters():
            if any(re.search(match, name) for match in freeze_list):
                print(f"Freezing parameter: {name}")
                param.requires_grad = False

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def decode_text_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        raise NotImplementedError

    def decode_audio_seq(
        self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def decode_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        raise NotImplementedError

    def forward_encoder(
        self, indices: Tensor, indices_lens: Tensor, decoder: TransformerDecoderV1, forward_func: Callable
    ) -> TransformerDecoderV1State:
        raise NotImplementedError

    def step_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State, decoder: TransformerDecoderV1
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        raise NotImplementedError

    def step_audio_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        raise NotImplementedError

    def step_text_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        raise NotImplementedError
