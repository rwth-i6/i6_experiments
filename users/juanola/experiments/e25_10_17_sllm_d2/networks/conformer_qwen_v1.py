__all__ = ["Model"]

from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union, Any

import tree
from functools import partial
import torch
import transformers
from torch import Tensor, nn
from transformers import Qwen2Config
from transformers.modeling_outputs import CausalLMOutputWithPast

from i6_models.assemblies.conformer.conformer_rel_pos_v1 import (
    ConformerConvolutionV2Config,
    ConformerMHSARelPosV1Config,
    ConformerPositionwiseFeedForwardV2Config,
    ConformerRelPosBlockV1Config,
    ConformerRelPosEncoderV1,
    ConformerRelPosEncoderV1Config,
)
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.masked_norm import MaskedBatchNorm1dV1
from i6_models.primitives.feature_extraction import (
    LogMelFeatureExtractionV1,  # NEEDED!
    LogMelFeatureExtractionV1Config,  # NEEDED!
    RasrCompatibleLogMelFeatureExtractionV1,
    RasrCompatibleLogMelFeatureExtractionV1Config,
)
from i6_models.primitives.specaugment import specaugment_v1_by_length

from .interfaces.aed_ctc_model_protocol import AedCtcModelProtocol
from .interfaces.base_encoder_decoder_model import BaseEncoderDecoderModel
from .linear_adapter_with_concat_downsampling import LinearAdapterWithConcatDownsampling
from .qwen2_decoder_state import Qwen2DecoderState


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


class Model(nn.Module, AedCtcModelProtocol,
            BaseEncoderDecoderModel[Qwen2DecoderState]):  # TODO: rename class -> will break in hardcoded places
    """
    Conformer encoder + Transformer decoder AED + CTC model
    similar to the RETURNN frontend implementation but using primitives from i6_models.

    Uses:
        - `RasrCompatibleLogMelFeatureExtractionV1` for feature extraction,
        - `VGG4LayerActFrontendV1` as convolutional frontend,
        - `ConformerRelPosEncoderV1` as encoder and
        - DECODER -> some LLM from HF transformers

    PARAMETER WARNING!! Linked to model_configs but refactoring will not see it!
    Advice: don't rename parameters
    """

    def __init__(
            self,
            # RETURNN get_model PARAMS
            epoch: int,
            step: int,
            *,

            # FEATURE EXTRACTION PARAMS
            feature_extraction_config: Optional[Dict[str, Any]] = None,
            sampling_rate: int,
            n_mels: int = 80,
            num_enc_layers: int,

            # ENCODER PARAMS
            encoder_dim: int,
            num_heads: int,

            rel_pos_clip: int = 16,
            pos_emb_dropout: float = 0.1,
            learnable_pos_emb: bool = True,
            with_linear_pos: bool = False,
            with_pos_bias: bool = False,
            separate_pos_emb_per_head: bool = False,

            # DECODER PARAMS
            config_path: Optional[str] = None,

            # VOCAB
            vocab_size: int,  # TODO: this should not be here
            bos_idx: int,
            eos_idx: int,
            blank_idx: Optional[int] = None,

            # RF DEFAULTS
            dropout: float = 0.1,
            dropout_broadcast_axes: Optional[Literal["B", "BT", "T"]] = "BT",
            specaug_start: Union[int, Tuple[int, int, int]] = 10,
            specaug_args: Optional[Dict[str, int]] = None,

            # OTHER
            aux_loss_layers: Sequence[int],  # fot the ctc stuff
            aux_logits_bias: bool = False,

            # Added later to use only parts of the model
            using_encoder:bool = True,
            using_decoder:bool = True,

            **_kwargs_unused,
    ):
        super().__init__()

        assert encoder_dim > 0
        assert vocab_size > 0
        assert len(aux_loss_layers) == len(set(aux_loss_layers))
        assert list(aux_loss_layers) == sorted(aux_loss_layers)
        assert 0 <= bos_idx < vocab_size
        assert 0 <= eos_idx < vocab_size
        assert bos_idx != eos_idx

        # GENERAL
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.num_labels = vocab_size
        if blank_idx is not None: # blank index is part of the vocabulary
            assert 0 <= blank_idx < vocab_size
            aux_out_dim = vocab_size
            self.blank_idx = blank_idx
        else: # blank index is not part of the vocabulary, make space for it
            aux_out_dim = vocab_size + 1
            self.blank_idx = vocab_size

        self.using_encoder = using_encoder
        self.using_decoder = using_decoder

        if using_encoder:
            # FEATURE EXTRACTION (used in forward encoder)
            if feature_extraction_config is None:
                mel_cfg = RasrCompatibleLogMelFeatureExtractionV1Config(
                    sample_rate=sampling_rate, win_size=25 / 1000, hop_size=10 / 1000, min_amp=1e-4, num_filters=n_mels
                )
                self.mel_frontend = RasrCompatibleLogMelFeatureExtractionV1(mel_cfg)
            else:
                assert "class" in feature_extraction_config
                feature_extraction_class_str = feature_extraction_config.pop("class")
                feature_extraction_class = eval(feature_extraction_class_str)
                feature_extraction_config_class = eval(feature_extraction_class_str + "Config")
                mel_cfg = feature_extraction_config_class(
                    sample_rate=sampling_rate,
                    **feature_extraction_config,
                )
                self.mel_frontend = feature_extraction_class(mel_cfg)

            # DATA AUGMENTATION - Spectrogram Augmentation (used in the forward encoder)
            self.spec_aug_args: _SpecAugArgs = {
                "time_min_num_masks": 1,
                "time_max_mask_per_n_frames": 100,
                "time_mask_max_size": 20,
                "freq_min_num_masks": 1,
                "freq_max_num_masks": 2,
                "freq_mask_max_size": n_mels // 5,
            }
            if specaug_args is not None:
                self.spec_aug_args.update(specaug_args)
            self.spec_aug_start = specaug_start

            # ENCODER
            frontend = ModuleFactoryV1(
                module_class=VGG4LayerActFrontendV1,
                cfg=VGG4LayerActFrontendV1Config(
                    in_features=n_mels,
                    conv1_channels=32,
                    conv2_channels=64,
                    conv3_channels=64,
                    conv4_channels=32,
                    conv_kernel_size=(3, 3),
                    conv_padding=None,
                    pool1_kernel_size=(1, 2),
                    pool1_padding=None,
                    pool1_stride=(3, 1),
                    pool2_kernel_size=(1, 2),
                    pool2_padding=None,
                    pool2_stride=(2, 1),
                    activation=nn.functional.relu,
                    out_features=encoder_dim,
                ),
            )
            block_cfg = ConformerRelPosBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                    input_dim=encoder_dim,
                    hidden_dim=encoder_dim * 4,
                    dropout=dropout,
                    activation=_relu_sq,
                    dropout_broadcast_axes=dropout_broadcast_axes,
                ),
                mhsa_cfg=ConformerMHSARelPosV1Config(
                    input_dim=encoder_dim,
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
                    channels=encoder_dim,
                    kernel_size=33,
                    dropout=dropout,
                    dropout_broadcast_axes=dropout_broadcast_axes,
                    activation=nn.functional.silu,
                    norm=MaskedBatchNorm1dV1(encoder_dim, eps=1e-3, momentum=0.1),
                ),
                modules=["ff", "mhsa", "conv", "ff"],
                scales=[0.5, 1.0, 1.0, 0.5],
            )
            enc_cfg = ConformerRelPosEncoderV1Config(num_layers=num_enc_layers, frontend=frontend, block_cfg=block_cfg)
            self.encoder = ConformerRelPosEncoderV1(enc_cfg)

            # AUX LAYERS (CTC - used in forward encoder)
            self.out_aux_logits = nn.ModuleList(
                [nn.Linear(encoder_dim, aux_out_dim, bias=aux_logits_bias) for _ in range(len(aux_loss_layers))]
            )
            self._out_fetch_layers = sorted(v - 1 for v in {*aux_loss_layers, enc_cfg.num_layers})

        if using_decoder:
            # DECODER
            qwen2_config: Qwen2Config = transformers.Qwen2Config.from_pretrained(config_path)
            self.decoder = transformers.Qwen2ForCausalLM(qwen2_config)

            self.num_labels = qwen2_config.vocab_size

            # Embedding
            self.decoder_embed_func = nn.Embedding(vocab_size, qwen2_config.hidden_size)

            # Adapter
            self.encoder_decoder_adapter = LinearAdapterWithConcatDownsampling(
                in_dim=encoder_dim,
                out_dim=qwen2_config.hidden_size,
            )

    def _apply_spec_aug(self, data: Tensor, data_len: Tensor) -> Tensor:
        """
        Uses returnn frontend wrapper for SpecAug -> data augmentation technique applied to audio features
        (usually spectrograms).

        # `self.specaug_start` configures whether we use the standard i6_models
        # SpecAugment-by-length or the stepwise-scheduled RF implementation.
        #
        # To use the i6_models implementation, `self.specaug_start` must be an
        # integer. It then configures the first epoch that SpecAug will be applied
        # in. Starting SpecAug right in the first epoch leads to convergence
        # problems.
        #
        # When using RF SpecAug, set `self.specaug_start` to a tuple of three train
        # step indices. These represent the points at which the RF implementation
        # will increase the amount of regularization.

        :param data:
        :param data_len:
        :return:
        """
        if not self.training:
            return data

        import returnn.frontend as rf
        from returnn.tensor import Dim

        if isinstance(self.spec_aug_start, int):
            if rf.get_run_ctx().epoch < self.spec_aug_start:
                return data
            return specaugment_v1_by_length(data, **self.spec_aug_args)

        batch_dim = Dim(int(data.shape[0]), name="batch")
        data_len = rf.convert_to_tensor(data_len.cpu(), dims=[batch_dim])
        time_dim = Dim(data_len, name="time")
        feature_dim = Dim(int(data.shape[-1]), name="feature")
        data = rf.convert_to_tensor(data, dims=[batch_dim, time_dim, feature_dim], feature_dim=feature_dim)
        data_w_spec_aug = rf.audio.specaugment(
            data,
            feature_dim=feature_dim,
            spatial_dim=time_dim,
            global_train_step_dependent=True,
            steps=self.spec_aug_start,
            max_consecutive_feature_dims=self.spec_aug_args["freq_mask_max_size"],
            max_consecutive_spatial_dims=self.spec_aug_args["time_mask_max_size"],
            num_spatial_mask_factor=self.spec_aug_args["time_max_mask_per_n_frames"],
        )
        return data_w_spec_aug.raw_tensor

    def forward(self, raw_audio: Tensor, raw_audio_lens: Tensor) -> Tuple[Tensor, List[Tensor], Tensor, Tensor]:
        """
        Forward the data through ONLY the encoder. (for training)

        :param raw_audio: raw audio tensor
        :param raw_audio_lens: raw audio lens tensor
        :returns: forward result tuple containing:
            - Encoder output (Tensor) [B, T, HiddenSize]
            - Aux Logits (Tensor List?)
            - Aux Logit Lengths (Tensor)
            - Out mask ? (Tensor)
        """
        if not self.using_encoder:
            raise Exception("Trying to use forward encoder for Model without encoder!")

        raw_audio = raw_audio.squeeze(dim=2).float()
        data, seq_lens = self.mel_frontend(raw_audio, raw_audio_lens)
        data = self._apply_spec_aug(data, seq_lens)
        data_mask = torch.less(torch.arange(data.shape[-2], device=data.device)[None, :], seq_lens[:, None])

        encoder_outputs, out_mask = self.encoder.forward(data, data_mask, return_layers=self._out_fetch_layers)

        assert len(self.out_aux_logits) <= len(encoder_outputs)
        out_aux_logits = [aux_linear(aux_out) for aux_linear, aux_out in zip(self.out_aux_logits, encoder_outputs)]
        out_seq_lens = out_mask.sum(dim=-1)

        return encoder_outputs[-1], out_aux_logits, out_seq_lens, out_mask  # [-1] from aed setup...

    def decode_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        """
        Main decoder forward function. (for training)

        :param x: labels [B, MaxTextLen]
        :param x_lens: labels [B]
        :param encoder_output: encoder output [B, T, F]
        :param encoder_output_lens: encoder output lens [B]
        :returns: decoder output [B, x_lens.max(), VocabSize]
        """
        if not self.using_decoder:
            raise Exception("Trying to use forward decoder for Model without decoder!")

        qwen_audio_features_in = self.encoder_decoder_adapter(encoder_output)  # [B, T, F]

        # Setup decoder(LLM) inputs
        qwen_input_embeds, qwen_attention_mask = self.get_qwen_input_embeds(
            qwen_audio_features_in,
            x,
            x_lens)

        # Decoder step
        qwen_output: CausalLMOutputWithPast = self.decoder.forward(
            inputs_embeds=qwen_input_embeds,
            attention_mask=qwen_attention_mask,
            logits_to_keep=x_lens.max().item(),
        )

        return qwen_output.logits  # [B, x_lens.max(), VocabSize]

    def get_qwen_input_embeds(self, audio_embeds: Tensor, text_tokens: Tensor, text_tokens_lens: Tensor) \
            -> Tuple[Tensor, Tensor]:
        """
        For now only feeding the encoded audio and the text labels.
        No prompt for now!
        :param audio_embeds: [B, T, F]
        :param text_tokens: [B, L]
        :param text_tokens_lens: [B]
        :return:
        """
        device = audio_embeds.device

        # Are they divided in batches?
        input_target_embeddings = self.decoder_embed_func(text_tokens)  # [B, L] -> [B, L, F]
        qwen_input_embeds = torch.cat([audio_embeds, input_target_embeddings], dim=1)  # [B, T+L, F]

        # Compute sequence lengths
        qwen_input_lens = audio_embeds.size(1) + text_tokens_lens  # [B]
        assert text_tokens_lens.ndim == 1
        qwen_input_lens = qwen_input_lens[:, None].expand(-1, qwen_input_embeds.size(1))

        # Build attention mask
        qwen_input_lens_range = torch.range(0, qwen_input_embeds.size(1) - 1)[None].expand(qwen_input_lens.size(0), -1)
        qwen_attention_mask = qwen_input_lens_range.to(device) < qwen_input_lens.to(device)

        return qwen_input_embeds, qwen_attention_mask

    def forward_encoder(self, raw_audio: Tensor, raw_audio_lens: Tensor, initial_beam_size: int) -> Qwen2DecoderState:
        """
        Forward the raw audio data through the encoder and initialize decoder state from it. (for inference)
        batch=1 (only one encoding/decoding) || now beams in encoder (only in decoder)
        """

        # Forward through encoder
        encoder_output, _, logits_lens, _ = self.forward(raw_audio, raw_audio_lens)

        # Prepare decoder input [adapter + mix with text imput] (could be also extracted, but not needed for now)
        qwen_audio_features_in = self.encoder_decoder_adapter(encoder_output)  # [B, T', HS']

        empty_tokens = torch.empty((raw_audio.shape[0], 0), dtype=torch.long,
                                   device=qwen_audio_features_in.device)  # [B, 0]
        empty_tokens_len = torch.tensor([0], device=qwen_audio_features_in.device).expand(
            qwen_audio_features_in.size(0))  # [B]

        qwen_input_embeds, _ = self.get_qwen_input_embeds(
            qwen_audio_features_in,
            empty_tokens,
            empty_tokens_len)

        # Package results in TransformerDecoderV1State
        initial_qwen2_decoder_state = {  # TODO: extract to class as initialize method
            "input_embeds": qwen_input_embeds[:, None].expand(-1, initial_beam_size, -1, -1),  # (B, b, T, F),
            "past_key_values": None,
        }

        return initial_qwen2_decoder_state

    def step_decoder(self, labels: Tensor, state: Qwen2DecoderState) -> Tuple[Tensor, Qwen2DecoderState]:
        """
        Perform a decoder step (for inference) -> only one new label prediction
        :type labels: Tensor - Previous generated labels
        :param labels: [Batch, Beam, Time=1]
        :param state: Decoder state
        :returns: decoder output [Batch, Beam, Time=1, L]
        """
        qwen_input_embeds = self.decoder.get_input_embeddings()(labels)
        #print("****qwen_input_embeds size", qwen_input_embeds.size())
        B, beam, T, F = qwen_input_embeds.shape  # noqa

        past_key_values = state["past_key_values"]

        if past_key_values is None:  # First Iteration
            # First step (use BOS + audio context)
            qwen_input_embeds_prefix = state["input_embeds"]
            qwen_input_embeds = torch.cat(
                [
                    qwen_input_embeds_prefix,
                    qwen_input_embeds,
                ],
                dim=-2  # time dim
            )  # (B, beam, T+l, F)
            B, beam, T, F = qwen_input_embeds.shape  # noqa
        else:  # Others
            past_key_values = tree.map_structure(
                partial(combine_batch_and_beam, batch_size=B, beam_size=beam), past_key_values,
            )  # [B*b,T+l,F]

        # Decoder Forward pass
        qwen_output: CausalLMOutputWithPast = self.decoder(
            inputs_embeds=qwen_input_embeds.view(B * beam, T, F),
            past_key_values=past_key_values,
            logits_to_keep=1,  # Only 1 step!
            use_cache=True,
        )

        # Update and return new state
        past_key_values = tree.map_structure(
            partial(separate_batch_and_beam, batch_size=B, beam_size=beam), qwen_output.past_key_values
        )
        new_state = {
            "input_embeds": None,
            "past_key_values": past_key_values,  # [B,b,T+l,F]
        }

        return qwen_output.logits.view(B, beam, 1, -1), new_state


def separate_batch_and_beam(state, *, batch_size: int, beam_size: int):
    if not isinstance(state, Tensor):
        return state

    return state.view(batch_size, beam_size, *state.shape[1:])


def combine_batch_and_beam(state, *, batch_size: int, beam_size: int):
    if not isinstance(state, Tensor):
        return state

    return state.view(batch_size * beam_size, *state.shape[2:])

# print(f" XXX shape = {XXX.size()}")
