__all__ = ["SllmV2"]

from functools import partial
from typing import Tuple

import torch
import tree
from torch import Tensor
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .conformer_qwen_v1 import Model, separate_batch_and_beam
from .qwen2_decoder_state import Qwen2DecoderState

class SllmV2(Model):
    """
    New Version. Fixes Beam search and how past_key_values are handled.
    """

    def step_decoder(self, labels: Tensor, state: Qwen2DecoderState) -> Tuple[Tensor, Qwen2DecoderState]:
        """
        Perform a decoder step (for inference) -> only one new label prediction
        :type labels: Tensor - Previous generated labels
        :param labels: [Batch, Beam, Time=1]
        :param state: Decoder state
        :returns: decoder output [Batch, Beam, Time=1, L]
        """
        qwen_input_embeds = self.decoder_embed_func(labels)
        # print("****qwen_input_embeds size", qwen_input_embeds.size())
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
                dim=-2,  # time dim
            )  # (B, beam, T+l, F)
            B, beam, T, F = qwen_input_embeds.shape  # noqa
        else:  # Others
            assert tree.is_nested(past_key_values)  # e.g., transformers.cache_utils.DynamicCache, isn't supported by tree.
            past_key_values = tree.map_structure(
                partial(combine_batch_and_beam_v2, batch_size=B, beam_size=beam),
                past_key_values,
            )  # [B*b,T+l,F]

        # Decoder Forward pass
        qwen_output: CausalLMOutputWithPast = self.decoder(
            inputs_embeds=qwen_input_embeds.view(B * beam, T, F),
            past_key_values=DynamicCache.from_legacy_cache(past_key_values),
            logits_to_keep=1,  # Only 1 step!
            use_cache=True,
        )

        # Update and return new state
        past_key_values = qwen_output.past_key_values
        if not tree.is_nested(past_key_values):
            # e.g., transformers.cache_utils.DynamicCache, aren't supported by tree.
            past_key_values = past_key_values.to_legacy_cache()
        past_key_values = tree.map_structure(
            partial(separate_batch_and_beam, batch_size=B, beam_size=beam), past_key_values
        )
        new_state = {
            "input_embeds": None,
            "past_key_values": past_key_values,  # [B,b,T+l,F]
        }

        return qwen_output.logits.view(B, beam, 1, -1), new_state

    def lm_step_decoder(self, labels: Tensor, state: Qwen2DecoderState) -> Tuple[Tensor, Qwen2DecoderState]:
        """
        Perform a decoder step (for inference) -> only one new label prediction
        :type labels: Tensor - Previous generated labels
        :param labels: [Batch, Beam, Time=1]
        :param state: Decoder state
        :returns: decoder output [Batch, Beam, Time=1, L]
        """
        qwen_input_embeds = self.decoder_embed_func(labels)
        # print("****qwen_input_embeds size", qwen_input_embeds.size())
        B, beam, T, F = qwen_input_embeds.shape  # noqa

        past_key_values = state["past_key_values"]
        if past_key_values is not None:
            assert tree.is_nested(past_key_values)  # e.g., transformers.cache_utils.DynamicCache, isn't supported by tree.
            past_key_values = tree.map_structure(
                partial(combine_batch_and_beam_v2, batch_size=B, beam_size=beam),
                past_key_values,
            )  # [B*b,T+l,F]

        # Decoder Forward pass
        qwen_output: CausalLMOutputWithPast = self.decoder.forward(
            inputs_embeds=qwen_input_embeds.view(B * beam, T, F),
            past_key_values=DynamicCache.from_legacy_cache(past_key_values),
            logits_to_keep=1,  # Only 1 step!
            use_cache=True,
        )

        # Update and return new state
        past_key_values = qwen_output.past_key_values
        if not tree.is_nested(past_key_values):
            # e.g., transformers.cache_utils.DynamicCache, aren't supported by tree.
            past_key_values = past_key_values.to_legacy_cache()
        past_key_values = tree.map_structure(
            partial(separate_batch_and_beam, batch_size=B, beam_size=beam), past_key_values
        )
        new_state = {
            "input_embeds": None,
            "past_key_values": past_key_values,  # [B,b,T+l,F]
        }

        return qwen_output.logits.view(B, beam, 1, -1), new_state

    def lm_decode_seq(self, text_tokens: Tensor, text_tokens_lens: Tensor) -> Tensor:
        """
        Main decoder forward function. (for training)

        :param x: labels [B, MaxTextLen]
        :param x_lens: labels [B]
        :param encoder_output: encoder output [B, T, F]
        :returns: decoder output [B, x_lens.max(), VocabSize]
        """
        device = text_tokens.device
        input_target_embeddings = self.decoder_embed_func(text_tokens)  # [B, L] -> [B, L, F]
        input_target_lens = text_tokens_lens[:, None].expand(-1, input_target_embeddings.size(1))

        # Build attention mask
        qwen_input_lens_range = torch.range(0, input_target_embeddings.size(1) - 1)[None].expand(input_target_lens.size(0), -1)
        qwen_attention_mask = qwen_input_lens_range.to(device) < input_target_lens.to(device)

        # Decoder step
        qwen_output: CausalLMOutputWithPast = self.decoder.forward(
            inputs_embeds=input_target_embeddings,
            attention_mask=qwen_attention_mask,
            logits_to_keep=text_tokens_lens.max().item(),
        )

        return qwen_output.logits  # [B, x_lens.max(), VocabSize]

def combine_batch_and_beam_v2(state, *, batch_size: int, beam_size: int):
    if not isinstance(state, Tensor):
        return state

    if state.size(1) == 1:  # first step, expand to beam.
        state = state.expand(batch_size, beam_size, *state.shape[2:])

    return state.reshape(batch_size * beam_size, *state.shape[2:])