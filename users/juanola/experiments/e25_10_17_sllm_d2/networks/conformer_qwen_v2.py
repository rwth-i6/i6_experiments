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

def combine_batch_and_beam_v2(state, *, batch_size: int, beam_size: int):
    if not isinstance(state, Tensor):
        return state

    if state.size(1) == 1:  # first step, expand to beam.
        state = state.expand(batch_size, beam_size, *state.shape[2:])

    return state.reshape(batch_size * beam_size, *state.shape[2:])