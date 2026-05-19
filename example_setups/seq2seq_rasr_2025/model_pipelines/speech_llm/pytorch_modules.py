import torch
import tree
from speech_llm.prefix_lm.model.definitions.speech_lm import SpeechLmV2
from transformers.cache_utils import DynamicCache


def _dynamic_cache_to_legacy_cache(dynamic_cache: DynamicCache) -> tuple:
    if hasattr(dynamic_cache, "to_legacy_cache"):
        return dynamic_cache.to_legacy_cache()

    legacy_cache = ()
    for layer in dynamic_cache.layers:
        legacy_cache += ((layer.keys, layer.values),)
    return legacy_cache


def _flatten_legacy_cache(past_key_values):
    flat = []
    for key, value in past_key_values:
        flat.extend([key, value])
    return tuple(flat)


def _legacy_cache_from_flat(flat_cache):
    return tuple((flat_cache[i], flat_cache[i + 1]) for i in range(0, len(flat_cache), 2))


def _pad_time_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    if multiple <= 1:
        return x

    time = x.size(1)
    pad_len = (multiple - time % multiple) % multiple
    pad = x.new_zeros((x.size(0), pad_len, x.size(2)))
    return torch.cat([x, pad], dim=1)


class SpeechLmEncoder(SpeechLmV2):
    def forward(self, raw_audio: torch.Tensor, raw_audio_lens: torch.Tensor):
        raw_audio = raw_audio.squeeze(dim=2)
        encoder_output, _, _, _ = self.encoder.forward(raw_audio, raw_audio_lens)
        return encoder_output.float()


class SpeechLmInitializer(SpeechLmV2):
    def forward(
        self,
        initial_prompt: torch.Tensor,
        encoder_states: torch.Tensor,
        encoder_states_size: torch.Tensor,
        suffix_prompt: torch.Tensor,
    ):
        initial_prompt = initial_prompt.to(torch.long)
        suffix_prompt = suffix_prompt.to(torch.long)
        batch_size = encoder_states.size(0)

        encoder_states = _pad_time_to_multiple(encoder_states, self.adapter.downsampling_factor)
        encoder_adapted = self.adapter(encoder_states)
        encoder_adapted_size = self.adapter.get_output_lengths(encoder_states_size.long())

        initial_prompt_embeds = self.decoder.embed_func(initial_prompt).expand(batch_size, -1, -1)
        suffix_prompt_embeds = self.decoder.embed_func(suffix_prompt).expand(batch_size, -1, -1)
        input_embeds = torch.cat([initial_prompt_embeds, encoder_adapted, suffix_prompt_embeds], dim=1)
        seq_len = input_embeds.size(1)
        positions = torch.arange(seq_len, device=input_embeds.device)
        attention_mask = torch.zeros(
            (1, 1, seq_len, seq_len),
            dtype=input_embeds.dtype,
            device=input_embeds.device,
        )
        attention_mask = attention_mask.masked_fill(
            positions[None, :] > positions[:, None],
            torch.finfo(input_embeds.dtype).min,
        )
        position_ids = positions.unsqueeze(0).expand(batch_size, -1)

        output = self.decoder.call_func(
            inputs_embeds=input_embeds,
            use_cache=True,
            logits_to_keep=1,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        scores = -torch.nn.functional.log_softmax(output.logits[:, -1, :].float(), dim=-1)
        past_key_values = output.past_key_values
        if not tree.is_nested(past_key_values):
            past_key_values = _dynamic_cache_to_legacy_cache(past_key_values)
        return (scores, encoder_adapted_size) + _flatten_legacy_cache(past_key_values)


class SpeechLmStep(SpeechLmV2):
    def forward(self, token: torch.Tensor, prefix_length: torch.Tensor, *past_key_values_flat: torch.Tensor):
        token = token.to(torch.long)
        prefix_length = prefix_length.to(torch.long)
        past_key_values = _legacy_cache_from_flat(past_key_values_flat)

        batch_size = token.size(0)
        max_prefix_length = past_key_values_flat[0].size(2)
        prefix_positions = torch.arange(max_prefix_length, device=prefix_length.device).unsqueeze(0)
        valid_prefix_mask = prefix_positions >= (max_prefix_length - prefix_length.unsqueeze(1))
        input_embeds = self.decoder.embed_func(token)
        masked_prefix = torch.where(
            valid_prefix_mask,
            torch.zeros((), dtype=input_embeds.dtype, device=input_embeds.device),
            torch.full(
                (),
                torch.finfo(input_embeds.dtype).min,
                dtype=input_embeds.dtype,
                device=input_embeds.device,
            ),
        )
        attention_mask = torch.cat(
            [
                masked_prefix,
                torch.zeros(
                    (batch_size, 1),
                    dtype=input_embeds.dtype,
                    device=input_embeds.device,
                ),
            ],
            dim=1,
        ).view(batch_size, 1, 1, max_prefix_length + 1)

        output = self.decoder.call_func(
            past_key_values=DynamicCache(past_key_values),
            inputs_embeds=input_embeds,
            use_cache=True,
            logits_to_keep=1,
            attention_mask=attention_mask,
            position_ids=prefix_length[:, None],
        )
        scores = -torch.nn.functional.log_softmax(output.logits[:, -1:, :].float(), dim=-1)
        past_key_values = output.past_key_values
        if not tree.is_nested(past_key_values):
            past_key_values = _dynamic_cache_to_legacy_cache(past_key_values)
        past_key_values = _flatten_legacy_cache(past_key_values)
        past_key_values = tuple(val[:, :, -1:, :] for val in past_key_values)
        return (scores,) + past_key_values


class SpeechLmCtc(SpeechLmV2):
    def forward(self, encoder_state: torch.Tensor) -> torch.Tensor:
        return -torch.nn.functional.log_softmax(self.out_aux_logits[-1](encoder_state), dim=-1)
