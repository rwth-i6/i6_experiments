import torch
import tree
from speech_llm.prefix_lm.model.definitions.speech_lm import SpeechLmV2
from transformers.cache_utils import DynamicCache


def encode_audio_to_encoder_states(model: SpeechLmV2, data: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        encoder_states, _ = SpeechLlmOnnxEncoder(model)(data, seq_len)
    return encoder_states


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


def _kv_cache_names(model: SpeechLmV2):
    num_layers = model.decoder.qwen_config.num_hidden_layers
    state_input_names = []
    state_output_names = []
    for layer_idx in range(num_layers):
        for kind in ("key", "value"):
            state_input_names.append(f"past_key_values.{layer_idx}.{kind}")
            state_output_names.append(f"present_key_values.{layer_idx}.{kind}")
    return state_input_names, state_output_names


class SpeechLmEncoder(SpeechLmV2):
    def forward(self, raw_audio: torch.Tensor, raw_audio_lens: torch.Tensor):
        encoder_states, _, _, _, _ = super().forward(raw_audio, raw_audio_lens)
        return encoder_states.float()


class SpeechLmInitializer(SpeechLmV2):
    def forward(self, initial_prompt: torch.Tensor, encoder_states: torch.Tensor, suffix_prompt: torch.Tensor):
        initial_prompt = initial_prompt.to(torch.long)
        suffix_prompt = suffix_prompt.to(torch.long)
        batch_size = encoder_states.size(0)

        encoder_adapted = self.adapter(encoder_states)

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
        return (scores,) + _flatten_legacy_cache(past_key_values)


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
