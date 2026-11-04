"""Qwen3 weights with the RASR ONNX transformer-state interface.

RASR supplies right-aligned prefix caches and expects ONLY new suffix K/V,
flattened to [batch,time,kv_heads*head_dim]. No chat template or sampling.
"""

import torch


class Qwen3RasrLm(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        cfg = model.config
        if cfg.model_type != "qwen3" or getattr(cfg, "use_sliding_window", False):
            raise ValueError("Only full-attention Qwen3 is supported")
        rope = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("Only default RoPE is supported")
        self.model = model.eval()
        self.num_layers = cfg.num_hidden_layers
        self.num_heads, self.kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.cache_dim = self.kv_heads * self.head_dim
        if self.num_heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("Unsupported attention dimensions")
        self.register_buffer("inv_freq", model.model.rotary_emb.inv_freq.detach().float().clone())

    @staticmethod
    def rotate_half(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)

    def forward(self, tokens, lengths, prefix_lengths, *states):
        batch, suffix = tokens.shape
        prefix = states[0].shape[1]
        pos = prefix_lengths[:, None].float() + torch.arange(suffix, device=tokens.device)[None, :].float()
        freqs = pos[:, :, None] * self.inv_freq[None, None, :]
        angles = torch.cat((freqs, freqs), dim=-1)[:, None, :, :]
        cos, sin = angles.cos(), angles.sin()
        pref_valid = torch.arange(prefix, device=tokens.device)[None, :] >= (prefix - prefix_lengths[:, None])
        steps = torch.arange(suffix, device=tokens.device)
        suffix_valid = (steps[None, None, :] <= steps[None, :, None]) & (steps[None, None, :] < lengths[:, None, None])
        valid = torch.cat((pref_valid[:, None, :].expand(-1, suffix, -1), suffix_valid), dim=-1)[:, None]
        x = self.model.model.embed_tokens(tokens.long())
        new_states = []
        for i, layer in enumerate(self.model.model.layers):
            residual = x
            h = layer.input_layernorm(x)
            att = layer.self_attn
            q = att.q_norm(att.q_proj(h).reshape(batch, suffix, self.num_heads, self.head_dim)).transpose(1, 2)
            k = att.k_norm(att.k_proj(h).reshape(batch, suffix, self.kv_heads, self.head_dim)).transpose(1, 2)
            v = att.v_proj(h).reshape(batch, suffix, self.kv_heads, self.head_dim).transpose(1, 2)
            q, k = q * cos + self.rotate_half(q) * sin, k * cos + self.rotate_half(k) * sin
            k_new = k.transpose(1, 2).reshape(batch, suffix, self.cache_dim)
            v_new = v.transpose(1, 2).reshape(batch, suffix, self.cache_dim)
            new_states.extend((k_new, v_new))
            k = (
                torch.cat((states[2 * i], k_new), dim=1)
                .reshape(batch, -1, self.kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                torch.cat((states[2 * i + 1], v_new), dim=1)
                .reshape(batch, -1, self.kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = k.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
            attention = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
            attention = attention.masked_fill(~valid, -1.0e30).softmax(dim=-1)
            h = (attention @ v).transpose(1, 2).reshape(batch, suffix, self.num_heads * self.head_dim)
            x = residual + att.o_proj(h)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        logits = self.model.lm_head(self.model.model.norm(x))
        return (-torch.log_softmax(logits.float(), dim=-1), *new_states)


def onnx_names(num_layers):
    inputs = ["tokens", "tokens:size1", "state_l000_k_in:size1"]
    outputs = ["scores"]
    metadata = {}
    axes = {
        "tokens": {0: "batch", 1: "suffix"},
        "tokens:size1": {0: "batch"},
        "state_l000_k_in:size1": {0: "batch"},
        "scores": {0: "batch", 1: "suffix"},
    }
    for i in range(num_layers):
        for kind in ("k", "v"):
            src, dst = f"state_l{i:03d}_{kind}_in", f"state_l{i:03d}_{kind}_out"
            inputs.append(src)
            outputs.append(dst)
            axes[src], axes[dst] = {0: "batch", 1: "prefix"}, {0: "batch", 1: "suffix"}
            metadata[f"STATE_{src}"] = dst
    return inputs, outputs, axes, metadata


class Qwen3TextVariantsRasrLm(Qwen3RasrLm):
    """Extra first-piece aliases, legal only when predicting immediately after BOS.

    Original columns and their probabilities are untouched. An alias is a second
    spelling of a token, not additional probability mass to renormalize. Its input
    embedding/history is exactly the original token's. No acoustic ids change.
    """

    def __init__(self, model, alias_ids):
        super().__init__(model)
        if not alias_ids or alias_ids != sorted(set(alias_ids)):
            raise ValueError("Expected a nonempty, sorted, unique alias map")
        if any(not 0 <= i < model.config.vocab_size for i in alias_ids):
            raise ValueError("Alias target outside Qwen vocabulary")
        self.register_buffer("alias_ids", torch.tensor(alias_ids, dtype=torch.long))
        self.register_buffer("input_ids", torch.tensor([*range(model.config.vocab_size), *alias_ids], dtype=torch.long))

    def forward(self, tokens, lengths, prefix_lengths, *states):
        scores, *new_states = super().forward(self.input_ids[tokens.long()], lengths, prefix_lengths, *states)
        positions = prefix_lengths[:, None] + torch.arange(tokens.shape[1], device=tokens.device)[None, :]
        initial = scores.index_select(-1, self.alias_ids).masked_fill(positions[:, :, None] != 0, 1.0e30)
        return (torch.cat((scores, initial), dim=-1), *new_states)


def verify_qwen_text_variants(model, session, sequences, alias_ids, atol=2e-3):
    """Full-vocab HF check plus alias scores, input remapping, masks and cached positions."""
    import numpy as np

    vocab = model.config.vocab_size
    aliases = {token: vocab + i for i, token in enumerate(alias_ids)}
    if not any(s[1] in aliases for s in sequences):
        raise ValueError("No test sequence exercises an initial-token alias")

    class BaseColumns:
        def run(self, outputs, feed):
            feed = dict(feed)
            tokens = feed["tokens"].copy()
            positions = feed["state_l000_k_in:size1"][:, None] + np.arange(tokens.shape[1])[None, :]
            # Replace the first content token even when BOS lives in the cache.
            for row, col in zip(*np.where(positions == 1)):
                tokens[row, col] = aliases.get(int(tokens[row, col]), int(tokens[row, col]))
            feed["tokens"] = tokens
            scores, *states = session.run(outputs, feed)
            if scores.shape[-1] != vocab + len(alias_ids):
                raise ValueError("Incorrect extended Qwen output dimension")
            expected = np.where(positions[:, :, None] == 0, scores[:, :, alias_ids], 1.0e30)
            if not np.array_equal(scores[:, :, vocab:], expected.astype(scores.dtype)):
                raise ValueError("Initial-token alias score/mask mismatch")
            return [scores[:, :, :vocab], *states]

    return verify_qwen_export(model, BaseColumns(), sequences, atol=atol)


def export_qwen_onnx(wrapper, filename):
    import onnx

    inputs, outputs, axes, metadata = onnx_names(wrapper.num_layers)
    args = (
        torch.ones((2, 3), dtype=torch.int32),
        torch.tensor([3, 2], dtype=torch.int32),
        torch.tensor([2, 1], dtype=torch.int32),
        *(torch.zeros((2, 2, wrapper.cache_dim)) for _ in range(2 * wrapper.num_layers)),
    )
    torch.onnx.export(
        wrapper,
        args,
        filename,
        input_names=inputs,
        output_names=outputs,
        dynamic_axes=axes,
        opset_version=17,
        dynamo=False,
        do_constant_folding=False,
        external_data=True,
    )
    model = onnx.load(filename, load_external_data=False)
    onnx.helper.set_model_props(model, metadata)
    onnx.save(model, filename)
    onnx.checker.check_model(filename)


def verify_qwen_export(model, session, sequences, atol=2e-3):
    """Compare HF full-prefix NLLs to ONNX cached, variable-length batches.

    Includes empty prefix, right-padding, right-aligned unequal prefixes, and
    cache reuse. Tests every vocabulary column at every valid suffix position.
    """
    import numpy as np

    inputs, outputs, _, _ = onnx_names(model.config.num_hidden_layers)
    dim = model.config.num_key_value_heads * model.config.head_dim
    maximum = 0.0
    if not sequences or any(len(s) < 2 for s in sequences):
        raise ValueError("Cache validation requires nonempty sequences of at least two tokens")

    def compare(actual, expected):
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError("Nonfinite Qwen ONNX/HF scores")
        return float(np.max(np.abs(actual - expected)))

    with torch.inference_mode():
        reference = [
            -model(torch.tensor([s]), use_cache=False).logits[0].float().log_softmax(-1).numpy() for s in sequences
        ]

    def forward(parts, caches, prefix_lengths):
        lengths = np.asarray([len(s) for s in parts], dtype=np.int32)
        tokens = np.zeros((len(parts), max(lengths)), dtype=np.int32)
        for i, s in enumerate(parts):
            tokens[i, : len(s)] = s
        feed = dict(zip(inputs, [tokens, lengths, np.asarray(prefix_lengths, dtype=np.int32), *caches]))
        return session.run(outputs, feed)

    empty = [np.zeros((len(sequences), 0, dim), dtype=np.float32) for _ in range(2 * model.config.num_hidden_layers)]
    full = forward(sequences, empty, [0] * len(sequences))
    for i, s in enumerate(sequences):
        delta = compare(full[0][i, : len(s)], reference[i])
        maximum = max(maximum, delta)
    cuts = [min(i + 1, len(s) - 1) for i, s in enumerate(sequences)]
    prefix = forward([s[:cut] for s, cut in zip(sequences, cuts)], empty, [0] * len(sequences))
    caches = []
    for state in prefix[1:]:
        aligned = np.zeros_like(state)
        for i, cut in enumerate(cuts):
            aligned[i, -cut:] = state[i, :cut]
        caches.append(aligned)
    suffix = forward([s[cut:] for s, cut in zip(sequences, cuts)], caches, cuts)
    for i, (s, cut) in enumerate(zip(sequences, cuts)):
        maximum = max(maximum, compare(suffix[0][i, : len(s) - cut], reference[i][cut:]))
    if not np.isfinite(maximum) or maximum > atol:
        raise ValueError(f"Qwen ONNX/HF score mismatch: max_abs_error={maximum}, tolerance={atol}")
    return maximum
