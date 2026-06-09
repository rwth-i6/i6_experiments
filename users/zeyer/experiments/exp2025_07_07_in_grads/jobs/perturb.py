"""Deterministic *non-parametric* model-degradation perturbations, complementing
``param_noise`` (which corrupts the weights). These degrade the model at inference
time without touching the weights, and -- unlike a discrete weight-corruption
threshold -- are expected to degrade recognition/alignment *smoothly*:

- input noise:      x += std * std(x) * eps        (Gaussian on the waveform; an SNR knob)
- activation noise: h += std * std(h) * eps         (per Linear output, forward hook)
- activation dropout: h = dropout(h, p) at eval      (per Linear output, forward hook)

All are seeded and per-tensor-std-scaled (relative perturbation uniform across
layers/utterances), so the SAME degraded model/input is reproduced across the
grad-align / forced-align / cross-attn jobs at a given level -> WBE and WER are
measured on the same condition.
"""

from __future__ import annotations

from typing import List


def apply_input_noise(wav, std: float, seed: int = 0):
    """Add Gaussian noise to a 1-D waveform, scaled by its own std. Returns a new array."""
    import numpy as np

    if not std:
        return wav
    wav = np.asarray(wav, dtype=np.float32)
    # vary the realization per utterance (length-derived) but stay deterministic
    g = np.random.RandomState((int(seed) * 100003 + wav.shape[0]) % (2**31 - 1))
    scale = std * float(max(wav.std(), 1e-8))
    return (wav + g.standard_normal(wav.shape).astype(np.float32) * scale).astype(np.float32)


def install_activation_perturbation(model, kind: str, level: float, seed: int = 0) -> List:
    """Register forward hooks on every nn.Linear so its output is perturbed at eval.

    kind="act_noise": out += level * std(out) * eps    (level = relative noise std)
    kind="act_dropout": out = dropout(out, p=level)     (level = drop prob)

    Returns the list of hook handles (call .remove() to undo). No-op if level==0.
    """
    import torch

    handles = []
    if not level:
        return handles
    assert kind in ("act_noise", "act_dropout"), kind
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    def noise_hook(_module, _inp, out):
        if not isinstance(out, torch.Tensor):
            return out
        scale = level * float(out.detach().float().std().clamp_min(1e-8).item())
        eps = torch.randn(out.shape, generator=g, dtype=torch.float32).to(dtype=out.dtype, device=out.device)
        return out + eps * scale

    def dropout_hook(_module, _inp, out):
        if not isinstance(out, torch.Tensor):
            return out
        return torch.nn.functional.dropout(out, p=level, training=True)

    hook = noise_hook if kind == "act_noise" else dropout_hook
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            handles.append(m.register_forward_hook(hook))
    return handles
