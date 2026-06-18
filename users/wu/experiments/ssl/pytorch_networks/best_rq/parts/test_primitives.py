"""
Unit tests for the BEST-RQ primitives (quantizer, mask, input-norm).

Pure-torch, CPU, no data needed. Run either with pytest or directly:

    PYTHONPATH=recipe:recipe/i6_models \
      /e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python \
      recipe/i6_experiments/users/wu/experiments/ssl/pytorch_networks/best_rq/parts/test_primitives.py
"""

from __future__ import annotations

import torch

from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.parts.input_norm import (
    masked_mean_var_norm,
)
from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.parts.mask import compute_span_mask
from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.parts.quantizer import (
    MultiCodebookRandomProjectionQuantizer,
)


def test_input_norm_zero_mean_unit_var_and_padding():
    torch.manual_seed(0)
    b, t, f = 4, 100, 80
    x = torch.randn(b, t, f) * 5.0 + 3.0
    lengths = torch.tensor([100, 75, 50, 10])
    y = masked_mean_var_norm(x, lengths)
    for i, n in enumerate(lengths.tolist()):
        valid = y[i, :n]
        assert valid.mean().abs() < 1e-4, valid.mean()
        # per-feature std ~1 -> overall std close to 1
        assert abs(float(valid.std(unbiased=False)) - 1.0) < 1e-2
        if n < t:
            assert y[i, n:].abs().max() == 0.0, "padding must be zeroed"
    print("ok input_norm: zero-mean/unit-var over valid frames, padding zeroed")


def test_quantizer_frozen_shapes_deterministic():
    q = MultiCodebookRandomProjectionQuantizer(
        input_dim=80, stack_size=4, codebook_dim=16, vocab_size=8192, num_codebooks=4, seed=42
    )
    # frozen: no trainable parameters, projection/codebook are buffers
    assert len(list(q.parameters())) == 0, "quantizer must have no trainable parameters"
    assert {n for n, _ in q.named_buffers()} == {"projection", "codebook"}
    # codebook is L2-normalized
    norms = q.codebook.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    b, t = 3, 401
    x = torch.randn(b, t, 80)
    lengths = torch.tensor([401, 300, 120])
    tgt, tlen = q(x, lengths)
    assert tgt.shape == (b, t // 4, 4), tgt.shape
    assert tgt.dtype == torch.int64
    assert int(tgt.min()) >= 0 and int(tgt.max()) < 8192
    assert torch.equal(tlen, lengths // 4)
    # deterministic
    tgt2, _ = q(x, lengths)
    assert torch.equal(tgt, tgt2)
    # no grad path
    assert not tgt.requires_grad
    print("ok quantizer: frozen, L2 codebook, shape [B,T//4,N], indices in range, deterministic")


def test_mask_padding_safe_min_masks_and_coverage():
    torch.manual_seed(0)
    b, t = 8, 400
    lengths = torch.tensor([400, 350, 300, 250, 200, 120, 60, 20])
    L = 10
    m = compute_span_mask(
        batch_size=b, time_size=t, lengths=lengths, mask_prob=0.04, mask_length=L,
        device=torch.device("cpu"), min_masks=1,
    )
    assert m.shape == (b, t) and m.dtype == torch.bool
    for i, n in enumerate(lengths.tolist()):
        assert not bool(m[i, n:].any()), "no masking allowed in padding"
        assert int(m[i, :n].sum()) >= 1, "min_masks not satisfied"
    # realized coverage over valid frames is in a sane range (heavy overlap allowed)
    cov = float(m.sum()) / float(lengths.sum())
    assert 0.05 < cov < 0.95, cov
    print(f"ok mask: padding-safe, >=1 span/seq, realized coverage={cov*100:.1f}%")


def test_mask_targets_alignment():
    """Mask is at subsampled rate; expanded x stack must align with full-rate features."""
    torch.manual_seed(0)
    b, t2, stack = 2, 100, 4
    lengths2 = torch.tensor([100, 60])
    m = compute_span_mask(
        batch_size=b, time_size=t2, lengths=lengths2, mask_prob=0.05, mask_length=10,
        device=torch.device("cpu"), min_masks=1,
    )
    m_full = m.repeat_interleave(stack, dim=1)  # [B, T2*stack]
    assert m_full.shape == (b, t2 * stack)
    # every masked subsampled frame corresponds to exactly `stack` masked full-rate frames
    assert int(m_full.sum()) == int(m.sum()) * stack
    print("ok mask alignment: repeat_interleave x stack preserves masked-frame correspondence")


def main():
    test_input_norm_zero_mean_unit_var_and_padding()
    test_quantizer_frozen_shapes_deterministic()
    test_mask_padding_safe_min_masks_and_coverage()
    test_mask_targets_alignment()
    print("\nALL PRIMITIVE TESTS PASSED")


if __name__ == "__main__":
    main()
