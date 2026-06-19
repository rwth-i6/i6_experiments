"""Standalone sanity tests for the CIF pooler (run with the conda speech_llm python).

Checks the properties the design depends on:
  1. completed tokens are convex combinations (contributing weights sum to 1.0);
  2. mass conservation: total scattered weight == sum(alpha) over valid frames;
  3. token count K == floor(total mass) (min-1 floor), and z_mask matches K;
  4. padding frames neither fire nor contribute;
  5. DIFFERENTIABILITY: CE-like gradient on z reaches the alpha predictor (non-zero grad).
"""

import torch

from cif import CIFAlphaPredictor, cif_pool, quantity_loss


def test_weight_sum_and_mass():
    torch.manual_seed(0)
    b, t, d = 3, 40, 8
    feats = torch.randn(b, t, d)
    alpha = torch.rand(b, t) * 0.6 + 0.2  # in (0.2, 0.8), <1 guaranteed
    lengths = torch.tensor([40, 30, 12])
    z, z_mask, diag = cif_pool(feats, alpha, lengths)

    # mass conservation: total weight placed (incl. dropped partial) == sum(alpha) over valid frames
    aval = (alpha * diag["frame_valid"]).sum(dim=1)
    torch.testing.assert_close(aval, diag["sum_alpha"], rtol=1e-4, atol=1e-4)

    # K == floor(total) (min 1) and z_mask agrees
    K_expected = torch.floor(diag["sum_alpha"] + 1e-4).long().clamp(min=1)
    assert torch.equal(diag["fires"], K_expected), (diag["fires"], K_expected)
    assert torch.equal(z_mask.sum(dim=1), K_expected)

    # completed tokens (all but possibly the last when total>=1) must be unit-weight convex combos.
    # Reconstruct per-token weight sums by pooling ones instead of features.
    ones = torch.ones(b, t, 1)
    wsum, _, _ = cif_pool(ones, alpha, lengths)  # [B,K,1] = per-token total weight
    wsum = wsum.squeeze(-1)
    for bi in range(b):
        kk = int(K_expected[bi])
        if diag["sum_alpha"][bi] >= 1.0:
            # all but the last kept token are fully completed -> weight 1.0
            comp = wsum[bi, : kk - 1] if kk >= 1 else wsum[bi, :0]
            if comp.numel():
                torch.testing.assert_close(comp, torch.ones_like(comp), rtol=1e-4, atol=1e-4)
    print("OK weight_sum_and_mass: K =", diag["fires"].tolist())


def test_padding_ignored():
    b, t, d = 2, 20, 4
    feats = torch.randn(b, t, d)
    alpha = torch.rand(b, t) * 0.5 + 0.3
    lengths = torch.tensor([20, 8])
    _, _, diag = cif_pool(feats, alpha, lengths)
    # padding frames of utt 1 (>=8) contribute zero alpha
    assert diag["alpha"][1, 8:].abs().sum().item() == 0.0
    print("OK padding_ignored")


def test_differentiable():
    torch.manual_seed(1)
    b, t, d = 2, 30, 16
    feats = torch.randn(b, t, d)  # frozen features (no grad)
    lengths = torch.tensor([30, 22])
    pred = CIFAlphaPredictor(d, kernel_size=5)
    alpha = pred(feats)
    z, z_mask, diag = cif_pool(feats, alpha, lengths)
    # a CE-like scalar on z only (quantity loss DETACHED) must still push gradient into the predictor
    loss = (z[z_mask] ** 2).mean()
    loss.backward()
    g = pred.proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum().item() > 0, "no CE->alpha gradient!"
    print("OK differentiable: |grad(proj.weight)| =", float(g.abs().sum()))


def test_quantity_loss():
    sum_alpha = torch.tensor([10.0, 5.0])
    lengths = torch.tensor([100, 50])  # 25 Hz; target 12.5 Hz -> Q* = T*0.5 = [50, 25]
    ql = quantity_loss(sum_alpha, lengths, target_rate_hz=12.5, frame_rate_hz=25.0)
    # rel = (10-50)/50=-0.8 ; (5-25)/25=-0.8 ; mean(0.64,0.64)=0.64
    torch.testing.assert_close(ql, torch.tensor(0.64), rtol=1e-4, atol=1e-4)
    print("OK quantity_loss:", float(ql))


if __name__ == "__main__":
    test_weight_sum_and_mass()
    test_padding_ignored()
    test_differentiable()
    test_quantity_loss()
    print("\nALL CIF TESTS PASSED")
